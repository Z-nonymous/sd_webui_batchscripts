import copy

import gradio as gr
import modules.scripts as scripts
import scripts.script_common as sc
from modules import shared
from modules.processing import Processed
from modules.processing import process_images
from modules.shared import state


def load_prompt_file(file):
    if file is None:
        lines = []
    else:
        lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]

    return None, "\n".join(lines), gr.update(lines=7)


class Script(scripts.Script):

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "Batch from PNGINFO"

    # Determines when the script should be shown in the dropdown menu via the
    # returned value. As an example:
    # is_img2img is True if the current tab is img2img, and False if it is txt2img.
    # Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):

        return not is_img2img

    # How the script's is displayed in the UI. See https://gradio.app/docs/#components
    # for the different UI components you can use and how to create them.
    # Most UI components can return a value, such as a boolean for a checkbox.
    # The returned values are passed to the run method as parameters.

    def ui(self, is_txt2img):

        script_overrides = gr.CheckboxGroup(label="Overrides", choices=sc.possible_overrides,
                                            value=sc.default_overrides)
        with gr.Accordion(label="Prompt overrides", open=False):
            prepend_prompt_text = gr.Textbox(label="Text to prepend", lines=1,
                                             elem_id=self.elem_id("prepend_prompt_text"))
            append_prompt = gr.Checkbox(label="Append text instead", elem_id=self.elem_id("append_prompt"))

        prompt_txt = gr.Textbox(label="List of prompt inputs", lines=1, elem_id=self.elem_id("prompt_txt"))
        file = gr.File(label="Upload prompt inputs", type='binary', elem_id=self.elem_id("file"))

        file.change(fn=load_prompt_file, inputs=[file], outputs=[file, prompt_txt, prompt_txt])

        # We start at one line. When the text changes, we jump to seven lines, or two lines if no \n.
        # We don't shrink back to 1, because that causes the control to ignore [enter], and it may
        # be unclear to the user that shift-enter is needed.
        prompt_txt.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=2), inputs=[prompt_txt],
                          outputs=[prompt_txt])
        return [prepend_prompt_text, append_prompt, prompt_txt, script_overrides]

    # This is where the additional processing is implemented. The parameters include
    # self, the model object "p" (a StableDiffusionProcessing class, see
    # processing.py), and the parameters returned by the ui method.
    # Custom functions can be defined here, and additional libraries can be imported
    # to be used in processing. The return value should be a Processed object, which is
    # what is returned by the process_images method.

    def run(self, p, prepend_prompt_text: str, append_prompt: bool, prompt_txt: str, script_overrides: list[str]):

        import modules.generation_parameters_copypaste as gpc
        from modules import sd_models as sdm

        def update_dict_keys(obj, mapping_dict):
            if isinstance(obj, dict):
                return {mapping_dict[k]: update_dict_keys(v, mapping_dict) for k, v in obj.items()}
            else:
                return obj

        def apply_checkpoint(x):
            info = sdm.get_closet_checkpoint_match(x)
            if info is None:
                raise RuntimeError(f"Unknown checkpoint: {x}")
            sdm.reload_model_weights(shared.sd_model, info)

        fallback_checkpoint = shared.opts.sd_model_checkpoint
        checkpoint_switched = False

        p.do_not_save_grid = True

        job_count = 0
        jobs = []
        overrides = []

        for block in prompt_txt.split('\n\n'):
            formated_args = {}
            res = gpc.parse_generation_parameters(block)
            args = update_dict_keys(res, sc.arg_mapping)

            for i in range(1, 21):
                args.pop(f'skip-{i}', None)

            for ovr in sc.possible_overrides:
                if ovr in script_overrides:
                    args.pop(sc.overrides_mapping.get(ovr, None))

            for arg, val in args.items():
                func = sc.prompt_tags.get(arg, None)
                assert func, f'unknown file setting: {arg}'
                formated_args[arg] = func(val)

            if (formated_args.get('hr_scale', 0) > 0) or (formated_args.get('hr_resize_x', 0) > 0) or (
                    formated_args.get('hr_resize_y', 0) > 0):
                formated_args['enable_hr'] = True
            else:
                formated_args['enable_hr'] = False

            if formated_args.get('face_restoration_model', False):
                formated_args['restore_faces'] = True

            seed_resize = formated_args.get('seed_resize', None)
            if seed_resize:
                formated_args['seed_resize_from_w'] = seed_resize[0]
                formated_args['seed_resize_from_h'] = seed_resize[1]
                formated_args.pop('restore_faces', None)

            override_settings = {}

            for setting_name in sc.override_list:
                value = formated_args.get(setting_name, None)
                if value is None:
                    continue
                override_settings[setting_name] = shared.opts.cast_value(setting_name, value)

            if prepend_prompt_text != '':
                if append_prompt:
                    formated_args['prompt'] = formated_args.get('prompt', '') + ' ,' + prepend_prompt_text
                else:
                    formated_args['prompt'] = prepend_prompt_text + ', ' + formated_args.get('prompt', '')

            job_count += formated_args.get("n_iter", p.n_iter)
            jobs.append(formated_args)
            overrides.append(override_settings)

        print(f"Will process {job_count} jobs.")

        images = []
        all_prompts = []
        infotexts = []

        def process_runlist(jobs, images, all_prompts, infotexts):
            for n, args in enumerate(jobs):
                state.job = f"{state.job_no + 1} out of {state.job_count}"

                copy_p = copy.copy(p)
                for k, v in args.items():
                    setattr(copy_p, k, v)

                copy_p.override_settings = overrides[n]
                copy_p.extra_generation_params = {}

                proc = process_images(copy_p)
                images += proc.images

                all_prompts += proc.all_prompts
                infotexts += proc.infotexts

        models_to_use = {j.get('sd_model_hash') for j in jobs}
        runs_list = {}
        for model in models_to_use:
            runs_list[model] = [j for j in jobs if j.get('sd_model_hash') == model]
        if None in runs_list.keys():
            process_runlist(runs_list[None], images, all_prompts, infotexts)
            runs_list.pop(None)
        if fallback_checkpoint in runs_list.keys():
            process_runlist(runs_list[fallback_checkpoint], images, all_prompts, infotexts)
            runs_list.pop(fallback_checkpoint)
        for model in runs_list.keys():
            apply_checkpoint(model)
            checkpoint_switched = True
            process_runlist(runs_list[model], images, all_prompts, infotexts)
        if checkpoint_switched:
            apply_checkpoint(fallback_checkpoint)

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
