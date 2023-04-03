import copy
import os

import gradio as gr
import modules.scripts as scripts
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


arg_mapping = {
    'Prompt': 'prompt',
    'Negative prompt': 'negative_prompt',
    'Steps': 'steps',
    'Sampler': 'sampler_name',
    'CFG scale': 'cfg_scale',
    'Seed': 'seed',
    'Size-1': 'width',
    'Size-2': 'height',
    'Model hash': 'sd_model_hash',  # goes to override
    'Clip skip': 'CLIP_stop_at_last_layers',  # goes to override instead of using clip_skip
    'Denoising strength': 'denoising_strength',
    'Hires upscale': 'hr_scale',
    'Hires upscaler': 'hr_upscaler',
    'Hires resize-1': 'hr_resize_x',
    'Hires resize-2': 'hr_resize_y',
    'Hires steps': 'hr_second_pass_steps',
    'First pass size-1': 'skip-1',
    'First pass size-2': 'skip-2',
    'Model': 'skip-3',
    'Hypernet': 'skip-4',
    'Hypernet strength': 'skip-5',
    'Aesthetic LR': 'skip-6',
    'Aesthetic text': 'skip-7',
    'Aesthetic slerp': 'skip-8',
    'Aesthetic steps': 'skip-9',
    'Aesthetic weight': 'skip-10',
    'Aesthetic embedding': 'skip-11',
    'Aesthetic slerp angle': 'skip-12',
    'Aesthetic text negative': 'skip-13',
    'Conditional mask weight': 'inpainting_mask_weight',  # goes to override
    'ENSD': 'eta_noise_seed_delta',  # goes to override
    'Noise multiplier': 'initial_noise_multiplier',  # goes to override
    'Eta': 'eta_ancestral',  # goes to override instead of using 'eta'
    # 'Eta': 'eta_ancestral',
    'Eta DDIM': 'eta_ddim',  # goes to override
    'Discard penultimate sigma': 'always_discard_next_to_last_sigma',  # goes to override
    'Variation seed': 'subseed',
    'Variation seed strength': 'subseed_strength',
    'Face restoration': 'face_restoration_model',
    'Mask blur': 'mask_blur',
    'Seed resize from': 'seed_resize'
}

override_list = [
    'sd_model_hash',
    'CLIP_stop_at_last_layers',
    'inpainting_mask_weight',
    'eta_noise_seed_delta',
    'initial_noise_multiplier',
    'eta_ancestral',
    'eta_ddim',
    'always_discard_next_to_last_sigma'
]


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


def process_seedresize_tag(tag):
    sp = tag.split('x')
    return (int(sp[0]), int(sp[1]))


prompt_tags = {
    "sd_model": None,
    "sd_model_hash": process_int_tag,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag,
    "clip_skip": process_int_tag,
    "CLIP_stop_at_last_layers": process_int_tag,
    "hr_scale": process_float_tag,
    "hr_upscaler": process_string_tag,
    "hr_resize_x": process_int_tag,
    "hr_resize_y": process_int_tag,
    "hr_second_pass_steps": process_int_tag,
    "eta_noise_seed_delta": process_int_tag,
    "eta_ancestral": process_float_tag,
    "eta": process_float_tag,
    "eta_ddim": process_float_tag,
    "denoising_strength": process_float_tag,
    "face_restoration_model": process_string_tag,
    "mask_blur": process_int_tag,
    "seed_resize": process_seedresize_tag
}


class Script(scripts.Script):

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "Batch from imagelist B"

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

        keep_src_hash = gr.Checkbox(label="Keep source image Model Hash", elem_id=self.elem_id("keep_src_hash"))
        prepend_prompt_text = gr.Textbox(label="Text to prepend", lines=1, elem_id=self.elem_id("prepend_prompt_text"))
        append_prompt = gr.Checkbox(label="Append text instead", elem_di=self.elem_id("append_prompt"))
        prompt_txt = gr.Textbox(label="List of prompt inputs", lines=1, elem_id=self.elem_id("prompt_txt"))
        file = gr.File(label="Upload prompt inputs", type='binary', elem_id=self.elem_id("file"))

        file.change(fn=load_prompt_file, inputs=[file], outputs=[file, prompt_txt, prompt_txt])

        # We start at one line. When the text changes, we jump to seven lines, or two lines if no \n.
        # We don't shrink back to 1, because that causes the control to ignore [enter], and it may
        # be unclear to the user that shift-enter is needed.
        prompt_txt.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=2), inputs=[prompt_txt],
                          outputs=[prompt_txt])
        return [keep_src_hash, prepend_prompt_text, append_prompt, prompt_txt]

    # This is where the additional processing is implemented. The parameters include
    # self, the model object "p" (a StableDiffusionProcessing class, see
    # processing.py), and the parameters returned by the ui method.
    # Custom functions can be defined here, and additional libraries can be imported
    # to be used in processing. The return value should be a Processed object, which is
    # what is returned by the process_images method.

    def run(self, p, keep_src_hash: bool, prepend_prompt_text: str, append_prompt: bool, prompt_txt: str):

        import modules.images as img
        import modules.generation_parameters_copypaste as gpc

        def update_dict_keys(obj, mapping_dict):
            if isinstance(obj, dict):
                return {mapping_dict[k]: update_dict_keys(v, mapping_dict) for k, v in obj.items()}
            else:
                return obj

        lines = [x.strip() for x in prompt_txt.splitlines()]
        lines = [x for x in lines if len(x) > 0]

        p.do_not_save_grid = True

        job_count = 0
        jobs = []
        overrides = []

        for line in lines:
            if line.startswith('"'):
                parts = line.split('" ')
                chemin = parts[0].strip('"')
                try:
                    or_batchsize = parts[1].strip('"')
                except IndexError:
                    or_batchsize = '1'

                if os.path.isfile(chemin):
                    formated_args = {}
                    f = open(chemin, "rb")
                    data = f.read()
                    f.close()
                    if data:
                        info = img.image_data(data)
                        res = gpc.parse_generation_parameters(info[0])
                        args = update_dict_keys(res, arg_mapping)

                        for i in range(1, 21):
                            args.pop(f'skip-{i}', None)

                        if not keep_src_hash:
                            args.pop('sd_model_hash', None)

                        for arg, val in args.items():
                            func = prompt_tags.get(arg, None)
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

                        for setting_name in override_list:
                            value = formated_args.pop(setting_name, None)
                            if value is None:
                                continue
                            override_settings[setting_name] = shared.opts.cast_value(setting_name, value)

                        try:
                            formated_args['batch_size'] = int(or_batchsize)
                        except ValueError:
                            continue

                        if prepend_prompt_text != '':
                            if append_prompt:
                                formated_args['prompt'] = formated_args.get('prompt', '') + ' ,' + prepend_prompt_text
                            else:
                                formated_args['prompt'] = prepend_prompt_text + ', ' + formated_args.get('prompt', '')

                        job_count += formated_args.get("n_iter", p.n_iter * formated_args.get('batch_size', 1))
                        jobs.append(formated_args)
                        overrides.append(override_settings)

        print(f"Will process {len(lines)} lines in {job_count} jobs.")

        images = []
        all_prompts = []
        infotexts = []
        for n, args in enumerate(jobs):
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            copy_p.override_settings = overrides[n]
            copy_p.override_settings_restore_afterwards = True
            copy_p.extra_generation_params = {}

            proc = process_images(copy_p)
            images += proc.images

            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
