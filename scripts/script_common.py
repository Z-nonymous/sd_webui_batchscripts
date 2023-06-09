import gradio as gr
import modules.scripts as scripts

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
    "sd_model_hash": process_string_tag,
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

overrides_mapping = {
    "Override Model": "sd_model_hash",
    "Override Seed": "seed",
    "Override Steps": "steps",
    "Override CFG Scale": "cfg_scale",
    "Override Sampler": "sampler_name",
    "Override Width": "width",
    "Override Height": "height"
}
possible_overrides = list(overrides_mapping.keys())
default_overrides = ["Override Model"]


def load_prompt_file(file):
    if file is None:
        return None, gr.update(), gr.update(lines=7)
    else:
        lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]
        return None, "\n".join(lines), gr.update(lines=7)


def ui(script: scripts.Script):
    script_overrides = gr.CheckboxGroup(label="Overrides", choices=possible_overrides, value=default_overrides)
    with gr.Accordion(label="Prompt overrides", open=False):
        prepend_prompt_text = gr.Textbox(label="Text to prepend", lines=1,
                                         elem_id=script.elem_id("prepend_prompt_text"))
        append_prompt = gr.Checkbox(label="Append text instead", elem_id=script.elem_id("append_prompt"))

    prompt_txt = gr.Textbox(label="List of prompt inputs", lines=1, elem_id=script.elem_id("prompt_txt"))
    file = gr.File(label="Upload prompt inputs", type='binary', elem_id=script.elem_id("file"))

    file.change(fn=load_prompt_file, inputs=[file], outputs=[file, prompt_txt, prompt_txt], show_progress=False)

    # We start at one line. When the text changes, we jump to seven lines, or two lines if no \n.
    # We don't shrink back to 1, because that causes the control to ignore [enter], and it may
    # be unclear to the user that shift-enter is needed.
    prompt_txt.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=2), inputs=[prompt_txt],
                      outputs=[prompt_txt], show_progress=False)
    return [prepend_prompt_text, append_prompt, prompt_txt, script_overrides]
