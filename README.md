# sd_webui_batchscripts
Some batch scripts for use with stable-diffusion-webui
All scripts require script_common.py

Install as an extension from webui

---

## batchimagesA.py
A custom script to use a list of images (containing generation parameters in png info) as a source for txt2img

This will help when you want to regenerate with a new model a bunch of images you previously generated.
You can choose to override some parameters with the settings in the UI (Seed, Steps, CFG Scale, Sampler, width, height)

The source textfile or input box must have one path to file per line

i.e. under windows
```
C:\My\Path\tofile1.png
X:\My\Path\to file2.png
```

### limitations:
- does not support Hypernetwork,
- does not support Aesthetic gradients.

---

## batchimagesB.py
Same as batchimagesA.py, but with support for a different batch_size per image.
A custom script to use a list of images (containing generation parameters in png info) as a source for txt2img

This will help when you want to regenerate with a new model a bunch of images you previously generated.
You can choose to override some parameters with the settings in the UI (Seed, Steps, CFG Scale, Sampler, width, height)

The source textfile or input box must have one path to file per line between quotes then a space then the `batch_size`

i.e. under windows
```
"C:\My\Path\tofile1.png" 1
"X:\My\Path\to file2.png" 4
```

### limitations:
- does not support Hypernetwork,
- does not support Aesthetic gradients.

---

## batchfrominfo.py
A custom script to use a list of prompt that use the same format as on copying from civitai.com (or from png info contained in generated images)

This will help you regenerate images from your prompt collection with a new model.

The source textfile or input bow must contain the information as extracted from image or copied from civitai.com 
Two entries must be separated by an empty line ('\n\n')

i.e.
```
a character standing
Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 10.0, Seed: 987654321, Size: 512x512, Model hash: 0123456789

a character standing
Negative prompt: too many fingers
Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 10.0, Seed: 987654321, Size: 512x512, Model hash: 0123456789
```

### limitations:
- does not support Hypernetwork,
- does not support Aesthetic gradients.

---
TODO (Maybe):
Support other specific parameters such as Hypernetworks or Aesthetic gradient