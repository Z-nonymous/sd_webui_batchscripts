# sd_webui_batchscripts
Some batch scripts for use with stable-diffusion-webui 
---
## batchimagesA.py
A custom script to use a list of images (with pnginfo) as a source for txt2img

This will help when you want to regenerate with a new model a bunch of images you previously generated.

textfile or input box must have one path to file per line

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

A custom script to use a list of images (with pnginfo) as a source for txt2img
This will help when you want to regenerate with a new model a bunch of images you previously generated.
textfile or input box must have one path to file per line between quotes then a space then the batch_size

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
A custom script to use a list of prompt that use the same format as on copying from civit AI (or from PNG info)

Two entries must be separated by an empty line ('\n\n')

i.e. under windows
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