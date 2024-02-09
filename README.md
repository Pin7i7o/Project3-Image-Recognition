---
license: mit
inference: false
pipeline_tag: image-to-text
tags:
- image-captioning
---
# FuseCap: Leveraging Large Language Models for Enriched Fused Image Captions

A framework designed to generate semantically rich image captions.

## Resources

- 💻 **Project Page**: For more details, visit the official [project page](https://rotsteinnoam.github.io/FuseCap/).

- 📝 **Read the Paper**: You can find the paper [here](https://arxiv.org/abs/2305.17718).
    
- 🚀 **Demo**: Try out our BLIP-based model [demo](https://huggingface.co/spaces/noamrot/FuseCap) trained using FuseCap.

- 📂 **Code Repository**: The code for FuseCap can be found in the [GitHub repository](https://github.com/RotsteinNoam/FuseCap).
  
- 🗃️ **Datasets**: The  fused captions datasets can be accessed from [here](https://github.com/RotsteinNoam/FuseCap#datasets).
  
#### Running the model

Our BLIP-based model can be run using the following code,

```python
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap").to(device)

img_url = 'https://huggingface.co/spaces/noamrot/FuseCap/resolve/main/bike.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

text = "a picture of "
inputs = processor(raw_image, text, return_tensors="pt").to(device)

out = model.generate(**inputs, num_beams = 3)
print(processor.decode(out[0], skip_special_tokens=True))
```

## Upcoming Updates

The official codebase, datasets and trained models for this project will be released soon.

## BibTeX

``` Citation
@inproceedings{rotstein2024fusecap,
  title={Fusecap: Leveraging large language models for enriched fused image captions},
  author={Rotstein, Noam and Bensa{\"\i}d, David and Brody, Shaked and Ganz, Roy and Kimmel, Ron},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5689--5700},
  year={2024}
}
```