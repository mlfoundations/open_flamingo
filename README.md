# 🦩 OpenFlamingo

[![PyPI version](https://badge.fury.io/py/open_flamingo.svg)](https://badge.fury.io/py/open_flamingo)

Blog post (coming soon) | Twitter thread (coming soon) | Paper (coming soon)

Welcome to our open source version of DeepMind's [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) model! In this repository, we provide a PyTorch implementation for training and evaluating OpenFlamingo models. We also provide an initial [OpenFlamingo 3B model](#api) trained on a new Multimodal C4 dataset. Please refer to our blog post for more details.

This repo is still under development. You can expect us to release better performing and larger Flamingo models soon. If you have any questions, please feel free to open an issue. We also welcome pull requests!

# Table of Contents
- [Installation](#installation)
- [Approach](#approach)
  * [Model architecture](#model-architecture)
- [Usage](#usage)
  * [Initializing an OpenFlamingo model](#initializing-an-openflamingo-model)
  * [Generating text](#generating-text)
- [Training](#training)
  * [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Future plans](#future-plans)
- [Team](#team)
- [Acknowledgments](#acknowledgments)
- [Citing](#citing)

# Installation

To create a conda environment for running OpenFlamingo, run

```
conda env create -f environment.yml
```

Alternatively, to install the package in an existing environment, run 
```
pip install open-flamingo
```

# Usage
We provide an initial [OpenFlamingo 3B model](https://huggingface.co/openflamingo/OpenFlamingo-3b) using a CLIP ViT-Large vision encoder and an OPT 1.3B language encoder. In general, we support any [CLIP vision encoder](https://huggingface.co/models?search=clip). For the language model, we support [LLaMA](https://huggingface.co/models?search=llama), [OPT](https://huggingface.co/models?search=opt), [GPT-Neo](https://huggingface.co/models?search=gpt-neo), [GPT-J](https://huggingface.co/models?search=gptj), and [Pythia](https://huggingface.co/models?search=pythia) models.

## Initializing an OpenFlamingo model
``` python
from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="openai/clip-vit-large-patch14",
    clip_processor_path="openai/clip-vit-large-patch14",
    lang_encoder_path="facebook/opt-1.3b",
    tokenizer_path="facebook/opt-1.3b",
)

# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)
```

## Generating text
Here is an example of generating text conditioned on interleaved images/text, in this case we will do few-shot image captioning.

``` python
from PIL import Image
import requests

"""
Step 1: Load images
"""
demo_image_one = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
    ).raw
)

demo_image_two = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
        stream=True
    ).raw
)

query_image = Image.open(
    requests.get(
        "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
        stream=True
    ).raw
)


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1 
 (this will always be one expect for video which we don't support yet), 
 channels = 3, height = 224, width = 224.
"""
vision_x = image_processor(images=[demo_image_one, demo_image_two, query_image], 
 return_tensors="pt")["pixel_values"]
vision_x = vision_x.unsqueeze(1).unsqueeze(0)


"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
    max_length=128,
    padding=True,
    return_tensors="pt",
)


"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text: ", tokenizer.decode(generated_text[0]))
```

# Approach
OpenFlamingo is a multimodal language model that can be used for a variety of tasks. It is trained on a large multimodal dataset (e.g. Multimodal C4) and can be used to generate text conditioned on interleaved images/text. For example, OpenFlamingo can be used to generate a caption for an image, or to generate a question given an image and a text passage. The benefit of this approach is that we are able to rapidly adapt to new tasks using in-context training.

## Model architecture
OpenFlamingo seeks to fuse pretrained a vision encoder and a language model using cross attention layers. The model architecture is shown below.

![OpenFlamingo architecture](docs/flamingo.png) 
Credit: [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)

# Training
To train a model, modify the following example command:
```
torchrun --nnodes=1 --nproc_per_node=2
train.py 
--run_name flamingo3B
--lm_path facebook/opt-1.3b \
--dataset_resampled \
--batch_size_mmc4 4 \
--batch_size_laion 8 \
--train_num_samples_mmc4 125000 \
--train_num_samples_laion 250000 \
--loss_multiplier_laion 0.2 \
--workers=6 \
--report_to_wandb \
--num_epochs 250 \
--lr_scheduler constant \
--warmup_steps 5000 \
--use_media_placement_augmentation \
--mmc4_textsim_threshold 30
```

## Dataset
We expect all our training datasets to be [WebDataset](https://github.com/webdataset/webdataset) shards.
We train our models on the [LAION 2B](https://huggingface.co/datasets/laion/laion2B-en) and Multimodal C4 (coming soon) datasets. By default the LAION 2B dataset is in WebDataset format if it is downloaded using the [img2dataset tool](https://github.com/rom1504/img2dataset) and Multimodal C4 comes packaged in the WebDataset format.


# Evaluation
We currently support running evaluations on [COCO](https://cocodataset.org/#home), [VQAv2](https://visualqa.org/index.html), [OKVQA](https://okvqa.allenai.org), [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), and [ImageNet](https://image-net.org/index.php). Note that currently these evaluations are ran in validation mode (as specified in the Flamingo paper). We will be adding support for running evaluations in test mode in the future. Moreover we will be adding support [very soon](https://github.com/mlfoundations/open_flamingo/pull/93) for querying demonstrations directly from the training dataset rather than picking a fixed set of demonstrations for evaluation. 


Before evaluating the model, you will need to install the coco evaluation package by running the following command:
```
pip install pycocoevalcap
```

To run evaluations on OKVQA you will need to run the following command:
```
import nltk
nltk.download('wordnet')
```

To evaluate the model, use script open_flamingo/eval/evaluate.py. For example, to evaluate the model on COCO and VQAv2, run the following command:

```
python evaluate.py
--lm_path facebook/opt-1.3b
--lm_tokenizer_path facebook/opt-1.3b
--clip_path openai/clip-vit-large-patch14
--checkpoint_path path/to/checkpoint.pt
--device 0
--coco_image_dir_path path/to/coco/images
--coco_annotations_json_path path/to/coco/captions_train2017.json
--vqav2_image_dir_path path/to/vqav2/images
--vqav2_annotations_json_path path/to/vqav2/v2_mscoco_train2014_annotations.json
--vqav2_questions_json_path path/to/vqav2/v2_OpenEnded_mscoco_train2014_questions.json
--eval_coco
--eval_vqav2
``` 

# Future plans
- [ ] Add support for video input
- [ ] Release better performing and larger OpenFlamingo models
- [ ] Expand our evaluation suite
- [ ] Add support for FSDP training

# Team


# Acknowledgments
This code is based on Lucidrains' [flamingo implementation](https://github.com/lucidrains/flamingo-pytorch) and David Hansmair's [flamingo-mini repo](https://github.com/dhansmair/flamingo-mini). Thank you for making your code public!

We would also like to thank [Jean-Baptiste Alayrac](https://www.jbalayrac.com) and [Antoine Miech](https://antoine77340.github.io) for their advice and thanks to [Stability AI](https://stability.ai) for providing us with compute resources to train these models.

# Citing
If you found this repository useful, please consider citing:

```
@software{open_flamingo,
    author = {...},
    title = {OpenFlamingo},
    month = march,
    year = 2023,
    note = {If you use this software, please cite it as below.},
    ... TBD}
```

```
@article{Alayrac2022FlamingoAV,
  title={Flamingo: a Visual Language Model for Few-Shot Learning},
  author={Jean-Baptiste Alayrac and Jeff Donahue and Pauline Luc and Antoine Miech and Iain Barr and Yana Hasson and Karel Lenc and Arthur Mensch and Katie Millican and Malcolm Reynolds and Roman Ring and Eliza Rutherford and Serkan Cabi and Tengda Han and Zhitao Gong and Sina Samangooei and Marianne Monteiro and Jacob Menick and Sebastian Borgeaud and Andy Brock and Aida Nematzadeh and Sahand Sharifzadeh and Mikolaj Binkowski and Ricardo Barreira and Oriol Vinyals and Andrew Zisserman and Karen Simonyan},
  journal={ArXiv},
  year={2022},
  volume={abs/2204.14198}
}
```
