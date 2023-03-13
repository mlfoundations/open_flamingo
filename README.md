# 🦩 OpenFlamingo
[![PyPI version](https://badge.fury.io/py/open-flamingo.svg)](https://badge.fury.io/py/open-flamingo)

[Blog post]() | [Twitter thread]() | Paper (coming soon)

Welcome to our open source implementation of DeepMind's [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) model! In this repository, we provide a PyTorch implementation for training and evaluating OpenFlamingo models. We also provide an initial [OpenFlamingo 3B model](#api) trained on the [Multimodal C4 dataset](#multimodal-c4-dataset-mmc4). Please refer to our blog post for more details.

This repo is still under development. You can expect us to release better performing and larger Flamingo models soon. If you have any questions, please feel free to open an issue. We also welcome pull requests!

# Table of Contents
- [Installation](#installation)
- [Approach](#approach)
  * [Model architecture](#model-architecture)
- [Usage](#usage)
  * [Initializing an OpenFlamingo model](#initializing-an-openflamingo-model)
  * [Generating text](#generating-text)
- [Multimodal C4 dataset (MMC4)](#multimodal-c4-dataset-mmc4)
- [Training](#training)
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
pip install open_flamingo
```

# Usage
We provide an initial [OpenFlamingo 3B model](https://huggingface.co/open-flamingo/flamingo3B) using a CLIP ViT-Large vision encoder and an OPT 1.3B language encoder. In general, we support any [CLIP vision encoder](https://huggingface.co/models?search=clip). For the language model, we support [OPT](https://huggingface.co/models?search=opt) models, [GPT-Neo](https://huggingface.co/models?search=gpt-neo), [GPT-J](https://huggingface.co/models?search=gptj), and [Pythia](https://huggingface.co/models?search=pythia) models.

## Initializing an OpenFlamingo model
``` python
from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="openai/clip-vit-large-patch14",
    clip_processor_path="openai/clip-vit-large-patch14",
    lang_encoder_path="facebook/opt-1.3b",
    tokenizer_path="facebook/opt-1.3b",
)

# If you have a checkpoint do:
model.load_state_dict(torch.load("path/to/checkpoint.pt"), strict=False)
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
        "https://upload.wikimedia.org/wikipedia/commons/a/ad/Football_in_Bloomington%2C_Indiana%2C_1996.jpg",
        stream=True
    ).raw
)

query_image = Image.open(
    requests.get(
        "https://upload.wikimedia.org/wikipedia/commons/e/e4/Latte_and_dark_coffee.jpg", 
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
vis_x = image_processor(images=[demo_image_one, demo_image_two, query_image], 
 return_tensors="pt")
vis_x = vis_x.unsqueeze(1).unsqueeze(1)


"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["<image>An image of two cats.<|endofchunk|><image>An image of a soccer player"
     " shooting a ball.<|endofchunk|><image>An image of"],
    max_length=128,
    padding=True,
    return_tensors="pt",
)


"""
Step 4: Generate text
"""
generated_text = model.generate(
    vis_x=vis_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20
)

print("Generated text: ", tokenizer.decode(generated_text[0]))
```

# Approach
OpenFlamingo is a multimodal language model that can be used for a variety of tasks. It is trained on a large multimodal dataset (e.g. [Multimodal C4](#multimodal-c4-dataset-mmc4)) and can be used to generate text conditioned on interleaved images/text. For example, OpenFlamingo can be used to generate a caption for an image, or to generate a question given an image and a text passage. The benefit of this approach is that we are able to rapidly adapt to new tasks using in-context training.

## Model architecture
OpenFlamingo seeks to fuse pretrained a vision encoder and a language model using cross attention layers. The model architecture is shown below.

![OpenFlamingo architecture](docs/flamingo.png) 
Credit: [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)

# Multimodal C4 dataset (MMC4)

Coming soon!

# Training
To train a model, modify the following example command:
```
torchrun --nnodes=1 --nproc_per_node=2
train.py 
--run_name flamingo3B
--batch_size_pile 8
--batch_size_laion 16
--train_num_samples_pile 10000
--train_num_samples_laion 20000
--laion_shards s3://s-datasets/laion5b/laion2B-data/{000000..231349}.tar
--pile_shards /fsx/home-anasawadalla/pile/shard-{000000..000169}.tar
--vision_encoder_path openai/clip-vit-large-patch14
--lm_path facebook/opt-1.3b
--dataset_resampled
--num_epochs 10
```

# Evaluation
Before evaluating the model, you will also need to install the coco evaluation package by running the following command:
```
pip install pycocoevalcap
```

To run evaluations on OKVQA you will need to run the following command:
```
import nltk
nltk.download('wordnet')
```

To evaluate the model, use script open_flamingo/eval/evaluate.py with the following arguments:

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
    ... TBD
```

```
@article{Alayrac2022Flamingo,
    title   = {Flamingo: a Visual Language Model for Few-Shot Learning},
    author  = {Jean-Baptiste Alayrac et al},
    year    = {2022}
}
```
