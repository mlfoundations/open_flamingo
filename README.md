# 🦩 OpenFlamingo
[![PyPI version](https://badge.fury.io/py/open-flamingo.svg)](https://badge.fury.io/py/open-flamingo)
[![Downloads](https://pepy.tech/badge/open-flamingo/week)](https://pepy.tech/project/open-flamingo/week)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Blog post](https://www.anasawadalla.com/blog/openflamingo) | [Twitter thread]() | Paper (coming soon)

Welcome to our open source implementation of DeepMind's [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) model! In this repository, we provide a PyTorch implementation for training and evaluating OpenFlamingo models. We also provide an initial [OpenFlamingo 3B model](#api) trained on the [Multimodal C4 dataset](#multimodal-c4-dataset-mmc4).

This repo is still under development. You can expect us to release better performing and larger Flamingo models soon. If you have any questions, please feel free to open an issue. We also welcome pull requests!

# Table of Contents
- [Installation](#installation)
- [API](#api)
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

# API
You can load a model using the following code:

``` python
from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="openai/clip-vit-base-patch32",
    clip_processor_path="openai/clip-vit-base-patch32",
    lang_encoder_path="facebook/opt-125m",
    tokenizer_path="facebook/opt-125m", 
)

# If you have a checkpoint do:
model.load_state_dict(torch.load("path/to/checkpoint.pt"), strict=False)
```
For how to generate using Flamingo look at examples/example.py


# Multimodal C4 dataset (MMC4)

Coming soon!

# Training
Currently, we only support OPT models on the language side and CLIP on the vision side.

To train a model, modify the following example command from the open_flamingo/train directory:
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

If this is your first time running this command you will need to run:
```
import nltk
nltk.download('punkt')
```
in a python shell before running training.

# Evaluation
Before evaluating the model, you will need to download the COCO and VQAv2 datasets. You will also need to install the coco evaluation package by running the following command:
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

# Team

# Acknowledgments
This code is based on Lucidrains' [flamingo implementation](https://github.com/lucidrains/flamingo-pytorch) and David Hansmair's [flamingo-mini repo](https://github.com/dhansmair/flamingo-mini). Thank you for making your code public!

We would also like to thank [Jean-Baptiste Alayrac](https://www.jbalayrac.com) and [Antoine Miech](https://antoine77340.github.io) for their advice.

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