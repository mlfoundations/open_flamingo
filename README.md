# 🦩 OpenFlamingo

[![PyPI version](https://badge.fury.io/py/open_flamingo.svg)](https://badge.fury.io/py/open_flamingo)

[Blog post](https://laion.ai/blog/open-flamingo/) | Paper (coming soon)

Welcome to our open source version of DeepMind's [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) model! In this repository, we provide a PyTorch implementation for training and evaluating OpenFlamingo models. We also provide an initial [OpenFlamingo 9B model](https://huggingface.co/openflamingo/OpenFlamingo-9B) trained on a new [Multimodal C4](https://github.com/allenai/mmc4) dataset. Please refer to our blog post for more details.

This repo is still under development, and we hope to release better performing and larger OpenFlamingo models soon. If you have any questions, please feel free to open an issue. We also welcome contributions!

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

To install the package in an existing environment, run 
```
pip install open-flamingo
```

or to create a conda environment for running OpenFlamingo, run
```
conda env create -f environment.yml
```

# Usage
We provide an initial [OpenFlamingo 9B model](https://huggingface.co/openflamingo/OpenFlamingo-9B) using a CLIP ViT-Large vision encoder and a LLaMA-7B language model. In general, we support any [CLIP vision encoder](https://huggingface.co/models?search=clip). For the language model, we support [LLaMA](https://huggingface.co/models?search=llama), [OPT](https://huggingface.co/models?search=opt), [GPT-Neo](https://huggingface.co/models?search=gpt-neo), [GPT-J](https://huggingface.co/models?search=gptj), and [Pythia](https://huggingface.co/models?search=pythia) models.

NOTE: To use LLaMA models, you will need to use this [script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) for converting LLaMA weights to HuggingFace format.

## Initializing an OpenFlamingo model
``` python
from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="<path to llama weights in HuggingFace format>",
    tokenizer_path="<path to llama tokenizer in HuggingFace format>",
    cross_attn_every_n_layers=4
)

# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
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
vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
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
OpenFlamingo is a multimodal language model that can be used for a variety of tasks. It is trained on a large multimodal dataset (e.g. [Multimodal C4](https://github.com/allenai/mmc4)) and can be used to generate text conditioned on interleaved images/text. For example, OpenFlamingo can be used to generate a caption for an image, or to generate a question given an image and a text passage. The benefit of this approach is that we are able to rapidly adapt to new tasks using in-context training.

## Model architecture
OpenFlamingo seeks to fuse a pretrained vision encoder and a language model using cross attention layers. The model architecture is shown below.

![OpenFlamingo architecture](docs/flamingo.png) 
Credit: [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)

# Training
To train a model, modify the following example command, which uses OPT 1.3B as an example LM:
```
torchrun --nnodes=1 --nproc_per_node=4 train.py \
--run_name flamingo3B \
--lm_path facebook/opt-1.3b \
--tokenizer_path facebook/opt-1.3b \
--dataset_resampled \
--laion_shards "/path/to/shards/shard-{0000..0999}.tar" \
--mmc4_shards "/path/to/shards/shard-{0000..0999}.tar" \
--batch_size_mmc4 4 \
--batch_size_laion 8 \
--train_num_samples_mmc4 125000 \
--train_num_samples_laion 250000 \
--loss_multiplier_laion 0.2 \
--workers=6 \
--num_epochs 250 \
--lr_scheduler constant \
--warmup_steps 5000 \
--use_media_placement_augmentation \
--mmc4_textsim_threshold 0.32
```

## Dataset
We expect all our training datasets to be [WebDataset](https://github.com/webdataset/webdataset) shards.
We train our models on the [LAION 2B](https://huggingface.co/datasets/laion/laion2B-en) and [Multimodal C4](https://github.com/allenai/mmc4) datasets. By default the LAION 2B dataset is in WebDataset format if it is downloaded using the [img2dataset tool](https://github.com/rom1504/img2dataset) and Multimodal C4 can be converted to the WebDataset format using this [script](https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/train/convert_mmc4_to_wds.py).


# Evaluation
We currently support running evaluations on [COCO](https://cocodataset.org/#home), [VQAv2](https://visualqa.org/index.html), [OKVQA](https://okvqa.allenai.org), [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), and [ImageNet](https://image-net.org/index.php). Note that currently these evaluations are ran in validation mode (as specified in the Flamingo paper). We will be adding support for running evaluations in test mode in the future.


To run evaluations on OKVQA you will need to run the following command:
```
import nltk
nltk.download('wordnet')
```

To evaluate the model, run the script at `open_flamingo/scripts/run_eval.sh`

# Future plans
- [ ] Add support for video input
- [ ] Release better performing and larger OpenFlamingo models
- [ ] Expand our evaluation suite
- [ ] Add support for FSDP training

# Team

OpenFlamingo is developed by:

[Anas Awadalla](https://anas-awadalla.streamlit.app/), [Irena Gao](https://i-gao.github.io/), [Joshua Gardner](https://homes.cs.washington.edu/~jpgard/), [Jack Hessel](https://jmhessel.com/), [Yusuf Hanafy](https://www.linkedin.com/in/yusufhanafy/), [Wanrong Zhu](https://wanrong-zhu.com/), [Kalyani Marathe](https://sites.google.com/uw.edu/kalyanimarathe/home?authuser=0), [Yonatan Bitton](https://yonatanbitton.github.io/), [Samir Gadre](https://sagadre.github.io/), [Jenia Jitsev](https://scholar.google.de/citations?user=p1FuAMkAAAAJ&hl=en), [Simon Kornblith](https://simonster.com/), [Pang Wei Koh](https://koh.pw/), [Gabriel Ilharco](https://gabrielilharco.com/), [Mitchell Wortsman](https://mitchellnw.github.io/), [Ludwig Schmidt](https://people.csail.mit.edu/ludwigs/).

The team is primarily from the University of Washington, Stanford, AI2, UCSB, and Google.

# Acknowledgments
This code is based on Lucidrains' [flamingo implementation](https://github.com/lucidrains/flamingo-pytorch) and David Hansmair's [flamingo-mini repo](https://github.com/dhansmair/flamingo-mini). Thank you for making your code public! We also thank the [OpenCLIP](https://github.com/mlfoundations/open_clip) team as we use their data loading code and take inspiration from their library design.

We would also like to thank [Jean-Baptiste Alayrac](https://www.jbalayrac.com) and [Antoine Miech](https://antoine77340.github.io) for their advice, [Rohan Taori](https://www.rohantaori.com/), [Nicholas Schiefer](https://nicholasschiefer.com/), [Deep Ganguli](https://hai.stanford.edu/people/deep-ganguli), [Thomas Liao](https://thomasliao.com/), [Tatsunori Hashimoto](https://thashim.github.io/), and [Nicholas Carlini](https://nicholas.carlini.com/) for their help with assessing the safety risks of our release, and to [Stability AI](https://stability.ai) for providing us with compute resources to train these models.

# Citing
If you found this repository useful, please consider citing:

```
@software{anas_awadalla_2023_7733589,
  author = {Awadalla, Anas and Gao, Irena and Gardner, Joshua and Hessel, Jack and Hanafy, Yusuf and Zhu, Wanrong and Marathe, Kalyani and Bitton, Yonatan and Gadre, Samir and Jitsev, Jenia and Kornblith, Simon and Koh, Pang Wei and Ilharco, Gabriel and Wortsman, Mitchell and Schmidt, Ludwig},
  title = {OpenFlamingo},
  month        = mar,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.1.1},
  doi          = {10.5281/zenodo.7733589},
  url          = {https://doi.org/10.5281/zenodo.7733589}
}
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
