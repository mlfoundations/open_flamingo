# OpenFlamingo Evaluation Suite

This is the evaluation module of OpenFlamingo. It contains a set of utilities for evaluating multimodal models on various benchmarking datasets.

*This module is a work in progress! We will be updating this README as it develops. In the meantime, if you notice an issue, please file a Bug Report or Feature Request [here](https://github.com/mlfoundations/open_flamingo/issues/new/choose).*

## Supported datasets

|Dataset|Task|Metric|Evaluation method|
|-------|----|------|-----------------|
|[COCO](https://arxiv.org/abs/1405.0312)|Captioning|CIDEr|Generation|
|[Flickr-30K](https://aclanthology.org/Q14-1006/)|Captioning|CIDEr|Generation|
|[VQAv2](https://arxiv.org/abs/1612.00837v3)|VQA|VQA accuracy|Generation|
|[OK-VQA](https://arxiv.org/abs/1906.00067)|VQA|VQA accuracy|Generation|
|[TextVQA](https://arxiv.org/abs/1904.08920)|VQA|VQA accuracy|Generation|
|[VizWiz](https://arxiv.org/abs/1802.08218)|VQA|VQA accuracy|Generation|
|[Hateful Memes](https://arxiv.org/abs/2005.04790)|Classification|ROC AUC|Logprobs|
|[ImageNet](https://arxiv.org/abs/1409.0575)|Classification|Top-1 accuracy|Logprobs|

When evaluating a model using `num_shots` shots, we sample the exemplars from the training split. Performance is evaluated on a disjoint test split, subsampled to `--num_samples` examples (or using the full test split if `--num_samples=-1`).

## Sample scripts
Our codebase uses DistributedDataParallel to parallelize evaluation by default, so please make sure to set the `MASTER_ADDR` and `MASTER_PORT` environment variables or use `torchrun`. We provide a sample Slurm evaluation script in `open_flamingo/open_flamingo/scripts/run_eval.sh`. 

We also support evaluating at a lower precision using the `--precision` flag. We find minimal difference between evaluating at full precision vs. amp_bf16.

To evaluate one of our pretrained checkpoints, we suggest first downloading a local copy of the weights, as follows:

```
# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
HF_TOKEN="<your-hf-token-here>"

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
checkpoint_path= hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", 
  "checkpoint.pt", 
  local_dir="openflamingo/OpenFlamingo-3B-vitl-mpt1b", 
  cache_dir="openflamingo/OpenFlamingo-3B-vitl-mpt1b", 
  local_dir_use_symlinks=False,
  token=HF_TOKEN)
print(checkpoint_path)
## openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt
```

This should place the OpenFlamingo model at the expected location in the evaluation script.

For TextVQA and VizWiz we expect annotations to be formatted differently than the original datasets. We provide the custom annotations in `open_flamingo/open_flamingo/eval/data/`. We have also uploaded all the annotation files in a [huggingface dataset](https://huggingface.co/datasets/openflamingo/eval_benchmark/tree/main) for easy access.

# Evaluating using RICES (Retrieval-based In-Context Example Selection)

We provide the option to evaluate using RICES, which is a method for selecting exemplars from the training set based on image similarity. This method was used in DeepMind's implementation for evaluating on ImageNet, but can be used for any dataset in our evaluation suite.

To use RICES, you must first create features for a benchmark's training set. We provide a script for doing so in `open_flamingo/open_flamingo/scripts/cache_rices_features.py`. This script will extract image features for a given dataset using a given CLIP model checkpoint. For example, to extract features for the COCO training set, you can run:

```bash
python cache_rices_features.py \
  --vision_encoder_path ViT-L-14 \
  --vision_encoder_pretrained openai \
  --batch_size 128 \
  --eval_coco \
  --coco_train_image_dir_path /path/to/coco/train2014 \
  --coco_val_image_dir_path /path/to/coco/val2014 \
  --coco_karpathy_json_path /path/to/coco/dataset_coco.json \
  --coco_annotations_json_path /path/to/coco/annotations/captions_train2014.json \
  --output_dir /path/to/coco/features 
```

This will create a directory at `/path/to/coco/features` containing a file named `coco.pkl` with the extracted features. You can then use this directory to evaluate using RICES by passing the `--rices` flag to the evaluation script, specifying the path to the features directory using the `--cached_demonstration_features` flag, and specifying the vision encoder to use for RICES using the `--rices_vision_encoder_path` and `--rices_vision_encoder_pretrained` flags.
