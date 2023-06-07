# OpenFlamingo Evaluation Suite

This is the evaluation module of OpenFlamingo. It contains a set of utilities for evaluating multimodal models on various benchmarking datasets.

*This module is a work in progress! We will be updating this README as it develops. In the meantime, if you notice an issue, please file a Bug Report or Feature Request [here](https://github.com/mlfoundations/open_flamingo/issues/new/choose).*

## Supported datasets

|Dataset|Task|Metric|Evaluation method|
|-------|----|------|-----------------|
|[COCO](https://arxiv.org/abs/1405.0312)|Captioning|CIDEr|Generation|
|[Flickr-30K](https://aclanthology.org/Q14-1006/)|Captioning|CIDEr|Generation|
|[VQAv2](https://arxiv.org/abs/1612.00837v3)|VQA|Accuracy|Generation|
|[OK-VQA](https://arxiv.org/abs/1906.00067)|VQA|Accuracy|Generation|
|[ImageNet](https://arxiv.org/abs/1409.0575)|Classification|Top-1 Accuracy|Logprobs|

When evaluating a model using `num_shots` shots, we sample the exemplars from the training split. Performance is evaluated on a disjoint test split, subsampled to `--num_samples` examples.

## Sample scripts

We provide a sample Slurm evaluation script in `open_flamingo/open_flamingo/scripts/run_eval.sh`.

Before running that script, we suggest to download a local copy of the OpenFlamingo model, as follows:

```
# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
HF_TOKEN="<your-hf-token-here>"

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B", "checkpoint.pt")
checkpoint_path= hf_hub_download("openflamingo/OpenFlamingo-3B", 
  "checkpoint.pt", 
  local_dir="openflamingo/OpenFlamingo-3B", 
  cache_dir="openflamingo/OpenFlamingo-3B", 
  local_dir_use_symlinks=False,
  token=HF_TOKEN)
print(checkpoint_path)
## openflamingo/OpenFlamingo-3B/checkpoint.pt
```

This should place the OpenFlamingo model at the expected location in the evaluation script.
