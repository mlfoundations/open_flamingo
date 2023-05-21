# OpenFlamingo Evaluation Suite

This is the evaluation module of OpenFlamingo. It contains a set of utilities for evaluating multimodal models on various benchmarking datasets.

*This module is a work in progress! We will be updating this README as it develops. In the meantime, if you notice an issue, please file a Bug Report or Feature Request [here](https://github.com/mlfoundations/open_flamingo/issues/new/choose).*

# Running the evaluation suite on OpenFlamingo-9B

The easiest way to run the evaluation suite is by using the script at `open_flamingo/open_flamingo/scripts/run_eval.sh`.

Before running that script, we suggest to download a local copy of the OpenFlamingo model, as follows:

```
# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
HF_TOKEN="<your-hf-token-here>"

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
checkpoint_path= hf_hub_download("openflamingo/OpenFlamingo-9B", 
  "checkpoint.pt", 
  local_dir="openflamingo/OpenFlamingo-9B", 
  cache_dir="openflamingo/OpenFlamingo-9B", 
  local_dir_use_symlinks=False,
  token=HF_TOKEN)
print(checkpoint_path)
## openflamingo/OpenFlamingo-9B/checkpoint.pt
```

This should place the OpenFlamingo model at the expected location in the evaluation script.
