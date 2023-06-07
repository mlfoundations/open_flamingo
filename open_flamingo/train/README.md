# OpenFlamingo Training
To train OpenFlamingo, please ensure your environment matches that of `environment.yml`.

## Data
Our codebase uses [WebDataset](https://github.com/webdataset/webdataset) to efficiently load `.tar` files containing image and text sequences. We recommend resampling shards with replacement during training using the `--dataset_resampled` flag. 

### LAION-2B Dataset
[LAION-2B](https://arxiv.org/abs/2210.08402) contains 2B web-scraped (image, text) pairs. 
We use [img2dataset](https://github.com/rom1504/img2dataset) to download this dataset into tar files; note that tar files should include pre-downloaded images.

### Multimodal C4 Dataset
We train on the full version of [Multimodal C4 (MMC4)](https://github.com/allenai/mmc4), which includes 103M documents of web-scraped, interleaved image-text sequences. During training, we truncate sequences to 256 text tokens and six images per sequence.

Our codebase expects `.tar` files containing `.json` files. We train with a version of MMC4 which encodes raw images in base64 within these jsons. Our perceiver resampler takes in these images as CLIP *patch* embeddings.

Unfortunately, the public relase of MMC4 does not include raw images in this format; instead the MMC4 team has released CLIP ViT-L/14 *projection features* for the images.
We include scripts in the `scripts/` directory to download MMC4 Core, the CLIP ViT-L/14 projection features, and package these into tarfile format.
You can switch our codebase to train using these projection embeddings via the `--use_mmc4_projection_features` flag.

### ChatGPT-generated sequences
We also train on custom ChatGPT-generated (image, text) sequences with images drawn from LAION. These sequences will be released soon.

## Distributed training
We provide a sample Slurm training script in `scripts/`. 

By default, `train.py` uses Pytorch's [DistributedDataParallel](https://pytorch.org/docs/stable/torch.nn.parallel.DistributedDataParallel.html) for training. 
To use [FullyShardedDataParallel](https://pytorch.org/docs/stable/fsdp.html), use the `--fsdp` flag. 

Some notes on FSDP:

* We recommend using the `--fsdp_use_orig_params` flag. If `--fsdp` is on without this flag, all language model embeddings will be unfrozen during training. (In contrast, the default behavior is to only train the newly added `<image>` and `<|endofchunk|>` tokens.)
    * Note: we've encountered issues using OPT with this flag. Other language models should be compatible.
* Our current FSDP wrapping strategy does not permit training language model embeddings that use tied weights (i.e., tied input / output embeddings). To train such models with FSDP, the language model embeddings must be frozen with the `--freeze_lm_embeddings` flag.

We also implement gradient checkpointing and mixed precision training. Use the `--gradient_checkpointing` and `--precision` arguments respectively.