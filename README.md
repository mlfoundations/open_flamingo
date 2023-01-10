# open_flamingo

An open source implementation of DeepMind's [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) model.

# Installation
To install the package, run the following command:
```
pip install -e .
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

## Demo
Alternatively, if you want to play around with the model without worrying about the code, you can use the demo streamlit in the examples directory.

First run:
```
pip install streamlit
pip install huggingface_hub
```

after that you need to authenticate into HuggingFace hub to access model weights:

```
huggingface-cli login
```

Then to run the demo, run the following command from the examples directory:
```
streamlit run demo.py
```

# Training instructions
Currently, we only support OPT models on the language side and CLIP on the vision side.

To train a model, modify the following example command from the open_flamingo/train directory:
```
torchrun --nnodes=1 --nproc_per_node=2
train.py 
--run_name flamingo3B
--batch_size 32
--shards /data/yfcc-tmp/cah/shards/shard_{000000..053008}.tar
--dataset_type image_text
--vision_encoder_path openai/clip-vit-large-patch14
--lm_path facebook/opt-1.3b
--train_num_samples 10000 (Number of samples per epoch)
--dataset_resampled
--num_epochs 500
```

## Additional arguments:

### Evaluation
To evaluate the model, use the --do_eval flag and specify the path to the data for evaluation using the --eval_coco_data_dir or --eval_okvqa_data_dir arguments.

### Wandb
To log to wandb, use the --report_to wandb flag. The run name will be specified using the --run_name argument. To specify the wandb project, use the --wandb_project argument and use wandb_entity to specify the wandb entity.

### Checkpointing
Checkpoints will be saved after each epoch in a directory name after the run_name argument. If there is already a checkpoints in that directory then it will try to resume training from the latest checkpoint in that directory. Additionally you can specify the --delete_previous_checkpoint flag to delete the previous checkpoint when saving the latest checkpoint. This should save you some space.

### Offline training
To run this script in offline mode (i.e. without downloading models from HuggingFace hub and syncing to Wandb), use the --offline flag. Additionally you will want to provide paths to local models and tokenizer using the --vision_encoder_path, clip_processor_path, --lm_path, and --tokenizer_path arguments.
