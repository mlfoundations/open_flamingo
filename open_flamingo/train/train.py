""" Main training script """

import argparse
import glob
import os
import random
import braceexpand

import numpy as np
import torch
import wandb
from data import get_data
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from train_utils import get_checkpoint, train_one_epoch
from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from open_flamingo import create_model_and_transforms


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vision_encoder_path", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument(
        "--clip_processor_path",
        default=None,
        type=str,
        help="path to clip processor defaults to vision_encoder_path",
    )
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)

    # From previous experiments other opt tokenizers may have a bug
    # so we default to this one in any case they should all be the same.
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="large model test",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size_c4", type=int, default=128)
    parser.add_argument("--batch_size_laion", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument(
        "--laion_shards",
        type=str,
        default="s3://s-datasets/laion5b/laion2B-data/{000000..231349}.tar" #"s3://s-datasets/laion5b/laion2B-data/{000000..231349}.tar",
    )
    parser.add_argument(
        "--c4_shards",
        type=str,
        default="/mmfs1/gscratch/efml/anasa2/data/c4/c4-interleaved-shard-{000000..000100}.tar",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--lr_scheduler", default="constant", type=str)
    parser.add_argument("--loss_multiplier_pile", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_laion", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples_c4", type=int, default=10000)
    parser.add_argument("--train_num_samples_laion", type=int, default=10000)
    parser.add_argument("--dataset_resampled", action="store_true")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--save_checkpoints_to_wandb", default=False, action="store_true"
    )
    parser.add_argument(
        "--wandb_project",
        default="open-flamingo",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        default="anas-awadalla",
        type=str,
    )
    parser.add_argument(
        "--c4_textsim_threshold",
        default=30,
        type=float,
    )

    # if torch.cuda.is_available():
    #   # This enables tf32 on Ampere GPUs which is only 8% slower than
    #   # float16 and almost as accurate as float32
    #   # This was a default in pytorch until 1.12
    #   torch.backends.cuda.matmul.allow_tf32 = True
    #   torch.backends.cudnn.benchmark = True
    #   torch.backends.cudnn.deterministic = False

    args = parser.parse_args()

    if args.laion_shards.startswith("s3"):
        args.laion_shards = f"pipe:aws s3 cp {args.laion_shards} -"

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    assert (args.train_num_samples_laion // args.batch_size_laion) == (
        args.train_num_samples_c4 // args.batch_size_c4
    ), "number of samples per epoch must be equal for pile and laion"

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device_id = init_distributed_device(args)

    random_seed(args.seed)

    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.clip_processor_path
        if args.clip_processor_path
        else args.vision_encoder_path,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        use_local_files=args.offline,
    )
    
    assert model.use_projection_vector is False, "projection vector not desired"

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    device_id = args.rank % torch.cuda.device_count()
    model = model.to(device_id)

    ddp_model = DDP(model, device_ids=[device_id])

    args.shards = list(braceexpand.braceexpand(args.laion_shards))
    args.dataset_type = "image_text"
    args.batch_size = args.batch_size_laion
    args.train_num_samples = args.train_num_samples_laion
    laion_dataset = get_data(args, image_processor, tokenizer)

    c4_shard_urls = []
    with open("/fsx/home-anasawadalla/shard_url_list.txt", "r") as f:
        for idx, line in enumerate(f):
            c4_shard_urls.append(line.strip())
            if idx == 53000:
                break
    
    # remove everything from the shard urls except the shard name
    c4_shard_urls = [shard_url.split("/")[-1] for shard_url in c4_shard_urls]
    # add the s3 prefix
    c4_shard_urls = [f"pipe:aws s3 cp s3://s-laion/flamingo/c4/{shard_url} -" for shard_url in c4_shard_urls]
            
    args.shards = c4_shard_urls

    args.dataset_type = "c4"
    args.batch_size = args.batch_size_c4
    args.train_num_samples = args.train_num_samples_c4
    pile_dataset = get_data(args, image_processor, tokenizer) # need to add in augmentation here according to args.use_media_placement_augmentation

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                "gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x
            )

        for n, p in model.named_parameters():
            # if p.requires_grad:
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)
                
        return [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(get_grouped_params(ddp_model), lr=args.learning_rate)

    total_training_steps = (
        (args.train_num_samples) // (args.batch_size * args.world_size)
    ) * args.num_epochs
    print(f"Total training steps: {total_training_steps}")
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # check if a checkpoint exists for this run
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
            )

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1

    ddp_model.train()
    
    # create a bloom filter to keep track of samples we have seen
    # this is to avoid duplicates
    from bloom_filter2 import BloomFilter
    bloom_filter = BloomFilter(max_elements=15000000, error_rate=0.001)

    for epoch in range(resume_from_epoch, args.num_epochs):
        laion_dataset.set_epoch(epoch)
        laion_loader = laion_dataset.dataloader
        pile_dataset.set_epoch(epoch)
        pile_loader = pile_dataset.dataloader

        train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            laion_loader=laion_loader,
            pile_loader=pile_loader,
            device_id=device_id,
            wandb=wandb,
            bloom_filter=bloom_filter,
        )

        if args.rank == 0:
            if not os.path.exists(args.run_name):
                os.makedirs(args.run_name)

            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(ddp_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }

            print(f"Saving checkpoint to {args.run_name}/checkpoint_{epoch}.pt")
            torch.save(checkpoint_dict, f"{args.run_name}/checkpoint_{epoch}.pt")
            if args.report_to_wandb and args.save_checkpoints_to_wandb:
                wandb.save(f"{args.run_name}/checkpoint_{epoch}.pt")

            if args.delete_previous_checkpoint:
                if epoch > 0:
                    os.remove(f"{args.run_name}/checkpoint_{epoch-1}.pt")

    if args.rank == 0:
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        torch.save(get_checkpoint(ddp_model), f"{args.run_name}/final_weights.pt")
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.run_name}/final_weights.pt")


if __name__ == "__main__":
    main()
