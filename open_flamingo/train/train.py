""" Main training script """
import argparse
import os
import torch
import wandb
import deepspeed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from open_flamingo import create_model_and_transforms, SUPPORTED_MODEL_FAMILIES
from data import get_data, SUPPORTED_DATASETS
from distributed import (
    init_distributed_device,
    world_info_from_env,
    get_fsdp_config,
    get_deepspeed_config,
)
from train_utils import (
    train_one_epoch,
    random_seed,
    load_deepspeed_checkpoint,
    find_most_recent_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_deepspeed_checkpoint,
)
from losses import (
    NextTokenPrediction,
)
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def main():
    parser = argparse.ArgumentParser()
    # model configuration args
    parser.add_argument(
        "--model_family", default="flamingo", type=str, choices=SUPPORTED_MODEL_FAMILIES
    )
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=1,
        help="how often to add a cross-attention layer after each transformer layer",
    )

    # training args
    parser.add_argument(
        "--run_name",
        type=str,
        default="openflamingo3B",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states. if there exists a checkpoint in the dir named run_name, we will resume from that checkpoint by default. If using deepspeed this should be a directory, not a file.",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="whether to train with gradient/activation checkpointing",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="we define an 'epoch' as a fixed number of examples (train_num_samples_mmc4, train_num_samples_laion), not a pass through the entire dataset",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )

    # data args
    for dataset_name in SUPPORTED_DATASETS:
        parser.add_argument(f"--batch_size_{dataset_name}", type=int, default=128)
        parser.add_argument(
            f"--loss_multiplier_{dataset_name}", type=float, default=1.0
        )
        parser.add_argument(
            f"--train_num_samples_{dataset_name}",
            type=int,
            default=10000,
            help="Number of samples in an 'epoch' for this dataset. Note that train_num_samples/batch_size must be the same for all datasets.",
        )
        parser.add_argument(
            f"--{dataset_name}_shards",
            type=str,
            help="Should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar. If None, we will not train on this dataset.",
        )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset_resampled", action="store_true")
    parser.add_argument(
        "--mmc4_textsim_threshold",
        default=30,
        type=float,
        help="threshold for filtering images in mmc4 based on image-text similarity",
    )
    parser.add_argument(
        "--mmc4_max_num_images",
        default=6,
        type=int,
        help="max number of images per sequence in mmc4 / chatgpt",
    )
    parser.add_argument(
        "--mmc4_min_num_images",
        default=1,
        type=int,
        help="min number of images per sequence in mmc4 / chatgpt",
    )

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

    # fsdp args
    parser.add_argument(
        "--fsdp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training. Not supported for some models, e.g. OPT.",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy", default="full", type=str, choices=["full", "hybrid"]
    )

    # deepspeed args
    parser.add_argument(
        "--deepspeed",
        default=False,
        action="store_true",
        help="Use deepspeed for distributed training.",
    )
    parser.add_argument(
        "--deepspeed_stage",
        default=2,
        type=int,
        help="DeepSpeed distributed training stage. 1: ZeRO-1 (optimizer sharding), 2: ZeRO-2 (optimizer + gradient sharding), 3: ZeRO-3 (optimizer + gradient + model sharding)",
    )

    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )

    args = parser.parse_args()

    # Validate args
    for dataset_name in SUPPORTED_DATASETS:
        shards_path = getattr(args, f"{dataset_name}_shards")
        if shards_path is not None and shards_path.startswith("s3"):
            args.laion_shards = f"pipe:aws s3 cp {args.laion_shards} -"

    datasets_to_train_on = [
        dataset_name
        for dataset_name in SUPPORTED_DATASETS
        if getattr(args, f"{dataset_name}_shards") is not None
    ]
    for i in range(len(datasets_to_train_on) - 1):
        assert getattr(args, f"train_num_samples_{datasets_to_train_on[i]}") // getattr(
            args, f"batch_size_{datasets_to_train_on[i]}"
        ) == getattr(
            args, f"train_num_samples_{datasets_to_train_on[i + 1]}"
        ) // getattr(
            args, f"batch_size_{datasets_to_train_on[i + 1]}"
        ), "Number of batches in each dataloader must be the same"

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.fsdp and args.deepspeed:
        raise ValueError("Select either FSDP or deepspeed for distributed training.")

    if args.fsdp:
        print(
            "Warning: FSDP is experimental and not fully supported. Preference should be given to Deepspeed."
        )
        assert (
            "dev" in torch.__version__ and torch.__version__ > "2.0.1"
        ), "FSDP requires torch nightly > 2.0.1"

    # Set up distributed training
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.deepspeed:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()
        ds_config = get_deepspeed_config(args)
        device_id = args.local_rank
    else:
        device_id = init_distributed_device(args)

    random_seed(args.seed)

    # Initialize model
    additional_kwargs = (
        {"cross_attn_every_n_layers": args.cross_attn_every_n_layers}
        if args.model_family == "flamingo"
        else {}
    )
    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        model_family=args.model_family,
        untie_embeddings=args.fsdp,  # untie embeddings for FSDP
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        **additional_kwargs,
    )
    random_seed(args.seed, args.rank)

    # Initialize wandb logging
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    # Load model checkpoint (on CPU)
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        # if args do not specify a checkpoint to resume from, resume from most recent checkpoint
        args.resume_from_checkpoint = find_most_recent_checkpoint(args)
    if (
        args.resume_from_checkpoint is not None and not args.deepspeed
    ):  # deepspeed handles checkpoint loading
        resume_from_epoch, checkpoint = load_checkpoint(args, model)

    # Initialize FSDP / DDP, and ensure the model is on GPU
    print(f"Initializing distributed training with {args.world_size} GPUs.")
    if args.fsdp:
        model.wrap_fsdp(
            get_fsdp_config(args, device_id), device_id
        )  # moves model to device_id
        ddp_model = model
    elif not args.deepspeed:
        model = model.to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])

    # Initialize gradient checkpointing
    if args.gradient_checkpointing:
        if args.deepspeed:
            raise ValueError(
                "gradient checkpointing currently not supported with deepspeed"
            )
        model.init_gradient_checkpointing()

    # Initialize optimizer
    params_with_wd, params_without_wd = model.group_params_by_weight_decay()
    optimizer = torch.optim.AdamW(
        [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )

    # load optimizer checkpoint
    if args.resume_from_checkpoint is not None and not args.deepspeed:
        osd = checkpoint["optimizer_state_dict"]
        if args.fsdp:
            osd = FSDP.optim_state_dict_to_load(osd, ddp_model, optimizer)
        optimizer.load_state_dict(osd)

    # Initialize datasets
    datasets = [
        get_data(args, image_processor, tokenizer, dataset_name)
        for dataset_name in datasets_to_train_on
    ]
    total_training_steps = (
        (args.train_num_samples_mmc4) // (args.batch_size_mmc4 * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    # Initialize lr scheduler
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # load lr scheduler checkpoint
    if args.resume_from_checkpoint is not None and not args.deepspeed:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    if args.deepspeed:
        ddp_model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            config=ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
        )
        if args.resume_from_checkpoint is not None:
            resume_from_epoch = load_deepspeed_checkpoint(args, ddp_model)

    # Start training!
    for epoch in range(resume_from_epoch, args.num_epochs):
        for dataset in datasets:
            dataset.set_epoch(epoch)
        train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            datasets=datasets,
            compute_loss_fn=NextTokenPrediction(),
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device_id=device_id,
            wandb=wandb,
        )

        if args.deepspeed:
            save_deepspeed_checkpoint(ddp_model, epoch, args)
        else:
            save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)


if __name__ == "__main__":
    main()
