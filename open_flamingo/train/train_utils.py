import time
from contextlib import suppress
import torch
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os
import shutil
import wandb
import glob
from data_utils import DataInfo
import random
import numpy as np


def train_one_epoch(
    args,
    model,
    epoch,
    datasets: [DataInfo],
    compute_loss_fn: callable,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    """
    Helper function for running one epoch of training.
    Handles logging, calling forward, backward, gradient clipping, and optimizer step.
    Args:
        args (argparse.Namespace): arguments from command line
        model: DDP / FSDP / Deepspeed wrapped model
        epoch (int): epoch number
        datasets (list): list of DataInfos, one for each dataset, to train on
        compute_loss_fn (callable): function that given the model and inputs, calls forward
            and returns a loss
        tokenizer: tokenizer for the language model
        optimizer: optimizer to step
        lr_scheduler: learning rate scheduler
        device_id (int): GPU device ID for this rank
        wandb: wandb object for logging
    """
    # calculate the number of steps in an epoch
    num_batches_per_epoch = datasets[0].dataloader.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    # set up model, autocast, and dtypes
    model.train()
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    # set up logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # loop through the batches in this epoch
    for step_num, batches in tqdm(
        enumerate(zip(*[dataset.dataloader for dataset in datasets])),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)
        global_step = step_num + epoch * num_batches_per_epoch

        # call compute_loss_fn on each dataset; call backward before continuing
        losses_to_log = {}
        batch_metadata_to_log = {}
        for dataset_ix, (images, (input_ids, attention_mask)) in enumerate(batches):
            # unpack the batch and move to device
            images = images.to(device_id, dtype=cast_dtype, non_blocking=True)
            input_ids = input_ids.to(device_id, non_blocking=True)
            attention_mask = attention_mask.to(device_id, non_blocking=True)

            # save some metadata for logging
            batch_metadata_to_log[
                f"{datasets[dataset_ix].name}_num_tokens"
            ] = attention_mask.sum().item()
            batch_metadata_to_log[f"{datasets[dataset_ix].name}_num_images"] = (
                (input_ids == model.media_token_id).sum().item()
            )

            # forward pass
            dataset_loss = compute_loss_fn(
                model=model,
                tokenizer=tokenizer,
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                autocast=autocast,
            )

            losses_to_log[f"loss_{datasets[dataset_ix].name}"] = dataset_loss.item()

            # scale loss and call backward
            dataset_loss *= (
                datasets[dataset_ix].loss_multiplier / args.gradient_accumulation_steps
            )
            if args.deepspeed:
                model.backward(dataset_loss)
            else:
                (dataset_loss).backward()

        # clip gradient norm
        if args.fsdp:
            model.clip_grad_norm_(1.0)
        elif not args.deepspeed:  # deepspeed handles clipping internally
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((step_num + 1) % args.gradient_accumulation_steps) == 0) or (
            step_num == num_batches_per_epoch - 1
        ):
            if args.deepspeed:
                model.step()
            else:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0 and args.report_to_wandb:
                # calculate samples per second
                throughput_metrics = compute_throughput(
                    args,
                    datasets,
                    batch_metadata_to_log,
                    step_time_m,
                )
                wandb.log(
                    {
                        "global_step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        **throughput_metrics,
                        **losses_to_log,
                    },
                    commit=True,
                )
                step_time_m.reset()
                data_time_m.reset()

        # Log loss to console
        if ((step_num + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {step_num+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Losses: "
                + "// ".join([f"{k}: {v:.3f}" for k, v in losses_to_log.items()])
            )


def get_cast_dtype(precision: str):
    """
    Parses the precision argument and returns the dtype to cast inputs to.
    """
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision, cache_enabled=True):
    """
    Parses the precision argument and returns an autocast context manager.
    """
    if precision == "amp":
        return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(
            dtype=torch.bfloat16, cache_enabled=cache_enabled
        )
    else:
        return suppress


def random_seed(seed=42, rank=0):
    """Seed everything"""
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


################################
# Helper functions for logging #
################################


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_throughput(
    args,
    datasets,
    batch_metadata,
    step_time_m,
):
    """
    Computes throughput metrics for logging, including samples per second and tokens per second.
    """
    log = {}
    for dataset in datasets:
        log[f"{dataset.name}_samples_per_second_per_gpu"] = (
            args.gradient_accumulation_steps * dataset.batch_size / step_time_m.val
        )
        log[f"{dataset.name}_samples_per_second"] = (
            log[f"{dataset.name}_samples_per_second_per_gpu"] * args.world_size
        )
        log[f"{dataset.name}_tokens_per_second_per_gpu"] = (
            args.gradient_accumulation_steps
            * batch_metadata[f"{dataset.name}_num_tokens"]
            / step_time_m.val
        )
        log[f"{dataset.name}_tokens_per_second"] = (
            log[f"{dataset.name}_tokens_per_second_per_gpu"] * args.world_size
        )  # this is an estimate based on rank 0
        log[f"{dataset.name}_images_per_second_per_gpu"] = (
            args.gradient_accumulation_steps
            * batch_metadata[f"{dataset.name}_num_images"]
            / step_time_m.val
        )
        log[f"{dataset.name}_images_per_second"] = (
            log[f"{dataset.name}_images_per_second_per_gpu"] * args.world_size
        )  # this is an estimate based on rank 0

    return log


####################################################
# Helper functions for checkpoint loading / saving #
####################################################


def find_most_recent_checkpoint(args):
    """
    Returns the path of the most recent checkpoint for a given run name.
    """
    if args.deepspeed:
        if os.path.exists(f"{args.run_name}/latest"):
            resume_from_checkpoint = args.run_name
            print(f"Found checkpoint {resume_from_checkpoint} for run {args.run_name}.")
        else:
            print(f"Found no checkpoints for run {args.run_name}.")
    else:
        checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(f"Found checkpoint {resume_from_checkpoint} for run {args.run_name}.")
    return resume_from_checkpoint


def load_checkpoint(args, model):
    """
    Loads a (non-Deepspeed) checkpoint into the model and returns the checkpoint + epoch to resume from.
    Does not load the optimizer or learning rate checkpoints, but these are included in the returned checkpoint dict.
    """
    if args.rank == 0:
        print(f"Loading checkpoint from {args.resume_from_checkpoint}")
    checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
    msd = checkpoint.pop("model_state_dict")
    msd = {k.replace("module.", ""): v for k, v in msd.items()}
    resume_from_epoch = checkpoint["epoch"] + 1
    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            **args.fsdp_checkpoint_config,
        )
    model.load_state_dict(msd, False)
    return resume_from_epoch, checkpoint


def load_deepspeed_checkpoint(args, ddp_model):
    """Loads a deepspeed checkpoint and returns the epoch to resume from."""
    if args.rank == 0:
        print(f"Loading checkpoint from {args.resume_from_checkpoint}")
    # We will not pass in a 'tag' and instead rely on 'latest' file in the checkpoint directory
    ddp_model.load_checkpoint(
        load_dir=args.resume_from_checkpoint,  # Note: this is the dir, not the file
        load_module_strict=False,
    )
    # read latest file to get epoch
    latest_file = os.path.join(args.resume_from_checkpoint, "latest")
    with open(latest_file, "r") as f:
        checkpoint_epoch = int(f.read().split("_")[-1])
    return checkpoint_epoch + 1


def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <image> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    # first, remove frozen params
    for name, p in model.named_parameters():
        if "fsdp" in name:
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")
    # second, remove additional duplicate params
    duplicate = lambda k: (
        "lang_model.old_decoder_blocks" in k
        or "lang_model.gated_cross_attn_layers" in k
    )
    filtered_dict = {
        key: value for key, value in state_dict.items() if not duplicate(key)
    }
    return filtered_dict


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """
    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            **args.fsdp_checkpoint_config,
        )
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if args.rank == 0:
        model_state = filter_state_dict_to_trainable(model, model_state)

        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }

        print(f"Saving checkpoint to {args.run_name}/checkpoint_{epoch}.pt")
        torch.save(checkpoint_dict, f"{args.run_name}/checkpoint_{epoch}.pt")
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.run_name}/checkpoint_{epoch}.pt")

        if args.delete_previous_checkpoint:
            if epoch > 0:
                os.remove(f"{args.run_name}/checkpoint_{epoch-1}.pt")


def save_deepspeed_checkpoint(model, epoch, args):
    """
    Save training checkpoint for deepspeed.
    """
    print(f"Saving checkpoint to {args.run_name}")
    model.save_checkpoint(
        save_dir=args.run_name,
        save_latest=True,
        tag=f"epoch_{epoch}",
        exclude_frozen_parameters=not args.gradient_checkpointing, # Save all parameters if gradient checkpointing is enabled
    )

    if args.rank == 0:
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.run_name}/epoch_{epoch}/mp_rank_00_model_states.pt")

        if args.delete_previous_checkpoint:
            if epoch > 0:  # remove checkpoint dir epoch_{epoch-1}
                shutil.rmtree(f"{args.run_name}/epoch_{epoch-1}")
