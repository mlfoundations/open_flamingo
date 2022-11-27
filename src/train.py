''' Main training script '''

import glob
import os
import argparse

import torch
import wandb
from eval.evaluate import (evaluate_coco,evaluate_vqa)
from tqdm import tqdm
import numpy as np
import random

from open_flamingo.factory import create_model_and_transforms
from distributed import init_distributed_device, world_info_from_env
from data import get_data
from train_utils import train_one_epoch

from transformers import get_constant_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP

def random_seed(seed=42, rank=0):
  torch.manual_seed(seed + rank)
  np.random.seed(seed + rank)
  random.seed(seed + rank)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--vision_encoder_path",
      default="openai/clip-vit-large-patch14",
      type=str)
  parser.add_argument(
      "--lm_path",
      default="facebook/opt-1.3b",
      type=str)
  parser.add_argument("--run_name", type=str, default="large model test",
                      help="used to name saving directory and wandb run")
  parser.add_argument("--num_epochs", type=int, default=1)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
  parser.add_argument("--do_eval", action="store_true")
  parser.add_argument("--resume_from_checkpoint", type=str, default=None)
  parser.add_argument("--shards", type=str, default="/data/yfcc-tmp/cah/shards/shard_{000000..053008}.tar")
  parser.add_argument("--eval_coco_data_dir", type=str, default="/data/yfcc-tmp/data/mscoco")
  parser.add_argument("--eval_okvqa_data_dir", type=str, default="/mmfs1/gscratch/efml/anasa2/data/ok-vqa/train")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--learning_rate", default=1e-4, type=float)
  parser.add_argument("--warmup_steps", default=5000, type=int)
  parser.add_argument("--weight_decay", default=0.1, type=float)
  parser.add_argument(
    "--precision",
    choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
    default="fp32",
    help="Floating point precision."
  )
  # data args
  parser.add_argument("--workers", type=int, default=1)
  parser.add_argument("--train_num_samples", type=int, default=None)
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
      help="Use horovod for distributed training."
  )
  parser.add_argument(
      "--no-set-device-rank",
      default=False,
      action="store_true",
      help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
  )
  # wandb args
  parser.add_argument(
    "--report_to_wandb",
    default=False,
    action="store_true"
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
  
  if torch.cuda.is_available():
    # This enables tf32 on Ampere GPUs which is only 8% slower than
    # float16 and almost as accurate as float32
    # This was a default in pytorch until 1.12
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


  args = parser.parse_args()
  
  args.dataset_type = "webdataset"
  args.local_rank, args.rank, args.world_size = world_info_from_env()
  
  device_id = init_distributed_device(args)
  
  random_seed(args.seed)
    
  model, image_processor, tokenizer = create_model_and_transforms(
    args.vision_encoder_path, args.lm_path)
  
  random_seed(args.seed, args.rank)

  print(f"Start running training on rank {args.rank}.")
  
  if args.rank == 0 and args.report_to_wandb:
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, config=vars(args))
  
  device_id = args.rank % torch.cuda.device_count()
  model = model.to(device_id)
  
  ddp_model = DDP(model, device_ids=[device_id])
  
  train_dataset = get_data(args, image_processor, tokenizer)
  
  def get_grouped_params(model):
    params_with_wd, params_without_wd = [], []
    apply_decay = lambda x: "gated_cross_attn_layer" in n and "ff_gate" not in n and "attn_gate" not in n and "norm" not in n and "bias" not in n
    for n, p in model.named_parameters():
        if apply_decay(n): params_with_wd.append(p)
        else: params_without_wd.append(p)
    return [{"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},]
      
  optimizer = torch.optim.AdamW(get_grouped_params(ddp_model), lr=args.learning_rate)
  lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
  
  # check if a checkpoint exists for this run
  if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
    checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
    if len(checkpoint_list) == 0:
      print(f"Found no checkpoints for run {args.run_name}.")
    else:
      args.resume_from_checkpoint = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
      print(f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}.")
    
  resume_from_epoch = 0
  if args.resume_from_checkpoint is not None:
    print(f"Loading checkpoint from {args.resume_from_checkpoint}")
    checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
    ddp_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    resume_from_epoch = checkpoint["epoch"]
  
  ddp_model.train()

  for epoch in range(resume_from_epoch, args.num_epochs):
    train_dataset.set_epoch(epoch)
    train_loader = train_dataset.dataloader
    
    train_one_epoch(args=args,
                    model=ddp_model,
                    epoch=epoch,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    train_loader=train_loader,
                    device_id=device_id,
                    wandb=wandb)
      
    if args.do_eval and args.rank == 0:  
      score = evaluate_coco(ddp_model, tokenizer, image_processor,
            data_dir=args.eval_coco_data_dir,
            batch_size=args.batch_size,
            num_samples=5000,
            device=device_id,
            wandb=wandb if args.report_to_wandb else None,
            step=(epoch+1)*len(train_loader))
      
      if args.report_to_wandb:
        wandb.log(score, step=(epoch+1)*len(train_loader), commit=False)

      vqa_score = evaluate_vqa(ddp_model, tokenizer, image_processor, benchmark_name="OKVQA",
                          data_dir=args.eval_okvqa_data_dir,
                          batch_size=args.batch_size,
                          num_samples=5000,
                          device=device_id,
                          wandb=wandb  if args.report_to_wandb else None,
                          step=(epoch+1)*len(train_loader))
      
      if args.report_to_wandb:
        wandb.log(vqa_score, step=(epoch+1)*len(train_loader), commit=True)
      
      ddp_model.train()
      
    if args.rank == 0:
      if not os.path.exists(args.run_name):
        os.makedirs(args.run_name)
        
      checkpoint_dict = {
          "epoch": epoch,
          "model_state_dict": ddp_model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
          "lr_scheduler_state_dict": lr_scheduler.state_dict(),
      }
      
      torch.save(checkpoint_dict, f"{args.run_name}/checkpoint_{epoch}.pt")
      if args.report_to_wandb:
        wandb.save(f"{args.run_name}/checkpoint_{epoch}.pt")
      
  if args.rank == 0:
    if not os.path.exists(args.run_name):
      os.makedirs(args.run_name)
    torch.save(ddp_model.state_dict(), f"{args.run_name}/final_weights.pt")    
    if args.report_to_wandb:
      wandb.save(f"{args.run_name}/final_weights.pt")
      
if __name__ == "__main__":
  main()
  