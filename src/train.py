''' Main training script '''

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
  parser.add_argument("--eval_steps", type=int, default=5000)
  parser.add_argument("--save_steps", type=int, default=10000)
  parser.add_argument("--shards", type=str, default="/data/yfcc-tmp/cah/shards/shard_{000000..053008}.tar")
  parser.add_argument("--eval_coco_data_dir", type=str, default="/data/yfcc-tmp/data/mscoco")
  parser.add_argument("--eval_okvqa_data_dir", type=str, default="/mmfs1/gscratch/efml/anasa2/data/ok-vqa/train")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--learning_rate", default=1e-4, type=float)
  parser.add_argument("--warmup_steps", default=5000, type=int)
  parser.add_argument("--weight_decay", default=0.1, type=float)
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
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name)
  
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
  ddp_model.train()
  
  lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

  num_steps = 0

  for epoch in range(args.num_epochs):
    train_dataset.set_epoch(epoch)
    train_loader = train_dataset.dataloader
    
    for batch in tqdm(train_loader, disable=args.rank != 0): 
      images = batch[0].to(device_id)
      
      # get all zero index elements from pairs in batch[1]
      input_ids = batch[1][0].to(device_id)
      attention_mask = batch[1][1].to(device_id)
      labels = input_ids.clone()
      
      # Do not compute loss on padding tokens
      labels[labels == tokenizer.pad_token_id] = -100
      # Do not compute loss on the media tokens and bos tokens
      labels[:, 0] = -100
      labels[:, 1] = -100
      labels.to(device_id)
      
      loss = ddp_model(images, input_ids, attention_mask=attention_mask, labels=labels)[0]
      divided_loss = loss / args.gradient_accumulation_steps
      divided_loss.backward()
      
      torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
      
      if ((num_steps+1) % args.gradient_accumulation_steps) == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
      
      if args.rank == 0 and args.report_to_wandb:
        wandb.log({"loss": loss.item()}, step=num_steps)
      
      if args.do_eval and args.rank == 0 and (((num_steps+1) % args.eval_steps) == 0):  
        score = evaluate_coco(ddp_model, tokenizer, image_processor,
              data_dir=args.eval_coco_data_dir,
              batch_size=args.batch_size,
              num_samples=5000,
              device=device_id,
              wandb=wandb if args.report_to_wandb else None,
              step=num_steps)
        
        if args.report_to_wandb:
          wandb.log(score, step=num_steps, commit=False)

        vqa_score = evaluate_vqa(ddp_model, tokenizer, image_processor, benchmark_name="OKVQA",
                            data_dir=args.eval_okvqa_data_dir,
                            batch_size=args.batch_size,
                            num_samples=5000,
                            device=device_id,
                            wandb=wandb  if args.report_to_wandb else None,
                            step=num_steps)
        
        if args.report_to_wandb:
          wandb.log(vqa_score, step=num_steps, commit=True)
        
        ddp_model.train()
        
      if args.rank == 0 and (((num_steps+1) % args.save_steps) == 0):
        if not os.path.exists(args.run_name):
          os.makedirs(args.run_name)
        torch.save(ddp_model.state_dict(), f"{args.run_name}/checkpoint_{num_steps}.pt")
        if args.report_to_wandb:
          wandb.save(f"{args.run_name}/checkpoint_{num_steps}.pt")
    
      num_steps+=1
    
  if args.rank == 0:
    if not os.path.exists(args.run_name):
      os.makedirs(args.run_name)
    torch.save(ddp_model.state_dict(), f"{args.run_name}/final_weights.pt")    
    if args.report_to_wandb:
      wandb.save(f"{args.run_name}/final_weights.pt")
      
if __name__ == "__main__":
  main()
  