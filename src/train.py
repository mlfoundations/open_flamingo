''' Main training script '''

import glob
import os
import argparse

import torch
import torchvision
import wandb
import webdataset as wds
from eval.evaluate import (evaluate_coco,evaluate_vqa)
from tqdm import tqdm

from open_flamingo.factory import create_model_and_transforms
from transformers import get_constant_schedule_with_warmup
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
        "--vision_encoder_path",
        default="openai/clip-vit-large-patch14",
        type=str)
  parser.add_argument(
      "--lm_path",
      default="facebook/opt-350m",
      type=str)
  parser.add_argument("--run_name", type=str, default="opt-350 + vit large")
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--num_workers", type=int, default=1)
  parser.add_argument("--eval_steps", type=int, default=10000)
  parser.add_argument("--save_steps", type=int, default=10000)
  args = parser.parse_args()
    
  model, image_processor, tokenizer = create_model_and_transforms(
    args.vision_encoder_path, args.lm_path)
  
  dist.init_process_group("nccl")
  rank = dist.get_rank()
  print(f"Start running training on rank {rank}.")
  
  if rank == 0:
    wandb.init(project="open-flamingo", entity="anas-awadalla", name=args.run_name)
  
  device_id = rank % torch.cuda.device_count()
  model = model.to(device_id)
  model = DDP(model, device_ids=[device_id])
  
  def preprocess_image(sample):  
    image = image_processor(images=sample, return_tensors="pt")[ "pixel_values"]
    # apply random horizontal flip and color jitter
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    # image = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(image)
    return image

  def preprocess_text(sample):
    tokenizer.padding_side = "right"
    sample = [(f"<image> {s.strip()} <|endofchunk|> {tokenizer.eos_token}") for s in sample]
    text = tokenizer(sample, max_length=64, padding="longest", truncation="only_first", return_tensors="pt")
    return text["input_ids"], text["attention_mask"]
  
  data_chunks_yfcc = glob.glob("/data/yfcc-tmp/yfcc/chunks_merged_1e3/*.tar")
  
  train_dataset = wds.DataPipeline(
      wds.SimpleShardList(data_chunks_yfcc),
      wds.split_by_node,
      wds.split_by_worker,
      wds.tarfile_to_samples(),
      wds.decode("pil"),
      wds.to_tuple("jpg;png", "txt"),
      wds.shuffle(1000),
      wds.batched(args.batch_size, partial=False),
      wds.map_tuple(preprocess_image, preprocess_text)
  ).with_epoch(500000 // torch.cuda.device_count()) # Hardcoded 500k steps to make each process have same num of samples
  
  train_loader = wds.WebLoader(train_dataset,
                               num_workers=args.num_workers,
                               batch_size=None)
  train_loader = train_loader
  
  def get_grouped_params(model):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if "gated_cross_attn_layer" in n and "ff_gate" not in n and "attn_gate" not in n: params_with_wd.append(p)
        else: params_without_wd.append(p)
    return [{"params": params_with_wd, "weight_decay": 0.1},
            {"params": params_without_wd, "weight_decay": 0.0},]
      
  optimizer = torch.optim.AdamW(get_grouped_params(model), lr=1e-4)
  model.train()
  
  lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=5000)

  for num_steps, batch in tqdm(enumerate(train_loader)):        
    images = batch[0].to(device_id)
    
    input_ids = batch[1][0].to(device_id)
    print(f"Step: {num_steps} Rank: {rank} Batch: {input_ids}")
    attention_mask = batch[1][1].to(device_id)
    labels = input_ids.clone()
    
    # Do not compute loss on padding tokens
    labels[labels == tokenizer.pad_token_id] = -100
    # Do not compute loss on the media tokens and bos tokens
    labels[:, 0] = -100
    labels[:, 1] = -100
    labels.to(device_id)
    
    loss = model(images, input_ids, attention_mask=attention_mask, labels=labels)[0]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    
    if rank == 0:
      for idx, layer in enumerate(model.module.lang_encoder.gated_cross_attn_layers):
        wandb.log({f"ff_gate {idx}": layer.ff_gate.item()}, step=num_steps, commit=False)
        wandb.log({f"attn_gate {idx}": layer.attn_gate.item()}, step=num_steps, commit=False)
        
      wandb.log({"loss": loss.item()}, step=num_steps)
      
    if rank == 0 and ((num_steps+1 % args.eval_steps) == 0):
      print(f"Step: {num_steps}, Loss: {loss.item()}")
        
      score = evaluate_coco(model, tokenizer, image_processor,
            data_dir="/data/yfcc-tmp/data/mscoco",
            batch_size=args.batch_size,
            num_samples=5,
            device=device_id,
            wandb=wandb,
            step=num_steps)
      
      wandb.log(score, step=num_steps, commit=False)

      vqa_score = evaluate_vqa(model, tokenizer, image_processor, benchmark_name="OKVQA",
                          data_dir="/mmfs1/gscratch/efml/anasa2/data/ok-vqa/train",
                          batch_size=args.batch_size,
                          num_samples=5,
                          device=device_id,
                          wandb=wandb,
                          step=num_steps)
      
      wandb.log(vqa_score, step=num_steps, commit=True)
      
      model.train()
      
    if rank == 0 and ((num_steps+1 % args.save_steps) == 0):
      if not os.path.exists(args.run_name):
        os.makedirs(args.run_name)
      torch.save(model.state_dict(), f"{args.run_name}/checkpoint_{num_steps}.pt")    
    
    dist.barrier()
    
  if rank == 0:
    if not os.path.exists(args.run_name):
      os.makedirs(args.run_name)
    torch.save(model.state_dict(), f"{args.run_name}/final_weight.pt")    
    
if __name__ == "__main__":
  main()
  