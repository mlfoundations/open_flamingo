''' Main training script '''

import glob
import io
import os

import imageio
import torch
import torchvision
import wandb
import webdataset as wds
from eval.evaluate import (evaluate_coco, evaluate_imagenet_zeroshot,
                           evaluate_vqa)
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from open_flamingo.factory import create_model_and_transforms
from transformers import get_constant_schedule_with_warmup, get_constant_schedule


def train(model, tokenizer, image_processor, batch_size, num_epochs, lr, device, save_dir=None, eval_steps=1, save_steps=None):
  """
  Trains the model on the training dataset. Report results to wandb.
  
  Args:
      model (_type_): _description_
      batch_size (_type_): _description_
      num_epochs (_type_): _description_
      lr (_type_): _description_
      weight_decay (_type_): _description_
      device (_type_): _description_
      save_dir (_type_): _description_
  """
  wandb.init(project="open-flamingo", entity="anas-awadalla")
  # wandb.config.update({"batch_size": batch_size, "num_epochs": num_epochs, "lr": lr, "weight_decay": weight_decay, "architecture": "CLIP + OPT 125M"})
  
  
  def preprocess_image(sample):  
    image = image_processor(images=sample, return_tensors="pt")[ "pixel_values"]
    # apply random horizontal flip and color jitter
    # image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    # image = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(image)
    return image

  def preprocess_text(sample):
    tokenizer.padding_side = "right"
    sample = [(f"<image> {s.strip()} <|endofchunk|> {tokenizer.eos_token}") for s in sample]
    text = tokenizer(sample, max_length=64, padding="longest", truncation="only_first", return_tensors="pt")
    return text["input_ids"], text["attention_mask"]
  
  def generation_preprocess_text(sample):
    sample = [(f"<image>") for s in sample]
    text = tokenizer(sample, max_length=64, padding="longest", truncation="only_first", return_tensors="pt")
    return text["input_ids"], text["attention_mask"]
  
  data_chunks = glob.glob("/data/yfcc-tmp/yfcc/chunks_merged_1e3/*.tar")

  validation_chunks = data_chunks[0]
  training_chunks = data_chunks[1:]
  
  dataset_size, batch_size = len(training_chunks), batch_size

  train_dataset = wds.DataPipeline(
      wds.SimpleShardList(training_chunks),
      wds.tarfile_to_samples(),
      wds.decode("pil"),
      wds.to_tuple("jpg;png", "txt"),
      wds.batched(batch_size),
      wds.map_tuple(preprocess_image, preprocess_text),
      wds.shuffle(),
  )
  
  train_loader = wds.WebLoader(train_dataset, num_workers=2, batch_size=None)
  
  validation_dataset = wds.DataPipeline(
      wds.SimpleShardList(validation_chunks),
      wds.tarfile_to_samples(),
      wds.decode("pil"),
      wds.to_tuple("jpg;png", "txt"),
      wds.batched(batch_size),
      wds.map_tuple(preprocess_image, generation_preprocess_text),
  )
  
  def get_grouped_params(model):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if "gated_cross_attn_layer" in n and "ff_gate" not in n and "attn_gate" not in n: params_with_wd.append(p)
        else: params_without_wd.append(p)
    return [{"params": params_with_wd, "weight_decay": 0.1},
            {"params": params_without_wd, "weight_decay": 0.0},]
    
      
  optimizer = torch.optim.AdamW(get_grouped_params(model), lr=lr)
  model.to(device)
  model.train()
  
  lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=5000)
    
  num_steps = 0
  for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):        
      images = batch[0].to(device)
      
      input_ids = batch[1][0].to(device)
      attention_mask = batch[1][1].to(device)
      labels = input_ids.clone()
      
      # Do not compute loss on padding tokens
      labels[labels == tokenizer.pad_token_id] = -100
      # Do not compute loss on the media tokens and bos tokens
      labels[:, 0] = -100
      labels[:, 1] = -100
      labels.to(device)
      
      # print a sample from input_ids
      # print("Training sample: ", tokenizer.decode(input_ids[0]))
      
      loss = model(images, input_ids, attention_mask=attention_mask, labels=labels)[0]
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
      
      for idx, layer in enumerate(model.lang_encoder.gated_cross_attn_layers):
        wandb.log({f"ff_gate {idx}": layer.ff_gate.item()}, step=num_steps, commit=False)
        wandb.log({f"attn_gate {idx}": layer.attn_gate.item()}, step=num_steps, commit=False)
        
      wandb.log({"loss": loss.item()}, step=num_steps)
      
      if num_steps % eval_steps == 0:
        print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")
        validation_loader = wds.WebLoader(validation_dataset, num_workers=2, batch_size=None)  
      
        # calculate validation loss
        model.eval()
        with torch.no_grad():
          total_loss = 0
          for j, batch in tqdm(enumerate(validation_loader)):
            images = batch[0].to(device)
            input_ids = batch[1][0].to(device)
            attention_mask = batch[1][1].to(device)
            labels = input_ids.clone()
            
            # Do not compute loss on padding tokens
            labels[labels == tokenizer.pad_token_id] = -100
            # Do not compute loss on the media tokens and bos tokens
            labels[:, 0] = -100
            labels[:, 1] = -100
            labels.to(device)
            
            loss = model(images, input_ids, attention_mask=attention_mask, labels=labels)[0]
            total_loss += loss.item()
            
            # forwards pass and check that the probabilities of the second token
            # output= model(images, input_ids, attention_mask=attention_mask)
            # get maximum probability token
            # next_token_logits = output[0][:, -1, :]
            # next_token = torch.argmax(next_token_logits, dim=-1)
            # print(next_token[:50])
                        
            # try to generate a sample
            generated_ids = model.generate(images, input_ids, attention_mask=attention_mask, max_length=15)
            print("Generated sample: ", tokenizer.batch_decode(generated_ids[:50]))
            
          wandb.log({"validation_loss": total_loss / j}, step=num_steps, commit=False)
          
        # imnet_score = evaluate_imagenet_zeroshot(model, tokenizer, image_processor,num_samples_per_class=10, num_classes=1000, device=0, batch_size=128)
        # wandb.log(imnet_score, step=num_steps)
        
        score = evaluate_coco(model, tokenizer, image_processor,
              data_dir="/data/yfcc-tmp/data/mscoco",
              batch_size=batch_size,
              num_samples=5000,
              device=device,
              wandb=wandb,
              step=num_steps,
              use_prompt=False)
        
        wandb.log(score, step=num_steps, commit=True)
        
        # score = evaluate_coco(model, tokenizer, image_processor,
        #       data_dir="/data/yfcc-tmp/data/mscoco",
        #       batch_size=batch_size,
        #       num_samples=5000,
        #       device=device,
        #       wandb=wandb,
        #       step=num_steps,
        #       use_prompt=True)
        # score = {f"prompt_{k}": v for k, v in score.items()}
        # wandb.log(score, step=num_steps, commit=True)

        # vqa_score = evaluate_vqa(model, tokenizer, image_processor, benchmark_name="OKVQA",
        #                     data_dir="/mmfs1/gscratch/efml/anasa2/data/ok-vqa/train",
        #                     batch_size=batch_size,
        #                     num_samples=5000,
        #                     device=device,
        #                     wandb=wandb,
        #                     step=num_steps,
        #                     use_prompt=False)
        
        # wandb.log(vqa_score, step=num_steps, commit=True)
        
        model.train()
        
      num_steps += 1
      if save_steps is not None and num_steps % save_steps == 0:
        if not os.path.exists(save_dir):
          os.makedirs(save_dir)
        torch.save(model.state_dict(), f"{save_dir}/checkpoint_{i}.pt")    
          
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    torch.save(model.state_dict(), f"{save_dir}/final_weight.pt")    
    
model, image_processor, tokenizer = create_model_and_transforms(
    "openai/clip-vit-large-patch14", "facebook/opt-350m")

train(model, tokenizer, image_processor, batch_size=128, num_epochs=2, lr=1e-4, device=0, save_dir="flamingo-small-no-gate-decay", eval_steps=5000, save_steps=10000)
