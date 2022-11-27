from tqdm import tqdm
import torch

def train_one_epoch(args, model, train_loader, tokenizer, optimizer, lr_scheduler, device_id, wandb):
    model.train()
    for num_steps, batch in tqdm(enumerate(train_loader), disable=args.rank != 0):      
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
        
        loss = model(images, input_ids, attention_mask=attention_mask, labels=labels)[0]
        divided_loss = loss / args.gradient_accumulation_steps
        divided_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if (((num_steps+1) % args.gradient_accumulation_steps) == 0) or (num_steps == len(train_loader)-1):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        if args.rank == 0 and args.report_to_wandb:
            wandb.log({"loss": loss.item()})
        