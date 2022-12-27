from contextlib import suppress
import torch
from tqdm import tqdm
from eval.evaluate import evaluate_coco, evaluate_vqa

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype

def get_autocast(precision):
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress

def train_one_epoch(args, model, epoch, train_loader, tokenizer, optimizer, lr_scheduler, device_id, wandb):
    num_batches_per_epoch = train_loader.num_batches
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    for num_steps, batch in tqdm(enumerate(train_loader), disable=args.rank != 0):
        global_step = num_steps + epoch * num_batches_per_epoch  
        images = batch[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        
        # get all zero index elements from pairs in batch[1]
        input_ids = batch[1][0].to(device_id, non_blocking=True)
        attention_mask = batch[1][1].to(device_id, dtype=cast_dtype, non_blocking=True)
        labels = input_ids.clone()
        
        # Do not compute loss on padding tokens
        labels[labels == tokenizer.pad_token_id] = -100
        # Do not compute loss on the media tokens and bos tokens
        labels[:, 0] = -100
        labels[:, 1] = -100
        labels.to(device_id)
        with autocast():
            loss = model(images, input_ids, attention_mask=attention_mask, labels=labels)[0]
        divided_loss = loss / args.gradient_accumulation_steps
        divided_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if (((num_steps+1) % args.gradient_accumulation_steps) == 0) or (num_steps == len(train_loader)-1):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        if args.rank == 0 and args.report_to_wandb:
            wandb.log({"loss": loss.item(), 'global_step': global_step})

def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]
    
    return state_dict

def run_eval_suite(args, model, tokenizer, image_processor, epoch, num_batches, device_id, wandb):
    score = evaluate_coco(model, tokenizer, image_processor,
                                  data_dir=args.eval_coco_data_dir,
                                  batch_size=args.batch_size,
                                  num_samples=5000,
                                  device=device_id,
                                  wandb=wandb if args.report_to_wandb else None,
                                  step=(epoch+1)*num_batches)

    print(f"CIDER score on COCO: {score['cider']}")

    if args.report_to_wandb:
        wandb.log(score, step=(epoch+1) *
                    num_batches, commit=False)

    vqa_score = evaluate_vqa(model, tokenizer, image_processor, benchmark_name="OKVQA",
                                data_dir=args.eval_okvqa_data_dir,
                                batch_size=args.batch_size,
                                num_samples=5000,
                                device=device_id,
                                wandb=wandb if args.report_to_wandb else None,
                                step=(epoch+1)*num_batches)

    print(f"VQA Accuracy on OKVQA: {vqa_score['vqa_accuracy']}")

    if args.report_to_wandb:
        wandb.log(vqa_score, step=(epoch+1) *
                    num_batches, commit=True)

