from contextlib import suppress

import torch
from einops import rearrange
from tqdm import tqdm


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def train_one_epoch(
    args,
    model,
    epoch,
    laion_loader,
    pile_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    num_batches_per_epoch_laion = laion_loader.num_batches
    num_batches_per_epoch_pile = pile_loader.num_batches

    print(f"Number of batches per epoch in laion: {num_batches_per_epoch_laion}")
    print(f"Number of batches per epoch in pile: {num_batches_per_epoch_pile}")

    # assert num_batches_per_epoch_laion == num_batches_per_epoch_pile, "Number of batches in laion and pile datasets must be the same"
    num_batches_per_epoch = num_batches_per_epoch_pile

    print(f"Number of batches per epoch: {num_batches_per_epoch}")

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()
    for num_steps, (batch_laion, batch_pile) in tqdm(
        enumerate(zip(laion_loader, pile_loader)), disable=args.rank != 0
    ):
        global_step = num_steps + epoch * num_batches_per_epoch

        #### LAION FORWARD PASS ####
        images = (
            batch_laion[0]
            .to(device_id, dtype=cast_dtype, non_blocking=True)
            .unsqueeze(1)
            .unsqueeze(1)
        )

        input_ids = batch_laion[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_laion[1][1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == media_token_id] = -100
        labels.to(device_id)

        with autocast():
            loss_laion = model(
                images,
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
        divided_loss_laion = loss_laion / args.gradient_accumulation_steps

        #### PILE FORWARD PASS ####
        images = batch_pile[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2)
        input_ids = torch.stack([x[0] for x in batch_pile[1]]).squeeze(1)
        attention_mask = torch.stack([x[1] for x in batch_pile[1]]).squeeze(1)
        
        # NOTE: irena: expected shape of clip_text_input_ids / attention_mask is (N, I, max_seq_len)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100

        # remove loss for any token before the first <image> token
        for i in range(labels.shape[0]):
            label_idx = 0
            while (
                label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id
            ):
                labels[i][label_idx] = -100
                label_idx += 1

        labels[labels == media_token_id] = -100
        labels.to(device_id)

        with autocast():
            loss_pile = model(
                images,
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
        divided_loss_pile = loss_pile / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_laion * args.loss_multiplier_laion
            + divided_loss_pile * args.loss_multiplier_pile
        )
        loss.backward()
        
        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if args.rank == 0 and args.report_to_wandb:
                wandb.log(
                    {
                        "loss_laion": divided_loss_laion.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )
                wandb.log(
                    {"loss_pile": divided_loss_pile.item(), "global_step": global_step},
                    commit=True,
                )


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict
