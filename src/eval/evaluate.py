import logging

import numpy as np
import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import more_itertools
import re

from .utils import compute_vqa_accuracy

from .image_net_zeroshot import imagenet_classnames


def evaluate_imagenet_zeroshot(model, tokenizer, image_processor, batch_size, num_samples=None, num_classes=10, device=-1):
    """Evaluate a model on ImageNet. Need to set a auth_token using huggingface-cli login to use the dataset.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        num_samples (int, optional): number of samples to evaluate on (for testing). Defaults to None (entire testset).
        num_classes (int, optional): number of classes to evaluate on (for testing).
        device (int, optional): device to use. Defaults to -1 (cpu).

    Returns:
        dict: dictionary of metrics
    """
    logging.info("Evaluating on ImageNet zero-shot...")

    if num_classes > 1000:
        raise ValueError("num_classes must be <= 1000")

    dataset = load_dataset("imagenet-1k", split="validation", streaming=True, use_auth_token=True)

    # if num_samples is not None:
    #     dataset = dataset.shuffle().select(range(num_samples))
    # dataset = dataset.map(lambda x: {"image": image_processor(x[0])["pixel_values"], "label": x[1]}, batched=True)

    bos_token = tokenizer.bos_token
    eoc_token = "<|endofchunk|>"

    imagenet_class_prompts = [f"<image> A photo of an {label} {eoc_token}" if label[0] in "aeiou" else f"<image> A photo of a {label} {eoc_token}" for label in imagenet_classnames[:num_classes]]
    imagenet_class_inputs = tokenizer(imagenet_class_prompts, padding="max_length", return_tensors="pt", max_length=20)

    model.eval()
    model.to(device if device >= 0 else "cpu")

    predictions = []

    with torch.inference_mode():
        for row in tqdm(dataset):
            image = image_processor(images=[row["image"]], return_tensors="pt")["pixel_values"]
            label = row["label"]

            per_class_loss = []

            # batched forward pass on imagenet_class_inputs
            for i in tqdm(range(0, len(imagenet_class_inputs["input_ids"]), batch_size)):
                batch = {k: v[i:i+batch_size].to(device if device >= 0 else "cpu") for k, v in imagenet_class_inputs.items()}
                # repeat image batch_size times
                images_inputs = image.repeat(min(batch_size, len(imagenet_class_inputs["input_ids"]) - i), 1, 1, 1)
                logits = model(images_inputs, batch["input_ids"], attention_mask=batch["attention_mask"])[0]
                labels = batch["input_ids"].clone()
                # convert padding tokens to -100 so they are ignored in loss
                labels[labels == tokenizer.pad_token_id] = -100

                # Loss computation from OPT code:
                # Compute per instance loss
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss(reduction="sum")
                # Flatten the tokens
                loss = loss_fct(
                    shift_logits.view(-1, model.lang_encoder.config.vocab_size), shift_labels.view(-1))
                loss = [loss_instance.item() for loss_instance, tokenize_input in zip([loss], tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True))]
                per_class_loss.extend(loss)

            predictions.append(np.argmin(per_class_loss))

            if len(predictions) == num_samples:
                break

    print("accuracy", (predictions == label).mean())
    return {"accuracy": (predictions == label).mean()}

def evaluate_text_vqa(model, tokenizer, image_processor, batch_size, max_generation_length=30, num_samples=None, device=-1):
    """Evaluate a model on TextVQA.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        num_samples (int, optional): number of samples to evaluate on (for testing). Defaults to None (entire test set).
        device (int, optional): device to use. Defaults to -1 (cpu).

    Returns:
        dict: dictionary of metrics
    """
    logging.info("Evaluating on TextVQA...")

    dataset = load_dataset("textvqa", split="validation")

    if num_samples is not None:
        dataset = dataset.shuffle().select(range(num_samples))

    model.eval()
    model.to(device if device >= 0 else "cpu")

    predictions = []
    
    def postprocess_generation(predictions):
        generated_tokens = predictions.split("answer:", 1)[1]
        return re.split("answer:|question:", generated_tokens, 1)[0]

    for batch in more_itertools.chunked(tqdm(dataset), batch_size):
        images = image_processor(
            images=[b["image"] for b in batch], return_tensors="pt")["pixel_values"]

        encodings = tokenizer([(f"<image> question:{row['question']} answer:") for row in batch], 
                              padding="longest",
                              truncation="only_first",
                              max_length=30,
                              return_tensors="pt")

        with torch.inference_mode():
            outputs = model.greedy_generate(images,
                                            encodings["input_ids"].to(device if device >= 0 else "cpu"),
                                            attention_mask=encodings["attention_mask"].to(device if device >= 0 else "cpu"), 
                                            max_length=len(encodings["input_ids"][0]) + max_generation_length,
                                            eoc_token_id=tokenizer.encode("<|endofchunk|>")[0],
                                            pad_token_id=tokenizer.pad_token_id)

            predictions.extend([postprocess_generation(out) for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)])
    
    return {"vqa_accuracy": compute_vqa_accuracy(predictions, [row["answers"] for row in dataset])}
