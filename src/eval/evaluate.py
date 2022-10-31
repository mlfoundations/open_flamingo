import logging
import re
import more_itertools
import numpy as np
import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from .image_net_zeroshot import imagenet_classnames
from .utils import compute_vqa_accuracy


def evaluate_imagenet_zeroshot(model, tokenizer, image_processor, batch_size, num_samples_per_class=10, num_classes=1000, device=-1, evaluation_stage="test"):
    """Evaluate a model on ImageNet. Need to set a auth_token using huggingface-cli login to use the dataset.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        num_samples_per_class (int, optional): number of samples to evaluate on (for testing). Defaults to 10. None for entire class split.
        num_classes (int, optional): number of classes to evaluate on (for testing). Defaults to 1000.
        device (int, optional): device to use. Defaults to -1 (cpu).
        evaluation_stage (str, optional): stage to evaluate on. Defaults to "test".

    Returns:
        dict: dictionary of metrics
    """
    logging.info("Evaluating on ImageNet zero-shot...")

    if num_classes > 1000:
        raise ValueError("num_classes must be <= 1000")

    if evaluation_stage == "test":  # As in flamingo paper we use validation set for testing
        dataset = load_dataset(
            "imagenet-1k", split="validation", streaming=True, use_auth_token=True)
    else:  # We use a subset of training set for validation
        dataset = load_dataset("imagenet-1k", split="train",
                               streaming=True, use_auth_token=True)

    eoc_token = "<|endofchunk|>"

    imagenet_class_prompts = [("<image> A photo of an ", f"{label} {eoc_token}") if label[0] in "aeiou" else (
        "<image> A photo of a ", f"{label} {eoc_token}") for label in imagenet_classnames[:num_classes]]

    model.eval()
    model.to(device if device >= 0 else "cpu")

    def evaluate_sample(row):
        # Finds the most likely class for a given image and returns predicted label
        image = image_processor(images=[row["image"]], return_tensors="pt")[
            "pixel_values"]
        label = row["label"]

        per_class_loss = []

        for batch in more_itertools.chunked(imagenet_class_prompts, batch_size):
            encodings = tokenizer(
                batch, padding="max_length", return_tensors="pt", max_length=20)
            encodings = {k: v.to(device if device >= 0 else "cpu")
                         for k, v in encodings.items()}

            # repeat image batch_size times
            images_inputs = image.repeat(len(batch), 1, 1, 1)
            images_inputs.to(device if device >= 0 else "cpu")

            logits = model(images_inputs, encodings["input_ids"],
                           attention_mask=encodings["attention_mask"])[0]

            labels = encodings["input_ids"].clone()
            # convert padding tokens to -100 so they are ignored in loss
            labels[labels == tokenizer.pad_token_id] = -100
            # convert all tokens in prefix until separator to -100 so they are ignored in loss
            for idx in range(len(labels)):
                end_of_prefix = labels[idx][1:].tolist().index(
                    tokenizer.eos_token_id) + 1
                labels[idx, :end_of_prefix + 1] = -100

            # Loss computation from OPT code:
            # Compute per instance loss
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss.view(logits.size(0), logits.size(1))

            # sum loss over all tokens and divide by number of variable tokens
            loss = loss.sum(dim=1) / (labels != -100).sum(dim=1).float()

            per_class_loss.extend(loss)

        return np.argmin(per_class_loss)

    predictions = []

    with torch.inference_mode():
        for label_idx in tqdm(range(num_classes), desc="Running inference"):
            per_class_dataset = dataset.filter(
                lambda example: example['label'] == label_idx).take(num_samples_per_class)
            for row in per_class_dataset:
                predictions.append(evaluate_sample(row) == label_idx)

    return {"accuracy": predictions.count(True) / len(predictions)}


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

    for batch in more_itertools.chunked(tqdm(dataset, desc="Running inference"), batch_size):
        images = image_processor(
            images=[b["image"] for b in batch], return_tensors="pt")["pixel_values"]

        encodings = tokenizer([(f"<image> question:{row['question']} answer:") for row in batch],
                              padding="longest",
                              truncation="only_first",
                              max_length=30,
                              return_tensors="pt")

        with torch.inference_mode():
            outputs = model.greedy_generate(images,
                                            encodings["input_ids"].to(
                                                device if device >= 0 else "cpu"),
                                            attention_mask=encodings["attention_mask"].to(
                                                device if device >= 0 else "cpu"),
                                            max_length=len(
                                                encodings["input_ids"][0]) + max_generation_length,
                                            eoc_token_id=tokenizer.encode(
                                                "<|endofchunk|>")[0],
                                            pad_token_id=tokenizer.pad_token_id)

            predictions.extend([postprocess_generation(
                out) for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)])

    return {"vqa_accuracy": compute_vqa_accuracy(predictions, [row["answers"] for row in dataset])}
