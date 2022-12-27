import logging

import more_itertools
import numpy as np
import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from .eval_datasets import COCODataset, OKVQADataset
from .image_net_zeroshot import imagenet_classnames
from .utils import (compute_cider, compute_vqa_accuracy,
                    postprocess_captioning_generation,
                    postprocess_vqa_generation)


def evaluate_imagenet_zeroshot(model, tokenizer, image_processor, batch_size, num_samples_per_class=10, num_classes=1000, device=-1, evaluation_stage="validation"):
    """Evaluate a model on ImageNet. Need to set a auth_token using huggingface-cli login to use the dataset.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        num_samples_per_class (int, optional): number of samples to evaluate on (for testing). Defaults to 10. None for entire class split.
        num_classes (int, optional): number of classes to evaluate on (for testing). Defaults to 1000.
        device (int, optional): device to use. Defaults to -1 (cpu).
        evaluation_stage (str, optional): stage to evaluate on. Defaults to "validation".

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
            "pixel_values"].to(device if device >= 0 else "cpu")
        label = row["label"]

        per_class_loss = []

        for batch in more_itertools.chunked(tqdm(imagenet_class_prompts), batch_size):
            encodings = tokenizer(
                batch, padding="longest", return_tensors="pt", max_length=128, truncation=True)
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
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss.view(logits.size(0), logits.size(1))

            # sum loss over all tokens and divide by number of variable tokens
            loss = loss.sum(dim=1) / (labels != -100).sum(dim=1).float()

            per_class_loss.extend(loss.cpu().numpy())

        return np.argmin(per_class_loss)

    predictions = []

    with torch.inference_mode():
        for label_idx in tqdm(range(num_classes), desc="Running inference"):
            per_class_dataset = dataset.filter(
                lambda example: example['label'] == label_idx).take(num_samples_per_class)
            for row in per_class_dataset:
                predictions.append(evaluate_sample(row) == label_idx)

    return {"imnet_accuracy": predictions.count(True) / len(predictions)}


def evaluate_coco(model, tokenizer, image_processor, batch_size, data_dir, max_generation_length=15, num_samples=5000, device=-1, evaluation_stage="validation", wandb=None, step=None):
    """Evaluate a model on COCO dataset.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        data_dir (str): path to data directory
        max_generation_length (int, optional): max generation length. Defaults to 30.
        num_samples (int, optional): number of samples to evaluate on (for testing). Defaults to 5000 samples.
        device (int, optional): device to use. Defaults to -1 (cpu).
        evaluation_stage (str, optional): stage to evaluate on. Can be "validation" or "test". Defaults to "validation".

    Returns:
        dict: dictionary of metrics
    """

    logging.info("Evaluating on COCO...")

    if evaluation_stage not in ["validation", "test"]:
        raise ValueError(
            "evaluation_stage must be either 'validation' or 'test'")

    dataset = COCODataset(data_dir, evaluation_stage)

    # get a random subset of the dataset
    np.random.seed(123)
    random_indices = np.random.choice(
        len(dataset), num_samples+2, replace=False)

    # get two samples from the dataset to use as prompts
    sample_one = dataset[random_indices[0]]
    sample_two = dataset[random_indices[1]]

    dataset = torch.utils.data.Subset(dataset, random_indices[2:])

    model.eval()

    predictions = []
    ground_truths = []

    for idx, batch in enumerate(more_itertools.chunked(tqdm(dataset, desc="Running inference"), batch_size)):
        images = image_processor(
            images=[b["image"] for b in batch], return_tensors="pt")["pixel_values"]

        tokenizer.padding_side = "left"
        encodings = tokenizer([f"<image> " for _ in batch],
                              padding="longest",
                              truncation="only_first",
                              max_length=64,
                              return_tensors="pt")

        with torch.inference_mode():
            if isinstance(model, torch.nn.DataParallel):
                outputs = model.module.generate(images.to(device if device >= 0 else "cpu"),
                                                encodings["input_ids"].to(
                    device if device >= 0 else "cpu"),
                    attention_mask=encodings["attention_mask"].to(
                    device if device >= 0 else "cpu"),
                    max_length=len(
                    encodings["input_ids"][0]) + max_generation_length)
            else:
                outputs = model.generate(images.to(device if device >= 0 else "cpu"),
                                         encodings["input_ids"].to(
                    device if device >= 0 else "cpu"),
                    attention_mask=encodings["attention_mask"].to(
                    device if device >= 0 else "cpu"),
                    max_length=len(
                    encodings["input_ids"][0]) + max_generation_length)

        outputs = outputs[:, len(encodings["input_ids"][0]):]
        new_predictions = [postprocess_captioning_generation(
            out) for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)]

        predictions.extend(new_predictions)
        ground_truths.extend([b["caption"] for b in batch])

        if wandb is not None:
            wandb.log({f"coco_{idx}": wandb.Image(batch[0]["image"], caption=f"Groundtruth: {batch[0]['caption']}\nPrediction: {new_predictions[0]}")},
                      step=step,
                      commit=False)

    return {"cider": compute_cider(predictions, ground_truths)}


def evaluate_vqa(model, tokenizer, image_processor, batch_size, benchmark_name="TextVQA", data_dir=None, max_generation_length=15, num_samples=None, device=-1, evaluation_stage="validation", wandb=None, step=None):
    """Evaluate a model on VQA datasets.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        benchmark_name (str, optional): benchmark to evaluate on can be "TextVQA" or "OKVQA". Defaults to "TextVQA".
        data_dir (str, optional): path to data directory. Defaults to None but needs to be defined for OKVQA.
        max_generation_length (int, optional): max generation length. Defaults to 30.
        num_samples (int, optional): number of samples to evaluate on (for testing). Defaults to None (entire test set).
        device (int, optional): device to use. Defaults to -1 (cpu).
        evaluation_stage (str, optional): stage to evaluate on. Can be "validation" or "test". Defaults to "validation".

    Returns:
        dict: dictionary of metrics
    """

    logging.info(f"Evaluating on {benchmark_name}...")

    if evaluation_stage not in ["validation", "test"]:
        raise ValueError(
            "evaluation_stage must be either 'validation' or 'test'")

    if benchmark_name == "TextVQA":
        if evaluation_stage == "validation":
            raise ValueError(
                "TextVQA does not have a validation set. Please use test set.")
        # validation set is reserved for testing
        dataset = load_dataset("textvqa", split="validation")
    elif benchmark_name == "OKVQA":
        if data_dir is None:
            raise ValueError("data_dir must be defined for OKVQA")
        dataset = OKVQADataset(data_dir, evaluation_stage)
    else:
        raise ValueError("benchmark_name must be either TextVQA or OKVQA")

    # get two samples to use as prompts
    prompt_one = dataset[0]
    prompt_two = dataset[1]

    if num_samples is not None:
        if benchmark_name == "TextVQA":
            dataset = dataset.shuffle().select(range(2, num_samples+2))
        elif benchmark_name == "OKVQA":  # OKVQA is not a huggingface dataset so we can't use select
            dataset = torch.utils.data.Subset(dataset, range(2, num_samples+2))

    model.eval()

    predictions = []
    idx = 0
    for batch in more_itertools.chunked(tqdm(dataset, desc="Running inference"), batch_size):
        images = image_processor(
            images=[b["image"] for b in batch], return_tensors="pt")["pixel_values"]

        tokenizer.padding_side = "left"

        encodings = tokenizer([(f"<image> question:{b['question']} answer:") for b in batch],
                              padding="longest",
                              truncation="only_first",
                              max_length=64,
                              return_tensors="pt")

        with torch.inference_mode():
            if isinstance(model, torch.nn.DataParallel):
                outputs = model.module.generate(images.to(device if device >= 0 else "cpu"),
                                                encodings["input_ids"].to(
                    device if device >= 0 else "cpu"),
                    attention_mask=encodings["attention_mask"].to(
                    device if device >= 0 else "cpu"),
                    max_length=len(
                    encodings["input_ids"][0]) + max_generation_length)
            else:
                outputs = model.generate(images.to(device if device >= 0 else "cpu"),
                                         encodings["input_ids"].to(
                    device if device >= 0 else "cpu"),
                    attention_mask=encodings["attention_mask"].to(
                    device if device >= 0 else "cpu"),
                    max_length=len(
                    encodings["input_ids"][0]) + max_generation_length)

        # get only the generated text
        outputs = outputs[:, len(encodings["input_ids"][0]):]

        new_predictions = [postprocess_vqa_generation(
            out) for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)]

        predictions.extend(new_predictions)

        if wandb is not None:
            wandb.log({f"vqa_{idx}": wandb.Image(
                batch[0]["image"], caption=f"Question: {batch[0]['question']}\nGroundtruth: {batch[0]['answers'][0]}\nPrediction: {new_predictions[0]}")}, step=step, commit=False)

        idx += 1
    return {"vqa_accuracy": compute_vqa_accuracy(predictions, [row["answers"] for row in dataset])}
