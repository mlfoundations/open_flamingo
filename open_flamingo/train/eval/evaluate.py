import logging

import more_itertools
import numpy as np
import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from .eval_datasets import COCODataset, OKVQADataset, VQAv2Dataset
from .image_net_zeroshot import imagenet_classnames
from .utils import compute_cider, compute_vqa_accuracy, postprocess_captioning_generation, postprocess_vqa_generation


def evaluate_imagenet_zeroshot(
    model,
    tokenizer,
    image_processor,
    batch_size,
    num_samples_per_class=10,
    num_classes=1000,
    device=-1,
    evaluation_stage="validation",
):
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
        dataset = load_dataset("imagenet-1k", split="validation", streaming=True, use_auth_token=True)
    else:  # We use a subset of training set for validation
        dataset = load_dataset("imagenet-1k", split="train", streaming=True, use_auth_token=True)

    eoc_token = "<|endofchunk|>"

    imagenet_class_prompts = [
        ("<image> A photo of an ", f"{label} {eoc_token}")
        if label[0] in "aeiou"
        else ("<image> A photo of a ", f"{label} {eoc_token}")
        for label in imagenet_classnames[:num_classes]
    ]

    model.eval()
    model.to(device if device >= 0 else "cpu")

    def evaluate_sample(row):
        # Finds the most likely class for a given image and returns predicted label
        image = image_processor(images=[row["image"]], return_tensors="pt")["pixel_values"].to(
            device if device >= 0 else "cpu"
        )
        label = row["label"]

        per_class_loss = []

        for batch in more_itertools.chunked(tqdm(imagenet_class_prompts), batch_size):
            encodings = tokenizer(batch, padding="longest", return_tensors="pt", max_length=128, truncation=True)
            encodings = {k: v.to(device if device >= 0 else "cpu") for k, v in encodings.items()}

            # repeat image batch_size times
            images_inputs = image.repeat(len(batch), 1, 1, 1)
            images_inputs.to(device if device >= 0 else "cpu")

            logits = model(images_inputs, encodings["input_ids"], attention_mask=encodings["attention_mask"])[0]

            labels = encodings["input_ids"].clone()
            # convert padding tokens to -100 so they are ignored in loss
            labels[labels == tokenizer.pad_token_id] = -100
            # convert all tokens in prefix until separator to -100 so they are ignored in loss
            for idx in range(len(labels)):
                end_of_prefix = labels[idx][1:].tolist().index(tokenizer.eos_token_id) + 1
                labels[idx, : end_of_prefix + 1] = -100

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
            per_class_dataset = dataset.filter(lambda example: example["label"] == label_idx).take(
                num_samples_per_class
            )
            for row in per_class_dataset:
                predictions.append(evaluate_sample(row) == label_idx)

    return {"accuracy": predictions.count(True) / len(predictions)}


def evaluate_coco(
    model,
    tokenizer,
    image_processor,
    batch_size,
    data_dir,
    max_generation_length=15,
    num_samples=5000,
    num_shots=8,
    device=-1,
    evaluation_stage="validation",
    wandb=None,
    step=None,
):
    """Evaluate a model on COCO dataset.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        data_dir (str): path to data directory
        max_generation_length (int, optional): max generation length. Defaults to 30.
        num_samples (int, optional): number of samples to evaluate on (for testing). Defaults to 5000 samples.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1 (cpu).
        evaluation_stage (str, optional): stage to evaluate on. Can be "validation" or "test". Defaults to "validation".

    Returns:
        dict: dictionary of metrics
    """

    logging.info("Evaluating on COCO...")

    if evaluation_stage not in ["validation", "test"]:
        raise ValueError("evaluation_stage must be either 'validation' or 'test'")

    dataset = COCODataset(data_dir, evaluation_stage)

    if num_samples + num_shots > len(dataset):
        raise ValueError(f"num_samples + num_shots must be less than {len(dataset)}")

    # get a random subset of the dataset
    np.random.seed(123)
    random_indices = np.random.choice(len(dataset), num_samples + num_shots, replace=False)

    # get in context samples
    in_context_samples = [dataset[i] for i in random_indices[:num_shots]]
    dataset = torch.utils.data.Subset(dataset, random_indices[num_shots:])

    def get_prompt(sample):
        return f"<image>{sample['caption'].strip()}<|endofchunk|>"

    context_text = "".join([get_prompt(sample) for sample in in_context_samples]) if num_shots > 0 else ""
    context_text += "<image>"
    print(context_text)

    if num_shots > 0:
        context_images = image_processor(images=[sample["image"] for sample in in_context_samples], return_tensors="pt")[
            "pixel_values"
        ]
        context_images = context_images.unsqueeze(1).unsqueeze(0)
    else:
        context_images = None

    model.eval()

    predictions = []
    ground_truths = []

    for idx, batch in enumerate(more_itertools.chunked(tqdm(dataset, desc="Running inference"), batch_size)):
        batch_images = None

        for b in batch:
            b_image = image_processor(images=[b["image"]], return_tensors="pt")["pixel_values"]
            b_image = b_image.unsqueeze(1).unsqueeze(0)
            b_image = torch.cat([context_images, b_image], dim=1) if num_shots > 0 else b_image

            if batch_images is None:
                batch_images = b_image
            else:
                batch_images = torch.cat([batch_images, b_image], dim=0)

        encodings = tokenizer([context_text for _ in batch], padding=False, truncation=False, return_tensors="pt")

        with torch.inference_mode():
            outputs = model.module.generate(
                batch_images.to(device if device >= 0 else "cpu"),
                encodings["input_ids"].to(device if device >= 0 else "cpu"),
                attention_mask=encodings["attention_mask"].to(device if device >= 0 else "cpu"),
                max_length=len(encodings["input_ids"][0]) + max_generation_length,
            )

        outputs = outputs[:, len(encodings["input_ids"][0]) :]
        new_predictions = [
            postprocess_captioning_generation(out) for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]

        predictions.extend(new_predictions)
        ground_truths.extend([b["caption"] for b in batch])

        if wandb is not None:
            wandb.log(
                {
                    f"coco_{idx}": wandb.Image(
                        batch[0]["image"],
                        caption=f"Groundtruth: {batch[0]['caption']}\nPrediction: {new_predictions[0]}",
                    )
                },
                step=step,
                commit=False,
            )

    return {"cider_for_coco": compute_cider(predictions, ground_truths)}


def evaluate_vqa(
    model,
    tokenizer,
    image_processor,
    batch_size,
    benchmark_name="TextVQA",
    data_dir=None,
    max_generation_length=15,
    num_samples=5000,
    num_shots=8,
    device=-1,
    evaluation_stage="validation",
    wandb=None,
    step=None,
):
    """Evaluate a model on VQA datasets.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        benchmark_name (str, optional): benchmark to evaluate on can be "TextVQA" or "OKVQA". Defaults to "TextVQA".
        data_dir (str, optional): path to data directory. Defaults to None but needs to be defined for OKVQA.
        max_generation_length (int, optional): max generation length. Defaults to 30.
        num_samples (int, optional): number of samples to evaluate on (for testing). Defaults to 5000.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1 (cpu).
        evaluation_stage (str, optional): stage to evaluate on. Can be "validation" or "test". Defaults to "validation".

    Returns:
        dict: dictionary of metrics
    """

    logging.info(f"Evaluating on {benchmark_name}...")

    if evaluation_stage not in ["validation", "test"]:
        raise ValueError("evaluation_stage must be either 'validation' or 'test'")

    if benchmark_name == "TextVQA":
        if evaluation_stage == "validation":
            raise ValueError("TextVQA does not have a validation set. Please use test set.")
        # validation set is reserved for testing
        dataset = load_dataset("textvqa", split="validation")
    elif benchmark_name == "OKVQA":
        if data_dir is None:
            raise ValueError("data_dir must be defined for OKVQA")
        dataset = OKVQADataset(data_dir, evaluation_stage)
    elif benchmark_name == "VQAv2":
        if evaluation_stage == "test":
            raise ValueError("VQAv2 does not have a test set. Please use validation set.")

        dataset = VQAv2Dataset(image_dir=data_dir)  # TODO: expose question_path and answers_path args
    else:
        raise ValueError("benchmark_name must be either TextVQA or OKVQA")

    if num_samples + num_shots > len(dataset):
        raise ValueError(f"num_samples + num_shots must be less than or equal to {len(dataset)}")

    # get context samples
    in_context_samples = [dataset[i] for i in range(num_shots)]

    def get_prompt(sample, train=True):
        return f"<image>question:{sample['question']} answer:{sample['answer'] if train else ''}{'<|endofchunk|>' if train else ''}"

    context_text = " ".join([get_prompt(s) for s in in_context_samples]) if num_shots > 0 else ""

    if num_shots > 0:
        context_images = image_processor(images=[s["image"] for s in in_context_samples], return_tensors="pt")[
            "pixel_values"
        ]
        context_images = context_images.unsqueeze(1).unsqueeze(0)
    else:
        context_images = None

    if num_samples is not None:
        if benchmark_name == "TextVQA":
            dataset = dataset.shuffle().select(range(num_samples, num_samples + num_samples))
        elif benchmark_name == "OKVQA" or benchmark_name == "VQAv2":  # not a huggingface dataset so we can't use select
            dataset = torch.utils.data.Subset(dataset, range(num_samples, num_samples + num_samples))

    model.eval()

    predictions = []
    idx = 0
    for batch in more_itertools.chunked(tqdm(dataset, desc="Running inference"), batch_size):
        batch_images = None
        batch_text = [context_text + get_prompt(b, train=False) for b in batch]
        print(batch_text)
        for b in batch:
            b_image = image_processor(images=[b["image"]], return_tensors="pt")["pixel_values"]
            b_image = b_image.unsqueeze(1).unsqueeze(0)
            # concatenate context_images and b_image
            b_image = torch.cat([context_images, b_image], dim=1) if num_shots > 0 else b_image

            if batch_images is None:
                batch_images = b_image
            else:
                batch_images = torch.cat([batch_images, b_image], dim=0)

        tokenizer.padding_side = "left"
        encodings = tokenizer(batch_text, padding="longest", truncation=True, max_length=256, return_tensors="pt")

        with torch.inference_mode():
            outputs = model.module.generate(
                batch_images.to(device if device >= 0 else "cpu"),
                encodings["input_ids"].to(device if device >= 0 else "cpu"),
                attention_mask=encodings["attention_mask"].to(device if device >= 0 else "cpu"),
                max_length=len(encodings["input_ids"][0]) + max_generation_length,
            )

        # get only the generated text
        outputs = outputs[:, len(encodings["input_ids"][0]) :]

        new_predictions = [
            postprocess_vqa_generation(out) for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]

        predictions.extend(new_predictions)

        if wandb is not None:
            wandb.log(
                {
                    f"{benchmark_name}_{idx}": wandb.Image(
                        batch[0]["image"],
                        caption=f"Question: {batch[0]['question']}\nGroundtruth: {batch[0]['answers'][0]}\nPrediction: {new_predictions[0]}",
                    )
                },
                step=step,
                commit=False,
            )

        idx += 1
    return {
        f"vqa_accuracy_for_{benchmark_name}": compute_vqa_accuracy(predictions, [row["answers"] for row in dataset])
    }
