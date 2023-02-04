import argparse
import logging

import more_itertools
import numpy as np
from open_flamingo.src.factory import create_model_and_transforms
import torch
from tqdm import tqdm

from .eval_datasets import COCODataset, VQAv2Dataset
from .utils import (compute_cider, compute_vqa_accuracy,
                    postprocess_captioning_generation,
                    postprocess_vqa_generation)

parser = argparse.ArgumentParser()
parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
parser.add_argument("--lm_tokenizer_path", type=str, default="facebook/opt-30b")
parser.add_argument("--clip_path", type=str, default="openai/clip-vit-large-patch14")
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--shots", nargs="+", default=[4, 32])
parser.add_argument("--num_trials", type=int, default=3, help="Number of trials to run for each shot using different demonstrations")
parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--coco_data_dir", type=str, required=True)
parser.add_argument("--vqa_data_dir", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")

def main():
    args = parser.parse_args()

    # load model
    flamingo, image_processor, tokenizer = create_model_and_transforms(
        args.clip_path, 
        args.clip_path,
        args.lm_path,
        args.lm_tokenizer_path,
    )
    
    flamingo.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"), strict=False)
    
    print("Evaluating on COCO...")
    for shot in args.shots:
        scores = []
        for trial in range(args.num_trials):
            cider_score = evaluate_coco(
                flamingo,
                tokenizer,
                image_processor,
                args.batch_size,
                args.coco_data_dir,
                num_samples=5000,
                num_shots=shot,
                device=args.device,
            )
            print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
            scores.append(cider_score)
        print(f"Shots {shot} Mean CIDEr score: {np.mean(scores)}")
        
    print("Evaluating on VQA...")
    for shot in args.shots:
        scores = []
        for trial in range(args.num_trials):
            vqa_score = evaluate_vqa(
                flamingo,
                tokenizer,
                image_processor,
                args.batch_size,
                args.vqa_data_dir,
                num_samples=5000,
                num_shots=shot,
                device=args.device,
            )
            print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
            scores.append(vqa_score)
        print(f"Shots {shot} Mean VQA score: {np.mean(scores)}")
        
if __name__ == "__main__":
    main() 

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
        raise ValueError(
            "evaluation_stage must be either 'validation' or 'test'")
    
    validation_support = COCODataset(data_dir, "validation")
    # select a random subset of num_shots samples from the validation set
    demonstration_indices = np.random.choice(
        len(validation_support), num_shots, replace=False)
    
    # get the demonstration samples
    demonstration_samples = [validation_support[i] for i in demonstration_indices]

    dataset = COCODataset(data_dir, "test")

    if num_samples + num_shots > len(dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than {len(dataset)}")

    # get a random subset of the dataset
    np.random.seed(123)
    random_indices = np.random.choice(
        len(dataset), num_samples + num_shots, replace=False)

    # get in context samples
    in_context_samples = [dataset[i] for i in random_indices[:num_shots]]
    dataset = torch.utils.data.Subset(dataset, random_indices[num_shots:])

    def get_prompt(sample):
        return f"<image>{sample['caption'].strip()}<|endofchunk|>"

    context_text = "".join(
        [get_prompt(sample) for sample in in_context_samples]) if num_shots > 0 else ""
    context_text += "<image>"
    print(context_text)

    if num_shots > 0:
        context_images = image_processor(
            images=[sample["image"] for sample in in_context_samples], return_tensors="pt"
        )["pixel_values"]
        context_images = context_images.unsqueeze(1).unsqueeze(0)
    else:
        context_images = None

    model.eval()

    predictions = []
    ground_truths = []

    for idx, batch in enumerate(more_itertools.chunked(tqdm(dataset, desc="Running inference"), batch_size)):
        batch_images = None

        for b in batch:
            b_image = image_processor(images=[b["image"]], return_tensors="pt")[
                "pixel_values"]
            b_image = b_image.unsqueeze(1).unsqueeze(0)
            b_image = torch.cat([context_images, b_image],
                                dim=1) if num_shots > 0 else b_image

            if batch_images is None:
                batch_images = b_image
            else:
                batch_images = torch.cat([batch_images, b_image], dim=0)

        encodings = tokenizer([context_text for _ in batch],
                              padding=False, truncation=False, return_tensors="pt")

        with torch.inference_mode():
            outputs = model.module.generate(
                batch_images.to(device if device >= 0 else "cpu"),
                encodings["input_ids"].to(device if device >= 0 else "cpu"),
                attention_mask=encodings["attention_mask"].to(
                    device if device >= 0 else "cpu"),
                max_length=len(encodings["input_ids"]
                               [0]) + max_generation_length,
            )

        outputs = outputs[:, len(encodings["input_ids"][0]):]
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
    benchmark_name="VQAv2",
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
        benchmark_name (str, optional): benchmark to evaluate on can be "VQAv2".
        data_dir (str, optional): path to data directory. Defaults to None.
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
        raise ValueError(
            "evaluation_stage must be either 'validation' or 'test'")

    if evaluation_stage == "test":
        raise ValueError(
            "VQAv2 does not have a test set. Please use validation set.")

    # TODO: expose question_path and answers_path args
    dataset = VQAv2Dataset(image_dir=data_dir)

    if num_samples + num_shots > len(dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than or equal to {len(dataset)}")

    # get demonstration samples
    in_context_samples = [dataset[i] for i in range(num_shots)]

    def get_prompt(sample, train=True):
        return f"<image>question:{sample['question']} answer:{sample['answers'][0] if train else ''}{'<|endofchunk|>' if train else ''}"

    context_text = " ".join([get_prompt(s)
                            for s in in_context_samples]) if num_shots > 0 else ""

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
        # not a huggingface dataset so we can't use select
        elif benchmark_name == "OKVQA" or benchmark_name == "VQAv2":
            dataset = torch.utils.data.Subset(
                dataset, range(num_samples, num_samples + num_samples))

    model.eval()

    predictions = []
    idx = 0
    for batch in more_itertools.chunked(tqdm(dataset, desc="Running inference"), batch_size):
        batch_images = None
        batch_text = [context_text + get_prompt(b, train=False) for b in batch]
        print(batch_text)
        for b in batch:
            b_image = image_processor(images=[b["image"]], return_tensors="pt")[
                "pixel_values"]
            b_image = b_image.unsqueeze(1).unsqueeze(0)
            # concatenate context_images and b_image
            b_image = torch.cat([context_images, b_image],
                                dim=1) if num_shots > 0 else b_image

            if batch_images is None:
                batch_images = b_image
            else:
                batch_images = torch.cat([batch_images, b_image], dim=0)

        tokenizer.padding_side = "left"
        encodings = tokenizer(batch_text, padding="longest",
                              truncation=True, max_length=256, return_tensors="pt")

        with torch.inference_mode():
            outputs = model.module.generate(
                batch_images.to(device if device >= 0 else "cpu"),
                encodings["input_ids"].to(device if device >= 0 else "cpu"),
                attention_mask=encodings["attention_mask"].to(
                    device if device >= 0 else "cpu"),
                max_length=len(encodings["input_ids"]
                               [0]) + max_generation_length,
            )

        # get only the generated text
        outputs = outputs[:, len(encodings["input_ids"][0]):]

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
