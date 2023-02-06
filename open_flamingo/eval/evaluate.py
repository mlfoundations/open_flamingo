import argparse
from collections import defaultdict
import json
import os
import uuid

import numpy as np
import torch

from open_flamingo.src.factory import create_model_and_transforms

from .coco_metric import compute_cider, postprocess_captioning_generation
from .eval_datasets import (CaptioningDataset, COCODataset, QADataset,
                            VQAv2Dataset)
from .vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation

parser = argparse.ArgumentParser()
parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
parser.add_argument("--lm_tokenizer_path", type=str,
                    default="facebook/opt-30b")
parser.add_argument("--clip_path", type=str,
                    default="openai/clip-vit-large-patch14")
parser.add_argument("--checkpoint_path", type=str, required=True)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[4, 32])
parser.add_argument("--num_trials", type=int, default=3,
                    help="Number of trials to run for each shot using different demonstrations")
parser.add_argument("--trial_seeds", nargs="+", default=[
                    0, 1, 2], help="Seeds to use for each trial for picking demonstrations and eval sets")

parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--device", type=str, default=0)

# Dataset arguments
parser.add_argument("--coco_image_dir_path", type=str,
                    default="/data/yfcc-tmp/data/mscoco/images")
parser.add_argument("--coco_annotations_json_path", type=str,
                    default="/data/yfcc-tmp/data/mscoco/annotations/captions_train2017.json")
parser.add_argument("--vqav2_image_dir_path", type=str,
                    default="/data/yfcc-tmp/data/vqa-v2/images")
parser.add_argument("--vqav2_questions_json_path", type=str,
                    default="/data/yfcc-tmp/data/vqa-v2/v2_OpenEnded_mscoco_train2014_questions.json")
parser.add_argument("--vqav2_annotations_json_path", type=str,
                    default="/data/yfcc-tmp/data/vqa-v2/v2_OpenEnded_mscoco_train2014_questions.json")

def main():
    args = parser.parse_args()

    # load model
    flamingo, image_processor, tokenizer = create_model_and_transforms(
        args.clip_path,
        args.clip_path,
        args.lm_path,
        args.lm_tokenizer_path,
    )

    flamingo.load_state_dict(torch.load(
        args.checkpoint_path, map_location="cpu"), strict=False)

    print("Evaluating on COCO...")
    for shot in args.shots:
        scores = []
        for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
            cider_score = evaluate_coco(
                model=flamingo,
                tokenizer=tokenizer,
                image_processor=image_processor,
                batch_size=args.batch_size,
                image_dir_path=args.coco_image_dir_path,
                annotations_json_path=args.coco_annotations_json_path,
                num_samples=5000,
                num_shots=shot,
                device=args.device,
                seed=seed,
                num_workers=args.num_workers,
            )
            print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
            scores.append(cider_score)
        print(f"Shots {shot} Mean CIDEr score: {np.mean(scores)}")

    print("Evaluating on VQAv2...")
    for shot in args.shots:
        scores = []
        for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
            vqa_score = evaluate_vqa(
                model=flamingo,
                tokenizer=tokenizer,
                image_processor=image_processor,
                batch_size=args.batch_size,
                num_samples=5000,
                num_shots=shot,
                device=args.device,
                seed=seed,
                image_dir_path=args.vqav2_image_dir_path,
                questions_json_path=args.vqav2_questions_json_path,
                annotations_json_path=args.vqav2_annotations_json_path,
                num_workers=args.num_workers,
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
    image_dir_path,
    annotations_json_path,
    seed=42,
    max_generation_length=10,
    num_beams=3,
    length_penalty=-2.0,
    num_samples=5000,
    num_shots=8,
    device=-1,
    num_workers=4
):
    """Evaluate a model on COCO dataset. 

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        image_dir_path (str, optional): path to the directory containing the images.
        annotations_json_path (str, optional): path to the json file containing the annotations.
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 10.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1.
        num_workers (int, optional): number of workers to use for dataloader. Defaults to 4.

    Returns:
        float: CIDEr score

    """

    full_dataset = COCODataset(
        image_dir_path=image_dir_path, annotations_path=annotations_json_path)

    if num_samples + num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than {len(full_dataset)}")

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + num_shots, replace=False)

    # get in context samples
    in_context_samples = [full_dataset[i] for i in random_indices[:num_shots]]
    eval_dataset = torch.utils.data.Subset(
        full_dataset, random_indices[num_shots:])

    dataset = CaptioningDataset(
        eval_dataset, image_processor, tokenizer, in_context_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()

    predictions = defaultdict()

    for batch in dataloader:
        batch_images = batch["images"].to(device if device >= 0 else "cpu")
        input_ids = batch["input_ids"].to(device if device >= 0 else "cpu")
        attention_mask = batch["attention_mask"].to(
            device if device >= 0 else "cpu")

        with torch.inference_mode():
            outputs = model.module.generate(
                batch_images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        outputs = outputs[:, len(input_ids[0]):]
        new_predictions = [
            postprocess_captioning_generation(out).replace('"', '') for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]

        for i, sample in enumerate(batch):
            predictions[sample["image_id"]] = new_predictions[i]

    predictions = list(
        {v['image_id']: v for v in predictions}.values())

    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f'cocoresults_{random_uuid}.json', 'w') as f:
        f.write(json.dumps(predictions, indent=4))

    metrics = compute_cider(
        result_path=f'cocoresults_{random_uuid}.json', annotations_path=annotations_json_path)

    # delete the temporary file
    os.remove(f'cocoresults_{random_uuid}.json')

    return metrics['CIDEr']*100.0


def evaluate_vqa(
    model,
    tokenizer,
    image_processor,
    batch_size,
    image_dir_path,
    questions_json_path,
    annotations_json_path,
    seed=42,
    max_generation_length=5,
    num_beams=3,
    length_penalty=-2.0,
    num_samples=5000,
    num_shots=8,
    device=-1,
    num_workers=4,
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        image_dir_path (str): path to image directory
        questions_json_path (str): path to questions json file
        annotations_json_path (str): path to annotations json file
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000 samples.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1 (cpu).
        num_workers (int, optional): number of workers to use. Defaults to 4.

    Returns:
        float: accuracy score
    """

    full_dataset = VQAv2Dataset(image_dir_path=image_dir_path,
                                question_path=questions_json_path,
                                annotations_path=annotations_json_path)

    if num_samples + num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than or equal to {len(full_dataset)}")

    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + num_shots, replace=False)

    in_context_samples = [full_dataset[i] for i in random_indices[:num_shots]]
    eval_dataset = torch.utils.data.Subset(
        full_dataset, random_indices[num_shots:])

    dataset = QADataset(eval_dataset, tokenizer,
                        image_processor, in_context_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    predictions = []

    for batch in dataloader:
        batch_images = batch["image"].to(device if device >= 0 else "cpu")
        input_ids = batch["input_ids"].to(device if device >= 0 else "cpu")
        attention_mask = batch["attention_mask"].to(
            device if device >= 0 else "cpu")

        with torch.inference_mode():
            outputs = model.module.generate(
                batch_images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

        outputs = outputs[:, len(input_ids[0]):]
        new_predictions = [
            postprocess_vqa_generation(out) for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]

        predictions.extend([{"answer": p, "question_id": row["question_id"]}
                           for p, row in zip(new_predictions, batch)])
    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f'vqaresults_{random_uuid}.json', 'w') as f:
        f.write(json.dumps(predictions, indent=4))

    acc = compute_vqa_accuracy(
        f'vqaresults_{random_uuid}.json', questions_json_path, annotations_json_path)

    # delete the temporary file
    os.remove(f'vqaresults_{random_uuid}.json')

    return acc
