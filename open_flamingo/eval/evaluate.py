import argparse
import importlib
import json
from math import ceil
import os
import random
import uuid
from collections import defaultdict
from typing import Callable

import more_itertools
import numpy as np
import torch
from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import COCOFlickrDataset, VQADataset, ImageNetDataset
from tqdm import tqdm

from open_flamingo.eval.ok_vqa_utils import postprocess_ok_vqa_generation
from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation
from open_flamingo.eval.classification import (
    compute_per_sample_probs,
    compute_per_sample_loss,
)
from open_flamingo.eval.imagenet_utils import (
    openai_imagenet_classnames,
    IMAGENET_1K_CLASS_ID_TO_LABEL,
)
from open_flamingo.eval import eval_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    default=[0],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples", type=int, default=5000, help="Number of samples to evaluate on"
)

parser.add_argument("--batch_size", type=int, default=8)

# Per-dataset evaluation flags
parser.add_argument(
    "--eval_coco",
    action="store_true",
    default=False,
    help="Whether to evaluate on COCO.",
)
parser.add_argument(
    "--eval_vqav2",
    action="store_true",
    default=False,
    help="Whether to evaluate on VQAV2.",
)
parser.add_argument(
    "--eval_ok_vqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on OK-VQA.",
)
parser.add_argument(
    "--eval_imagenet",
    action="store_true",
    default=False,
    help="Whether to evaluate on ImageNet.",
)

parser.add_argument(
    "--eval_flickr30",
    action="store_true",
    default=False,
    help="Whether to evaluate on Flickr30.",
)

# Dataset arguments

## Flickr30 Dataset
parser.add_argument(
    "--flickr_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--flickr_annotations_json_path",
    type=str,
    help="Path to the dataset_flickr30k_coco_style.json file.",
    default=None,
)

## COCO Dataset
parser.add_argument(
    "--coco_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default=None,
)

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_annotations_json_path",
    type=str,
    default=None,
)

## OK-VQA Dataset
parser.add_argument(
    "--ok_vqa_image_dir_path",
    type=str,
    help="Path to the vqav2/train2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_train2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_train2014_annotations.json file.",
    default=None,
)

## Imagenet dataset
parser.add_argument("--imagenet_root", type=str, default="/tmp")

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)


def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"open_flamingo.eval.models.{args.model}")
    eval_model = module.EvalModel(leftovers)

    results = defaultdict(list)

    if args.eval_flickr30:
        print("Evaluating on Flickr30...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_coco_flickr(
                    eval_model=eval_model,
                    batch_size=args.batch_size,
                    image_dir_path=args.flickr_image_dir_path,
                    annotations_json_path=args.flickr_annotations_json_path,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    seed=seed,
                    is_flickr=True,
                )
                print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                scores.append(cider_score)
            print(f"Shots {shot} Mean CIDEr score: {np.mean(scores)}")
            results["flickr30"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )
    results = defaultdict(list)

    if args.eval_coco:
        print("Evaluating on COCO...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_coco_flickr(
                    eval_model=eval_model,
                    batch_size=args.batch_size,
                    image_dir_path=args.coco_image_dir_path,
                    annotations_json_path=args.coco_annotations_json_path,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    seed=seed,
                )
                print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                scores.append(cider_score)
            print(f"Shots {shot} Mean CIDEr score: {np.mean(scores)}")
            results["coco"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.eval_ok_vqa:
        print("Evaluating on OK-VQA...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                ok_vqa_score = evaluate_vqa(
                    eval_model=eval_model,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    seed=seed,
                    image_dir_path=args.ok_vqa_image_dir_path,
                    questions_json_path=args.ok_vqa_questions_json_path,
                    annotations_json_path=args.ok_vqa_annotations_json_path,
                    vqa_dataset="ok_vqa",
                )
                print(f"Shots {shot} Trial {trial} OK-VQA score: {ok_vqa_score}")
                scores.append(ok_vqa_score)
            print(f"Shots {shot} Mean OK-VQA score: {np.mean(scores)}")
            results["ok_vqa"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.eval_vqav2:
        print("Evaluating on VQAv2...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    eval_model=eval_model,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    seed=seed,
                    image_dir_path=args.vqav2_image_dir_path,
                    questions_json_path=args.vqav2_questions_json_path,
                    annotations_json_path=args.vqav2_annotations_json_path,
                    vqa_dataset="vqa",
                )
                print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                scores.append(vqa_score)
            print(f"Shots {shot} Mean VQA score: {np.mean(scores)}")
            results["vqav2"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.eval_imagenet:
        print("Evaluating on ImageNet...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                imagenet_score = evaluate_imagenet(
                    eval_model=eval_model,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    seed=seed,
                    imagenet_root=args.imagenet_root,
                )
                print(
                    f"Shots {shot} Trial {trial} " f"ImageNet score: {imagenet_score}"
                )
                scores.append(imagenet_score)
            print(f"Shots {shot} Mean ImageNet score: {np.mean(scores)}")
            results["imagenet"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def prepare_eval_samples_and_dataset(full_dataset, random_indices, query_set_size):
    # get in context samples
    in_context_samples = [full_dataset[i] for i in random_indices[:query_set_size]]
    eval_dataset = torch.utils.data.Subset(
        full_dataset, random_indices[query_set_size:]
    )
    return in_context_samples, eval_dataset


def get_context_text(
    get_prompt: Callable[[dict], str],
    in_context_samples,
    effective_num_shots,
    num_shots,
) -> str:
    context_text = (
        "".join([get_prompt(s) for s in in_context_samples])
        if effective_num_shots > 0
        else ""
    )

    if num_shots == 0:
        context_text = context_text.replace("<image>", "")
    return context_text


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


def evaluate_coco_flickr(
    eval_model,
    batch_size,
    image_dir_path,
    annotations_json_path,
    seed=42,
    max_generation_length=20,
    num_beams=3,
    length_penalty=-2.0,
    num_samples=5000,
    query_set_size=2048,
    num_shots=8,
    is_flickr=False,
):
    """Evaluate a model on COCO dataset.

    Args:
        eval_model (eval_model.EvalModel): model to evaluate
        batch_size (int): batch size
        image_dir_path (str, optional): path to the directory containing the images.
        annotations_json_path (str, optional): path to the json file containing the annotations.
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 10.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000.
        query_set_size (int, optional): number of samples to use for query set. Defaults to 2048.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        num_workers (int, optional): number of workers to use for dataloader. Defaults to 4.
        is_flickr (bool): defines if that data is COCO or Flickr. Defaults to False (COCO).

    Returns:
        float: CIDEr score

    """

    full_dataset = COCOFlickrDataset(
        image_dir_path=image_dir_path,
        annotations_path=annotations_json_path,
        is_flickr=is_flickr,
    )
    effective_num_shots = num_shots if num_shots > 0 else 2
    random_indices = get_random_indices(num_samples, query_set_size, full_dataset, seed)

    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset,
        random_indices=random_indices,
        query_set_size=query_set_size,
    )

    def get_prompt(sample):
        return f"<image>Output:{sample['caption'].strip()}<|endofchunk|>"

    predictions = defaultdict()

    desc = "Running inference Flickr30" if is_flickr else "Running inference COCO"

    for batch in more_itertools.chunked(tqdm(eval_dataset, desc=desc), batch_size):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch)
        )

        batch_images = []
        batch_text = []
        for i in range(len(batch)):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch[i]["image"]])

            context_text = get_context_text(
                get_prompt,
                in_context_samples=batch_demo_samples[i],
                effective_num_shots=effective_num_shots,
                num_shots=num_shots,
            )
            batch_text.append(f"{context_text}<image>Output:")

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        new_predictions = [
            postprocess_captioning_generation(out).replace('"', "") for out in outputs
        ]

        for i, sample in enumerate(batch):
            predictions[sample["image_id"]] = {
                "caption": new_predictions[i],
            }

    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    results_path = (
        f"flickrresults_{random_uuid}.json"
        if is_flickr
        else f"cocoresults_{random_uuid}.json"
    )
    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_id": k, "caption": predictions[k]["caption"]}
                    for k in predictions
                ],
                indent=4,
            )
        )

    metrics = compute_cider(
        result_path=results_path,
        annotations_path=annotations_json_path,
    )

    # delete the temporary file
    os.remove(results_path)

    return metrics["CIDEr"] * 100.0


def evaluate_vqa(
    eval_model,
    batch_size,
    image_dir_path,
    questions_json_path,
    annotations_json_path,
    seed=42,
    max_generation_length=5,
    num_beams=3,
    length_penalty=-2.0,
    num_samples=5000,
    query_set_size=2048,
    num_shots=8,
    vqa_dataset="vqa",
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0.

    Args:
        eval_model (eval_model.EvalModel): model to evaluate
        batch_size (int): batch size
        image_dir_path (str): path to image directory
        questions_json_path (str): path to questions json file
        annotations_json_path (str): path to annotations json file
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000 samples.
        query_set_size (int, optional): size of the query set. Defaults to 2048.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        num_workers (int, optional): number of workers to use. Defaults to 4.
        vqa_dataset (string): type of vqa dataset: currently supports vqa, ok_vqa. Defaults to vqa.
    Returns:
        float: accuracy score
    """

    full_dataset = VQADataset(
        image_dir_path=image_dir_path,
        question_path=questions_json_path,
        annotations_path=annotations_json_path,
        vqa_dataset=vqa_dataset,
    )

    effective_num_shots = num_shots if num_shots > 0 else 2

    if num_samples + effective_num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than or equal to {len(full_dataset)}"
        )

    random_indices = get_random_indices(num_samples, query_set_size, full_dataset, seed)

    def get_prompt(sample, train=True):
        return f"<image>Question:{sample['question'].strip()} Short Answer:{sample['answers'][0].strip() if train else ''}{'<|endofchunk|>' if train else ''}"

    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset,
        random_indices=random_indices,
        query_set_size=query_set_size,
    )

    predictions = []

    for batch in more_itertools.chunked(
        tqdm(eval_dataset, desc="Running inference"), batch_size
    ):
        batch_demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, effective_num_shots, len(batch)
        )

        batch_images = []
        batch_text = []
        for i in range(len(batch)):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch[i]["image"]])

            context_text = get_context_text(
                get_prompt,
                in_context_samples=batch_demo_samples[i],
                effective_num_shots=effective_num_shots,
                num_shots=num_shots,
            )
            batch_text.append(context_text + get_prompt(batch[i], train=False))

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        process_function = (
            postprocess_vqa_generation
            if vqa_dataset == "vqa"
            else postprocess_ok_vqa_generation
        )

        new_predictions = map(process_function, outputs)

        predictions.extend(
            [
                {"answer": p, "question_id": sample["question_id"]}
                for p, sample in zip(new_predictions, batch)
            ]
        )
    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"{vqa_dataset}results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))

    acc = compute_vqa_accuracy(
        f"{vqa_dataset}results_{random_uuid}.json",
        questions_json_path,
        annotations_json_path,
    )

    # delete the temporary file
    os.remove(f"{vqa_dataset}results_{random_uuid}.json")

    return acc


def evaluate_imagenet(
    eval_model,
    batch_size: int,
    imagenet_root: str,
    seed: int = 42,
    num_samples: int = 5000,
    num_shots: int = 8,
):
    """
    Evaluate a model on ImageNet dataset.

    Args:
        eval_model (eval_model.EvalModel): model to evaluate
        batch_size (int): batch size
        imagenet_root (str): path to imagenet root for the specified split.
        seed (int, optional): random seed. Defaults to 42.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000 samples.
        num_shots (int, optional): number of shots to use. Defaults to 8.

    Returns:
        float: accuracy score
    """
    if not hasattr(eval_model, "model") or not hasattr(eval_model, "tokenizer"):
        raise NotImplementedError(
            "evaluate_imagenet is currently only supported for OpenFlamingo " "models"
        )
    model, tokenizer = eval_model.model, eval_model.tokenizer

    full_dataset = ImageNetDataset(root=imagenet_root)

    effective_num_shots = num_shots if num_shots > 0 else 2

    if num_samples + effective_num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than or equal to "
            f"{len(full_dataset)} "
        )

    random_indices = get_random_indices(
        num_samples, effective_num_shots, full_dataset, seed
    )

    eoc_token = "<|endofchunk|>"
    eoc_token_id = tokenizer.additional_special_tokens_ids[
        tokenizer.additional_special_tokens.index(eoc_token)
    ]

    # Padding from right allows efficient precomputing of context activations.
    tokenizer.padding_side = "right"

    def _imagenet_prompt(class_name, is_context: bool = True):
        """Construct an imagenet prompt for a given label."""
        prefix = "<image>A photo of a "
        if is_context:
            return prefix + class_name.strip()
        else:
            # Not a context example; insert EOS token before the class name
            # so that we can compute the loss on the class name tokens only.
            return prefix + tokenizer.eos_token + class_name.strip()

    def get_imagenet_prompt(x: dict, is_context: bool = True) -> str:
        """Construct an ImageNet prompt for an example, using its label."""
        return _imagenet_prompt(x["class_name"], is_context=is_context)

    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset,
        random_indices=random_indices,
        query_set_size=effective_num_shots,  # NOTE: here we replace query_set_size with effective_num_shots but this is not the ideal evaluation setting.
        # TODO: We should add a query_set_size argument to the function and use it to randomly sample the context for each example.
        # This will be more consistent with the evaluation setting in the paper but will require some reworking of the caching.
    )

    device = eval_model.device

    model.eval()
    # Predictions based on the class target sequence with the maximal
    # predicted probability
    predictions_max_prob = []
    # Predictions based on the class target sequence with the minimal loss on
    # the model logits
    predictions_min_loss = []
    labels = []

    context_images = [x["image"] for x in in_context_samples] if num_shots > 0 else []
    context_text = get_context_text(
        get_imagenet_prompt,
        in_context_samples=in_context_samples,
        effective_num_shots=effective_num_shots,
        num_shots=num_shots,
    )

    # kwargs to use when calling tokenizer
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": 256,
    }

    for i, batch in enumerate(more_itertools.chunked(eval_dataset, batch_size)):
        print(f"processing batch {i} of {ceil(len(eval_dataset) / batch_size)}")
        batch_per_class_probs = []
        batch_per_class_losses = []
        batch_images = eval_model._prepare_images(
            [context_images + [x["image"]] for x in batch]
        )

        # Process the images only once.
        batch_images = batch_images.to(device)
        model._encode_vision_x(vision_x=batch_images)

        # Process the context text only once.
        context_encodings = tokenizer([context_text] * batch_size, **tokenizer_kwargs)
        context_ids = context_encodings["input_ids"].to(device)
        context_len = context_ids.shape[-1]
        context_precomputed = model(
            None,
            context_ids,
            use_cached_vision_x=True,
            clear_conditioned_layers=False,
            use_cache=True,
        )

        # For each ImageNet class, construct the output prompt, compute a
        # forward pass, and store the results.
        for imagenet_class_name in tqdm(openai_imagenet_classnames):
            batch_text = [
                context_text + _imagenet_prompt(imagenet_class_name, False) + eoc_token
            ] * batch_size

            full_batch_encodings = tokenizer(batch_text, **tokenizer_kwargs)

            # full_batch_input_ids has shape [batch_size, seq_len], but we
            # only need to run inference on the [batch_size,
            # context_len:] inputs that have not been precomputed and
            # vary per class.
            full_batch_input_ids = full_batch_encodings["input_ids"].to(device)
            full_batch_attention_mask = full_batch_encodings["attention_mask"].to(
                device
            )

            # Sanity check that the encoded inputs with context are the same
            # as the encoded context alone, for every example in the batch
            assert torch.all(
                context_ids[0, :] == full_batch_input_ids[:, :context_len]
            ).item()

            # Clone the nested structure of the past key values
            past_key_values = tuple(
                [
                    tuple([x.clone() for x in inner])
                    for inner in context_precomputed.past_key_values
                ]
            )

            # Compute the outputs without recomputing context representations.
            outputs = model(
                vision_x=None,
                lang_x=full_batch_input_ids[:, context_len:],
                attention_mask=full_batch_attention_mask,
                use_cached_vision_x=True,
                clear_conditioned_layers=False,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = torch.concat((context_precomputed.logits, outputs.logits), 1)

            per_sample_probs = compute_per_sample_probs(
                encodings=full_batch_encodings,
                tokenizer=tokenizer,
                logits=logits,
                eoc_token_id=eoc_token_id,
            )
            per_sample_loss = compute_per_sample_loss(
                encodings=full_batch_encodings,
                tokenizer=tokenizer,
                logits=logits,
                eoc_token_id=eoc_token_id,
            )
            batch_per_class_probs.append(per_sample_probs.detach())
            batch_per_class_losses.append(per_sample_loss.detach())

        # Tensor of shape [batch_size, 1000] where the [i,j]th element is
        # the (probability or loss) for batch element i on imagenet class j.
        batch_probs = torch.stack(batch_per_class_probs, 1)
        batch_losses = torch.stack(batch_per_class_losses, 1)

        predictions_max_prob.extend(torch.argmax(batch_probs, 1).detach().tolist())
        predictions_min_loss.extend(torch.argmin(batch_losses, 1).detach().tolist())
        labels.extend(x["class_id"] for x in batch)

    acc_max_prob = (np.array(predictions_max_prob) == np.array(labels)).mean()
    acc_min_loss = (np.array(predictions_min_loss) == np.array(labels)).mean()
    print(f"[DEBUG] ImageNet accuracy with max prob method is {acc_max_prob}")
    print(f"[DEBUG] ImageNet accuracy with min loss method is {acc_min_loss}")
    print(f"[DEBUG] printing ImageNet predictions and labels:")
    for yhat_prob, yhat_loss, y in zip(
        predictions_max_prob, predictions_min_loss, labels
    ):
        print(
            " " * 30 + f"label: {IMAGENET_1K_CLASS_ID_TO_LABEL[y]}"
            f"\nprediction (max prob method): "
            f"{IMAGENET_1K_CLASS_ID_TO_LABEL[yhat_prob]}"
            f"\nprediction (min loss method): "
            f"{IMAGENET_1K_CLASS_ID_TO_LABEL[yhat_loss]}\n"
            "#" * 25
        )
    return acc_max_prob


if __name__ == "__main__":
    main()
