import argparse
import importlib
import json
import os
import random
import uuid
from collections import defaultdict

from einops import repeat
import more_itertools
import numpy as np
import torch


from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import CaptionDataset, VQADataset, ImageNetDataset
from tqdm import tqdm


from eval_datasets import VQADataset, ImageNetDataset
from open_flamingo.eval.imagenet_utils import (
    openai_imagenet_classnames,
    IMAGENET_1K_CLASS_ID_TO_LABEL,
)

from eval_model import BaseEvalModel

from open_flamingo.eval.ok_vqa_utils import postprocess_ok_vqa_generation
from open_flamingo.src.flamingo import Flamingo
from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation

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
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples", type=int, default=5000, help="Number of samples to evaluate on"
)
parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Size of demonstration query set"
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
    "--eval_vizwiz",
    action="store_true",
    default=False,
    help="Whether to evaluate on VizWiz.",
)
parser.add_argument(
    "--eval_textvqa",
    action="store_true",
    default=False,
    help="Whether to evaluate on TextVQA.",
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
    "--flickr_karpathy_json_path",
    type=str,
    help="Path to the dataset_flickr30k.json file.",
    default=None,
)
parser.add_argument(
    "--flickr_annotations_json_path",
    type=str,
    help="Path to the dataset_flickr30k_coco_style.json file.",
)
## COCO Dataset
parser.add_argument(
    "--coco_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_val_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_karpathy_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default=None,
)

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_annotations_json_path",
    type=str,
    default=None,
)

## OK-VQA Dataset
parser.add_argument(
    "--ok_vqa_train_image_dir_path",
    type=str,
    help="Path to the vqav2/train2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_train_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_train2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_train_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_train2014_annotations.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_image_dir_path",
    type=str,
    help="Path to the vqav2/val2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_val2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_val2014_annotations.json file.",
    default=None,
)

## VizWiz Dataset
parser.add_argument(
    "--vizwiz_train_image_dir_path",
    type=str,
    help="Path to the vizwiz train images directory.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_image_dir_path",
    type=str,
    help="Path to the vizwiz test images directory.",
    default=None,
)
parser.add_argument(
    "--vizwiz_train_questions_json_path",
    type=str,
    help="Path to the vizwiz questions json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_train_annotations_json_path",
    type=str,
    help="Path to the vizwiz annotations json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_questions_json_path",
    type=str,
    help="Path to the vizwiz questions json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_annotations_json_path",
    type=str,
    help="Path to the vizwiz annotations json file.",
    default=None,
)

# TextVQA Dataset
parser.add_argument(
    "--textvqa_image_dir_path",
    type=str,
    help="Path to the textvqa images directory.",
    default=None,
)
parser.add_argument(
    "--textvqa_train_questions_json_path",
    type=str,
    help="Path to the textvqa questions json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_train_annotations_json_path",
    type=str,
    help="Path to the textvqa annotations json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_test_questions_json_path",
    type=str,
    help="Path to the textvqa questions json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_test_annotations_json_path",
    type=str,
    help="Path to the textvqa annotations json file.",
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

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    if args.eval_flickr30:
        print("Evaluating on Flickr30k...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_captioning(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="flickr",
                )
                print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                scores.append(cider_score)
            print(f"Shots {shot} Mean CIDEr score: {np.mean(scores)}")
            results["flickr30"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.eval_coco:
        print("Evaluating on COCO...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_captioning(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="coco",
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
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="ok_vqa",
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
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vqav2",
                )
                print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                scores.append(vqa_score)
            print(f"Shots {shot} Mean VQA score: {np.mean(scores)}")
            results["vqav2"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.eval_vizwiz:
        print("Evaluating on VizWiz...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vizwiz_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vizwiz",
                )
                print(f"Shots {shot} Trial {trial} VizWiz score: {vizwiz_score}")
                scores.append(vizwiz_score)
            print(f"Shots {shot} Mean VizWiz score: {np.mean(scores)}")
            results["vizwiz"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)}
            )

    if args.eval_textvqa:
        print("Evaluating on TextVQA...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                textvqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="textvqa",
                )
                print(f"Shots {shot} Trial {trial} TextVQA score: {textvqa_score}")
                scores.append(textvqa_score)
            print(f"Shots {shot} Mean TextVQA score: {np.mean(scores)}")
            results["textvqa"].append(
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
            f"num_samples + query_set_size must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def get_query_set(train_dataset, query_set_size, seed):
    np.random.seed(seed)
    query_set = np.random.choice(len(train_dataset), query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, seed):
    np.random.seed(seed)
    random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    return torch.utils.data.Subset(test_dataset, random_indices)


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def evaluate_captioning(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    max_generation_length: int = 20,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    num_shots: int = 8,
    dataset_name: str = "coco",
):
    """Evaluate a model on COCO dataset.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 20.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr". Defaults to "coco".
    Returns:
        float: CIDEr score

    """

    if dataset_name == "coco":
        image_train_dir_path = args.coco_train_image_dir_path
        image_val_dir_path = args.coco_val_image_dir_path
        annotations_path = args.coco_karpathy_json_path
    elif dataset_name == "flickr":
        image_train_dir_path = (
            args.flickr_image_dir_path
        )  # Note: calling this "train" for consistency with COCO but Flickr only has one split for images
        image_val_dir_path = None
        annotations_path = args.flickr_karpathy_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=True,
        dataset_name=dataset_name,
    )

    test_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataset = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        seed,
    )

    in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)

    predictions = defaultdict()

    for batch in more_itertools.chunked(
        tqdm(test_dataset, desc=f"Running inference {dataset_name.upper()}"),
        args.batch_size,
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

            context_text = "".join(
                [
                    eval_model.get_caption_prompt(caption=x["caption"].strip())
                    for x in batch_demo_samples[i]
                ]
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(context_text + eval_model.get_caption_prompt())

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
    results_path = f"{dataset_name}results_{uuid.uuid4()}.json"

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
        annotations_path=args.coco_annotations_json_path
        if dataset_name == "coco"
        else args.flickr_annotations_json_path,
    )

    # delete the temporary file
    os.remove(results_path)

    return metrics["CIDEr"] * 100.0


def evaluate_vqa(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    num_shots: int = 8,
    dataset_name: str = "vqav2",
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
    Returns:
        float: accuracy score
    """

    if dataset_name == "ok_vqa":
        train_image_dir_path = args.ok_vqa_train_image_dir_path
        train_questions_json_path = args.ok_vqa_train_questions_json_path
        train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        test_image_dir_path = args.ok_vqa_test_image_dir_path
        test_questions_json_path = args.ok_vqa_test_questions_json_path
        test_annotations_json_path = args.ok_vqa_test_annotations_json_path
    elif dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
    elif dataset_name == "vizwiz":
        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path
    elif dataset_name == "textvqa":
        train_image_dir_path = args.textvqa_image_dir_path
        train_questions_json_path = args.textvqa_train_questions_json_path
        train_annotations_json_path = args.textvqa_train_annotations_json_path
        test_image_dir_path = args.textvqa_image_dir_path
        test_questions_json_path = args.textvqa_test_questions_json_path
        test_annotations_json_path = args.textvqa_test_annotations_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = VQADataset(
        image_dir_path=train_image_dir_path,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=dataset_name,
    )

    test_dataset = VQADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataset = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        seed,
    )

    in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)
    predictions = []

    for batch in more_itertools.chunked(
        tqdm(test_dataset, desc=f"Running inference {dataset_name.upper()}"),
        args.batch_size,
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

            context_text = "".join(
                [
                    eval_model.get_vqa_prompt(
                        question=x["question"], answer=x["answers"][0]
                    )
                    for x in batch_demo_samples[i]
                ]
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(
                context_text + eval_model.get_vqa_prompt(question=batch[i]["question"])
            )

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        process_function = (
            postprocess_ok_vqa_generation
            if dataset_name == "ok_vqa"
            else postprocess_vqa_generation
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
    with open(f"{dataset_name}results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))

    acc = compute_vqa_accuracy(
        f"{dataset_name}results_{random_uuid}.json",
        test_questions_json_path,
        test_annotations_json_path,
    )

    # delete the temporary file
    os.remove(f"{dataset_name}results_{random_uuid}.json")

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
        eval_model (BaseEvalModel): model to evaluate
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
    np.random.seed(seed)
    model, tokenizer = eval_model.model, eval_model.tokenizer
    assert isinstance(model, Flamingo)

    train_dataset = ImageNetDataset(os.path.join(imagenet_root, "train"))
    val_dataset = ImageNetDataset(os.path.join(imagenet_root, "val"))

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)
    tokenizer.padding_side = (
        "left"  # For generation padding tokens should be on the left
    )

    acc1 = 0
    acc5 = 0
    prompt_text = "<image>A photo of a"

    val_iterator = more_itertools.chunked(val_dataset, batch_size)
    for batch_idx, batch in enumerate(val_iterator):
        batch_images = []
        batch_text = []

        for idx in range(len(batch)):
            # Choose a different set of random context samples for each sample
            # from the training set
            context_indices = np.random.choice(
                len(train_dataset), effective_num_shots, replace=False
            )

            in_context_samples = [train_dataset[i] for i in context_indices]

            vision_x = [
                eval_model.image_processor(data["image"]).unsqueeze(0)
                for data in in_context_samples
            ] + [eval_model.image_processor(batch[idx]["image"]).unsqueeze(0)]
            batch_images.append(torch.cat(vision_x, dim=0))

            context_class_names = [
                in_context_samples[i]["class_name"] for i in range(effective_num_shots)
            ]
            context_text = "".join(
                f"{prompt_text} {classname}<|endofchunk|>"
                for classname in context_class_names
            )
            batch_text.append(context_text)

        # shape [B, T_img, C, h, w]
        vision_x = torch.stack(batch_images, dim=0)
        # shape [B, T_img, 1, C, h, w] where 1 is the frame dimension
        vision_x = vision_x.unsqueeze(2)
        model._encode_vision_x(vision_x.cuda())

        # Cache the context text: tokenize context and prompt,
        # e.g. '<context> a picture of a '
        ctx_and_prompt_tokenized = tokenizer(
            [context_text + prompt_text + " " for context_text in batch_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        with torch.no_grad():
            precomputed = model(
                vision_x=None,
                lang_x=ctx_and_prompt_tokenized["input_ids"].cuda(),
                attention_mask=ctx_and_prompt_tokenized["attention_mask"].cuda(),
                clear_conditioned_layers=False,
                use_cached_vision_x=True,
                use_cache=True,
            )

        def _detach_pkvs(pkvs):
            """Detach a set of past key values."""
            return tuple([tuple([x.detach() for x in inner]) for inner in pkvs])

        precomputed_pkvs = _detach_pkvs(precomputed.past_key_values)

        precomputed_logits = precomputed.logits.detach()

        overall_probs = []
        for imagenet_class_name in tqdm(openai_imagenet_classnames):
            past_key_values = None
            # Tokenize only the class name and iteratively decode the model's
            # predictions for this class.
            classname_tokens = tokenizer(
                imagenet_class_name, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].cuda()

            if classname_tokens.ndim == 1:  # Case: classname is only 1 token
                classname_tokens = torch.unsqueeze(classname_tokens, 1)

            classname_tokens = repeat(
                classname_tokens, "b s -> (repeat b) s", repeat=batch_size
            )

            # Compute the outputs one token at a time, using cached
            # activations.

            # Initialize the elementwise predictions with the last set of
            # logits from precomputed; this will correspond to the predicted
            # probability of the first position/token in the imagenet
            # classname. We will append the logits for each token to this
            # list (each element has shape [B, 1, vocab_size]).
            elementwise_logits = [precomputed_logits[:, -2:-1, :]]

            for token_idx in range(classname_tokens.shape[1]):
                _lang_x = classname_tokens[:, token_idx].reshape((-1, 1))
                with torch.no_grad():
                    outputs = model(
                        vision_x=None,
                        lang_x=_lang_x,
                        clear_conditioned_layers=False,
                        use_cached_vision_x=True,
                        past_key_values=(
                            past_key_values if token_idx > 0 else precomputed_pkvs
                        ),
                        use_cache=True,
                    )
                past_key_values = _detach_pkvs(outputs.past_key_values)
                elementwise_logits.append(outputs.logits.detach())

            # logits/probs has shape [B, classname_tokens + 1, vocab_size]
            logits = torch.concat(elementwise_logits, 1)
            probs = torch.softmax(logits, dim=-1).detach()

            # collect the probability of the generated token -- probability
            # at index 0 corresponds to the token at index 1.
            probs = probs[:, :-1, :]  # shape [B, classname_tokens, vocab_size]

            gen_probs = torch.gather(probs, 2, classname_tokens[:, :, None]).squeeze(-1)

            class_prob = torch.prod(gen_probs, 1).detach().cpu().numpy()
            overall_probs.append(class_prob)

        overall_probs = np.row_stack(overall_probs).T  # shape [B, num_classes]

        def topk(probs_ary: np.ndarray, k: int) -> np.ndarray:
            """Return the indices of the top k elements in probs_ary."""
            return np.argsort(probs_ary)[::-1][:k]

        for i in range(batch_size):
            top5 = [
                IMAGENET_1K_CLASS_ID_TO_LABEL[pred]
                for pred in topk(overall_probs[i], 5)
            ]

            y_i = batch[i]["class_name"]
            acc5 += int(y_i in set(top5))
            acc1 += int(y_i == top5[0])

            print(
                f"DEBUG: batch {idx} elem {i} of {batch_size}:"
                f"label {y_i} // top5 {top5}"
            )

        examples_seen = (batch_idx + 1) * batch_size
        print(
            "eval {}/{}: acc@1 ({}), acc@5 ({})".format(
                examples_seen, num_samples, acc1 / examples_seen, acc5 / examples_seen
            )
        )
        if batch_idx * batch_size >= num_samples - 1:
            break

    return float(acc1) / num_samples


if __name__ == "__main__":
    main()
