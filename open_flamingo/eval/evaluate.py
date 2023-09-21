import argparse
import json
import os
import uuid
import random
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import utils
import math
from tqdm import tqdm
import wandb
import time

from open_flamingo.eval.eval_models import (
    SUPPORTED_MODELS,
    ZERO_SHOT_ONLY_MODELS,
    get_eval_model,
    BaseEvalModel,
)
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env

from open_flamingo.eval.rices import RICES
from open_flamingo.eval.classification_utils import (
    IMAGENET_CLASSNAMES,
    HM_CLASSNAMES,
    WATERBIRDS_CLASSNAMES,
    CAMELYON17_CLASSNAMES,
)
from open_flamingo.eval.coco_metric import compute_cider, postprocess_captioning_generation
from open_flamingo.eval.eval_datasets import (
    SUPPORTED_TASKS,
    CaptionDataset,
    VQADataset,
    ImageNetDataset,
    HatefulMemesDataset,
    WILDSDataset,
)
from open_flamingo.eval.ok_vqa_utils import postprocess_ok_vqa_generation
from open_flamingo.eval.vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    choices=SUPPORTED_MODELS,
    help="Model to evaluate.",
    default="open_flamingo",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)
parser.add_argument(
    "--report_to_wandb",
    action="store_true",
)
parser.add_argument(
    "--wandb_project",
    type=str,
)
parser.add_argument(
    "--wandb_entity",
    type=str,
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
parser.add_argument("--true_zero_shot", action="store_true")
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Size of demonstration query set"
)

parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument(
    "--no_caching_for_classification",
    action="store_true",
    help="Whether to skip using key-value caching for classification evals, which usually speeds it up.",
)
parser.add_argument(
    "--classification_prompt_ensembling",
    action="store_true",
    help="Whether to use prompt ensembling (average log-likelihoods over permutations of in-context examples)",
)
parser.add_argument(
    "--classification_num_classes_in_demos",
    type=int,
    default=None,
    help="If set, demonstrations use class-conditional sampling with this many classes. Otherwise, random sampling.",
)
parser.add_argument(
    "--rices",
    action="store_true",
    help="Whether to use RICES for evaluation. If False, uses random demonstrations.",
)
parser.add_argument(
    "--rices_vision_encoder_path",
    default="ViT-L-14",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--rices_vision_encoder_pretrained",
    default="openai",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--cached_demonstration_features",
    default=None,
    help="Directory where rices features for all choices of in-context examples are stored as a pkl file with the dataset name. If None, features are re-computed by script.",
)

# Distributed evaluation
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training.",
)
parser.add_argument(
    "--local_rank",
    default=0,
    type=int,
    help="Rank of distributed process (default: 0). Usually overwritten by world_info_from_env()",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
)
parser.add_argument(
    "--deepspeed",
    default=False,
    action="store_true",
    help="Whether to use deepspeed for distributed inference.",
)

# Per-dataset evaluation flags
for ds in SUPPORTED_TASKS:
    parser.add_argument(
        f"--eval_{ds}",
        action="store_true",
        default=False,
        help=f"Whether to evaluate on {ds}.",
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
parser.add_argument(
    "--vqav2_final_test_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_test2015_questions.json file containing all test questions. This is required to format the predictions for EvalAI.",
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

## Hateful Memes dataset
parser.add_argument(
    "--hateful_memes_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--hateful_memes_train_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--hateful_memes_test_annotations_json_path",
    type=str,
    default=None,
)

## WILDS datasets
parser.add_argument(
    "--wilds_root",
    type=str,
    default=".",
)


def eval_dataset(
    args,
    dataset_name,
    eval_model,
    results,
    eval_fn,
    **eval_kwargs,
):
    """Helper function to evaluate a dataset for all shots and seeds."""
    print(f"Evaluating on {dataset_name}...")

    # load cached demonstration features for RICES
    if args.cached_demonstration_features is not None:
        cached_features = torch.load(
            f"{args.cached_demonstration_features}/{dataset_name}.pkl",
            map_location="cpu",
        )
    else:
        cached_features = None

    for shot in args.shots:
        scores = defaultdict(list)
        for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
            trial_results = eval_fn(
                args,
                eval_model=eval_model,
                num_shots=shot,
                seed=seed,
                dataset_name=dataset_name,
                cached_features=cached_features,
                **eval_kwargs,
            )
            if args.rank == 0:
                # log to results file
                for metric_name, metric_value in trial_results.items():
                    print(f"Shots {shot} Trial {trial} {metric_name}: {metric_value}")
                    scores[metric_name].append(metric_value)
                # log to wandb table
                if args.report_to_wandb:
                    if trial == 0:
                        wandb_table = wandb.Table(
                            columns=["seed"] + list(trial_results.keys())
                        )
                    wandb_table.add_data(seed, *list(trial_results.values()))

        if args.rank == 0:
            means = {
                metric_name: np.nanmean(metric_values)
                for metric_name, metric_values in scores.items()
            }
            stds = {
                metric_name: np.nanstd(metric_values)
                for metric_name, metric_values in scores.items()
            }
            print(f"Shots {shot} Mean scores: {means}")

            # log to results file
            setting_results = {}
            for metric_name, metric_value in means.items():
                setting_results.update(
                    {
                        f"trials_{metric_name}": scores[metric_name],
                        f"mean_{metric_name}": means[metric_name],
                        f"std_{metric_name}": stds[metric_name],
                    }
                )
            results[dataset_name].append(
                {"shots": shot, "seeds": args.trial_seeds, **setting_results}
            )

            # log to wandb
            if args.report_to_wandb:
                setting_results = {
                    f"{dataset_name}/{k}": v
                    for k, v in setting_results.items()
                    if "trials" not in k
                }
                wandb.log(
                    {
                        f"{dataset_name}/results": wandb_table,
                        **setting_results,
                    },
                    step=shot,
                    commit=True,
                )


def main():
    args, leftovers = parser.parse_known_args()

    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    model_args["device"] = device_id

    # initialize model
    eval_model = get_eval_model(args.model, model_args, init_on_device=args.deepspeed)
    eval_model.init_distributed(
        world_size=args.world_size,
        use_deepspeed=args.deepspeed,
    )

    # Validate args
    if args.model in ZERO_SHOT_ONLY_MODELS and args.shots != [0]:
        raise ValueError(f"Only 0 shot eval is supported for {args.model}")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    for dataset in SUPPORTED_TASKS:
        if (
            getattr(args, f"eval_{dataset}")
            and dataset not in eval_model.supported_tasks
        ):
            raise ValueError(f"Model {args.model} does not support {dataset}.")

    # set up wandb
    if args.rank == 0 and args.report_to_wandb:
        cfg_dict = vars(args)
        cfg_dict.update(model_args)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=cfg_dict,
        )

    # Run through dataset flags
    results = defaultdict(list)
    if args.eval_flickr30:
        eval_dataset(
            args,
            dataset_name="flickr30",
            eval_model=eval_model,
            results=results,
            eval_fn=evaluate_captioning,
        )

    if args.eval_coco:
        eval_dataset(
            args,
            dataset_name="flickr30",
            eval_model=eval_model,
            results=results,
            eval_fn=evaluate_captioning,
        )

    if args.eval_ok_vqa:
        eval_dataset(
            args,
            dataset_name="ok_vqa",
            eval_model=eval_model,
            results=results,
            eval_fn=evaluate_vqa,
        )

    if args.eval_vqav2:
        eval_dataset(
            args,
            dataset_name="vqav2",
            eval_model=eval_model,
            results=results,
            eval_fn=evaluate_vqa,
        )

    if args.eval_vizwiz:
        eval_dataset(
            args,
            dataset_name="vizwiz",
            eval_model=eval_model,
            results=results,
            eval_fn=evaluate_vqa,
        )

    if args.eval_textvqa:
        eval_dataset(
            args,
            dataset_name="textvqa",
            eval_model=eval_model,
            results=results,
            eval_fn=evaluate_vqa,
            max_generation_length=10,
        )

    if args.eval_imagenet:
        eval_dataset(
            args,
            dataset_name="imagenet",
            eval_model=eval_model,
            results=results,
            eval_fn=evaluate_classification,
            use_prompt_ensembling=args.classification_prompt_ensembling,
            no_kv_caching=args.no_caching_for_classification,
        )

    if args.eval_hateful_memes:
        eval_dataset(
            args,
            dataset_name="hateful_memes",
            eval_model=eval_model,
            results=results,
            eval_fn=evaluate_classification,
            use_prompt_ensembling=args.classification_prompt_ensembling,
            no_kv_caching=args.no_caching_for_classification,
        )

    if args.eval_waterbirds:
        eval_dataset(
            args,
            dataset_name="waterbirds",
            eval_model=eval_model,
            results=results,
            eval_fn=evaluate_classification,
            use_prompt_ensembling=args.classification_prompt_ensembling,
            no_kv_caching=args.no_caching_for_classification,
        )

    if args.eval_camelyon17:
        eval_dataset(
            args,
            dataset_name="camelyon17",
            eval_model=eval_model,
            results=results,
            eval_fn=evaluate_classification,
            use_prompt_ensembling=args.classification_prompt_ensembling,
            no_kv_caching=args.no_caching_for_classification,
        )

    # Write all results to a json
    if args.rank == 0 and args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


def evaluate_captioning(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 20,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "coco",
    cached_features=None,
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
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr30". Defaults to "coco".
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.
    Returns:
        float: CIDEr score

    """
    utils.random_seed(seed, args.rank)

    if dataset_name == "coco":
        image_train_dir_path = args.coco_train_image_dir_path
        image_val_dir_path = args.coco_val_image_dir_path
        annotations_path = args.coco_karpathy_json_path
        prompt_fn = eval_model.get_coco_prompt
    elif dataset_name == "flickr30":
        image_train_dir_path = (
            args.flickr_image_dir_path
        )  # Note: calling this "train" for consistency with COCO but Flickr only has one split for images
        image_val_dir_path = None
        annotations_path = args.flickr_karpathy_json_path
        prompt_fn = eval_model.get_flickr30_prompt
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=True,
        dataset_name=dataset_name if dataset_name != "nocaps" else "coco",
    )

    test_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    effective_num_shots = utils.compute_effective_num_shots(num_shots, args.model, args.true_zero_shot)

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples,
        args.batch_size,
    )

    # subset of the training set to sample context images from
    query_set = utils.get_query_set(train_dataset, args.query_set_size)
    if args.rices:
        rices_dataset = RICES(
            query_set,
            eval_model.device,
            args.batch_size,
            cached_features=cached_features[query_set.indices],
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )

    utils.random_seed(seed, args.rank)
    predictions = defaultdict()
    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name.upper()}",
        disable=args.rank != 0,
    ):
        if args.rices:
            batch_demo_samples = rices_dataset.find(batch["image"], effective_num_shots)
        else:
            batch_demo_samples = utils.sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch["image"])
            )

        batch_images, batch_text = [], []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])

            context_text = "".join(
                [
                    prompt_fn(caption=x["caption"].strip()) + "\n"
                    for x in batch_demo_samples[i]
                ]
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(context_text + prompt_fn())

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        new_predictions = [
            postprocess_captioning_generation(out).replace('"', "") for out in outputs
        ]

        if args.rank == 0:
            for i in range(len(batch_text)):
                print("Context:", batch_text[i], "\n", "Generated:", new_predictions[i])

        for i, sample_id in enumerate(batch["image_id"]):
            predictions[sample_id] = {
                "caption": new_predictions[i],
            }

    # all gather
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of dicts

    if args.rank != 0:
        return None

    all_predictions = {
        k: v for d in all_predictions for k, v in d.items()
    }  # merge dicts

    # save the predictions to a temporary file
    results_path = f"{dataset_name}results_{uuid.uuid4()}.json"

    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_id": k, "caption": all_predictions[k]["caption"]}
                    for k in all_predictions
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

    return {"CIDEr": metrics["CIDEr"] * 100.0}


def evaluate_vqa(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "vqav2",
    cached_features=None,
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
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.
    Returns:
        float: accuracy score
    """
    utils.random_seed(seed, args.rank)

    if dataset_name == "ok_vqa":
        train_image_dir_path = args.ok_vqa_train_image_dir_path
        train_questions_json_path = args.ok_vqa_train_questions_json_path
        train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        test_image_dir_path = args.ok_vqa_test_image_dir_path
        test_questions_json_path = args.ok_vqa_test_questions_json_path
        test_annotations_json_path = args.ok_vqa_test_annotations_json_path
        prompt_fn = eval_model.get_ok_vqa_prompt
    elif dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
        prompt_fn = eval_model.get_vqav2_prompt
    elif dataset_name == "vizwiz":
        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path
        prompt_fn = eval_model.get_vizwiz_prompt
    elif dataset_name == "textvqa":
        train_image_dir_path = args.textvqa_image_dir_path
        train_questions_json_path = args.textvqa_train_questions_json_path
        train_annotations_json_path = args.textvqa_train_annotations_json_path
        test_image_dir_path = args.textvqa_image_dir_path
        test_questions_json_path = args.textvqa_test_questions_json_path
        test_annotations_json_path = args.textvqa_test_annotations_json_path
        prompt_fn = eval_model.get_textvqa_prompt
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

    effective_num_shots = utils.compute_effective_num_shots(num_shots, args.model, args.true_zero_shot)

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples,
        args.batch_size,
    )

    query_set = utils.get_query_set(train_dataset, args.query_set_size)
    if args.rices:
        rices_dataset = RICES(
            query_set,
            eval_model.device,
            args.batch_size,
            cached_features=cached_features[query_set.indices],
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )

    utils.random_seed(seed, args.rank)
    predictions = []
    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name}",
        disable=args.rank != 0,
    ):
        if args.rices:
            batch_demo_samples = rices_dataset.find(batch["image"], effective_num_shots)
        else:
            batch_demo_samples = utils.sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch["image"])
            )

        batch_images, batch_text = [], []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])

            context_text = "".join(
                [
                    prompt_fn(
                        question=x["question"], answer=x["answers"][0]
                    )
                    + "\n"
                    for x in batch_demo_samples[i]
                ]
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(
                context_text + prompt_fn(question=batch["question"][i])
            )

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        if args.rank == 0:
            for i in range(len(batch["image"])):
                print("Context:", batch_text[i])
                print("Prediction:", outputs[i])
                print()

        process_function = (
            postprocess_ok_vqa_generation
            if dataset_name == "ok_vqa"
            else postprocess_vqa_generation
        )

        new_predictions = map(process_function, outputs)

        for new_prediction, sample_id in zip(new_predictions, batch["question_id"]):
            predictions.append({"answer": new_prediction, "question_id": sample_id})

    # all gather
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists

    if args.rank != 0:
        return None

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten

    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"{dataset_name}results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(all_predictions, indent=4))

    if test_annotations_json_path is not None:
        acc = compute_vqa_accuracy(
            f"{dataset_name}results_{random_uuid}.json",
            test_questions_json_path,
            test_annotations_json_path,
        )
        # delete the temporary file
        os.remove(f"{dataset_name}results_{random_uuid}.json")

    else:
        print("No annotations provided, skipping accuracy computation.")
        acc = None
        if dataset_name == "vqav2":
            from open_flamingo.scripts.fill_vqa_testdev_results import (
                fill_vqav2_test_json,
            )

            fill_fn = fill_vqav2_test_json
        elif dataset_name == "vizwiz":
            from open_flamingo.scripts.fill_vqa_testdev_results import (
                fill_vizwiz_test_json,
            )

            fill_fn = fill_vizwiz_test_json
        else:
            print(
                "Temporary file saved to ", f"{dataset_name}results_{random_uuid}.json"
            )
            return

        fill_fn(
            f"{dataset_name}results_{random_uuid}.json",
            f"{dataset_name}-testdev_{eval_model.lm_name}_{num_shots}_{'rices' if args.rices else 'random'}_{seed}.json",
            args.vqav2_final_test_questions_json_path
            if dataset_name == "vqav2"
            else args.vizwiz_test_questions_json_path,
        )
        print(
            "Test-dev results saved to ",
            f"{dataset_name}-testdev_{eval_model.lm_name}_{num_shots}_{'rices' if args.rices else 'random'}_{seed}.json",
        )
        os.remove(f"{dataset_name}results_{random_uuid}.json")

    return {"accuracy": acc}


def evaluate_classification(
    args: argparse.Namespace,
    eval_model,
    seed: int = 42,
    num_shots: int = 8,
    dataset_name: str = "imagenet",
    cached_features=None,
    no_kv_caching=False,
    use_prompt_ensembling: bool = False,
):
    """
    Evaluate a model on classification dataset.

    Args:
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        no_kv_caching (bool): whether to disable key-value caching
        dataset_name (str, optional): dataset name. Defaults to "imagenet".
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.

    Returns:
        float: accuracy score
    """
    if args.model != "open_flamingo":
        raise NotImplementedError(
            "evaluate_classification is currently only supported for OpenFlamingo"
        )

    if dataset_name == "imagenet":
        train_dataset = ImageNetDataset(os.path.join(args.imagenet_root, "train"))
        test_dataset = ImageNetDataset(os.path.join(args.imagenet_root, "val"))
        prompt_fn = lambda x, test: eval_model.get_imagenet_prompt(
            label=x["class_name"] if not test else None
        )
        all_class_names = IMAGENET_CLASSNAMES
        k = 5
    elif dataset_name == "hateful_memes":
        train_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_train_annotations_json_path,
        )
        test_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_test_annotations_json_path,
        )
        prompt_fn = lambda x, test: eval_model.get_hateful_memes_prompt(
            text=x["ocr"], label=x["class_name"] if not test else None
        )
        all_class_names = HM_CLASSNAMES
        k = 1
    elif dataset_name in ("waterbirds",):  # subpopulation shift datasets
        train_dataset = WILDSDataset(
            dataset_name=dataset_name,
            split="train",
            root_dir=args.wilds_root,
        )
        test_dataset = WILDSDataset(
            dataset_name=dataset_name,
            split="test",
            root_dir=args.wilds_root,
        )
        prompt_fn = lambda x, test: eval_model.get_waterbirds_prompt(
            label=x["class_name"] if not test else None
        )
        all_class_names = WATERBIRDS_CLASSNAMES
        k = 1
    elif dataset_name == "camelyon17":
        train_dataset = WILDSDataset(
            dataset_name=dataset_name,
            split="train",
            root_dir=args.wilds_root,
        )
        test_dataset = WILDSDataset(
            dataset_name=dataset_name,
            split="test",
            root_dir=args.wilds_root,
        )
        prompt_fn = lambda x, test: eval_model.get_camelyon17_prompt(
            label=x["class_name"] if not test else None
        )
        all_class_names = CAMELYON17_CLASSNAMES
        k = 1
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    class_id_to_name = dict(zip(range(len(all_class_names)), all_class_names))

    effective_num_shots = utils.compute_effective_num_shots(num_shots, args.model, args.true_zero_shot)

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples,
        args.batch_size,
    )

    query_set = utils.get_query_set(train_dataset, args.query_set_size)
    assert hasattr(query_set, 'class_id_array')
    if args.rices:
        rices_dataset = RICES(
            query_set,
            eval_model.device,
            args.batch_size,
            cached_features=cached_features[query_set.indices],
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )

    utils.random_seed(seed, args.rank)
    predictions = []
    prompt_time_m = utils.AverageMeter()
    rank_time_m = utils.AverageMeter()

    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name}",
        disable=args.rank != 0,
    ):

        end = time.time()
        if args.classification_num_classes_in_demos is not None:
            batch_classes, batch_demo_samples = utils.sample_class_conditional_batch_demos_from_query_set(
                batch["class_id"], args.classification_num_classes_in_demos, query_set, effective_num_shots,
            )
            if args.rices: 
                batch_classes = torch.LongTensor(batch_classes)
                num_samples = effective_num_shots // args.classification_num_classes_in_demos * torch.ones_like(batch_classes)
                num_samples += torch.tensor([int(i < effective_num_shots % args.classification_num_classes_in_demos) for i in range(args.classification_num_classes_in_demos)]).unsqueeze(0).repeat(len(batch_classes), 1)
                batch_demo_samples = rices_dataset.find_filtered(
                    utils.repeat_interleave(batch["image"], args.classification_num_classes_in_demos), 
                    num_samples.view(-1),
                    [torch.where(query_set.class_id_array == class_id)[0].tolist() for class_id in batch_classes.view(-1)],
                )
                batch_demo_samples = utils.reshape_nested_list(batch_demo_samples, (len(batch_classes), effective_num_shots))
        else:
            if not args.rices:
                batch_demo_samples = utils.sample_batch_demos_from_query_set(
                    query_set, effective_num_shots, len(batch["image"])
                )
            else:
                batch_demo_samples = rices_dataset.find(
                    batch["image"], 
                    effective_num_shots, 
                )

        prompt_time_m.update(time.time() - end)
        end = time.time()

        # set up prompt ensembling
        num_permutations = (
            min(6, math.factorial(effective_num_shots)) if use_prompt_ensembling else 1
        )
        logprobs = []
        for _ in range(num_permutations):
            batch_images, batch_text = [], []
            for i in range(len(batch["image"])):
                if use_prompt_ensembling:
                    random.shuffle(batch_demo_samples[i])

                if effective_num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                batch_images.append(context_images + [batch["image"][i]])

                context_text = "".join(
                    [prompt_fn(x, test=False) for x in batch_demo_samples[i]]
                )

                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")

                batch_text.append(
                    context_text + prompt_fn({k: batch[k][i] for k in batch}, test=True)
                )

            # get predicted class names
            logprobs.append(
                eval_model.get_rank_classifications(
                    batch_text,
                    batch_images,
                    all_class_names,
                    use_cache=(not no_kv_caching),
                    normalize_length=False,
                )
            )

        # ensemble logprobs together
        logprobs = torch.mean(torch.stack(logprobs, dim=-1), dim=-1).to(dtype=torch.float32)
        rank_time_m.update(time.time() - end)

        (
            predicted_class_ixs,
            predicted_classnames,
            predicted_logprobs,
        ) = utils.get_predicted_classnames(
            logprobs,
            k,
            class_id_to_name,
        )

        # dev: print some results
        print("Context:", batch_text[0], "\n", "Generated:", predicted_classnames[0][0], "\n", "True:", batch["class_name"][0])

        # compute accuracy
        for i, topk in enumerate(predicted_classnames):
            y_i = batch["class_name"][i]
            score = torch.exp(
                predicted_logprobs[i][0] - torch.logsumexp(logprobs[i], dim=0)
            ).item()
            pred_info = {
                "id": batch["id"][i],
                "gt_label": y_i,
                "gt_id": batch["class_id"][i],
                "pred_label": topk[0],
                "pred_score": score,
                "pred_class_id": predicted_class_ixs[i][0].item(),
            }
            if "metadata" in batch:
                pred_info["metadata"] = batch["metadata"][i].tolist()
            predictions.append(pred_info)

        if args.rank == 0:
            print(f"Avg prompt loading time: {prompt_time_m.avg}")
            print(f"Avg rank classification w/ ensembling time: {rank_time_m.avg}")
        
        end = time.time()

    # all gather
    gloo_pg = torch.distributed.new_group(backend="gloo")
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions, group=gloo_pg)  # list of dicts

    if args.rank != 0:
        return

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten

    if dataset_name == "hateful_memes":
        # return ROC-AUC score
        greater_label = max(all_class_names)
        gts = [pred["gt_label"] for pred in all_predictions]
        pred_scores = [
            pred["pred_score"]
            if pred["pred_label"] == greater_label
            else 1 - pred["pred_score"]
            for pred in all_predictions
        ]
        return {"roc_auc": roc_auc_score(gts, pred_scores)}
    elif dataset_name == "waterbirds":
        # return avg and worst group accuracies
        y_pred = torch.Tensor([pred["pred_class_id"] for pred in all_predictions])
        y_true = torch.Tensor([pred["gt_id"] for pred in all_predictions])
        metadata = torch.stack([pred["metadata"] for pred in all_predictions])
        all_results = test_dataset.dataset.eval(y_pred, y_true, metadata)[0]
        return all_results
    else:
        # return top-1 accuracy
        acc1 = sum(
            int(pred["gt_label"] == pred["pred_label"]) for pred in all_predictions
        )
        return {"top-1-accuracy": float(acc1) / len(all_predictions)}


if __name__ == "__main__":
    main()