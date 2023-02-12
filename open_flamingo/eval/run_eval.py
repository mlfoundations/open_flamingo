import argparse
import os
import time

import numpy as np
import torch
from evaluate import evaluate_coco, evaluate_vqa

import wandb
from open_flamingo.src.factory import create_model_and_transforms

parser = argparse.ArgumentParser()
parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
parser.add_argument("--lm_tokenizer_path", type=str,
                    default="facebook/opt-30b")
parser.add_argument("--clip_path", type=str,
                    default="openai/clip-vit-large-patch14")

parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory containing checkpoints")

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4])
parser.add_argument(
    "--num_trials",
    type=int,
    default=3,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    default=[0, 2, 4],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples", type=int, default=5000, help="Number of samples to evaluate on"
)

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--device", type=int, default=0)

# Dataset arguments
parser.add_argument(
    "--coco_image_dir_path",
    type=str,
    default="/fsx/home-anasawadalla/data/coco/train2017",
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default="/fsx/home-anasawadalla/data/coco/annotations/captions_train2017.json",
)
parser.add_argument(
    "--vqav2_image_dir_path",
    type=str,
    default="/fsx/home-anasawadalla/data/vqav2/train2014",
)
parser.add_argument(
    "--vqav2_questions_json_path",
    type=str,
    default="/fsx/home-anasawadalla/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json",
)
parser.add_argument(
    "--vqav2_annotations_json_path",
    type=str,
    default="/fsx/home-anasawadalla/data/vqav2/v2_mscoco_train2014_annotations.json",
)

# Wandb arguments
parser.add_argument(
    "--report_to_wandb",
    action="store_true", 
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="open-flamingo",
)
parser.add_argument(
    "--wandb_entity",
    type=str,
    default="anas-awadalla",
)
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default="online-eval",
)

def run_evaluation_suite(args):
    flamingo, image_processor, tokenizer = create_model_and_transforms(
        args.clip_path,
        args.clip_path,
        args.lm_path,
        args.lm_tokenizer_path,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")[
        "model_state_dict"]
    # remove the "module." prefix from the keys
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    flamingo.load_state_dict(checkpoint, strict=False)
    flamingo.to(args.device if args.device >= 0 else "cpu")

    results = {"coco": [], "vqav2": []}

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
                num_samples=args.num_samples,
                num_shots=shot,
                device=args.device,
                seed=seed,
            )
            print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
            scores.append(cider_score)
        print(f"Shots {shot} Mean CIDEr score: {np.mean(scores)}")
        results["coco"].append(
            {"shots": shot, "trials": scores, "mean": np.mean(scores)})

    print("Evaluating on VQAv2...")
    for shot in args.shots:
        scores = []
        for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
            vqa_score = evaluate_vqa(
                model=flamingo,
                tokenizer=tokenizer,
                image_processor=image_processor,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                num_shots=shot,
                device=args.device,
                seed=seed,
                image_dir_path=args.vqav2_image_dir_path,
                questions_json_path=args.vqav2_questions_json_path,
                annotations_json_path=args.vqav2_annotations_json_path,
            )
            print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
            scores.append(vqa_score)
        print(f"Shots {shot} Mean VQA score: {np.mean(scores)}")
        results["vqav2"].append(
            {"shots": shot, "trials": scores, "mean": np.mean(scores)})

    return results


def main():
    args = parser.parse_args()

    if args.report_to_wandb:
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity, name=args.wandb_run_name)

    evaluated_checkpoints = set()
    while True:
        # check for new checkpoints
        checkpoints = set([f for f in os.listdir(
            args.model_dir) if f.startswith('checkpoint')])

        if len(checkpoints) > 0:
            # remove already evaluated checkpoints
            checkpoints.difference_update(evaluated_checkpoints)
            
            # sort checkpoints by epoch
            checkpoints = sorted(checkpoints, key=lambda x: int(
                x.split('_')[-1].split('.')[0]))
            
            for path in checkpoints:
                # pick the last checkpoint
                checkpoint = os.path.join(args.model_dir, path)
                epoch = int(checkpoint.split('_')[-1].split('.')[0])
                print('found new checkpoint: {}'.format(checkpoint))
                # evaluate the model
                args.checkpoint_path = checkpoint
                results = run_evaluation_suite(args)
                evaluated_checkpoints.add(checkpoint)

                if args.report_to_wandb:
                    for dataset in results:
                        for result in results[dataset]:
                            metric_name = f"{dataset} {'cider' if dataset == 'coco' else 'vqa accuracy'} (shots = {result['shots']})"
                            wandb.log({metric_name: result['mean'], "epoch": epoch})
        else:
            print('no checkpoint found, waiting for 10 mins...')
            time.sleep(10 * 60)

if __name__ == "__main__":
    main()
    
