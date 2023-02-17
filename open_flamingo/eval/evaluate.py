import argparse
import json
import os
import uuid
from collections import defaultdict
from typing import Callable, Any

import more_itertools
import numpy as np
import torch
from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import COCODataset, VQAv2Dataset, ImageNetDataset
from tqdm import tqdm
from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation
from open_flamingo.eval.classification import \
    postprocess_classification_generation, compute_classification_accuracy

from open_flamingo.src.factory import create_model_and_transforms
from open_flamingo.src.flamingo import Flamingo

parser = argparse.ArgumentParser()
parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
parser.add_argument("--lm_tokenizer_path", type=str,
                    default="facebook/opt-30b")
parser.add_argument("--clip_path", type=str,
                    default="openai/clip-vit-large-patch14")
parser.add_argument("--checkpoint_path", type=str, required=True)

parser.add_argument("--results_file", type=str, default=None,
                    help="JSON file to save results")

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 8], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=3,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    default=[0, 1, 2],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples", type=int, default=5000, help="Number of samples to evaluate on"
)

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--device", type=int, default=0)

# Per-dataset evaluation flags
parser.add_argument("--eval_coco", action="store_true", default=False,
        help="Whether to evaluate on COCO.")

parser.add_argument("--eval_vqav2", action="store_true", default=False,
        help="Whether to evaluate on VQAV2.")
parser.add_argument("--eval_imagenet", action="store_true", default=False,
                    help="Whether to evaluate on ImageNet.")

# Dataset arguments

## COCO Dataset
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

## VQAV2 Dataset
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

## Imagenet dataset
parser.add_argument("--imagenet_root",
                    type=str,
                    default="/tmp")


def main():
    args = parser.parse_args()

    # load model
    flamingo, image_processor, tokenizer = create_model_and_transforms(
        args.clip_path,
        args.clip_path,
        args.lm_path,
        args.lm_tokenizer_path,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")[
        "model_state_dict"
    ]
    # remove the "module." prefix from the keys
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    flamingo.load_state_dict(checkpoint, strict=False)
    flamingo.to(args.device if args.device >= 0 else "cpu")

    results = defaultdict(list)

    if args.eval_coco:

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

    if args.eval_vqav2:

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

    if args.eval_imagenet:

        print("Evaluating on ImageNet...")
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                imagenet_score = evaluate_imagenet(
                    model=flamingo,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    imagenet_root=args.imagenet_root
                )
                print(f"Shots {shot} Trial {trial} "
                      f"ImageNet score: {imagenet_score}")
                scores.append(imagenet_score)
            print(f"Shots {shot} Mean ImageNet score: {np.mean(scores)}")
            results["imagenet"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)})

    if args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


def get_random_indices(num_samples, effective_num_shots, full_dataset, seed):


    if num_samples + effective_num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + effective_num_shots, replace=False
    )
    return random_indices


def prepare_eval_samples_and_dataset(full_dataset, random_indices,
                                     effective_num_shots):
    # get in context samples
    in_context_samples = [full_dataset[i]
                          for i in random_indices[:effective_num_shots]]
    eval_dataset = torch.utils.data.Subset(
        full_dataset, random_indices[effective_num_shots:])
    return in_context_samples, eval_dataset


def get_context_images(image_processor, in_context_samples, num_shots):
    if num_shots > 0:
        context_images = image_processor(
            images=[s["image"] for s in in_context_samples],
            return_tensors="pt",
        )["pixel_values"]
        context_images = context_images.unsqueeze(1).unsqueeze(0)
    else:
        context_images = None
    return context_images


def get_context_text(get_prompt: Callable[[dict], str], in_context_samples,
                     effective_num_shots, num_shots) -> str:
    context_text = (
        "".join([get_prompt(s) for s in in_context_samples]
                ) if effective_num_shots > 0 else ""
    )

    if num_shots == 0:
        context_text = context_text.replace("<image>", "")
    return context_text


def prepare_batch_images(batch, image_processor, context_images,
                         num_shots):
    batch_images = None
    for b in batch:
        b_image = image_processor(images=[b["image"]], return_tensors="pt")[
            "pixel_values"
        ]
        b_image = b_image.unsqueeze(1).unsqueeze(0)
        b_image = (
            torch.cat([context_images, b_image], dim=1)
            if num_shots > 0
            else b_image
        )

        if batch_images is None:
            batch_images = b_image
        else:
            batch_images = torch.cat([batch_images, b_image], dim=0)
    return batch_images


def get_outputs(model, batch_images, device, attention_mask,
                max_generation_length, num_beams, length_penalty, input_ids):
    with torch.inference_mode():
        outputs = model.generate(
            batch_images.to(device if device >= 0 else "cpu"),
            input_ids.to(device if device >= 0 else "cpu"),
            attention_mask=attention_mask.to(
                device if device >= 0 else "cpu"),
            max_new_tokens=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

    outputs = outputs[:, len(input_ids[0]):]
    return outputs

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
        image_dir_path=image_dir_path, annotations_path=annotations_json_path
    )
    effective_num_shots = num_shots if num_shots > 0 else 2
    random_indices = get_random_indices(num_samples, effective_num_shots,
                                        full_dataset, seed)

    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset, random_indices=random_indices,
        effective_num_shots=effective_num_shots)

    context_images = get_context_images(image_processor=image_processor,
                                        in_context_samples=in_context_samples,
                                        num_shots=num_shots)

    model.eval()

    def get_prompt(sample):
        return f"<image>Output:{sample['caption'].strip()}<|endofchunk|>"

    context_text = get_context_text(get_prompt,
                                    in_context_samples=in_context_samples,
                                    effective_num_shots=effective_num_shots,
                                    num_shots=num_shots)

    predictions = defaultdict()

    for batch in more_itertools.chunked(
        tqdm(eval_dataset, desc="Running inference"), batch_size
    ):
        batch_images = prepare_batch_images(batch=batch,
                                            image_processor=image_processor,
                                            context_images=context_images,
                                            num_shots=num_shots)

        batch_text = [context_text + "<image>Output:" for _ in batch]
        tokenizer.padding_side = "left"
        encodings = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        outputs = get_outputs(model=model,
                              batch_images=batch_images,
                              device=device,
                              attention_mask=attention_mask,
                              max_generation_length=max_generation_length,
                              num_beams=num_beams,
                              length_penalty=length_penalty,
                              input_ids=input_ids)

        new_predictions = [
            postprocess_captioning_generation(out).replace('"', "")
            for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]

        for i, sample in enumerate(batch):
            predictions[sample["image_id"]] = {
                "caption": new_predictions[i],
            }

    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"cocoresults_{random_uuid}.json", "w") as f:
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
        result_path=f"cocoresults_{random_uuid}.json",
        annotations_path=annotations_json_path,
    )

    # delete the temporary file
    os.remove(f"cocoresults_{random_uuid}.json")

    return metrics["CIDEr"] * 100.0


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

    full_dataset = VQAv2Dataset(
        image_dir_path=image_dir_path,
        question_path=questions_json_path,
        annotations_path=annotations_json_path,
    )

    effective_num_shots = num_shots if num_shots > 0 else 2

    if num_samples + effective_num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than or equal to {len(full_dataset)}"
        )

    random_indices = get_random_indices(num_samples, effective_num_shots,
                                        full_dataset, seed)

    def get_prompt(sample, train=True):
        return f"<image>Question:{sample['question'].strip()} Answer:{sample['answers'][0].strip() if train else ''}{'<|endofchunk|>' if train else ''}"

    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset, random_indices=random_indices,
        effective_num_shots=effective_num_shots)

    model.eval()
    predictions = []

    context_images = get_context_images(image_processor=image_processor,
                                        in_context_samples=in_context_samples,
                                        num_shots=num_shots)

    context_text = get_context_text(get_prompt,
                                    in_context_samples=in_context_samples,
                                    effective_num_shots=effective_num_shots,
                                    num_shots=num_shots)

    for batch in more_itertools.chunked(
        tqdm(eval_dataset, desc="Running inference"), batch_size
    ):
        batch_images = prepare_batch_images(batch=batch,
                                            image_processor=image_processor,
                                            context_images=context_images,
                                            num_shots=num_shots)

        batch_text = [context_text + get_prompt(s, train=False) for s in batch]

        tokenizer.padding_side = "left"
        encodings = tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = encodings["input_ids"].to(device if device >= 0 else "cpu")
        attention_mask = encodings["attention_mask"].to(
            device if device >= 0 else "cpu"
        )

        outputs = get_outputs(model=model,
                              batch_images=batch_images,
                              device=device,
                              attention_mask=attention_mask,
                              max_generation_length=max_generation_length,
                              num_beams=num_beams,
                              length_penalty=length_penalty,
                              input_ids=input_ids)
        new_predictions = [
            postprocess_vqa_generation(out)
            for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]

        predictions.extend(
            [
                {"answer": p, "question_id": sample["question_id"]}
                for p, sample in zip(new_predictions, batch)
            ]
        )
    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"vqaresults_{random_uuid}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))

    acc = compute_vqa_accuracy(
        f"vqaresults_{random_uuid}.json", questions_json_path, annotations_json_path
    )

    # delete the temporary file
    os.remove(f"vqaresults_{random_uuid}.json")

    return acc


def evaluate_imagenet(
    model: Flamingo,
    tokenizer,
    image_processor,
    batch_size: int,
    imagenet_root: str,
    seed: int = 42,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    num_samples: int = 5000,
    num_shots: int = 8,
    device: int = -1,
):
    """
    Evaluate a model on ImageNet dataset.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        imagenet_root (str): path to imagenet root for the specified split.
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

    full_dataset = ImageNetDataset(root=imagenet_root)

    effective_num_shots = num_shots if num_shots > 0 else 2

    if num_samples + effective_num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than or equal to {len(full_dataset)}"
        )

    random_indices = get_random_indices(num_samples, effective_num_shots,
                                        full_dataset, seed)

    def get_prompt(sample: dict) -> str:
        return f"<image>Output: A photo of a " \
               f"{sample['class_name'].strip()}<|endofchunk|>"

    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset, random_indices=random_indices,
        effective_num_shots=effective_num_shots)

    model.eval()
    predictions = []

    context_images = get_context_images(image_processor=image_processor,
                                        in_context_samples=in_context_samples,
                                        num_shots=num_shots)

    context_text = get_context_text(get_prompt,
                                    in_context_samples=in_context_samples,
                                    effective_num_shots=effective_num_shots,
                                    num_shots=num_shots)

    for batch in more_itertools.chunked(
            tqdm(eval_dataset, desc="Running inference"), batch_size
    ):
        batch_images = prepare_batch_images(batch=batch,
                                            image_processor=image_processor,
                                            context_images=context_images,
                                            num_shots=num_shots)

        batch_text = [context_text + "<image>Output:" for _ in batch]

        tokenizer.padding_side = "left"
        encodings = tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        device = device if device >= 0 else "cpu"
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        batch_images = batch_images.to(device)

        outputs = model(batch_images, input_ids, attention_mask)

        labels = encodings["input_ids"].clone()
        # convert padding tokens to -100 so they are ignored in loss
        labels[labels == tokenizer.pad_token_id] = -100

        # Convert all tokens in prefix until separator to -100 so they are
        # ignored in loss (loss is only computed on token IDs >= 0)
        for idx in range(len(labels)):
            end_of_prefix = labels[idx][1:].tolist().index(
                tokenizer.eos_token_id) + 1
            labels[idx, :end_of_prefix + 1] = -100

        # Loss computation from OPT code:
        # Compute per instance loss
        # Shift so that tokens < n predict n
        # shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
        # loss = loss.view(outputs.size(0), outputs.size(1))

        # sum loss over all tokens and divide by number of variable tokens
        loss = loss.sum(dim=1) / (labels != -100).sum(dim=1).float()

        import ipdb;ipdb.set_trace()



    ##############################################################
    ##### Compute "strict" accuracy based on generated text ######
    ##############################################################
    #
    #
        outputs = get_outputs(model=model,
                              batch_images=batch_images,
                              device=device,
                              attention_mask=attention_mask,
                              max_generation_length=max_generation_length,
                              num_beams=num_beams,
                              length_penalty=length_penalty,
                              input_ids=input_ids)
    #
    #     new_predictions = [
    #         postprocess_classification_generation(out).replace('"', "")
    #         for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #     ]
    #
    #     predictions.extend(
    #         [
    #             {"prediction": p, "class_label": sample["class_name"]}
    #             for p, sample in zip(new_predictions, batch)
    #         ]
    #     )
    #
    # print("[DEBUG] sample of predictions and labels for debugging:")
    # for p in predictions[:16]:
    #     print(f"\t prediction: {p['prediction']}")
    #     print(f"\t label: {p['class_label']}")
    # return compute_classification_accuracy(predictions)


if __name__ == "__main__":
    main()
