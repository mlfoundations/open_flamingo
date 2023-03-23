import argparse
import json
from math import ceil
import os
import uuid
from collections import defaultdict

import more_itertools
import numpy as np
import torch
import torch.multiprocessing as mp
from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import COCOFlickrDataset, VQADataset, ImageNetDataset
from tqdm import tqdm

from open_flamingo.eval.ok_vqa_utils import postprocess_ok_vqa_generation
from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation

from open_flamingo.eval.evaluation_utils import FlamingoModelLoader, \
    get_context_images, get_context_text, prepare_batch_images
from open_flamingo.eval.imagenet_utils import openai_imagenet_classnames, \
    IMAGENET_1K_CLASS_ID_TO_LABEL

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
    "--num_samples", type=int, default=5000,
    help="Number of samples to evaluate on"
)

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--device", type=int, default=0)

# Per-dataset evaluation flags
parser.add_argument("--eval_coco", action="store_true", default=False,
                    help="Whether to evaluate on COCO.")
parser.add_argument("--eval_vqav2", action="store_true", default=False,
                    help="Whether to evaluate on VQAV2.")
parser.add_argument("--eval_ok_vqa", action="store_true", default=False,
                    help="Whether to evaluate on OK-VQA.")
parser.add_argument("--eval_imagenet", action="store_true", default=False,
                    help="Whether to evaluate on ImageNet.")

parser.add_argument("--eval_flickr30", action="store_true", default=False,
                    help="Whether to evaluate on Flickr30.")

# Dataset arguments

## COCO Dataset
parser.add_argument(
    "--coco_image_dir_path",
    type=str,
    help="Path to the coco/train2017 directory.",
    default=None,
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    help="Path to the coco/annotations/captions_train2017.json file.",
    default=None,
)

## Flickr30 Dataset
parser.add_argument(
    "--flickr_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None
)
parser.add_argument(
    "--flickr_annotations_json_path",
    type=str,
    help="Path to the dataset_flickr30k_coco_style.json file.",
    default=None,
)

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_image_dir_path",
    type=str,
    help="Path to the vqav2/train2014 directory.",
    default=None,
)
parser.add_argument(
    "--vqav2_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_train2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--vqav2_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_train2014_annotations.json file.",
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
parser.add_argument("--imagenet_root",
                    type=str,
                    default="/tmp")


def main():
    args = parser.parse_args()

    flamingo_loader = FlamingoModelLoader(
        clip_path=args.clip_path,
        lm_path=args.lm_path,
        lm_tokenizer_path=args.lm_tokenizer_path,
        checkpoint_path=args.checkpoint_path)

    results = defaultdict(list)

    if args.eval_flickr30:
        print("Evaluating on Flickr30...")
        flamingo, image_processor, tokenizer = flamingo_loader.load(args.device)
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_coco_flickr(
                    model=flamingo,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    batch_size=args.batch_size,
                    image_dir_path=args.flickr_image_dir_path,
                    annotations_json_path=args.flickr_annotations_json_path,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    is_flickr=True
                )
                print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                scores.append(cider_score)
            print(f"Shots {shot} Mean CIDEr score: {np.mean(scores)}")
            results["flickr30"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)})
    results = defaultdict(list)

    if args.eval_coco:

        print("Evaluating on COCO...")
        flamingo, image_processor, tokenizer = flamingo_loader.load(args.device)
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_coco_flickr(
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

    if args.eval_ok_vqa:

        print("Evaluating on OK-VQA...")
        flamingo, image_processor, tokenizer = flamingo_loader.load(args.device)
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                ok_vqa_score = evaluate_vqa(
                    model=flamingo,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    num_shots=shot,
                    device=args.device,
                    seed=seed,
                    image_dir_path=args.ok_vqa_image_dir_path,
                    questions_json_path=args.ok_vqa_questions_json_path,
                    annotations_json_path=args.ok_vqa_annotations_json_path,
                    vqa_dataset='ok_vqa'
                )
                print(
                    f"Shots {shot} Trial {trial} OK-VQA score: {ok_vqa_score}")
                scores.append(ok_vqa_score)
            print(f"Shots {shot} Mean OK-VQA score: {np.mean(scores)}")
            results["ok_vqa"].append(
                {"shots": shot, "trials": scores, "mean": np.mean(scores)})

    if args.eval_vqav2:

        print("Evaluating on VQAv2...")
        flamingo, image_processor, tokenizer = flamingo_loader.load(args.device)
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
                    vqa_dataset='vqa'
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
                    flamingo_loader=flamingo_loader,
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


def evaluate_coco_flickr(
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
        is_flickr=False,
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
        is_flickr (bool): defines if that data is COCO or Flickr. Defaults to False (COCO).

    Returns:
        float: CIDEr score

    """

    full_dataset = COCOFlickrDataset(
        image_dir_path=image_dir_path, annotations_path=annotations_json_path,
        is_flickr=is_flickr,
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

    desc = 'Running inference Flickr30' if is_flickr else 'Running inference COCO'

    for batch in more_itertools.chunked(
            tqdm(eval_dataset, desc=desc), batch_size
    ):
        batch_images = prepare_batch_images(batch=batch,
                                            image_processor=image_processor,
                                            context_images=context_images,
                                            num_shots=num_shots)

        batch_text = [context_text + "<image>Output:" for _ in batch]
        tokenizer.padding_side = "left"
        encodings = tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=512,
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
    results_path = f"flickrresults_{random_uuid}.json" if is_flickr \
        else f"cocoresults_{random_uuid}.json"
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
        vqa_dataset='vqa'
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
            tqdm(eval_dataset, desc=f"Running inference {vqa_dataset}"),
            batch_size
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
            padding="longest",
            truncation=True,
            max_length=512,
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

        process_function = postprocess_vqa_generation if vqa_dataset == 'vqa' else postprocess_ok_vqa_generation

        new_predictions = [
            process_function(out)
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
    with open(f"{vqa_dataset}results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))

    acc = compute_vqa_accuracy(
        f"{vqa_dataset}results_{random_uuid}.json", questions_json_path,
        annotations_json_path,
    )

    # delete the temporary file
    os.remove(f"{vqa_dataset}results_{random_uuid}.json")

    return acc


def evaluate_imagenet(
        flamingo_loader: FlamingoModelLoader,
        batch_size: int,
        imagenet_root: str,
        seed: int = 42,
        num_samples: int = 5000,
        num_shots: int = 8,
        device: int = -1,
):
    """
    Evaluate a model on ImageNet dataset.

    Args:
        flamingo_loader: FlamingoModelLoader to use.
        batch_size (int): batch size
        imagenet_root (str): path to imagenet root for the specified split.
        seed (int, optional): random seed. Defaults to 42.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000 samples.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1 (cpu).

    Returns:
        float: accuracy score
    """
    # flamingo, image_processor, tokenizer = flamingo_loader.load(device)

    full_dataset = ImageNetDataset(root=imagenet_root)

    effective_num_shots = num_shots if num_shots > 0 else 2

    if num_samples + effective_num_shots > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than or equal to "
            f"{len(full_dataset)} ")

    random_indices = get_random_indices(num_samples, effective_num_shots,
                                        full_dataset, seed)

    in_context_samples, eval_dataset = prepare_eval_samples_and_dataset(
        full_dataset=full_dataset, random_indices=random_indices,
        effective_num_shots=effective_num_shots)

    # Predictions based on the class target sequence with the maximal
    # predicted probability
    predictions_max_prob = []
    # Predictions based on the class target sequence with the minimal loss on
    # the model logits
    predictions_min_loss = []
    labels = []

    # kwargs to use when calling tokenizer
    tokenizer_kwargs = {'return_tensors': 'pt',
                        'padding': True,
                        'truncation': True,
                        'max_length': 256}

    for i, batch in enumerate(more_itertools.chunked(eval_dataset, batch_size)):
        print(f"processing batch {i} of {ceil(len(eval_dataset) / batch_size)}")

        # For each ImageNet class, construct the output prompt, compute a
        # forward pass, and store the results.
        from open_flamingo.eval.imagenet_utils import infer
        mp.set_start_method('spawn')
        queue = mp.Queue()
        processes = []
        device_count = torch.cuda.device_count()

        for rank in range(device_count):
            p = mp.Process(target=infer, kwargs={
                "rank": rank,
                "queue": queue,
                "flamingo_loader": flamingo_loader,
                "batch": batch,
                "in_context_samples": in_context_samples,
                "batch_size": batch_size,
                "tokenizer_kwargs": tokenizer_kwargs,
                "num_shots": num_shots,
                "effective_num_shots": effective_num_shots,
            })
            p.start()
            processes.append(p)

        manager = mp.Manager()
        return_dict = manager.dict()

        for class_id, class_name in enumerate(openai_imagenet_classnames):
            queue.put((class_id, class_name, return_dict))

        for _ in range(device_count):
            queue.put(None)  # sentinel value to signal subprocesses to exit
        for p in processes:
            p.join()  # wait for all subprocesses to finish

        batch_per_class_probs = [val for _, val in sorted(return_dict.items(),
                                                          key=lambda x: x[0])]
        import ipdb;
        ipdb.set_trace()

        # Tensor of shape [batch_size, 1000] where the [i,j]th element is
        # the (probability or loss) for batch element i on imagenet class j.
        batch_probs = torch.stack(batch_per_class_probs, 1)

        predictions_max_prob.extend(torch.argmax(batch_probs, 1).tolist())

        labels.extend(x['class_id'] for x in batch)

    acc_max_prob = (np.array(predictions_max_prob) == np.array(labels)).mean()

    print(f"[DEBUG] ImageNet accuracy with max prob method is {acc_max_prob}")

    print(f"[DEBUG] printing ImageNet predictions and labels:")
    for yhat_prob, y in zip(predictions_max_prob, labels):
        print(" " * 30 + f"label: {IMAGENET_1K_CLASS_ID_TO_LABEL[y]}"
                         f"\nprediction (max prob method): "
                         f"{IMAGENET_1K_CLASS_ID_TO_LABEL[yhat_prob]}"
                         "#" * 25)
    return acc_max_prob


if __name__ == "__main__":
    main()
