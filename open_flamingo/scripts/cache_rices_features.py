"""
Cache CLIP features for all images in training split in preparation for RICES
"""
import argparse
import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
    )
)
from eval.rices import RICES
from eval.eval_datasets import (
    CaptionDataset,
    VQADataset,
    ImageNetDataset,
    HatefulMemesDataset,
)
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save the cached features.",
)
parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
parser.add_argument("--batch_size", default=256)

# Per-dataset flags
parser.add_argument(
    "--eval_coco",
    action="store_true",
    default=False,
    help="Whether to cache COCO.",
)
parser.add_argument(
    "--eval_vqav2",
    action="store_true",
    default=False,
    help="Whether to cache VQAV2.",
)
parser.add_argument(
    "--eval_ok_vqa",
    action="store_true",
    default=False,
    help="Whether to cache OK-VQA.",
)
parser.add_argument(
    "--eval_vizwiz",
    action="store_true",
    default=False,
    help="Whether to cache VizWiz.",
)
parser.add_argument(
    "--eval_textvqa",
    action="store_true",
    default=False,
    help="Whether to cache TextVQA.",
)
parser.add_argument(
    "--eval_imagenet",
    action="store_true",
    default=False,
    help="Whether to cache ImageNet.",
)
parser.add_argument(
    "--eval_flickr30",
    action="store_true",
    default=False,
    help="Whether to cache Flickr30.",
)
parser.add_argument(
    "--eval_hateful_memes",
    action="store_true",
    default=False,
    help="Whether to cache Hateful Memes.",
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

## VizWiz Dataset
parser.add_argument(
    "--vizwiz_train_image_dir_path",
    type=str,
    help="Path to the vizwiz train images directory.",
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


def main():
    args, leftovers = parser.parse_known_args()
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    if args.eval_flickr30:
        print("Caching Flickr30k...")
        train_dataset = CaptionDataset(
            image_train_dir_path=args.flickr_image_dir_path,
            image_val_dir_path=None,
            annotations_path=args.flickr_karpathy_json_path,
            is_train=True,
            dataset_name="flickr",
        )
        rices_dataset = RICES(
            train_dataset,
            device_id,
            args.batch_size,
            vision_encoder_path=args.vision_encoder_path,
            vision_encoder_pretrained=args.vision_encoder_pretrained,
        )
        torch.save(
            rices_dataset.features,
            os.path.join(args.output_dir, "flickr30.pkl"),
        )

    if args.eval_coco:
        print("Caching COCO...")
        train_dataset = CaptionDataset(
            image_train_dir_path=args.coco_train_image_dir_path,
            image_val_dir_path=args.coco_val_image_dir_path,
            annotations_path=args.coco_karpathy_json_path,
            is_train=True,
            dataset_name="coco",
        )
        rices_dataset = RICES(
            train_dataset,
            device_id,
            args.batch_size,
            vision_encoder_path=args.vision_encoder_path,
            vision_encoder_pretrained=args.vision_encoder_pretrained,
        )
        torch.save(
            rices_dataset.features,
            os.path.join(args.output_dir, "coco.pkl"),
        )

    if args.eval_ok_vqa:
        print("Caching OK-VQA...")
        train_dataset = VQADataset(
            image_dir_path=args.ok_vqa_train_image_dir_path,
            question_path=args.ok_vqa_train_questions_json_path,
            annotations_path=args.ok_vqa_train_annotations_json_path,
            is_train=True,
            dataset_name="ok_vqa",
        )
        rices_dataset = RICES(
            train_dataset,
            device_id,
            args.batch_size,
            vision_encoder_path=args.vision_encoder_path,
            vision_encoder_pretrained=args.vision_encoder_pretrained,
        )
        torch.save(
            rices_dataset.features,
            os.path.join(args.output_dir, "ok_vqa.pkl"),
        )

    if args.eval_vizwiz:
        print("Caching VizWiz...")
        train_dataset = VQADataset(
            image_dir_path=args.vizwiz_train_image_dir_path,
            question_path=args.vizwiz_train_questions_json_path,
            annotations_path=args.vizwiz_train_annotations_json_path,
            is_train=True,
            dataset_name="vizwiz",
        )
        rices_dataset = RICES(
            train_dataset,
            device_id,
            args.batch_size,
            vision_encoder_path=args.vision_encoder_path,
            vision_encoder_pretrained=args.vision_encoder_pretrained,
        )
        torch.save(
            rices_dataset.features,
            os.path.join(args.output_dir, "vizwiz.pkl"),
        )

    if args.eval_vqav2:
        print("Caching VQAv2...")
        train_dataset = VQADataset(
            image_dir_path=args.vqav2_train_image_dir_path,
            question_path=args.vqav2_train_questions_json_path,
            annotations_path=args.vqav2_train_annotations_json_path,
            is_train=True,
            dataset_name="vqav2",
        )
        rices_dataset = RICES(
            train_dataset,
            device_id,
            args.batch_size,
            vision_encoder_path=args.vision_encoder_path,
            vision_encoder_pretrained=args.vision_encoder_pretrained,
        )
        torch.save(
            rices_dataset.features,
            os.path.join(args.output_dir, "vqav2.pkl"),
        )

    if args.eval_textvqa:
        print("Caching TextVQA...")
        train_dataset = VQADataset(
            image_dir_path=args.textvqa_image_dir_path,
            question_path=args.textvqa_train_questions_json_path,
            annotations_path=args.textvqa_train_annotations_json_path,
            is_train=True,
            dataset_name="textvqa",
        )
        rices_dataset = RICES(
            train_dataset,
            device_id,
            args.batch_size,
            vision_encoder_path=args.vision_encoder_path,
            vision_encoder_pretrained=args.vision_encoder_pretrained,
        )
        torch.save(
            rices_dataset.features,
            os.path.join(args.output_dir, "textvqa.pkl"),
        )

    if args.eval_hateful_memes:
        print("Caching Hateful Memes...")
        train_dataset = HatefulMemesDataset(
            image_dir_path=args.hateful_memes_image_dir_path,
            annotations_path=args.hateful_memes_train_annotations_json_path,
        )
        rices_dataset = RICES(
            train_dataset,
            device_id,
            args.batch_size,
            vision_encoder_path=args.vision_encoder_path,
            vision_encoder_pretrained=args.vision_encoder_pretrained,
        )
        torch.save(
            rices_dataset.features,
            os.path.join(args.output_dir, "hateful_memes.pkl"),
        )


if __name__ == "__main__":
    main()
