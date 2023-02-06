import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


# class OKVQADataset(Dataset):
#     def __init__(self, data_dir, split):
#         if split not in ["validation", "test"]:
#             raise ValueError("Split must be either validation or test")
#         self.data_type = "train2014" if split == "validation" else "val2014"
#         self.data_dir = data_dir
#         self.annotations = json.load(
#             open(f"{self.data_dir}/mscoco_{self.data_type}_annotations.json"))["annotations"]
#         self.questions = json.load(open(f"{self.data_dir}/OpenEnded_mscoco_{self.data_type}_questions.json"))[
#             "questions"
#         ]

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         image = Image.open(
#             f"{self.data_dir}/{self.data_type}/COCO_{self.data_type}_{self.annotations[idx]['image_id']:012d}.jpg"
#         )
#         question = self.questions[idx]["question"]
#         answers = [x["answer"] for x in self.annotations[idx]["answers"]]
#         return {"image": image, "question": question, "answers": answers}


class COCODataset(Dataset):
    def __init__(self,
                 image_dir_path="/mmfs1/gscratch/efml/anasa2/data/coco/train2017/",
                 annotations_path="/mmfs1/gscratch/efml/anasa2/data/coco/annotations/captions_train2017.json"):
        self.image_dir_path = image_dir_path
        self.annotations = json.load(open(annotations_path))["annotations"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image = Image.open(
            f"{self.image_dir_path}/{self.annotations[idx]['image_id']:012d}.jpg")
        caption = self.annotations[idx]["caption"]
        return {"image": image, "caption": caption, "image_id": self.annotations[idx]["image_id"]}


class VQAv2Dataset(Dataset):
    def __init__(
        self,
        image_dir_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/train2014/",
        question_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json",
        annotations_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_mscoco_train2014_annotations.json",
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        self.answers = json.load(open(annotations_path, "r"))["annotations"]
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        image = Image.open(os.path.join(self.image_dir_path,
                           f"COCO_train2014_{question['image_id']:012d}.jpg"))

        return {"image": image, "question": question["question"], "answers": [a["answer"] for a in answers["answers"]], "": question["question_id"]}


class QADataset(Dataset):
    def __init__(self, vqa_dataset, image_processor, tokenizer, in_context_samples):
        self.vqa_dataset = vqa_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.in_context_samples = in_context_samples

        self.context = " ".join([self.get_prompt(s)
                                for s in in_context_samples])
        self.demo_images = self.image_processor(
            images=[s["image"] for s in in_context_samples], return_tensors="pt")["pixel_values"]
        self.demo_images = self.demo_images.unsqueeze(1).unsqueeze(0)

    def get_prompt(sample, train=True):
        return f"<image>Question:{sample['question']} Answer:{sample['answers'][0] if train else ''}{'<|endofchunk|>' if train else ''}"

    def __len__(self):
        return len(self.vqa_dataset)

    def __getitem__(self, idx):
        item = self.vqa_dataset[idx]
        image = self.image_processor(images=[item["image"]], return_tensors="pt")[
            "pixel_values"].unsqueeze(1).unsqueeze(0)
        image = torch.cat([self.demo_images, image], dim=1)

        self.tokenizer.padding_side = "left"
        text = self.tokenizer(self.context + self.get_prompt(item, train=False), padding="longest",
                              truncation=True, max_length=512, return_tensors="pt")

        return {"image": image, "input_ids": text["input_ids"], "attention_mask": text["attention_mask"], "question_id": item["question_id"]}


class CaptioningDataset(Dataset):
    def __init__(self, coco_dataset, image_processor, tokenizer, in_context_samples):
        self.coco_dataset = coco_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.in_context_samples = in_context_samples

        self.context = " ".join([self.get_prompt(s)
                                for s in in_context_samples])
        self.demo_images = self.image_processor(
            images=[s["image"] for s in in_context_samples], return_tensors="pt")["pixel_values"]
        self.demo_images = self.demo_images.unsqueeze(1).unsqueeze(0)

    def get_prompt(sample, train=True):
        return f"<image>Output:{sample['caption']}{'<|endofchunk|>' if train else ''}"

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        item = self.coco_dataset[idx]
        image = self.image_processor(images=[item["image"]], return_tensors="pt")[
            "pixel_values"].unsqueeze(1).unsqueeze(0)
        image = torch.cat([self.demo_images, image], dim=1)

        self.tokenizer.padding_side = "left"
        text = self.tokenizer(self.context + self.get_prompt(item, train=False), padding="longest",
                              truncation=True, max_length=512, return_tensors="pt")

        return {"image": image, "input_ids": text["input_ids"], "attention_mask": text["attention_mask"], "image_id": item["image_id"]}
