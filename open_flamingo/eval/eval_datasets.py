import json
import os

from PIL import Image
from torch.utils.data import Dataset


class OKVQADataset(Dataset):
    def __init__(self, data_dir, split):
        if split not in ["validation", "test"]:
            raise ValueError("Split must be either validation or test")
        # using the train2014 set for validation
        self.data_type = "train2014" if split == "validation" else "val2014"
        self.data_dir = data_dir
        self.annotations = json.load(
            open(f"{self.data_dir}/mscoco_{self.data_type}_annotations.json"))["annotations"]
        self.questions = json.load(open(f"{self.data_dir}/OpenEnded_mscoco_{self.data_type}_questions.json"))[
            "questions"
        ]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image = Image.open(
            f"{self.data_dir}/{self.data_type}/COCO_{self.data_type}_{self.annotations[idx]['image_id']:012d}.jpg"
        )
        question = self.questions[idx]["question"]
        answers = [x["answer"] for x in self.annotations[idx]["answers"]]
        return {"image": image, "question": question, "answers": answers}


class COCODataset(Dataset):
    def __init__(self, data_dir, split):
        if split not in ["validation", "test"]:
            raise ValueError("Split must be either validation or test")
        # using the train2014 set for validation
        self.data_type = "train2017" if split == "validation" else "val2017"
        self.data_dir = data_dir
        self.annotations = json.load(open(
            f"{self.data_dir}/annotations/captions_{self.data_type}.json"))["annotations"]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image = Image.open(
            f"{self.data_dir}/{self.data_type}/{self.annotations[idx]['image_id']:012d}.jpg")
        caption = self.annotations[idx]["caption"]
        return {"image": image, "caption": caption, "image_id": self.annotations[idx]["image_id"]}


class VQAv2Dataset(Dataset):
    def __init__(
        self,
        question_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json",
        answers_path="/mmfs1/gscratch/efml/anasa2/data/vqav2/v2_mscoco_train2014_annotations.json",
        image_dir="/mmfs1/gscratch/efml/anasa2/data/vqav2/train2014/",
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        self.answers = json.load(open(answers_path, "r"))["annotations"]
        self.image_dir = image_dir

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        image_path = os.path.join(
            self.image_dir, f"COCO_train2014_{question['image_id']:012d}.jpg")
        image = Image.open(image_path)

        return {"image": image, "question": question["question"], "answers": [a["answer"] for a in answers["answers"]], "question_id": question["question_id"]}
