import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from open_flamingo.eval.imagenet_utils import IMAGENET_1K_CLASS_ID_TO_LABEL


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


class COCOFlickrDataset(Dataset):
    def __init__(
        self,
        image_dir_path="/mmfs1/gscratch/efml/anasa2/data/coco/train2017/",
        annotations_path="/mmfs1/gscratch/efml/anasa2/data/coco/annotations/captions_train2017.json",
        is_flickr=False,
    ):
        self.image_dir_path = image_dir_path
        self.annotations = json.load(open(annotations_path))["annotations"]
        self.is_flickr = is_flickr

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, idx):
        if self.is_flickr:
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']}.jpg"
        else:
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']:012d}.jpg"


    def __getitem__(self, idx):
        image = Image.open(self.get_img_path(idx))
        caption = self.annotations[idx]["caption"]
        return {
            "image": image,
            "caption": caption,
            "image_id": self.annotations[idx]["image_id"],
        }


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
        image = Image.open(
            os.path.join(
                self.image_dir_path, f"COCO_train2014_{question['image_id']:012d}.jpg"
            )
        )

        return {
            "image": image,
            "question": question["question"],
            "answers": [a["answer"] for a in answers["answers"]],
            "question_id": question["question_id"],
        }


class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""
    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return {
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label  # human-readable name of ImageNet class
        }

