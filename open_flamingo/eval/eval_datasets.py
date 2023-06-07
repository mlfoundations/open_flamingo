import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_train_dir_path,
        annotations_path,
        is_train,
        dataset_name,
        image_val_dir_path=None,
    ):
        self.image_train_dir_path = image_train_dir_path
        self.image_val_dir_path = image_val_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name

        full_annotations = json.load(open(annotations_path))["images"]

        for i in range(len(full_annotations)):
            if self.is_train and full_annotations[i]["split"] != "train":
                continue
            elif not self.is_train and full_annotations[i]["split"] != "test":
                continue

            self.annotations.append(full_annotations[i])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
                if self.annotations[idx]["filepath"] == "train2014"
                else os.path.join(
                    self.image_val_dir_path, self.annotations[idx]["filename"]
                )
            )
        elif self.dataset_name == "flickr":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
            )
        image.load()
        caption = self.annotations[idx]["sentences"][0]["raw"]
        return {
            "image": image,
            "caption": caption,
            "image_id": self.annotations[idx]["cocoid"]
            if self.dataset_name == "coco"
            else self.annotations[idx]["filename"].split(".")[0],
        }


class VQADataset(Dataset):
    def __init__(
        self, image_dir_path, question_path, annotations_path, is_train, dataset_name
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        self.answers = json.load(open(annotations_path, "r"))["annotations"]
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok-vqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_train2014_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_val2014_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        return {
            "image": image,
            "question": question["question"],
            "answers": [a["answer"] for a in answers["answers"]],
            "question_id": question["question_id"],
        }

class NoCapsDataset(Dataset):
    def __init__(self, image_dir_path, annotations_path):
        self.image_dir_path = image_dir_path
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        
        self.images = annotations["images"]
        self.annotations = annotations["annotations"]
        
        # assert annotations is 10x the size of images
        assert len(self.annotations) == 10*len(self.images)
            
    def __len__(self):
        return len(self.annotations)//10 # 10 captions per image listed sequentially
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx*10]
        image = Image.open(os.path.join(self.image_dir_path, self.images[idx]["file_name"]))
        image.load()
        return {
            "image": image,
            "caption": annotation["caption"],
            "image_id": annotation["image_id"]
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
            "class_name": target_label,  # human-readable name of ImageNet class
        }


class HatefulMemesDataset(Dataset):
    def __init__(self, image_dir_path, annotations_path):
        self.image_dir_path = image_dir_path
        with open(annotations_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir_path, annotation["img"].split("/")[-1])
        image = Image.open(img_path)
        image.load()
        return {
            "image": image,
            "ocr": annotation["text"],
            "class_name": "yes" if annotation["label"] == 1 else "no",
            "class_id": annotation["label"],
        }

class ScienceQADataset(Dataset):
    def __init__(self, image_dir_path, annotations_path, is_train):
        self.image_dir_path = image_dir_path
        with open(annotations_path, "r") as f:
            self.annotations = json.load(f)
            
        # remove entries with no image or split is wrong
        self.q_ids = [key for key in self.annotations if (self.annotations[key]["image"] != None) and (self.annotations[key]["split"] == "train" if is_train else self.annotations[key]["split"] == "test")]

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, idx):
        annotation = self.annotations[self.q_ids[idx]]
        img_path = os.path.join(self.image_dir_path, self.q_ids[idx], "image.png")
        image = Image.open(img_path)
        image.load()
        return {
            "image": image,
            "context": annotation["lecture"],
            "question": annotation["question"],
            "choices": annotation["choices"],
            "class_name": annotation["choices"][annotation["answer"]],
            "class_id": annotation["answer"],
        }

class IconQADataset(Dataset):
    def __init__(self, image_dir_path):
        self.image_dir_paths = [os.path.join(image_dir_path, d) for d in os.listdir(image_dir_path)]
        
    def __len__(self):
        return len(self.image_dir_paths)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.image_dir_paths[idx], "image.png")
        image = Image.open(img_path)
        image.load()
        
        with open(os.path.join(self.image_dir_paths[idx], "data.json"), "r") as f:
            annotation = json.load(f)
        
        return {
            "image": image,
            "question": annotation["question"],
            "choices": annotation["choices"],
            "class_name": annotation["choices"][annotation["answer"]],
            "class_id": annotation["answer"],
        }
        
import requests
from io import BytesIO
class VSRDataset(Dataset):
    def __init__(self, annotations_path):
        with open(annotations_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]
            
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        url = annotation["image_link"]
        # load image from url
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image.load()
        
        # image = Image.open()
        # image.load()
        return {
            "image": image,
            "caption": annotation["caption"],
            "class_name": "(True)" if annotation["label"] == 1 else "(False)",
            "class_id": annotation["label"],
        }