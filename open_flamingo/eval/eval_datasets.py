import json
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Mapping

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from open_flamingo.eval.imagenet_utils import IMAGENET_1K_CLASS_ID_TO_LABEL


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
            self,
            image_dir_path,
            question_path,
            annotations_path,
            is_train,
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        self.answers = json.load(open(annotations_path, "r"))["annotations"]
        self.image_dir_path = image_dir_path
        self.is_train = is_train

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        return os.path.join(
            self.image_dir_path,
            f"COCO_train2014_{question['image_id']:012d}.jpg"
            if self.is_train
            else f"COCO_val2014_{question['image_id']:012d}.jpg",
        )

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


def topk(probs_ary: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the top k elements in probs_ary."""
    return np.argsort(probs_ary)[::-1][:k]


@dataclass
class ClassificationDataset:
    """Class to hold a classification dataset for evals.

    All Dataset objects (train_dataset, val_dataset, test_dataset)
        should return a dictionary containing at least the
        following keys: image, class_id, class_name. See
        ImageNetDataset for an example.
    """
    train_dataset: Dataset
    prompts: Sequence[str] = field(
        metadata={"help": "A sequence of prompts to be used during evaluation;"
                          "e.g. 'A photo of a'. It is recommended to 'strip' the prompt (remove leading/trailing "
                          "spaces) for best performance."}
    )
    class_id_to_label: Mapping[int, str] = field(
        metadata={"help": "mapping of numeric class IDs to string class names/labels."
                          "Downstream metrics will be evaluated against the mapped strings."})
    val_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None

    def get_in_context_samples(self, num: int, **kwargs) -> Sequence[int]:
        """Fetch a set of `num` in-context sample indices."""
        return np.random.choice(
            len(self.train_dataset), num, replace=False
        )

    def metric_fn(self, labels: Sequence[int], outputs: Sequence[float]) -> Mapping[str, float]:
        """
        Compute metrics for a set of labels and predictions.

        labels: An array-like of shape [batch_size,]
        outputs: Model outputs; an array-like of shape [batch_size, num_classes]. The
            [i,j]^th element of outputs should correspond to the probability
            that the i^th observation has numeric class label j.
        """
        batch_size = len(labels)

        # Sanity check that batch size is consistent
        assert len(outputs) == len(labels)

        # Sanity check that outputs has same dimension as class mapping.
        assert outputs.shape[1] == len(self.class_id_to_label)

        acc5 = 0.
        acc1 = 0.

        for i in range(batch_size):
            top5 = [
                self.class_id_to_label[pred]
                for pred in topk(outputs[i], 5)
            ]

            y_i = labels[i]["class_name"]
            acc5 += int(y_i in set(top5))
            acc1 += int(y_i == top5[0])

            print(
                f"[DEBUG]: elem {i} of {batch_size}:"
                f"label {y_i} // top5 {top5}"
            )
        return {"acc1": acc1, "acc5": acc5}


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
