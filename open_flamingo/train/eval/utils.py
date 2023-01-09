import re

from .cider import Cider
from .vqa_eval import VQAEval


def compute_vqa_accuracy(predictions, ground_truth):
    """Compute the VQA accuracy metric.

    Args:
        predictions (List): list of predictions
        ground_truth (List[List]): list of all possible ground truth answers

    Returns:
        float: VQA accuracy
    """
    return VQAEval().evaluate(predictions, ground_truth)


def postprocess_vqa_generation(predictions):
    return re.split("answer:|question:|Question:|Answer:", predictions, 1)[0]


def compute_cider(predictions, ground_truth):
    """Compute the CIDEr metric.

    Args:
        predictions (List): list of predictions
        ground_truth (List[List]): list of all possible ground truth answers

    Returns:
        float: CIDEr score
    """
    metric = Cider()
    # convert to dict with a stub image_id as key to match the CIDEr API
    predictions = {i: [p] for i, p in enumerate(predictions)}
    ground_truth = {i: [g] for i, g in enumerate(ground_truth)}

    return metric.compute_score(ground_truth, predictions)[0]


def postprocess_captioning_generation(predictions):
    return predictions.split("Output:", 1)[0]
