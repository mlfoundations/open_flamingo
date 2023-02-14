from typing import Dict, Sequence
import re
import numpy as np


def postprocess_classification_generation(predictions) -> str:
    return re.split("Prompt|Completion", predictions, 1)[0]


def compute_classification_accuracy(
        predictions: Sequence[Dict[str, str]]) -> float:
    """Compute the accuracy of a sequence of predictions."""

    def _preprocess_fn(s):
        """Function to preprocess both targets and predictions."""
        return s.lower()

    is_correct = [
        _preprocess_fn(x["prediction"]) == _preprocess_fn(x["class_label"])
        for x in predictions]

    return np.mean(is_correct).item()
