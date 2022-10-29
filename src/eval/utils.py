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
