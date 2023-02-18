from typing import Dict, Sequence
import re
import numpy as np
import torch

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


def compute_per_sample_loss(encodings, tokenizer, outputs):
    """Helper function to compute per-sample classification loss.

    Assumes <eos token> is used to separate inputs from targets in the
    prompt text
    """
    labels = encodings["input_ids"].clone()
    # convert padding tokens to -100 so they are ignored in loss
    labels[labels == tokenizer.pad_token_id] = -100

    # Convert all tokens in prefix until separator to -100 so they are
    # ignored in loss (loss is only computed on token IDs >= 0)
    for idx in range(len(labels)):
        # Find the location of the last token of prefix *from right*,
        # since the first non-padding token of the sequence will also be
        # eos_token (because bos_token and eos_token are the same for
        # the tokenizer).
        end_of_prefix = -labels[idx].tolist()[::-1].index(
            tokenizer.eos_token_id) - 1
        labels[idx, :end_of_prefix + 1] = -100

    # Shift so that tokens < n predict n. The shifted tensors both have
    # shape [batch_size, seq_len - 1].
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    # Loss is computed token-wise, on Tensors of shape
    # [batch_size * (seq_len - 1), vocab_size]
    # and returns a loss tensor of shape
    # [batch_size * (seq_len - 1)]. Most of the tokens will be masked
    # in this computation.
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                   shift_labels.view(-1).to(device))

    # Reshape to [batch_size, seq_len - 1]
    loss = loss.view(shift_logits.size(0), shift_logits.size(1)).cpu()

    # Compute per-element loss : sum loss over all tokens and divide by
    # number of variable tokens to obtain tensor of shape [batch_size,]
    loss = loss.sum(dim=1) / (labels != -100).sum(dim=1).float()
    return loss