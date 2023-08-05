import numpy as np
import torch
import random
import torch.nn as nn
from contextlib import suppress


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def compute_effective_num_shots(num_shots, model_type):
    """
    Compute the effective number of shots for a given model type.
    For example, following Flamingo, 0-shot OF evaluations use two text-only shots.
    """
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    """
    Sample random demonstrations from the query set.
    """
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


def get_query_set(train_dataset, query_set_size):
    """
    Get a subset of the training dataset to use as the query set.
    """
    query_set = np.random.choice(len(train_dataset), query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, batch_size):
    """
    Subset the test dataset and return a DataLoader.
    """
    random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    dataset = torch.utils.data.Subset(test_dataset, random_indices)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )
    return loader


def get_indices_of_unique(x):
    """
    Return the indices of x that correspond to unique elements.
    If value v is unique and two indices in x have value v, the first index is returned.
    """
    unique_elements = torch.unique(x)
    first_indices = []
    for v in unique_elements:
        indices = torch.where(x == v)[0]
        first_indices.append(indices[0])  # Take the first index for each unique element
    return torch.tensor(first_indices)


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model


def get_predicted_classnames(logprobs, k, class_id_to_name):
    """
    Args:
        - logprobs shape (B, Y) containing logprobs for each classname
        - k: number for top-k
        - class_id_to_name: dict mapping class index to classname

    Returns:
        - top-k predicted classnames shape (B, k) type str
        - top-k logprobs shape (B, k) type float
    """
    # convert indices to classnames
    _, predictions = torch.topk(logprobs, k=k, dim=1)  # shape (B, k)
    predicted_classnames = [
        [class_id_to_name[ix] for ix in item] for item in predictions.tolist()
    ]
    predicted_logprobs = torch.gather(logprobs, 1, predictions)
    return predicted_classnames, predicted_logprobs


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
