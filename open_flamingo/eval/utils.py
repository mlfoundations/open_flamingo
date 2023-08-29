import numpy as np
import torch
import random
import torch.nn as nn
from torch.utils.data import Subset
from contextlib import suppress
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


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


def compute_effective_num_shots(num_shots, model_type, true_zero_shot=False):
    """
    Compute the effective number of shots for a given model type.
    For example, following Flamingo, 0-shot OF evaluations use two text-only shots.
    """
    if model_type == "open_flamingo" and not true_zero_shot:
        return num_shots if num_shots > 0 else 2
    return num_shots


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    """
    Sample random demonstrations with replacement from the query set.
    Returns a torch Subset
    """
    return [Subset(query_set, np.random.choice(len(query_set), num_samples, replace=True)) for _ in range(batch_size)]

def sample_class_conditional_batch_demos_from_query_set(
    batch_class_ids,
    num_classes: int,
    query_set: Subset,
    num_samples: int,
):
    """
    Two-stage demo sampling procedure.
    1. Sample num_classes classes to include in the demo, being sure to include the true class (in batch_class_ids)
        Classes are only sampled from the classes in the query set.
        if the batch_class_ids contains classes not in the query set, raises an error.
    2. For each sampled class, sample floor(num_samples / num_classes); the remainder gets distributed among
        random classes. If there are fewer than num_classes samples, sample with replacement.
    Returns a list of torch Subsets
    """
    # sanity checks
    all_classes = torch.unique(query_set.class_id_array)
    assert num_classes <= len(all_classes), "Attempting to select more classes in the demo than there are classes in the dataset."
    if not isinstance(batch_class_ids, torch.Tensor):
        batch_class_ids = torch.LongTensor(batch_class_ids)
    if torch.any(~torch.isin(batch_class_ids, all_classes)):
        raise ValueError("batch_class_ids contains classes not in the query set.")
    if num_samples < num_classes:
        raise ValueError("num_samples must be >= num_classes.")
                         
    # sample classes + demos per class
    sampled_classes, sampled_demos = [], []
    samples_per_class = num_samples // num_classes
    leftover_samples = num_samples % num_classes
    for y in batch_class_ids:
        if isinstance(y, torch.Tensor): y = y.item()
        other_classes = np.setdiff1d(all_classes, [y]).tolist()
        classes = random.sample(other_classes, num_classes - 1) + [y]
        random.shuffle(classes)
        sampled_classes.append(classes)
        demos = [
            sample_examples_from_class(
                query_set,
                yp,
                samples_per_class + int(i < leftover_samples),
                replace_if_insufficient=True,
            )
            for i, yp in enumerate(classes)
        ]
        demos = [item for sublist in demos for item in sublist]
        random.shuffle(demos) # otherwise, examples will be in class chunks
        sampled_demos.append(Subset(query_set, demos))
        
    return sampled_classes, sampled_demos


def sample_examples_from_class(dataset, y, num_samples, replace_if_insufficient=False):
    """
    Given a class id y and a torch dataset containing examples from multiple classes,
    samples num_samples examples from class y uniformly at random.
    Returns: indices of selected examples
    """
    class_indices = torch.where(dataset.class_id_array == y)[0].tolist()
    selected_indices = random.sample(
        class_indices, min(num_samples, len(class_indices))
    )
    if len(selected_indices) < num_samples:
        print(f"Warning: insufficient samples in query set for class {y}, sampling with replacement={replace_if_insufficient}")
        if replace_if_insufficient:
            selected_indices += random.choices(
                class_indices, k=num_samples - len(selected_indices)
            )

    return selected_indices


def get_query_set(train_dataset, query_set_size):
    """
    Get a subset of the training dataset to use as the query set. Returns a torch Subset.
    Adds the "indices" attribute containing the indices of each example in the original set.
    """
    if query_set_size == -1: 
        train_dataset.indices = np.arange(len(train_dataset))
        return train_dataset
    query_set_indices = np.random.choice(len(train_dataset), query_set_size, replace=False)
    query_set = Subset(train_dataset, query_set_indices)
    if hasattr(train_dataset, "class_id_array"):
        query_set.class_id_array = train_dataset.class_id_array[query_set_indices]
        if len(np.unique(query_set.class_id_array)) != len(np.unique(train_dataset.class_id_array)):
            print(f"Warning: query set does not contain examples from all classes; {len(np.unique(query_set.class_id_array))} remaining classes.")
    query_set.indices = query_set_indices
    return query_set


def prepare_eval_samples(test_dataset, num_samples, batch_size):
    """
    Subset the test dataset and return a DataLoader.
    """
    if num_samples != -1:
        random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        dataset = Subset(test_dataset, random_indices)
    else:
        dataset = test_dataset
    sampler = DistributedSampler(dataset)
    loader = DataLoader(
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
        - top-k predicted class ixs (B, k) type int
        - top-k predicted classnames shape (B, k) type str
        - top-k logprobs shape (B, k) type float
    """
    # convert indices to classnames
    _, predictions = torch.topk(logprobs, k=k, dim=1)  # shape (B, k)
    predicted_classnames = [
        [class_id_to_name[ix] for ix in item] for item in predictions.tolist()
    ]
    predicted_logprobs = torch.gather(logprobs, 1, predictions)
    return predictions, predicted_classnames, predicted_logprobs


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

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def repeat_interleave(list, n):
    """
    Mimics torch.repeat_interelave for a list of arbitrary objects
    """
    return [item for item in list for _ in range(n)]

def reshape_nested_list(original_list, shape: tuple):
    """
    Reshapes a 2D list into a 2D list of shape shape
    """
    assert len(shape) == 2
    outer_list, inner_list = [], []
    for list in original_list:
        for x in list:
            inner_list.append(x)
            if len(inner_list) == shape[1]:
                outer_list.append(inner_list)
                inner_list = []
    if len(outer_list) != shape[0]:
        raise ValueError(f"List could not be reshaped to {shape}")
    return outer_list
