import ast
import functools
import json
import logging
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from multiprocessing import Value


import braceexpand
import torch
import torchvision
import webdataset as wds
from nltk import sent_tokenize
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import base64
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)
from webdataset_utils import ResampledShards2, SharedEpoch, DataInfo, get_shard_stats, prepare_dataloader, add_detshuffle2_step, add_pile_data_processing_step, add_interleaved_data_processing_step, add_image_text_data_processing_step, add_tar_to_samples_step
from image_text_utils import preprocess_image, preprocess_text
from interleaved_utils import preprocess_interleaved_data
from pile_utils import preprocess_pile

from PIL import Image
import io


Image.MAX_IMAGE_PIXELS = 1000000000

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None




def create_dataset_pipeline_without_resampling(input_shards, shared_epoch, args, preprocess_data_fn):
    pipeline = [wds.SimpleShardList(input_shards)]
    add_detshuffle2_step(shared_epoch, args, pipeline)
    add_tar_to_samples_step(pipeline)
    if args.dataset_type == "interleaved":
        add_interleaved_data_processing_step(args, preprocess_data_fn, pipeline)
    elif args.dataset_type == "pile":
        add_pile_data_processing_step(args, preprocess_data_fn, pipeline)
    elif args.data_type == "image_text":
        add_image_text_data_processing_step(args, preprocess_data_fn, pipeline)
    dataset = wds.DataPipeline(*pipeline)
    return dataset

def create_dataset_pipeline_with_resampling(input_shards, shared_epoch, args, preprocess_data_fn):
    pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    add_tar_to_samples_step(pipeline)
    if args.dataset_type == "interleaved":
        add_interleaved_data_processing_step(args, preprocess_data_fn, pipeline)
    elif args.dataset_type == "pile":
        add_pile_data_processing_step(args, preprocess_data_fn, pipeline)
    elif args.dataset_type == "image_text":
        add_image_text_data_processing_step(args, preprocess_data_fn, pipeline)
    dataset = wds.DataPipeline(*pipeline)
    return dataset

def get_pile_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    input_shards = args.shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_shard_stats(args, input_shards)

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_pile_fn = functools.partial(preprocess_pile, clip_processor=image_processor, tokenizer=tokenizer)
    preprocess_fn = [preprocess_pile_fn]

    if resampled:
        dataset = create_dataset_pipeline_with_resampling(input_shards,shared_epoch, args, preprocess_fn)
    else:
        dataset = create_dataset_pipeline_without_resampling(input_shards, shared_epoch, args, preprocess_fn)
        assert (num_shards >= args.workers * args.world_size), "number of shards must be >= total workers"

    dataloader = prepare_dataloader(args, floor, num_samples, dataset)
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_laion_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    input_shards = args.shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_shard_stats(args, input_shards)

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(preprocess_image, image_processor=image_processor)
    preprocess_text_fn = functools.partial(preprocess_text, tokenizer=tokenizer)

    preprocess_fn = [preprocess_image_fn, preprocess_text_fn]
    if resampled:
        dataset = create_dataset_pipeline_with_resampling(input_shards,shared_epoch, args, preprocess_fn)
    else:
        dataset = create_dataset_pipeline_without_resampling(input_shards, shared_epoch, args, preprocess_fn)
        assert (num_shards >= args.workers * args.world_size), "number of shards must be >= total workers"

    dataloader = prepare_dataloader(args, floor, num_samples, dataset)
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

def get_interleaved_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    input_shards = args.shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_shard_stats(args, input_shards)

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_interleaved_data_fn = functools.partial(preprocess_interleaved_data, text_tokenizer=tokenizer, image_processor = image_processor)
    preprocess_fn = [preprocess_interleaved_data_fn]

    if resampled:
        dataset = create_dataset_pipeline_with_resampling(input_shards,shared_epoch, args, preprocess_fn)
    else:
        dataset = create_dataset_pipeline_without_resampling(input_shards, shared_epoch, args, preprocess_fn)
        assert (num_shards >= args.workers * args.world_size), "number of shards must be >= total workers"

    dataloader = prepare_dataloader(args, floor, num_samples, dataset)
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)
    
def get_dataset_fn(dataset_type):
    if dataset_type == "image_text":
        return get_laion_dataset
    elif dataset_type == "pile":
        return get_pile_dataset
    elif dataset_type == "interleaved":
        return get_interleaved_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, image_processor, tokenizer, epoch=0):
    return get_dataset_fn(args.dataset_type)(
        args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer
    )