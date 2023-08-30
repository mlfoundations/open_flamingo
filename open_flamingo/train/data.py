"""
Preprocess and load datasets for training.
"""

import functools
import io
import json
import math
import re
import random
import numpy as np
import torch
import torchvision
import webdataset as wds
from PIL import Image
import base64

from data_utils import *

Image.MAX_IMAGE_PIXELS = 1000000000
N_CHANNELS = 3
MIN_KB = 10
_SHARD_SHUFFLE_SIZE = 10
_SHARD_SHUFFLE_INITIAL = 5
_SAMPLE_SHUFFLE_SIZE = 100
_SAMPLE_SHUFFLE_INITIAL = 50

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def preprocess_image(sample, image_processor):
    """
    Convert images to tensors for training.
    Augmentations: random horizontal flip.
    Normalization handled by wds.
    """
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    return image


def filter_no_caption_or_no_image(sample):
    """
    Filter out LAION samples with no caption or no image.
    """
    return ("txt" in sample) and (
        "png" in sample or "jpg" in sample or "jpeg" in sample
    )


def preprocess_laion_text(sample, tokenizer, max_tokens=128):
    """
    Preprocess text for LAION.
    Captions are truncated to 32 tokens by default.
    """
    tokenizer.padding_side = "right"
    prefix = "Below is an instruction that describes a task. "+ "Write a response that appropriately completes the request.\n\n" + "### Instruction:\n{Instruction}\n\n### Response:"

    sample = [
        prefix.format(Instruction=random.choice(CAPTION_TEMPLATES)) + s.strip() + "<|endofchunk|>" for s in sample
    ]
    text = tokenizer(
        sample,
        max_length=max_tokens,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]


def preprocess_gpt_interleaved(
    info, tokenizer, clip_processor, min_num_images, max_num_images, max_tokens=256
):
    """
    Preprocess a ChatGPT-generated image-text sequence.
    """
    text = info["example"]
    text = re.sub(r"_!_IMAGE\d+_!_", "<|endofchunk|><image>", text)

    # convert images from base64 to PIL
    images = []
    for image_key in range(1, len(info["image_map"]) + 1):
        image_base64 = info["image_map"][f"_!_IMAGE{image_key}_!_"]["base64_image"]
        rawbytes = base64.b64decode(image_base64)
        images.append(Image.open(io.BytesIO(rawbytes)).convert("RGB"))

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (max_num_images - len(images_tensors), 3, 224, 224), dtype=torch.float
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )

    indices = [m.start() for m in re.finditer("<image>", text)]
    if len(indices) > max_num_images:
        start_index = indices[max_num_images - 1]
        text = text[:start_index]

    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images after truncation
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )
    if num_images < min_num_images:
        raise ValueError(f"Fewer than {min_num_images} images in sample")

    return (images_tensors, (text_tensor["input_ids"], text_tensor["attention_mask"]))


def preprocess_interleaved(
    sample,
    tokenizer,
    clip_processor,
    sim_threshold,
    min_num_images,
    max_num_images,
    max_tokens=256,
):
    """
    Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
    is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
    """
    info = json.loads(sample[0])
    if "is_gpt" in info:
        return preprocess_gpt_interleaved(
            info, tokenizer, clip_processor, min_num_images, max_num_images, max_tokens
        )

    sentences = info["text_list"]
    sim_matrix = info["similarity_matrix"]

    # convert images from base64 to PIL and filter based on image-text similarity
    images, sentence_ixs = [], []
    for sample_image, sim_vec in zip(info["image_info"], sim_matrix):
        if "image_base64" not in sample_image:
            continue
        image_base64 = sample_image["image_base64"]
        rawbytes = base64.b64decode(image_base64)

        sim_ix = np.argmax(sim_vec)
        sim_score = sim_vec[sim_ix]

        # filter to images >= 10KB
        if len(rawbytes) // 1000 <= MIN_KB:
            continue
        if sim_score < sim_threshold:
            continue
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

        images.append(image)
        sentence_ixs.append(sim_ix)

    if len(images) == 0:
        raise ValueError("No images in sample")

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (
                max_num_images - len(images_tensors),
                N_CHANNELS,
                images_tensors[0].shape[1],
                images_tensors[0].shape[2],
            ),
            dtype=torch.float,
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    # add in <image> and <eoc> tokens
    for ix in sentence_ixs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"
    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )
    if num_images < min_num_images:
        raise ValueError(f"Fewer than {min_num_images} images in sample")
    elif (
        num_images == 1 and random.random() <= 0.5
    ):  # 50% chance of keeping single image samples
        raise ValueError("Only one image in sample")

    # avoid the situation where there's one <image> token and it's at the end
    if (
        num_images == 1
        and text_tensor["input_ids"][:, -1]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    ):
        raise ValueError(
            "Only one image at the end of sample, so labels will all be -100"
        )

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )


def get_mmc4_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for MMC4 / ChatGPT sequences
    """
    input_shards = args.mmc4_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_mmc4
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    preprocess_fn = functools.partial(
        preprocess_interleaved,
        clip_processor=image_processor,
        tokenizer=tokenizer,
        sim_threshold=args.mmc4_textsim_threshold,
        min_num_images=args.mmc4_min_num_images,
        max_num_images=args.mmc4_max_num_images,
    )

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.to_tuple("json", handler=log_and_continue),
            wds.map(preprocess_fn, handler=log_and_continue),
            wds.batched(args.batch_size_mmc4, partial=False),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_mmc4 * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_laion_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for LAION data
    """
    input_shards = args.laion_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_laion
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # create two preprocess functions that take in the passed in image_processor and tokenizer
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_laion_text, tokenizer=tokenizer)

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.to_tuple("jpg;png;jpeg", "txt", handler=log_and_continue),
            wds.batched(args.batch_size_laion, partial=False),
            wds.map_tuple(
                preprocess_image_fn, preprocess_text_fn, handler=log_and_continue
            ),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_laion * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

def preprocess_arxiv(sample, image_processor, tokenizer, max_tokens=256, max_num_images=5):
    info = json.loads(sample[0])
    caption = info["caption"]
    images_bytes = info["images_bytes"]

    # convert images from base64 to PIL and filter based on image-text similarity
    images = []
    for sample_image in images_bytes:
        image_base64 = sample_image
        rawbytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
        images.append(image)

    # preprocess and pad images
    images_tensors = preprocess_image(images, image_processor)
    
    # number of images
    num_images = min(len(images_tensors), max_num_images)
    
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (
                max_num_images - len(images_tensors),
                N_CHANNELS,
                images_tensors[0].shape[1],
                images_tensors[0].shape[2],
            ),
            dtype=torch.float,
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # truncate image_tensors to max_num_images
    images_tensors = images_tensors[:max_num_images]
        
    prefix = "Below is an instruction that describes a task. "+ "Write a response that appropriately completes the request.\n\n" + f"### Instruction:\nDescribe the findings of the figure.\n\n### Response:"
    
    text = f"{'<image>' * num_images}{caption}<|endofchunk|>"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        prefix + text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )

def get_arxiv_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for Arxiv data
    """
    input_shards = args.arxiv_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_arxiv
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    preprocess_fn = functools.partial(preprocess_arxiv, image_processor=image_processor, tokenizer=tokenizer)

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.to_tuple("json", handler=log_and_continue),
            wds.map(preprocess_fn, handler=log_and_continue),
            wds.batched(args.batch_size_arxiv, partial=False),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_arxiv * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

import torchvision
import io
import json
from PIL import Image
from transformers import AutoTokenizer
import open_clip
import cv2
import numpy as np
import tempfile
import random
import torch
import imageio
from PIL import Image
import numpy as np
from io import BytesIO


def read_frames_from_bytes(mp4_bytes):
    # Create an in-memory binary stream from the bytes
    mp4_stream = BytesIO(mp4_bytes)

    # Read the video using imageio
    video_reader = imageio.get_reader(mp4_stream, format="mp4")

    # Get the video's frame rate
    frame_rate = (
        video_reader.get_meta_data()["fps"]
        if video_reader.get_meta_data()["fps"] != None
        else 30
    )

    frames = []
    frame_count = 0
    for frame in video_reader:
        # Capture one frame per second
        if (
            frame_count % int(round(frame_rate)) == 0
        ):  # Round the frame rate to the nearest integer
            # Convert the imageio image (RGB) to a PIL image (RGB)
            frame_pil = Image.fromarray(np.array(frame))
            frames.append(frame_pil)

        frame_count += 1

    if len(frames) <= 1:
        # save the mp4 to a file to debug
        # with open("/fsx/home-anasawadalla/bad_mp4.mp4", "wb") as f:
        #     f.write(mp4_bytes)

        raise ValueError("No frames extracted from video. len(frames):", len(frames))
    return frames


def get_random_one_minute_subset(chunks):
    # Get the total duration of the transcript
    total_duration = chunks[-1]["timestamp"][1]
    if total_duration == None:
        total_duration = chunks[-1]["timestamp"][0]

    # If the total duration is less than 60 seconds, return the whole transcript
    if total_duration < 60:
        return chunks

    # Choose a random start time
    start_time = random.uniform(0, total_duration - 60)

    # Find the starting chunk
    start_chunk_index = next(
        i for i, chunk in enumerate(chunks) if chunk["timestamp"][1] > start_time
    )

    # Collect chunks until the total duration reaches 60 seconds or we have 5 chunks
    subset_chunks = []
    current_duration = 0
    for chunk in chunks[start_chunk_index:]:
        subset_chunks.append(chunk)
        current_duration += chunk["timestamp"][1] - chunk["timestamp"][0]
        if current_duration >= 60 or len(subset_chunks) == 5:
            break

    return subset_chunks


def get_evenly_spaced_frames(chunks, frames):
    chunk_frames = []

    for chunk in chunks:
        start_time, end_time = map(int, chunk["timestamp"])

        # Extract the frames for this chunk
        chunk_frame_indices = np.linspace(
            start_time, end_time, 5, endpoint=False
        ).astype(int)
        try:
            chunk_frames_list = [frames[i] for i in chunk_frame_indices]
        except Exception as e:
            raise ValueError(
                f"Error extracting frames: {e} chunk_frame_indices:{chunk_frame_indices} start_time:{start_time} end_time:{end_time} len(frames):{len(frames)}"
            )
        chunk_frames.append(chunk_frames_list)

    return chunk_frames


def preprocess_video(sample, tokenizer, clip_processor, max_tokens=512):
    """
    Preprocess a video sample.
    """
    mp4_bytes = sample[0]
    json_transcript = sample[1]
    json_transcript = json.loads(json_transcript)

    if "chunks" not in json_transcript:
        raise ValueError("No chunks in sample")

    frames = read_frames_from_bytes(mp4_bytes)
    subset_chunks = get_random_one_minute_subset(json_transcript["chunks"])
    chunk_frames = get_evenly_spaced_frames(subset_chunks, frames)

    # if there are more than 5 chunks, truncate to 5 chunks and 5 frames
    if len(chunk_frames) > 5:
        chunk_frames = chunk_frames[:5]
        subset_chunks = subset_chunks[:5]

    # Encode the transcript from subset_chunks
    text = "<|endofchunk|><image>".join([chunk["text"] for chunk in subset_chunks])

    # remove the first <|endofchunk|> from the transcript
    text = text.replace("<|endofchunk|>", "", 1)

    # Tokenize the transcript
    text_tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
    )

    # Encode the frames for each frame separately
    video_tensors = []
    for chunk_frames_list in chunk_frames:
        video_tensors.append(
            preprocess_image(chunk_frames_list, clip_processor).unsqueeze(0)
        )

    video = torch.cat(video_tensors, dim=0)

    # pad the video to 5 images
    if video.shape[0] < 5:
        zero_padding = torch.zeros(
            (5 - video.shape[0], 5, 3, 224, 224), dtype=torch.float
        )
        video = torch.cat((video, zero_padding), dim=0)

    return video, (text_tokens["input_ids"], text_tokens["attention_mask"])


def get_video_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for Video data
    """
    input_shards = args.video_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_video
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    preprocess_fn = functools.partial(
        preprocess_video, clip_processor=image_processor, tokenizer=tokenizer
    )

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.to_tuple("mp4", "json", handler=log_and_continue),
            wds.map(preprocess_fn, handler=log_and_continue),
            wds.batched(args.batch_size_video, partial=False),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_video * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_fn(dataset_type):
    """
    Helper function to get the dataset function based on the dataset type
    """
    if dataset_type == "image_text":
        return get_laion_dataset
    elif dataset_type == "mmc4":
        return get_mmc4_dataset
    elif dataset_type == "video":
        return get_video_dataset
    elif dataset_type == "arxiv":
        return get_arxiv_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, image_processor, tokenizer, dataset_type, epoch=0):
    """
    Interface for getting the webdatasets
    """
    return get_dataset_fn(dataset_type)(
        args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer
    )
