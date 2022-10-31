import logging

import numpy as np
import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import more_itertools
import re

from .utils import compute_vqa_accuracy

def evaluate_text_vqa(model, tokenizer, image_processor, batch_size, max_generation_length=30, num_samples=None, device=-1):
    """Evaluate a model on TextVQA.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor (transformers.ImageProcessor): image processor for the model
        batch_size (int): batch size
        num_samples (int, optional): number of samples to evaluate on (for testing). Defaults to None (entire test set).
        device (int, optional): device to use. Defaults to -1 (cpu).

    Returns:
        dict: dictionary of metrics
    """
    logging.info("Evaluating on TextVQA...")

    dataset = load_dataset("textvqa", split="validation")

    if num_samples is not None:
        dataset = dataset.shuffle().select(range(num_samples))

    model.eval()
    model.to(device if device >= 0 else "cpu")

    predictions = []
    
    def postprocess_generation(predictions):
        generated_tokens = predictions.split("answer:", 1)[1]
        return re.split("answer:|question:", generated_tokens, 1)[0]

    for batch in more_itertools.chunked(tqdm(dataset), batch_size):
        images = image_processor(
            images=[b["image"] for b in batch], return_tensors="pt")["pixel_values"]

        encodings = tokenizer([(f"<image> question:{row['question']} answer:") for row in batch], 
                              padding="longest",
                              truncation="only_first",
                              max_length=30,
                              return_tensors="pt")

        with torch.inference_mode():
            outputs = model.greedy_generate(images,
                                            encodings["input_ids"].to(device if device >= 0 else "cpu"),
                                            attention_mask=encodings["attention_mask"].to(device if device >= 0 else "cpu"), 
                                            max_length=len(encodings["input_ids"][0]) + max_generation_length,
                                            eoc_token_id=tokenizer.encode("<|endofchunk|>")[0],
                                            pad_token_id=tokenizer.pad_token_id)

            predictions.extend([postprocess_generation(out) for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)])
    
    return {"vqa_accuracy": compute_vqa_accuracy(predictions, [row["answers"] for row in dataset])}
