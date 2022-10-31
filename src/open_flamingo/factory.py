import logging

import torch
from transformers import AutoTokenizer, CLIPProcessor, CLIPVisionModel

from .flamingo import Flamingo
from .flamingo_lm import OPTForCausalLMFlamingo


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    lang_encoder_path: str,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder. 
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip vision encoder
        lang_encoder_path (str): path to pretrained language encoder

    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    logging.info("Initializing Flamingo model...")

    vision_encoder, image_processor = get_clip_vision_encoder(
        clip_vision_encoder_path)
    vision_encoder.requires_grad_(False)

    text_tokenizer = AutoTokenizer.from_pretrained(lang_encoder_path)
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens({
        'additional_special_tokens': ['<|endofchunk|>']
    })

    lang_encoder = OPTForCausalLMFlamingo.from_pretrained(lang_encoder_path)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    for layer in lang_encoder.get_decoder().layers:
        layer.requires_grad_(False)
    lang_encoder.perceiver_resampler.requires_grad_(True)
    for layer in lang_encoder.gated_cross_attn:
        layer.requires_grad_(True)
    lang_encoder.get_decoder().embed_tokens.requires_grad_(True)

    model = Flamingo(vision_encoder, lang_encoder)

    logging.info(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    return model, image_processor, text_tokenizer


def get_clip_vision_encoder(path):
    return CLIPVisionModel.from_pretrained(path), CLIPProcessor.from_pretrained(path)
