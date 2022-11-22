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

    for p in vision_encoder.parameters():
        p.requires_grad = False
    
    text_tokenizer = AutoTokenizer.from_pretrained('facebook/opt-30b')
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens({
        'additional_special_tokens': ['<|endofchunk|>', '<image>']
    })

    lang_encoder = OPTForCausalLMFlamingo.from_pretrained(lang_encoder_path).to("cpu")
    lang_encoder.resize_token_embeddings(len(text_tokenizer))
    
    model = Flamingo(vision_encoder, lang_encoder, text_tokenizer.encode("<|endofchunk|>")[-1], text_tokenizer.encode("<image>")[-1])

    for p in lang_encoder.get_decoder().layers.parameters():
        p.requires_grad = False
        
    for p in model.perceiver_resampler.parameters():
        p.requires_grad = True

    for p in lang_encoder.gated_cross_attn_layers.parameters():
        p.requires_grad = True
        
    lang_encoder.get_input_embeddings().weight.requires_grad = True

    print(f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    return model, image_processor, text_tokenizer


def get_clip_vision_encoder(path):
    return CLIPVisionModel.from_pretrained(path).to("cpu"), CLIPProcessor.from_pretrained(path)
