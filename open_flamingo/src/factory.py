from transformers import AutoTokenizer, CLIPProcessor, CLIPModel

from .flamingo import Flamingo
from .flamingo_lm import OPTForCausalLMFlamingo


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_processor_path: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    use_local_files: bool = False,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model
        clip_processor_path (str): path to pretrained clip processor
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        use_local_files (bool, optional): whether to use local files. Defaults to False.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    vision_encoder = CLIPModel.from_pretrained(
        clip_vision_encoder_path, local_files_only=use_local_files
    )
    image_processor = CLIPProcessor.from_pretrained(
        clip_processor_path, local_files_only=use_local_files
    )

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=use_local_files
    )
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )

    lang_encoder = OPTForCausalLMFlamingo.from_pretrained(
        lang_encoder_path, local_files_only=use_local_files
    )
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        **flamingo_kwargs,
    )
    
    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
    
    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    model.perceiver.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    model.lang_encoder.get_input_embeddings().requires_grad_(True)
    model.lang_encoder.model.decoder.final_layer_norm.requires_grad_(True)

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer
