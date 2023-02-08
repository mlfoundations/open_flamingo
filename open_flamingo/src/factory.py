from transformers import AutoTokenizer, CLIPProcessor, CLIPVisionModel

from .flamingo import Flamingo
from .flamingo_lm import OPTForCausalLMFlamingo


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_processor_path: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    use_local_files: bool = False,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip vision encoder
        clip_processor_path (str): path to pretrained clip processor
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        use_local_files (bool, optional): whether to use local files. Defaults to False.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    vision_encoder = CLIPVisionModel.from_pretrained(
        clip_vision_encoder_path, local_files_only=use_local_files
    )
    image_processor = CLIPProcessor.from_pretrained(
        clip_processor_path, local_files_only=use_local_files
    )

    for p in vision_encoder.parameters():
        p.requires_grad = False

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
    )

    for p in lang_encoder.get_decoder().layers.parameters():
        p.requires_grad = False

    for p in lang_encoder.gated_cross_attn_layers.parameters():
        p.requires_grad = True

    lang_encoder.get_input_embeddings().weight.requires_grad = True

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer
