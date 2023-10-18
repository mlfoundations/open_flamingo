from typing import Optional
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip

from .flamingo import Flamingo
from .kosmos import Kosmos
from .blip import BLIP
from .utils import hasattr_recursive, setattr_recursive

SUPPORTED_MODEL_FAMILIES = ("flamingo", "kosmos", "blip")


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_model_path: str,
    tokenizer_path: str,
    model_family: str = "flamingo",
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    cache_dir: Optional[str] = None,
    gradient_checkpointing: bool = False,
    verbose: bool = True,
    **model_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_model_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
        cache_dir (str, optional): path to cache directory for downloading OpenClip/HF weights.
        gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        verbose (bool, optional): whether to print model info. Defaults to True.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """

    assert model_family in SUPPORTED_MODEL_FAMILIES

    # load vision encoder
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,
        pretrained=clip_vision_encoder_pretrained,
        cache_dir=cache_dir,
    )
    vision_encoder.visual.output_tokens = True
    vision_encoder = vision_encoder.visual
    vis_hidden_dim = open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
        "width"
    ]

    # load tokenizer and ensure there is a pad token
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token_id = text_tokenizer.eos_token_id

    # load langauge model
    lang_model = AutoModelForCausalLM.from_pretrained(
        lang_model_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    check_embedding_fns(lang_model)

    # init the model
    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_model)

    if model_family == "flamingo":
        model = Flamingo(
            vision_encoder=vision_encoder,
            lang_model=lang_model,
            vis_feature_dim=vis_hidden_dim,
            initial_tokenizer_len=len(text_tokenizer),
            gradient_checkpointing=gradient_checkpointing,
            decoder_layers_attr_name=decoder_layers_attr_name,
            pad_token_id=text_tokenizer.pad_token_id,
            **model_kwargs,
        )

    elif model_family == "kosmos":
        model = Kosmos(
            vision_encoder=vision_encoder,
            lang_model=lang_model,
            vis_feature_dim=vis_hidden_dim,
            initial_tokenizer_len=len(text_tokenizer),
            gradient_checkpointing=gradient_checkpointing,
            pad_token_id=text_tokenizer.pad_token_id,
            decoder_layers_attr_name=decoder_layers_attr_name,
            **model_kwargs,
        )

    elif model_family == "blip":
        model = BLIP(
            vision_encoder=vision_encoder,
            lang_model=lang_model,
            vis_feature_dim=vis_hidden_dim,
            initial_tokenizer_len=len(text_tokenizer),
            gradient_checkpointing=gradient_checkpointing,
            pad_token_id=text_tokenizer.pad_token_id,
            decoder_layers_attr_name=decoder_layers_attr_name,
            **model_kwargs,
        )

    # add special tokens to the tokenizer and language models
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": list(model.special_tokens.values())}
    )
    model.lang_model.config.vocab_size = len(text_tokenizer)
    model.set_special_token_ids(
        {
            v: text_tokenizer.convert_tokens_to_ids(v)
            for v in model.special_tokens.values()
        }
    )

    # freeze appropriate parameters
    model.set_trainable()

    # log model info
    if verbose:
        print(
            f"{model_family} model initialized with {model.num_trainable_params:,} trainable parameters"
        )
        print(f"========== Trainable Parameters\n{model.num_trainable_params_per_module}")
        print(f"========== Total Parameters\n{model.num_params_per_module}\n==========")
    return model, image_processor, text_tokenizer


def _infer_decoder_layers_attr_name(model):
    """
    Infer the name of the attribute storing the decoder layers (as a ModuleList) in the model.
    """
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
}


def check_embedding_fns(lang_model):
    """Checks for and attempts to set {get/set}_{input/output}_embeddings functions to the model"""
    if not has_fn(lang_model, "get_input_embeddings"):
        if hasattr_recursive(lang_model, "transformer.wte"):  # MPT
            lang_model.get_input_embeddings = lambda: lang_model.transformer.wte
        elif hasattr_recursive(lang_model, "model.decoder.embed_tokens"):  # OPT
            lang_model.get_input_embeddings = lambda: lang_model.decoder.embed_tokens
        else:
            raise ValueError(
                "We require the language encoder to have a get_input_embeddings method but we couldn't determine the name of the input embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "set_input_embeddings"):
        if hasattr_recursive(lang_model, "transformer.wte"):  # MPT
            lang_model.set_input_embeddings = lambda x: setattr_recursive(
                lang_model, "transformer.wte", x
            )
        elif hasattr_recursive(lang_model, "model.decoder.embed_tokens"):  # OPT
            lang_model.set_input_embeddings = lambda x: setattr_recursive(
                lang_model, "model.decoder.embed_tokens", x
            )
        else:
            raise ValueError(
                "We require the language encoder to have a set_input_embeddings method but we couldn't determine the name of the input embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "get_output_embeddings"):
        if hasattr_recursive(lang_model, "lm_head"):
            lang_model.get_output_embeddings = lambda: lang_model.lm_head
        else:
            raise ValueError(
                "We require the language encoder to have a get_output_embeddings method but we couldn't determine the name of the output embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "set_output_embeddings"):
        if hasattr_recursive(lang_model, "lm_head"):
            lang_model.set_output_embeddings = lambda x: setattr_recursive(
                lang_model, "lm_head", x
            )
        else:
            raise ValueError(
                "We require the language encoder to have a set_output_embeddings method but we couldn't determine the name of the output embeddings attribute. Please supply this manually in factory.py."
            )


def has_fn(model, fn_name):
    """Try to call the fn_name function on the model"""
    try:
        getattr(model, fn_name)()
        return True
    except:
        return False
