from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip

from .flamingo import Flamingo
from .kosmos import Kosmos
from .utils import extend_instance


def create_model_and_transforms(
    model_family: str,
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_model_path: str,
    tokenizer_path: str,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    cache_dir: Optional[str] = None,
    gradient_checkpointing: bool = False,
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
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """

    assert model_family in ("flamingo", "kosmos")

    # load vision encoder
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,
        pretrained=clip_vision_encoder_pretrained,
        cache_dir=cache_dir,
    )
    vision_encoder.visual.output_tokens = True
    vision_encoder = vision_encoder.visual

    # load tokenizer and ensure there is a pad token
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # load langauge model
    lang_model = AutoModelForCausalLM.from_pretrained(
        lang_model_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    ## hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_model_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings

        extend_instance(lang_model, EmbeddingFnMixin)

    # init the model
    if model_family == "flamingo":
        if decoder_layers_attr_name is None:
            decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_model)

        model = Flamingo(
            vision_encoder=vision_encoder,
            lang_model=lang_model,
            vis_feature_dim=open_clip.get_model_config(clip_vision_encoder_path)[
                "vision_cfg"
            ]["width"],
            tokenizer_vocab_size=len(text_tokenizer),
            gradient_checkpointing=gradient_checkpointing,
            decoder_layers_attr_name=decoder_layers_attr_name,
            pad_token=text_tokenizer.pad_token,
            **model_kwargs,
        )

    elif model_family == "kosmos":
        model = Kosmos(
            vision_encoder=vision_encoder,
            lang_model=lang_model,
            vis_feature_dim=open_clip.get_model_config(clip_vision_encoder_path)[
                "vision_cfg"
            ]["width"],
            tokenizer_vocab_size=len(text_tokenizer),
            gradient_checkpointing=gradient_checkpointing,
            pad_token=text_tokenizer.pad_token,
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

    # freeze appropraite parameters
    model.set_trainable()
    print(
        f"{model_family} model initialized with {model.num_trainable_params:,} trainable parameters"
    )
    return model, image_processor, text_tokenizer


def _infer_decoder_layers_attr_name(model):
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
