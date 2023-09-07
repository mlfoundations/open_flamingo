from torch import nn
from .helpers import PerceiverResampler
from .vlm import VLMWithCrossAttention


class Flamingo(VLMWithCrossAttention):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_model: nn.Module,
        vis_feature_dim: int,
        tokenizer_vocab_size: int,
        pad_token: str,
        cross_attn_every_n_layers: int = 1,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_model (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            tokenizer_vocab_size (int): size of the tokenizer vocab
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
            decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """
        self._special_tokens = {
            "eoc_token": "<|endofchunk|>",
            "media_token": "<image>",
            "pad_token": pad_token,
        }
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=PerceiverResampler(dim=vis_feature_dim),
            lang_model=lang_model,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_vocab_size=tokenizer_vocab_size,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            decoder_layers_attr_name=decoder_layers_attr_name,
        )

    def set_trainable(self):
        """
        Freeze everything except: perceiver, gated_cross_attn_layers, and inserted LM input embeddings
        """
        self.requires_grad_(False)
        self.vision_tokenizer.requires_grad_(True)
        self.lang_model.gated_cross_attn_layers.requires_grad_(True)
        self.lang_model.get_output_embeddings().set_requires_grad(
            require_regular_grad=False,
            require_additional_grad=True,
        )
        self.lang_model.get_input_embeddings().set_requires_grad(
            require_regular_grad=False,
            require_additional_grad=True,
        )
