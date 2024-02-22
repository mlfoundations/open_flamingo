from typing import List, Optional, Tuple, Union
from torch import nn
import torch
from .helpers import PerceiverResampler
from .vlm import VLMWithCrossAttention

class Flamingo(VLMWithCrossAttention):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_model: nn.Module,
        vis_feature_dim: int,
        initial_tokenizer_len: int,
        pad_token_id: int,
        cross_attn_every_n_layers: int = 1,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_model (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            initial_tokenizer_len (int): size of the tokenizer vocab
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
            decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """
        self._special_tokens = {
            "eoc_token": "<|endofchunk|>",
            "media_token": "<image>",
        }
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=PerceiverResampler(dim=vis_feature_dim),
            lang_model=lang_model,
            gradient_checkpointing=gradient_checkpointing,
            initial_tokenizer_len=initial_tokenizer_len,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            decoder_layers_attr_name=decoder_layers_attr_name,
            pad_token_id=pad_token_id,
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

    def _should_apply_weight_decay(self, parameter_name):
        """
        Flamingo applies 0.1 weight decay to cross attention parameters
        """
        return "gated_cross_attn" in parameter_name

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return super().generate(vision_x, lang_x, attention_mask, past_key_values, past_media_locations, past_vision_tokens, eos_token_id=self.eoc_token_id, **kwargs)