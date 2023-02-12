from typing import List, Optional, Tuple, Union

import torch.nn as nn

from .helpers import GatedCrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive


class FlamingoLayer(nn.Module):
    def __init__(self, gated_cross_attn_layer, decoder_layer):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        self.media_locations = media_locations

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):

        if self.vis_x is None:
            raise ValueError("vis_x must be conditioned before forward pass")

        if self.media_locations is None:
            raise ValueError("media_locations must be conditioned before forward pass")

        lang_x = self.gated_cross_attn_layer(
            lang_x, self.vis_x, media_locations=self.media_locations
        )
        lang_x = self.decoder_layer(
            lang_x,
            attention_mask=attention_mask,
            **decoder_layer_kwargs
        )
        return lang_x

class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """
    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(self, media_token_id, vis_hidden_size):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.

        Args:
            media_token_id (_type_): _description_
            vis_hidden_size (_type_): _description_
        """
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    dim=self.config.hidden_size, dim_visual=vis_hidden_size
                )
                for _ in self._get_decoder_layers()
            ]
        )
        self._set_decoder_layers(nn.ModuleList(
            [
                FlamingoLayer(gated_cross_attn_layer, decoder_layer)
                for gated_cross_attn_layer, decoder_layer in zip(
                    self.gated_cross_attn_layers, self._get_decoder_layers()
                )
            ]
        ))
        self.media_token_id = media_token_id
        self.initalized_flamingo = True
        self.register_forward_pre_hook(self._compute_media_locations)

    def _compute_media_locations(self, module, args):
        if not self.initalized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )
        # we always pass in 1 positional argument (input_ids)        
        input_ids = args[0]
        media_locations = input_ids == self.media_token_id
        for layer in self._get_decoder_layers():
            layer.condition_media_locations(media_locations)
            
    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
