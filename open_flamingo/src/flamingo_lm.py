import random

import torch.nn as nn
from torch.utils.checkpoint import checkpoint


from .helpers import GatedCrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive


class FlamingoLayer(nn.Module):
    def __init__(self, gated_cross_attn_layer, decoder_layer, grad_checkpointing):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None
        self._grad_checkpointing = grad_checkpointing

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None and self.media_locations is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        self.media_locations = media_locations

    def condition_use_cached_media(self, use_cached_media):
        self.use_cached_media = use_cached_media

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Cross attention
        if self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")

            if self.media_locations is None:
                raise ValueError("media_locations must be conditioned before forward pass")

            if self._grad_checkpointing:
                lang_x = checkpoint(
                    self.gated_cross_attn_layer,
                    lang_x,
                    self.vis_x,
                    use_reentrant=False,
                    media_locations=self.media_locations,
                    use_cached_media=self.use_cached_media,
                )
            else:
                lang_x = self.gated_cross_attn_layer(
                    lang_x,
                    self.vis_x,
                    media_locations=self.media_locations,
                    use_cached_media=self.use_cached_media,
                )

        # Normal decoder layer
        if self._grad_checkpointing:
            lang_x = checkpoint(
                self.decoder_layer,
                lang_x, 
                use_reentrant=False,
                attention_mask=attention_mask, **decoder_layer_kwargs
            )
        else:
            lang_x = self.decoder_layer(
                lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
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

    def init_flamingo(
        self,
        media_token_id,
        vis_hidden_size,
        cross_attn_every_n_layers,
        grad_checkpointing,
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        self.old_decoder_blocks = self._get_decoder_layers()
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    dim=self.config.hidden_size, dim_visual=vis_hidden_size
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self.init_flamingo_layers(grad_checkpointing)
        self.media_token_id = media_token_id
        self.initialized_flamingo = True
        self._generating = False

    def init_flamingo_layers(self, grad_checkpointing):
        """
        Re initializes the FlamingoLayers. 
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    FlamingoLayer(gated_cross_attn_layer, decoder_layer, grad_checkpointing)
                    for gated_cross_attn_layer, decoder_layer in zip(
                        self.gated_cross_attn_layers, self.old_decoder_blocks
                    )
                ]
            )
        )

    def forward(self, *input, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else input[0]
        media_locations = input_ids == self.media_token_id
        
        # if there are media already cached and we're generating and there are no media tokens in the input, 
        # we'll assume that ALL input tokens should attend to the last previous media that is cached. 
        # this is especially important for HF generate() compatibility,
        # which calls forward() repeatedly one token at a time (with no media tokens)
        # TODO: refactor this out of flamingo_lm and into flamingo? so that it mirrors vis_x
        use_cached_media_locations = self._generating and self.is_conditioned() and not media_locations.any()

        for layer in self.get_decoder().layers:
            if not use_cached_media_locations: layer.condition_media_locations(media_locations)
            layer.condition_use_cached_media(use_cached_media_locations)

        return super().forward(
            *input, **kwargs
        )  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_use_cached_media(None)
