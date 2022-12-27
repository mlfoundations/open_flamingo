'''
Main Flamingo class
Uses gated cross attention with Perceiver resampler
'''

import torch
from torch import nn

from .helpers import GatedCrossAttentionBlock, PerceiverResampler


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

    def forward(self,
                lang_x,
                attention_mask=None,
                **kwargs):

        if self.vis_x is None:
            raise ValueError("vis_x must be conditioned before forward pass")

        if self.media_locations is None:
            raise ValueError(
                "media_locations must be conditioned before forward pass")

        lang_x = self.gated_cross_attn_layer(
            lang_x, self.vis_x, media_locations=self.media_locations)
        lang_x = self.decoder_layer(lang_x,
                                    attention_mask=attention_mask,
                                    **kwargs)
        return lang_x


class Flamingo(nn.Module):
    def __init__(self, vision_encoder: nn.Module, lang_encoder: nn.Module, eoc_token_id: int, media_token_id: int):
        """
        Args:
            vision_encoder (nn.Module): Any vision encoder
            lang_encoder (nn.Module): Any auto-regressive language model
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id

        self.vision_encoder = vision_encoder
        self.lang_encoder = lang_encoder
        self.gated_cross_attn_layers = nn.ModuleList([GatedCrossAttentionBlock(
            dim=self.lang_encoder.config.hidden_size, dim_visual=self.vision_encoder.config.hidden_size) for _ in range(len(self.get_lm_decoder_layers()))])

        flamingo_layers = self._create_flamingo_layers()

        # replace language model decoder layers with Flamingo layers
        if "OPT" in self.lang_encoder.__class__.__name__:
            self.lang_encoder.get_decoder().layers = flamingo_layers
        elif "GPTNeoX" in self.lang_encoder.__class__.__name__:  # For Pythia models
            self.lang_encoder.gpt_neox.layers = flamingo_layers
        else:
            self.lang_encoder.transformer.h = flamingo_layers

        self.perceiver_resampler = PerceiverResampler(
            dim=vision_encoder.config.hidden_size, depth=6)

        # replace language model forward pass with Flamingo forward pass
        self.lang_encoder.original_forward = self.lang_encoder.forward
        self.lang_encoder.forward = self.lm_forward

    def _create_flamingo_layers(self):
        return nn.ModuleList([FlamingoLayer(gated_cross_attn_layer, decoder_layer)
                              for gated_cross_attn_layer, decoder_layer in zip(self.gated_cross_attn_layers, self.get_lm_decoder_layers())])

    def forward(self, vision_x: torch.Tensor, lang_x: torch.Tensor, attention_mask: torch.Tensor = None, labels: torch.Tensor = None):
        self._process_media(vision_x)

        output = self.lang_encoder.forward(
            lang_x, attention_mask=attention_mask, labels=labels)

        self.clear_conditioned_layers()
        return output

    def generate(self, vision_x: torch.Tensor, lang_x: torch.Tensor, max_length: int, attention_mask: torch.Tensor = None, num_beams=1, temperature=1.0, top_k=0, top_p=1.0, no_repeat_ngram_size=0, length_penalty=1.0, num_return_sequences=1, do_sample=False, early_stopping=False):
        """ 
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
            lang_x (torch.Tensor): Language input
            max_length (int): Maximum length of the output
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            num_beams (int, optional): Number of beams. Defaults to 1.
            temperature (float, optional): Temperature. Defaults to 1.0.
            top_k (int, optional): Top k. Defaults to 0.
            top_p (float, optional): Top p. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
            num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
            do_sample (bool, optional): Do sample. Defaults to False.
            early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        self._process_media(vision_x)

        output = self.lang_encoder.generate(
            lang_x,
            attention_mask=attention_mask,
            max_length=max_length,
            eos_token_id=self.eoc_token_id,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            early_stopping=early_stopping
        )

        self.clear_conditioned_layers()
        return output

    def clear_conditioned_layers(self):
        for layer in self.get_lm_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)

    def _process_media(self, vision_x):
        """
        Compute media tokens from vision input by passing it through vision encoder, resampling and conditioning language model.
        """
        vision_attended = self.vision_encoder(vision_x).last_hidden_state
        vision_attended = self.perceiver_resampler(vision_attended)
        for layer in self.get_lm_decoder_layers():
            layer.condition_vis_x(vision_attended)

    def lm_forward(
        self,
        input_ids,
        attention_mask=None,
        **kwargs
    ):

        # condition language model layers on media locations
        media_locations = input_ids == self.media_token_id
        for layer in self.get_lm_decoder_layers():
            layer.condition_media_locations(media_locations)

        return self.lang_encoder.original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def get_lm_decoder_layers(self):
        if "OPT" in self.lang_encoder.__class__.__name__:
            return self.lang_encoder.get_decoder().layers
        elif "GPTNeoX" in self.lang_encoder.__class__.__name__:  # For Pythia models
            return self.lang_encoder.gpt_neox.layers
        else:
            return self.lang_encoder.transformer.h

    def freeze_backbones(self):
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        for p in self.get_lm_decoder_layers().parameters():
            p.requires_grad = False

        for p in self.perceiver_resampler.parameters():
            p.requires_grad = True

        for p in self.gated_cross_attn_layers.parameters():
            p.requires_grad = True

        self.lang_encoder.get_input_embeddings().weight.requires_grad = True
