import torch
from einops import rearrange
from torch import nn

from .flamingo_lm import OPTForCausalLMFlamingo
from .helpers import PerceiverResampler


class Flamingo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: OPTForCausalLMFlamingo,
        eoc_token_id: int,
        media_token_id: int,
        encoded_vis_dim: int = None,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIP Model 
                We assume vision_encoder has a vision_model whose forward() fn outputs 
                something with a .last_hidden_state attribute
            lang_encoder (OPTForCausalLMFlamingo): An instance of OPTForCausalLMFlamingo
            encoded_vis_dim (int, optional): In forward(), vision inputs are either passed
                in pre-encoded or encoded using visual encoder into a final shape (..., D)
                This parameter pre-specifies D, which needs to be the same across all datasets
                (e.g. both Pile and LAION) in order to use one set of cross-attention layers.
                Defaults to the vision_encoder's hidden_size.
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        
        self.vis_dim = encoded_vis_dim if encoded_vis_dim is not None else vision_encoder.config.vision_config.hidden_size

        self.vision_encoder = vision_encoder
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.lang_encoder = lang_encoder
        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            vis_hidden_size=self.vis_dim,
        )

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        is_vision_encoded: bool = False,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) or (B, T_img, n, D) depending on is_vision_encoded
                See _process_media() for more details
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            is_vision_encoded (bool, optional): Whether vision input has already been encoded outside
                of the Flamingo model. Defaults to False. This is useful when training on the Pile.
                See _process_media() for more details.
        """
        self._process_media(vision_x, is_vision_encoded=is_vision_encoded)

        output = self.lang_encoder(lang_x, attention_mask=attention_mask, labels=labels)

        self.lang_encoder.clear_conditioned_layers()
        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        num_beams=1,
        max_new_tokens=None,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        no_repeat_ngram_size=0,
        prefix_allowed_tokens_fn=None,
        length_penalty=1.0,
        num_return_sequences=1,
        do_sample=False,
        early_stopping=False,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            max_length (int, optional): Maximum length of the output. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            num_beams (int, optional): Number of beams. Defaults to 1.
            max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
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
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        self._process_media(vision_x)

        output = self.lang_encoder.generate(
            lang_x,
            attention_mask=attention_mask,
            eos_token_id=self.eoc_token_id,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            early_stopping=early_stopping,
        )

        self.lang_encoder.clear_conditioned_layers()
        return output

    def _process_media(self, vision_x: torch.Tensor, is_vision_encoded: bool):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                if not is_vision_encoded: shape (B, T_img, F, C, H, W)
                else: shape (B, T_img, n, D)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
            is_vision_encoded (bool, optional): Whether vision input has already been encoded outside
                of the Flamingo model. Defaults to False. This is useful when training on the Pile
                and passing in text projections as vision_x.
                vision_x will NOT be passed through the PerceiverResampler if is_vision_encoded.

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        # check shapes
        if is_vision_encoded:
            assert vision_x.ndim == 4, "vision_x should be of shape (b, T_img, n, D)"
        else:
            assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
            b, T, F = vision_x.shape[:3]
            assert F == 1, "Only single frame supported"

        # encode images
        if not is_vision_encoded:
            vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
            with torch.no_grad():
                vision_x = self.vision_encoder.vision_model(vision_x).last_hidden_state 
            vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
            vision_x = self.perceiver(vision_x) # reshapes to (b, T, n, d)

        assert vision_x.shape[-1] == self.vis_dim, f"Expected the last dim of vision_x to be {self.vis_dim} as passed to the constructor."

        for layer in self.lang_encoder.get_decoder().layers:
            layer.condition_vis_x(vision_x)
