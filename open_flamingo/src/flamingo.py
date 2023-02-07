import torch
from einops import rearrange
from torch import nn

from .flamingo_lm import OPTForCausalLMFlamingo


class Flamingo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: OPTForCausalLMFlamingo,
        eoc_token_id: int,
        media_token_id: int,
    ):
        """
        Args:
            vision_encoder (nn.Module): Any vision encoder
            lang_encoder (OPTForCausalLMFlamingo): An instance of OPTForCausalLMFlamingo
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id

        self.vision_encoder = vision_encoder
        self.lang_encoder = lang_encoder
        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            vis_hidden_size=self.vision_encoder.config.projection_dim,
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
            lang_x (torch.Tensor): Language input
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            is_vision_encoded (bool, optional): Whether vision input is already encoded. Defaults to False. This is useful
            in training when passing text projections as 'vision' input for PILE input.
        """
        self._process_media(vision_x, is_vision_encoded)

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
            lang_x (torch.Tensor): Language input
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

    def _process_media(self, vision_x: torch.Tensor, is_vision_encoded: bool = False):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.

        Args:
            vision_x (torch.Tensor): Vision input
            is_vision_encoded (bool, optional): Whether vision input is already encoded. Defaults to False. This is useful
            in training when passing text projections as 'vision' input.
        """
        # rearrange code taken from https://github.com/dhansmair/flamingo-mini
        b, N, T = vision_x.shape[:3]

        assert T == 1, "Only single frame supported"
        if not is_vision_encoded:
            vision_x = rearrange(vision_x, "b N T c h w -> (b N T) c h w")
            with torch.no_grad():
                vision_x = self.vision_encoder.get_image_features(vision_x)
                vision_x = vision_x / vision_x.norm(p=2, dim=-1, keepdim=True)
                vision_x = vision_x.unsqueeze(1)
            vision_x = rearrange(vision_x, "(b N 1) v d -> b N v d", b=b, N=N)
        else:
            vision_x = rearrange(vision_x, "b N 1 v d -> b N v d", b=b, N=N)

        for layer in self.lang_encoder.get_decoder().layers:
            layer.condition_vis_x(vision_x)
