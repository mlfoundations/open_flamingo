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
        vis_dim: int = None,
        use_projection_vector: bool = False,
        use_media_placement_augmentation: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (OPTForCausalLMFlamingo): An instance of OPTForCausalLMFlamingo
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int, optional): Dimension of the visual features. Defaults to CLIP's vision_encoder's hidden size.
                Visual features are projected to match this shape along the last dimension.
            use_projection_vector (bool, optional): Whether to use the CLIP projection output for the visual features.
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.use_projection_vector = use_projection_vector
        self.use_media_placement_augmentation = use_media_placement_augmentation

        self.vis_dim = (
            vis_dim
            if vis_dim is not None
            else vision_encoder.config.vision_config.hidden_size
        )

        self.vision_encoder = vision_encoder
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.lang_encoder = lang_encoder
        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            vis_hidden_size=self.vis_dim,
            use_media_placement_augmentation=self.use_media_placement_augmentation
        )

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        pseudovision_x: torch.Tensor = None,
        pseudovision_attention_mask: torch.Tensor = None,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            pseudovision_x (torch.Tensor, optional): Input ids for text to be used as pseudoimages.
                When training on the Pile, we use text as "pseudoimages" by encoding with the CLIP text encoder.
                shape (B, T_img, m) where m is the sequence length
            pseudovision_attention_mask (torch.Tensor, optional): Attention mask for pseudovision_x.
        """
        assert (vision_x is not None) ^ (
            pseudovision_x is not None
        ), "Must provide either vision_x or pseudovision_x"
        vis_x = self._process_media(
            vision_x=vision_x,
            pseudovision_x=pseudovision_x,
            pseudovision_attention_mask=pseudovision_attention_mask,
        )

        output = self.lang_encoder(
            lang_x, attention_mask=attention_mask, labels=labels, vis_x = vis_x)

        # self.lang_encoder.clear_conditioned_layers()
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

        self._process_media(vision_x=vision_x)

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

    def _process_media(
        self,
        vision_x: torch.Tensor = None,
        pseudovision_x: torch.Tensor = None,
        pseudovision_attention_mask: torch.Tensor = None,
    ):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
            pseudovision_x (torch.Tensor, optional): Input ids for text to be used as pseudoimages.
                shape (B, T_img, m) where m is the sequence length
            pseudovision_attention_mask (torch.Tensor, optional): Attention mask for pseudovision_x.
        """
        assert (vision_x is None) ^ (
            pseudovision_x is None), "Must provide either vision_x or pseudovision_x"

        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x)
        elif pseudovision_x is not None:
            vision_features = self._encode_pseudovision_x(
                pseudovision_x, pseudovision_attention_mask
            )

        return vision_features

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Encode real vision inputs
        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            if self.use_projection_vector:
                vision_x = self.vision_encoder.get_image_features(vision_x)
                vision_x = vision_x / vision_x.norm(p=2, dim=-1, keepdim=True)
                # add a dimension v to match perceiver input
                vision_x = vision_x.unsqueeze(-2)
            else:
                vision_x = self.vision_encoder.vision_model(
                    vision_x).last_hidden_state
        vision_x = rearrange(
            vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        
        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)
            
        return vision_x

    def _encode_pseudovision_x(
        self, pseudovision_x: torch.Tensor, pseudovision_attention_mask: torch.Tensor
    ):
        """
        Encode text inputs as pseudoimages
        """
        assert (
            pseudovision_x.ndim == 3
        ), "pseudovision_x should be of shape (b, T_img, m)"
        b, T = pseudovision_x.shape[:2]

        pseudovision_x = rearrange(pseudovision_x, "b T m -> (b T) m")
        pseudovision_attention_mask = rearrange(
            pseudovision_attention_mask, "b T m -> (b T) m"
        )
        with torch.no_grad():
            pseudovision_x = self.vision_encoder.get_text_features(
                input_ids=pseudovision_x, attention_mask=pseudovision_attention_mask
            )
            pseudovision_x = pseudovision_x / pseudovision_x.norm(
                p=2, dim=-1, keepdim=True
            )

        pseudovision_x = rearrange(
            pseudovision_x, "(b T) d -> b T 1 1 d", b=b, T=T)
        
        pseudovision_x = self.perceiver(pseudovision_x)
        return pseudovision_x
