from torch import nn
from .helpers import QFormerWithProjection
from .vlm import VLMWithLanguageStream


class BLIP(VLMWithLanguageStream):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_model: nn.Module,
        vis_feature_dim: int,
        initial_tokenizer_len: int,
        pad_token_id: int,
        gradient_checkpointing: bool = False,
    ):
        """
        Language stream VLM that uses a Q-former, similar to BLIP.
        Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py

        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            initial_tokenizer_len (int): size of the tokenizer vocab
            padding_token_id (int): id of the padding token. None if no padding token; then a padding token
                will be inserted into self.special_tokens, which factory.py fills after creating new tokens
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """
        self._special_tokens = {
            "media_token": "<image>",
        }
        lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=QFormerWithProjection(
                dim_input=vis_feature_dim, dim_out=lang_embedding_dim
            ),
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            gradient_checkpointing=gradient_checkpointing,
            pad_token_id=pad_token_id,
        )

    def set_trainable(self):
        """
        Freeze everything except the Q-former and the inserted LM embeddings
        """
        self.requires_grad_(False)
        self.vision_tokenizer.requires_grad_(True)
        self.lang_model.get_output_embeddings().set_requires_grad(
            require_regular_grad=False,
            require_additional_grad=True,
        )
        self.lang_model.get_input_embeddings().set_requires_grad(
            require_regular_grad=False,
            require_additional_grad=True,
        )

    def _should_apply_weight_decay(self, parameter_name):
        """BLIP applies 0.05 weight decay to everything"""
        return True
