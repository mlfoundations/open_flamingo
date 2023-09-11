from torch import nn
from .helpers import PerceiverResampler
from .vlm import VLMWithLanguageStream


class Kosmos(VLMWithLanguageStream):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_model: nn.Module,
        vis_feature_dim: int,
        tokenizer_vocab_size: int,
        pad_token: str,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            tokenizer_vocab_size (int): size of the tokenizer vocab
            padding_token_id (int): id of the padding token. None if no padding token; then a padding token
                will be inserted into self.special_tokens, which factory.py fills after creating new tokens
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """
        self._special_tokens = {
            "media_token": "<image>",
            "pad_token": pad_token,
        }
        lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=PerceiverResampler(
                dim=vis_feature_dim, dim_inner=lang_embedding_dim
            ),
            lang_model=lang_model,
            tokenizer_vocab_size=tokenizer_vocab_size,
            gradient_checkpointing=gradient_checkpointing,
        )

    def set_trainable(self):
        """
        Unfreeze everything except the vision_encoder
        """
        self.requires_grad_(True)
        self.vision_encoder.requires_grad_(False)

    def wrap_fsdp(self, wrapper_kwargs, device_id):
        raise NotImplementedError