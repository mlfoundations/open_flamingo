from torch import nn
from .helpers import LinearPatchProjection
from .vlm import VLMWithLanguageStream


class Llava(VLMWithLanguageStream):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_model: nn.Module,
        vis_feature_dim: int,
        initial_tokenizer_len: int,
        pad_token_id: int,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
    ):
        """
        Language stream VLM that uses a linear projection, similar to Llava.

        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            initial_tokenizer_len (int): size of the tokenizer vocab
            padding_token_id (int): id of the padding token. None if no padding token; then a padding token
                will be inserted into self.special_tokens, which factory.py fills after creating new tokens
            decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """
        self._special_tokens = {
            "media_token": "<image>",
        }
        lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]

        if vision_encoder.__class__.__name__ == "TimmModel":
            grid_size = vision_encoder.trunk.patch_embed.grid_size
        else:
            grid_size = vision_encoder.grid_size

        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=LinearPatchProjection(dim_visual=vis_feature_dim,
                                                    dim_out=lang_embedding_dim,
                                                    num_patches=grid_size[0] * grid_size[1]),
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            gradient_checkpointing=gradient_checkpointing,
            decoder_layers_attr_name=decoder_layers_attr_name,
            pad_token_id=pad_token_id,
        )

    def set_trainable(self):
        """
        Unfreeze everything except the vision_encoder
        """
        self.requires_grad_(True)
        self.vision_encoder.requires_grad_(False)

    def _should_apply_weight_decay(self, parameter_name):
        return True
