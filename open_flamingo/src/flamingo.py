from torch import nn
from .helpers import PerceiverResampler
from .vlm import VLMWithCrossAttention


class Flamingo(VLMWithCrossAttention):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_model: nn.Module,
        vis_feature_dim: int,
        tokenizer_vocab_size: int,
        pad_token: str,
        cross_attn_every_n_layers: int = 1,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_model (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            tokenizer_vocab_size (int): size of the tokenizer vocab
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
            decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """
        self._special_tokens = {
            "eoc_token": "<|endofchunk|>",
            "media_token": "<image>",
            "pad_token": pad_token,
        }
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=PerceiverResampler(dim=vis_feature_dim),
            lang_model=lang_model,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_vocab_size=tokenizer_vocab_size,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            decoder_layers_attr_name=decoder_layers_attr_name,
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

    def wrap_fsdp(self, wrapper_kwargs, device_id):
        """
        Manually wraps submodules for FSDP and move other parameters to device_id.

        Why manually wrap?
        - all parameters within the FSDP wrapper must have the same requires_grad.
            We have a mix of frozen and unfrozen parameters.
        - model.vision_encoder.visual needs to be individually wrapped or encode_vision_x errors
            See: https://github.com/pytorch/pytorch/issues/82461#issuecomment-1269136344

        The rough wrapping structure is:
        - FlamingoModel
            - FSDP(FSDP(vision_encoder))
            - FSDP(FSDP(perceiver))
            - lang_model
                - FSDP(FSDP(input_embeddings))
                - FlamingoLayers
                    - FSDP(FSDP(gated_cross_attn_layer))
                    - FSDP(FSDP(decoder_layer))
                - FSDP(FSDP(output_embeddings))
                - other parameters

        Known issues:
        - Our FSDP strategy is not compatible with tied embeddings. If the LM embeddings are tied,
            train with DDP or set the --freeze_lm_embeddings flag to true.
        - With FSDP + gradient ckpting, one can increase the batch size with seemingly no upper bound.
            Although the training curves look okay, we found that downstream performance dramatically
            degrades if the batch size is unreasonably large (e.g., 100 MMC4 batch size for OPT-125M).

        FAQs about our FSDP wrapping strategy:
        Why double wrap?
        As of torch==2.0.1, FSDP's _post_forward_hook and _post_backward_hook
        only free gathered parameters if the module is NOT FSDP root.

        Why unfreeze the decoder_layers?
        See https://github.com/pytorch/pytorch/issues/95805
        As of torch==2.0.1, FSDP's _post_backward_hook is only registed if the flat param
        requires_grad=True. We need the postback to fire to avoid OOM.
        To effectively freeze the decoder layers, we exclude them from the optimizer.

        What is assumed to be frozen v. unfrozen?
        We assume that the model is being trained under normal Flamingo settings
        with these lines being called in factory.py:
            ```
            # Freeze all parameters
            model.requires_grad_(False)
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

            # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
            model.perceiver.requires_grad_(True)
            model.lang_model.gated_cross_attn_layers.requires_grad_(True)
            [optional] model.lang_model.get_input_embeddings().requires_grad_(True)
            ```
        """
        print(
            "WARNING: FSDP is not designed for training with a mix of frozen and unfrozen parameters. "
            + "This experimental workaround results in a significant drop in GPU power usage."
        )

        from torch.distributed.fsdp.wrap import (
            enable_wrap,
            wrap,
        )
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )
        from .utils import apply_with_stopping_condition

        # unfreeze the decoder layers
        for block in self.lang_model.old_decoder_blocks:
            block.requires_grad_(True)

        # wrap in FSDP
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            self.perceiver = wrap(wrap(self.perceiver))
            self.lang_model.old_decoder_blocks = nn.ModuleList(
                wrap(wrap(block)) for block in self.lang_model.old_decoder_blocks
            )
            self.lang_model.gated_cross_attn_layers = nn.ModuleList(
                wrap(wrap(layer)) if layer is not None else None
                for layer in self.lang_model.gated_cross_attn_layers
            )
            self.lang_model.init_flamingo_layers(self._use_gradient_checkpointing)
            if hasattr(self.lang_model.get_input_embeddings(), "additional_embedding"):
                # wrap additional_embedding and original embedding separately, this is the case when using decoupled embeddings
                self.lang_model.get_input_embeddings().additional_embedding = wrap(
                    wrap(self.lang_model.get_input_embeddings().additional_embedding)
                )
                self.lang_model.get_input_embeddings().weight = wrap(
                    wrap(self.lang_model.get_input_embeddings().weight)
                )
            else:
                self.lang_model.set_input_embeddings(
                    wrap(wrap(self.lang_model.get_input_embeddings()))
                )

            if hasattr(self.lang_model.get_output_embeddings(), "additional_fc"):
                # wrap additional_fc and original embedding separately, this is the case when using decoupled linear layer
                self.lang_model.get_output_embeddings().additional_fc = wrap(
                    wrap(self.lang_model.get_output_embeddings().additional_fc)
                )
                self.lang_model.get_output_embeddings().weight = wrap(
                    wrap(self.lang_model.get_output_embeddings().weight)
                )
                if self.lang_model.get_output_embeddings().bias is not None:
                    self.lang_model.get_output_embeddings().bias = wrap(
                        wrap(self.lang_model.get_output_embeddings().bias)
                    )
            else:
                self.lang_model.set_output_embeddings(
                    wrap(wrap(self.lang_model.get_output_embeddings()))
                )

            self.vision_encoder = wrap(wrap(self.vision_encoder))  # frozen

        # manually move non-FSDP managed parameters to device_id
        # these are all in lang_model
        apply_with_stopping_condition(
            module=self.lang_model,
            apply_fn=lambda m: m.to(device_id),
            apply_condition=lambda m: len(list(m.children())) == 0,
            stopping_condition=lambda m: isinstance(m, FSDP),
        )

        # exclude the original decoder layers from the optimizer
        for block in self.lang_model.old_decoder_blocks:
            for p in block.parameters():
                p.exclude_from_optimizer = True

        # set up clip_grad_norm_ function
        def clip_grad_norm_(max_norm):
            self.perceiver.clip_grad_norm_(max_norm)
            for layer in self.lang_model.gated_cross_attn_layers:
                if layer is not None:
                    layer.clip_grad_norm_(max_norm)
            self.lang_model.get_input_embeddings().clip_grad_norm_(max_norm)

        self.clip_grad_norm_ = clip_grad_norm_
