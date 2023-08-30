import torch
from einops import rearrange
from torch import nn
from typing import Optional
from torch.nn import CrossEntropyLoss
from .helpers import PerceiverResampler
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from .utils import apply_with_stopping_condition

def torch_stack_with_padding(list_of_tensors, padding_value=0, padding_side="right"):
    max_tokens = max(tensor.size(0) for tensor in list_of_tensors)
    padded_tensors = []
    for tensor in list_of_tensors:
        num_tokens = tensor.size(0)
        if len(tensor.size()) == 1:
            padding = torch.full(
                (max_tokens - num_tokens,),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            padding = torch.full(
                (max_tokens - num_tokens, tensor.size(1)),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        padded_tensor = (
            torch.cat((tensor, padding), dim=0)
            if padding_side == "right"
            else torch.cat((padding, tensor), dim=0)
        )
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors)


class MLLM(nn.Module):
    def __init__(
        self, language_model, vision_model, vis_dim, media_token_id, padding_token_id
    ):
        super().__init__()
        self.language_model = language_model
        self.vision_model = vision_model.visual
        self.vis_dim = vis_dim
        self.media_token_id = media_token_id
        self.padding_token_id = padding_token_id
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.language_projection = nn.Linear(
            self.vis_dim, self.language_model.config.hidden_size
        )

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_model(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.perceiver(vision_x)
        language_model_inputs = self.language_projection(vision_x)

        return language_model_inputs

    def forward(
        self,
        vision_x: torch.FloatTensor,
        lang_x: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else False
        )

        vision_x = self._encode_vision_x(vision_x)

        lang_embeds = self.language_model.get_input_embeddings()(lang_x)

        if attention_mask is None:
            attention_mask = torch.ones_like(lang_x)

        labels = lang_x if labels is None else labels

        if vision_x is not None:
            multimodal_embeds = []
            multimodal_labels = []
            multimodal_attention_mask = []

            for i in range(lang_embeds.shape[0]):
                # get index of <image> tokens in lang_x[i]
                image_token_idxs = torch.where(lang_x[i] == self.media_token_id)[0]
                # since an image is represented by 64 tokens, we need to offset the image_token_idxs by 64 except for the first image
                for j, img_idx in enumerate(image_token_idxs):
                    image_token_idxs[j] += 63 * j

                new_embed = lang_embeds[i].clone()
                new_attention_mask = (
                    attention_mask[i].clone() if attention_mask is not None else None
                )
                new_label = labels[i].clone()

                for img_num, img_idx in enumerate(image_token_idxs):
                    new_embed = torch.cat(
                        (
                            new_embed[:img_idx],
                            vision_x[i][img_num],
                            new_embed[img_idx + 1 :],
                        ),
                        dim=0,
                    )

                    new_attention_mask = torch.cat(
                        (
                            new_attention_mask[:img_idx],
                            torch.ones(64, dtype=torch.long).to(attention_mask.device),
                            new_attention_mask[img_idx + 1 :],
                        ),
                        dim=0,
                    )

                    new_label = torch.cat(
                        (
                            new_label[:img_idx],
                            torch.ones(64, dtype=torch.long).to(labels.device) * -100,
                            new_label[img_idx + 1 :],
                        ),
                        dim=0,
                    )

                multimodal_embeds.append(new_embed)
                multimodal_attention_mask.append(new_attention_mask)
                multimodal_labels.append(new_label)

            multimodal_embeds = torch_stack_with_padding(
                multimodal_embeds, padding_value=self.padding_token_id
            )
            multimodal_attention_mask = torch_stack_with_padding(
                multimodal_attention_mask, padding_value=0
            )
            multimodal_labels = torch_stack_with_padding(
                multimodal_labels, padding_value=-100
            )
        else:
            multimodal_embeds = lang_embeds
            multimodal_attention_mask = attention_mask
            multimodal_labels = labels

        outputs = self.language_model(
            inputs_embeds=multimodal_embeds,
            attention_mask=multimodal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if multimodal_labels is not None:
            multimodal_labels = multimodal_labels.to(logits.device)
            logits = logits[:, -multimodal_labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = multimodal_labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")

            loss = loss_fct(
                shift_logits.view(-1, self.language_model.config.vocab_size),
                shift_labels.view(-1),
            )

        return (loss, logits) if loss is not None else (logits,)

    @torch.no_grad()
    def generate(
        self,
        vision_x: torch.FloatTensor,
        lang_x: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        batch_size = vision_x.shape[0]

        if attention_mask is None:
            attention_mask = torch.ones(lang_x.shape, dtype=torch.long).to(
                lang_x.device
            )

        vision_x = self._encode_vision_x(vision_x)

        lang_embeds = self.language_model.get_input_embeddings()(lang_x)

        if vision_x is not None:
            multimodal_embeds = []
            multimodal_attention_mask = []

            for i in range(lang_embeds.shape[0]):
                # get index of <image> tokens in lang_x[i]
                image_token_idxs = torch.where(lang_x[i] == self.media_token_id)[0]
                # since an image is represented by 64 tokens, we need to offset the image_token_idxs by 64 except for the first image
                for j, img_idx in enumerate(image_token_idxs):
                    image_token_idxs[j] += 63 * j

                new_embed = lang_embeds[i].clone()
                new_attention_mask = (
                    attention_mask[i].clone() if attention_mask is not None else None
                )

                for img_num, img_idx in enumerate(image_token_idxs):
                    new_embed = torch.cat(
                        (
                            new_embed[:img_idx],
                            vision_x[i][img_num],
                            new_embed[img_idx + 1 :],
                        ),
                        dim=0,
                    )

                    new_attention_mask = torch.cat(
                        (
                            new_attention_mask[:img_idx],
                            torch.ones(64, dtype=torch.long).to(attention_mask.device),
                            new_attention_mask[img_idx + 1 :],
                        ),
                        dim=0,
                    )

                multimodal_embeds.append(new_embed)
                multimodal_attention_mask.append(new_attention_mask)

            multimodal_embeds = torch_stack_with_padding(
                multimodal_embeds,
                padding_value=self.padding_token_id,
                padding_side="left",
            )
            multimodal_attention_mask = torch_stack_with_padding(
                multimodal_attention_mask, padding_value=0, padding_side="left"
            )
        else:
            multimodal_embeds = lang_embeds
            multimodal_attention_mask = attention_mask

        outputs = self.language_model.generate(
            input_ids=None,
            inputs_embeds=multimodal_embeds,
            attention_mask=multimodal_attention_mask,
            **generate_kwargs,
        )

        return outputs

    def wrap_fsdp(self, wrapper_kwargs, device_id, lm_trainable=False):
        """
        Manually wraps submodules for FSDP and move other parameters to device_id.

        Why manually wrap?
        - all parameters within the FSDP wrapper must have the same requires_grad.
            We have a mix of frozen and unfrozen parameters.
        - model.vision_model.visual needs to be individually wrapped or encode_vision_x errors
            See: https://github.com/pytorch/pytorch/issues/82461#issuecomment-1269136344

        The rough wrapping structure is:
        - FlamingoModel
            - FSDP(FSDP(vision_model))
            - FSDP(FSDP(perceiver))
            - language_model
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
            model.language_model.gated_cross_attn_layers.requires_grad_(True)
            [optional] model.language_model.get_input_embeddings().requires_grad_(True)
            ```
        """
        # unfreeze the decoder layers
        for p in self.language_model.parameters():
            p.requires_grad_(True)

        # wrap in FSDP
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            self.language_projection = wrap(wrap(self.language_projection))
            self.perceiver = wrap(wrap(self.perceiver))
            
            self.language_model.model.layers = nn.ModuleList(
                wrap(wrap(block)) for block in self.language_model.model.layers
            )
            self.language_model.set_input_embeddings(
                wrap(wrap(self.language_model.get_input_embeddings()))
            )
            self.language_model.set_output_embeddings(
                wrap(wrap(self.language_model.get_output_embeddings()))
            )
            self.vision_model = wrap(wrap(self.vision_model))  # frozen

        # manually move non-FSDP managed parameters to device_id
        # these are all in language_model
        apply_with_stopping_condition(
            module=self.language_model,
            apply_fn=lambda m: m.to(device_id),
            apply_condition=lambda m: len(list(m.children())) == 0,
            stopping_condition=lambda m: isinstance(m, FSDP),
        )

        if not lm_trainable:
            # exclude the original decoder layers from the optimizer
            for p in self.language_model.parameters():
                p.exclude_from_optimizer = True

        # set up clip_grad_norm_ function
        def clip_grad_norm_(max_norm):
            self.perceiver.clip_grad_norm_(max_norm)
            self.language_projection.clip_grad_norm_(max_norm)
            # TODO: clip the decoder layers if they are unfrozen
            if lm_trainable:
                self.language_model.parameters().clip_grad_norm_(max_norm)

        self.clip_grad_norm_ = clip_grad_norm_
