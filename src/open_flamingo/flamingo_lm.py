import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import OPTForCausalLM

from .helpers import GatedCrossAttentionBlock, PerceiverResampler

# Had to redefine the OPT model as it will be more difficult to iterate layer by layer (which should just be defined in the forward pass)
# Forward pass code modified from https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/models/opt/modeling_opt.py


class FlamingoLayer(nn.Module):
    def __init__(self, perciever_layer, gated_cross_attn_layer, decoder_layer):
        super().__init__()
        self.perciever_layer = perciever_layer
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer

    def forward(self,
                vis_x,
                lang_x,
                attention_mask=None,
                layer_head_mask=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False):

        vis_x = self.perciever_layer(vis_x)
        lang_x = self.gated_cross_attn_layer(lang_x, vis_x)
        lang_x = self.decoder_layer(lang_x,
                                    attention_mask=attention_mask,
                                    layer_head_mask=layer_head_mask,
                                    past_key_value=past_key_value,
                                    output_attentions=output_attentions,
                                    use_cache=use_cache)
        return lang_x


class OPTForCausalLMFlamingo(OPTForCausalLM):
    def __init__(self, config, perceiver_depth=2):
        super().__init__(config)

        self.gated_cross_attn = [
            GatedCrossAttentionBlock(dim=self.config.hidden_size)
            for _ in range(len(self.get_decoder().layers))
        ]

        self.perceiver_resampler = [
            PerceiverResampler(dim=self.config.hidden_size,
                               depth=perceiver_depth)
            for _ in range(len(self.get_decoder().layers))
        ]

        self.get_decoder().layers = nn.ModuleList([FlamingoLayer(perceiver_layer, gated_cross_attn_layer, decoder_layer)
                                                   for perceiver_layer, gated_cross_attn_layer, decoder_layer in
                                                   zip(self.perceiver_resampler, self.gated_cross_attn, self.get_decoder().layers)])

    def forward(
        self,
        vision_attended: torch.Tensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple:
        r"""
        Args:
            vision_attended (:obj:`torch.Tensor` of shape :obj:`(batch_size, vision_sequence_length)`):
                The output of the vision encoder.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using [`OPTTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.get_decoder().embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        pos_embeds = self.get_decoder().embed_positions(
            attention_mask, past_key_values_length)

        attention_mask = self.get_decoder()._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        if self.get_decoder().project_in is not None:
            inputs_embeds = self.get_decoder().project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None and attn_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, decoder_layer in enumerate(self.get_decoder().layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.get_decoder().gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    vision_attended,
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    vision_attended,
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=(
                        head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.get_decoder().final_layer_norm is not None:
            hidden_states = self.get_decoder().final_layer_norm(hidden_states)

        if self.get_decoder().project_out is not None:
            hidden_states = self.get_decoder().project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        logits = self.lm_head(hidden_states).contiguous()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        output = (logits,) + (next_cache, all_hidden_states, all_self_attns)
        return (loss,) + output if loss is not None else output

    def generate(self, **kwargs):
        return NotImplementedError("Generation is not implemented for this model.")
