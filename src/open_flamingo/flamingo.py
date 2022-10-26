'''
Main Flamingo class
Uses gated cross attention with Perceiver resampler
'''

from typing import List

import torch
from torch import nn

from .flamingo_lm import OPTForCausalLMFlamingo
from .helpers import GatedCrossAttentionBlock, PerceiverResampler


class Flamingo(nn.Module):
    def __init__(self, vision_encoder: nn.Module, lang_encoder: OPTForCausalLMFlamingo):
        """
        Args:
            vision_encoder (nn.Module): Any vision encoder
            lang_encoder (OPTForCausalLMFlamingo): An instance of OPTForCausalLMFlamingo
        """
        super().__init__()
        self.vision_encoder = vision_encoder
        self.lang_encoder = lang_encoder

    def forward(self, vision_x: torch.Tensor, lang_x: torch.Tensor):
        vision_attended = self.vision_encoder(vision_x)
        return self.lang_encoder(vision_attended.last_hidden_state, lang_x)

    def greedy_generate(self, vision_x: torch.Tensor, lang_x: torch.Tensor, max_length: int, eoc_token_id: int):
        """ Adapted from https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/generation_utils.py#L1637
        This is a barebones implementation of greedy decoding. We should work on better methods later.

        Args:
            vision_x (torch.Tensor): A batch of images
            lang_x (torch.Tensor): A tensor of shape (batch_size, sequence_length)
            max_length (int): Maximum length of the generated text sequence if no EOC token is encountered
            eoc_token_id (int): End of chunk token id used to terminate generation
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        if eoc_token_id is None:
            raise ValueError(
                "eoc_token_id must be provided to determine end of generation")

        unfinished_sequences = lang_x.new(lang_x.shape[0]).fill_(1)

        while True:
            output = self.forward(vision_x, lang_x)[0][:, -1, :] # just get logits for final token
            # get token with highest probability
            new_tokens = torch.argmax(output, dim=-1)
            lang_x = torch.cat((lang_x, new_tokens[:, None]), dim=-1)

            unfinished_sequences.mul_(new_tokens.ne(eoc_token_id).long())
            
            if unfinished_sequences.max() == 0 or lang_x.shape[1] >= max_length:
                break

        return lang_x
