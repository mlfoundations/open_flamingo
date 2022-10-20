'''
Flamingo model
Uses gated cross attention with Perceiver resampler
'''

from helpers import GatedCrossAttentionBlock, PerceiverResampler
from torch import nn

class Flamingo(nn.Module):
  '''
    parameters:
      vision_encoder: any vision encoder
      lang_decoder: ModuleList containing language decoder blocks
  '''
  def __init__(self, vision_encoder, lang_decoder):
    self.vision_encoder = vision_encoder
    self.lang_decoder = lang_decoder
    
    self.gated_cross_attn = nn.ModuleList([
      GatedCrossAttentionBlock(dim)
      for _ in range(len(self.lang_decoder))
    ])
    
    self.perceiver_resampler = nn.ModuleList([
      PerceiverResampler(dim)
      for _ in range(len(self.lang_decoder))
    ])

  def forward(self, vision_input, text_input):
    vision_attended = self.vision_encoder(vision_input)
    
    for i in range(len(self.lang_decoder)):
      vision_input = self.perceiver_resampler[i](vision_attended)
      text_input = self.gated_cross_attn[i](text_input, vision_input)
      text_input = self.lang_decoder[i](text_input)
   
    return text_input
