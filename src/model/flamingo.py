'''
Main Flamingo class
Uses gated cross attention with Perceiver resampler
'''

from helpers import GatedCrossAttentionBlock, PerceiverResampler

class Flamingo(nn.Module):
  '''
    parameters:
      vision_encoder: any vision encoder
      lang_encoder: ModuleList containing language encoder blocks
  '''
  def __init__(self, vision_encoder, lang_encoder):
    self.vision_encoder = vision_encoder
    self.lang_encoder = lang_encoder
    
    self.gated_cross_attn = nn.ModuleList([
      GatedCrossAttentionBlock(dim)
      for _ in range(len(lang_encoder))
    ])
    
    self.perceiver_resampler = nn.ModuleList([
      PerceiverResampler(dim)
      for _ in range(len(lang_encoder))
    ])

  def forward(self, vision_x, lang_x):
    vision_attended = self.vision_encoder(vision_x)
    
    for i in range(len(self.lang_encoder)):
      vision_x = self.perceiver_resampler[i](vision_attended)
      lang_x = self.gated_cross_attn[i](lang_x, vision_x)
      lang_x = self.language_encoder[i](lang_x)
   
   return lang_x
