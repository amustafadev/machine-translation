# imports
import torch
from torch import nn


class Transformer(nn.Module):
  # constructor
  def __init__(
          self,
          num_classes: int,
          d_model: int,
          nhead: int,
          num_encoder_layers: int,
          num_decoder_layers: int,
          dim_feedforward: int
  ):
    # super constructor
    super().__init__()

    # constants
    self.nhead = nhead

    # modules
    self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                      num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)
    self.linear = nn.Linear(in_features=d_model, out_features=num_classes)
    self.softmax = nn.Softmax(dim=2)

  # forward prop
  def forward(self, x: torch.Tensor, y: torch.Tensor):
    # target mask
    if self.training:
      T = y.shape[0]
      mask = torch.ones(size=(T, T)) * float('-inf')
      mask = torch.triu(input=mask, diagonal=1)
      mask = mask.to(x.device)
    else:
      mask = None

    # forward
    out = self.transformer(src=x, tgt=y, tgt_mask=mask)
    out = self.linear(out)

    # return
    return out
