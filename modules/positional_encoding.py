# imports
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
  # constructor
  def __init__(self, d_model: int, dropout: float, seq_length: int):
    # super constructor
    super().__init__()

    # modules
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(seq_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_length, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = x + self.pe[:x.size(0)]
    out = self.dropout(out)
    return out
