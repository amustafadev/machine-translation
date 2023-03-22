# imports
from torch import nn

from modules.transformer import Transformer
from modules.positional_encoding import PositionalEncoding


class Model(nn.Module):
  # constructor
  def __init__(
      self,
      num_classes: int,
      d_model: int,
      nhead: int,
      num_encoder_layers: int,
      num_decoder_layers: int,
      num_src_embeddings: int,
      dropout: float = 0.1,
      seq_length: int = 128,
      padding_idx: int = 0,
      dim_feedforward: int = 2048,
  ):
    # super constructor
    super().__init__()

    # modules
    self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, seq_length=seq_length)
    self.src_embedder = nn.Embedding(num_embeddings=num_src_embeddings, embedding_dim=d_model, padding_idx=padding_idx)
    self.tgt_embedder = nn.Embedding(num_embeddings=num_classes, embedding_dim=d_model, padding_idx=padding_idx)
    self.transformer = Transformer(
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward
    )

  def forward(self, src, tgt):
    src = self.src_embedder(src)
    src = self.pos_encoder(src)

    tgt = self.tgt_embedder(tgt)
    tgt = self.pos_encoder(tgt)

    src = src.permute(dims=(1, 0, 2))
    tgt = tgt.permute(dims=(1, 0, 2))

    out = self.transformer(src, tgt)
    return out


# run file
if __name__ == '__main__':
  pass
