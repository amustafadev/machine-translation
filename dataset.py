# imports
import torch
import pandas as pd
from torch.utils.data.dataset import Dataset

import config
from vocabulary import Vocabulary


class TranslationDataset(Dataset):
  def __init__(
      self,
      csv_path: str,
      src_vocab: Vocabulary,
      tgt_vocab: Vocabulary,
      seq_length: int,
      train: bool = True,
      val_split: float = 0.1
  ) -> None:

    src_vocab.tokenizer._add_bos = True
    src_vocab.tokenizer._add_eos = True
    self.src_vocab = src_vocab

    tgt_vocab.tokenizer._add_bos = True
    tgt_vocab.tokenizer._add_eos = True
    self.tgt_vocab = tgt_vocab

    self.seq_length = seq_length

    df = pd.read_csv(csv_path)

    src_data = df[src_vocab.model_prefix]
    tgt_data = df[tgt_vocab.model_prefix]

    assert len(src_data) == len(tgt_data), 'source and target data must be of the same length!'

    split_index = int(len(src_data) * val_split)
    src_data = src_data.iloc[:-split_index] if train else src_data.iloc[-split_index:]
    tgt_data = tgt_data.iloc[:-split_index] if train else tgt_data.iloc[-split_index:]

    self.src_data = src_data.tolist()
    self.tgt_data = tgt_data.tolist()

  def __getitem__(self, idx: int):
    src_string = self.src_data[idx].strip().lower()
    src_encoded = self.src_vocab.encode_as_ids(sentence=src_string)
    src_sequence = src_encoded + [self.src_vocab.word2id['<PAD>']] * (self.seq_length - len(src_encoded) - 2)
    src_tensor = torch.LongTensor(src_sequence)

    tgt_string = self.tgt_data[idx].strip().lower()
    tgt_encoded = self.tgt_vocab.encode_as_ids(sentence=tgt_string)
    tgt_sequence = tgt_encoded + [self.tgt_vocab.word2id['<PAD>']] * (self.seq_length - len(tgt_encoded) - 1)
    tgt_tensor = torch.LongTensor(tgt_sequence)

    return src_tensor, tgt_tensor

  def __len__(self):
    return len(self.src_data)


# run file
if __name__ == '__main__':

  src_vocab = Vocabulary(data_path=f'{config.VOCABULARY_FOLDER}/eng.txt')
  tgt_vocab = Vocabulary(data_path=f'{config.VOCABULARY_FOLDER}/asl.txt')

  dataset = TranslationDataset(
      csv_path='./datasets/eng_asl.csv',
      src_vocab=src_vocab,
      tgt_vocab=tgt_vocab,
      seq_length=128,
      train=False
  )

  out = dataset[0]
  pass
