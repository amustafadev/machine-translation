# imports
import os
import json
import config
from typing import List
import sentencepiece as spm
from collections import Counter

from utils import get_file_name_from_path


class Vocabulary:
  def __init__(
      self,
      data_path: str,
      min_frequency: int = 2,
  ):
    self.data_path = data_path
    self.model_prefix = get_file_name_from_path(path=data_path)
    self.min_frequency = min_frequency
    self.tokenizer = None
    self.word2id = {}
    self.id2word = {}

    self.load_or_train_tokenizer()
    self.build_vocab()

  def load_or_train_tokenizer(self):
    model_file = f'{config.TOKENIZER_FOLDER}/{self.model_prefix}.model'
    if os.path.exists(model_file):
      self.tokenizer = spm.SentencePieceProcessor(model_file=model_file)
    else:
      self.train_tokenizer()

  def train_tokenizer(self):
    model_prefix = f'{config.TOKENIZER_FOLDER}/{self.model_prefix}'

    spm.SentencePieceTrainer.Train(
        input=self.data_path,
        model_prefix=model_prefix,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UKN]',
        bos_piece='[SOS]',
        eos_piece='[END]'
    )

    self.tokenizer = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')

  def build_vocab(self):
    counter = Counter()
    with open(self.data_path, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip().lower()
        counter.update(self.tokenizer.encode_as_pieces(line))

    special_tokens = ['<PAD>', '<UKN>', '<SOS>', '<END>']

    for token in special_tokens:
      self.word2id[token] = len(self.word2id)
      self.id2word[len(self.id2word)] = token

    for token, freq in counter.most_common():
      if freq >= self.min_frequency and token not in special_tokens:
        self.word2id[token] = len(self.word2id)
        self.id2word[len(self.id2word)] = token

  def encode_as_ids(self, sentence: str) -> List[int]:
    return self.tokenizer.encode_as_ids(sentence)

  def decode_ids(self, ids: List[int]) -> str:
    return self.tokenizer.decode_ids(ids)

  def save_vocab(self, path: str):
    with open(path, 'w', encoding='utf-8') as f:
      json.dump({'word2id': self.word2id, 'id2word': self.id2word}, f, indent=4)

  @classmethod
  def load_vocab(cls, path: str) -> 'Vocabulary':
    with open(path, 'r', encoding='utf-8') as f:
      vocab_dict = json.load(f)
    vocab = cls.__new__(cls)
    vocab.data_path = None
    vocab.model_prefix = None
    vocab.vocab_size = None
    vocab.special_tokens = None
    vocab.min_frequency = None
    vocab.tokenizer = None
    vocab.word2id = vocab_dict['word2id']
    vocab.id2word = vocab_dict['id2word']
    return vocab

  def __len__(self):
    return len(self.tokenizer)


# run file
if __name__ == '__main__':
  model_prefix = 'eng'

  # initialize the vocabulary object for language
  vocab = Vocabulary(
      data_path=f'{config.VOCABULARY_FOLDER}/{model_prefix}.txt',
      min_frequency=2
  )

  # save vocabulary
  vocab.save_vocab(path=f'{config.VOCABULARY_FOLDER}/{model_prefix}.json')
