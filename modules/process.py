# imports
import torch
from typing import List
from fuzzywuzzy import fuzz

from vocabulary import Vocabulary


def decode_batch(batch: torch.Tensor, vocabulary: Vocabulary):
  batch = batch.detach().tolist()
  string = vocabulary.decode_ids(ids=batch)
  return string


def calculate_accuracy(truth: List[str], predictions: List[str]):
  acc = 0
  num_examples = len(truth)
  for i in range(num_examples):
    acc += fuzz.ratio(truth[i], predictions[i])
  acc = acc / (num_examples * 100)
  return acc


def process_text_for_tensorboard(truth: List[str], predictions: List[str]):
  text = f'**True**: {truth[0]}<br>**Prediction**: {predictions[0]}'
  return text
