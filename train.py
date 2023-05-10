# imports
import torch
from adabelief_pytorch import AdaBelief

import config
from model import Model
from modules import Trainer
from vocabulary import Vocabulary
from dataset import TranslationDataset


# vocabularies
src_vocab = Vocabulary(data_path=f'{config.VOCABULARY_FOLDER}/eng.txt')
tgt_vocab = Vocabulary(data_path=f'{config.VOCABULARY_FOLDER}/asl.txt')


# datasets
train_dataset = TranslationDataset(csv_path='./datasets/eng_asl.csv', src_vocab=src_vocab, tgt_vocab=tgt_vocab, seq_length=config.SEQ_LEN)
val_dataset = TranslationDataset(csv_path='./datasets/eng_asl.csv', src_vocab=src_vocab, tgt_vocab=tgt_vocab, seq_length=config.SEQ_LEN, train=False)


# model
model = Model(
    num_classes=len(tgt_vocab),
    d_model=config.DIM,
    nhead=config.NUM_HEADS,
    num_encoder_layers=config.NUM_ENCODERS,
    num_decoder_layers=config.NUM_DECODERS,
    num_src_embeddings=len(src_vocab)
)


# optimizer
optimizer = AdaBelief(params=model.parameters(), lr=config.LR, betas=config.BETAS, weight_decay=config.DECAY, print_change_log=False)


# loss
criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab.tokenizer.pad_id())


# training
trainer = Trainer(model=model, train_datset=train_dataset, loss_criterion=criterion, optimizer=optimizer, validation_dataset=val_dataset)
trainer.fit()
