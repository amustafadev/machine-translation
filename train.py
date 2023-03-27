# imports
import torch

import config
from model import Model
from modules import Trainer
from vocabulary import Vocabulary
from dataset import TranslationDataset


# vocabularies
src_vocab = Vocabulary(data_path=f'{config.VOCABULARY_FOLDER}/eng.txt')
tgt_vocab = Vocabulary(data_path=f'{config.VOCABULARY_FOLDER}/asl.txt')


# datasets
train_dataset = TranslationDataset(csv_path='./datasets/eng_asl.csv', src_vocab=src_vocab, tgt_vocab=tgt_vocab, seq_length=128)
val_dataset = TranslationDataset(csv_path='./datasets/eng_asl.csv', src_vocab=src_vocab, tgt_vocab=tgt_vocab, seq_length=128, train=False)


# model
model = Model(
    num_classes=len(tgt_vocab),
    d_model=512,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=1,
    num_src_embeddings=len(src_vocab)
)


# optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LR, betas=config.BETAS)


# loss
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)


# training
trainer = Trainer(model=model, train_datset=train_dataset, loss_criterion=criterion, optimizer=optimizer, validation_dataset=val_dataset)
trainer.fit()
