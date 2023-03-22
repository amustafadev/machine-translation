# Imports
import os
import torch
import config
import shutil
import numpy as np
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset
import torch.nn.modules.loss as losses
from torch.utils.data import DataLoader

from utils import *

from .process import *
from .tensorboard import Tensorboard


# Class: Trainer
class Trainer():
  # Constructor
  def __init__(self, model: torch.nn, train_datset: Dataset, loss_criterion: losses, optimizer: optim.Optimizer, validation_dataset: Dataset = None) -> None:
    # Dataloaders
    self.trainloader = DataLoader(dataset=train_datset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    self.valloader = None
    if validation_dataset:
      self.valloader = DataLoader(dataset=validation_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
      self.valloader = None if len(self.valloader) == 0 else self.valloader

    # Loss and Optimizer
    self.criterion = loss_criterion
    self.optimizer = optimizer

    # Device
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    self.model = model.to(self.device)

    # Save Folder
    now = datetime.now().strftime("%y-%m-%d %H-%M-%S")
    save_folder = f'{config.SAVE_FOLDER}/{config.EXP_NAME}/{config.MODEL_NAME}/{now}'

    # Weights
    self.weights_folder = f'{save_folder}/{config.WEIGHTS_FOLDER}'

    # Tensorboard
    log_dir = f'{save_folder}/{config.TENSORBOARD_FOLDER}'
    self.tensorboard = Tensorboard(log_dir=log_dir)

  # Function: Train
  def fit(self):
    # Set Mode
    self.model.train()

    # Copy Config File
    os.makedirs(name=self.weights_folder, exist_ok=True)
    save_path = f'{self.weights_folder}/config.py'
    shutil.copy(config.__file__, save_path)

    # Print Stats
    print('\nDATA STATS:')
    print(f'Total Training Examples    : {self.trainloader.dataset.__len__()}')
    print(f'Total Validation Examples  : {self.valloader.dataset.__len__()}')
    print(f'Number of batches per Epoch: {self.trainloader.__len__()}')

    print('\nModel STATS:')
    print(f'Device              : {self.device}')

    # Initialize training metrics
    self.best_loss, self.best_acc = np.inf, 0.0
    self.save_loss = np.inf

    # Start Training
    print('\nTRAINING:')
    for epoch in range(config.NUM_EPOCHS):
      # Initialize epoch metrics
      train_acc, train_loss = 0.0, 0.0

      # Update Epoch Number
      self.cur_epoch = epoch

      # Loop over batch
      for batch_idx, (src, tgt) in enumerate(self.trainloader):
        # Device
        src = src.to(self.device)
        tgt = tgt.to(self.device)

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # align target
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # Forward + Backward + Optimize
        outputs = self.model(src, tgt_in)
        loss = self.criterion(outputs.permute(1, 2, 0), tgt_out)
        loss.backward()
        self.optimizer.step()

        # decode
        truth = decode_batch(batch=src, vocabulary=self.trainloader.dataset.src_vocab)
        predictions = decode_batch(batch=outputs.argmax(dim=-1).permute(1, 0), vocabulary=self.trainloader.dataset.tgt_vocab)

        # Batch accuracy
        train_acc *= batch_idx
        train_acc += calculate_accuracy(truth=truth, predictions=predictions)
        train_acc /= (batch_idx + 1)

        # Batch loss
        train_loss *= batch_idx
        train_loss += loss.item()
        train_loss /= (batch_idx + 1)

        # Print Stats in Terminal
        if batch_idx == 0 or batch_idx == len(self.trainloader) - 1 or (batch_idx + 1) % config.PRINT_AFTER_BATCHES == 0:
          print(f'Epoch: {epoch + 1}/{config.NUM_EPOCHS}\t Batch: {batch_idx + 1:5d} | Loss: {train_loss:.3f}\t Accuracy: {train_acc:.3f}')

      # Update Tensorbaord
      self.tensorboard.add_scalar(tag='train/loss', value=train_loss)
      self.tensorboard.add_text(tag='train/output', text=process_text_for_tensorboard(truth=truth, predictions=predictions))

      # Validation Set Evaluate
      if self.valloader:
        val_loss, val_acc = self.evaluate_val()
        print(f'Val Loss: {val_loss:.3f}  Best Loss: {self.best_loss:.3f}  Val Accuracy: {val_acc:.3f}  Best Accuracy: {self.best_acc:.3f}\n')

    # End
    print('TRAINING COMPLETE.\n')

  # Function: Evaluate on Test Set
  def evaluate_val(self):
    # Set Mode
    self.model.eval()

    # Initialize log vars
    val_acc = 0.0
    val_loss = 0.0

    # Loop over batch
    for batch_idx, (src, tgt) in enumerate(self.valloader):
      # Device
      src = src.to(self.device)
      tgt = tgt.to(self.device)

      # align target
      tgt_in = tgt[:, :-1]
      tgt_out = tgt[:, 1:]

      # Forward + Backward + Optimize
      outputs = self.model(src, tgt_in)
      loss = self.criterion(outputs.permute(1, 2, 0), tgt_out)

      # decode
      truth = decode_batch(batch=src, vocabulary=self.valloader.dataset.src_vocab)
      predictions = decode_batch(batch=outputs.argmax(dim=-1).permute(1, 0), vocabulary=self.valloader.dataset.tgt_vocab)

      # Batch accuracy
      val_acc *= batch_idx
      val_acc += calculate_accuracy(truth=truth, predictions=predictions)
      val_acc /= (batch_idx + 1)

      # Batch loss
      val_loss *= batch_idx
      val_loss += loss.item()
      val_loss /= (batch_idx + 1)

    # Update Best Metrics
    self.best_loss = np.min([val_loss, self.best_loss])
    self.best_acc = np.max([val_acc, self.best_acc])

    # Update Tensorboard
    self.tensorboard.add_scalar(tag='val/loss', value=val_loss)
    self.tensorboard.add_scalar(tag='val/acc', value=val_acc)
    self.tensorboard.add_text(tag='val/output', text=process_text_for_tensorboard(truth=truth, predictions=predictions))

    # Save Best Loss
    if config.SAVE_BEST_LOSS and val_loss == self.best_loss:
      self.save(loss=val_loss, acc=val_acc)

    # Save Checkpoint
    elif config.SAVE_CHECKPOINTS and (self.cur_epoch + 1) % config.CHECKPOINT_EPOCHS == 0 and (self.cur_epoch + 1) < config.NUM_EPOCHS:
      if config.CHECKPOINT_LOSS_CHECK and (val_loss > self.save_loss):
        pass
      else:
        self.save(loss=val_loss, acc=val_acc)

    # Reset Mode
    self.model.train()

    # Return
    return val_loss, val_acc

  # Function: Save Model
  def save(self, loss, acc):
    # Model Save Folder
    save_folder = f'{self.weights_folder}/best'
    os.makedirs(name=save_folder, exist_ok=True)

    # Save Model State
    save_path = f'{save_folder}/model_state.pt'
    torch.save(obj=self.model.state_dict(), f=save_path)

    # Save Optimizer State
    save_path = f'{save_folder}/optim_state.pt'
    torch.save(obj=self.optimizer.state_dict(), f=save_path)

    # Update Save Loss
    self.save_loss = loss

    # Print
    print(f'Model Saved. Loss: {loss:.3f}  Accuracy: {acc:.3f}\n')
