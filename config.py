# vocabulary config
TOKENIZER_FOLDER = 'tokenizers'
VOCABULARY_FOLDER = 'vocabularies'


# dataset config
SEQ_LEN = 128
VAL_SPLIT = 0.1


# model config
DIM = 128
NUM_HEADS = 2
NUM_ENCODERS = 1
NUM_DECODERS = 1


# train config
EXP_NAME = 'test'
MODEL_NAME = f'e{NUM_ENCODERS}d{NUM_DECODERS}h{NUM_HEADS}v{DIM}'

LR = 1e-2
BETAS = [0.9, 0.99]

BATCH_SIZE = 32
NUM_WORKERS = 8

NUM_EPOCHS = 100
PRINT_AFTER_BATCHES = 250

SAVE_FOLDER = 'save'
WEIGHTS_FOLDER = 'weights'
TENSORBOARD_FOLDER = 'runs'
CHECKPOINTS_FOLDER = 'checkpoints'

SAVE_CHECKPOINTS = False
CHECKPOINT_EPOCHS = 10
CHECKPOINT_LOSS_CHECK = True

SAVE_BEST_LOSS = True
