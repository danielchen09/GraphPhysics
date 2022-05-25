import torch


SEED = 42
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MAX_EPOCH = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
VAL_INTERVAL = 100
SAVE_INTERVAL = 500
N_LINKS = 3
N_RUNS = 5000
N_STEPS = 100
NOISE = 1e-2
MOMENTUM = 0
LR_DECAY = 0.975

DATASET_PATH = 'datasets/swimmer3_dataset_general.pkl'
CHECKPOINT_DIR = 'checkpoints'
MODEL_DIR = 'models'

DEBUG = True