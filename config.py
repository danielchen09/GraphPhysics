from collections import namedtuple
import torch
import dm_control.suite.swimmer as swimmer
from dm_control import suite
import random


# files & general setting
SEED = 42
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
LINUX = False
DATASET_PATH = 'datasets/general_4_20k'
CHECKPOINT_DIR = 'checkpoints'
MODEL_DIR = 'models'
WORKING_DATASET_SIZE = 500

# training setting
MAX_EPOCH = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
VAL_INTERVAL = 100
SAVE_INTERVAL = 500

# dataset setting
N_RUNS = 20000
N_STEPS = 100

# rendering
N_LINKS = 6
BODY_LENGTH = 0.05

# hyperparameters
NOISE = 1e-4
LR_DECAY = 0.975
LR_DECAY_STEP = 50000
DROPOUT = 0.3
DM_HIDDEN_FEATURES = [512, 512, 512, 512]
USE_STATIC_ATTRS = True
USE_NORMALIZATION = True

# dynamic replay buffer
DRP_ALPHA = 0.1
DRP_BETA = 1.0

DEBUG = False

# MPC
MPC_LR = 1e-2
MPC_EPOCHS = 100
MPC_HORIZON = 100
