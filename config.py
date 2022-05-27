import torch
import dm_control.suite.swimmer as swimmer
from dm_control import suite


SEED = 42
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
LINUX = False
DATASET_PATH = 'datasets/swimmer6_dataset_general_7k.pkl'
CHECKPOINT_DIR = 'checkpoints'
MODEL_DIR = 'models'

MAX_EPOCH = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
VAL_INTERVAL = 100
SAVE_INTERVAL = 500
N_LINKS = 6
N_RUNS = 7000
N_STEPS = 100

DRAW_X_IDX = 0
DRAW_Y_IDX = 1
BODY_LENGTH = 0.05
ANGLE_IDX = 0

NOISE = 0
LR_DECAY = 0.975
LR_DECAY_STEP = 50000
ROTATION_LOSS_WEIGHT = 16
DROPOUT = 0.2
DM_HIDDEN_FEATURES = [256, 256]
GNN_TYPE = 'graphconv' # graphconv, gatconv
N_HEADS = 2

DEBUG = False

def GET_ENVIRONMENT():
    return swimmer.swimmer(N_LINKS)
    # return suite.load('cheetah', 'run')