from dm_control import suite
from matplotlib import animation
import numpy as np
import mujoco
import copy
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from scipy.spatial.transform import Rotation as R
from dataset import MujocoDataset
from env import *
from torch_geometric.utils.convert import from_networkx, to_networkx
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm
import pandas as pd

from utils import *
from render import generate_video
from env import *
from dataset import MujocoDataset
from render import Renderer
from train import test

envs = ['swimmer', 'cheetah', 'acrobot', 'pendulum']
data = []
for env in envs:
    error = test('', env=env, save_path=f'results/test_{env}.mp4')
    data.append(['GN', env, error])
    const_error = test('constant', env=env, trials=1, save_path='test_result.mp4')
    data.append(['Constant', env, const_error])

df = pd.DataFrame(data, columns=['group', 'column', 'val'])
df.pivot("column", "group", "val").plot(kind='bar')
plt.ylabel('error')
plt.xlabel('environment')

plt.savefig('test.png')
plt.show()