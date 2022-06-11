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

from utils import *
from render import generate_video


cec = CheetahEnvCreator()
nx.draw(cec.base_graphs[0])
plt.show()