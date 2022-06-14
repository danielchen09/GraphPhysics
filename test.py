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
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from utils import *
from render import generate_video
from env import *
from dataset import MujocoDataset

class TestData(Data):
    def __init__(self, G):
        print(G is None)
        g = from_networkx(G)
        super(TestData, self).__init__(torch.randn(5, 13), g.edge_index, torch.randn(8, 7))
        self.z = torch.randn(1)
        self.G = G

    def update(self):
        self.x = torch.randn(5, 13)

class TestDataset(Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
    
    def __getitem__(self, idx):
        G = nx.path_graph(5).to_directed()
        return TestData(G), torch.tensor([5, 5])

    def __len__(self):
        return 2

# ds = TestDataset()
# dl = DataLoader(ds, batch_size=2)
# it = iter(dl)
# breakpoint()
# d = next(it)
# print(d)

ds = MujocoDataset(CompositeEnvCreator(), n_runs=1, n_steps=20)
for x in ds:
    print(x[0] is None)
dl = DataLoader(ds, batch_size=2)
for x in dl:
    print(x)