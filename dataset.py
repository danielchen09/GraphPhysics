from email.mime import base
from re import L
from tkinter.ttk import setup_master
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils.convert import from_networkx
import numpy as np
import dm_control
import dm_control.suite.swimmer as swimmer
import random
import os
from tqdm import tqdm

import config
from utils import *
from graphs import Graph, GraphData


class MujocoDataset(Dataset):
    def __init__(self, 
                 env_creator,
                 env_name='',
                 n_runs=config.N_RUNS, 
                 n_steps=config.N_STEPS,
                 noise=0,
                 dataset_path=config.DATASET_PATH,
                 load_from_path=False,
                 save=False,
                 shuffle=False):
        self.env_creator = env_creator
        self.env_name = env_name
        self.n_runs = n_runs
        self.n_steps = n_steps
        self.noise = noise
        self.dataset_path = dataset_path
        self.save = save
        self.shuffle = shuffle
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        self.subset_idx = 0

        self.classes = {key: [] for key in self.env_creator.keys}
        self.data = self.load_save() if load_from_path else None
        if self.data is None:
            self.generate_dataset()
        for i, data in enumerate(self.data):
            self.classes[data[-1]].append(i)
    
    def generate_dataset(self):
        self.subset_idx = 0
        self.data = []
        for _ in tqdm(range(self.n_runs), desc='generating dataset', disable=self.n_runs <= 10):
            self.data += self.generate_run()
            if len(self.data) // self.n_steps >= config.WORKING_DATASET_SIZE and self.save:
                self.save_dataset()
                self.subset_idx += 1
                self.data = []
        if len(self.data) > 0 and self.save:
            self.save_dataset()
        self.subset_idx = 0
        if self.save:
            self.data = self.load_save()

    def get_path(self):
        return f'{self.dataset_path}/{self.subset_idx}.pkl'

    def load_save(self):
        print(f'Load dataset {self.get_path()}')
        path = self.get_path()
        if not os.path.exists(path):
            print('Existing dataset not found')
            return None
        dataset = load_pickle(path)
        if self.env_name != dataset['env'] or self.n_steps != dataset['n_steps'] or self.n_runs != dataset['n_runs']:
            print('Dataset information mismatch')
            return None
        if self.shuffle:
            random.shuffle(dataset['data'])
        self.classes = {key: [] for key in self.env_creator.keys}
        for i, data in enumerate(dataset['data']):
            self.classes[data[-1]].append(i)
        return dataset['data']

    def save_dataset(self):
        print(f'Saved dataset {self.get_path()}')
        dataset = {
            'env': self.env_name,
            'data': self.data,
            'n_steps': self.n_steps,
            'n_runs': self.n_runs
        }
        save_pickle(self.get_path(), dataset)

    def generate_run(self):
        run_data = []
        env, env_key = self.env_creator.create_env()
        env.reset()
        last_obs = get_state(env, self.env_creator.geom_names[env_key])
        action_space = env.action_spec()
        random_state = np.random.RandomState(config.SEED)

        for _ in range(self.n_steps):
            action = random_state.uniform(
                action_space.minimum,
                action_space.maximum,
                action_space.shape
            )
            env.step(action)
            new_obs = get_state(env, self.env_creator.geom_names[env_key])
            run_data.append((last_obs, action, new_obs, env_key))
            last_obs = new_obs
        return run_data

    def __len__(self):
        return self.n_runs * self.n_steps

    def __getitem__(self, idx):
        dataset_size = config.WORKING_DATASET_SIZE * self.n_steps
        subset_idx = idx // dataset_size
        if subset_idx != self.subset_idx:
            self.subset_idx = subset_idx
            self.data = self.load_save()
        sampled_class = self.env_creator.sample_key()
        if not self.save:
            sampled_idx = idx % dataset_size
        else:
            sampled_idx = np.random.choice(self.classes[sampled_class], 1)[0]
        obs_old, action, obs_new, env_key = self.data[sampled_idx]
        base_graph = self.env_creator.base_graphs[env_key]
        geom_names = self.env_creator.geom_names[env_key]
        edge_order = self.env_creator.edge_orders[env_key]
        node_attrs_new = get_dynamic_node_attrs(obs_new, geom_names)
        node_attrs_old = get_dynamic_node_attrs(obs_old, geom_names)
        center = center_attrs(node_attrs_old, (0, 3))
        center_attrs(node_attrs_new, (0, 3), center)
        node_attrs_old = add_noise(node_attrs_old, self.noise)
        edge_index = from_networkx(base_graph).edge_index
        graph_old = GraphData(
            edge_index,
            None,
            node_attrs_old,
            edge_attrs=get_dynamic_edge_attrs(action, base_graph.edges, edge_order)
        )
        static_graph = GraphData(
            edge_index,
            self.env_creator.global_attrs[env_key],
            self.env_creator.static_node_attrs[env_key],
            edge_attrs=self.env_creator.static_edge_attrs[env_key]
        )
        info = {
            'center': center,
            'env_key': env_key
        }
        return graph_old, static_graph, Data(x=node_attrs_old), Data(x=node_attrs_new), info

def test():
    from render import Renderer, generate_video
    from env import CompositeEnvCreator
    import matplotlib.pyplot as plt
    from env import SwimmerEnvCreator, CheetahEnvCreator, WalkerEnvCreator
    renderer = Renderer()
    ds = MujocoDataset(WalkerEnvCreator(), n_runs=1, n_steps=1)
    G = ds[0][0].G
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(e[0], e[1]): f"{e[2]['edge_attr'].item():.2f}" for e in G.edges(data=True)})
    plt.show()

    
if __name__ == '__main__':
    test()