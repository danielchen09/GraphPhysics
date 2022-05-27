from re import L
import torch
from torch.utils.data import Dataset
import numpy as np
import dm_control
import dm_control.suite.swimmer as swimmer
import random
import os
from tqdm import tqdm

import config
from utils import *
from graphs import Graph


class MujocoDataset(Dataset):
    def __init__(self, 
                 env,
                 env_name='',
                 n_runs=config.N_RUNS, 
                 n_steps=config.N_STEPS,
                 noise=0,
                 dataset_path=config.DATASET_PATH,
                 load_from_path=False,
                 save=False):
        self.env = env
        self.env_name = env_name
        self.n_runs = n_runs
        self.n_steps = n_steps
        self.noise = noise
    
        self.action_space = self.env.action_spec()
        self.random_state = np.random.RandomState(config.SEED)
        
        self.body_names, self.name2id = get_body_name_and_id(env)
        self.body_idx = [self.name2id[name] for name in self.body_names]

        self.base_graph = get_graph(self.env)
        self.edge_order = get_edge_order(self.env)

        self.data = self.load_save(dataset_path) if load_from_path else None
        if self.data is None:
            self.data = []
            for _ in tqdm(range(self.n_runs), desc='generating dataset'):
                self.data += self.generate_run()
            if save:
                self.save_dataset(dataset_path)
    
    def load_save(self, path):
        if not os.path.exists(path):
            return None
        dataset = load_pickle(path)
        if self.env_name != dataset['env'] or self.n_steps != dataset['n_steps'] or self.n_runs != dataset['n_runs']:
            return None
        return dataset['data']

    def save_dataset(self, path):
        dataset = {
            'env': self.env_name,
            'data': self.data,
            'n_steps': self.n_steps,
            'n_runs': self.n_runs
        }
        save_pickle(path, dataset)

    def generate_run(self):
        run_data = []
        self.env.reset()
        last_obs = get_state(self.env)
        for _ in range(self.n_steps):
            action = self.random_state.uniform(
                self.action_space.minimum,
                self.action_space.maximum,
                self.action_space.shape
            )
            self.env.step(action)
            new_obs = get_state(self.env)
            run_data.append((last_obs, action, new_obs))
            last_obs = new_obs
        return run_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs_old, action, obs_new = self.data[idx]
        node_attrs_new = get_dynamic_node_attrs(obs_new, self.body_names)
        node_attrs_old = get_dynamic_node_attrs(obs_old, self.body_names)
        center = center_attrs(node_attrs_old, (0, 3))
        center_attrs(node_attrs_new, (0, 3), center)
        node_attrs_old = add_noise(node_attrs_old, self.noise)
        graph_old = Graph.from_nx_graph(
                        self.base_graph.copy(), 
                        None, 
                        node_attrs_old, 
                        get_dynamic_edge_attrs(action, self.base_graph.edges, self.edge_order)
                    )
        return graph_old, node_attrs_old, node_attrs_new, center

    def get_collate_fn(self, concat=True):
        def collate_fn(inp):
            if concat:
                x = Graph.empty()
                for graph, _, _, _ in inp:
                    x = Graph.union(x, graph)
                x = [x]
            else:
                x = [graph for graph, _, _, _ in inp]
            y_old = torch.cat([y for _, y, _, _ in inp], dim=0)
            y_new = torch.cat([y for _, _, y, _ in inp], dim=0)
            center = torch.cat([center for _, _, _, center in inp], dim=0)
            return x, y_old, y_new, center
        return collate_fn


# class MujocoSequenceDataset(MujocoDataset):
#     def __init__(self, 
#                  env,
#                  env_name='',
#                  n_runs=config.N_RUNS, 
#                  n_steps=config.N_STEPS,
#                  noise=0,
#                  dataset_path=config.DATASET_PATH,
#                  load_from_path=False,
#                  save=False):
#         super(MujocoSequenceDataset, self).__init__(env, env_name, n_runs, n_steps, noise, dataset_path, load_from_path, save)
    
#     def __getitem__(self, idx):
        