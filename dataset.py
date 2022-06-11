from email.mime import base
from re import L
from tkinter.ttk import setup_master
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

        self.data = self.load_save() if load_from_path else None
        if self.data is None:
            self.generate_dataset()
    
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
        obs_old, action, obs_new, env_key = self.data[idx % dataset_size]
        base_graph = self.env_creator.base_graphs[env_key]
        geom_names = self.env_creator.geom_names[env_key]
        edge_order = self.env_creator.edge_orders[env_key]
        node_attrs_new = get_dynamic_node_attrs(obs_new, geom_names)
        node_attrs_old = get_dynamic_node_attrs(obs_old, geom_names)
        center = center_attrs(node_attrs_old, (0, 3))
        center_attrs(node_attrs_new, (0, 3), center)
        node_attrs_old = add_noise(node_attrs_old, self.noise)
        graph_old = Graph.from_nx_graph(
                        base_graph.copy(), 
                        None, 
                        node_attrs_old, 
                        get_dynamic_edge_attrs(action, base_graph.edges, edge_order)
                    )
        info = {
            'center': center,
            'env_key': env_key,
            'static_node_attrs': self.env_creator.static_node_attrs[env_key]
        }
        return graph_old, node_attrs_old, node_attrs_new, info

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
            center = torch.cat([info['center'] for _, _, _, info in inp], dim=0)
            static_node_attrs = torch.cat([info['static_node_attrs'] for _, _, _, info in inp], dim=0)
            return x, y_old, y_new, center, static_node_attrs
        return collate_fn

def test():
    from render import Renderer, generate_video
    from env import CompositeEnvCreator
    renderer = Renderer()
    ds = MujocoDataset(CompositeEnvCreator(env_args={'swimmer': {'n_links':6}}), n_runs=1, n_steps=20)
    frames = []
    for g_old, n_old, n_new, info in ds:
        center_attrs(n_old, (0, 3), -info['center'])
        frames.append(renderer.render(n_old, ds.env_creator, info['env_key']))
    
    generate_video(frames)

if __name__ == '__main__':
    test()