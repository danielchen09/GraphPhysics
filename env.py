import dm_control.suite.swimmer as swimmer
from dm_control import suite
import random

from utils import *
import config

class EnvCreator:
    def __init__(self, keys, geom_names, camera_configs=None):
        self.keys = keys
        self.geom_names = geom_names
        self.camera_configs = camera_configs
        if self.camera_configs is None:
            self.camera_configs = {key: [3, 90, 0] for key in keys}
        self.envs = {key: self._create_env(key) for key in keys}
        self.base_graphs = {key: get_graph(self.envs[key], geom_names[key]) for key in keys}
        self.static_node_attrs = {key: get_static_node_attrs(self.envs[key], geom_names[key]) for key in keys}
        self.edge_orders = {key: get_edge_order(self.envs[key], geom_names[key]) for key in keys}
        self.sizes = {key: get_geom_sizes(self.envs[key], geom_names[key]) for key in keys}

    def _create_env(self, key):
        pass
    
    def create_env(self):
        key = random.choice(self.keys)
        return self.envs[key], key

class CompositeEnvCreator(EnvCreator):
    def __init__(self, env_creators=None, env_args={}):
        def get_arg(k):
            return env_args[k] if k in env_args else {}

        self.env_creators = env_creators
        if env_creators is None:
            self.env_creators = [
                SwimmerEnvCreator(**get_arg('swimmer'), n_links=6), 
                CheetahEnvCreator(**get_arg('cheetah')),
                WalkerEnvCreator(**get_arg('walker'))
            ]
        keys = []
        geom_names = {}
        camera_configs = {}
        for i, env_creator in enumerate(self.env_creators):
            for key in env_creator.keys:
                keys.append((i, key))
                geom_names[(i, key)] = env_creator.geom_names[key]
                camera_configs[(i, key)] = env_creator.camera_configs[key]
        super(CompositeEnvCreator, self).__init__(keys, geom_names, camera_configs)
    
    def _create_env(self, key):
        env_idx, k = key
        return self.env_creators[env_idx]._create_env(k)

class SwimmerEnvCreator(EnvCreator):
    def __init__(self, n_links=-1):
        keys = [n_links]
        if n_links == -1:
            keys = [i for i in range(3, config.N_LINKS)]
        
        geom_names = {}
        camera_config = {}
        for key in keys:
            geom_names[key] = ['visual'] + [f'visual_{i}' for i in range(key - 1)] + ['ground']
            camera_config[key] = [1, 90, -90]
        super(SwimmerEnvCreator, self).__init__(keys, geom_names, camera_config)

    def _create_env(self, key):
        return swimmer.swimmer(key)
    
class CheetahEnvCreator(EnvCreator):
    def __init__(self):
        geom_names = [
            'torso', 
            'bthigh', 
            'bshin', 
            'bfoot', 
            'fthigh', 
            'fshin', 
            'ffoot',
            'ground'
        ]
        super(CheetahEnvCreator, self).__init__([0], {0: geom_names})

    def _create_env(self, key):
        return suite.load('cheetah', 'run')

class WalkerEnvCreator(EnvCreator):
    def __init__(self):
        geom_names = [
            'torso', 
            'right_thigh', 
            'right_leg', 
            'right_foot', 
            'left_thigh', 
            'left_leg', 
            'left_foot',
            'floor'
        ]
        super(WalkerEnvCreator, self).__init__([0], {0: geom_names})
    
    def _create_env(self, key):
        return suite.load('walker', 'run')