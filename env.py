import dm_control.suite.swimmer as swimmer
from dm_control import suite
import numpy as np

from utils import *
import config

class EnvCreator:
    def __init__(self, name, keys, geom_names, camera_configs=None, env_creators=None, use_root=True, include_fn=None):
        if include_fn is None:
            include_fn = lambda c, p: p != 'world'
        self.name = name
        self.keys = keys
        self.geom_names = geom_names
        self.camera_configs = camera_configs
        if self.camera_configs is None:
            self.camera_configs = {key: [4, 90, 0] for key in keys}
        self.envs = {key: self._create_env(key) for key in keys}
        self.global_attrs = {key: get_static_global_attrs(self.envs[key]) for key in keys}
        self.base_graphs = {key: get_graph(self.envs[key], geom_names[key]) for key in keys}
        self.static_node_attrs = {key: get_static_node_attrs(self.envs[key], geom_names[key]) for key in keys}
        self.edge_orders = {key: get_edge_order(self.envs[key], geom_names[key], include_fn, use_root=use_root) for key in keys}
        self.sizes = {key: get_geom_sizes(self.envs[key], geom_names[key]) for key in keys}
        self.static_edge_attrs = {key: get_static_edge_attrs(self.envs[key], self.base_graphs[key].edges, self.edge_orders[key], use_root=use_root) for key in self.keys}

        self.env_error = {key: 1 for key in self.keys}
        if env_creators is None:
            self.env_creators = [self]

    def update_error(self, key, error):
        if key not in self.env_error:
            self.env_error[key] = error
            return
        self.env_error[key] = config.DRP_ALPHA * error + (1 - config.DRP_ALPHA) * self.env_error[key]

    def get_probs(self):
        s = sum([self.env_error[key] ** config.DRP_BETA for key in self.keys])
        return [(self.env_error[key] ** config.DRP_BETA) / s for key in self.keys]

    def print_probs(self):
        print(self.get_probs())
    
    def print_errors(self):
        for k, v in self.env_error.items():
            print(f'{k}: {v}')

    def _create_env(self, key):
        pass
    
    def sample_key(self):
        return np.random.choice(self.keys, 1, p=self.get_probs())[0]

    def create_env(self, key=None):
        if key is None:
            key = self.sample_key()
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
                PendulumEnvCreator(**get_arg('pendulum')),
                AcrobotEnvCreator(**get_arg('acrobot'))
            ]
        keys = []
        geom_names = {}
        camera_configs = {}
        for i, env_creator in enumerate(self.env_creators):
            for key in env_creator.keys:
                keys.append(f'{i},{key}')
                geom_names[f'{i},{key}'] = env_creator.geom_names[key]
                camera_configs[f'{i},{key}'] = env_creator.camera_configs[key]
        super(CompositeEnvCreator, self).__init__('Composite', keys, geom_names, camera_configs, self.env_creators)
    
    def _create_env(self, key):
        env_idx, k = key.split(',')
        return self.env_creators[int(env_idx)]._create_env(int(k))

    def print_probs(self):
        for env_creator, prob in zip(self.env_creators, self.get_probs()):
            print(f'{env_creator}: {prob}')

class SwimmerEnvCreator(EnvCreator):
    def __init__(self, n_links=-1):
        keys = [n_links]
        if n_links == -1:
            keys = [i for i in range(3, config.N_LINKS)]
        
        geom_names = {}
        camera_config = {}
        for key in keys:
            geom_names[key] = ['visual'] + [f'visual_{i}' for i in range(key - 1)] + ['ground']
            camera_config[key] = [2, 90, -90]
        super(SwimmerEnvCreator, self).__init__('Swimmer', keys, geom_names, camera_config)

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
        super(CheetahEnvCreator, self).__init__('Cheetah', [0], {0: geom_names})

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
        super(WalkerEnvCreator, self).__init__('Walker', [0], {0: geom_names})
    
    def _create_env(self, key):
        return suite.load('walker', 'run')

class PendulumEnvCreator(EnvCreator):
    def __init__(self):
        geom_names = [
            'pole',
            'floor'
        ]
        include_fn = lambda c, p: True  # include all joints
        super(PendulumEnvCreator, self).__init__('Pendulum', [0], {0: geom_names}, use_root=False, include_fn=include_fn)
    
    def _create_env(self, key):
        return suite.load('pendulum', 'swingup')

class AcrobotEnvCreator(EnvCreator):
    def __init__(self):
        geom_names = [
            'upper_arm',
            'lower_arm',
            'floor'
        ]
        super(AcrobotEnvCreator, self).__init__('Acrobot', [0], {0: geom_names}, use_root=False)

    def _create_env(self, key):
        return suite.load('acrobot', 'swingup')