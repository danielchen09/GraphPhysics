from re import M
import numpy as np
from pyrsistent import b

import ddpg_cfg

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=ddpg_cfg.BUFFER_CAPACITY, batch_size=ddpg_cfg.BATCH_SIZE):
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity), dtype=np.float32)
        self.capacity = capacity
        self.batch_size = batch_size
        self.ptr, self.size = 0, 0
    
    def store(self, obs, act, next_obs, reward):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.next_obs_buf[self.ptr] = next_obs
        self.reward_buf[self.ptr] = reward
        self.ptr = (self.ptr + 1) % self.size
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return {
            'obs': self.obs_buf[idxs],
            'act': self.act_buf[idxs],
            'next_obs': self.next_obs_buf[idxs],
            'reward': self.reward_buf[idxs]
        }
    
    def __len__(self):
        return self.size
    