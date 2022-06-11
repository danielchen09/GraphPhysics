from torch import nn

from actor import Actor
from critic import Critic
from noise import OUNoise
import ddpg_cfg

class DDPGAgent:
    def __init__(self, obs_dim, action_spec):
        self.action_spec = action_spec
        act_dim = action_spec.shape[0]
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim, act_dim)
        self.noise = OUNoise(act_dim)

        self.steps = 0

    def select_action(self, state):
        if self.steps < ddpg_cfg.INIT_STEPS:
            
    
    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        return self