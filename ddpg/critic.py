from torch import nn

import ddpg_cfg
from utils import *

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_features=ddpg_cfg.CRITIC_HIDDEN_FEATURES):
        super(Critic, self).__init__()

        features = [obs_dim + act_dim] + hidden_features + [1]
        self.net = make_nn(features)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)
