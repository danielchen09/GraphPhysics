from torch import nn

import ddpg_cfg
from utils import *


class Actor(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=ddpg_cfg.ACTOR_HIDDEN_FEATURES):
        super(Actor, self).__init__()

        features = [in_features] + hidden_features + [out_features]
        self.net = make_nn(features, last_act=nn.Tanh)
    
    def forward(self, s):
        return self.net(s)
