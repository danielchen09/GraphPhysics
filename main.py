from model import ForwardModel
from dataset import SwimmerDataset
from graphs import Graph
from utils import *
from torch.utils.data import DataLoader
import dm_control.suite.swimmer as swimmer
from torch import nn
from dm_control import suite
import mujoco
import math



random_state = np.random.RandomState(42)
env = suite.load('cheetah', 'run', task_kwargs={'random': random_state})
spec = env.action_spec()
action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
o = env.reset()
env.step(action)
names = env.physics.model.names.decode('utf-8')
body_names = [names[i:names.index('\x00', i)] for i in env.physics.model.name_bodyadr]
objid = {name: mujoco.mj_name2id(env.physics.model._model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names}
def get_obj_vel(name):
    res = np.zeros((6, 1))
    mujoco.mj_objectVelocity(env.physics.model._model, env.physics.data._data, mujoco.mjtObj.mjOBJ_BODY, objid[name], res, 0)
    return res.reshape(-1)
print({name: get_obj_vel(name) for name in body_names})
print({i: env.physics.named.data.xpos[i] for i in body_names})
print({i: q2e(*env.physics.named.data.xquat[i]) for i in body_names})
print(env.physics.named.data.ctrl)
print(env.physics.named.data.qpos)
breakpoint()

# env = suite.load('cheetah', 'run')
# g = get_graph(env)
# print(g.nodes(data=True))
# print(g.edges(data=True))
# breakpoint()