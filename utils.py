import copy
from platform import node
from turtle import pos
from torch import nn
import numpy as np
import torch
import pickle
from datetime import datetime
import networkx as nx
import mujoco
import math

import config

def get_date():
    return datetime.now().strftime("%m%d%H%M%S")

def save_checkpoint(model, optimizer, norm_in, norm_out, name=''):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'norm_in': norm_in,
        'norm_out': norm_out
    }
    path = f'{config.CHECKPOINT_DIR}/{name}'
    if name == '':
        path = f'{config.CHECKPOINT_DIR}/checkpoint_{get_date}.ckpt'
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, lr=config.LEARNING_RATE):
    ckpt = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return ckpt['norm_in'], ckpt['norm_out']

def get_obs(time_step):
    return copy.deepcopy(time_step.observation)

def get_obj_vel(env, objids, name):
    res = np.zeros((6, 1))
    mujoco.mj_objectVelocity(env.physics.model._model, env.physics.data._data, mujoco.mjtObj.mjOBJ_BODY, objids[name], res, 0)
    return res.reshape(-1)

def get_body_name_and_id(env):
    names = env.physics.model.names.decode('utf-8')
    body_names = [names[i:names.index('\x00', i)] for i in env.physics.model.name_bodyadr]
    body_names.remove('world')
    objids = {name: mujoco.mj_name2id(env.physics.model._model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names}
    return body_names, objids

def get_edge_order(env):  
    body_names, name2id = get_body_name_and_id(env)  
    id2name = {v: k for k, v in name2id.items()}
    parent = env.physics.named.model.body_parentid
    joint_body = env.physics.named.model.jnt_bodyid
    njnt = env.physics.model.njnt
    edge_order = {}
    for order, i in enumerate(range(3, njnt)):
        edge_order[id2name[joint_body[i]], id2name[parent[joint_body[i]]]] = order
        edge_order[id2name[parent[joint_body[i]]], id2name[joint_body[i]]] = order
    return edge_order


def get_graph(env):
    body_names, objids = get_body_name_and_id(env)
    id2name = {v: k for k, v in objids.items()}
    edges = []
    for c_name in body_names:
        c_id = objids[c_name]
        p_name = id2name[env.physics.named.model.body_parentid[c_id]]
        edges.append([c_name, p_name])
    G = nx.from_edgelist(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def get_state(env):
    body_names, objids = get_body_name_and_id(env)
    id2name = {v: k for k, v in objids.items()}
    parent = env.physics.named.model.body_parentid
    joints = env.physics.named.data.qpos
    joint_body = env.physics.named.model.jnt_bodyid
    njnt = env.physics.model.njnt

    return {
        'position': {name: env.physics.named.data.xpos[name] for name in body_names},
        'velocity': {name: get_obj_vel(env, objids, name) for name in body_names},
        'rotation': {name: env.physics.named.data.xquat[name] for name in body_names},
        'joints': {(id2name[joint_body[jid]], id2name[parent[joint_body[jid]]]): joints[jid] for jid in range(3, njnt)}
    }

def get_pos(env, n_links):
    idx = ['head'] + [f'segment_{i}' for i in range(n_links - 1)]
    pos = []
    for i in idx:
        pos += [
            env.physics.named.data.xpos[i, 'x'], 
            env.physics.named.data.xpos[i, 'y'],
            np.arctan2(-env.physics.named.data.xmat[i][1], env.physics.named.data.xmat[i][0])
        ]
    return np.array(pos)

def add_noise_dict(x, scale=0.1):
    for k in x.keys():
        if type(x[k]) is dict:
            x[k] = add_noise_dict(x[k], scale)
        else:
            x[k] = add_noise_np(x[k], scale)
    return x

def add_noise_np(x, scale=0.1):
    noise = np.random.normal(scale=scale, size=x.shape)
    return x + noise

def normalize(x, mean, std):
    return (x - mean) / (std + 1e-6)

def invnormalize(x, mean, std):
    return x * (std + 1e-6) + mean

def to_cpu(x):
    if type(x) is torch.Tensor:
        if x.is_cuda:
            x = x.cpu()
        x = x.detach()
    return x

def make_nn(features):
    layers = []
    for i in range(len(features) - 1):
        layers.append(nn.Linear(features[i], features[i + 1]))
        if i < len(features) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def get_dynamic_node_attrs(obs):
    n_links = len(obs['joints']) + 1
    node_attrs = []
    for i in range(n_links):
        velocities = obs['body_velocities'][i * 3:(i + 1) * 3]
        positions = obs['pos'][i * 3:(i + 1) * 3]
        node_attrs.append(np.hstack((positions, velocities)))
    return torch.tensor(node_attrs)

def get_dynamic_edge_attrs(obs, action):
    n_links = len(obs['joints']) + 1
    edge_attrs = []
    for i in range(n_links - 1):
        edge_attrs.append(np.array([obs['joints'][i], action[i]]))
        edge_attrs.append(np.array([obs['joints'][i], action[i]]))
    return torch.tensor(edge_attrs)

def get_dynamic_node_attrs2(obs, body_names):
    node_attrs = []
    for name in body_names:
        node_attrs.append(np.hstack((obs['position'][name], obs['rotation'][name], obs['velocity'][name])))
    return torch.tensor(node_attrs)

def get_dynamic_edge_attrs2(action, edges, edge_order):
    edge_attrs = []
    for edge in edges:
        if edge not in edge_order:
            continue
        edge_attrs.append(np.array([action[edge_order[edge]]]))
    return torch.tensor(edge_attrs)

def get_static_global_attrs(env):
    opt = env.physics.model.opt
    return torch.tensor([
        opt.timestep,
        *opt.gravity,
        *opt.wind,
        *opt.magnetic,
        opt.density,
        opt.viscosity,
        opt.impratio,
        opt.o_margin,
        *opt.o_solref,
        *opt.o_solimp
    ])

def get_static_node_attrs(env):
    model = env.physics.model
    return np.hstack((
        model.body_mass.reshape(-1, 1),
        model.body_pos,
        model.body_quat,
        model.body_inertia,
        model.body_ipos,
        model.body_iquat
    ))[1:, :]

def get_static_edge_attrs(env):
    model = env.physics.model
    return 

def swimmer_params_from_data(obs, action):
    n_links = len(obs['joints']) + 1
    return nx.path_graph(n_links).to_directed(), None, get_dynamic_node_attrs(obs), get_dynamic_edge_attrs(obs, action)

def save_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def q2e(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    a = -np.pi - np.array([roll_x, pitch_y, yaw_z]) # in radians
    a = a[np.where(a % np.pi != 0)]
    return a[0] if len(a) > 0 else 0