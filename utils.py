import copy
from torch import nn
import numpy as np
import torch
import pickle
from datetime import datetime
import networkx as nx
import mujoco
import math
import glob
import os

import config

def get_date():
    return datetime.now().strftime("%m%d%H%M%S")

def load_latest_checkpoint():
    checkpoints = glob.glob(f'{config.CHECKPOINT_DIR}/*.ckpt')
    checkpoints.sort(key=lambda x: os.path.getmtime(x))
    name = checkpoints[-1].split('/')[-1]
    print(f'using checkpoint: {name}')
    return name

def save_checkpoint(model, optimizer, scheduler, logger, norm_in, norm_out, name=None):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'norm_in': norm_in.cpu(),
        'norm_out': norm_out.cpu(),
        'epoch': logger.epochs,
        'step': logger.steps,
        'lr': scheduler.get_last_lr()[0],
        'log_dir': logger.log_dir
    }
    if name is None:
        name = logger.name + '.ckpt'
    path = f'{config.CHECKPOINT_DIR}/{name}'
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = ckpt['lr'][0] if isinstance(ckpt['lr'], (tuple, list)) else ckpt['lr']
    ckpt['norm_in'].to(config.DEVICE)
    ckpt['norm_out'].to(config.DEVICE)
    return ckpt

def m2q(mat):
    q = np.zeros((4, 1))
    mujoco.mju_mat2Quat(q, mat)
    return q.reshape(-1)

def one_hot(x, size):
    if type(x) is not np.ndarray:
        x = np.array(x).reshape(-1)
    a = np.zeros((x.shape[0], size))
    a[np.arange(x.shape[0]), x] = 1
    return a

def get_obj_vel(env, objids, name):
    res = np.zeros((6, 1))
    mujoco.mj_objectVelocity(env.physics.model._model, env.physics.data._data, mujoco.mjtObj.mjOBJ_GEOM, objids[name], res, 0)
    return res.reshape(-1)

def get_body_name_and_id(env):
    names = env.physics.model.names.decode('utf-8')
    body_names = [names[i:names.index('\x00', i)] for i in env.physics.model.name_bodyadr]
    objids = {name: mujoco.mj_name2id(env.physics.model._model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names}
    return body_names, objids

def get_edge_order(env, geom_names):  
    body_names, name2id = get_body_name_and_id(env) 
    id2name = {v: k for k, v in name2id.items()}
    body2geom = {id2name[env.physics.named.model.geom_bodyid[name]]: name for name in geom_names}
 
    parent = env.physics.named.model.body_parentid
    joint_body = env.physics.named.model.jnt_bodyid
    njnt = env.physics.model.njnt
    edge_order = {}
    for order, i in enumerate(range(3 if njnt > 3 else 0, njnt)):
        u = body2geom[id2name[joint_body[i]]]
        v = body2geom[id2name[parent[joint_body[i]]]]
        edge_order[u, v] = order
        edge_order[v, u] = order
    return edge_order

def get_geom_sizes(env, geom_names):
    return env.physics.named.model.geom_size[geom_names]

def get_graph(env, geom_names):
    body_names, objids = get_body_name_and_id(env)
    id2name = {v: k for k, v in objids.items()}
    # breakpoint()
    geom2body = {name: id2name[env.physics.named.model.geom_bodyid[name]] for name in geom_names}
    body2geom = {v: k for k, v in geom2body.items()}
    edges = []
    for g_name in geom_names:
        if g_name in ['ground', 'floor']:
            continue
        c_name = geom2body[g_name]
        c_id = objids[c_name]
        p_id = env.physics.named.model.body_parentid[c_id]
        if p_id not in id2name or p_id == 0:
            continue
        p_name = id2name[p_id]
        edges.append([g_name, body2geom[p_name]])
    G = nx.from_edgelist(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.add_node('ground')
    for node in G.nodes:
        if node == 'ground':
            continue
        G.add_edge('ground', node)
    return G.to_directed()

def get_state(env, geom_names):
    objids = {name: mujoco.mj_name2id(env.physics.model._model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in geom_names}

    return {
        'position': {name: copy.deepcopy(env.physics.named.data.geom_xpos[name]) for name in geom_names},
        'velocity': {name: get_obj_vel(env, objids, name) for name in geom_names},
        'rotation': {name: copy.deepcopy(m2q(env.physics.named.data.geom_xmat[name])) for name in geom_names},
        'size': {name: copy.deepcopy(env.physics.named.model.geom_size[name]) for name in geom_names}
    }

def get_node_attrs(state, geom_names, attr_names):
    node_attrs = []
    for name in geom_names:
        row = []
        for attr_name in attr_names:
            row.append(state[attr_name][name])
        node_attrs.append(np.hstack(row))
    return torch.tensor(node_attrs)

def get_dynamic_node_attrs(obs, geom_names):
    return get_node_attrs(obs, geom_names, ['position', 'rotation', 'velocity'])

def get_dynamic_edge_attrs(action, edges, edge_order):
    edge_attrs = []
    for edge in edges:
        if edge in edge_order:
            edge_attrs.append(np.array([action[edge_order[edge]]]))
        else:
            edge_attrs.append(np.array([0]))
    return torch.tensor(edge_attrs)

def get_static_state(env, geom_names):
    model = env.physics.named.model
    body_names, objids = get_body_name_and_id(env)
    id2name = {v: k for k, v in objids.items()}
    geom2body = {name: id2name[env.physics.named.model.geom_bodyid[name]] for name in geom_names}
    return {
        'mass': {name: copy.deepcopy(model.body_mass[geom2body[name]].reshape(-1)) for name in geom_names},
        'size': {name: copy.deepcopy(model.geom_size[name]) for name in geom_names},
        'type': {name: one_hot(model.geom_type[name], 8).reshape(-1) for name in geom_names},
        'body_pos': {name: copy.deepcopy(model.body_pos[geom2body[name]]) for name in geom_names},
        'body_quat': {name: copy.deepcopy(model.body_quat[geom2body[name]]) for name in geom_names},
        'geom_pos': {name: copy.deepcopy(model.geom_pos[name]) for name in geom_names},
        'geom_quat': {name: copy.deepcopy(model.geom_quat[name]) for name in geom_names},
        'ipos': {name: copy.deepcopy(model.body_ipos[geom2body[name]]) for name in geom_names},
        'iquat': {name: copy.deepcopy(model.body_iquat[geom2body[name]]) for name in geom_names}
    }

def get_static_node_attrs(env, geom_names):
    attr_names = ['mass', 'size', 'type', 'body_pos', 'body_quat', 'geom_pos', 'geom_quat', 'ipos', 'iquat']
    state = get_static_state(env, geom_names)
    return get_node_attrs(state, geom_names, attr_names)

def add_noise(x, scale=0.1):
    noise = np.random.normal(scale=scale, size=x.shape)
    return x + noise

def center_attrs(node_attrs, indicies=(0, 3), mean=None):
    attr = node_attrs[:, indicies[0]:indicies[1]]
    if mean is None:
        mean = attr.mean(dim=0)
    node_attrs[:, indicies[0]:indicies[1]] -= mean
    return mean.reshape(1, -1).repeat(node_attrs.shape[0], 1)

def normalize(x, mean, std):
    if config.USE_NORMALIZATION:
        return (x - mean) / (std + 1e-6)
    return x

def invnormalize(x, mean, std):
    if config.USE_NORMALIZATION:
        return x * (std + 1e-6) + mean
    return x

def to_cpu(x):
    if type(x) is torch.Tensor:
        if x.is_cuda:
            x = x.cpu()
        x = x.detach()
    return x

def make_nn(features, last_act=None):
    layers = []
    for i in range(len(features) - 1):
        layers.append(nn.Linear(features[i], features[i + 1]))
        if i < len(features) - 2:
            layers.append(nn.ReLU())
    if last_act is not None:
        layers.apend(last_act())
    return nn.Sequential(*layers)

def save_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

