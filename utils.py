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

def save_checkpoint(model, optimizer, scheduler, logger, norm_in, norm_out, name=None):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'norm_in': norm_in,
        'norm_out': norm_out,
        'epoch': logger.epochs,
        'step': logger.steps,
        'lr': scheduler.get_last_lr(),
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
            param_group['lr'] = ckpt['lr']
    return ckpt

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
        p_id = env.physics.named.model.body_parentid[c_id]
        if p_id not in id2name:
            continue
        p_name = id2name[p_id]
        edges.append([c_name, p_name])
    G = nx.from_edgelist(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G.to_directed()

def get_state(env):
    body_names, objids = get_body_name_and_id(env)
    id2name = {v: k for k, v in objids.items()}
    parent = env.physics.named.model.body_parentid
    joints = env.physics.named.data.qpos
    joint_body = env.physics.named.model.jnt_bodyid
    njnt = env.physics.model.njnt

    return {
        'position': {name: copy.deepcopy(env.physics.named.data.xpos[name]) for name in body_names},
        'velocity': {name: get_obj_vel(env, objids, name) for name in body_names},
        'rotation': {name: copy.deepcopy(env.physics.named.data.xquat[name]) for name in body_names},
        'joints': {(id2name[joint_body[jid]], id2name[parent[joint_body[jid]]]): joints[jid] for jid in range(3, njnt)},
    }

def get_static_state(env):
    body_names, objids = get_body_name_and_id(env)
    model = env.physics.named.model
    return {
        'mass': {name: copy.deepcopy(model.body_mass[name].reshape(-1, 1)) for name in body_names},
        'pos': {name: copy.deepcopy(model.body_pos[name]) for name in body_names},
        'quat': {name: copy.deepcopy(model.body_quat[name]) for name in body_names},
        'inertia': {name: copy.deepcopy(model.body_intertia[name]) for name in body_names},
        'ipos': {name: copy.deepcopy(model.body_ipos[name]) for name in body_names},
        'iquat': {name: copy.deepcopy(model.body_iquat[name]) for name in body_names}
    }

def add_noise(x, scale=0.1):
    noise = np.random.normal(scale=scale, size=x.shape)
    return x + noise

def center_attrs(node_attrs, indicies, mean=None):
    attr = node_attrs[:, indicies[0]:indicies[1]]
    if mean is None:
        mean = attr.mean(dim=0)
    node_attrs[:, indicies[0]:indicies[1]] -= mean
    return mean.reshape(1, -1).repeat(node_attrs.shape[0], 1)

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

def get_dynamic_node_attrs(obs, body_names):
    node_attrs = []
    for name in body_names:
        node_attrs.append(np.hstack((obs['position'][name], obs['rotation'][name], obs['velocity'][name])))
    return torch.tensor(node_attrs)

def get_dynamic_edge_attrs(action, edges, edge_order):
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

def save_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

