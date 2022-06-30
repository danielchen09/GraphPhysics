from bdb import Breakpoint
from numpy import outer
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from torch import nn
from tqdm import tqdm
import dm_control.suite.swimmer as swimmer
import glob
import os

import config
from model import ConstantModel, ForwardModel, GCNForwardModel
from dataset import MujocoDataset
from graphs import Graph
from env import AcrobotEnvCreator, CheetahEnvCreator, CompositeEnvCreator, PendulumEnvCreator, SwimmerEnvCreator, WalkerEnvCreator
from render import generate_video, fig2array, Renderer
from utils import *
from logger import Logger
from normalizer import GraphNormalizer, Normalizer
import argparse

def evaluate_rollout(model, norm_in, norm_out, env_creators=None):
    model.eval()
    predictions = []
    actuals = []
    inputs = []
    errors = []
    infos = []

    run = MujocoDataset(CompositeEnvCreator(env_creators=env_creators), n_runs=1, n_steps=20, noise=config.NOISE if config.DEBUG else 0, save=False)
    dl = DataLoader(run, batch_size=1)
    with torch.no_grad():
        last_obs = run[0][0].node_attrs
        n_geoms = last_obs.shape[0]
        for graph, static_graph, y_old, y_new, info in dl:
            y_old, y_new, center = y_old.x, y_new.x, info['center']
            graph.x = last_obs
            prediction = model.predict(graph, norm_in, norm_out, static_graph=static_graph)
            last_obs = copy.deepcopy(prediction)

            center_attrs(prediction, mean=-center)
            center_attrs(y_new, mean=-center)
            center_attrs(y_old, mean=-center)

            predictions.append(prediction)
            actuals.append(y_new)
            inputs.append(y_old)
            infos.append(info['env_key'][0])
            errors.append(torch.abs(prediction - y_new).mean()) 
    return np.sum(errors) / n_geoms, (predictions, actuals, inputs, run.env_creator, infos)

def render_rollout(predictions, actuals, inputs, env_creator, infos, save_path='result.mp4', n_links=config.N_LINKS):
    renderer = Renderer()
    frames = []
    for prediction, actual, inp, env_key in tqdm(zip(predictions, actuals, inputs, infos), desc='generating video'):
        frame = []
        frame.append(renderer.render(prediction, env_creator, env_key))
        frame.append(renderer.render(actual, env_creator, env_key))
        if config.DEBUG:
            frame.append(renderer.render(inp))
        frames.append(np.hstack(frame))
    generate_video(frames, name=save_path)

def train(ds, model, optimizer, train_loader, loss_fn, norm_in, norm_out, logger, scheduler):
    step = 0

    best_error = {env_creator.name: -1 for env_creator in ds.env_creator.env_creators}

    for epoch in range(config.MAX_EPOCH):
        for x, static_graph, y_old, y_new, info in tqdm(train_loader, desc=f'epoch {epoch}/{config.MAX_EPOCH}'):
            y_old, y_new = y_old.x, y_new.x
            model.train()
            graph_old = x.to(config.DEVICE)
            y = y_new - y_old
            y = y.to(config.DEVICE)

            norm_out.update(y)
            y = norm_out.normalize(y)

            norm_in.update(graph_old)
            norm_in.normalize(graph_old)
            
            if config.USE_STATIC_ATTRS:
                graph_old.concat(static_graph)

            optimizer.zero_grad()
            y_pred = model(graph_old)
            loss = loss_fn(y_pred.float(), y.float())
            loss.backward()
            optimizer.step()

            logger.log_loss('training_loss', loss.item())

            if step % config.VAL_INTERVAL == 0:
                for i, env_creator in enumerate(ds.env_creator.env_creators):
                    error, rollout = evaluate_rollout(model, norm_in, norm_out, [env_creator])
                    ds.env_creator.update_error(f"{i},{rollout[-1][0].split(',')[1]}", error)
                    logger.log_loss('val error', error)
                    if error < best_error[env_creator.name] or best_error[env_creator.name] == -1:
                        print(f'best error achieved at step {step}: {error} for {env_creator.name} environment')
                        ds.env_creator.print_probs()
                        ds.env_creator.print_errors()
                        best_error[env_creator.name] = error
                        render_rollout(*rollout, f'results/{env_creator.name}.mp4')
                        save_checkpoint(model, optimizer, scheduler, logger, norm_in, norm_out, name='forward_model.mdl')
            
            if step % config.SAVE_INTERVAL == 0:
                save_checkpoint(model, optimizer, scheduler, logger, norm_in, norm_out)

            step += 1
            logger.step()
            scheduler.step()
        logger.step_epoch()

def run_train(checkpoint=None):
    ds = MujocoDataset(CompositeEnvCreator(), dataset_path=config.DATASET_PATH, n_runs=config.N_RUNS, n_steps=config.N_STEPS, load_from_path=True, save=True, noise=config.NOISE, shuffle=True)
    sample_graph, static_graph, _, _, info = ds[0]
    global_features = static_graph.global_attrs.shape[-1]
    node_features = sample_graph.x.shape[-1] + static_graph.x.shape[-1]
    edge_features = sample_graph.edge_attr.shape[-1] + static_graph.edge_attr.shape[-1]
    out_features = sample_graph.x.shape[-1]
    model = ForwardModel(global_features, node_features, edge_features, out_features).to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    train_loader = DataLoader(ds, batch_size=config.BATCH_SIZE)
    if checkpoint is not None:
        ckpt = load_checkpoint(f'{config.CHECKPOINT_DIR}/{checkpoint}', model, optimizer)
    norm_in = GraphNormalizer() if checkpoint is None else ckpt['norm_in']
    norm_out = Normalizer() if checkpoint is None else ckpt['norm_out']

    def get_loss_fn():
        mse = nn.MSELoss()
        def loss_fn_no_split(y_pred, y):
            return mse(y_pred, y)
        return loss_fn_no_split

    logger = Logger(model=model) if checkpoint is None else Logger(steps=ckpt['step'], model=model)
    scheduler = StepLR(optimizer, config.LR_DECAY_STEP, gamma=config.LR_DECAY)
    train(ds, model, optimizer, train_loader, get_loss_fn(), norm_in, norm_out, logger=logger, scheduler=scheduler)


def test(name='', save_path='test_result.mp4', env='', trials=20):
    env_creators = {
        'swimmer': [SwimmerEnvCreator(n_links=6)],
        'cheetah': [CheetahEnvCreator()],
        'acrobot': [AcrobotEnvCreator()],
        'pendulum': [PendulumEnvCreator()],
        '': None
    }
    constant = name == 'constant'
    if len(name) == 0 or name == 'constant':
        name = load_latest_checkpoint()

    ds = MujocoDataset(CompositeEnvCreator(), n_runs=1, n_steps=1)
    sample_graph, static_graph, _, _, info = ds[0]
    global_features = static_graph.global_attrs.shape[-1]
    node_features = sample_graph.x.shape[-1] + static_graph.x.shape[-1]
    edge_features = sample_graph.edge_attr.shape[-1] + static_graph.edge_attr.shape[-1]
    out_features = sample_graph.x.shape[-1]

    model = ForwardModel(global_features, node_features, edge_features, out_features).to(config.DEVICE)
    ckpt = load_checkpoint(f'{config.CHECKPOINT_DIR}/{name}', model)
    if constant:
        model = ConstantModel()
    min_error = 999
    min_rollout = None
    for _ in range(trials):
        e, r = evaluate_rollout(model, ckpt['norm_in'], ckpt['norm_out'], env_creators=env_creators[env])
        if e < min_error:
            min_error = e
            min_rollout = r
    print(min_error)
    render_rollout(*min_rollout, save_path=save_path)
    return min_error
