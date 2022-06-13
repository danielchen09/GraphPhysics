from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import dm_control.suite.swimmer as swimmer
import glob
import os

import config
from model import ForwardModel, GCNForwardModel
from dataset import MujocoDataset
from graphs import Graph
from env import CheetahEnvCreator, CompositeEnvCreator, SwimmerEnvCreator, WalkerEnvCreator
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

    env_creators = [WalkerEnvCreator()]
    
    run = MujocoDataset(CompositeEnvCreator(env_creators=env_creators), n_runs=1, n_steps=20, noise=config.NOISE if config.DEBUG else 0, save=False)
    with torch.no_grad():
        last_obs = run[0][0].node_attrs
        n_geoms = last_obs.shape[0]
        for graph, y_old, y_new, info in run:
            center = info['center']
            graph.update(None, last_obs, None)
            prediction = model.predict(graph, norm_in, norm_out, static_node_attrs=info['static_node_attrs'])
            last_obs = copy.deepcopy(prediction)

            center_attrs(prediction, mean=-center)
            center_attrs(y_new, mean=-center)
            center_attrs(y_old, mean=-center)

            predictions.append(prediction)
            actuals.append(y_new)
            inputs.append(y_old)
            infos.append(info['env_key'])
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

def train(model, optimizer, train_loader, loss_fn, norm_in, norm_out, logger, scheduler):
    best_error = -1
    step = 0

    renderer = Renderer()
    for epoch in range(config.MAX_EPOCH):
        for x, y_old, y_new, center, static_node_attrs in tqdm(train_loader, desc=f'epoch {epoch}/{config.MAX_EPOCH}'):
            model.train()
            graph_old = x[0].to(config.DEVICE)
            y = y_new - y_old
            y = y.to(config.DEVICE)

            norm_out.update(y)
            y = norm_out.normalize(y)

            norm_in.update(graph_old)
            graph_old = norm_in.normalize(graph_old)
            if config.USE_STATIC_ATTRS:
                static_node_attrs = static_node_attrs.to(config.DEVICE)
                graph_old.concat_node(static_node_attrs)

            optimizer.zero_grad()
            y_pred = model(graph_old)
            loss = loss_fn(y_pred.float(), y.float())
            loss.backward()
            optimizer.step()

            logger.log_loss('training_loss', loss.item())

            if step % config.VAL_INTERVAL == 0:
                error, rollout = evaluate_rollout(model, norm_in, norm_out)
                logger.log_loss('val error', error)
                if error < best_error or best_error == -1:
                    print(f'best error achieved at step {step}: {error}')
                    best_error = error
                    render_rollout(*rollout)
                    save_checkpoint(model, optimizer, scheduler, logger, norm_in, norm_out, name='forward_model.mdl')
            
            if step % config.SAVE_INTERVAL == 0:
                save_checkpoint(model, optimizer, scheduler, logger, norm_in, norm_out)

            step += 1
            logger.step()
            scheduler.step()
        logger.step_epoch()

def run_train(checkpoint=None):
    ds = MujocoDataset(WalkerEnvCreator(), dataset_path=config.DATASET_PATH, n_runs=config.N_RUNS, n_steps=config.N_STEPS, load_from_path=True, save=True, noise=config.NOISE, shuffle=True)
    sample_graph, _, _, info = ds[0]
    in_shape = sample_graph.node_attrs.shape[1]
    if config.USE_STATIC_ATTRS:
        in_shape += info['static_node_attrs'].shape[-1]
    model = GCNForwardModel(in_shape, sample_graph.node_attrs.shape[1]).to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    train_loader = DataLoader(ds, batch_size=config.BATCH_SIZE, collate_fn=ds.get_collate_fn(), shuffle=False)
    if checkpoint is not None:
        ckpt = load_checkpoint(f'{config.CHECKPOINT_DIR}/{checkpoint}', model, optimizer)
    norm_in = GraphNormalizer() if checkpoint is None else ckpt['norm_in']
    norm_out = Normalizer() if checkpoint is None else ckpt['norm_out']

    def get_loss_fn():
        # 3p4r6v
        mse = nn.MSELoss()
        def loss_fn_no_split(y_pred, y):
            return mse(y_pred, y)
        return loss_fn_no_split

    logger = Logger() if checkpoint is None else Logger(steps=ckpt['step'])
    scheduler = StepLR(optimizer, config.LR_DECAY_STEP, gamma=config.LR_DECAY)
    train(model, optimizer, train_loader, get_loss_fn(), norm_in, norm_out, logger=logger, scheduler=scheduler)


def test(name='', save_path='test_result.mp4', env=''):
    env_creators = {
        'swimmer': [SwimmerEnvCreator()],
        'cheetah': [CheetahEnvCreator()],
        'walker': [WalkerEnvCreator()],
        '': None
    }
    if len(name) == 0:
        name = load_latest_checkpoint()
    in_shape = 13 + (33 if config.USE_STATIC_ATTRS else 0)
    model = GCNForwardModel(in_shape, 13).to(config.DEVICE)
    ckpt = load_checkpoint(f'{config.CHECKPOINT_DIR}/{name}', model)
    min_error = 999
    min_rollout = None
    for _ in range(20):
        e, r = evaluate_rollout(model, ckpt['norm_in'], ckpt['norm_out'], env_creators=env_creators[env])
        if e < min_error:
            min_error = e
            min_rollout = r
    print(min_error)
    render_rollout(*min_rollout, save_path=save_path)
