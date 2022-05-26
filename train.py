from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import dm_control.suite.swimmer as swimmer

import config
from model import ForwardModel, GCNForwardModel
from dataset import MujocoDataset, SwimmerDataset
from graphs import Graph
from render import generate_video, draw, fig2array
from utils import *
from logger import Logger
from normalizer import GraphNormalizer, Normalizer
import argparse

def evaluate_rollout(model, norm_in, norm_out):
    model.eval()
    predictions = []
    actuals = []
    errors = []
    run = MujocoDataset(swimmer.swimmer(config.N_LINKS), n_runs=1, n_steps=20, shuffle=False, noise=0, save=False)
    with torch.no_grad():
        last_obs = run[0][0].node_attrs
        for graph, y_old, y_new in run:
            graph.update(None, last_obs, None)
            prediction = model.predict(graph, norm_in, norm_out)
            last_obs = prediction
            predictions.append(prediction)
            actuals.append(y_new)
            errors.append(torch.abs(prediction - y_new).mean())
    return np.sum(errors), (predictions, actuals)

def render_rollout(predictions, actuals, save_path='result.mp4', n_links=config.N_LINKS):
    frames = []
    for prediction, actual in tqdm(zip(predictions, actuals), desc='generating video'):
        fig = draw(n_links, prediction, title='predicted')
        buf_pred = fig2array(fig)
        fig = draw(n_links, actual, title='actual')
        buf_actual = fig2array(fig)
        frames.append(np.hstack((buf_actual, buf_pred)))
    generate_video(frames, name=save_path)

def train(model, optimizer, train_loader, loss_fn, norm_in, norm_out, logger=None, scheduler=None):
    best_error = -1
    date = get_date()
    step = 0

    for epoch in range(config.MAX_EPOCH):
        for x, y_old, y_new in tqdm(train_loader, desc=f'epoch {epoch}/{config.MAX_EPOCH}'):
            model.train()
            graph_old = x[0].to(config.DEVICE)
            y = y_new - y_old
            y = y.to(config.DEVICE)

            norm_out.update(y)
            y = norm_out.normalize(y)

            norm_in.update(graph_old)
            graph_old = norm_in.normalize(graph_old)

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
                    save_checkpoint(model, optimizer, norm_in, norm_out, name='forward_model.mdl')
            
            if step % config.SAVE_INTERVAL == 0:
                save_checkpoint(model, optimizer, norm_in, norm_out, name=f'{date}.ckpt')

            step += 1
            logger.step()
            if scheduler is not None:
                scheduler.step()

def train_swimmer():
    ds = MujocoDataset(swimmer.swimmer(config.N_LINKS), n_runs=config.N_RUNS, n_steps=config.N_STEPS, load_from_path=True, save=True, noise=config.NOISE)
    sample_graph, _, _ = ds[0]
    # model = ForwardModel(0, sample_graph.node_attrs.shape[1], sample_graph.edge_attrs.shape[1]).to(config.DEVICE)
    model = GCNForwardModel(sample_graph.node_attrs.shape[1], sample_graph.node_attrs.shape[1]).to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    train_loader = DataLoader(ds, batch_size=config.BATCH_SIZE, collate_fn=ds.get_collate_fn(), shuffle=True)
    def get_loss_fn():
        # 3p4r6v
        mse = nn.MSELoss()
        def angle_loss(y_pred, y):
            return torch.mean(torch.sin(torch.sum(y_pred * y, dim=-1)) ** 2)

        def loss_fn(y_pred, y):
            y_pred_c = torch.cat([y_pred[:, :3], y_pred[:, 7:]], dim=-1)
            y_pred_a = y_pred[:, 3:7]
            y_c = torch.cat([y[:, :3], y[:, 7:]], dim=-1)
            y_a = y[:, 3:7]
            return mse(y_pred_c, y_c) + config.ROTATION_LOSS_WEIGHT * angle_loss(y_pred_a, y_a)
        return loss_fn

    logger = Logger()
    scheduler = StepLR(optimizer, config.LR_DECAY_STEP, gamma=config.LR_DECAY)
    train(model, optimizer, train_loader, get_loss_fn(), GraphNormalizer(), Normalizer(), logger=logger, scheduler=scheduler)


def test(name='forward_model.mdl', save_path='test_result.mp4'):
    ds = MujocoDataset(swimmer.swimmer(config.N_LINKS), n_runs=1000, n_steps=100, load_from_path=True, save=True)
    sample_graph, _, _ = ds[0]
    model = GCNForwardModel(sample_graph.node_attrs.shape[1], sample_graph.node_attrs.shape[1]).to(config.DEVICE)
    norm_in, norm_out = load_checkpoint(f'{config.CHECKPOINT_DIR}/{name}', model, None)
    error, rollout = evaluate_rollout(model, norm_in, norm_out)
    print(error)
    render_rollout(*rollout, save_path=save_path)

