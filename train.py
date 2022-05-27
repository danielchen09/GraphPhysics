from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import dm_control.suite.swimmer as swimmer

import config
from model import ForwardModel, GCNForwardModel
from dataset import MujocoDataset
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
    inputs = []
    errors = []
    run = MujocoDataset(config.GET_ENVIRONMENT(), n_runs=1, n_steps=20, noise=config.NOISE if config.DEBUG else 0, save=False)
    with torch.no_grad():
        last_obs = run[0][0].node_attrs
        for graph, y_old, y_new, center in run:
            graph.update(None, last_obs, None)
            prediction = model.predict(graph, norm_in, norm_out)
            last_obs = copy.deepcopy(prediction)

            center_attrs(prediction, (0, 3), -center)
            center_attrs(y_new, (0, 3), -center)
            center_attrs(y_old, (0, 3), -center)

            predictions.append(prediction)
            actuals.append(y_new)
            inputs.append(y_old)
            errors.append(torch.abs(prediction - y_new).mean())
    return np.sum(errors), (predictions, actuals, inputs)

def render_rollout(predictions, actuals, inputs, save_path='result.mp4', n_links=config.N_LINKS):
    frames = []
    for prediction, actual, inputs in tqdm(zip(predictions, actuals, inputs), desc='generating video'):
        frame = []
        fig = draw(n_links, prediction, title='predicted')
        buf_pred = fig2array(fig)
        frame.append(buf_pred)
        fig = draw(n_links, actual, title='actual')
        buf_actual = fig2array(fig)
        frame.append(buf_actual)
        if config.DEBUG:
            fig = draw(n_links, inputs, title='input')
            frame.append(fig2array(fig))
        frames.append(np.hstack(frame))
    generate_video(frames, name=save_path)

def train(model, optimizer, train_loader, loss_fn, norm_in, norm_out, logger, scheduler):
    best_error = -1
    step = 0

    for epoch in range(config.MAX_EPOCH):
        for x, y_old, y_new, center in tqdm(train_loader, desc=f'epoch {epoch}/{config.MAX_EPOCH}'):
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
                    save_checkpoint(model, optimizer, scheduler, logger, norm_in, norm_out, name='forward_model.mdl')
            
            if step % config.SAVE_INTERVAL == 0:
                save_checkpoint(model, optimizer, scheduler, logger, norm_in, norm_out)

            step += 1
            logger.step()
            scheduler.step()
        logger.step_epoch()

def train_swimmer(checkpoint=None):
    ds = MujocoDataset(config.GET_ENVIRONMENT(), n_runs=config.N_RUNS, n_steps=config.N_STEPS, load_from_path=True, save=True, noise=config.NOISE)
    sample_graph, _, _, _ = ds[0]
    # model = ForwardModel(0, sample_graph.node_attrs.shape[1], sample_graph.edge_attrs.shape[1]).to(config.DEVICE)
    model = GCNForwardModel(sample_graph.node_attrs.shape[1], sample_graph.node_attrs.shape[1]).to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    train_loader = DataLoader(ds, batch_size=config.BATCH_SIZE, collate_fn=ds.get_collate_fn(), shuffle=True)
    if checkpoint is not None:
        ckpt = load_checkpoint(checkpoint, model, optimizer)
    norm_in = GraphNormalizer() if checkpoint is None else ckpt['norm_in']
    norm_out = Normalizer() if checkpoint is None else ckpt['norm_out']

    def get_loss_fn():
        # 3p4r6v
        mse = nn.MSELoss()
        def angle_loss(q1, q2):
            return torch.mean(torch.sin(torch.sum(q1 * q2, dim=-1)) ** 2)

        def loss_fn(y_pred, y):
            y_pred_c = torch.cat([y_pred[:, :3], y_pred[:, 7:]], dim=-1)
            y_pred_a = y_pred[:, 3:7]
            y_c = torch.cat([y[:, :3], y[:, 7:]], dim=-1)
            y_a = y[:, 3:7]
            return mse(y_pred_c, y_c) + config.ROTATION_LOSS_WEIGHT * angle_loss(y_pred_a, y_a)
        
        def loss_fn_no_split(y_pred, y):
            return mse(y_pred, y)
        return loss_fn_no_split

    logger = Logger() if checkpoint is None else Logger(log_name=ckpt['log_dir'], steps=ckpt['step'])
    scheduler = StepLR(optimizer, config.LR_DECAY_STEP, gamma=config.LR_DECAY)
    train(model, optimizer, train_loader, get_loss_fn(), norm_in, norm_out, logger=logger, scheduler=scheduler)


def test(name='forward_model.mdl', save_path='test_result.mp4'):
    ds = MujocoDataset(config.GET_ENVIRONMENT(), n_runs=1000, n_steps=100, load_from_path=True, save=True)
    sample_graph, _, _ = ds[0]
    model = GCNForwardModel(sample_graph.node_attrs.shape[1], sample_graph.node_attrs.shape[1]).to(config.DEVICE)
    ckpt = load_checkpoint(f'{config.CHECKPOINT_DIR}/{name}', model)
    error, rollout = evaluate_rollout(model, ckpt['norm_in'], ckpt['norm_out'])
    print(error)
    render_rollout(*rollout, save_path=save_path)

