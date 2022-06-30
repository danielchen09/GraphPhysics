from datetime import datetime
from tensorboard import program
import tensorflow as tf
import wandb

from utils import *

class WandbLogger:
    def __init__(self, steps=0, log_name=None, model=None):
        self.name = get_date() if log_name is None else log_name.split('.')[0]
        wandb.init(project="Graph Physics", entity="danielchen09", name=self.name)
        if model is not None:
            wandb.watch(model)
        self.steps = steps
        self.epochs = 0
        self.logs = []
    
    def log_drp(self, env_creator):
        data = [[key, prob] for key, prob in zip(env_creator.keys, env_creator.get_probs())]
        table = wandb.Table(data=data, columns=['env', 'weight'])
        wandb.log({'dynamic replay buffer weights': wandb.plot.bar(table, 'env', 'weight', title='dynamic replay buffer weights')})

    def log_loss(self, name, loss):
        self.logs.append((name, loss))
        if len(self.logs) >= 100:
            for n, l in self.logs:
                wandb.log({n: l})
            self.logs = []

    def step(self):
        self.steps += 1
    
    def step_epoch(self):
        self.epochs += 1

class Logger:
    def __init__(self, port=6006, log_name=None, steps=0, model=None):
        self.name = get_date() if log_name is None else log_name.split('.')[0]
        self.log_dir = './logs/' + self.name
        self.writer = tf.summary.create_file_writer(self.log_dir, flush_millis=1000)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir, '--reload_interval', '1', '--port', f'{port}'])
        self.url = tb.launch()
        print(f"Tensorflow listening on {self.url} log_dir: {self.log_dir}")
        self.steps = steps
        self.epochs = 0

    def log_loss(self, name, value):
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=self.steps)

    def log_drp(self, *args, **kwargs):
        pass

    def step(self):
        self.steps += 1
    
    def step_epoch(self):
        self.epochs += 1