from model import ForwardModel
from graphs import Graph
from utils import *
from torch.utils.data import DataLoader
import dm_control.suite.swimmer as swimmer
from torch import nn
from dm_control import suite
import mujoco
import math
import argparse
from train import run_train, test
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--linux', type=str, default='f')
    parser.add_argument('--save_path', type=str, default='test_result.mp4')
    parser.add_argument('--epochs', type=int, default=config.MAX_EPOCH)
    parser.add_argument('--env', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()
    if args.debug:
        print('\n\ndebug mode on\n\n')
        config.DEBUG = True
    if args.device not in ['cpu', 'cuda:0']:
        print('Invalid device, options: --device=[cpu|cuda:0]')
        return
    if len(args.dataset) > 0:
        config.DATASET_PATH = ('' if args.dataset.startswith('datasets/') else 'datasets/') + args.dataset
        
    config.DEVICE = args.device
    config.LINUX = args.linux.lower() in ['yes', 'y', 'true', 't']
    config.MAX_EPOCH = args.epochs

    if args.mode == 'train':
        checkpoint = args.checkpoint if args.checkpoint != '' else None
        if checkpoint == 'latest':
            checkpoint = load_latest_checkpoint()
        run_train(checkpoint)
    elif args.mode == 'test':
        test(args.model, args.save_path, env=args.env)
    else:
        print('Invalid mode, options: --mode=[train|test]')

if __name__ == '__main__':
    main()