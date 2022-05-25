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
import argparse
from train import train_swimmer, test

def main():
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    if args.debug:
        print('debug mode on')
        config.DEBUG = True
    if args.device not in ['cpu', 'cuda:0']:
        print('Invalid device, options: --device=[cpu|cuda:0]')
        return
    config.DEVICE = args.device

    if args.mode == 'train':
        train_swimmer()
    elif args.mode == 'test':
        test(args.model)
    else:
        print('Invalid mode, options: --mode=[train|test]')

if __name__ == '__main__':
    main()