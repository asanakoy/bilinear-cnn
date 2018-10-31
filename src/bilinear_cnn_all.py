#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune all layers for bilinear CNN.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./src/bilinear_cnn_all.py --base_lr 0.05 \
        --batch_size 64 --epochs 100 --weight_decay 5e-4
"""
from __future__ import print_function
from __future__ import division

import os

import torch
import torchvision

import cub200
from bcnn import BCNN
from trainer import BCNNTrainer


def main():
    import argparse

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser(
        description='Train bilinear CNN on CUB200.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, required=True,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        required=True, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int, required=True,
                        help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        required=True, help='Weight decay.')
    parser.add_argument('--model', dest='model', type=str, required=True,
                        help='Model name')
    parser.add_argument('--lr_scheduler', type=str,
                        default='reduce_on_plateau',
                        choices=['reduce_on_plateau',
                                 'fixed'],
                        help='LR scheduler')
    args = parser.parse_args()
    print(args)
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    project_root = os.popen('pwd').read().strip()
    print('Project root:', project_root)
    model_dir = os.path.join(project_root, 'model', args.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    path = {
        'cub200': os.path.expanduser('~/workspace/datasets/cub200'),
        'model': model_dir,
        'ckpt_path': os.path.join(model_dir, 'vgg_16_fc_epoch_best.pth')
    }

    manager = BCNNTrainer(vars(args), path, ckpt_basename='vgg_16_all')
    # manager.getStat()
    manager.train()


if __name__ == '__main__':
    main()
