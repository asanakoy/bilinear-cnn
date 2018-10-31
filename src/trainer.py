# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import numpy as np
import os
import shutil
import sys

import torch
import torchvision
from tqdm import tqdm

import cub200
from bcnn import BCNN


class BCNNTrainer(object):
    """Manager class to train bilinear CNN.

    Attributes:
        _options: Hyperparameters.
        _path: Useful paths.
        _net: Bilinear CNN.
        _criterion: Cross-entropy loss.
        _solver: SGD with momentum.
        _scheduler: Reduce learning rate by a fator of 0.1 when plateau.
        _train_loader: Training data.
        _test_loader: Testing data.
    """
    def __init__(self, options, path, ckpt_basename='vgg_16'):
        """Prepare the network, criterion, solver, and data.

        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path
        self.ckpt_basename = ckpt_basename
        # Network.
        self._net = BCNN(freeze_features=True)
        #self._net = torch.nn.DataParallel(self._net)
        self._net.features = torch.nn.DataParallel(self._net.features)
        self._net.cuda()

        if 'ckpt_path' in self._path:
            if os.path.exists(self._path['ckpt_path']):
                print('Continue from', self._path['ckpt_path'])
                self._net.load_state_dict(torch.load(self._path['ckpt_path']))
            else:
                print('Ckpt {} not found!'.format(self._path['ckpt_path']))
        print(self._net)
        # Criterion.
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Solver.
        self._solver = torch.optim.SGD(
            self._net.fc.parameters(), lr=self._options['base_lr'],
            momentum=0.9, weight_decay=self._options['weight_decay'])
        if self._options['lr_scheduler'] == 'reduce_on_plateau':
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._solver, mode='max', factor=0.1, patience=5, verbose=True,
                threshold=1e-4, min_lr=1e-6)
        elif self._options['lr_scheduler'] == 'fixed':
            self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._solver,
                lambda epoch: 1.0)
        else:
            raise ValueError('Unknown scheduler:', self._options['lr_scheduler'])

	# Imagenet normalization
	normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
	    normalize
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
      	    normalize
        ])
        train_data = cub200.CUB200(
            root=self._path['cub200'], train=True, download=True,
            transform=train_transforms)
        test_data = cub200.CUB200(
            root=self._path['cub200'], train=False, download=True,
            transform=test_transforms)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=16,
            shuffle=False, num_workers=4, pin_memory=True)

    def train(self):
        """Train the network."""
        print('Training.')
        self._net.train()
        best_acc = 0.0
        best_epoch = None
        for epoch in range(self._options['epochs']):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            for batch_idx, (X, y) in enumerate(self._train_loader):
                # Data.
                X = X.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y)
                epoch_loss.append(loss.data.item())
                # Prediction.
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()
                # Backward pass.
                loss.backward()
                self._solver.step()
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                                 % (epoch, self._options['epochs'], batch_idx + 1,
                                    len(self._train_loader), loss.data.item(), (100. * num_correct) / num_total))
                sys.stdout.flush()
            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._test_loader)
            print('\nEpoch\tTrain loss\tTrain acc\tTest acc')
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (epoch + 1, np.mean(epoch_loss), train_acc, test_acc))
            self._scheduler.step(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                print('*', end='')
                # Save model onto disk.
                save_path = os.path.join(self._path['model'], '{}_epoch_{}.pth'.format(self.ckpt_basename, epoch + 1))
                save_path_best = os.path.join(self._path['model'], '{}_epoch_best.pth'.format(self.ckpt_basename))
                torch.save(self._net.state_dict(), save_path)
                shutil.copy(save_path, save_path_best)
        print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):
        """Compute the train/test accuracy.

        Args:
            data_loader: Train/Test DataLoader.

        Returns:
            Train/Test accuracy in percentage.
        """
        self._net.eval()
        num_correct = 0
        num_total = 0
        for X, y in data_loader:
            # Data.
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with torch.no_grad():
                # Prediction.
                score = self._net(X)
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()
        return 100 * num_correct / num_total

    def getStat(self):
        """Get the mean and std value for a certain dataset."""
        print('Compute mean and variance for training data.')
        train_data = cub200.CUB200(
            root=self._path['cub200'], train=True,
            transform=torchvision.transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=4,
            pin_memory=True)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for X, _ in tqdm(train_loader):
            for d in range(3):
                mean[d] += X[:, d, :, :].mean()
                std[d] += X[:, d, :, :].std()
        mean.div_(len(train_data))
        std.div_(len(train_data))
        print(mean)
        print(std)

