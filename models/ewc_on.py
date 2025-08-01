# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class EwcOn(ContinualModel):
    """Continual learning via online EWC."""
    NAME = 'ewc_on'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--e_lambda', type=float, required=True,
                            help='lambda weight for EWC')
        parser.add_argument('--gamma', type=float, required=True,
                            help='gamma parameter for EWC online')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(EwcOn, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = None

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = self.args.e_lambda * (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset):
        fish = torch.zeros_like(self.net.get_params())

        for j, data in enumerate(dataset.train_loader):
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.net.get_grads() ** 2

        fish /= (len(dataset.train_loader) * self.args.batch_size)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += fish

        self.checkpoint = self.net.get_params().data.clone()

    def get_penalty_grads(self):
        return self.args.e_lambda * 2 * self.fish * (self.net.get_params().data - self.checkpoint)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        if self.checkpoint is not None:
            self.net.set_grads(self.get_penalty_grads())
        loss = self.loss(outputs, labels)
        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        return loss.item()