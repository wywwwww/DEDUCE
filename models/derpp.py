# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from copy import deepcopy
import torch
import copy
import random
from collections import OrderedDict
epsilon = 1E-20

from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from utils.gum import GUM
import numpy as np
from torch.utils.data import random_split
import torch.nn.functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser

def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.temp = copy.deepcopy(self.net).to(self.device)
        self.plas = copy.deepcopy(self.net).to(self.device)
        self.temp_opt = torch.optim.SGD(self.temp.parameters(), lr=0.01)
        self.plas_opt = torch.optim.SGD(self.plas.parameters(), lr=0.01)

        lr = self.args.lr
        weight_decay = 0.0001
        self.delta = 0.0001
        self.tau = 0.00001

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = {}
        self.fish = {}
        for name, param in self.net.named_parameters():
            self.fish[name] = torch.zeros_like(param).to(self.device)

        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        self.grad_dims = []

        self.phi = 0.00001
        self.utility_function = "contribution"
        self.maturity_threshold = 1000

        self.prev_unlearn_flag = None
        self.buffer_grad = None
        # Global Unlearning Module (GUM)
        self.gum = GUM(net=self.net,
                             hidden_activation="relu",
                             replacement_rate=self.phi,
                             decay_rate=0.99,
                             util_type=self.utility_function,
                             maturity_threshold=self.maturity_threshold,
                             device=self.device)
        self.current_features = []
        self.times = 0

    # Using gradient conflict analysis strategy for detecting negative transfer
    def observe(self, inputs, labels, not_aug_inputs, task_id=None, gradients=None, unlearn_flag=None):
        grad = []
        gradients = []
        if task_id > 0:
            self.buffer_grad = self.cal_buffer(task_id)
        if task_id is not None and task_id != self.prev_unlearn_flag and task_id > 0:
            print(f"task {task_id - 1} unlearn times: {self.times}")
            self.prev_unlearn_flag = task_id
            self.times = 0
        self.opt.zero_grad()
        current_features = []
        outputs, features = self.net(inputs, feature_list=current_features)
        loss = self.loss(outputs, labels)
        grads = torch.autograd.grad(loss, self.net.parameters(), retain_graph=True, create_graph=False)
        if self.buffer_grad is not None:
            for g in grads:
                if g is not None:
                    grad.append(g.view(-1))
            gradients.append(torch.cat(grad))
            gradients = torch.cat(gradients)

            tolerance = 0.0  # define tolerance
            # L2
            grad_norm = torch.norm(gradients, p=2)
            buffer_grad_norm = torch.norm(self.buffer_grad, p=2)

            # gradient conflict analysis
            left = torch.dot(gradients, self.buffer_grad)
            right = tolerance * grad_norm * buffer_grad_norm
            if left<=right:
                # activate Local Unlearning Module
                self.lum(inputs=inputs, labels=labels)
                self.times = self.times + 1

        self.opt.zero_grad()
        outputs, features = self.net(inputs, feature_list=current_features)
        loss = self.loss(outputs, labels)
        prev_params = {name: param.clone() for name, param in self.net.named_parameters()}
        loss += self.ewc_loss(prev_params=prev_params, lambda_ewc=0.1)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs, feas = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs, feas= self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()
        self.gum.gen_and_test(current_features, fish=self.fish)
        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)

        return loss.item(), features

    # Local Unlearning Module (LUM)
    def lum(self, inputs, labels):

        self.temp.load_state_dict(self.net.state_dict())
        self.temp.train()
        outputs, feas = self.temp(inputs)
        unlearn_loss = - F.cross_entropy(outputs, labels)
        regularization_loss = 0
        lambda_score = 0.1
        if self.checkpoint:
            for (initial_name, initial_param), (current_name, current_param) in zip(self.checkpoint.items(),
                                                                          self.temp.named_parameters()):
                if initial_name == current_name:
                    regularization_loss += torch.sum((current_param - initial_param) ** 2)
        unlearn_loss = unlearn_loss + lambda_score * regularization_loss
        self.temp_opt.zero_grad()
        unlearn_loss.backward()
        self.temp_opt.step()

        for (model_name, model_param), (temp_name, temp_param) in zip(self.net.named_parameters(), self.temp.named_parameters()):
                weight_update = temp_param - model_param
                model_param_norm = model_param.norm()
                weight_update_norm = weight_update.norm() + epsilon
                norm_update = model_param_norm / weight_update_norm * weight_update
                identity = torch.ones_like(self.fish[model_name])
                with torch.no_grad():
                    model_param.add_(self.delta * torch.mul(1.0/(identity + 0.001*self.fish[model_name]), norm_update + 0.001*torch.randn_like(norm_update)))

    def ewc_loss(self, prev_params, lambda_ewc):
        loss = 0
        for temp_name, temp_param in self.temp.named_parameters():
            if temp_name in prev_params:
                loss += (self.fish[temp_name] * (temp_param - prev_params[temp_name]).pow(2)).sum()
        return lambda_ewc * loss

    def end_task(self, dataset):
        self.temp.load_state_dict(self.net.state_dict())
        fish = {}
        for name, param in self.temp.named_parameters():
            fish[name] = torch.zeros_like(param).to(self.device)

        for j, data in enumerate(dataset.train_loader):
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.temp_opt.zero_grad()
                output, feas = self.temp(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()

                for name, param in self.temp.named_parameters():
                    fish[name] +=  exp_cond_prob * param.grad ** 2

        for name, param in self.temp.named_parameters():
            fish[name] /= (len(dataset.train_loader) * self.args.batch_size)
       
        for key in self.fish:
                self.fish[key] *= self.tau
                self.fish[key] += fish[key].to(self.device)

        for name, param in self.net.named_parameters():
            self.checkpoint[name] = param.data.clone()
        self.temp_opt.zero_grad()

        return self.fish

    def cal_buffer(self, task_number):
        classes_per_task = 10
        if self.buffer.is_empty():
            print("Buffer is empty")
            return torch.tensor([])

        buf_inputs, buf_labels, buf_task_labels = self.buffer.get_data(
            self.args.buffer_size, transform=self.transform)
        current_task_label = list(range(classes_per_task * task_number, classes_per_task * (task_number + 1)))

        mask = ~torch.isin(buf_labels, torch.tensor(current_task_label, device=buf_labels.device))
        filtered_inputs = buf_inputs[mask]
        filtered_labels = buf_labels[mask]

        if filtered_inputs.shape[0] == 0:
            print("Except: current task buffer is empty")
            return torch.tensor([])

        gradients = []
        unique_tasks = torch.arange(classes_per_task)

        for i in unique_tasks:
            task_mask = (filtered_labels // classes_per_task) == i
            task_inputs = filtered_inputs[task_mask]
            task_labels = filtered_labels[task_mask]

            if task_inputs.shape[0] == 0:
                continue

            num_samples = task_inputs.shape[0]
            num_batches = (num_samples + self.args.minibatch_size - 1) // self.args.minibatch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.args.minibatch_size
                end_idx = min(start_idx + self.args.minibatch_size, num_samples)

                sampled_inputs = task_inputs[start_idx:end_idx]
                sampled_labels = task_labels[start_idx:end_idx]

                self.opt.zero_grad()
                buf_outputs, feas = self.net(sampled_inputs)
                buffer_loss = self.loss(buf_outputs, sampled_labels)
                grads = torch.autograd.grad(buffer_loss, self.net.parameters(), retain_graph=False, create_graph=False)

                grad_vector = torch.cat([g.view(-1) for g in grads if g is not None])
                gradients.append(grad_vector)

        if len(gradients) > 0:
            avg_gradient = torch.stack(gradients).mean(dim=0)
        else:
            avg_gradient = torch.tensor([])

        return avg_gradient


    def compute_leep_score(self, source_model_probs, target_labels):

        N, C_s = source_model_probs.shape
        target_classes = torch.unique(target_labels)
        C_t = len(target_classes)

        # 1. calculate P(y_t, y_s)
        joint_prob = torch.zeros((C_t, C_s)).to(self.device)
        for i in range(N):
            y_t = target_labels[i] % C_t
            joint_prob[y_t, :] += source_model_probs[i, :]
        joint_prob /= N
        # 2. calculate P(y_s)
        source_marginal_prob = joint_prob.sum(axis=0, keepdims=True)
        # 3. calculate P(y_t | y_s) = P(y_t, y_s) / P(y_s)
        conditional_prob = joint_prob / (source_marginal_prob + 1e-10)
        # 4. calculate LEEP
        expected_probs_list = []
        for i in range(N):
            expected_probs =conditional_prob[target_labels[i] % C_t, :] * source_model_probs[i, :]
            expected_probs_list.append(expected_probs)
        expected_probs_tensor = torch.stack(expected_probs_list, dim=0)
        sum_expected_probs = torch.sum(expected_probs_tensor, dim=1)
        leep_score = torch.mean(torch.log(sum_expected_probs + 1e-10))

        return leep_score

    def target_upperbound(self, dataset, task_number):

        classes_per_task = 10
        if self.buffer.is_empty():
            print("Buffer is empty")
            return torch.tensor([])

        buf_inputs, buf_labels, buf_task_labels = self.buffer.get_data(
            self.args.buffer_size, transform=self.transform)
        current_task_label = list(range(classes_per_task * task_number, classes_per_task * (task_number + 1)))
        current_mask = ~torch.isin(buf_labels, torch.tensor(current_task_label, device=buf_labels.device))
        source_inputs = buf_inputs[current_mask]
        source_labels = buf_labels[current_mask]
        target_inputs = buf_inputs[~current_mask]
        target_labels = buf_labels[~current_mask]
        num_target = target_inputs.shape[0]
        num_source = source_inputs.shape[0]
        if num_source >= num_target:
            indices = torch.randperm(num_source)[:num_target]
            sampled_source_inputs = source_inputs[indices]
        else:
            print("source domain smaller than target domain")

        domain_target_labels = torch.ones(num_target, dtype=torch.long)
        domain_source_labels = torch.zeros(num_target, dtype=torch.long)
        domain_inputs = torch.cat([sampled_source_inputs, target_inputs], dim=0).to(self.device)
        domain_labels = torch.cat([domain_source_labels, domain_target_labels], dim=0).to(self.device)
        indices_source = torch.randperm(len(domain_inputs))
        shuffled_domain_inputs = domain_inputs[indices_source]
        shuffled_domain_labels = domain_labels[indices_source]

        num_samples = 2 * num_target
        train_size = int(0.7 * num_samples)
        test_size = num_samples - train_size
        train_inputs, test_inputs = random_split(shuffled_domain_inputs, [train_size, test_size])
        train_labels, test_labels = random_split(shuffled_domain_labels, [train_size, test_size])

        self.plas.load_state_dict(self.net.state_dict())
        source_model_probs, feas = self.plas(target_inputs)
        source_model_probs = F.softmax(source_model_probs, dim=1)
        # calculate LEEP
        leep_score = self.compute_leep_score(source_model_probs, target_labels)
        c = 1
        lameda = c * abs(leep_score)

        # training domain classifier
        self.ideal = dataset.get_backbone()
        self.ideal.to(self.device)
        self.ideal.train()
        in_features = self.ideal.fc.in_features
        self.ideal.fc = nn.Linear(in_features, 2).to(self.device)
        nn.init.xavier_uniform_(self.ideal.fc.weight)
        nn.init.zeros_(self.ideal.fc.bias)

        for param in self.ideal.parameters():
            param.requires_grad = False
        for param in self.ideal.fc.parameters():
            param.requires_grad = True

        # Define the optimizer for the new network
        self.ideal_opt = torch.optim.SGD(self.ideal.parameters(), lr=0.01)
        num_samples = train_size
        num_batches = (num_samples + self.args.minibatch_size - 1) // self.args.minibatch_size
        num_epochs = 500
        for epoch in range(num_epochs):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.args.minibatch_size
                end_idx = min(start_idx + self.args.minibatch_size, num_samples)
                sampled_inputs = train_inputs[start_idx:end_idx]
                sampled_labels = train_labels[start_idx:end_idx]
                self.ideal_opt.zero_grad()
                sampled_outputs, feas = self.ideal(sampled_inputs)
                domain_loss = self.loss(sampled_outputs, sampled_labels)
                domain_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ideal.parameters(), max_norm=5.0)
                self.ideal_opt.step()
        self.ideal.eval()

        outputs, feas = self.ideal(test_inputs[:])
        _, pred = torch.max(outputs.data, 1)
        fp = torch.sum((pred == 1) & (test_labels[:] == 0)).item()
        fn = torch.sum((pred == 0) & (test_labels[:] == 1)).item()
        fp_rate = fp / torch.sum(test_labels[:] == 0)
        fn_rate = fn / torch.sum(test_labels[:] == 1)
        domain_error = abs(2*(1-(fp_rate+fn_rate)))
        return self.ideal, lameda, domain_error


