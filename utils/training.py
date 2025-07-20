# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
# from timm.utils import accuracy
from torch.ao.nn.quantized.functional import threshold

from utils.conf import get_device
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
import copy
import time
import wandb
from utils.buffer import Buffer
import torch.nn.functional as F
from models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Average(lst):
    return sum(lst) / len(lst)

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    if type(dataset.N_CLASSES_PER_TASK) == list:
        THIS_TASK_START = int(np.sum(dataset.N_CLASSES_PER_TASK[:k]))
        THIS_TASK_END = int(np.sum(dataset.N_CLASSES_PER_TASK[:k+1]))
    else:
        THIS_TASK_START = k * dataset.N_CLASSES_PER_TASK
        THIS_TASK_END = (k + 1) * dataset.N_CLASSES_PER_TASK

    outputs[:, :THIS_TASK_START] = -float('inf')
    outputs[:, THIS_TASK_END:] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """

    status = model.module.net.training
    model.module.net.eval()
    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        # print('testing task index', k)
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for i, data in enumerate(test_loader):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.module.device), labels.to(model.module.device)
                if 'class-il' not in model.module.COMPATIBILITY:
                    outputs, feas = model(inputs, k)
                else:
                    outputs, feas = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.module.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.module.net.train(status)
    return accs, accs_mask_classes

def evaluate_previous(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """

    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []

    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs, feas = model(inputs, k)
                else:
                    outputs, feas = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes

def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.module.net.to(model.module.device)
    results, results_mask_classes = [], []
    model_dict = {}

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.module.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.module.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.module.NAME != 'icarl' and model.module.NAME != 'pnn':
        random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    source_error_list = []
    source_mask_error_list = []
    backbone = dataset.get_backbone()
    MODEL_loss = dataset.get_loss()
    initial_model = get_model(args, backbone, MODEL_loss, dataset.get_transform())
    initial_model = initial_model.to(device)
    for t in range(dataset.N_TASKS):
        model.module.net.train()
        train_loader, test_loader = dataset.get_data_loaders()

        if hasattr(model.module, 'begin_task'):
            model.module.begin_task(dataset)
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        unlearn_flag = 1
        unlearn_times = 0
        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.module.args.n_epochs):
            # using transferability bound for detecting negative transfer
            # if epoch <= model.module.args.n_epochs * 0.3:
            #     for i, data in enumerate(train_loader):
            #         inputs, labels, not_aug_inputs = data
            #         inputs, labels = inputs.to(initial_model.device), labels.to(initial_model.device)
            #         not_aug_inputs = not_aug_inputs.to(initial_model.device)
            #         loss, features = initial_model.observe(inputs, labels, not_aug_inputs, t)
            #         progress_bar(i, len(train_loader), epoch, t, loss)
            #     temp_accs, temp_mask_accs = evaluate_previous(initial_model, dataset)
            #     source_error_list.append(temp_accs)
            #     source_mask_error_list.append(temp_mask_accs)
            #
            # if epoch == model.module.args.n_epochs * 0.3 and t > 0:
            #     source_error = 100 - torch.mean(torch.tensor(source_mask_error_list[t-1]))
            #     target_error = 100 - source_mask_error_list[t][t]
            #     # source_error = 100 - sum(source_error_list[:t]) / len(source_error_list[:t])
            #     domain_model = get_model(args, backbone, MODEL_loss, dataset.get_transform())
            #     domain_model = domain_model.to(device)
            #     domain_model.buffer = model.module.buffer
            #     domain_classifier, lambda, error = domain_model.target_upperbound(dataset, t)
            #     target_upper_bound = source_error + (error/2) + lambda
            #     if target_error >= target_upper_bound:
            #         unlearn_flag = 0
            #         print(f"Task {t+1} activate unlearn module")

            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.module.device)
                    labels = labels.to(model.module.device)
                    not_aug_inputs = not_aug_inputs.to(model.module.device)
                    logits = logits.to(model.module.device)
                    loss = model.module.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.module.device), labels.to(
                        model.module.device)
                    not_aug_inputs = not_aug_inputs.to(model.module.device)
                    loss, features = model.module.observe(inputs, labels, not_aug_inputs, t, gradients, unlearn_flag)

                progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)

            if scheduler is not None:
                scheduler.step()

        if hasattr(model.module, 'end_task'):
            model.module.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        bwt = backward_transfer(results)
        bwt_mask_classes = backward_transfer(results_mask_classes)
        print("bwt:", bwt)
        print("bwt_mask_classes:", bwt_mask_classes)

        mean_acc = np.mean(accs, axis=1)
        print("Each task in CIL:", accs[0])
        print("Each task in TIL:", accs[1])

        print("unlearn times of new task:", unlearn_times)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.module.NAME != 'icarl' and model.module.NAME != 'pnn':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))