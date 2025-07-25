# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
from backbone import MammothBackbone


from torch import Tensor
from typing import Callable, List, Optional, Type, Union
from torchvision.utils import _log_api_usage_once


def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SequentialWithKeywordArguments(torch.nn.Sequential):

    """
    Sequential module that allows the use of keyword arguments in the forward pass
    """

    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor, feature_list: list = None) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if feature_list is not None: feature_list.append(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut_out = self.shortcut(identity)

        out += shortcut_out
        out = self.relu(out)

        if feature_list is not None: feature_list.append(out)

        return out


class ResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.fc = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> SequentialWithKeywordArguments:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithKeywordArguments(*layers)

    def forward(self, x: torch.Tensor, feature_list: list = None) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :param returnt: return type (a string among 'out', 'features', 'all')
        :return: output tensor (output_classes)
        """

        out = relu(self.bn1(self.conv1(x))) # 64, 32, 32
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)

        if feature_list is not None: feature_list.append(out)

        out = self.layer1(out, feature_list=feature_list)  # -> 64, 32, 32
        out = self.layer2(out, feature_list=feature_list)  # -> 128, 16, 16
        out = self.layer3(out, feature_list=feature_list)  # -> 256, 8, 8
        out = self.layer4(out, feature_list=feature_list)  # -> 512, 4, 4

        if feature_list is not None: feature_list.pop(-1)

        out = avg_pool2d(out, out.shape[2]) # -> 512, 1, 1
        feature = out.view(out.size(0), -1)  # 512

        if feature_list is not None: feature_list.append(feature)
        out = self.fc(feature)
        return out, feature

def resnet18(nclasses: int, nf: int=64) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)
