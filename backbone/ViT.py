import torch
import torch.nn as nn
import timm
from typing import List
from backbone import MammothBackbone

class ViTBackbone(MammothBackbone):
    def __init__(self, num_classes: int, model_name: str = 'vit_base_patch16_224', pretrained: bool = True):
        super(ViTBackbone, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)

        self.vit.head = nn.Identity()
        self.embed_dim = self.vit.embed_dim
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x: torch.Tensor, feature_list: list = None):
        feature = self.vit(x)
        if feature_list is not None:
            feature_list.append(feature)

        out = self.fc(feature)
        return out, feature
