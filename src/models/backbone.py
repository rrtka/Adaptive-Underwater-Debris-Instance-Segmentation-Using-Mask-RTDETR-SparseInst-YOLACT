"""
Backbone implementation for Mask RT-DETR.
Extracts multi-scale feature hierarchies at strides 8, 16, 32 (C3, C4, C5).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict


class ResNetBackbone(nn.Module):
    """
    ResNet backbone extracting multi-scale feature representations.
    Supports ResNet-18, ResNet-34, ResNet-50.
    """
    def __init__(self, name: str = "resnet50", pretrained: bool = False):
        super().__init__()
        self.name = name

        if name == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.out_channels = [128, 256, 512]
        elif name == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            self.out_channels = [128, 256, 512]
        else:  # resnet50
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.out_channels = [512, 1024, 2048]

        # Stage 0: stem
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # Stages
        self.layer1 = base.layer1  # Stride 4
        self.layer2 = base.layer2  # Stride 8  (C3)
        self.layer3 = base.layer3  # Stride 16 (C4)
        self.layer4 = base.layer4  # Stride 32 (C5)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)  # Stride 8
        c4 = self.layer3(c3)  # Stride 16
        c5 = self.layer4(c4)  # Stride 32

        return {"c3": c3, "c4": c4, "c5": c5}
