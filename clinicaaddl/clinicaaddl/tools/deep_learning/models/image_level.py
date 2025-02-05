# coding: utf8

from .modules import PadMaxPool3d, Flatten, BasicBlock, Bottleneck, SEBasicBlock, SEBottleneck, ResNet
import torch.nn as nn

"""
All the architectures are built here
"""


class Conv5_FC3(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv5_FC3_mni(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Extensive preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_mni, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 4 * 5 * 4, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv6_FC3(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv6_FC3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(256 * 3 * 4 * 3, 1000),
            nn.ReLU(),

            nn.Linear(1000, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 256, 3, 4, 3]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

class ResNet18(ResNet):
    def __init__(self, n_classes=3, expanded=False):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=n_classes, expanded=expanded)


class ResNet50(ResNet):
    def __init__(self, n_classes=3, expanded=False):
        super().__init__(Bottleneck, [3, 4, 6, 3], num_classes=n_classes, expanded=expanded)

class SEResNet18(ResNet):
    def __init__(self, n_classes=3, expanded=False):
        super().__init__(SEBasicBlock, [2, 2, 2, 2], num_classes=n_classes, num_channels=1, expanded=expanded)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

class SEResNet50(ResNet):
    def __init__(self, n_classes=3, expanded=False):
        super().__init__(SEBottleneck, [3, 4, 6, 3], num_classes=n_classes, num_channels=1, expanded=expanded)

        
class ResNet18Expanded(ResNet18):
    def __init__(self, n_classes=3, expanded=True):
        super().__init__(n_classes, expanded)


class ResNet50Expanded(ResNet50):
    def __init__(self, n_classes=3, expanded=True):
        super().__init__(n_classes, expanded)

class SEResNet18Expanded(SEResNet18):
    def __init__(self, n_classes=3, expanded=True):
        super().__init__(n_classes, expanded)



class SEResNet50Expanded(SEResNet50):
    def __init__(self, n_classes=3, expanded=True):
        super().__init__(n_classes, expanded)
