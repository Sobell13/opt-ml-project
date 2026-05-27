"""Model definitions for the mini-project baseline."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Small CNN for Fashion-MNIST/CIFAR-style image classification.

    Default input_channels=1 is for Fashion-MNIST.
    Architecture:
        Conv(32) -> ReLU -> MaxPool
        Conv(64) -> ReLU -> MaxPool
        Flatten
        Linear(128) -> ReLU
        Linear(10)
    """

    def __init__(self, input_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(name: str = "small_cnn", input_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """Factory for models used by run.py."""
    name = name.lower()
    if name == "small_cnn":
        return SmallCNN(input_channels=input_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name}")
