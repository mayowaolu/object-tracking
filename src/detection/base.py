import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseDetector(ABC, nn.Module):

    @abstractmethod
    def __init__(self, config: "ModelConfig"):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass of the model

        Args:
            x: Input tensor
        
        Returns:
            Output tensor.

        """
        pass

    @abstractmethod
    def calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss given predictions and ground truth targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.

        Returns:
            Loss tensor.
        """
        pass