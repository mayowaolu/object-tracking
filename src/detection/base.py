import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from config import ModelConfig

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
    def predict(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss given predictions and ground truth targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.

        Returns:
            Loss tensor.
        """
        pass

    def load_weights(self, filepath: str) -> None:
        """
        Load pre-trained weights
        """
        pass

    def save_weights(self, filepath: str):
        """
        Save the model's weights
        """
        pass