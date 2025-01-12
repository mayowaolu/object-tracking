from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Configuration for the model architecture.
    """
    num_layers: int = 4
    num_filters: int = 32

@dataclass
class TrainingConfig:
    """
    Configuration for training the model.
    """
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 10