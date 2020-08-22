"""Classes for storing model configuration information."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimiserConfig:
    """Class for storing optimiser configuration information."""

    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class ModelConfig:
    """Class for storing model configuration information."""

    optimiser: OptimiserConfig
