"""Classes storing model configuration information."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimiserConfig:
    """Class storing optimiser configuration information."""

    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class ModelDataSubsetConfig:
    """Class storing data subset information for model training."""

    split: str  # TODO Link to existing dataset config enums
    version: str  # TODO Link to existing dataset config enums


@dataclass(frozen=True)
class ModelDataConfig:
    """Class storing dataset information for model training."""

    artifact: str  # TODO support local paths
    train: ModelDataSubsetConfig
    val: ModelDataSubsetConfig
    test: ModelDataSubsetConfig


@dataclass(frozen=True)
class ModelConfig:
    """Class for storing model configuration information."""

    optimiser: OptimiserConfig
    data: ModelDataConfig
