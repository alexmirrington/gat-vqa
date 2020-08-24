"""Classes storing model configuration information."""
from dataclasses import dataclass
from typing import List, Optional


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
class ModelFeatureConfig:
    """Class storing information abeout features used for model training."""

    name: str  # TODO Link to dataset config enums
    artifact: Optional[str]  # TODO support local paths


@dataclass(frozen=True)
class ModelDataConfig:
    """Class storing dataset information for model training."""

    features: List[ModelFeatureConfig]
    train: ModelDataSubsetConfig
    val: ModelDataSubsetConfig
    test: ModelDataSubsetConfig


@dataclass(frozen=True)
class ModelConfig:
    """Class for storing model configuration information."""

    optimiser: OptimiserConfig
    data: ModelDataConfig
