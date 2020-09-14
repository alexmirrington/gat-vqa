"""Classes for storing training configuration information."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class OptimiserName(Enum):
    """An enum specifying supported optimiser types."""

    ADAM = "adam"
    SGD = "sgd"


@dataclass
class DataloaderConfig:
    """A class specifying the common fields used across all data loaders."""

    batch_size: int
    workers: int

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.batch_size <= 0:
            raise ValueError(f"Field {self.batch_size} must be greater than 0.")

        if self.workers < 0:
            raise ValueError(
                f"Field {self.workers} must be greater than or equal to 0."
            )


@dataclass
class OptimiserConfig:
    """Class storing optimiser configuration information."""

    name: OptimiserName
    momentum: Optional[float]
    learning_rate: float
    weight_decay: float
    grad_clip: Optional[float]
    schedule: bool

    def __post_init__(self) -> None:
        """Validate fields after dataclass construction."""
        if self.name == OptimiserName.ADAM and self.momentum is not None:
            raise ValueError(
                f"Field {self.momentum=} must be {None} when using an Adam optimiser."
            )


@dataclass
class DataSubsetConfig:
    """Class storing data subset information for model training."""

    split: str
    version: Optional[str] = None
    subset: Tuple[float, float] = (0.0, 1.0)


@dataclass
class FeatureConfig:
    """Class storing information abeout features used for model training."""

    name: str
    artifact: Optional[str]


@dataclass
class TrainingDataConfig:
    """Class storing dataset information for model training."""

    features: List[FeatureConfig]
    train: DataSubsetConfig
    val: DataSubsetConfig
    test: DataSubsetConfig


@dataclass
class TrainingConfig:
    """Class for storing training configuration information."""

    epochs: int
    log_step: int
    dataloader: DataloaderConfig
    optimiser: OptimiserConfig
    data: TrainingDataConfig

    def __post_init__(self) -> None:
        """Validate fields after dataclass construction."""
        if self.epochs <= 0:
            raise ValueError(f"Field {self.epochs} must be positive.")

        if self.log_step <= 0:
            raise ValueError(f"Field {self.log_step} must be strictly positive.")
