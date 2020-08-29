"""Classes for storing general configuration information."""
from dataclasses import dataclass
from typing import Union

from .clevr import CLEVRDatasetConfig
from .gqa import GQADatasetConfig
from .model import ModelConfig
from .preprocessing import PreprocessingConfig
from .training import TrainingConfig


@dataclass(frozen=True)
class Config:
    """A class containing configuration information such as model parameters."""

    dataset: Union[CLEVRDatasetConfig, GQADatasetConfig]
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    model: ModelConfig
