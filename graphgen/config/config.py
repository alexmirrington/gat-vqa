"""Classes for storing general configuration information."""
from dataclasses import dataclass
from typing import Union

from .clevr import CLEVRDatasetConfig
from .gqa import GQADatasetConfig
from .model import (
    E2EMultiGCNModelConfig,
    FasterRCNNModelConfig,
    MultiGCNModelConfig,
    ReasoningMultiGCNModelConfig,
)
from .preprocessing import PreprocessingConfig
from .training import TrainingConfig


@dataclass
class Config:
    """A class containing configuration information such as model parameters."""

    dataset: Union[CLEVRDatasetConfig, GQADatasetConfig]
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    model: Union[
        FasterRCNNModelConfig,
        E2EMultiGCNModelConfig,
        MultiGCNModelConfig,
        ReasoningMultiGCNModelConfig,
    ]
