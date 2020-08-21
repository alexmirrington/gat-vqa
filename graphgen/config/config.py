"""Classes for storing general configuration information."""
from dataclasses import dataclass
from typing import Union

from .clevr import CLEVRDatasetConfig
from .dataloader import DataloaderConfig
from .gqa import GQADatasetConfig
from .logging import LoggingConfig
from .preprocessing import PreprocessingConfig


@dataclass(frozen=True)
class Config:
    """A class containing configuration information such as model parameters."""

    dataset: Union[CLEVRDatasetConfig, GQADatasetConfig]
    preprocessing: PreprocessingConfig
    dataloader: DataloaderConfig
    logging: LoggingConfig
