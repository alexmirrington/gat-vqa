"""Classes for storing training configuration information."""
from dataclasses import dataclass

from .training import DataConfig, DataloaderConfig


@dataclass
class PredictionConfig:
    """Class for storing prediction configuration information."""

    dataloader: DataloaderConfig
    data: DataConfig
