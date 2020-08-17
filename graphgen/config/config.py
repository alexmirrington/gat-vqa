"""Classes for storing general configuration information."""
from dataclasses import dataclass
from typing import Union

from .dataloader import DataloaderConfig
from .gqa import GQADatasetConfig


@dataclass(frozen=True)
class Config:
    """A class containing configuration information such as model parameters."""

    dataset: Union[GQADatasetConfig]
    dataloader: DataloaderConfig
