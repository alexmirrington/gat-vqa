"""Classes for storing general configuration information."""
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .clevr import CLEVRDatasetConfig
from .dataloader import DataloaderConfig
from .gqa import GQADatasetConfig
from .logging import LoggingConfig


@dataclass(frozen=True)
class Config:
    """A class containing configuration information such as model parameters."""

    cache: Path
    dataset: Union[CLEVRDatasetConfig, GQADatasetConfig]
    dataloader: DataloaderConfig
    logging: LoggingConfig

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if not self.cache.exists() or not self.cache.is_dir():
            raise ValueError(f"Field {self.cache=} must be a valid directory.")
