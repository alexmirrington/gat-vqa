"""Classes for storing general configuration information."""
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .gqa import GQADatasetConfig


@dataclass(frozen=True)
class Config:
    """A class containing configuration information such as model parameters."""

    dataset: Union[GQADatasetConfig]
    cache: Path
