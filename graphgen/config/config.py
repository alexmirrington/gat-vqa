"""Classes for storing general configuration information."""
from dataclasses import dataclass
from typing import Union

from .dataset import GQADatasetConfig


@dataclass(frozen=True)
class Config:
    """A class containing configuration information such as model parameters."""

    dataset: Union[GQADatasetConfig]
