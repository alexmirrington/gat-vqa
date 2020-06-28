"""Classes and enums for storing configuration-related information."""
from dataclasses import dataclass
from enum import Enum


class Dataset(Enum):
    """An enum specifying the valid values for a dataset."""

    GQA = "gqa"


@dataclass
class Config:
    """A class containing configuration information such as model parameters."""

    dataset: Dataset
