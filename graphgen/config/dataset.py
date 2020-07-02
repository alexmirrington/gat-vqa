"""Classes and enums for storing dataset-related configuration information."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..datasets.gqa import GQASplit, GQAVersion


class DatasetName(Enum):
    """An enum containing a list of valid dataset names."""

    GQA = "gqa"


@dataclass(frozen=True)
class DatasetConfig:
    """A class specifying the common fields used across all datasets."""

    name: DatasetName
    root: Path

    def __post_init__(self) -> None:
        """Perform post-init checks on the `root` field."""
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"Field {self.root=} must be a valid directory.")


@dataclass(frozen=True)
class GQADatasetConfig(DatasetConfig):
    """A class specifying the valid values for a GQA dataset config."""

    split: GQASplit
    version: GQAVersion

    def __post_init__(self) -> None:
        """Perform post-init checks on the `name` field."""
        if self.name != DatasetName.GQA:
            raise ValueError(f"Field {self.name=} must be equal to {DatasetName.GQA}")
