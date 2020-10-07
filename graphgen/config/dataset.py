"""Classes for storing dataset-related configuration information."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DatasetName(Enum):
    """An enum containing a list of valid dataset names."""

    GQA = "gqa"
    CLEVR = "clevr"


@dataclass
class DatasetConfig:
    """A class specifying the common fields used across all datasets."""

    name: DatasetName


@dataclass
class DatasetFilemap:
    """A class specifying the common fields used across all dataset filemaps."""

    root: Path

    def __post_init__(self) -> None:
        """Perform post-init checks on the `root` field."""
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"Field {self.root=} must be a valid directory.")
