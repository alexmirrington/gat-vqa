"""Classes for storing filemap-related configuration information."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from .dataset import DatasetConfig, DatasetFilemap, DatasetName


class CLEVRSplit(Enum):
    """An enum specifying possible values for CLEVR dataset splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class CLEVRFilemap(DatasetFilemap):
    """A class defining the paths to relevant CLEVR dataset files.

    All paths are defined relative to the dataset's root directory.
    """

    questions_dir: Path = Path("questions")
    images_dir: Path = Path("images")
    scene_graphs_dir: Path = Path("scenes")

    def image_path(self, split: CLEVRSplit, image_id: Optional[str] = None) -> Path:
        """Get the path to an image file."""
        if image_id is not None:
            return (
                self.root
                / self.images_dir
                / str(split.value)
                / f"CLEVR_{split.value}_{image_id}.png"
            )
        return self.root / self.images_dir / str(split.value)

    def question_path(self, split: CLEVRSplit) -> Path:
        """Get the path to the questions JSON file for a given split."""
        return self.root / self.questions_dir / f"CLEVR_{split.value}_questions.json"

    def scene_graph_path(self, split: CLEVRSplit) -> Path:
        """Get the path to the scene graphs JSON file for a given split."""
        return self.root / self.scene_graphs_dir / f"CLEVR_{split.value}_scenes.json"


@dataclass
class CLEVRDatasetConfig(DatasetConfig):
    """A class specifying the valid values for a CLEVR dataset config."""

    split: CLEVRSplit
    filemap: CLEVRFilemap

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != DatasetName.CLEVR:
            raise ValueError(f"Field {self.name=} must be equal to {DatasetName.CLEVR}")
