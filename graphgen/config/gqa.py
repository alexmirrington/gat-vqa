"""Classes for storing GQA configuration information."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Set

from .dataset import DatasetConfig, DatasetFilemap, DatasetName


class GQASplit(Enum):
    """An enum specifying possible values for GQA dataset splits."""

    TRAIN = "train"
    VAL = "val"
    DEV = "testdev"
    TEST = "test"
    CHALLENGE = "challenge"


class GQAVersion(Enum):
    """An enum specifying possible values for GQA dataset versions."""

    BALANCED = "balanced"
    ALL = "all"


class GQAFeatures(Enum):
    """An enum specifying possible values for GQA features."""

    IMAGES = "images"
    OBJECTS = "objects"
    SPATIAL = "spatial"
    SCENE_GRAPHS = "scene_graphs"


@dataclass(frozen=True)
class GQAFilemap(DatasetFilemap):
    """A class defining the paths to relevant GQA dataset files.

    All paths are defined relative to the dataset's root directory.
    """

    questions_dir: Path = Path("questions")
    images_dir: Path = Path("images", "images")
    objects_dir: Path = Path("images", "objects")
    spatial_dir: Path = Path("images", "spatial")
    scene_graphs_dir: Path = Path("sceneGraphs")

    def image_path(self, image_id: Optional[str] = None) -> Path:
        """Get the path to an image file."""
        if image_id is not None:
            return self.root / self.images_dir / f"{image_id}.jpg"
        return self.root / self.images_dir

    def object_path(self, chunk_id: Optional[int] = None) -> Path:
        """Get the path to a Faster-RCNN object features file."""
        if chunk_id is not None:
            return self.root / self.objects_dir / f"gqa_objects_{chunk_id}.h5"
        return self.root / self.objects_dir

    def object_meta_path(self) -> Path:
        """Get the path to the Faster-RCNN object features metadata file."""
        return self.root / self.objects_dir / "gqa_objects_info.json"

    def spatial_path(self, chunk_id: Optional[int] = None) -> Path:
        """Get the path to one/all ResNet-101 spatial features file(s)."""
        if chunk_id is not None:
            return self.root / self.spatial_dir / f"gqa_spatial_{chunk_id}.h5"
        return self.root / self.spatial_dir

    def spatial_meta_path(self) -> Path:
        """Get the path to the ResNet-101 spatial features metadata file."""
        return self.root / self.spatial_dir / "gqa_spatial_info.json"

    def question_path(
        self,
        split: GQASplit,
        version: GQAVersion,
        chunked: bool = False,
        chunk_id: Optional[int] = None,
    ) -> Path:
        """Get the path to the questions JSON file for a given split and version."""
        if chunked:
            if chunk_id is None:
                return (
                    self.root
                    / self.questions_dir
                    / f"{split.value}_{version.value}_questions"
                )
            return (
                self.root
                / self.questions_dir
                / f"{split.value}_{version.value}_questions"
                / f"{split.value}_{version.value}_questions_{chunk_id}.json"
            )
        return (
            self.root
            / self.questions_dir
            / f"{split.value}_{version.value}_questions.json"
        )

    def scene_graph_path(self, split: GQASplit) -> Path:
        """Get the path to the scene graphs JSON file for a given split."""
        return self.root / self.scene_graphs_dir / f"{split.value}_sceneGraphs.json"


@dataclass(frozen=True)
class GQADatasetConfig(DatasetConfig):
    """A class specifying the valid values for a GQA dataset config."""

    version: GQAVersion
    split: GQASplit
    features: Set[GQAFeatures]
    filemap: GQAFilemap

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.name != DatasetName.GQA:
            raise ValueError(f"Field {self.name=} must be equal to {DatasetName.GQA}")
