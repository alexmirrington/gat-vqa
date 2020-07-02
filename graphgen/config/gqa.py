"""Classes for storing filemap-related configuration information."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Union

from .dataset import DatasetConfig, DatasetName


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


@dataclass
class GQAFilemap:
    """A class defining the paths to relevant GQA dataset files.

    All paths are defined relative to the dataset's root directory.
    """

    questions_dir: Path = Path("questions")
    images_dir: Path = Path("images", "images")
    objects_dir: Path = Path("images", "objects")
    spatial_dir: Path = Path("images", "spatial")
    scene_graphs_dir: Path = Path("sceneGraphs")

    def images(self, image_id: Optional[str] = None) -> Union[Path, Iterable[Path]]:
        """Get the path to one/all image files."""
        if image_id is None:
            return self.images_dir.iterdir()
        return self.images_dir / f"{image_id}.jpg"

    def objects(self, chunk_id: Optional[int] = None) -> Union[Path, Iterable[Path]]:
        """Get the path to one/all Faster-RCNN object features file(s)."""
        if chunk_id is None:
            return [
                path for path in list(self.objects_dir.iterdir()) if path.match("*.h5")
            ]
        return self.objects_dir / f"gqa_objects_{chunk_id}.h5"

    def objects_meta(self) -> Path:
        """Get the path to the Faster-RCNN object features metadata file."""
        return self.objects_dir / "gqa_objects_info.json"

    def spatial(self, chunk_id: Optional[int] = None) -> Union[Path, Iterable[Path]]:
        """Get the path to one/all ResNet-101 spatial features file(s)."""
        if chunk_id is None:
            return [
                path for path in list(self.spatial_dir.iterdir()) if path.match("*.h5")
            ]
        return self.spatial_dir / f"gqa_spatial_{chunk_id}.h5"

    def spatial_meta(self) -> Path:
        """Get the path to the ResNet-101 spatial features metadata file."""
        return self.spatial_dir / "gqa_spatial_info.json"

    def questions(self, split: GQASplit, version: GQAVersion) -> Iterable[Path]:
        """Get the path to the questions JSON file for a given split and version."""
        if split == GQASplit.TRAIN and version == GQAVersion.ALL:
            return tuple(
                (
                    self.questions_dir / f"{split.value}_{version.value}_questions"
                ).iterdir()
            )
        return (self.questions_dir / f"{split.value}_{version.value}_questions.json",)

    def scene_graphs(self, split: GQASplit) -> Path:
        """Get the path to the scene graphs JSON file for a given split."""
        if split in (GQASplit.TRAIN, GQASplit.VAL):
            return self.scene_graphs_dir / f"{split.value}_sceneGraphs.json"
        raise ValueError(f"No scene graphs exist for split {split}.")


@dataclass(frozen=True)
class GQADatasetConfig(DatasetConfig):
    """A class specifying the valid values for a GQA dataset config."""

    split: GQASplit
    version: GQAVersion
    filemap: GQAFilemap = GQAFilemap()

    def __post_init__(self) -> None:
        """Perform post-init checks on the `name` field."""
        if self.name != DatasetName.GQA:
            raise ValueError(f"Field {self.name=} must be equal to {DatasetName.GQA}")
