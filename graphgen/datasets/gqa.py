"""A torch-compatible GQA dataset implementation."""
import os.path
from enum import Enum
from pathlib import Path
from typing import Any

import torch.utils.data

from .utilities import ChunkedJSONDataset


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


class GQAQuestions(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that retrieves GQA question samples."""

    def __init__(self, root: Path, split: GQASplit, version: GQAVersion) -> None:
        """Initialise a `GQAQuestions` instance.

        Params:
        -------
        `root`: A path to the root directory of the GQA dataset.
        `split`: The dataset split to use.
        `version`: The dataset version to use.

        Returns:
        --------
        None
        """
        super().__init__()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Parameter {root=} must be a directory.")

        self._root = root
        self._split = split
        self._version = version

        self._filemap = self._init_filemap()
        self._data = ChunkedJSONDataset(
            self._filemap["questions"][self.split][self.version]
        )

    def _init_filemap(self) -> Any:
        filemap: Any = {"questions": {}}
        for split in GQASplit:
            filemap["questions"][split] = {}
            for version in GQAVersion:
                path = self._root / "questions"
                if split == GQASplit.TRAIN and version == GQAVersion.ALL:
                    path = path / f"{split.value}_{version.value}_questions"
                else:
                    path = path / f"{split.value}_{version.value}_questions.json"
                if not os.path.exists(path):
                    raise ValueError(f"No file or folder exists at path {path=}")
                filemap["questions"][split][version] = path
        return filemap

    @property
    def root(self) -> Path:
        """Get the dataset root directory."""
        return self._root

    @property
    def split(self) -> GQASplit:
        """Get the dataset split."""
        return self._split

    @property
    def version(self) -> GQAVersion:
        """Get the dataset version."""
        return self._version

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._data)

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        return self._data[index]
