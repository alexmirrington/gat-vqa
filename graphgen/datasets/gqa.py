"""A torch-compatible GQA dataset implementation."""
import os.path
from enum import Enum
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

    def __init__(self, root: str, split: GQASplit, version: GQAVersion) -> None:
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
        self._root = root
        self._split = split
        self._version = version

        self._filemap = self._init_filemap()
        self._data = ChunkedJSONDataset(
            self._filemap["questions"][self.split][self.version]
        )

    def _init_filemap(self) -> Any:
        return {
            "questions": {
                split: {
                    version: os.path.join(
                        self.root,
                        "questions",
                        f"{split.value}_{version.value}_questions",
                    )
                    if split == GQASplit.TRAIN and version == GQAVersion.ALL
                    else os.path.join(
                        self.root,
                        "questions",
                        f"{split.value}_{version.value}_questions.json",
                    )
                    for version in GQAVersion
                }
                for split in GQASplit
            }
        }

    @property
    def root(self) -> str:
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
