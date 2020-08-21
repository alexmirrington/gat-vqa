"""Utilities for loading data from one or more files."""
from abc import abstractmethod
from pathlib import Path
from typing import Any, Tuple

import torch.utils.data


class ChunkedDataset(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that loads data from one or more files."""

    def __init__(self, root: Path) -> None:
        """Initialise a `ChunkedDataset` instance.

        Params:
        -------
        `root`: A path to a single file or a folder containing multiple files
        (chunks) at its top level.

        Returns:
        --------
        None
        """
        super().__init__()

        if not isinstance(root, Path):
            raise TypeError(f"Parameter {root=} must be of type {Path.__name__}.")

        if not root.exists():
            raise ValueError(f"Parameter {root=} must point to a file or directory.")

        if root.is_dir() and len(tuple(root.iterdir())) == 0:
            raise ValueError(f"Parameter {root=} must point to a non-empty directory.")

        self._root = root
        self._chunks: Tuple[Path, ...] = tuple(
            sorted(self._root.iterdir())
        ) if self._root.is_dir() else (self._root,)

    @property
    def root(self) -> Path:
        """Get the root of the dataset, either a single file or a directory."""
        return self._root

    @property
    def chunks(self) -> Tuple[Path, ...]:
        """Get a tuple of containing the paths to the chunks in the dataset."""
        return self._chunks

    @property
    @abstractmethod
    def chunk_sizes(self) -> Tuple[int, ...]:
        """Get the length of each of the chunks in the dataset."""
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        """Get the length of the dataset."""
        raise NotImplementedError()
