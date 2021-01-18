"""Utilities for defining datasets with non-integral keys."""
from abc import ABC, abstractmethod
from typing import Iterator

import torch.utils.data


class KeyedDataset(ABC, torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that loads data from one or more files."""

    @abstractmethod
    def keys(self) -> Iterator[str]:
        """Get the dataset's keys."""

    @abstractmethod
    def key_to_index(self, key: str) -> int:
        """Get index of a given key in the dataset."""
