"""Utilities for loading data from one or more image files."""
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple

import torchvision.transforms as T
from PIL import Image
from torch import Tensor

from .chunked_dataset import ChunkedDataset
from .keyed_dataset import KeyedDataset


class ImageFolderDataset(ChunkedDataset, KeyedDataset):
    """A torch-compatible dataset that loads data from one or more image files."""

    def __init__(
        self, root: Path, transform: Optional[Callable[[Image.Image], Tensor]] = None
    ) -> None:
        """Initialise an `ImageFolderDataset` instance.

        Params:
        -------
        `root`: A path to a single image file or a folder containing multiple
        image files at its top level.

        Returns:
        --------
        None
        """
        super().__init__(root)
        self._key_to_idx: Dict[str, int] = {
            img.stem: idx for idx, img in enumerate(self._chunks)
        }
        self._transform = transform
        self._chunk_sizes = tuple([1 for _ in range(len(self._key_to_idx))])

    @property
    def chunk_sizes(self) -> Tuple[int, ...]:
        """Get the length of each of the chunks in the dataset."""
        return self._chunk_sizes

    def __getitem__(self, index: int) -> Tensor:
        """Get an item from the dataset at a given index."""
        img_path = self._chunks[index]
        img = Image.open(img_path)
        transform = self._transform if self._transform is not None else T.ToTensor()
        return transform(img)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._chunks)

    def keys(self) -> Iterator[str]:
        """Get the dataset's keys."""
        return iter(self._key_to_idx.keys())

    def key_to_index(self, key: str) -> int:
        """Get index of a given key in the dataset."""
        return self._key_to_idx[key]
