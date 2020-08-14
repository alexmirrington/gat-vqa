"""A torch-compatible GQA images dataset implementation."""
from typing import Any

import torch.utils.data

from ...config.gqa import GQAFilemap
from ..utilities import ImageFolderDataset


class GQAImages(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that retrieves GQA image samples."""

    def __init__(self, filemap: GQAFilemap) -> None:
        """Initialise a `GQAImages` instance.

        Params:
        -------
        `filemap`: The filemap to use when determining where to load data from.

        Returns:
        --------
        None
        """
        super().__init__()
        if not isinstance(filemap, GQAFilemap):
            raise TypeError(
                f"Parameter {filemap=} must be of type {GQAFilemap.__name__}."
            )

        images_root = filemap.image_path()
        if not images_root.exists():
            raise ValueError(
                f"Parameter {filemap=} does not refer to a valid images directory."
            )

        self._filemap = filemap
        self._data = ImageFolderDataset(images_root)

    @property
    def filemap(self) -> GQAFilemap:
        """Get the dataset's filemap."""
        return self._filemap

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._data)

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        return self._data[index]

    def key_to_index(self, key: str) -> Any:
        """Get the index of the image in the dataset with a given image id."""
        return self._data.key_to_index(key)
