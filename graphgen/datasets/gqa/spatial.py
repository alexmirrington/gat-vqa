"""A torch-compatible GQA spatial features dataset implementation."""
import json
from typing import Any

import torch.utils.data

from ...config.gqa import GQAFilemap
from ..utilities import ChunkedHDF5Dataset


class GQASpatial(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that retrieves GQA spatial feature samples."""

    def __init__(self, filemap: GQAFilemap) -> None:
        """Initialise a `GQASpatial` instance.

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

        self._filemap = filemap
        # Validate the spatial data root file/directory
        spatial_root = self._filemap.spatial_path()
        if not spatial_root.exists():
            raise ValueError(
                f"Parameter {filemap=} does not refer to a valid spatial "
                "features directory."
            )

        spatial_meta = self._filemap.spatial_meta_path()
        with open(spatial_meta, "r") as file:
            meta = json.load(file)
            chunk_map = {
                key: (self._filemap.spatial_path(val["file"]), val["idx"])
                for key, val in meta.items()
            }
        print(chunk_map)

        self._data = ChunkedHDF5Dataset(spatial_root, chunk_map)

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
        """Get the index of the spatial feature in the dataset with a given image id."""
        return self._data.key_to_index(key)
