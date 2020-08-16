"""A torch-compatible GQA spatial features dataset implementation."""
import json

import torch
from torch import Tensor

from ...config.gqa import GQAFilemap
from ..utilities import ChunkedHDF5Dataset


class GQASpatial(ChunkedHDF5Dataset):
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
        if not isinstance(filemap, GQAFilemap):
            raise TypeError(
                f"Parameter {filemap=} must be of type {GQAFilemap.__name__}."
            )

        root = filemap.spatial_path()
        if not root.exists():
            raise ValueError(
                f"Parameter {filemap=} does not refer to a valid spatial "
                "features directory."
            )

        spatial_meta = filemap.spatial_meta_path()
        with open(spatial_meta, "r") as file:
            meta = json.load(file)
            chunk_map = {
                key: (filemap.spatial_path(val["file"]), val["idx"])
                for key, val in meta.items()
            }

        super().__init__(root, chunk_map)

        self._filemap = filemap

    @property
    def filemap(self) -> GQAFilemap:
        """Get the dataset's filemap."""
        return self._filemap

    def __getitem__(self, index: int) -> Tensor:
        """Get an item from the dataset at a given index."""
        sample = super().__getitem__(index)
        return torch.tensor(sample["features"])  # pylint: disable=not-callable
