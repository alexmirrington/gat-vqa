"""A torch-compatible GQA object features dataset implementation."""
import json
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from ...config.gqa import GQAFilemap
from ..utilities import ChunkedHDF5Dataset


class GQAObjects(ChunkedHDF5Dataset):
    """A torch-compatible dataset that retrieves GQA object feature samples."""

    def __init__(
        self,
        filemap: GQAFilemap,
        transform: Optional[
            Callable[
                [Tensor, Tensor, Dict[str, Any]], Tuple[Tensor, Tensor, Dict[str, Any]]
            ]
        ] = None,
    ) -> None:
        """Initialise a `GQAObjects` instance.

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

        root = filemap.object_path()
        if not root.exists():
            raise ValueError(
                f"Parameter {filemap=} does not refer to a valid object "
                "features directory."
            )

        objects_meta = filemap.object_meta_path()
        with open(objects_meta, "r") as file:
            meta = json.load(file)
            chunk_map = {
                key: (filemap.object_path(val["file"]), val["idx"])
                for key, val in meta.items()
            }

        super().__init__(root, chunk_map)

        self._meta = {
            self.key_to_index(image_id): obj_meta for image_id, obj_meta in meta.items()
        }
        self._filemap = filemap
        self._transform = transform

    @property
    def filemap(self) -> GQAFilemap:
        """Get the dataset's filemap."""
        return self._filemap

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Get an item from the dataset at a given index."""
        data = super().__getitem__(index)
        meta = self._meta[index]
        if self._transform is not None:
            return self._transform(
                torch.tensor(data["features"]),  # pylint: disable=not-callable
                torch.tensor(data["bboxes"]),  # pylint: disable=not-callable
                meta,
            )
        return (
            torch.tensor(data["features"]),  # pylint: disable=not-callable
            torch.tensor(data["bboxes"]),  # pylint: disable=not-callable
            meta,
        )
