"""A torch-compatible GQA object features dataset implementation."""
import json
from typing import Any

import numpy as np
import torch.utils.data
from PIL import Image

from ...config.gqa import GQAFilemap


class GQAImages(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that retrieves GQA image samples."""

    def __init__(self, filemap: GQAFilemap) -> None:
        """Initialise a `GQAObjects` instance.

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

        # Validate the objects data root file/directory
        objects_root = self._filemap.object_path()
        if not objects_root.exists():
            raise ValueError(
                f"Parameter {filemap=} does not refer to a valid object "
                "features directory."
            )

        image_meta = self._filemap.object_meta_path()
        with open(image_meta, "r") as file:
            meta = json.load(file)
            self._index_to_key = list(meta.keys())
            self._index_to_meta = [
                {
                    "width": val["width"],
                    "height": val["height"],
                    "objects": val["objectsNum"],
                }
                for val in meta.values()
            ]
            self._key_to_index = {
                key: idx for idx, key in enumerate(self._index_to_key)
            }

    @property
    def filemap(self) -> GQAFilemap:
        """Get the dataset's filemap."""
        return self._filemap

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._index_to_key)

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        img_id = self._index_to_key[index]
        img_path = self._filemap.image_path(img_id)
        img = Image.open(img_path)
        return np.asarray(img)

    def key_to_index(self, key: str) -> Any:
        """Get the index of the image in the dataset with a given image id."""
        return self._key_to_index[key]
