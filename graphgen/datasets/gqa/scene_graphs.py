"""A torch-compatible GQA scene graphs dataset implementation."""
from typing import Any

import torch.utils.data

from ...config.gqa import GQAFilemap, GQASplit
from ..utilities import ChunkedJSONDataset


class GQASceneGraphs(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that retrieves GQA scene graph samples."""

    def __init__(self, filemap: GQAFilemap, split: GQASplit) -> None:
        """Initialise a `GQASceneGraphs` instance.

        Params:
        -------
        `filemap`: The filemap to use when determining where to load data from.
        `split`: The dataset split to use.

        Returns:
        --------
        None
        """
        super().__init__()
        if not isinstance(filemap, GQAFilemap):
            raise TypeError(
                f"Parameter {filemap=} must be of type {GQAFilemap.__name__}."
            )

        if not isinstance(split, GQASplit):
            raise TypeError(f"Parameter {split=} must be of type {GQASplit.__name__}.")

        if split not in (GQASplit.TRAIN, GQASplit.VAL):
            raise ValueError(
                f"Parameter {split=} must be one of {(GQASplit.TRAIN, GQASplit.VAL)}."
            )

        self._filemap = filemap
        self._split = split

        # Validate the scene_graphs root file/directory
        scene_graphs_root = self._filemap.scene_graph_path(self._split,)
        if not scene_graphs_root.exists():
            raise ValueError(
                f"Parameter {filemap=} does not refer to a valid scene graph"
                f"file/folder for {split=}."
            )

        self._data = ChunkedJSONDataset(scene_graphs_root)

    @property
    def filemap(self) -> GQAFilemap:
        """Get the dataset's filemap."""
        return self._filemap

    @property
    def split(self) -> GQASplit:
        """Get the dataset split."""
        return self._split

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._data)

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        return self._data[index]

    def key_to_index(self, key: str) -> Any:
        """Get the index of the scene graph in the dataset with a given question id."""
        return self._data.key_to_index(key)
