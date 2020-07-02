"""Dataset abstractions and other data-related utilities."""
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch.utils.data


class ChunkedJSONDataset(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that loads data from one or more JSON files."""

    def __init__(self, root: Path) -> None:
        """Initialise a `ChunkedJSONDataset` instance.

        Params:
        -------
        `root`: A path to a single JSON file or a folder containing multiple
        JSON files (chunks) at its top level.

        Returns:
        --------
        None
        """
        super().__init__()

        self._root = root
        self._chunk_cache = None

        self._chunk_map: Dict[Path, Tuple[str, ...]] = {}
        if self._root.is_dir():
            self._chunk_map = {
                self._root / name: () for name in sorted(self._root.iterdir())
            }
        else:
            self._chunk_map = {self._root: ()}

        # Load top-level JSON keys into `self.chunk_map`
        for chunk_name in self._chunk_map.keys():
            with open(chunk_name, "r") as chunk:
                chunk_data = json.load(chunk)
                self._chunk_map[chunk_name] = tuple(chunk_data.keys())
                self._chunk_cache = {chunk_name: chunk_data}
                del chunk_data

    def get_chunk_path_and_local_index(self, index: int) -> Tuple[Path, int]:
        """Get the path of the chunk containing the data item at index `index`.

        Params:
        -------
        `index`: An index in the range `[0, len(self))`, the index of the data
        item to retrieve.

        Returns:
        --------
        A tuple containing the name of the chunk containing the data item at
        index `index` and the index of that data item within its chunk.
        """
        if index < 0:
            raise ValueError(f"Parameter {index=} must be greater than or equal to 0")

        cumulative_count = 0
        for chunk_name, chunk_keys in self._chunk_map.items():
            if cumulative_count <= index < cumulative_count + len(chunk_keys):
                return chunk_name, index - cumulative_count
        raise ValueError(
            f"Parameter {index=} must be less than or equal to the total "
            "number of keys across all chunks."
        )

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        # Get the name of the chunk the index belongs to and its corresponding
        # chunk-local index in the range [0, len(self.chunk_map[chunk_name]))
        chunk_name, local_index = self.get_chunk_path_and_local_index(index)

        # Load the correct chunk into memory if not cached
        if self._chunk_cache is None or chunk_name not in self._chunk_cache.keys():
            with open(chunk_name, "r") as chunk:
                self._chunk_cache = {chunk_name: json.load(chunk)}

        # Get item key and return item
        item_key = self._chunk_map[chunk_name][local_index]
        return self._chunk_cache[chunk_name][item_key]

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return sum([len(chunk_keys) for chunk_keys in self._chunk_map.values()])
