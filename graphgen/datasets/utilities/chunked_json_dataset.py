"""Utilities for loading data from one or more JSON files."""
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from .chunked_dataset import ChunkedDataset
from .keyed_dataset import KeyedDataset


class ChunkedJSONDataset(ChunkedDataset, KeyedDataset):
    """A torch-compatible dataset that loads data from one or more JSON files."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, root: Path) -> None:
        """Initialise a `ChunkedJSONDataset` instance.

        Params:
        -------
        `root`: A path to a single JSON file or a folder containing multiple
        JSON files (chunks) at its top level.

        `cache`: A path to a directory that preprocessed files can be saved in.

        `preprocessor`: A callable that preprocesses a single sample of the data.
        Preprocessing occurs on dataset creation, and preprocessed data is saved
        to disk.
        """
        super().__init__(root)

        self._key_to_idx: Dict[str, int] = {}
        self._idx_to_key: List[str] = []
        self._chunk_sizes: Tuple[int, ...] = ()
        self._chunk_cache: Dict[int, Tuple[Any, ...]] = {}

        # Load top-level JSON keys into `self.chunk_map`
        cum_idx = 0
        chunk_sizes = []
        for chunk_idx, chunk_name in enumerate(self._chunks):
            with open(chunk_name, "r") as chunk:
                chunk_data = json.load(chunk)
                chunk_size = len(chunk_data)
                self._key_to_idx.update(
                    {key: cum_idx + idx for idx, key in enumerate(chunk_data.keys())}
                )
                self._idx_to_key += list(chunk_data.keys())
                chunk_sizes.append(chunk_size)
                cum_idx += chunk_size
                del chunk_data

        self._chunk_sizes = tuple(chunk_sizes)

    @property
    def chunk_sizes(self) -> Tuple[int, ...]:
        """Get the length of each of the chunks in the dataset."""
        return self._chunk_sizes

    def _get_chunk_local_idx(self, index: int) -> Tuple[int, int]:
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
        chunk_start_idx = 0
        for chunk_idx, chunk_size in enumerate(self._chunk_sizes):
            if chunk_start_idx <= index < chunk_start_idx + chunk_size:
                return chunk_idx, index - chunk_start_idx
            chunk_start_idx += chunk_size

        raise IndexError(
            f"Parameter {index=} must be less than or equal to the total "
            "number of keys across all chunks."
        )

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        # Get the index of the chunk the given index belongs to and its
        # corresponding chunk-local index in the range
        # [0, self._chunk_sizes[chunk_idx]))
        key = self._idx_to_key[index]
        chunk_idx, local_idx = self._get_chunk_local_idx(index)
        # Load the correct chunk into memory if not cached
        if self._chunk_cache is None or chunk_idx not in self._chunk_cache.keys():
            with open(self._chunks[chunk_idx], "r") as chunk:
                self._chunk_cache = {chunk_idx: tuple(json.load(chunk).values())}

        return key, self._chunk_cache[chunk_idx][local_idx]

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return sum(self._chunk_sizes)

    def keys(self) -> Iterator[str]:
        """Get the dataset's keys."""
        return iter(self._key_to_idx.keys())

    def key_to_index(self, key: str) -> int:
        """Get index of a given key in the dataset."""
        return self._key_to_idx[key]
