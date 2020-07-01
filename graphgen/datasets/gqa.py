"""Module containing a torch-compatible GQA dataset implementation."""
import json
import os.path
from enum import Enum
from typing import Any, Dict, Tuple

import torch.utils.data


class GQASplit(Enum):
    """An enum specifying possible values for GQA dataset splits."""

    TRAIN = "train"
    VAL = "val"
    DEV = "testdev"
    TEST = "test"
    CHALLENGE = "challenge"


class GQAVersion(Enum):
    """An enum specifying possible values for GQA dataset versions."""

    BALANCED = "balanced"
    ALL = "all"


class ChunkedJSONDataset(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that loads data from one or more JSON files."""

    def __init__(self, root: str) -> None:
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

        self._chunk_map: Dict[str, Tuple[str, ...]] = {}
        if os.path.isdir(self._root):
            self._chunk_map = {
                os.path.join(self._root, name): ()
                for name in sorted(os.listdir(self._root))
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
                print(os.path.relpath(chunk_name, self._root))
                print(self._chunk_map[chunk_name][:10])
                print(self._chunk_cache[chunk_name][self._chunk_map[chunk_name][0]])

    def get_chunk_name_and_local_index(self, index: int) -> Tuple[str, int]:
        """Get the name of the chunk containing the data item at index `index`.

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
            f"Parameter {index=} must be less than or equal to the total ",
            "number of keys across all chunks",
        )

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        # Get the name of the chunk the index belongs to and its corresponding
        # chunk-local index in the range [0, len(self.chunk_map[chunk_name]))
        chunk_name, local_index = self.get_chunk_name_and_local_index(index)

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


class GQAQuestions(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that retrieves GQA question samples."""

    def __init__(self, root: str, split: GQASplit, version: GQAVersion) -> None:
        """Initialise a `GQAQuestions` instance.

        Params:
        -------
        `root`: A path to the root directory of the GQA dataset.
        `split`: The dataset split to use.
        `version`: The dataset version to use.

        Returns:
        --------
        None
        """
        super().__init__()
        self._root = root
        self._split = split
        self._version = version

        self._filemap = self._init_filemap()
        self._data = ChunkedJSONDataset(
            self._filemap["questions"][self.split][self.version]
        )

    def _init_filemap(self) -> Any:
        return {
            "questions": {
                split: {
                    version: os.path.join(
                        self.root,
                        "questions",
                        f"{split.value}_{version.value}_questions",
                    )
                    if split == GQASplit.TRAIN and version == GQAVersion.ALL
                    else os.path.join(
                        self.root,
                        "questions",
                        f"{split.value}_{version.value}_questions.json",
                    )
                    for version in GQAVersion
                }
                for split in GQASplit
            }
        }

    @property
    def root(self) -> str:
        """Get the dataset root directory."""
        return self._root

    @property
    def split(self) -> GQASplit:
        """Get the dataset split."""
        return self._split

    @property
    def version(self) -> GQAVersion:
        """Get the dataset version."""
        return self._version

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._data)

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        return self._data[index]
