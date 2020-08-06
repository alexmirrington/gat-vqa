"""Dataset abstractions and other data-related utilities."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import h5py
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

        if not isinstance(root, Path):
            raise TypeError(f"Parameter {root=} must be of type {Path.__name__}.")

        if not root.exists():
            raise ValueError(f"Parameter {root=} must point to a file or directory.")

        if root.is_dir() and len(tuple(root.iterdir())) == 0:
            raise ValueError(f"Parameter {root=} must point to a non-empty directory.")

        self._root = root
        self._chunks: Tuple[Path, ...] = ()
        self._chunk_sizes: Tuple[int, ...] = ()
        self._chunk_cache = None

        self._key_to_idx: Dict[str, int] = {}

        if self._root.is_dir():
            self._chunks = tuple(sorted(self._root.iterdir()))
        else:
            self._chunks = (self._root,)

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
                self._chunk_cache = {chunk_idx: tuple(chunk_data.values())}
                chunk_sizes.append(chunk_size)
                cum_idx += chunk_size
                del chunk_data

        self._chunk_sizes = tuple(chunk_sizes)

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

        raise ValueError(
            f"Parameter {index=} must be less than or equal to the total "
            "number of keys across all chunks."
        )

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        # Get the index of the chunk the given index belongs to and its
        # corresponding chunk-local index in the range
        # [0, self._chunk_sizes[chunk_idx]))
        chunk_idx, local_idx = self._get_chunk_local_idx(index)
        # Load the correct chunk into memory if not cached
        if self._chunk_cache is None or chunk_idx not in self._chunk_cache.keys():
            with open(self._chunks[chunk_idx], "r") as chunk:
                self._chunk_cache = {chunk_idx: tuple(json.load(chunk).values())}

        return self._chunk_cache[chunk_idx][local_idx]

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return sum(self._chunk_sizes)

    def key_to_index(self, key: str) -> int:
        """Get index of a given key in the dataset."""
        return self._key_to_idx[key]


class ChunkedHDF5Dataset(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that loads data from one or more HDF5 files."""

    def __init__(
        self, root: Path, chunk_map: Optional[Dict[str, Tuple[Path, int]]] = None
    ) -> None:
        """Initialise a `ChunkedHDF5Dataset` instance.

        Params:
        -------
        `root`: A path to a single HDF5 file or a folder containing multiple
        HDF5 files (chunks) at its top level.

        Returns:
        --------
        None
        """
        super().__init__()

        if not isinstance(root, Path):
            raise TypeError(f"Parameter {root=} must be of type {Path.__name__}.")

        if not root.exists():
            raise ValueError(f"Parameter {root=} must point to a file or directory.")

        if root.is_dir() and len(tuple(root.iterdir())) == 0:
            raise ValueError(f"Parameter {root=} must point to a non-empty directory.")

        self._root = root
        self._key_to_idx: Optional[Dict[str, int]] = {}
        self._chunks: Tuple[Path, ...] = ()
        self._chunk_sizes: Tuple[int, ...] = ()

        self._chunks, self._chunk_sizes, dataset_shapes = self._load_dataset_metadata()

        # Determine concatenated dataset shape and create virtual dataset
        cat_shapes = {
            dataset: (sum(self._chunk_sizes),) + shape
            for dataset, shape in dataset_shapes.items()
        }
        layouts = {
            dataset: h5py.VirtualLayout(shape=shape)
            for dataset, shape in cat_shapes.items()
        }
        for dataset, shape in dataset_shapes.items():
            chunk_start = 0
            for chunk_idx, chunk_size in enumerate(self._chunk_sizes):
                source = h5py.VirtualSource(
                    self._chunks[chunk_idx], dataset, shape=(chunk_size,) + shape,
                )
                layouts[dataset][chunk_start : chunk_start + chunk_size] = source
                chunk_start += chunk_size

        self._tempdir = TemporaryDirectory()
        self._vds = Path(self._tempdir.name) / str(uuid4())
        with h5py.File(self._vds, "w") as file:
            for dataset, layout in layouts.items():
                file.create_virtual_dataset(dataset, layout)

        self._data = h5py.File(self._vds, "r")

        self._key_to_idx = (
            {
                image_id: sum(
                    [self._chunk_sizes[i] for i in range(self._chunks.index(chunk))]
                )
                + idx
                for image_id, (chunk, idx) in chunk_map.items()
            }
            if chunk_map is not None
            else None
        )

    def _load_dataset_metadata(
        self,
    ) -> Tuple[Tuple[Path, ...], Tuple[int, ...], Dict[str, Tuple[Any, ...]]]:
        chunks = (
            list(sorted(self._root.iterdir())) if self._root.is_dir() else [self._root]
        )

        dataset_shapes: Dict[str, Tuple[Any, ...]] = {}
        chunk_sizes: List[int] = []
        to_remove = []
        for chunk_idx, chunk_name in enumerate(chunks):
            try:
                with h5py.File(chunk_name, "r") as chunk:
                    # Get shapes of datasets
                    shape = {
                        dataset: tuple(chunk[dataset].shape) for dataset in chunk.keys()
                    }

                    # Set dataset shapes
                    if len(dataset_shapes) == 0:
                        dataset_shapes = {
                            dataset: val[1:] for dataset, val in shape.items()
                        }

                    # Enforce datasets are the same length in first dimension.
                    chunk_size = None
                    for dataset_shape in shape.values():
                        if chunk_size is None:
                            chunk_size = dataset_shape[0]
                            chunk_sizes.append(chunk_size)
                        elif chunk_size != dataset_shape[0]:
                            raise ValueError(
                                f"Chunk {chunk_name} must have datasets of ",
                                "equal length.",
                            )

                    # Enforce dataset existence and check shapes across chunks
                    for dataset, val in shape.items():
                        if dataset not in dataset_shapes.keys():
                            raise ValueError(
                                f"Chunk {chunk_name} must only contain ",
                                "datasets present in other chunks.",
                            )
                        if dataset_shapes[dataset] != val[1:]:
                            raise ValueError(
                                f"Dataset {dataset} in chunk {chunk_name} must ",
                                "have the same shape as its corresponding ",
                                "dataset in other chunks. Expected shape ",
                                f"{dataset_shapes[dataset][0]} but got {val[1:]}",
                            )
            except OSError:
                to_remove.append(chunk_name)

        for chunk in to_remove:
            chunks.remove(chunk)

        return tuple(chunks), tuple(chunk_sizes), dataset_shapes

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        return {key: self._data[key][index] for key in self._data.keys()}

    def __len__(self) -> int:
        """Get the length of the dataset."""
        key = list(self._data.keys())[0]
        return len(self._data[key])

    def key_to_index(self, key: str) -> int:
        """Get index of a given key in the dataset."""
        if self._key_to_idx is None:
            key_int = int(key)
            if 0 <= key_int < len(self):
                return key_int
        elif key in self._key_to_idx:
            return self._key_to_idx[key]
        raise ValueError(f"Parameter {key=} is not a valid key for the dataset.")
