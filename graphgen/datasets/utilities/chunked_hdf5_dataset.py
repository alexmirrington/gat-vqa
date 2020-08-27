"""Utilities for loading data from one or more HDF5 files."""
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterator, List, Optional, Tuple

import h5py

from .chunked_dataset import ChunkedDataset
from .keyed_dataset import KeyedDataset


class ChunkedHDF5Dataset(ChunkedDataset, KeyedDataset):
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
        super().__init__(root)

        self._key_to_idx: Optional[Dict[str, int]] = {}
        self._chunk_sizes: Tuple[int, ...] = ()
        self._chunks, self._chunk_sizes, dataset_shapes = self._load_dataset_metadata()

        if len(self._chunks) == 0:
            raise ValueError(
                f"Parameter {root=} must contain one or more readable HDF5 ",
                "files if it is a directory, otherwise should be a readable ",
                "HDF5 file.",
            )

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
                    self._chunks[chunk_idx],
                    dataset,
                    shape=(chunk_size,) + shape,
                )
                layouts[dataset][chunk_start : chunk_start + chunk_size] = source
                chunk_start += chunk_size

        self._tempdir = TemporaryDirectory()
        self._vds = Path(self._tempdir.name) / "vds.h5"
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
        chunks = list(self._chunks)
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

    @property
    def chunk_sizes(self) -> Tuple[int, ...]:
        """Get the length of each of the chunks in the dataset."""
        return self._chunk_sizes

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        return {dset: self._data[dset][index] for dset in self._data.keys()}

    def __len__(self) -> int:
        """Get the length of the dataset."""
        key = list(self._data.keys())[0]
        return len(self._data[key])

    def keys(self) -> Iterator[str]:
        """Get the dataset's keys."""
        if self._key_to_idx is None:
            return iter([str(key) for key in range(len(self))])
        return iter(self._key_to_idx.keys())

    def key_to_index(self, key: str) -> int:
        """Get index of a given key in the dataset."""
        if self._key_to_idx is None:
            try:
                key_int = int(key)
                if 0 <= key_int < len(self):
                    return key_int
            except ValueError as ex:
                raise KeyError(
                    f"Parameter {key=} is not a valid key for the dataset."
                ) from ex
        elif key in self._key_to_idx:
            return self._key_to_idx[key]
        raise KeyError(f"Parameter {key=} is not a valid key for the dataset.")
