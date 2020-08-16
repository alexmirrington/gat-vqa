"""Dataset abstractions and other data-related utilities."""
import json
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import h5py
import torch
import torch.utils.data
from PIL import Image
from torch import Tensor
from torchvision import transforms


class ChunkedDataset(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that loads data from one or more files."""

    def __init__(self, root: Path) -> None:
        """Initialise a `ChunkedDataset` instance.

        Params:
        -------
        `root`: A path to a single file or a folder containing multiple files
        (chunks) at its top level.

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
        self._chunks: Tuple[Path, ...] = tuple(
            sorted(self._root.iterdir())
        ) if self._root.is_dir() else (self._root,)

    @property
    def root(self) -> Path:
        """Get the root of the dataset, either a single file or a directory."""
        return self._root

    @property
    def chunks(self) -> Tuple[Path, ...]:
        """Get a tuple of containing the paths to the chunks in the dataset."""
        return self._chunks

    @property
    @abstractmethod
    def chunk_sizes(self) -> Tuple[int, ...]:
        """Get the length of each of the chunks in the dataset."""
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        """Get the length of the dataset."""
        raise NotImplementedError()


class ChunkedJSONDataset(ChunkedDataset):
    """A torch-compatible dataset that loads data from one or more JSON files."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        root: Path,
        tempdir: Optional[Path] = None,
        preprocessor: Optional[Callable[[Any], Any]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Initialise a `ChunkedJSONDataset` instance.

        Params:
        -------
        `root`: A path to a single JSON file or a folder containing multiple
        JSON files (chunks) at its top level.

        `tempdir`: A path to a directory that preprocessed files can be saved in.
        Preprocessed files are removed when the dataset is unloaded from memory,
        though files may persist if a process crashes. If `tempdir` is `None`,
        a system temporary directory will be used.

        `preprocessor`: A callable that preprocesses a single sample of the data.
        Preprocessing occurs on dataset creation, and preprocessed data is saved
        to disk.

        `transform`: A function that is applied to each sample in __getitem__,
        i.e. applied to the result of the `preprocessor` function for a sample,
        or to raw samples if `preprocessor` is `None`.
        """
        super().__init__(root)

        self._key_to_idx: Dict[str, int] = {}
        self._chunk_sizes: Tuple[int, ...] = ()
        self._chunk_cache: Dict[int, Tuple[Any, ...]] = {}

        self._tempdir = TemporaryDirectory(dir=tempdir)
        self._preprocessor = preprocessor
        self._transform = transform

        # Load top-level JSON keys into `self.chunk_map`
        cum_idx = 0
        chunk_sizes = []
        preprocessed_chunks = []
        for chunk_idx, chunk_name in enumerate(self._chunks):
            with open(chunk_name, "r") as chunk:
                chunk_data = json.load(chunk)
                chunk_size = len(chunk_data)
                self._key_to_idx.update(
                    {key: cum_idx + idx for idx, key in enumerate(chunk_data.keys())}
                )
                chunk_sizes.append(chunk_size)
                cum_idx += chunk_size
                # Preprocess data
                preprocessed_data = None
                if self._preprocessor is not None:
                    preprocessed_data = {
                        key: self._preprocessor(val) for key, val in chunk_data.items()
                    }
                del chunk_data
                # Save preprocessed data
                if preprocessed_data is not None:
                    pchunk = Path(self._tempdir.name) / f"{chunk_idx}.json"
                    with open(pchunk, "w") as file:
                        json.dump(preprocessed_data, file)
                    preprocessed_chunks.append(pchunk)
                del preprocessed_data

        self._chunk_sizes = tuple(chunk_sizes)
        if self._preprocessor is not None:
            self._root = Path(self._tempdir.name)
            self._chunks = tuple(preprocessed_chunks)

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
        chunk_idx, local_idx = self._get_chunk_local_idx(index)
        # Load the correct chunk into memory if not cached
        if self._chunk_cache is None or chunk_idx not in self._chunk_cache.keys():
            with open(self._chunks[chunk_idx], "r") as chunk:
                self._chunk_cache = {chunk_idx: tuple(json.load(chunk).values())}

        result = self._chunk_cache[chunk_idx][local_idx]
        if self._transform is not None:
            result = self._transform(result)

        return result

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return sum(self._chunk_sizes)

    def key_to_index(self, key: str) -> int:
        """Get index of a given key in the dataset."""
        return self._key_to_idx[key]


class ChunkedHDF5Dataset(ChunkedDataset):
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
                    self._chunks[chunk_idx], dataset, shape=(chunk_size,) + shape,
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

    def key_to_index(self, key: str) -> int:
        """Get index of a given key in the dataset."""
        if self._key_to_idx is None:
            try:
                key_int = int(key)
                if 0 <= key_int < len(self):
                    return key_int
            except ValueError:
                raise KeyError(f"Parameter {key=} is not a valid key for the dataset.")
        elif key in self._key_to_idx:
            return self._key_to_idx[key]
        raise KeyError(f"Parameter {key=} is not a valid key for the dataset.")


class ImageFolderDataset(ChunkedDataset):
    """A torch-compatible dataset that loads data from one or more image files."""

    def __init__(
        self, root: Path, transform: Optional[Callable[[Image.Image], Tensor]] = None
    ) -> None:
        """Initialise an `ImageFolderDataset` instance.

        Params:
        -------
        `root`: A path to a single image file or a folder containing multiple
        image files at its top level.

        Returns:
        --------
        None
        """
        super().__init__(root)
        self._key_to_idx: Dict[str, int] = {
            img.stem: idx for idx, img in enumerate(self._chunks)
        }
        self._transform = transform
        self._chunk_sizes = tuple([1 for _ in range(len(self._key_to_idx))])

    @property
    def chunk_sizes(self) -> Tuple[int, ...]:
        """Get the length of each of the chunks in the dataset."""
        return self._chunk_sizes

    def __getitem__(self, index: int) -> Tensor:
        """Get an item from the dataset at a given index."""
        img_path = self._chunks[index]
        img = Image.open(img_path)
        transform = (
            self._transform if self._transform is not None else transforms.ToTensor()
        )
        return transform(img)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._chunks)

    def key_to_index(self, key: str) -> int:
        """Get index of a given key in the dataset."""
        return self._key_to_idx[key]


class ChunkedRandomSampler:
    """Custom sampler that performs a chunked shuffle fo maximise cache hits."""

    def __init__(
        self, data_source: ChunkedDataset, generator: torch.Generator = None
    ) -> None:
        """Create a new ChunkedRandomSampler instance.

        Params:
        -------
        `data_source`: Dataset to sample from.
        `generator`: Generator used in sampling.
        """
        self.data_source = data_source
        self.generator = generator

    def __iter__(self) -> Iterable[int]:
        """Get an iterator for the sampler instance."""
        chunk_bounds = self.data_source.chunk_sizes
        # Permute items inside chunks
        perms = []
        start = 0
        for bound in chunk_bounds:
            perms.append(
                (torch.randperm(bound, generator=self.generator) + start).tolist()
            )
            start += bound
        # Permute chunks
        chunk_perm = torch.randperm(len(perms), generator=self.generator).tolist()
        result = []
        for cidx in chunk_perm:
            result += perms[cidx]
        return iter(result)

    def __len__(self) -> int:
        """Get the length of the sampler's data source."""
        return len(self.data_source)
