"""A torch-compatible GQA questions dataset implementation."""
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from ...config.gqa import GQAFilemap, GQASplit, GQAVersion
from ...utilities.preprocessing import Preprocessor
from ..utilities import ChunkedDataset, ChunkedJSONDataset, PreprocessedJSONDataset


class GQAQuestions(ChunkedDataset):
    """A torch-compatible dataset that retrieves GQA question samples."""

    def __init__(
        self,
        filemap: GQAFilemap,
        split: GQASplit,
        version: GQAVersion,
        cache: Optional[Path] = None,
        preprocessor: Optional[Preprocessor] = None,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Initialise a `GQAQuestions` instance.

        Params:
        -------
        `filemap`: The filemap to use when determining where to load data from.

        `split`: The dataset split to use.

        `version`: The dataset version to use.

        `cache`: A path to a directory that preprocessed files can be saved in.
        If `cache` is `None`, a system temporary directory will be used.

        `preprocessor`: A callable that preprocesses a single sample of the data.
        Preprocessing occurs on dataset creation, and preprocessed data is saved
        to disk.

        `transform`: A function that is applied to each sample in __getitem__,
        i.e. applied to the result of the `preprocessor` function for a sample,
        or to raw samples if `preprocessor` is `None`.
        """
        # Validate parameters
        if not isinstance(filemap, GQAFilemap):
            raise TypeError(
                f"Parameter {filemap=} must be of type {GQAFilemap.__name__}."
            )

        if not isinstance(split, GQASplit):
            raise TypeError(f"Parameter {split=} must be of type {GQASplit.__name__}.")

        if not isinstance(version, GQAVersion):
            raise TypeError(
                f"Parameter {version=} must be of type {GQAVersion.__name__}."
            )

        root = filemap.question_path(
            split,
            version,
            chunked=(split == GQASplit.TRAIN and version == GQAVersion.ALL),
        )

        if not root.exists():
            raise ValueError(
                f"Parameter {filemap=} does not refer to a valid questions"
                f"file or directory for {split=} and {version=}."
            )
        super().__init__(root)

        self._data = ChunkedJSONDataset(root)
        if preprocessor is not None and cache is not None:
            self._data = PreprocessedJSONDataset(
                self._data, cache=cache, preprocessor=preprocessor
            )

        self._filemap = filemap
        self._split = split
        self._version = version
        self._transform = transform

    @property
    def filemap(self) -> GQAFilemap:
        """Get the dataset's filemap."""
        return self._filemap

    @property
    def split(self) -> GQASplit:
        """Get the dataset split."""
        return self._split

    @property
    def version(self) -> GQAVersion:
        """Get the dataset version."""
        return self._version

    @property
    def chunk_sizes(self) -> Tuple[int, ...]:
        """Get the length of each of the chunks in the dataset."""
        return self._data.chunk_sizes

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        item = self._data[index]
        if self._transform is not None:
            return self._transform(item)
        return item

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._data)

    def key_to_index(self, key: str) -> int:
        """Get index of a given key in the dataset."""
        return self._data.key_to_index(key)
