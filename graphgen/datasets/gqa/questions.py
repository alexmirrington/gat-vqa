"""A torch-compatible GQA questions dataset implementation."""
from pathlib import Path
from typing import Any, Callable, Optional

from ...config.gqa import GQAFilemap, GQASplit, GQAVersion
from ..utilities import ChunkedJSONDataset


class GQAQuestions(ChunkedJSONDataset):
    """A torch-compatible dataset that retrieves GQA question samples."""

    def __init__(
        self,
        filemap: GQAFilemap,
        split: GQASplit,
        version: GQAVersion,
        tempdir: Optional[Path] = None,
        preprocessor: Optional[Callable[[Any], Any]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Initialise a `GQAQuestions` instance.

        Params:
        -------
        `filemap`: The filemap to use when determining where to load data from.

        `split`: The dataset split to use.

        `version`: The dataset version to use.

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

        super().__init__(
            root, tempdir=tempdir, preprocessor=preprocessor, transform=transform
        )

        self._filemap = filemap
        self._split = split
        self._version = version

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
