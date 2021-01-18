"""A torch-compatible GQA questions dataset implementation."""
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
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Initialise a `GQAQuestions` instance.

        Params:
        -------
        `filemap`: The filemap to use when determining where to load data from.

        `split`: The dataset split to use.

        `version`: The dataset version to use.

        `preprocessor`: A callable that preprocesses a single sample of the data.
        Preprocessing occurs on dataset creation, and preprocessed data is saved
        to disk.
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

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        qid, question = super().__getitem__(index)
        question["questionId"] = qid
        if self._transform is not None:
            return self._transform(question)
        return question
