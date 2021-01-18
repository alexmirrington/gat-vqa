"""A torch-compatible GQA images dataset implementation."""
from typing import Callable, Optional

from PIL import Image
from torch import Tensor

from ...config.gqa import GQAFilemap
from ..utilities import ImageFolderDataset


class GQAImages(ImageFolderDataset):
    """A torch-compatible dataset that retrieves GQA image samples."""

    def __init__(
        self,
        filemap: GQAFilemap,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
    ) -> None:
        """Initialise a `GQAImages` instance.

        Params:
        -------
        `filemap`: The filemap to use when determining where to load data from.
        """
        if not isinstance(filemap, GQAFilemap):
            raise TypeError(
                f"Parameter {filemap=} must be of type {GQAFilemap.__name__}."
            )

        root = filemap.image_path()
        if not root.exists():
            raise ValueError(
                f"Parameter {filemap=} does not refer to a valid images directory."
            )

        super().__init__(root, transform)

        self._filemap = filemap

    @property
    def filemap(self) -> GQAFilemap:
        """Get the dataset's filemap."""
        return self._filemap
