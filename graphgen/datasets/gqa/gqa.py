"""A torch-compatible GQA dataset implementation."""
from typing import Any, Optional

import torch.utils.data

from ...config.gqa import GQAFilemap, GQASplit, GQAVersion
from .images import GQAImages
from .objects import GQAObjects
from .questions import GQAQuestions
from .scene_graphs import GQASceneGraphs
from .spatial import GQASpatial


class GQA(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that retrieves GQA samples."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, filemap: GQAFilemap, split: GQASplit, version: GQAVersion
    ) -> None:
        """Initialise a `GQA` instance.

        Params:
        -------
        `filemap`: The filemap to use when determining where to load data from.
        `split`: The dataset split to use.
        `version`: The dataset version to use.

        Returns:
        --------
        None
        """
        super().__init__()
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

        self._filemap = filemap
        self._split = split
        self._version = version

        self._images = GQAImages(filemap)
        self._objects = GQAObjects(filemap)
        self._scene_graphs = (
            GQASceneGraphs(filemap, self._split)
            if self._split in (GQASplit.TRAIN, GQASplit.VAL)
            else None
        )
        self._spatial = GQASpatial(filemap)
        self._questions = GQAQuestions(filemap, self._split, self._version)

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
    def images(self) -> GQAImages:
        """Get the image portion of the dataset."""
        return self._images

    @property
    def objects(self) -> GQAObjects:
        """Get the objects portion of the dataset."""
        return self._objects

    @property
    def scene_graphs(self) -> Optional[GQASceneGraphs]:
        """Get the scene graphs portion of the dataset."""
        return self._scene_graphs

    @property
    def spatial(self) -> GQASpatial:
        """Get the spatial portion of the dataset."""
        return self._spatial

    @property
    def questions(self) -> GQAQuestions:
        """Get the questions portion of the dataset."""
        return self._questions

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        question = self._questions[index]
        image_id = question["imageId"]
        image = self._images[self._images.key_to_index(image_id)]
        objects = self._objects[self._objects.key_to_index(image_id)]
        spatial = self._spatial[self._spatial.key_to_index(image_id)]
        scene_graph = None
        if self._scene_graphs is not None:
            scene_graph = self._scene_graphs[self._scene_graphs.key_to_index(image_id)]

        return (question, image, spatial, objects, scene_graph)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._questions)

    def key_to_index(self, key: str) -> Any:
        """Get the index of the question in the dataset with a given question id."""
        return self._questions.key_to_index(key)
