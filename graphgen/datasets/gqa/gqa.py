"""A torch-compatible GQA dataset implementation."""
from typing import Any, Optional

import torch.utils.data

from ...config.gqa import GQASplit, GQAVersion
from .images import GQAImages
from .objects import GQAObjects
from .questions import GQAQuestions
from .scene_graphs import GQASceneGraphs
from .spatial import GQASpatial


class GQA(torch.utils.data.Dataset):  # type: ignore
    """A torch-compatible dataset that retrieves GQA samples."""

    def __init__(
        self,
        questions: GQAQuestions,
        images: Optional[GQAImages] = None,
        objects: Optional[GQAObjects] = None,
        spatial: Optional[GQASpatial] = None,
        scene_graphs: Optional[GQASceneGraphs] = None,
    ) -> None:
        """Initialise a `GQA` instance.

        Params:
        -------
        `split`: The dataset split to use.

        `version`: The dataset version to use.

        `questions`: The GQAQuestions dataset to use. This dataset must have the
        same split and version as the one supplied.
        """
        super().__init__()

        self._questions = questions
        self._images = images
        self._objects = objects
        self._spatial = spatial

        if scene_graphs is not None and scene_graphs.split != self._questions.split:
            raise ValueError(
                f"{GQASceneGraphs.__name__} split does not match ",
                f"{GQAQuestions.__name__} split.",
            )

        self._scene_graphs = scene_graphs

        self._split = self._questions.split
        self._version = self._questions.version

    @property
    def split(self) -> GQASplit:
        """Get the dataset split."""
        return self._split

    @property
    def version(self) -> GQAVersion:
        """Get the dataset version."""
        return self._version

    @property
    def questions(self) -> GQAQuestions:
        """Get the scene graphs portion of the dataset."""
        return self._questions

    @property
    def images(self) -> Optional[GQAImages]:
        """Get the images portion of the dataset."""
        return self._images

    @property
    def objects(self) -> Optional[GQAObjects]:
        """Get the object features portion of the dataset."""
        return self._objects

    @property
    def spatial(self) -> Optional[GQASpatial]:
        """Get the spatial features portion of the dataset."""
        return self._spatial

    @property
    def scene_graphs(self) -> Optional[GQASceneGraphs]:
        """Get the scene graphs portion of the dataset."""
        return self._scene_graphs

    def __getitem__(self, index: int) -> Any:
        """Get an item from the dataset at a given index."""
        question = self._questions[index]
        result = {"question": question}

        image_id = question["imageId"]

        if self._images is not None:
            result["image"] = self._images[self._images.key_to_index(image_id)]

        if self._objects is not None:
            objects, boxes = self._objects[self._objects.key_to_index(image_id)]
            result["objects"] = objects
            result["boxes"] = boxes

        if self._spatial is not None:
            spatial = self._spatial[self._spatial.key_to_index(image_id)]
            result["spatial"] = spatial

        if self._scene_graphs is not None:
            scene_graph = self._scene_graphs[self._scene_graphs.key_to_index(image_id)]
            result["scene_graph"] = scene_graph

        return result

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._questions)

    def key_to_index(self, key: str) -> Any:
        """Get the index of the question in the dataset with a given question id."""
        return self._questions.key_to_index(key)
