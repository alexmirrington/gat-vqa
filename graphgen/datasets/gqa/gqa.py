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
        split: GQASplit,
        version: GQAVersion,
        questions: Optional[GQAQuestions] = None,
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

        if not isinstance(split, GQASplit):
            raise TypeError(f"Parameter {split=} must be of type {GQASplit.__name__}")

        if not isinstance(version, GQAVersion):
            raise TypeError(
                f"Parameter {version=} must be of type {GQAVersion.__name__}"
            )

        if questions is not None and questions.split != split:
            raise ValueError(
                f"{GQAQuestions.__name__} split does not match ",
                f"{GQA.__name__} split.",
            )

        if questions is not None and questions.version != version:
            raise ValueError(
                f"{GQAQuestions.__name__} version does not match ",
                f"{GQA.__name__} version.",
            )

        if scene_graphs is not None and scene_graphs.split != split:
            raise ValueError(
                f"{GQASceneGraphs.__name__} split does not match ",
                f"{GQA.__name__} split.",
            )

        self._split = split
        self._version = version
        self._questions = questions
        self._images = images
        self._objects = objects
        self._spatial = spatial
        self._scene_graphs = scene_graphs

    @property
    def split(self) -> GQASplit:
        """Get the dataset split."""
        return self._split

    @property
    def version(self) -> GQAVersion:
        """Get the dataset version."""
        return self._version

    @property
    def questions(self) -> Optional[GQAQuestions]:
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
        result = {}
        if self._questions is not None:
            question = self._questions[index]
            result["question"] = question

        if self._images is not None:
            image_id = (
                self._images.key_to_index(question["imageId"])
                if self._questions is not None
                else index
            )
            result["image"] = self._images[image_id]

        if self._objects is not None:
            image_id = (
                self._objects.key_to_index(question["imageId"])
                if self._questions is not None
                else index
            )
            objects, boxes = self._objects[image_id]
            result["objects"] = objects
            result["boxes"] = boxes

        if self._spatial is not None:
            image_id = (
                self._spatial.key_to_index(question["imageId"])
                if self._questions is not None
                else index
            )
            spatial = self._spatial[image_id]
            result["spatial"] = spatial

        if self._scene_graphs is not None:
            image_id = (
                self._scene_graphs.key_to_index(question["imageId"])
                if self._questions is not None
                else index
            )
            scene_graph = self._scene_graphs[image_id]
            result["scene_graph"] = scene_graph

        if len(result) == 0:
            raise IndexError("No keys exist for this dataset.")

        return result

    def __len__(self) -> int:
        """Get the length of the dataset."""
        for dataset in (
            self._questions,
            self._images,
            self._spatial,
            self._objects,
            self.scene_graphs,
        ):
            if dataset is not None:
                return len(dataset)
        return 0

    def key_to_index(self, key: str) -> Any:
        """Get the index of the question in the dataset with a given question id."""
        for dataset in (
            self._questions,
            self._images,
            self._spatial,
            self._objects,
            self.scene_graphs,
        ):
            if dataset is not None:
                return dataset.key_to_index(key)
        raise KeyError("No keys exist for this dataset.")
