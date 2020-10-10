"""Common schema definitions."""

from typing import List, Optional, Tuple, TypedDict

import torch
from torch_geometric.data import Data


class BoundingBox:
    """Class wrapper for storing bounding box information."""

    def __init__(
        self,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
        label: Optional[int] = None,
        score: Optional[float] = None,
    ) -> None:
        """Initialise a bounding box instance."""
        self._coords = (min_x, min_y, max_x, max_y)
        self._label = label
        self._score = score

    @property
    def min_x(self) -> int:
        """Get the minimum x value of the box, measured from the left of the image."""
        return self._coords[0]

    @property
    def min_y(self) -> int:
        """Get the minimum y value of the box, measured from the top of the image."""
        return self._coords[1]

    @property
    def max_x(self) -> int:
        """Get the maximum x value of the box, measured from the left of the image."""
        return self._coords[2]

    @property
    def max_y(self) -> int:
        """Get the maximum y value of the box, measured from the top of the image."""
        return self._coords[3]

    @property
    def label(self) -> Optional[int]:
        """Get the bounding box's label."""
        return self._label

    @property
    def score(self) -> Optional[float]:
        """Get the bounding box's score."""
        return self._score


class Question(TypedDict):
    """Serialisable representation of a question instance, from any dataset."""

    questionId: str
    imageId: str
    question: str
    tokens: List[int]
    dependencies: List[List[int]]
    answer: Optional[int]


class TrainableQuestion(TypedDict):
    """Trainable representation of a question instance, from any dataset."""

    questionId: str
    imageId: str
    tokens: torch.Tensor
    dependencies: Data
    answer: Optional[int]


class SceneGraph(TypedDict):
    """Serialisable representation of a scene graph instance, from any dataset."""

    imageId: str
    boxes: List[Tuple[int, int, int, int]]
    labels: List[str]
    attributes: List[List[str]]
    relations: List[str]
    coos: Tuple[List[int], List[int]]
    indexed_labels: List[int]
    indexed_attributes: List[List[int]]
    indexed_relations: List[int]


class TrainableSceneGraph(TypedDict):
    """Trainable representation of a scene graph instance, from any dataset."""

    imageId: str
    boxes: torch.FloatTensor  # FloatTensor[N, 4] (x1, y1, x2, y2) format
    labels: List[str]
    graph: Data
