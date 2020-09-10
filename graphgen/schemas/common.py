"""Common schema definitions."""

from typing import List, Optional, Tuple, TypedDict

import torch
from torch_geometric.data import Data


class Question(TypedDict):
    """Serialisable representation of a question instance, from any dataset."""

    questionId: str
    imageId: str
    question: str
    tokens: List[str]  # List[str] over List[int] for GloVe vector lookup.
    dependencies: List[List[int]]
    answer: Optional[int]


class TrainableQuestion(TypedDict):
    """Trainable representation of a question instance, from any dataset."""

    questionId: str
    imageId: str
    embeddings: torch.Tensor
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
    attributes: torch.Tensor
    objects: Data
