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
    dependencies: Data
    answer: Optional[int]


class SceneGraph(TypedDict):
    """Serialisable representation of a scene graph instance, from any dataset."""

    imageId: str
    boxes: List[Tuple[int, int, int, int]]
    labels: List[int]
    # TODO work out relations format. We probs need to preprocess relation ids


class TrainableSceneGraph(TypedDict):
    """Trainable representation of a scene graph instance, from any dataset."""

    imageId: str
    boxes: torch.FloatTensor  # FloatTensor[N, 4] (x1, y1, x2, y2) format
    labels: torch.IntTensor  # Int64Tensor[N] class labels for boxes
