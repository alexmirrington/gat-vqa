"""Common schema definitions."""

from typing import List, TypedDict

from torch_geometric.data import Data


class Question(TypedDict):
    """Serialisable representation of a question instance, from any dataset."""

    imageId: str
    question: str
    tokens: List[int]
    dependencies: List[List[int]]


class TrainableQuestion(TypedDict):
    """Trainable representation of a question instance, from any dataset."""

    imageId: str
    dependencies: Data
