"""Common schema definitions."""

from typing import List, Optional, TypedDict

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
