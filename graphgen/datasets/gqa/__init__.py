"""A torch-compatible GQA dataset implementation."""
from .gqa import GQA as GQA
from .images import GQAImages as GQAImages
from .objects import GQAObjects as GQAObjects
from .questions import GQAQuestions as GQAQuestions
from .scene_graphs import GQASceneGraphs as GQASceneGraphs
from .spatial import GQASpatial as GQASpatial

__all__ = [
    GQA.__name__,
    GQAImages.__name__,
    GQAObjects.__name__,
    GQAQuestions.__name__,
    GQASceneGraphs.__name__,
    GQASpatial.__name__,
]
