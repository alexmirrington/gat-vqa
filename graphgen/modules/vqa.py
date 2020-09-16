"""Implementation of a geneal multimodal reasoning network."""
from typing import Any, Optional

import torch
from torch_geometric.data import Batch

from .question import AbstractQuestionModule
from .reasoning import AbstractReasoningModule
from .scene import AbstractSceneGraphModule


class VQA(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B905
    """Network that uses multiple GCN/BiLSTM inputs for its question and \
    knowledge-base representations, and a reasoning module for predictions."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        reasoning_module: AbstractReasoningModule,
        question_module: AbstractQuestionModule,
        scene_graph_module: AbstractSceneGraphModule,
        question_embeddings: Optional[torch.nn.Embedding] = None,
        scene_graph_embeddings: Optional[torch.nn.Embedding] = None,
    ) -> None:
        """Create a `VQA` model."""
        super().__init__()
        self.reasoning_module = reasoning_module
        self.question_module = question_module
        self.scene_graph_module = scene_graph_module
        self.question_embeddings = question_embeddings
        self.scene_graph_embeddings = scene_graph_embeddings

    def forward(self, question_graph: Batch, scene_graph: Batch) -> Any:
        """Propagate data through the model."""
        # Lookup scene graph embeddings if supplied.
        if self.scene_graph_embeddings is not None:
            # scene_graph.x should be tensor of indices
            # assert len(scene_graph.x.size()) == 1
            scene_graph.x = self.scene_graph_embeddings(scene_graph.x.long())

        # Lookup question embeddings if supplied.
        if self.question_embeddings is not None:
            # scene_graph.x should be tensor of indices
            # assert len(scene_graph.x.size()) == 1
            question_graph.x = self.question_embeddings(question_graph.x.long())

        words, question = self.question_module(question_graph)
        knowledge = self.scene_graph_module(scene_graph)
        return self.reasoning_module(words, question, knowledge)
