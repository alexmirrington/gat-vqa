"""BottomUp network with multiple GCN/BiLSTM inputs."""
from typing import Any, Optional

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


class BottomUpMultiGCN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B905
    """BottomUp network that uses multiple GCN/BiLSTM inputs for its question \
    and knowledge-base representations."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        reasoning_module: torch.nn.Module,
        question_module: torch.nn.Module,
        scene_graph_module: Optional[torch.nn.Module],
    ) -> None:
        """Create a `BottomUpMultiGCN` model."""
        super().__init__()
        self.reasoning_module = reasoning_module
        self.question_module = question_module
        self.scene_graph_module = scene_graph_module

    def forward(self, dependencies: Batch, objects: Batch) -> Any:
        """Propagate data through the model."""
        # pylint: disable=too-many-locals

        if isinstance(self.question_module, torch.nn.LSTM):
            # Get dense text features for MAC contextual words
            dense_text_feats, question_lengths = to_dense_batch(
                dependencies.x, batch=dependencies.batch
            )
            question_lengths = torch.sum(question_lengths, dim=1)
            batch_size = dense_text_feats.size(0)
            packed_text_feats = torch.nn.utils.rnn.pack_padded_sequence(
                dense_text_feats,
                question_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.question_module(packed_text_feats)
            question = torch.cat([h_n[0], h_n[1]], -1)
            h_n = (  # Don't know if this is necessary
                h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)
            )
        else:
            _, question = self.question_module(dependencies)

        # Get scene graph feats
        if isinstance(self.scene_graph_module, torch.nn.LSTM):
            # Get dense text features for MAC contextual words
            dense_object_feats, num_objects = to_dense_batch(
                objects.x, batch=objects.batch
            )
            num_objects = torch.sum(num_objects, dim=1)
            batch_size = dense_object_feats.size(0)
            # Assume we have at least one object for samples with zero objects
            num_objects = torch.clamp(num_objects, min=1)
            packed_object_feats = torch.nn.utils.rnn.pack_padded_sequence(
                dense_object_feats,
                num_objects,
                batch_first=True,
                enforce_sorted=False,
            )
            lstm_out, _ = self.scene_graph_module(packed_object_feats)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
            knowledge = lstm_out  # (batch_size, max_objects, hidden_dim)
        else:
            sg_feats = objects.x
            if self.scene_gcn is not None:
                sg_feats, _ = self.scene_gcn(objects)
            # (batch_size, max_objects, object_dim)
            knowledge, _ = to_dense_batch(sg_feats, batch=objects.batch)

        # `question` has size (batch, hidden_dim), and is traditionally the
        # concatenated outputs of the forward and backward question BiLSTM passes.
        # `knowledge` has size (batch_size, knowledge_feat_count, knowledge_feat_dim).
        return self.reasoning_module(question, knowledge)
