"""MAC netowrk with dual GCN inputs."""
from typing import Any, Optional

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from .sparse import GAT, GCN


class MACMultiGCN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """MAC network that uses multiple GCN inputs for its question and \
    knowledge-base representations."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        mac_network: torch.nn.Module,
        question_module: torch.nn.Module,
        scene_graph_module: Optional[torch.nn.Module],
        scene_graph_embeddings: Optional[torch.nn.Embedding],
    ) -> None:
        """Create a multi-gcn model with bidirectional attention."""
        super().__init__()
        self.mac_network = mac_network
        self.question_module = question_module
        self.scene_graph_module = scene_graph_module
        self.scene_graph_embeddings = scene_graph_embeddings

        self.sg_proj = (
            torch.nn.Linear(
                self.scene_graph_module.shape[-1], self.mac_network.hidden_dim
            )
            if isinstance(self.scene_graph_module, (GCN, GAT))
            and self.scene_graph_module.shape[-1] != self.mac_network.hidden_dim
            else None
        )

    def forward(self, dependencies: Batch, objects: Batch) -> Any:
        """Propagate data through the model."""
        # pylint: disable=too-many-locals

        # Apply scene graph embeddings if they exist.
        if self.scene_graph_embeddings is not None:
            assert len(objects.x.size()) == 1  # x should be tensor of indices
            objects.x = self.scene_graph_embeddings(objects.x.long())

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
            lstm_out, (h_n, _) = self.question_module(packed_text_feats)
            question = torch.cat([h_n[0], h_n[1]], -1)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
            h_n = (  # Don't know if this is necessary
                h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)
            )
            contextual_words = lstm_out
        else:
            text_feats, question = self.question_module(dependencies)
            dense_text_feats, question_lengths = to_dense_batch(
                text_feats, batch=dependencies.batch
            )
            question_lengths = torch.sum(question_lengths, dim=1)
            # Pack and pad again for MAC network
            packed_text_feats = torch.nn.utils.rnn.pack_padded_sequence(
                dense_text_feats,
                question_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            contextual_words, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_text_feats, batch_first=True
            )

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
            scene_graph_feats = lstm_out  # (batch_size, max_objects, hidden_dim)
            if self.sg_proj is not None:
                scene_graph_feats = self.sg_proj(scene_graph_feats)
        else:
            sg_feats = objects.x
            if self.scene_graph_module is not None:
                sg_feats, _ = self.scene_graph_module(objects)
            # (batch_size, max_objects, object_dim)
            scene_graph_feats, _ = to_dense_batch(sg_feats, batch=objects.batch)
            if self.sg_proj is not None:
                scene_graph_feats = self.sg_proj(scene_graph_feats)

        scene_graph_feats = torch.transpose(scene_graph_feats, 1, 2)

        # MAC network expects a tuple of tensors, (contextual_words, question, img).
        # `contextual_words` has size (batch_size, max_word_length, hidden_dim),
        # and is traditionally the output of the last BiLSTM layer at each timestep.
        # `question` has size (batch, hidden_dim), and is traditionally the
        # concatenated outputs of the forward and backward question BiLSTM passes.
        # `img` has size (batch_size, hidden_dim, img_feat_dim).

        return self.mac_network(contextual_words, question, scene_graph_feats)
