"""MAC netowrk with dual GCN inputs."""
from typing import Any, Optional

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


class MACMultiGCN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """MAC network that uses multiple GCN inputs for its question and \
    knowledge-base representations."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        mac_network: torch.nn.Module,
        question_module: torch.nn.Module,
        scene_gcn: Optional[torch.nn.Module],
    ) -> None:
        """Create a multi-gcn model with bidirectional attention."""
        super().__init__()
        self.mac_network = mac_network
        self.question_module = question_module
        self.scene_gcn = scene_gcn
        self.sg_proj = (
            torch.nn.Linear(self.scene_gcn.shape[-1], self.mac_network.hidden_dim)
            if self.scene_gcn is not None
            else None
        )

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
        sg_feats = objects.x
        if self.scene_gcn is not None:
            sg_feats, _ = self.scene_gcn(objects)
        scene_graph_feats, _ = to_dense_batch(sg_feats, batch=objects.batch)

        # Pad scene graph feats to the correct dimension, or limit if too big
        # if scene_graph_feats.size(1) >= self.max_objects:
        #     scene_graph_feats = scene_graph_feats[:, : self.max_objects, :]
        # else:
        #     scene_graph_feats = torch.cat(
        #         (
        #             scene_graph_feats,
        #             torch.zeros(
        #                 (
        #                     scene_graph_feats.size(0),
        #                     self.max_objects - scene_graph_feats.size(1),
        #                     scene_graph_feats.size(2),
        #                 )
        #             ).to(scene_graph_feats.device),
        #         ),
        #         dim=1,
        #     )

        scene_graph_feats = torch.transpose(self.sg_proj(scene_graph_feats), 1, 2)

        # MAC network expects a tuple of tensors, (contextual_words, question, img).
        # `contextual_words` has size (batch_size, max_word_length, hidden_dim),
        # and is traditionally the output of the last BiLSTM layer at each timestep.
        # `question` has size (batch, hidden_dim), and is traditionally the
        # concatenated outputs of the forward and backward question BiLSTM passes.
        # `img` has size (batch_size, hidden_dim, img_feat_dim).

        return self.mac_network((contextual_words, question, scene_graph_feats))
