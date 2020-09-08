"""MAC netowrk with dual GCN inputs."""
from typing import Any, Sequence

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GATConv, GCNConv
from torch_geometric.utils import to_dense_batch

from ..config.model import GCNName


class MACMultiGCN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """MAC network that uses multiple GCN inputs for its question and \
    knowledge-base representations."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        mac_network: torch.nn.Module,
        text_gcn_shape: Sequence[int],
        text_gcn_conv: GCNName,
        scene_gcn_shape: Sequence[int],
        scene_gcn_conv: GCNName,
        max_objects: int = 100,  # GQA default max objects
    ) -> None:
        """Create a multi-gcn model with bidirectional attention."""
        super().__init__()
        self.mac_network = mac_network
        self.max_objects = max_objects

        self.text_gcn_layers = torch.nn.ModuleList([])
        for idx in range(1, len(text_gcn_shape)):
            if text_gcn_conv == GCNName.GCN:
                self.text_gcn_layers.append(
                    GCNConv(text_gcn_shape[idx - 1], text_gcn_shape[idx])
                )
            elif text_gcn_conv == GCNName.GAT:
                self.text_gcn_layers.append(
                    GATConv(text_gcn_shape[idx - 1], text_gcn_shape[idx], heads=1)
                )
            else:
                raise NotImplementedError()

        self.scene_gcn_layers = torch.nn.ModuleList([])
        for idx in range(1, len(scene_gcn_shape)):
            if scene_gcn_conv == GCNName.GCN:
                self.scene_gcn_layers.append(
                    GCNConv(scene_gcn_shape[idx - 1], scene_gcn_shape[idx])
                )
            elif scene_gcn_conv == GCNName.GAT:
                self.scene_gcn_layers.append(
                    GATConv(scene_gcn_shape[idx - 1], scene_gcn_shape[idx], heads=1)
                )
            else:
                raise NotImplementedError()

    def forward(self, dependencies: Batch, objects: Batch) -> Any:
        """Propagate data through the model."""
        # pylint: disable=too-many-locals

        text_feats = dependencies.x
        sg_feats = objects.x
        for idx in range(min(len(self.text_gcn_layers), len(self.scene_gcn_layers))):
            # Get text and scene feats for current layer
            text_gcn_layer = self.text_gcn_layers[idx]
            scene_gcn_layer = self.scene_gcn_layers[idx]

            text_feats = text_gcn_layer(text_feats, dependencies.edge_index)
            sg_feats = scene_gcn_layer(sg_feats, objects.edge_index)

            # Apply dense bidirectional attention to node features
            # TODO  multihead_attn = torch.nn.MultiheadAttention(300, self.heads)

        # Apply any leftover conv layers for text gcn
        for leftover_idx in range(idx + 1, len(self.text_gcn_layers)):
            text_gcn_layer = self.text_gcn_layers[leftover_idx]
            text_feats = text_gcn_layer(text_feats, dependencies.edge_index)

        # Apply any leftover conv layers for scene gcn
        for leftover_idx in range(idx + 1, len(self.scene_gcn_layers)):
            scene_gcn_layer = self.scene_gcn_layers[leftover_idx]
            sg_feats = scene_gcn_layer(sg_feats, objects.edge_index)

        # MAC network expects a tuple of tensors, (contextual_words, question, img).
        # `contextual_words` has size (batch_size, max_word_length, hidden_dim),
        # and is traditionally the output of the last BiLSTM layer at each timestep.
        # `question` has size (batch, hidden_dim), and is traditionally the
        # concatenated outputs of the forward and backward question BiLSTM passes.
        # `img` has size (batch_size, hidden_dim, img_feat_dim).

        contextual_words, _ = to_dense_batch(text_feats, batch=dependencies.batch)
        # TODO consider self-attention pooling over contextual words instead
        # of global mean pooling.
        question = global_mean_pool(text_feats, batch=dependencies.batch)
        scene_graph_feats, _ = to_dense_batch(sg_feats, batch=objects.batch)

        # Pad scene graph feats to the correct dimension, or limit if too big
        if scene_graph_feats.size(1) >= self.max_objects:
            scene_graph_feats = scene_graph_feats[:, : self.max_objects, :]
        else:
            scene_graph_feats = torch.cat(
                (
                    scene_graph_feats,
                    torch.zeros(
                        (
                            scene_graph_feats.size(0),
                            self.max_objects - scene_graph_feats.size(1),
                            scene_graph_feats.size(2),
                        )
                    ).to(scene_graph_feats.device),
                ),
                dim=1,
            )

        out = self.mac_network((contextual_words, question, scene_graph_feats))

        return out
