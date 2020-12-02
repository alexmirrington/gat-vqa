"""A RNN scene-graph-processing module."""
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from .abstract_scene_graph_module import AbstractSceneGraphModule


class RNNSceneGraphModule(AbstractSceneGraphModule):
    """Implementation of a RNN scene-graph-processing module."""

    def __init__(self, rnn: torch.nn.RNNBase) -> None:
        """Initialise a `RNNSceneGraphModule` instance."""
        super().__init__()
        self.rnn = rnn

    def forward(self, scene_graph: Batch) -> torch.Tensor:
        """Propagate data through the module.

        Params:
        -------
        `scene_graph`: A batch of data containing edge indices in COO format that
        specify dependencies between scene graph nodes, as well as embeddings
        for each nodes in the graph.

        Returns:
        --------
        `scene_graph`: A dense tensor of size `(batch_size, max_object_count,
        object_feature_dim)`, the processed scene graph features for each graph
        in the batch.
        """
        # Get dense object features input to RNN
        dense_node_feats, num_nodes = to_dense_batch(
            scene_graph.x, batch=scene_graph.batch
        )
        # Assume we have at least one node for samples with zero nodes, required
        # for `pack_padded_sequence`.
        num_nodes = torch.sum(num_nodes, dim=1)
        num_nodes = torch.clamp(num_nodes, min=1)
        packed_object_feats = torch.nn.utils.rnn.pack_padded_sequence(
            dense_node_feats,
            num_nodes.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        knowledge, _ = self.rnn(packed_object_feats)
        knowledge, _ = torch.nn.utils.rnn.pad_packed_sequence(
            knowledge, batch_first=True
        )
        return knowledge
