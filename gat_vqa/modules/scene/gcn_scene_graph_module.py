"""A GCN scene-graph-processing module."""
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from ..sparse import AbstractGCN
from .abstract_scene_graph_module import AbstractSceneGraphModule


class GCNSceneGraphModule(AbstractSceneGraphModule):
    """Implementation of a GCN scene-graph-processing module."""

    def __init__(self, gcn: AbstractGCN) -> None:
        """Initialise a `GCNSceneGraphModule` instance."""
        super().__init__()
        self.gcn = gcn

    def forward(self, scene_graph: Batch) -> torch.Tensor:
        """Propagate data through the module.

        Params:
        -------
        `scene_graph`: A batch of data containing edge indices in COO format that
        specify dependencies between scene graph nodes, as well as embeddings
        for each nodes in the graph.

        Returns:
        --------
        `knowledge`: A dense tensor of size `(batch_size, max_object_count,
        object_feature_dim)`, the processed scene graph features for each graph
        in the batch.
        """
        knowledge, _ = self.gcn(scene_graph)
        knowledge, _ = to_dense_batch(knowledge, batch=scene_graph.batch)
        return knowledge
