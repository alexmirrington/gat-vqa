"""A identity scene graph module with no trainable parameters."""

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from .abstract_scene_graph_module import AbstractSceneGraphModule


class IdentitySceneGraphModule(AbstractSceneGraphModule):
    """Implementation of an identity scene graph module with no trainable parameters."""

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
        # Get dense object features and return
        knowledge, _ = to_dense_batch(scene_graph.x, batch=scene_graph.batch)
        return knowledge
