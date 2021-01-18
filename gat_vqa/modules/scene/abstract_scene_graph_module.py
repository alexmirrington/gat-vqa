"""Abstract classes for all scene-graph-processing modules."""
from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Batch


class AbstractSceneGraphModule(torch.nn.Module, ABC):  # type: ignore
    """Abstract base class for all scene-graph-processing modules."""

    @abstractmethod
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
