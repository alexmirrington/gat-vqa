"""Basic GCN implementation."""
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv

from .abstract_gcn import AbstractGCN


class GCN(AbstractGCN):
    """Basic GCN implementation."""

    def __init__(
        self,
        shape: Sequence[int],
        pooling: Optional[Callable[..., torch.Tensor]],
    ) -> None:
        """Create a GCN with layer sizes according to `shape`."""
        super().__init__(shape, pooling)
        self.shape = shape
        self.layers = torch.nn.ModuleList([])
        for idx in range(1, len(shape)):
            self.layers.append(GCNConv(shape[idx - 1], shape[idx]))
        self.pooling = pooling

    def forward(self, graphs: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform a forward GCN pass."""
        feats, edge_index = graphs.x, graphs.edge_index

        for layer in self.layers[:-1]:
            feats = layer(feats, edge_index)
            # TODO experiment with dropout and activations
        feats = self.layers[-1](feats, edge_index)

        if self.pooling is not None:
            pooled_feats = self.pooling(feats, graphs.batch)
            return feats, pooled_feats
        return feats, None
