"""Basic GCN implementation."""
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv

from .abstract_gcn import AbstractGCN


class GAT(AbstractGCN):
    """Basic GAT implementation."""

    def __init__(
        self,
        shape: Sequence[int],
        pooling: Optional[Callable[..., torch.Tensor]],
        heads: int,
        concat: bool,
    ) -> None:
        """Create a GAT with layer sizes according to `shape`."""
        super().__init__(shape, pooling)
        self.layers = torch.nn.ModuleList([])
        for idx in range(1, len(shape)):
            self.layers.append(
                GATConv(
                    shape[idx - 1],
                    shape[idx] // heads if concat else shape[idx],
                    heads=heads,
                    concat=concat,
                )
            )
        self.pool = pooling
        self.shape = shape
        self.concat = concat

    def forward(self, graphs: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform a forward GAT pass."""
        feats, edge_index = graphs.x, graphs.edge_index

        for layer in self.layers[:-1]:
            feats = layer(feats, edge_index)
            # TODO experiment with dropout and activations
        feats = self.layers[-1](feats, edge_index)

        if self.pool is not None:
            pooled_feats = self.pool(feats, graphs.batch)
            return feats, pooled_feats
        return feats, None
