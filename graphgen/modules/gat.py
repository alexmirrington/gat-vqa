"""Basic GCN implementation."""
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Basic GAT implementation."""

    def __init__(
        self,
        shape: Sequence[int],
        heads: int,
        pool_func: Optional[Callable[..., torch.Tensor]],
    ) -> None:
        """Create a GCN with layer sizes according to `shape`."""
        super().__init__()
        self.conv_layers = torch.nn.ModuleList([])
        for idx in range(1, len(shape)):
            self.conv_layers.append(GATConv(shape[idx - 1], shape[idx], heads=heads))
        self.pool = pool_func
        self.shape = shape

    def forward(self, data: Data) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform a forward GCN pass."""
        feats, edge_index = data.x, data.edge_index

        for layer in self.conv_layers[:-1]:
            feats = layer(feats, edge_index)
        feats = self.conv_layers[-1](feats, edge_index)

        if self.pool is not None:
            pooled_feats = self.pool(feats, data.batch)
            return feats, pooled_feats
        return feats, None
