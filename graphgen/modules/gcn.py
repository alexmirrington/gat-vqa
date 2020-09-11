"""Basic GCN implementation."""
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Basic GCN implementation."""

    def __init__(
        self,
        shape: Sequence[int],
        pool_func: Optional[Callable[..., torch.Tensor]],
    ) -> None:
        """Create a GCN with layer sizes according to `shape`."""
        super().__init__()
        self.shape = shape
        self.conv_layers = torch.nn.ModuleList([])
        for idx in range(1, len(shape)):
            self.conv_layers.append(GCNConv(shape[idx - 1], shape[idx]))
        self.pool = pool_func

    def forward(self, data: Data) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform a forward GCN pass."""
        feats, edge_index = data.x, data.edge_index

        for layer in self.conv_layers[:-1]:
            feats = layer(feats, edge_index)
            feats = F.relu(feats)
            # feats = F.dropout(feats, training=self.training)
        feats = self.conv_layers[-1](feats, edge_index)

        if self.pool is not None:
            pooled_feats = self.pool(feats, data.batch)
            return feats, pooled_feats
        return feats, None
