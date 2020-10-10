"""Module containing various torch hooks."""
from typing import Any

import torch
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import to_dense_adj


class GATConvAttentionHook:
    """Callable hook for retrieving attention maps from a `GATConv` layer."""

    def __init__(self) -> None:
        """Initialise the hook."""
        self.result = None

    def __call__(self, module: torch.nn.Module, ipt: Any, output: Any) -> None:
        """Get the attention weights from a `torch_geometric.nn.conv.GATConv` layer."""
        if not isinstance(module, GATConv):
            return
        if isinstance(output, tuple) and len(output) > 1:
            edge_index, edge_attr = output[1]
            adj = (
                to_dense_adj(edge_index=edge_index, edge_attr=edge_attr)
                .squeeze()
                .detach()
                .numpy()
            )
            self.result = adj
        self.result = None
