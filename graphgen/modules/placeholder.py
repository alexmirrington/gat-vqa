"""Placeholder wrapper model."""
from typing import Any, List

import torch
from torch_geometric.data import Data

from .gcn import GCN
from .graph_rcnn import GraphRCNN


class Placeholder(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Placeholder model for wrapping submodules during development."""

    def __init__(self, grcnn: GraphRCNN, dep_gcn: GCN) -> None:
        """Create a placeholder model."""
        super().__init__()
        self.dep_gcn = dep_gcn
        self.grcnn = grcnn

    def forward(self, images: List[torch.Tensor], deps: Data) -> Any:
        """Propagate data through the model."""
        grcnn_results = self.grcnn(images)
        dep_gcn_results = self.dep_gcn(deps)

        return grcnn_results, dep_gcn_results
