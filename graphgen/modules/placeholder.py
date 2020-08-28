"""Placeholder wrapper model."""
from typing import Any, List, Optional

import torch
from torch_geometric.data import Data

from .gcn import GCN
from .graph_rcnn import GraphRCNN, GraphRCNNTarget


class Placeholder(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Placeholder model for wrapping submodules during development."""

    def __init__(self, grcnn: GraphRCNN, dep_gcn: GCN) -> None:
        """Create a placeholder model."""
        super().__init__()
        self.dep_gcn = dep_gcn
        self.grcnn = grcnn

    def forward(
        self,
        deps: Data,
        images: List[torch.Tensor],
        bbox_targets: Optional[List[GraphRCNNTarget]] = None,
    ) -> Any:
        """Propagate data through the model."""
        grcnn_results = self.grcnn(images, bbox_targets)
        dep_gcn_results = self.dep_gcn(deps)

        return dep_gcn_results, grcnn_results
