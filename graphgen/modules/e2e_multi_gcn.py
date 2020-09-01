"""Placeholder wrapper model."""
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .faster_rcnn import FasterRCNNTarget
from .gcn import GCN
from .graph_rcnn import GraphRCNN


class E2EMultiGCN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """End to end multi-gcn model with built in FasterRCNN object detector."""

    def __init__(
        self, out_dim: int, grcnn: GraphRCNN, dep_gcn: GCN, obj_semantic_gcn: GCN
    ) -> None:
        """Create an `E2EMultiGCN` model."""
        super().__init__()
        self.dep_gcn = dep_gcn
        self.obj_semantic_gcn = obj_semantic_gcn
        self.grcnn = grcnn

        in_dim = (
            self.dep_gcn.conv_layers[-1].out_channels
            + self.obj_semantic_gcn.conv_layers[-1].out_channels
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(in_dim, (in_dim + out_dim) // 2),
            torch.nn.Linear((in_dim + out_dim) // 2, out_dim),
        )

    def forward(
        self,
        dependencies: Data,
        images: List[torch.Tensor],
        targets: Optional[List[FasterRCNNTarget]] = None,
    ) -> Any:
        """Propagate data through the model."""
        rcnn_loss, semantic_gcn_batch = self.grcnn(images, targets)
        dep_gcn_results = self.dep_gcn(dependencies)
        semantic_gcn_batch = semantic_gcn_batch.to("cuda")  # TODO handle devices better
        semantic_gcn_results = self.obj_semantic_gcn(semantic_gcn_batch)

        fused_feats = self.fusion(
            torch.cat((dep_gcn_results, semantic_gcn_results), dim=1)
        )
        return rcnn_loss, F.log_softmax(fused_feats, dim=1)
