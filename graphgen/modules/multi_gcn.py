"""Multi-gcn model."""
from typing import Any, Union

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .gat import GAT
from .gcn import GCN

# from .bidirectional_attention import BidirectionalAttention
# from block import fusions


class MultiGCN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Multi-gcn model that operates on pre-extracted FasterRCNN features or \
    grount-truth scene graph data."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        num_answer_classes: int,
        text_syntactic_gcn: Union[GCN, GAT],
        scene_gcn: Union[GCN, GAT],
    ) -> None:
        """Create a multi-gcn model with bidirectional attention."""
        super().__init__()
        self.num_answer_classes = num_answer_classes

        # Create GCNs TODO use these instead of individual GATConv layers
        self.text_gcn = text_syntactic_gcn
        self.scene_gcn = scene_gcn
        # self.txt_semantic_gcn = txt_semantic_gcn
        # self.obj_feat_gcn = obj_feat_gcn
        # self.obj_pos_gcn = obj_pos_gcn

        # self.txt_obj_biatt_0 = BidirectionalAttention(
        #     key_shape=(obj_feat_gcn.shape[1], obj_feat_gcn.shape[1]),
        #     query_shape=(txt_semantic_gcn.shape[1], txt_semantic_gcn.shape[1]),
        #     heads=1,
        #     bidirectional=True,
        # )

        # LINEAR FUSION
        # Create 2-layer MLP fusion net that takes in conatenated feats from gcns.
        in_dim = self.text_gcn.shape[-1] + self.scene_gcn.shape[-1]
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(in_dim, (in_dim + num_answer_classes) // 2),
            torch.nn.Linear((in_dim + num_answer_classes) // 2, num_answer_classes),
        )

        # self.block = fusions.BlockTucker((300, 300), num_answer_classes)

        # ALIGNMENT EXPERIMENT TODO
        # self.alignment = PairwiseAlignment(
        #     (
        #         txt_syntactic_gcn.shape[-1],
        #         txt_semantic_gcn.shape[-1],
        #         obj_feat_gcn.shape[-1]
        #     ),
        #     proj_dim=256,
        #     heads=4
        # )
        # self.fusion = None

    def forward(self, dependencies: Data, objects: Data) -> Any:
        """Propagate data through the model."""
        # pylint: disable=too-many-locals

        pooled_text_feats = self.text_gcn(dependencies)
        pooled_object_feats = self.scene_gcn(objects)

        fused_feats = self.fusion(
            torch.cat([pooled_text_feats, pooled_object_feats], dim=1)
        )

        # fused_feats = self.block((pooled_text_feats, pooled_object_feats))

        return F.log_softmax(fused_feats, dim=1)
