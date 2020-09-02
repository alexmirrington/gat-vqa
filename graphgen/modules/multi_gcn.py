"""Placeholder wrapper model."""
from typing import Any, List, Union

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

from .bidirectional_attention import BidirectionalAttention
from .gcn import GCN
from .relpn import RelPN


class MultiGCN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Multi-gcn model that operates on pre-extracted FasterRCNN features or \
    grount-truth scene graph data."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        num_answer_classes: int,
        num_object_classes: int,
        txt_dependency_gcn: GCN,
        obj_semantic_gcn: GCN,
    ) -> None:
        """Create a multi-gcn model with bidirectional attention."""
        super().__init__()
        self.num_answer_classes = num_answer_classes
        self.num_object_classes = num_object_classes

        # Create text dependency GCN.
        # self.txt_dependency_gcn = txt_dependency_gcn  # TODO uncomment

        # Create object semantic relation proposal network and gcn
        self.obj_semantic_relpn = RelPN(num_object_classes)
        # self.obj_semantic_gcn = obj_semantic_gcn  # TODO uncomment

        self.txt_dependency_gat_0 = GCNConv(
            txt_dependency_gcn.shape[0], txt_dependency_gcn.shape[1]
        )
        self.obj_semantic_gat_0 = GCNConv(
            obj_semantic_gcn.shape[0], obj_semantic_gcn.shape[1]
        )

        # use object semantic features as an attention query on the dependency
        # parser feats. i.e. we are querying obj-semantic feats to exctract information
        # to propagate to the dependency parser feats.
        self.biatt_0 = BidirectionalAttention(
            key_shape=(txt_dependency_gcn.shape[1], txt_dependency_gcn.shape[1]),
            query_shape=(obj_semantic_gcn.shape[1], obj_semantic_gcn.shape[1]),
            heads=3,
            bidirectional=True,
        )

        self.txt_dependency_gat_1 = GCNConv(
            txt_dependency_gcn.shape[1], txt_dependency_gcn.shape[2]
        )
        self.txt_dependency_gat_2 = GCNConv(
            txt_dependency_gcn.shape[2], txt_dependency_gcn.shape[3]
        )

        # Create 2-layer MLP fusion net that takes in conatenated feats from gcns.
        in_dim = txt_dependency_gcn.shape[-1] + obj_semantic_gcn.shape[-1]
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(in_dim, (in_dim + num_answer_classes) // 2),
            torch.nn.Linear((in_dim + num_answer_classes) // 2, num_answer_classes),
        )

    def forward(
        self,
        dependencies: Data,
        boxes: List[torch.FloatTensor],
        labels: Union[List[torch.IntTensor], List[torch.FloatTensor]],
    ) -> Any:
        """Propagate data through the model."""
        # Convert labels to one hot if they are not probabilty distributions.
        class_logits = [
            F.one_hot(img_labels, self.num_object_classes)
            .type(torch.FloatTensor)
            .to("cuda")
            if len(img_labels.size()) == 1
            else img_labels
            for img_labels in labels
        ]

        # Get RelPN relationship predictions
        relations, scores = self.obj_semantic_relpn(class_logits, boxes)

        # Use class_logits directly, as class_logits are indexed relative to
        # number of proposals
        obj_semantics = Batch.from_data_list(
            [
                Data(
                    edge_index=img_relations.t(),
                    edge_attr=img_scores.unsqueeze(-1),
                    x=img_logits,  # x has shape [num_nodes, num_node_features]
                )
                for img_relations, img_scores, img_logits in zip(
                    relations, scores, class_logits
                )
            ]
        ).to(
            "cuda"
        )  # TODO handle devices better

        dependencies.x = self.txt_dependency_gat_0(
            dependencies.x, dependencies.edge_index
        )
        obj_semantics.x = self.obj_semantic_gat_0(
            obj_semantics.x, obj_semantics.edge_index
        )
        # print(f"{dependencies=}")
        # print(f"{obj_semantics=}")
        dependencies.x, obj_semantics.x = self.biatt_0(dependencies.x, obj_semantics.x)
        # print(f"{dependencies=}")
        # print(f"{obj_semantics=}")
        dependencies.x = self.txt_dependency_gat_1(
            dependencies.x, dependencies.edge_index
        )
        # print(f"{dependencies=}")
        # print(f"{obj_semantics=}")
        dependencies.x = self.txt_dependency_gat_2(
            dependencies.x, dependencies.edge_index
        )
        # print(f"{dependencies=}")
        # print(f"{obj_semantics=}")
        dep_pooled_feats = global_mean_pool(dependencies.x, dependencies.batch)
        obj_pooled_feats = global_mean_pool(obj_semantics.x, obj_semantics.batch)

        fused_feats = self.fusion(
            torch.cat([dep_pooled_feats, obj_pooled_feats], dim=1)
        )
        return F.log_softmax(fused_feats, dim=1)
