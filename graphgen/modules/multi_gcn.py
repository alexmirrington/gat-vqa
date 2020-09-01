"""Placeholder wrapper model."""
from typing import Any, List, Union

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from .gcn import GCN
from .relpn import RelPN


class MultiGCN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Multi-gcn model that operates on pre-extracted FasterRCNN features or \
    grount-truth scene graph data."""

    def __init__(
        self,
        num_answer_classes: int,
        num_object_classes: int,
        txt_dependency_gcn: GCN,
        obj_semantic_gcn: GCN,
    ) -> None:
        """Create a placeholder model."""
        super().__init__()
        self.num_answer_classes = num_answer_classes
        self.num_object_classes = num_object_classes

        # Create text dependency GCN.
        self.txt_dependency_gcn = txt_dependency_gcn

        # Create object semantic relation proposal network and gcn
        self.obj_semantic_relpn = RelPN(num_object_classes)
        self.obj_semantic_gcn = obj_semantic_gcn

        # create fusion layer
        in_dim = (
            self.txt_dependency_gcn.conv_layers[-1].out_channels
            + self.obj_semantic_gcn.conv_layers[-1].out_channels
        )
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
        obj_semantic_gcn_batch = Batch.from_data_list(
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

        dep_gcn_results = self.txt_dependency_gcn(dependencies)
        semantic_gcn_results = self.obj_semantic_gcn(obj_semantic_gcn_batch)

        fused_feats = self.fusion(
            torch.cat((dep_gcn_results, semantic_gcn_results), dim=1)
        )
        return F.log_softmax(fused_feats, dim=1)
