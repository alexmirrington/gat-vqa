"""Placeholder wrapper model."""
from typing import Any, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import negative_sampling

from ..config.model import GCNConvName


class MultiGCN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Multi-gcn model that operates on pre-extracted FasterRCNN features or \
    grount-truth scene graph data."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        num_answer_classes: int,
        num_object_classes: int,
        num_relation_classes: int,
        txt_syntactic_gcn: GCN,
        txt_semantic_gcn: GCN,
        obj_feat_gcn: GCN,
        obj_pos_gcn: GCN,
    ) -> None:
        """Create a multi-gcn model with bidirectional attention."""
        super().__init__()
        self.num_answer_classes = num_answer_classes
        self.num_object_classes = num_object_classes
        self.num_relation_classes = num_relation_classes

        # Create GCNs TODO use these instead of individual GATConv layers
        self.txt_syntactic_gcn = txt_syntactic_gcn
        self.txt_semantic_gcn = txt_semantic_gcn
        self.obj_feat_gcn = obj_feat_gcn
        self.obj_pos_gcn = obj_pos_gcn

        # Positional relation proposal network selects the most likely relation
        # candidates based on bounding box positions
        self.positional_relpn = SpatialRelPN(4)
        self.txt_semantic_relpn = BiLSTMRelPN(300, 256)
        # TODO experiment with training signals from rcnn feats w/ sg bboxes
        # self.object_feat_relpn = SpatialRelPN(2048)

        # Positional relational classifier takes in bounding boxes only.
        self.positional_rel_classifier = torch.nn.Sequential(
            torch.nn.Linear(8, (8 + num_relation_classes) // 2),
            torch.nn.Linear((8 + num_relation_classes) // 2, num_relation_classes),
        )
        # TODO experiment with training signals from rcnn feats w/ sg bboxes
        # self.object_feat_rel_classifier = torch.nn.Sequential(
        #     torch.nn.Linear(4096, 512),
        #     torch.nn.Linear(512, num_relation_classes),
        # )

        # LINEAR FUSION
        # Create 2-layer MLP fusion net that takes in conatenated feats from gcns.
        in_dim = (
            txt_syntactic_gcn.shape[-1]
            + txt_semantic_gcn.shape[-1]
            + obj_feat_gcn.shape[-1]
            + obj_pos_gcn.shape[-1]
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(in_dim, num_answer_classes),
            torch.nn.Linear(num_answer_classes, num_answer_classes),
        )

    def forward(
        self,
        word_embeddings: Union[PackedSequence, torch.Tensor],
        dependencies: Data,
        rcnn_objects: Sequence[torch.FloatTensor],
        rcnn_boxes: Sequence[torch.FloatTensor],
        gt_boxes: Sequence[torch.FloatTensor],
        gt_labels: Sequence[torch.IntTensor],
        gt_relations: Sequence[torch.IntTensor],
        gt_coos: Sequence[torch.LongTensor],
        widths: Sequence[int],
        heights: Sequence[int],
    ) -> Any:
        """Propagate data through the model."""
        # pylint: disable=too-many-locals
        # Get RelPN relationship predictions
        (
            semantic_relations,
            semantic_scores,
            semantic_feats,
        ) = self.txt_semantic_relpn(word_embeddings)

        # Normalise bounding boxes to between 0 and 1
        whs = [
            torch.tensor((w, h, w, h)).to("cuda") for w, h in zip(widths, heights)
        ]  # pylint: disable=not-callable
        rcnn_normed_boxes, gt_normed_boxes = zip(
            *[
                (img_rcnn_boxes / img_whs, img_gt_boxes / img_whs)
                for img_whs, img_rcnn_boxes, img_gt_boxes in zip(
                    whs, rcnn_boxes, gt_boxes
                )
            ]
        )

        # Provide supervised signals to relpn and relcls models
        posrelpn_losses = []
        posrelcls_losses = []
        if self.training:

            for (img_gt_coos, img_gt_boxes, img_gt_relations) in zip(
                gt_coos, gt_normed_boxes, gt_relations
            ):
                # Perform balanced positive and negative sampling for training
                # img_samples: (3, num_edges)
                if img_gt_coos.numel() == 0:
                    continue
                img_pos_samples = img_gt_coos
                img_pos_targets = torch.ones(img_pos_samples.size(1)).to("cuda")
                img_neg_samples = negative_sampling(
                    img_gt_coos, num_nodes=len(img_gt_boxes)
                )
                img_neg_targets = torch.zeros(img_neg_samples.size(1)).to("cuda")
                img_samples = torch.cat((img_pos_samples, img_neg_samples), dim=1)
                img_targets = torch.cat((img_pos_targets, img_neg_targets), dim=0)

                # Get positional relpn losses
                img_score_adj = self.positional_relpn.similarity(img_gt_boxes)
                img_scores = torch.stack(
                    [img_score_adj[i][j] for i, j in img_samples.t()]
                ).to(
                    "cuda"
                )  # TODO check index order  # TODO make more efficient
                posrelpn_losses.append(F.binary_cross_entropy(img_scores, img_targets))

                # Get classification losses for each of the positive sample relations
                selected_normed_boxes = torch.stack(
                    [
                        torch.cat((img_gt_boxes[i], img_gt_boxes[j]), dim=0)
                        for i, j in img_pos_samples.t()
                    ],
                    dim=0,
                ).to("cuda")
                img_relation_preds = F.log_softmax(
                    self.positional_rel_classifier(selected_normed_boxes)
                )
                posrelcls_losses.append(
                    F.nll_loss(img_relation_preds, img_gt_relations)
                )

        # Get top-k non-maximally-selected proposals based on positional relpn
        # pre_nms_limit=None, iou_threshold=1.0, post_nms_limit=None
        # implies we don't filter relations at all, just retrieve scores
        rel_coos, rel_scores, rel_intersections = self.positional_relpn(
            rcnn_boxes, rcnn_boxes, iou_threshold=0.9
        )
        # Get image preds for rcnn_boxes
        relationship_predictions = [
            self.positional_rel_classifier(
                torch.stack(
                    [
                        torch.cat((img_boxes[i], img_boxes[j]), dim=0)
                        for i, j in img_rel_coos
                    ],
                    dim=0,
                )
            )
            for img_rel_coos, img_boxes in zip(rel_coos, rcnn_normed_boxes)
        ]  # TODO make more efficient, maybe with torch.gather

        # Build rcnn proposal graph
        obj_pos_graph = Batch.from_data_list(
            [
                Data(
                    edge_index=img_relations.t(),
                    # edge_attr=obj_scores,
                    x=img_rels,  # x has shape (num_nodes, num_node_features)
                )
                for img_relations, img_scores, img_rels in zip(
                    rel_coos, rel_scores, relationship_predictions
                )
            ]
        ).to("cuda")
        obj_feat_graph = Batch.from_data_list(
            [
                Data(
                    edge_index=img_relations.t(),
                    # edge_attr=obj_scores,
                    x=img_feats,  # x has shape (num_nodes, num_node_features)
                )
                for img_relations, img_scores, img_feats in zip(
                    rel_coos, rel_scores, rcnn_objects
                )
            ]
        ).to("cuda")
        txt_sem_graph = Batch.from_data_list(
            [
                Data(
                    edge_index=txt_relations,
                    # edge_attr=txt_scores,
                    x=txt_feats,
                )
                for txt_relations, txt_scores, txt_feats in zip(
                    semantic_relations,
                    semantic_scores,
                    semantic_feats,  # pad_packed_sequence(word_embeddings)[0]
                )
            ]
        ).to("cuda")
        txt_syn_pooled_feats = self.txt_syntactic_gcn(dependencies)
        obj_pos_pooled_feats = self.obj_pos_gcn(obj_pos_graph)
        obj_feat_pooled_feats = self.obj_feat_gcn(obj_feat_graph)
        txt_sem_pooled_feats = self.txt_semantic_gcn(txt_sem_graph)
        fused_feats = self.fusion(
            torch.cat(
                [
                    txt_syn_pooled_feats,
                    obj_pos_pooled_feats,
                    obj_feat_pooled_feats,
                    txt_sem_pooled_feats,
                ],
                dim=1,
            )
        )
        return F.log_softmax(fused_feats, dim=1), {
            "relpn": torch.mean(torch.stack(posrelpn_losses)),
            "relcls": torch.mean(torch.stack(posrelcls_losses)),
        }
