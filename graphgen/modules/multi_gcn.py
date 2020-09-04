"""Placeholder wrapper model."""
from typing import Any, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool

from .bidirectional_attention import BidirectionalAttention
from .bilstm_relpn import BiLSTMRelPN
from .gcn import GCN
from .relpn import SpatialRelPN


class MultiGCN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Multi-gcn model that operates on pre-extracted FasterRCNN features or \
    grount-truth scene graph data."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        num_answer_classes: int,
        num_object_classes: int,
        txt_syntactic_gcn: GCN,
        txt_semantic_gcn: GCN,
        obj_feat_gcn: GCN,
        obj_pos_gcn: GCN,
    ) -> None:
        """Create a multi-gcn model with bidirectional attention."""
        super().__init__()
        self.num_answer_classes = num_answer_classes
        self.num_object_classes = num_object_classes

        # Create GCNs TODO use these instead of individual GATConv layers
        # self.txt_syntactic_gcn = txt_syntactic_gcn
        # self.txt_semantic_gcn = txt_semantic_gcn
        # self.obj_feat_gcn = obj_feat_gcn
        # self.obj_pos_gcn = obj_pos_gcn

        # Create relpns
        self.obj_relpn = SpatialRelPN(2048)
        self.txt_semantic_relpn = BiLSTMRelPN(300, 256)

        self.txt_syntactic_gcn_0 = GATConv(
            txt_syntactic_gcn.shape[0], txt_syntactic_gcn.shape[1]
        )
        self.txt_semantic_gcn_0 = GATConv(
            txt_semantic_gcn.shape[0], txt_semantic_gcn.shape[1]
        )
        self.obj_feat_gcn_0 = GATConv(obj_feat_gcn.shape[0], obj_feat_gcn.shape[1])
        self.obj_pos_gcn_0 = GATConv(obj_pos_gcn.shape[0], obj_pos_gcn.shape[1])

        self.txt_syntactic_gcn_1 = GATConv(
            txt_syntactic_gcn.shape[1], txt_syntactic_gcn.shape[2]
        )
        self.txt_semantic_gcn_1 = GATConv(
            txt_semantic_gcn.shape[1], txt_semantic_gcn.shape[2]
        )
        self.obj_feat_gcn_1 = GATConv(obj_feat_gcn.shape[1], obj_feat_gcn.shape[2])
        self.obj_pos_gcn_1 = GATConv(obj_pos_gcn.shape[1], obj_pos_gcn.shape[2])

        self.txt_syntactic_gcn_2 = GATConv(
            txt_syntactic_gcn.shape[2], txt_syntactic_gcn.shape[3]
        )
        self.txt_semantic_gcn_2 = GATConv(
            txt_semantic_gcn.shape[2], txt_semantic_gcn.shape[3]
        )
        self.obj_feat_gcn_2 = GATConv(obj_feat_gcn.shape[2], obj_feat_gcn.shape[3])
        self.obj_pos_gcn_2 = GATConv(obj_pos_gcn.shape[2], obj_pos_gcn.shape[3])

        self.txt_obj_biatt_0 = BidirectionalAttention(
            key_shape=(obj_feat_gcn.shape[1], obj_feat_gcn.shape[1]),
            query_shape=(txt_semantic_gcn.shape[1], txt_semantic_gcn.shape[1]),
            heads=1,
            bidirectional=True,
        )

        self.txt_obj_biatt_1 = BidirectionalAttention(
            key_shape=(obj_feat_gcn.shape[2], obj_feat_gcn.shape[2]),
            query_shape=(txt_semantic_gcn.shape[2], txt_semantic_gcn.shape[2]),
            heads=1,
            bidirectional=True,
        )

        self.txt_obj_biatt_2 = BidirectionalAttention(
            key_shape=(obj_feat_gcn.shape[3], obj_feat_gcn.shape[3]),
            query_shape=(txt_semantic_gcn.shape[3], txt_semantic_gcn.shape[3]),
            heads=1,
            bidirectional=True,
        )

        # LINEAR FUSION
        # Create 2-layer MLP fusion net that takes in conatenated feats from gcns.
        in_dim = (
            txt_syntactic_gcn.shape[-1]
            + txt_semantic_gcn.shape[-1]
            + obj_feat_gcn.shape[-1]
            + 8
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(in_dim, (in_dim + num_answer_classes) // 2),
            torch.nn.Linear((in_dim + num_answer_classes) // 2, num_answer_classes),
        )

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

    def forward(
        self,
        word_embeddings: Union[PackedSequence, torch.Tensor],
        dependencies: Data,
        objects: torch.FloatTensor,
        boxes: torch.FloatTensor,
    ) -> Any:
        """Propagate data through the model."""
        # pylint: disable=too-many-locals

        # Alias for readability
        txt_syn_graph = dependencies

        # Get RelPN relationship predictions
        obj_relations, obj_scores, intersections = self.obj_relpn(
            objects, boxes
        )  # TODO initialise GATConv with edge scores (img_scores.unsqueeze(-1))?
        # TODO consider whether to use Glove or LSTM feats as node features
        (
            semantic_relations,
            semantic_scores,
            semantic_feats,
        ) = self.txt_semantic_relpn(word_embeddings)

        # BUILD GRAPHS
        # Build rcnn feature graph
        obj_feat_graph = Batch.from_data_list(
            [
                Data(
                    edge_index=img_relations.t(),
                    # edge_attr=obj_scores,
                    x=img_objects,  # x has shape (num_nodes, num_node_features)
                )
                for img_relations, img_scores, img_objects in zip(
                    obj_relations, obj_scores, objects
                )
            ]
        ).to(
            "cuda"
        )  # TODO handle devices better

        # Build rcnn proposal graph
        selected_boxes = [
            torch.stack(
                [
                    torch.cat((img_boxes[i], img_boxes[j]), dim=0)
                    for i, j in img_relation_pairs
                ],
                dim=0,
            )
            for img_relation_pairs, img_boxes in zip(obj_relations, boxes)
        ]  # TODO make more efficient, maybe with torch.gather
        obj_pos_graph = Batch.from_data_list(
            [
                Data(
                    edge_index=img_relations.t(),
                    # edge_attr=obj_scores,
                    x=img_boxes,  # x has shape (num_nodes, num_node_features)
                )
                for img_relations, img_scores, img_boxes in zip(
                    obj_relations, obj_scores, selected_boxes
                )
            ]
        ).to("cuda")

        # build text semantic graph
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
        ).to(
            "cuda"
        )  # TODO handle devices better

        txt_syn_graph.x = self.txt_syntactic_gcn_0(
            txt_syn_graph.x, txt_syn_graph.edge_index
        )
        txt_sem_graph.x = self.txt_semantic_gcn_0(
            txt_sem_graph.x, txt_sem_graph.edge_index
        )
        obj_feat_graph.x = self.obj_feat_gcn_0(
            obj_feat_graph.x, obj_feat_graph.edge_index
        )
        obj_pos_graph.x = self.obj_pos_gcn_0(obj_pos_graph.x, obj_pos_graph.edge_index)

        obj_feat_graph.x, txt_sem_graph.x = self.txt_obj_biatt_0(
            obj_feat_graph.x, txt_sem_graph.x
        )

        txt_syn_graph.x = self.txt_syntactic_gcn_1(
            txt_syn_graph.x, txt_syn_graph.edge_index
        )
        txt_sem_graph.x = self.txt_semantic_gcn_1(
            txt_sem_graph.x, txt_sem_graph.edge_index
        )
        obj_feat_graph.x = self.obj_feat_gcn_1(
            obj_feat_graph.x, obj_feat_graph.edge_index
        )
        obj_pos_graph.x = self.obj_pos_gcn_1(obj_pos_graph.x, obj_pos_graph.edge_index)

        obj_feat_graph.x, txt_sem_graph.x = self.txt_obj_biatt_1(
            obj_feat_graph.x, txt_sem_graph.x
        )

        txt_syn_graph.x = self.txt_syntactic_gcn_2(
            txt_syn_graph.x, txt_syn_graph.edge_index
        )
        txt_sem_graph.x = self.txt_semantic_gcn_2(
            txt_sem_graph.x, txt_sem_graph.edge_index
        )
        obj_feat_graph.x = self.obj_feat_gcn_2(
            obj_feat_graph.x, obj_feat_graph.edge_index
        )
        obj_pos_graph.x = self.obj_pos_gcn_2(obj_pos_graph.x, obj_pos_graph.edge_index)

        obj_feat_graph.x, txt_sem_graph.x = self.txt_obj_biatt_2(
            obj_feat_graph.x, txt_sem_graph.x
        )

        # LINEAR FUSION
        txt_syn_pooled_feats = global_mean_pool(txt_syn_graph.x, txt_syn_graph.batch)
        txt_sem_pooled_feats = global_mean_pool(txt_sem_graph.x, txt_sem_graph.batch)
        obj_feat_pooled_feats = global_mean_pool(obj_feat_graph.x, obj_feat_graph.batch)
        obj_pos_pooled_feats = global_mean_pool(obj_pos_graph.x, obj_pos_graph.batch)

        fused_feats = self.fusion(
            torch.cat(
                [
                    txt_syn_pooled_feats,
                    obj_feat_pooled_feats,
                    txt_sem_pooled_feats,
                    obj_pos_pooled_feats,
                ],
                dim=1,
            )
        )

        # ALIGNMENT EXPERIMENT TODO
        # aligned_feats = self.alignment(
        #     (txt_syn_graph.x, txt_sem_graph.x, obj_pos_graph.x)
        # )
        # common_dim = 128
        # if self.fusion is None:
        #     self.fusion = torch.nn.ModuleList([
        #         torch.nn.Linear(feat.size(0), common_dim)
        #         for feat in aligned_feats
        #     ])
        # fused_feats = [
        #     mlp(feats.t()) for mlp, feats in zip(self.fusion, aligned_feats)
        # ]

        # [(num_query_nodes, num_heads * self.proj_dim), ...]
        # cat_alignments = torch.cat(alignments, dim=0)

        return F.log_softmax(fused_feats, dim=1)
