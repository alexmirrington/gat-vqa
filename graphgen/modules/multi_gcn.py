"""Placeholder wrapper model."""
from typing import Any, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

from .bidirectional_attention import BidirectionalAttention
from .bilstm_relpn import BiLSTMRelPN
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
        txt_syntactic_gcn: GCN,
        txt_semantic_gcn: GCN,
        obj_positional_gcn: GCN,
    ) -> None:
        """Create a multi-gcn model with bidirectional attention."""
        super().__init__()
        self.num_answer_classes = num_answer_classes
        self.num_object_classes = num_object_classes

        # Create GCNs.
        # self.txt_syntactic_gcn = txt_syntactic_gcn  # TODO uncomment
        # self.txt_semantic_gcn = txt_semantic_gcn  # TODO uncomment
        # self.obj_positional_gcn = obj_positional_gcn  # TODO uncomment

        # Create relpns
        self.obj_positional_relpn = RelPN(2048)
        self.txt_contextual_relpn = BiLSTMRelPN(300, 256)

        self.txt_syntactic_gcn_0 = GCNConv(
            txt_syntactic_gcn.shape[0], txt_syntactic_gcn.shape[1]
        )
        self.txt_semantic_gcn_0 = GCNConv(
            txt_semantic_gcn.shape[0], txt_semantic_gcn.shape[1]
        )
        self.obj_positional_gcn_0 = GCNConv(
            obj_positional_gcn.shape[0], obj_positional_gcn.shape[1]
        )
        self.txt_syntactic_gcn_1 = GCNConv(
            txt_syntactic_gcn.shape[1], txt_syntactic_gcn.shape[2]
        )
        self.txt_semantic_gcn_1 = GCNConv(
            txt_semantic_gcn.shape[1], txt_semantic_gcn.shape[2]
        )
        self.obj_positional_gcn_1 = GCNConv(
            obj_positional_gcn.shape[1], obj_positional_gcn.shape[2]
        )
        self.txt_syntactic_gcn_2 = GCNConv(
            txt_syntactic_gcn.shape[2], txt_syntactic_gcn.shape[3]
        )
        self.txt_semantic_gcn_2 = GCNConv(
            txt_semantic_gcn.shape[2], txt_semantic_gcn.shape[3]
        )
        self.obj_positional_gcn_2 = GCNConv(
            obj_positional_gcn.shape[2], obj_positional_gcn.shape[3]
        )
        # use object semantic features as an attention query on the dependency
        # parser feats. i.e. we are querying obj-semantic feats to exctract information
        # to propagate to the dependency parser feats.

        self.txt_obj_biatt_0 = BidirectionalAttention(
            key_shape=(obj_positional_gcn.shape[1], obj_positional_gcn.shape[1]),
            query_shape=(txt_semantic_gcn.shape[1], txt_semantic_gcn.shape[1]),
            heads=1,
            bidirectional=True,
        )
        self.txt_obj_biatt_1 = BidirectionalAttention(
            key_shape=(obj_positional_gcn.shape[2], obj_positional_gcn.shape[2]),
            query_shape=(txt_semantic_gcn.shape[2], txt_semantic_gcn.shape[2]),
            heads=1,
            bidirectional=True,
        )

        self.txt_obj_biatt_2 = BidirectionalAttention(
            key_shape=(obj_positional_gcn.shape[3], obj_positional_gcn.shape[3]),
            query_shape=(txt_semantic_gcn.shape[3], txt_semantic_gcn.shape[3]),
            heads=1,
            bidirectional=True,
        )

        # Create 2-layer MLP fusion net that takes in conatenated feats from gcns.
        in_dim = (
            txt_syntactic_gcn.shape[-1]
            + txt_semantic_gcn.shape[-1]
            + obj_positional_gcn.shape[-1]
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(in_dim, (in_dim + num_answer_classes) // 2),
            torch.nn.Linear((in_dim + num_answer_classes) // 2, num_answer_classes),
        )

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
        txt_syn_data = dependencies

        # Get RelPN relationship predictions
        relations, scores = self.obj_positional_relpn(objects, boxes)
        (
            contextual_relations,
            contextual_scores,
            contextual_feats,
        ) = self.txt_contextual_relpn(word_embeddings)
        obj_pos_data = Batch.from_data_list(
            [
                Data(
                    edge_index=img_relations.t(),
                    edge_attr=img_scores.unsqueeze(-1),
                    x=img_objects,  # x has shape (num_nodes, num_node_features)
                )
                for img_relations, img_scores, img_objects in zip(
                    relations, scores, objects
                )
            ]
        ).to(
            "cuda"
        )  # TODO handle devices better

        # TODO consider whether to use Glove of LSTM feats as node features
        txt_ctx_data = Batch.from_data_list(
            [
                Data(
                    edge_index=txt_relations,
                    edge_attr=txt_scores,
                    x=txt_feats,
                )
                for txt_relations, txt_scores, txt_feats in zip(
                    contextual_relations,
                    contextual_scores,
                    contextual_feats,  # pad_packed_sequence(word_embeddings)[0]
                )
            ]
        ).to(
            "cuda"
        )  # TODO handle devices better

        txt_syn_data.x = self.txt_syntactic_gcn_0(
            txt_syn_data.x, txt_syn_data.edge_index
        )
        txt_ctx_data.x = self.txt_semantic_gcn_0(
            txt_ctx_data.x, txt_ctx_data.edge_index
        )
        obj_pos_data.x = self.obj_positional_gcn_0(
            obj_pos_data.x, obj_pos_data.edge_index
        )

        obj_pos_data.x, txt_ctx_data.x = self.txt_obj_biatt_0(
            obj_pos_data.x, txt_ctx_data.x
        )

        txt_syn_data.x = self.txt_syntactic_gcn_1(
            txt_syn_data.x, txt_syn_data.edge_index
        )
        txt_ctx_data.x = self.txt_semantic_gcn_1(
            txt_ctx_data.x, txt_ctx_data.edge_index
        )
        obj_pos_data.x = self.obj_positional_gcn_1(
            obj_pos_data.x, obj_pos_data.edge_index
        )

        obj_pos_data.x, txt_ctx_data.x = self.txt_obj_biatt_1(
            obj_pos_data.x, txt_ctx_data.x
        )

        txt_syn_data.x = self.txt_syntactic_gcn_2(
            txt_syn_data.x, txt_syn_data.edge_index
        )
        txt_ctx_data.x = self.txt_semantic_gcn_2(
            txt_ctx_data.x, txt_ctx_data.edge_index
        )
        obj_pos_data.x = self.obj_positional_gcn_2(
            obj_pos_data.x, obj_pos_data.edge_index
        )

        obj_pos_data.x, txt_ctx_data.x = self.txt_obj_biatt_2(
            obj_pos_data.x, txt_ctx_data.x
        )

        txt_syn_pooled_feats = global_mean_pool(txt_syn_data.x, txt_syn_data.batch)
        txt_sem_pooled_feats = global_mean_pool(txt_ctx_data.x, txt_ctx_data.batch)
        obj_pos_pooled_feats = global_mean_pool(obj_pos_data.x, obj_pos_data.batch)

        fused_feats = self.fusion(
            torch.cat(
                [txt_syn_pooled_feats, obj_pos_pooled_feats, txt_sem_pooled_feats],
                dim=1,
            )
        )
        return F.log_softmax(fused_feats, dim=1)
