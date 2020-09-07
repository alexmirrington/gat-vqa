"""Multi-gcn model."""
import math
from typing import Any, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from .gat import GAT
from .gcn import GCN

# from .bidirectional_attention import BidirectionalAttention


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

        self.max_qn_length = 28  # max question length across entire dataset
        self.truncate_qn_length = self.max_qn_length

        # LINEAR FUSION
        # Create 2-layer MLP fusion net that takes in conatenated feats from gcns.
        in_dim = self.truncate_qn_length * (
            self.text_gcn.shape[-1] + self.scene_gcn.shape[-1]
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(in_dim, num_answer_classes),
            torch.nn.Linear(num_answer_classes, num_answer_classes),
        )

    def forward(self, dependencies: Data, objects: Data) -> Any:
        """Propagate data through the model."""
        # pylint: disable=too-many-locals

        _ = self.text_gcn(dependencies)
        _ = self.scene_gcn(objects)

        # Attention alignment like in "aligned dual channel gcns for vqa"

        # (batch_size, max_question_length, num_word_feats)
        question_words, question_lengths = to_dense_batch(
            dependencies.x, batch=dependencies.batch
        )
        question_lengths = question_lengths.type(torch.IntTensor).sum(dim=-1)
        # print(f"{question_lengths=}")
        # Self attention over words to determine which are important
        word_alignment = torch.bmm(
            question_words, torch.transpose(question_words, 1, 2)
        ) / math.sqrt(
            question_words.size(-1)
        )  # (batch_size, max_question_length, max_question_length)
        word_alignment = torch.softmax(
            word_alignment, dim=-1
        )  # (batch_size, max_question_length, max_question_length)
        word_self_attention = torch.bmm(
            word_alignment, question_words
        )  # (batch_size, max_question_length, num_word_feats)
        # print(f"{word_self_attention.size()=}")

        # (batch_size, max_object_count, num_object_feats)
        # Attention over scene objects to determine which are important to
        # attended question
        sg_object_feats, sg_object_counts = to_dense_batch(
            objects.x, batch=objects.batch
        )
        sg_object_counts = sg_object_counts.type(torch.IntTensor).sum(dim=-1)
        # print(f"{sg_object_counts=}")
        sg_alignment = torch.bmm(
            word_self_attention, torch.transpose(sg_object_feats, 1, 2)
        ) / math.sqrt(
            sg_object_feats.size(-1)
        )  # (batch_size, max_question_length, max_object_count)
        sg_alignment = torch.softmax(
            sg_alignment, dim=-1
        )  # (batch_size, max_question_length, max_object_count)
        word_sg_attention = torch.bmm(
            sg_alignment, sg_object_feats
        )  # (batch_size, max_question_length, num_object_feats)
        # print(f"{word_sg_attention.size()=}")

        # Pack the attention sequences according to number of words
        packed_self_attns = pack_padded_sequence(
            word_self_attention,
            question_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_word_sg_attns = pack_padded_sequence(
            word_sg_attention, question_lengths, batch_first=True, enforce_sorted=False
        )
        padded_self_attns, _ = pad_packed_sequence(
            packed_self_attns, total_length=self.max_qn_length, batch_first=True
        )
        padded_self_attns = padded_self_attns[:, : self.truncate_qn_length, :]
        padded_word_sg_attns, _ = pad_packed_sequence(
            packed_word_sg_attns, total_length=self.max_qn_length, batch_first=True
        )
        padded_word_sg_attns = padded_word_sg_attns[:, : self.truncate_qn_length, :]

        fused_feats = self.fusion(
            torch.cat(
                [
                    padded_self_attns.flatten(start_dim=1),
                    padded_word_sg_attns.flatten(start_dim=1),
                ],
                dim=1,
            )
        )

        return F.log_softmax(fused_feats, dim=1)
