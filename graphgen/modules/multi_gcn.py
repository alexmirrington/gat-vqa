"""Multi-gcn model."""
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.data import Data
from torch_geometric.nn.conv import GATConv, GCNConv
from torch_geometric.utils import to_dense_batch

from ..config.model import GCNConvName


class MultiGCN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Multi-gcn model that operates on pre-extracted FasterRCNN features or \
    grount-truth scene graph data."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        num_answer_classes: int,
        text_gcn_shape: Sequence[int],
        text_gcn_conv: GCNConvName,
        scene_gcn_shape: Sequence[int],
        scene_gcn_conv: GCNConvName,
    ) -> None:
        """Create a multi-gcn model with bidirectional attention."""
        super().__init__()
        self.num_answer_classes = num_answer_classes

        self.text_gcn_layers = torch.nn.ModuleList([])
        for idx in range(1, len(text_gcn_shape)):
            if text_gcn_conv == GCNConvName.GCN:
                self.text_gcn_layers.append(
                    GCNConv(text_gcn_shape[idx - 1], text_gcn_shape[idx])
                )
            elif text_gcn_conv == GCNConvName.GAT:
                self.text_gcn_layers.append(
                    GATConv(text_gcn_shape[idx - 1], text_gcn_shape[idx], heads=1)
                )
            else:
                raise NotImplementedError()

        self.scene_gcn_layers = torch.nn.ModuleList([])
        for idx in range(1, len(scene_gcn_shape)):
            if scene_gcn_conv == GCNConvName.GCN:
                self.scene_gcn_layers.append(
                    GCNConv(scene_gcn_shape[idx - 1], scene_gcn_shape[idx])
                )
            elif scene_gcn_conv == GCNConvName.GAT:
                self.scene_gcn_layers.append(
                    GATConv(scene_gcn_shape[idx - 1], scene_gcn_shape[idx], heads=1)
                )
            else:
                raise NotImplementedError()

        self.max_qn_length = 29  # max question length across entire dataset
        self.truncate_qn_length = self.max_qn_length

        # LINEAR FUSION
        # Create 2-layer MLP fusion net that takes in conatenated feats from gcns.
        in_dim = self.truncate_qn_length * (text_gcn_shape[-1] + scene_gcn_shape[-1])
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(in_dim, num_answer_classes),
            torch.nn.Linear(num_answer_classes, num_answer_classes),
        )

    def forward(self, dependencies: Data, objects: Data) -> Any:
        """Propagate data through the model."""
        # pylint: disable=too-many-locals

        for idx in range(min(len(self.text_gcn_layers), len(self.scene_gcn_layers))):
            # Get text and scene feats for current layer
            text_gcn_layer = self.text_gcn_layers[idx]
            scene_gcn_layer = self.scene_gcn_layers[idx]

            dependencies.x = text_gcn_layer(dependencies.x, dependencies.edge_index)
            objects.x = scene_gcn_layer(objects.x, objects.edge_index)

            # Apply dense bidirectional attention to node features
            # TODO  multihead_attn = torch.nn.MultiheadAttention(300, self.heads)

        # Apply any leftover conv layers for text gcn
        for leftover_idx in range(idx + 1, len(self.text_gcn_layers)):
            text_gcn_layer = self.text_gcn_layers[leftover_idx]
            dependencies.x = text_gcn_layer(dependencies.x, dependencies.edge_index)

        # Apply any leftover conv layers for scene gcn
        for leftover_idx in range(idx + 1, len(self.scene_gcn_layers)):
            scene_gcn_layer = self.scene_gcn_layers[leftover_idx]
            objects.x = scene_gcn_layer(objects.x, objects.edge_index)

        # Attention alignment like in "aligned dual channel gcns for vqa"
        # question_words: (batch_size, max_question_length, num_word_feats)
        question_words, question_lengths = to_dense_batch(
            dependencies.x, batch=dependencies.batch
        )
        question_lengths = question_lengths.type(torch.IntTensor).sum(dim=-1)

        # Self attention over words to determine which are important
        word_alignment = torch.bmm(
            question_words, torch.transpose(question_words, 1, 2)
        ) / math.sqrt(
            question_words.size(-1)
        )  # word_alignment: (batch_size, max_question_length, max_question_length)
        word_alignment = torch.softmax(
            word_alignment, dim=-1
        )  # word_alignment: (batch_size, max_question_length, max_question_length)
        word_self_attention = torch.bmm(
            word_alignment, question_words
        )  # word_self_attention: (batch_size, max_question_length, num_word_feats)

        # Attention over scene objects to determine which object features are
        # important to the self-attended question
        # sg_object_feats: (batch_size, max_object_count, num_object_feats)
        sg_object_feats, sg_object_counts = to_dense_batch(
            objects.x, batch=objects.batch
        )
        sg_object_counts = sg_object_counts.type(torch.IntTensor).sum(dim=-1)
        sg_alignment = torch.bmm(
            word_self_attention, torch.transpose(sg_object_feats, 1, 2)
        ) / math.sqrt(
            sg_object_feats.size(-1)
        )  # sg_alignment: (batch_size, max_question_length, max_object_count)
        sg_alignment = torch.softmax(
            sg_alignment, dim=-1
        )  # sg_alignment: (batch_size, max_question_length, max_object_count)
        word_sg_attention = torch.bmm(
            sg_alignment, sg_object_feats
        )  # word_sg_attention: (batch_size, max_question_length, num_object_feats)

        # Pack and unpack the attention sequences according to number of words
        # to ensure consistent question length across all batches for fusion
        # layer compatibility.
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
