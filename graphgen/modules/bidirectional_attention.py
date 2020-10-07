"""Implementation of a pairwise general attention layer."""
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter


class BidirectionalAttention(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """Implementation of a pairwise general attention layer, whereby general \
    attention is computed between node features of multiple graphs.

    References:
    -----------
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/\
    torch_geometric/nn/conv/gat_conv.html#GATConv
    https://arxiv.org/abs/1710.10903
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        key_shape: Tuple[int, int],
        query_shape: Tuple[int, int],
        heads: int = 1,
        bidirectional: bool = True,
        return_attention_weights: bool = False,
    ):
        """Create a `BidirectionalAttention` module."""
        super().__init__()

        # Num in and out channels for the attention key graph's node features.
        self.key_shape = key_shape
        # Num in and out channels for the attention query graph's node features.
        self.query_shape = query_shape
        # The number of attention heads to use.
        self.heads = heads

        # Whether to use both the key and query as attention values or just the key.
        self.bidirectional = bidirectional
        self.return_attention_weights = return_attention_weights

        # Create linear layers for each of the input features
        self.k_lin = Linear(
            key_shape[0], heads * key_shape[1], bias=False
        )  # TODO handle biases
        self.q_lin = Linear(
            query_shape[0], heads * query_shape[1], bias=False
        )  # TODO handle biases

        self.attn_weight = Parameter(
            torch.Tensor(self.heads, key_shape[1], query_shape[1])
        )  # TODO support other attention types if vectors are same dimension

        # TODO decide whether to have a shared set of projection weights for
        # k -> q and q -> k spaces. My first thought is yes
        # TODO decide whether to have one set of prejection weights for each head.
        # Projection weights are for mapping output space of query to key and
        # vice versa if bidirectional is True.
        # TODO work out if single weight matrix is good enough
        self.projection_weight = Parameter(torch.Tensor(key_shape[1], query_shape[1]))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the module's parameters to their defaults."""
        torch.nn.init.kaiming_uniform_(self.attn_weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.projection_weight, a=math.sqrt(5))

    def forward(
        self,
        x_k: Tensor,  # (num_nodes_k, num_in_feats), num_in_feats = self.key_shape[0]
        x_q: Tensor,  # (num_nodes_q, num_in_feats), num_in_feats = self.query_shape[0]
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Propagate data through the module."""
        c_k, c_q = self.key_shape[1], self.query_shape[1]

        # Apply linear layers to each node feature set
        x_k_proj = torch.transpose(self.k_lin(x_k).view(-1, self.heads, c_k), 0, 1)
        x_q_proj = torch.transpose(self.q_lin(x_q).view(-1, self.heads, c_q), 0, 1)

        # x_k has shape (self.heads, num_nodes_k, c_k)
        # x_q has shape (self.heads, num_nodes_q, c_q)
        # self.attn_weights has shape (H, c_k, c_q)
        # x_k W x_q^T => W should be (H, num_nodes_k, num_nodes_q)
        alphas = torch.bmm(
            torch.bmm(x_k_proj, self.attn_weight),
            torch.transpose(x_q_proj, 1, 2),
        )

        # Calculate attention weighted sums of node features from the query graph
        # for each node in the key graph. If bidirectional, do the converse, using
        # query as key and key as query.

        # alpha_k_feats has shape (self.heads, num_nodes_k, c_q)
        # For each attention head, the (num_nodes_k, c_q) matrix represents
        # an attention-weighted sum of the node features in x_q for each node in x_k

        # alpha_q_feats has shape (self.heads, num_nodes_q, c_k)
        # For each attention head, the (num_nodes_q, c_k) matrix represents
        # an attention-weighted sum of the node features in x_k for each node in x_q

        alpha_ks = None
        alpha_qs = None
        x_k_update = 0
        x_q_update = 0

        alpha_ks = F.softmax(alphas, dim=2)
        alpha_k_feats = torch.bmm(alpha_ks, x_q_proj)  # TODO add more weights?

        # Map query feature space back to key feature space
        # TODO experiment with different multihead pooling (concat, average etc)
        alpha_k_agg = torch.mean(alpha_k_feats, dim=0)
        x_k_update = torch.mm(alpha_k_agg, self.projection_weight.t())

        if self.bidirectional:
            alpha_qs = F.softmax(
                torch.transpose(alphas, 1, 2), dim=2
            )  # Try tanh/additive attention for stability?
            alpha_q_feats = torch.bmm(alpha_qs, x_k_proj)  # TODO add more weights?

            # TODO experiment with different multihead pooling (concat, average etc)
            alpha_q_agg = torch.mean(alpha_q_feats, dim=0)
            x_q_update = torch.mm(alpha_q_agg, self.projection_weight)

        # TODO consider placing a sigmoid gate on the update, so the model can
        # learn to cut off attention signals if needed.
        if self.return_attention_weights:
            return x_k + x_k_update, x_q + x_q_update, alpha_ks, alpha_qs
        return x_k + x_k_update, x_q + x_q_update


class PairwiseAlignment(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """A `PairwiseAlignment` fusion module."""

    def __init__(
        self,
        feat_dims: Sequence[int],
        proj_dim: int,
        heads: int,
        aggregate: Optional[str] = "concat",
    ):
        """Create a `PairwiseAlignment` module."""
        super().__init__()
        self.feat_dims = feat_dims
        self.proj_dim = proj_dim
        self.heads = heads
        self.aggregate = aggregate

        self.linear_projections = torch.nn.ModuleList(
            [
                torch.nn.Linear(feat_dim, heads * self.proj_dim, bias=True)
                for feat_dim in self.feat_dims
            ]
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Propagate data through the module."""
        projected_feats = [
            torch.transpose(proj(feat).view(-1, self.heads, self.proj_dim), 0, 1)
            for proj, feat in zip(self.linear_projections, feats)
        ]
        print(f"{[pfeat.size() for pfeat in projected_feats]=}")
        aligned = []
        for key_feats in projected_feats:
            for query_feats in projected_feats:
                # key_feats (self.heads, num_key_nodes, self.proj_dim)
                # query_feats (self.heads, num_query_nodes, self.proj_dim)

                # Compute dotprod attention
                # TODO reduce redundant attention calcs and compute bidirectional
                # attention here
                alignment = (
                    torch.bmm(query_feats, torch.transpose(key_feats, 1, 2))
                    / self.proj_dim
                )
                # alignment (self.heads, num_query_nodes, num_key_nodes)
                alignment = torch.softmax(
                    alignment, dim=2
                )  # Softmax across attention keys
                # Obtain the parts of the value (key) that are relevant to the query,
                # i.e. we have tensor of shape
                # (self.heads, num_query_nodes, self.proj_dim),
                # the weighted sum of each projected key feature for each query node.
                aligned_values = torch.bmm(alignment, key_feats)
                if self.aggregate is None:
                    aligned.append(aligned_values)
                elif self.aggregate == "concat":
                    aligned.append(
                        torch.flatten(
                            torch.transpose(aligned_values, 0, 1), start_dim=1
                        )
                    )
                    print(f"{aligned[-1].size()}")
                elif self.aggregate == "mean":
                    aligned.append(torch.mean(aligned_values, dim=0))
                else:
                    raise NotImplementedError()
        return aligned
