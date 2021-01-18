"""Implementation of a MAC recurrent cell read unit."""
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .utils import xavier_uniform_linear


class ReadUnit(nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """A MAC recurrent cell read unit."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        memory_dim: int = 512,  # TODO consider different kb_dim
        variational_dropout: bool = True,
        dropout: float = 0.15,
    ):
        """Initialise the read unit."""
        super().__init__()
        self.memory_dim = memory_dim
        self.variational_dropout = variational_dropout
        self.dropout = dropout

        self.read_dropout = nn.Dropout(self.dropout)
        self.mem_proj = xavier_uniform_linear(self.memory_dim, self.memory_dim)
        self.kb_proj = xavier_uniform_linear(self.memory_dim, self.memory_dim)
        self.concat = xavier_uniform_linear(self.memory_dim * 2, self.memory_dim)
        self.concat2 = xavier_uniform_linear(self.memory_dim, self.memory_dim)
        self.attn = xavier_uniform_linear(self.memory_dim, 1)

    def forward(
        self,
        memories: Sequence[torch.Tensor],
        know: torch.Tensor,
        controls: Sequence[torch.Tensor],
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate data through the model."""
        # Step 1: knowledge base / memory interactions
        last_mem = memories[-1]
        if self.training:
            if self.variational_dropout:
                last_mem = memories[-1] * masks
            else:
                last_mem = self.read_dropout(memories[-1])
        know = self.read_dropout(know)
        proj_mem = self.mem_proj(last_mem).unsqueeze(1)
        # proj_know is (batch_size, num_objects, memory_dim)
        proj_know = self.kb_proj(know)
        concat = torch.cat(
            [
                proj_mem * proj_know,
                proj_know,  # This is originally set by the flag `readMemConcatProj`
            ],
            2,
        )

        # Project memory interactions back to hidden dimension, this is enabled
        # in GQA configs in ofifcial repo, even though not reported in the paper.
        concat = self.concat2(
            F.elu(self.concat(concat))
        )  # if readMemProj ++ second projection and nonlinearity if readMemAct

        # Step 2: compute interactions with control (if config.readCtrl)
        attn = F.elu(concat * controls[-1].unsqueeze(1))

        # if readCtrlConcatInter torch.cat([interactions, concat])

        # optionally concatenate knowledge base elements

        # optional nonlinearity

        attn = self.read_dropout(attn)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(2)

        read = (attn * know).sum(1)

        return read
