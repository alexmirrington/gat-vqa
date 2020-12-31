"""Implementation of a MAC recurrent cell control unit."""
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .utils import xavier_uniform_linear


class ControlUnit(nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """A MAC recurrent cell control unit."""

    def __init__(self, control_dim: int = 512, length: int = 12) -> None:
        """Initialise the control unit.

        Params:
        -------
        `dim`: The dimension of the control vector.
        `length`: The length of the overall MAC network, i.e. the max number of
        reasoning steps the network intends to perform.

        Returns:
        --------
        `None`
        """
        super().__init__()
        self.control_dim = control_dim
        self.shared_control_proj = (
            xavier_uniform_linear(  # Could make this question dim to control dim
                self.control_dim, self.control_dim
            )
        )
        self.position_aware = nn.ModuleList()
        for i in range(length):
            self.position_aware.append(
                xavier_uniform_linear(self.control_dim, self.control_dim)
            )
        self.control_question = xavier_uniform_linear(
            self.control_dim * 2, self.control_dim
        )
        self.attn = xavier_uniform_linear(self.control_dim, 1)

    def forward(
        self,
        context: torch.Tensor,
        question: torch.Tensor,
        controls: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Propagate data through the model."""
        cur_step = len(controls) - 1

        # prepare question input to control
        question = F.elu(  # torch.tanh
            self.shared_control_proj(question)  # Included in MACCell.__call__
        )  # TODO: avoid repeating call

        # question = torch.tanh(  # ELU?
        #     self.shared_control_proj(question)  # Included in MACCell.__call__
        # )  # TODO: avoid repeating call

        position_aware = self.position_aware[cur_step](question)
        # Unshared due to --controlInputUnshared being set by default in
        # original code base.

        # Compute "continuous" control state given previous control and question.
        # control inputs: question and previous control. Runs if given CLEVR args1.txt,
        # but not included for GQA
        # control = controls[-1]
        # control_question = torch.cat([control, position_aware], 1)
        # control_question = self.control_question(control_question)
        # control_question = control_question.unsqueeze(1)

        control_question = position_aware.unsqueeze(1)

        context_prod = control_question * context

        # ++ optionally concatenate words (= context)

        # optional projection (if config.controlProj) --> stacks another
        # linear after activation

        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        # only valid if self.inwords == self.outwords
        next_control = (attn * context).sum(1)

        return next_control, attn.squeeze(-1)
