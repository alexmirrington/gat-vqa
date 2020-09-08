"""Implementation of a MAC recurrent cell write unit."""
import torch
import torch.nn.functional as F
from torch import nn

from .utils import xavier_uniform_linear


class WriteUnit(nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """A MAC recurrent cell write unit."""

    def __init__(self, hidden_dim: int = 512, self_attention: bool = False) -> None:
        """Initialise the write unit."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_attention = self_attention

        if self.self_attention:
            self.control = xavier_uniform_linear(self.hidden_dim, self.hidden_dim)
            self.attn = xavier_uniform_linear(self.hidden_dim, 1)
            self.concat = xavier_uniform_linear(self.hidden_dim * 3, self.hidden_dim)
        else:
            self.concat = xavier_uniform_linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(
        self, memories: torch.Tensor, retrieved: torch.Tensor, controls: torch.Tensor
    ) -> torch.Tensor:
        """Propagate data through the model."""
        # optionally project info if config.writeInfoProj:

        # optional info nonlinearity if writeInfoAct != 'NON'

        # compute self-attention vector based on previous controls and memories
        if self.self_attention:
            self_control = controls[-1]
            self_control = self.control(self_control)
            controls_cat = torch.stack(controls[:-1], 2)
            attn = self_control.unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            # next_mem = self.W_s(attn_mem) + self.W_p(concat)

        prev_mem = memories[-1]
        # get write unit inputs: previous memory, the new info,
        # optionally self-attention / control
        concat = torch.cat([retrieved, prev_mem], 1)

        if self.self_attention:
            concat = torch.cat([concat, attn_mem], 1)

        # project memory back to memory dimension if config.writeMemProj
        concat = self.concat(concat)

        # optional memory nonlinearity

        # write unit gate moved to RNNWrapper

        # optional batch normalization

        next_mem = concat

        return next_mem
