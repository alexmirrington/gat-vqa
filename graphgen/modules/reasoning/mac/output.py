"""Implementation of a MAC cell output module."""
import torch
from torch import nn
from torch.nn.init import xavier_uniform_


class OutputUnit(nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """An output classifier for a MAC network."""

    def __init__(
        self, input_dim: int = 512, output_dim: int = 28, dropout: float = 0.15
    ) -> None:
        """Initialise an `OutputUnit` instance."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.question_proj = nn.Linear(self.input_dim, self.input_dim)
        self.classifier_out = nn.Sequential(
            nn.Dropout(p=self.dropout),  # output dropout outputDropout=0.85
            nn.Linear(self.input_dim * 2, self.input_dim + self.output_dim // 2),
            nn.ELU(),
            nn.Dropout(p=self.dropout),  # output dropout outputDropout=0.85
            nn.Linear(self.input_dim + self.output_dim // 2, self.output_dim),
        )
        xavier_uniform_(self.classifier_out[1].weight)
        xavier_uniform_(self.classifier_out[4].weight)

    def forward(self, last_mem: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        """Propagate data through the model."""
        question = self.question_proj(question)
        cat = torch.cat([last_mem, question], 1)
        out = self.classifier_out(cat)
        return out
