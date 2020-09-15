"""Implementation of a MAC cell and recurrent network."""
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from ..abstract_reasoning_module import AbstractReasoningModule
from .control import ControlUnit
from .output import OutputUnit
from .read import ReadUnit
from .utils import xavier_uniform_linear
from .write import WriteUnit


class MACCell(nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """A MAC recurrent cell, containing a control, read and write unit."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        hidden_dim: int = 512,
        random_control_init: bool = False,
        memory_variational_dropout: bool = True,
        memory_variational_dropout_amount: float = 0.15,
        control: Optional[ControlUnit] = None,
        read: Optional[ReadUnit] = None,
        write: Optional[WriteUnit] = None,
    ) -> None:
        """Create a `MACCell` instance."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.random_control_init = random_control_init
        self.memory_variational_dropout = memory_variational_dropout
        self.memory_dropout = memory_variational_dropout_amount

        if control is not None and control.control_dim != self.hidden_dim:
            raise ValueError(
                f"MACCell parameter {self.hidden_dim=} and \
                control unit parameter {control.control_dim} must be equal."
            )

        self.control = (
            ControlUnit(control_dim=self.hidden_dim) if control is None else control
        )

        if read is not None and read.memory_dim != self.hidden_dim:
            raise ValueError(
                f"MACCell parameter {self.hidden_dim=} and \
                read unit parameter {read.memory_dim} must be equal."
            )

        if (
            read is not None
            and read.variational_dropout != self.memory_variational_dropout
        ):
            raise ValueError(
                f"MACCell parameter {self.memory_variational_dropout=} and \
                read unit parameter {read.variational_droput} must be equal."
            )

        self.read = (
            ReadUnit(
                memory_dim=self.hidden_dim,
                variational_dropout=self.memory_variational_dropout,
            )
            if read is None
            else read
        )

        if write is not None and write.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"MACCell parameter {self.hidden_dim=} and \
                write unit parameter {write.hidden_dim} must be equal."
            )

        self.write = WriteUnit(hidden_dim=self.hidden_dim) if write is None else write

        self.mem_0 = nn.Parameter(torch.zeros(1, self.hidden_dim))
        # control0 is most often question, other times (eg. args2.txt) it's a
        # learned parameter, initialized as random normal
        if random_control_init:
            self.control_0 = nn.Parameter(torch.zeros(1, self.hidden_dim))

    @staticmethod
    def get_mask(data: torch.Tensor, dropout: float) -> torch.Tensor:
        """Get a dropout mask from a tensor."""
        mask = torch.empty_like(data).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)
        return mask

    def init_hidden(
        self, batch_size: int, question: torch.Tensor
    ) -> Tuple[Tuple[List[torch.Tensor], List[torch.Tensor]], Optional[torch.Tensor]]:
        """Initialise the cell hidden state."""
        if not self.random_control_init:
            control = question
        else:
            control = self.control_0.expand(batch_size, self.hidden_dim)
        memory = self.mem_0.expand(batch_size, self.hidden_dim)
        if self.training and self.memory_variational_dropout:
            memory_mask = self.get_mask(memory, self.memory_dropout)
        else:
            memory_mask = None

        controls = [control]
        memories = [memory]

        return (controls, memories), (memory_mask)

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        state: Sequence[torch.Tensor],
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate data through the model."""
        words, question, img = inputs
        controls, memories = state

        control = self.control(words, question, controls)
        controls.append(control)

        read = self.read(memories, img, controls, masks)
        # if config.writeDropout < 1.0:     dropouts["write"]
        memory = self.write(memories, read, controls)
        memories.append(memory)

        return controls, memories


class MACNetwork(AbstractReasoningModule):
    """A wrapper for a sequence of MAC cells that handles propagation and output."""

    def __init__(
        self,
        length: int = 12,
        hidden_dim: int = 512,
        memory_gate: bool = True,
        memory_gate_bias: float = 1.0,
        classifier: Optional[OutputUnit] = None,
        cell: Optional[MACCell] = None,
    ):
        """Initialise a `RecurrentWrapper` intsance."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.length = length
        self.memory_gate = memory_gate
        self.memory_gate_bias = memory_gate_bias

        if cell is not None and cell.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"MACNetwork parameter {self.hidden_dim=} and \
                cell parameter {cell.hidden_dim} must be equal."
            )

        if classifier is not None and classifier.input_dim != self.hidden_dim:
            raise ValueError(
                f"MACNetwork parameter {self.hidden_dim=} and \
                classifier parameter {classifier.input_dim} must be equal."
            )

        if cell is not None and len(cell.control.position_aware) != self.length:
            raise ValueError(
                f"MACNetwork parameter {self.length_dim=} must be equal to the \
                length of cell control unit parameter {cell.control.position_aware}."
            )

        self.cell = MACCell(hidden_dim=self.hidden_dim) if cell is None else cell
        self.classifier = (
            OutputUnit(input_dim=self.hidden_dim) if classifier is None else classifier
        )

        self.gate = xavier_uniform_linear(self.hidden_dim, 1)

    def forward(
        self, words: torch.Tensor, question: torch.Tensor, knowledge: torch.Tensor
    ) -> torch.Tensor:
        """Propagate data through the module.

        Params:
        -------
        `words`: Tensor of size `(batch_size, max_question_length, output_dim)`,
        traditionally the output of the last LSTM/GRU layer at each timestep.
        `question`: Tensor of size  `(batch_size, hidden_dim)`, traditionally the
        last hidden state of a LSTM or GRU model.
        `knowledge`: Tensor of size `(batch_size, max_object_count,
        object_feature_dim)`, typically a tensor of object features from
        Faster-RCNN, spatial features from a CNN (where an `object` is
        interpreted as a spatial region) or scene graph object features.

        Returns:
        --------
        `predictions`: Tensor of size `(batch_size, num_answers)`, activations
        across all possible answer classes.
        """
        state, masks = self.cell.init_hidden(question.size(0), question)

        for _ in range(1, self.length + 1):
            state = self.cell((words, question, knowledge), state, masks)

            # memory gate
            if self.memory_gate:
                controls, memories = state
                gate = torch.sigmoid(self.gate(controls[-1]) + self.memory_gate_bias)
                memories[-1] = gate * memories[-2] + (1 - gate) * memories[-1]

        _, memories = state

        out = self.classifier(memories[-1], question)

        return out


class OriginalMACNetwork(nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """Implementation of a MAC network, including question and image stem modules."""

    def __init__(
        self,
        hidden_dim: int = 512,
        length: int = 12,
        vocab_size: int = 90,
        embedding_dim: int = 300,
        cnn_dropout: float = 0.18,
        bilstm_dropout: float = 0.08,
    ):
        """Initialise a `MACNetwork` instance."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.length = length

        self.conv = nn.Sequential(
            nn.Dropout(p=cnn_dropout),  # stem dropout stemDropout=0.82
            nn.Conv2d(1024, self.hidden_dim, 3, padding=1),
            nn.ELU(),
            nn.Dropout(p=cnn_dropout),  # stem dropout stemDropout=0.82
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.ELU(),
        )
        self.question_dropout = nn.Dropout(bilstm_dropout)
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # if bi: (bidirectional)
        hidden_dim = int(self.hidden_dim / 2)
        self.lstm = nn.LSTM(
            self.embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # choose different wrappers for no-act/actSmooth/actBaseline
        # TODO config passthrough
        self.actmac = MACNetwork(hidden_dim=self.hidden_dim, length=self.length)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the network's parameters."""
        # From original implementation
        xavier_uniform_(self.embed.weight)
        xavier_uniform_(self.conv[1].weight)
        self.conv[1].bias.data.zero_()
        xavier_uniform_(self.conv[4].weight)
        self.conv[4].bias.data.zero_()

    def forward(
        self,
        image: torch.Tensor,
        question: torch.Tensor,
        question_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate data through the model."""
        batch_size = question.size(0)

        img = self.conv(image)
        img = img.view(batch_size, self.hidden_dim, -1)
        img = img.permute(0, 2, 1)

        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(
            embed, question_lengths, batch_first=True
        )
        lstm_out, (h_n, _) = self.lstm(embed)
        question = torch.cat([h_n[0], h_n[1]], -1)
        question = self.question_dropout(question)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        h_n = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)
        out = self.actmac(lstm_out, question, img)

        return out
