"""Implementation of a relation proposal network."""

import math
from typing import Tuple, Union

import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class BiLSTMRelPN(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method
    """Implementation of a BiLSTM-based relation proposal network, for proposing \
    contextual relationships between question words."""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0
    ):
        """Construct a `BiLSTMRelPN` instance."""
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = torch.nn.LSTM(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            dropout=self.dropout,
        )

    def forward(
        self, sentences: Union[PackedSequence, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a set of relation pairs and scores for each sentence."""
        # Get LSTM outputs from last layer for each token
        lstm_out, (h_n, c_n) = self.lstm(sentences)

        # Pad lstm_out if it was a PackedSequence, ready for linear layer
        if isinstance(lstm_out, PackedSequence):
            lstm_out = pad_packed_sequence(lstm_out)[0]
        # lstm_out is of size:
        # (seq_len, batch_size, num_directions * hidden_size)

        # Make batch the first dimension
        lstm_out = torch.transpose(lstm_out, 0, 1)

        # print(f"{lstm_out.size()=}")

        # Compute scaled dot product self-attention over lstm outputs
        alignment = torch.bmm(lstm_out, torch.transpose(lstm_out, 1, 2)) / math.sqrt(
            lstm_out.size(-1)
        )
        # alignment has size (batch_size, seq_len, seq_len)
        # print(f"{alignment.size()=}")

        # Get most K important vectors for each vector, this will become a directed
        # adjacency matrix where each vector is connected to K others. A higher K
        # thus gives a more dense graph. Note self-loops are neither guaranteed
        # nor explicitly removed.
        k = 3
        seq_len = lstm_out.size(1)
        sorted_alignment, sorted_alignment_indices = torch.sort(
            alignment, dim=-1, descending=True
        )
        # print(f"{sorted_alignment.size()=}")
        # print(f"{sorted_alignment_indices.size()=}")

        adj_list = sorted_alignment_indices[:, :, :k]
        # print(f"{adj_list.size()=}")
        # print(f"{adj_list[0]=}")
        # adj_list has size (batch_size, seq_len, k)
        indices = (
            torch.arange(seq_len)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(adj_list.size(0), 1, k)
            .to("cuda")
        )
        # print(f"{indices.size()=}")
        # print(f"{indices[0]=}")

        coo = torch.stack((adj_list.flatten(1, 2), indices.flatten(1, 2)), dim=1)
        # coo has size (batch_size, 2, seq_len * k)
        # print(f"{coo.size()=}")
        # print(f"{coo[0]=}")

        # Convert the selected alignment indices to COO format
        # alignment = F.softmax(alignment, dim=-1)

        # Self attention context vectors
        # context_vectors = torch.bmm(alignment, lstm_out)
        return coo, sorted_alignment[:, :, :k], lstm_out
