"""A RNN question-processing module."""
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


class RNNQuestionModule(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """Implementation of a RNN question-processing module."""

    def __init__(self, rnn: torch.nn.RNNBase) -> None:
        """Initialise a `RNNQuestionModule` instance."""
        super().__init__()
        self.rnn = rnn

    def forward(self, question_graph: Batch) -> torch.Tensor:
        """Propagate data through the module.

        Params:
        -------
        `question_graph`: A batch of data containing edge indices in COO format that
        specify dependencies between question words, as well as word embeddings
        for each word in the sentence.

        Returns:
        --------
        `words`: Tensor of size `(batch_size, max_question_length, output_dim)`,
        a set of processed embeddings of size `output_dim` for each word in the
        question, for each question in the batch.
        `question`: Tensor of size  `(batch_size, hidden_dim)`, an embedding of
        the question as a whole.
        """
        # Get dense word features for BiLSTM input
        dense_text_feats, question_lengths = to_dense_batch(
            question_graph.x, batch=question_graph.batch
        )
        question_lengths = torch.sum(question_lengths, dim=1)
        batch_size = dense_text_feats.size(0)
        packed_text_feats = torch.nn.utils.rnn.pack_padded_sequence(
            dense_text_feats,
            question_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        words, (h_n, _) = self.rnn(packed_text_feats)
        question_graph = (
            torch.cat([h_n[0], h_n[1]], -1) if self.rnn.bidirectional else h_n[0]
        )
        words, _ = torch.nn.utils.rnn.pad_packed_sequence(words, batch_first=True)
        # TODO verify if this is necessary
        h_n = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)
        return words, question_graph
