"""A GCN question-processing module."""
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from ..sparse import AbstractGCN


class GCNQuestionModule(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """Implementation of a GCN question-processing module."""

    def __init__(self, gcn: AbstractGCN) -> None:
        """Initialise a `GCNQuestionModule` instance."""
        super().__init__()
        self.gcn = gcn

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
        `question`: Tensor of size `(batch_size, hidden_dim)`, an embedding of
        the question as a whole.
        """
        word_feats, pooled_feats = self.gcn(question_graph)
        dense_word_feats, question_lengths = to_dense_batch(
            word_feats, batch=question_graph.batch
        )
        question_lengths = torch.sum(question_lengths, dim=1)
        # TODO verify if this is necessary: pack and pad again for MAC compatibility
        packed_word_feats = torch.nn.utils.rnn.pack_padded_sequence(
            dense_word_feats,
            question_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        words, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_word_feats, batch_first=True
        )
        question = pooled_feats
        return words, question
