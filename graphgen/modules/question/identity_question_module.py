"""An identity question module with no trainable parameters."""
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


class IdentityQuestionModule(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """Implementation of an identity question module with no trainable parameters."""

    def forward(  # pylint: disable=no-self-use
        self, question_graph: Batch
    ) -> torch.Tensor:
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
        the question as a whole. For the `IdentityQuestionModule`, we take the
        mean of all words in the question, similarly to the `GCNQuestionModule`.
        """
        # Get dense word features for BiLSTM input
        dense_text_feats, _ = to_dense_batch(
            question_graph.x, batch=question_graph.batch
        )
        words, question = dense_text_feats, torch.mean(dense_text_feats, dim=1)
        return words, question
