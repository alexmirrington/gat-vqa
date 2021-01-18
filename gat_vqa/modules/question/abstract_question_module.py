"""Abstract classes for all question-processing modules."""
from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Batch


class AbstractQuestionModule(torch.nn.Module, ABC):  # type: ignore
    """Abstract base class for all question-processing modules."""

    @abstractmethod
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
