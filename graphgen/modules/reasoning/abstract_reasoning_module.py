"""Abstract classes for VQA reasoning."""
from abc import ABC, abstractmethod

import torch


class AbstractReasoningModule(torch.nn.Module, ABC):  # type: ignore
    """Abstract base class for all VQA reasoning modules."""

    @abstractmethod
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
