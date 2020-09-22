"""A CNN question-processing module.

References
----------
https://www.aclweb.org/anthology/D14-1181/
"""
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


class CNNQuestionModule(torch.nn.Module):  # type: ignore  # pylint: disable=abstract-method  # noqa: B950
    """Implementation of a CNN question-processing module.

    References
    ----------
    https://www.aclweb.org/anthology/D14-1181/
    """

    def __init__(self, input_dim: int = 300, out_channels: int = 512) -> None:
        """Initialise a `CNNQuestionModule` instance."""
        super().__init__()
        # h = max_question_length, w = output_dim
        # kernels convolve over all dimensions of each word, and either 3, 4 or 5
        # sentence words each is in https://www.aclweb.org/anthology/D14-1181/
        self.convs = torch.nn.ModuleList(
            [  # TODO out_channels division remainders
                torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=out_channels // 4,
                    kernel_size=(3, input_dim),
                    padding=(1, 0),
                ),  # outputs size (n, 100, n_words, 1)  # TODO experiment with zero pad  # noqa: B950
                torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=out_channels // 4,
                    kernel_size=(5, input_dim),
                    padding=(2, 0),
                ),  # output size (n, 100, n_words, 1)    # TODO experiment with zero pad  # noqa: B950
                torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=out_channels // 4,
                    kernel_size=(7, input_dim),
                    padding=(3, 0),
                ),  # output size (n, 100, n_words, 1)    # TODO experiment with zero pad  # noqa: B950
                torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=out_channels // 4,
                    kernel_size=(9, input_dim),
                    padding=(4, 0),
                ),  # outputs size (n, 100, n_words, 1)  # TODO experiment with zero pad  # noqa: B950
            ]
        )

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
        # Get dense word features for cnn input
        dense_text_feats, question_lengths = to_dense_batch(
            question_graph.x, batch=question_graph.batch
        )  # dense_text_feats: (batch_size, max_question_length, output_dim)
        question_lengths = torch.sum(question_lengths, dim=1)
        dense_text_feats = dense_text_feats.unsqueeze(1)
        pooled_feats = []
        word_feats = []
        for conv in self.convs:
            convolved_feats = conv(dense_text_feats)  # (n, out_channels, n_words, 1)
            convolved_feats = convolved_feats.squeeze()  # (n, out_channels, n_words)
            word_feats.append(
                convolved_feats.transpose(1, 2)
            )  # (n, n_words, out_channels)
            pooled, _ = torch.max(convolved_feats, dim=-1)  # (n, out_channels)
            pooled_feats.append(pooled)
        question = torch.cat(pooled_feats, dim=-1)
        words = torch.cat(word_feats, dim=-1)
        return words, question
