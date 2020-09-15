"""Implementation of Bottom-Up VQA model by Anderson et. al.

References:
-----------
https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Bottom-Up_and_\
Top-Down_CVPR_2018_paper.html
"""
import torch
import torch.nn.functional as F
from torch.nn.init import calculate_gain, xavier_uniform_

from ..abstract_reasoning_module import AbstractReasoningModule
from .gated_tanh import GatedTanh


class BottomUp(AbstractReasoningModule):
    """Bottom-up attention VQA model."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        question_dim: int = 512,
        knowledge_dim: int = 900,
        hidden_dim: int = 512,
        output_dim: int = 1843,
    ) -> None:
        """Create a bottom-up attention VQA model."""
        super().__init__()
        self.question_dim = question_dim
        self.knowledge_dim = knowledge_dim
        self.hidden_dim = hidden_dim

        self.question_knowledge_attn_weights = torch.nn.Parameter(
            torch.empty(hidden_dim)
        )
        self.question_knowledge_attn_gate = GatedTanh(
            question_dim + knowledge_dim, hidden_dim
        )
        self.question_proj = GatedTanh(question_dim, hidden_dim)
        self.knowledge_proj = GatedTanh(knowledge_dim, hidden_dim)
        self.classifier_0 = GatedTanh(hidden_dim, hidden_dim)
        self.classifier_1 = torch.nn.Parameter(torch.empty(hidden_dim, output_dim))

    def reset_parameters(self) -> None:
        """Reset the module's parameters."""
        xavier_uniform_(
            self.question_knowledge_attn_weights,
            gain=calculate_gain("tanh"),
        )
        xavier_uniform_(self.classifier_1, gain=calculate_gain("tanh"))

    def forward(
        self, words: torch.Tensor, question: torch.Tensor, knowledge: torch.Tensor
    ) -> torch.Tensor:
        """Propagate data through the model.

        Params:
        -------
        `words`: Tensor of size `(batch_size, max_question_length, output_dim)`,
        traditionally the output of the last LSTM/GRU layer at each timestep.
        `question`: question tensor of shape (batch_size, question_dim),
        typically the output of an LSTM or GRU.
        `knowledge`: knowledge-base tensor of shape
        (batch_size, knowledge_feat_count, knowledge_dim)
        """
        expanded_question = question.unsqueeze(1).expand(
            question.size(0), knowledge.size(1), question.size(1)
        )  # (batch_size, knowledge_feat_count, question_dim)
        attn = self.question_knowledge_attn_gate(
            torch.cat((expanded_question, knowledge), dim=2)
        )  # (batch_size, knowledge_feat_count, hidden_dim)
        attn = torch.matmul(
            attn, self.question_knowledge_attn_weights
        )  # (batch_size, knowledge_feat_count)
        attn = F.softmax(attn, dim=1)  # (batch_size, knowledge_feat_count)
        attended_knowledge = torch.bmm(
            attn.unsqueeze(1), knowledge
        ).squeeze()  # (batch_size, knowledge_dim)
        projected_question = self.question_proj(question)  # (batch_size, hidden_dim)
        projected_knowledge = self.question_proj(
            attended_knowledge
        )  # (batch_size, hidden_dim)
        joint_embedding = (
            projected_question * projected_knowledge
        )  # (batch_size, hidden_dim)
        intermediate_out = self.classifier_0(
            joint_embedding
        )  # (batch_size, hidden_dim)
        out = torch.mm(intermediate_out, self.classifier_1)
        return out  # TODO BCE loss. Not strictly necessary for GQA but not bad practice
