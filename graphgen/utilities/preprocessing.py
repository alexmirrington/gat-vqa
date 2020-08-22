"""Module containing utilities for preprocessing data."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import stanza
import torch
from torch_geometric.data import Data
from torchtext.vocab import GloVe
from tqdm import tqdm

from ..schemas.common import Question, TrainableQuestion
from ..schemas.gqa import GQAQuestion
from .generators import slice_sequence


class Preprocessor(ABC):
    """Abstract base class for all preprocessors."""

    @abstractmethod
    def __call__(self, data: Sequence[Any]) -> Sequence[Any]:
        """Preprocess a sequence of data."""
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Determine if this preprocessor is equal to another. Equal \
        preprocessors should have equal outputs given the same data."""
        raise NotImplementedError()


def dep_coordinate_list(
    doc: stanza.Document, directed: bool = False
) -> List[List[List[int]]]:
    """Construct a co-ordinate list (COO) representation of the dependency \
    graph for each sentence in a list of sentences (`doc`).

    Returns:
    --------
    A 3D list of dimensions [len(doc.sentences), 2, n_edges] representing \
    the graph connectivity in COO format for each sentence.
    """
    doc_coords = []
    for sent in doc.sentences:
        outgoing = []  # Nodes with outgoing edges (rows)
        incoming = []  # Nodes with incoming edges (cols)
        # deprels = []
        for word in sent.words:
            # deprels.append(word.deprel)
            curr = word.id - 1  # incoming node, zero indexed
            head = word.head - 1  # outgoing node, zero indexed
            if head >= 0:  # Ignore root token which has index of -1
                outgoing.append(curr)
                incoming.append(head)
                if not directed:
                    outgoing.append(head)
                    incoming.append(curr)
        doc_coords.append([outgoing, incoming])

    return doc_coords


class QuestionTransformer:
    """Class for applying transformations to questions."""

    def __init__(self) -> None:
        """Initialise a `QuestionTransformer` instance."""
        self.vectors = GloVe(name="6B", dim=300)

    def __call__(self, data: Question) -> TrainableQuestion:
        """Transform data into a trainable format and look up word embeddings."""
        adjacency = torch.tensor(data["dependencies"])  # pylint: disable=not-callable
        node_features = self.vectors.get_vecs_by_tokens(
            data["tokens"], lower_case_backup=True
        )
        return {
            "imageId": data["imageId"],
            "dependencies": Data(edge_index=adjacency, x=node_features),
            "answer": data["answer"],
        }


class GQAQuestionPreprocessor(Preprocessor):
    """Class for preprocessing questions."""

    def __init__(
        self, frozen: bool = False, answer_to_index: Optional[Dict[str, int]] = None
    ) -> None:
        """Create a `QuestionPreprocessor` instance."""
        self._answer_to_index = answer_to_index if answer_to_index is not None else {}
        self.frozen = frozen
        self._question_pipeline = stanza.Pipeline(
            lang="en",
            processors={
                "tokenize": "default",
                "pos": "default",
                "lemma": "default",
                "depparse": "default",
            },
            verbose=False,
        )

    @property
    def answer_to_index(self) -> Dict[str, int]:
        """Get the string to int mapping of answers to indices, based on the \
        data that has been processed so far."""
        return self._answer_to_index.copy()

    @answer_to_index.setter
    def answer_to_index(self, value: Dict[str, int]) -> None:
        self._answer_to_index = value

    def _process_questions(
        self, questions: List[str]
    ) -> Tuple[List[List[str]], List[List[List[int]]]]:
        qdoc = self._question_pipeline("\n\n".join(questions))
        tokens = [[word.text for word in sent.words] for sent in qdoc.sentences]
        graphs = dep_coordinate_list(qdoc)
        return tokens, graphs

    def _process_answers(self, answers: List[str]) -> List[int]:
        result: List[int] = []
        for answer in answers:
            if (
                answer is not None
                and answer not in self._answer_to_index
                and not self.frozen
            ):
                self._answer_to_index[answer] = len(self._answer_to_index)
            result.append(self._answer_to_index[answer] if answer is not None else None)
        return result

    def __call__(self, data: Sequence[GQAQuestion]) -> List[Question]:
        """Preprocess a question sample."""
        result: List[Question] = []
        step = 1024
        start = 0
        for questions in tqdm(
            slice_sequence(data, step),
            total=len(data) // step + 1,
            desc="preprocessing: ",
        ):
            answers = self._process_answers([q.get("answer", None) for q in questions])
            tokens, graphs = self._process_questions([q["question"] for q in questions])

            # If one sample has a multi-sentence question, zip should fail.
            result += [
                {
                    "imageId": question["imageId"],
                    "question": question["question"],
                    "answer": answer,
                    "tokens": tokens_,
                    "dependencies": graph,
                }
                for question, answer, tokens_, graph in zip(
                    questions, answers, tokens, graphs
                )
            ]
            start += step
        return result

    def __eq__(self, other: Any) -> bool:
        """Determine if this preprocessor is equal to another. Equal \
        preprocessors should have equal outputs given the same data."""
        if not isinstance(other, self.__class__):
            return False
        return (
            self.frozen == other.frozen
            and self.answer_to_index == other.answer_to_index
        )
