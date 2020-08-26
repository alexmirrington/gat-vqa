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


class QuestionPreprocessor(Preprocessor):
    """Abstract base class for all question preprocessors."""

    @property
    @abstractmethod
    def index_to_answer(self) -> List[str]:
        """Get the `int` to `str` mapping of indices to answers, based on the \
        data that has been processed so far."""

    @abstractmethod
    def __call__(self, data: Sequence[Any]) -> Sequence[Question]:
        """Preprocess a sequence of data."""


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


class GQAQuestionPreprocessor(QuestionPreprocessor):
    """Class for preprocessing questions."""

    def __init__(self, answer_to_index: Optional[Dict[str, int]] = None) -> None:
        """Create a `QuestionPreprocessor` instance."""
        self._answer_to_index = answer_to_index if answer_to_index is not None else {}
        self._index_to_answer = (
            list(answer_to_index.keys()) if answer_to_index is not None else []
        )
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
    def index_to_answer(self) -> List[str]:
        """Get the `int` to `str` mapping of indices to answers, based on the \
        data that has been processed so far."""
        return self._index_to_answer.copy()

    @index_to_answer.setter
    def index_to_answer(self, value: List[str]) -> None:
        """Set the int to str mapping of indices to answers."""
        self._index_to_answer = value
        self._answer_to_index = {key: idx for idx, key in enumerate(value)}

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
            if answer is not None and answer not in self._answer_to_index:
                # Unknown vocab, add to dict. It is OK to add val and test
                # answers to the dict, as we still have no training signal for
                # those in the training set, hence there is no reason to freeze
                # the answer vocab.
                self._answer_to_index[answer] = len(self._answer_to_index)
                self._index_to_answer.append(answer)
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
                    "questionId": question["questionId"],
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
            "questionId": data["questionId"],
            "imageId": data["imageId"],
            "dependencies": Data(edge_index=adjacency, x=node_features),
            "answer": data["answer"],
        }
