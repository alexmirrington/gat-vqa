"""Module containing utilities for preprocessing data."""
from typing import Dict, List, Tuple

import stanza
import torch
from torch_geometric.data import Data
from torchtext.vocab import GloVe
from tqdm import tqdm

from ..schemas.common import Question, TrainableQuestion
from ..schemas.gqa import GQAQuestion
from .generators import slice_dict


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


class GQAQuestionPreprocessor:
    """Class for preprocessing questions."""

    def __init__(self) -> None:
        """Create a `QuestionPreprocessor` instance."""
        self._answer_to_class: Dict[str, int] = {}
        self._word_to_index: Dict[str, int] = {}
        self._question_pipeline = stanza.Pipeline(
            lang="en",
            processors={
                "tokenize": "default",
                "pos": "default",
                "lemma": "default",
                "depparse": "default",
            },
        )

    def _process_questions(
        self, questions: List[str]
    ) -> Tuple[List[List[str]], List[List[List[int]]]]:
        qdoc = self._question_pipeline("\n\n".join(questions))
        tokens: List[List[str]] = []
        for sent in qdoc.sentences:
            tokens.append([])
            for word in sent.words:
                if word.text not in self._word_to_index.keys():
                    self._word_to_index[word.text] = len(self._word_to_index)
                tokens[-1].append(word.text)
        graphs = GQAQuestionPreprocessor.dep_coordinate_list(qdoc)
        return tokens, graphs

    def _process_answers(self, answers: List[str]) -> List[int]:
        result: List[int] = []
        for answer in answers:
            if answer is not None:
                if answer not in self._answer_to_class:
                    self._answer_to_class[answer] = len(self._answer_to_class)
            result.append(self._answer_to_class[answer] if answer is not None else None)
        return result

    def __call__(self, data: Dict[str, GQAQuestion]) -> Dict[str, Question]:
        """Preprocess a question sample."""
        result: Dict[str, Question] = {}
        step = 1024
        for subset in tqdm(
            slice_dict(data, step), total=len(data) // step + 1, desc="preprocessing: "
        ):
            qids, questions = zip(*subset.items())
            answers = self._process_answers([q.get("answer", None) for q in questions])
            tokens, graphs = self._process_questions([q["question"] for q in questions])

            # If one sample has a multi-sentence question, zip should fail.
            for qid, question, answer, tokens_, graph in zip(
                qids, questions, answers, tokens, graphs
            ):
                result[qid] = {
                    "imageId": question["imageId"],
                    "question": question["question"],
                    "answer": answer,
                    "tokens": tokens_,
                    "dependencies": graph,
                }

        return result

    @staticmethod
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
