"""Module containing utilities for preprocessing data."""
from typing import Dict, List

import stanza
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from ..schemas.common import Question, TrainableQuestion
from ..schemas.gqa import GQAQuestion
from .generators import slice_dict


def custom_transform(data: Question) -> TrainableQuestion:
    """Transform data into data for torch-geometric."""
    adjacency = torch.tensor(data["dependencies"])  # pylint: disable=not-callable
    node_features = torch.tensor(data["tokens"])  # pylint: disable=not-callable
    return {
        "imageId": data["imageId"],
        "dependencies": Data(edge_index=adjacency, x=node_features.unsqueeze(-1)),
    }


class GQAQuestionPreprocessor:
    """Class for preprocessing questions."""

    def __init__(self) -> None:
        """Create a `QuestionPreprocessor` instance."""
        self.word_to_index: Dict[str, int] = {}
        self._pipeline = stanza.Pipeline(
            lang="en",
            processors={
                "tokenize": "default",
                "pos": "default",
                "lemma": "default",
                "depparse": "default",
            },
        )

    def __call__(self, data: Dict[str, GQAQuestion]) -> Dict[str, Question]:
        """Preprocess a question sample."""
        result: Dict[str, Question] = {}
        step = 1024
        for subset in tqdm(slice_dict(data, step), total=len(data) // step + 1):
            qids, questions = zip(*subset.items())
            doc = self._pipeline("\n\n".join([q["question"] for q in questions]))
            tokens: List[List[int]] = []
            for sent in doc.sentences:
                tokens.append([])
                for word in sent.words:
                    if word.text not in self.word_to_index.keys():
                        self.word_to_index[word.text] = len(self.word_to_index)
                    tokens[-1].append(self.word_to_index[word.text])
            graphs = GQAQuestionPreprocessor.dep_coordinate_list(doc)
            # If there are multi-sentence questions, zip will fail.
            for qid, question, qtokens, graph in zip(qids, questions, tokens, graphs):
                result[qid] = {
                    "imageId": question["imageId"],
                    "question": question["question"],
                    "tokens": qtokens,
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
