"""Module containing utilities for preprocessing data."""
from time import perf_counter
from typing import Any, Dict, List

import stanza


class QuestionPreprocessor:
    """Class for preprocessing questions."""

    KEY_MASK = ("imageId", "question", "answer")
    VOCAB_MASK = ("question", "answer")

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

    @staticmethod
    def dep_coordinate_list(
        doc: stanza.Document, directed: bool = False
    ) -> List[List[List[int]]]:
        """Construct a co-ordinate list (COO) representation of the dependency \
        graph for each sentence in a list of sentences (`doc`).

        Returns:
        --------
        A 3D list of dimensions [len(doc.sentences), 2, n_edges] representing \
        the graph connectivity in COO format for each sentence
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
                if head >= 0:  # Ignore root token (indexed -1)
                    outgoing.append(head)
                    incoming.append(curr)
                    if not directed:
                        outgoing.append(curr)
                        incoming.append(head)
            doc_coords.append([outgoing, incoming])

        return doc_coords

    def __call__(self, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a question sample."""
        result = {}
        batch_size = 8192
        batch_start = 0
        while batch_start < len(questions):
            start = perf_counter()
            qids, qns = zip(
                *(
                    list(questions.items())[
                        batch_start : min(batch_start + batch_size, len(questions))
                    ]
                )
            )
            doc = self._pipeline("\n\n".join([q["question"] for q in qns]))
            graphs = QuestionPreprocessor.dep_coordinate_list(doc)
            for qid, question, processed, graph in zip(
                qids, qns, doc.sentences, graphs
            ):  # If there are multi-sentence questions, zip will fail.
                result[qid] = {
                    key: val for key, val in question.items() if key in self.KEY_MASK
                }
                result[qid]["question"] = {
                    "raw": result[qid]["question"],
                    "tokens": [word.text for word in processed.words],
                    "dependencies": graph,
                }
            batch_start += batch_size
            end = perf_counter()
            print(
                (
                    f"[{batch_start}/{len(questions)}]",
                    f"({batch_size/(end - start):.2f} it/s)",
                ),
                end="\r",
            )

        # Populate word_to_index dict
        # for key, val in result.items():
        #     if key in self.VOCAB_MASK:
        #         lval = val.lower()
        #         lval = re.sub(r"[^\w\s]", "", lval)
        #         tokens = []
        #         for word in lval.split():
        #             if word not in self.word_to_index.keys():
        #                 self.word_to_index[word] = len(self.word_to_index)
        #             tokens.append(self.word_to_index[word])
        #         result[key] = tokens

        return result
