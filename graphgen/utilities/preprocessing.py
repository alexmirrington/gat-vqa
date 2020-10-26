"""Module containing utilities for preprocessing data."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import stanza
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from ..config.model import SceneGraphAggregationName, SceneGraphConfig
from ..schemas.common import (
    Question,
    SceneGraph,
    TrainableQuestion,
    TrainableSceneGraph,
)
from ..schemas.gqa import GQAQuestion, GQASceneGraph, GQASceneGraphObject
from .generators import slice_sequence


class QuestionPreprocessor:
    """Abstract base class for all question preprocessors."""

    def __init__(
        self,
        answer_to_index: Optional[Dict[str, int]] = None,
        word_to_index: Optional[Dict[str, int]] = None,
    ) -> None:
        """Create a `QuestionPreprocessor` instance."""
        self._answer_to_index = answer_to_index if answer_to_index is not None else {}
        self._index_to_answer = (
            list(answer_to_index.keys()) if answer_to_index is not None else []
        )
        self._word_to_index = word_to_index if word_to_index is not None else {}
        self._index_to_word = (
            list(word_to_index.keys()) if word_to_index is not None else []
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

    @property
    def index_to_word(self) -> List[str]:
        """Get the `int` to `str` mapping of indices to question words, based \
        on the data that has been processed so far."""
        return self._index_to_word.copy()

    @index_to_word.setter
    def index_to_word(self, value: List[str]) -> None:
        """Set the int to str mapping of indices to question words."""
        self._index_to_word = value
        self._word_to_index = {key: idx for idx, key in enumerate(value)}

    def __call__(self, data: Sequence[Any]) -> List[Question]:
        """Preprocess a question sample."""
        raise NotImplementedError()


class SceneGraphPreprocessor:
    """Abstract base class for all scene graph preprocessors."""

    def __init__(
        self,
        object_to_index: Optional[Dict[str, int]] = None,
        attr_to_index: Optional[Dict[str, int]] = None,
        rel_to_index: Optional[Dict[str, int]] = None,
    ) -> None:
        """Create a `SceneGraphPreprocessor` instance."""
        self._object_to_index = object_to_index if object_to_index is not None else {}
        self._attr_to_index = attr_to_index if attr_to_index is not None else {}
        self._rel_to_index = rel_to_index if rel_to_index is not None else {}

    @property
    def object_to_index(self) -> Dict[str, int]:
        """Get the `int` to `str` mapping of indices to object classes, based \
        on the data that has been processed so far."""
        return self._object_to_index.copy()

    @object_to_index.setter
    def object_to_index(self, value: Dict[str, int]) -> None:
        """Set the int to str mapping of indices to objects."""
        self._object_to_index = value

    @property
    def attr_to_index(self) -> Dict[str, int]:
        """Get the `int` to `str` mapping of indices to attribute classes, based \
        on the data that has been processed so far."""
        return self._attr_to_index.copy()

    @attr_to_index.setter
    def attr_to_index(self, value: Dict[str, int]) -> None:
        """Set the int to str mapping of indices to attributes."""
        self._attr_to_index = value

    @property
    def rel_to_index(self) -> Dict[str, int]:
        """Get the `int` to `str` mapping of indices to relation classes, based \
        on the data that has been processed so far."""
        return self._rel_to_index.copy()

    @rel_to_index.setter
    def rel_to_index(self, value: Dict[str, int]) -> None:
        """Set the int to str mapping of indices to relations."""
        self._rel_to_index = value

    def __call__(self, data: Sequence[Any]) -> List[SceneGraph]:
        """Preprocess a scene graph sample."""
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


class GQAQuestionPreprocessor(QuestionPreprocessor):
    """Class for preprocessing GQA questions."""

    def __init__(
        self,
        answer_to_index: Optional[Dict[str, int]] = None,
        word_to_index: Optional[Dict[str, int]] = None,
    ) -> None:
        """Create a `GQAQuestionPreprocessor` instance."""
        super().__init__(answer_to_index, word_to_index)
        self._question_pipeline = stanza.Pipeline(
            lang="en",
            processors={
                "tokenize": "default",
                "pos": "default",
                "lemma": "default",
                "depparse": "default",
            },
            verbose=False,
            dir=".stanza",
        )

    def _process_questions(
        self, questions: List[str]
    ) -> Tuple[List[List[int]], List[List[List[int]]]]:
        qdoc = self._question_pipeline("\n\n".join(questions))
        tokens: List[List[int]] = []
        for sent in qdoc.sentences:
            tokens.append([])
            for word in sent.words:
                if word.text not in self._index_to_word:
                    self._word_to_index[word.text] = len(self._word_to_index)
                    self._index_to_word.append(word.text)
                tokens[-1].append(self._word_to_index[word.text])
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


class GQASceneGraphPreprocessor(SceneGraphPreprocessor):
    """Class for preprocessing GQA scene graphs."""

    def _process_objects(  # pylint: disable=no-self-use
        self, objects: List[Dict[str, GQASceneGraphObject]]
    ) -> Tuple[
        List[List[Tuple[int, int, int, int]]],
        List[List[str]],
        List[List[List[str]]],
        List[Tuple[List[int], List[int]]],
        List[List[str]],
    ]:
        boxes: List[List[Tuple[int, int, int, int]]] = []
        labels: List[List[str]] = []
        attrs: List[List[List[str]]] = []
        coos: List[Tuple[List[int], List[int]]] = []
        relations: List[List[str]] = []

        for obj_dict in objects:
            labels.append([])
            boxes.append([])
            attrs.append([])
            coos.append(([], []))
            relations.append([])
            obj_key_to_idx = {key: idx for idx, key in enumerate(obj_dict.keys())}
            for obj_key, obj_data in obj_dict.items():
                name = obj_data["name"]
                box = (
                    obj_data["x"],
                    obj_data["y"],
                    obj_data["x"] + obj_data["w"],
                    obj_data["y"] + obj_data["h"],
                )
                labels[-1].append(name)
                if name not in self._object_to_index:
                    self._object_to_index[name] = len(self._object_to_index)
                boxes[-1].append(box)
                attrs[-1].append(obj_data["attributes"])
                for attr in obj_data["attributes"]:
                    if attr not in self._attr_to_index:
                        self._attr_to_index[attr] = len(self._attr_to_index)
                # Populate relation indices
                for relation in obj_data["relations"]:
                    coos[-1][0].append(obj_key_to_idx[obj_key])
                    coos[-1][1].append(obj_key_to_idx[relation["object"]])
                    relations[-1].append(relation["name"])
                    if relation["name"] not in self._rel_to_index:
                        self._rel_to_index[relation["name"]] = len(self._rel_to_index)
        return boxes, labels, attrs, coos, relations

    def __call__(self, data: Sequence[GQASceneGraph]) -> List[SceneGraph]:
        """Preprocess a scene graph sample."""
        result: List[SceneGraph] = []
        step = 1024
        start = 0
        for questions in tqdm(
            slice_sequence(data, step),
            total=len(data) // step + 1,
            desc="preprocessing: ",
        ):
            boxes, labels, attrs, coos, rels = self._process_objects(
                [d["objects"] for d in data]
            )
            result += [
                {
                    "imageId": scene["imageId"],
                    "boxes": boxes_,
                    "labels": labels_,
                    "attributes": attrs_,
                    "relations": rels_,
                    "coos": coos_,
                    "indexed_labels": [self._object_to_index[lbl] for lbl in labels_],
                    "indexed_attributes": [
                        [self._attr_to_index[attr] for attr in obj_attrs]
                        for obj_attrs in attrs_
                    ],
                    "indexed_relations": [self._rel_to_index[rel] for rel in rels_],
                }
                for scene, boxes_, labels_, attrs_, coos_, rels_ in zip(
                    data, boxes, labels, attrs, coos, rels
                )
            ]
            start += step
        return result


class QuestionTransformer:
    """Class for applying transformations to questions."""

    def __call__(self, data: Question) -> TrainableQuestion:
        """Transform data into a trainable format and look up word embeddings."""
        dependencies = torch.tensor(  # pylint: disable=not-callable
            data["dependencies"]
        )
        tokens = torch.tensor(data["tokens"])  # pylint: disable=not-callable
        return {
            "questionId": data["questionId"],
            "imageId": data["imageId"],
            "tokens": tokens,
            "dependencies": Data(edge_index=dependencies, x=tokens),
            "answer": data["answer"],
        }


class SceneGraphTransformer:
    """Class for applying transformations to scene graphs."""

    def __init__(
        self,
        num_objects: int,
        num_relations: int,
        num_attributes: int,
        graph: SceneGraphConfig,
        embeddings: Optional[torch.nn.Embedding],
    ) -> None:
        """Initialise a `SceneGraphTransformer` instance."""
        self.num_objects = num_objects
        self.num_relations = num_relations
        self.num_attributes = num_attributes
        self.graph = graph
        self.embeddings = embeddings

    def build_graph(
        self,
        object_coos: Tuple[List[int], List[int]],
        objects: List[int],
        relations: List[int],
        attributes: List[List[int]],
        add_skip_edges: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a graph containing objects, relation and attribute nodes."""
        # pylint: disable=too-many-locals
        # Add obj->relation, relation->obj and attr->object edges
        sources = object_coos[0] if add_skip_edges else []
        targets = object_coos[1] if add_skip_edges else []
        feats = []

        # Add objects to node features
        if self.embeddings is None:
            feats.append(torch.tensor(objects))  # pylint: disable=not-callable
        else:
            feats.append(
                self.embeddings(
                    torch.tensor(  # pylint:disable=not-callable
                        objects, dtype=torch.long
                    )
                )
            )
        offset_idx = len(objects)

        # Add source->relation and relation->target edges
        for idx in range(len(relations)):
            source_node = object_coos[0][idx]
            target_node = object_coos[1][idx]
            # Treat every relation as unique, i.e. two "on" relations will
            # correspond to two nodes in the final graph. The relation node
            # has index `len(objects) + idx` in the graph.
            sources += [source_node, offset_idx + idx]
            targets += [offset_idx + idx, target_node]

        # Add relations to node features
        if self.embeddings is None:
            feats.append(
                torch.tensor(relations)  # pylint: disable=not-callable
                + self.num_objects
            )
        else:
            feats.append(
                self.embeddings(
                    torch.tensor(  # pylint:disable=not-callable
                        relations, dtype=torch.long
                    )
                    + self.num_objects
                )
            )
        offset_idx += len(relations)

        # Add attr->object edges
        attr_to_idx: Dict[Any, int] = {}
        for obj_idx, obj_attrs in enumerate(attributes):
            for attr in obj_attrs:
                # We don't treat attributes as unique, since all edges are
                # outgoing from attributes, i.e. two "blue" relations will
                # correspond to one node in the final graph, with outgoing
                # edges to all blue objects.
                if attr not in attr_to_idx:
                    attr_to_idx[attr] = len(attr_to_idx)
                sources.append(offset_idx + attr_to_idx[attr])
                targets.append(obj_idx)

        # Add attributes to node features
        # (requires python 3.7+, to assert order of inserted keys in attr_to_idx)
        if self.embeddings is None:
            feats.append(
                torch.tensor(list(attr_to_idx.keys()))  # pylint: disable=not-callable
                + self.num_objects
                + self.num_relations
            )
        else:
            feats.append(
                self.embeddings(
                    torch.tensor(  # pylint:disable=not-callable
                        list(attr_to_idx.keys()), dtype=torch.long
                    )
                    + self.num_objects
                    + self.num_relations
                )
            )
        return (
            torch.tensor(  # pylint:disable=not-callable
                (sources, targets), dtype=torch.long
            ),
            torch.cat(feats, dim=0),
        )

    def build_concat_mean_graph(
        self,
        object_coos: Tuple[List[int], List[int]],
        objects: List[int],
        relations: List[int],
        attributes: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a knowledgebase of concatenated glove vectors."""
        assert self.embeddings is not None
        sources = object_coos[0]
        targets = object_coos[1]
        feats = torch.tensor([])  # pylint:disable=not-callable
        # Mean pool relations
        if len(objects) > 0:
            feats = self.embeddings(
                torch.tensor(objects, dtype=torch.long)  # pylint:disable=not-callable
            )
            rels: List[List[int]] = [[] for _ in objects]  # relations for each object
            for idx, rel in enumerate(relations):
                source_obj_idx = sources[idx]
                target_obj_idx = targets[idx]
                # TODO flag to determine whether to add source
                rels[source_obj_idx].append(rel)
                rels[target_obj_idx].append(rel)
            rel_feats = torch.stack(
                [
                    torch.mean(
                        self.embeddings(
                            torch.tensor(  # pylint:disable=not-callable
                                obj_rels, dtype=torch.long
                            )
                            + self.num_objects
                        ),
                        axis=0,
                    )
                    if len(obj_rels) > 0
                    else torch.zeros(self.embeddings.embedding_dim)
                    for obj_rels in rels
                ],
                dim=0,
            )
            assert rel_feats.size() == feats.size()
            attr_feats = torch.stack(
                [
                    torch.mean(
                        self.embeddings(
                            torch.tensor(  # pylint:disable=not-callable
                                obj_attrs, dtype=torch.long
                            )
                            + self.num_objects
                            + self.num_relations
                        ),
                        axis=0,
                    )
                    if len(obj_attrs) > 0
                    else torch.zeros(self.embeddings.embedding_dim)
                    for obj_attrs in attributes
                ],
                dim=0,
            )
            assert attr_feats.size() == feats.size()

            return (
                torch.tensor(  # pylint:disable=not-callable
                    (sources, targets), dtype=torch.long
                ),
                torch.cat((feats, rel_feats, attr_feats), dim=1),
            )
        return (
            torch.tensor(  # pylint:disable=not-callable
                (sources, targets), dtype=torch.long
            ),
            feats,
        )

    def __call__(self, data: SceneGraph) -> TrainableSceneGraph:
        """Transform data into a trainable format."""
        objects = data["indexed_labels"]
        relations = data["indexed_relations"]
        attributes = data["indexed_attributes"]
        if self.graph.aggregation is None:
            coos, feats = self.build_graph(
                data["coos"],
                objects,
                relations,
                attributes,
                self.graph.object_skip_edges,
            )
        elif (
            self.graph.aggregation
            == SceneGraphAggregationName.PER_OBJ_CONCAT_MEAN_REL_ATTR
        ):
            coos, feats = self.build_concat_mean_graph(
                data["coos"], objects, relations, attributes
            )
        else:
            raise NotImplementedError()

        return {
            "imageId": data["imageId"],
            "boxes": torch.tensor(  # pylint: disable=not-callable
                data["boxes"], dtype=torch.float
            ),
            "labels": data["labels"],
            "attributes": data["attributes"],
            "relations": data["relations"],
            "graph": Data(edge_index=coos, x=feats),
        }


class ObjectTransformer:
    """Class for applying transformations to rcnn object features."""

    def __init__(self) -> None:
        """Initialise a `ObjectsTransformer` instance."""

    def __call__(
        self, objects: torch.Tensor, boxes: torch.Tensor, meta: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform data into a trainable format."""
        num_objects = meta["objectsNum"]
        return (objects[:num_objects], boxes[:num_objects])


@dataclass
class PreprocessorCollection:
    """Wrapper class for storing a preprocessor feature mappings."""

    questions: QuestionPreprocessor
    scene_graphs: SceneGraphPreprocessor


@dataclass
class DatasetCollection:
    """Wrapper class for storing a train and val dataset tuple."""

    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    images: torch.utils.data.Dataset
