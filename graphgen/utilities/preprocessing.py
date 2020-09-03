"""Module containing utilities for preprocessing data."""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import stanza
import torch
from torch_geometric.data import Data
from torchtext.vocab import GloVe
from tqdm import tqdm

from ..schemas.common import (
    Question,
    SceneGraph,
    TrainableQuestion,
    TrainableSceneGraph,
)
from ..schemas.gqa import GQAQuestion, GQASceneGraph, GQASceneGraphObject
from .generators import slice_sequence

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class QuestionPreprocessor:
    """Abstract base class for all question preprocessors."""

    def __init__(self, answer_to_index: Optional[Dict[str, int]] = None) -> None:
        """Create a `QuestionPreprocessor` instance."""
        self._answer_to_index = answer_to_index if answer_to_index is not None else {}
        self._index_to_answer = (
            list(answer_to_index.keys()) if answer_to_index is not None else []
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

    def __call__(self, data: Sequence[Any]) -> List[Question]:
        """Preprocess a question sample."""
        raise NotImplementedError()


class SceneGraphPreprocessor:
    """Abstract base class for all scene graph preprocessors."""

    def __init__(self, object_to_index: Optional[Dict[str, int]] = None) -> None:
        """Create a `SceneGraphPreprocessor` instance."""
        self._object_to_index = object_to_index if object_to_index is not None else {}

    @property
    def object_to_index(self) -> Dict[str, int]:
        """Get the `int` to `str` mapping of indices to object classes, based \
        on the data that has been processed so far."""
        return self._object_to_index.copy()

    @object_to_index.setter
    def object_to_index(self, value: Dict[str, int]) -> None:
        """Set the int to str mapping of indices to objects."""
        self._object_to_index = value

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

    def __init__(self, answer_to_index: Optional[Dict[str, int]] = None) -> None:
        """Create a `GQAQuestionPreprocessor` instance."""
        super().__init__(answer_to_index)
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


class GQASceneGraphPreprocessor(SceneGraphPreprocessor):
    """Class for preprocessing GQA scene graphs."""

    def __init__(
        self,
        object_to_index: Optional[Dict[str, int]] = None,
        use_coco_classes: bool = False,
    ) -> None:
        """Create a `GQASceneGraphPreprocessor` instance."""
        super().__init__(object_to_index)

        self._coco_vectors: Optional[torch.Tensor] = None

        self._glove = GloVe(name="6B", dim=300)

        if use_coco_classes:
            self._coco_vectors = torch.stack(
                [self._embed_word(word) for word in COCO_INSTANCE_CATEGORY_NAMES]
            )
            self._similarity = torch.nn.CosineSimilarity(dim=1)

    def _embed_word(self, word: str) -> torch.Tensor:
        return torch.sum(
            self._glove.get_vecs_by_tokens(word.split(), lower_case_backup=True), dim=0
        )

    def _process_objects(
        self, objects: List[Dict[str, GQASceneGraphObject]]
    ) -> Tuple[List[List[Tuple[int, int, int, int]]], List[List[int]]]:
        boxes: List[List[Tuple[int, int, int, int]]] = []
        labels: List[List[int]] = []
        for obj_dict in objects:
            labels.append([])
            boxes.append([])
            for obj_data in obj_dict.values():
                name = obj_data["name"]
                box = (
                    obj_data["x"],
                    obj_data["y"],
                    obj_data["x"] + obj_data["w"],
                    obj_data["y"] + obj_data["h"],
                )
                if obj_data["w"] > 0 and obj_data["h"] > 0:
                    if name is not None and name not in self._object_to_index:
                        if (
                            self._coco_vectors is not None
                            and self._similarity is not None
                        ):
                            # Map word to COCO class via cosine similarity
                            name_embedding = (
                                self._embed_word(name)
                                .unsqueeze(0)
                                .repeat(self._coco_vectors.size(0), 1)
                                .unsqueeze(-1)
                            )
                            similarities = self._similarity(
                                self._coco_vectors.unsqueeze(-1), name_embedding
                            )
                            idx = int(torch.argmax(similarities, dim=0))
                            self._object_to_index[name] = idx
                        else:
                            # Unknown vocab, add to dict. It is OK to add val
                            # object names to the dict, as we still have no training
                            # signal for those in the training set, hence there is no
                            # reason to freeze the object vocab.
                            self._object_to_index[name] = len(self._object_to_index)
                    labels[-1].append(self._object_to_index[name])
                    boxes[-1].append(box)

        return boxes, labels

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
            boxes, labels = self._process_objects([d["objects"] for d in data])
            result += [
                {"imageId": scene["imageId"], "boxes": boxes_, "labels": labels_}
                for scene, boxes_, labels_ in zip(data, boxes, labels)
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
        dependencies = torch.tensor(  # pylint: disable=not-callable
            data["dependencies"]
        )
        embeddings = self.vectors.get_vecs_by_tokens(
            data["tokens"], lower_case_backup=True
        )
        return {
            "questionId": data["questionId"],
            "imageId": data["imageId"],
            "embeddings": embeddings,
            "dependencies": Data(edge_index=dependencies, x=embeddings),
            "answer": data["answer"],
        }


class SceneGraphTransformer:
    """Class for applying transformations to scene graphs."""

    def __init__(self) -> None:
        """Initialise a `SceneGraphTransformer` instance."""

    def __call__(self, data: SceneGraph) -> TrainableSceneGraph:
        """Transform data into a trainable format."""
        return {
            "imageId": data["imageId"],
            "boxes": torch.tensor(  # pylint: disable=not-callable
                data["boxes"], dtype=torch.float
            ),
            "labels": torch.tensor(  # pylint: disable=not-callable
                data["labels"], dtype=torch.int64
            ),
        }
