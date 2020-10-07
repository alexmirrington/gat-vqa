"""Schema definitions for the GQA dataset."""

from typing import Dict, List, Optional, TypedDict

# Alternative declaration to allow "global" as a key.
GQAQuestionGroup = TypedDict(
    "GQAQuestionGroup", {"global": Optional[str], "local": Optional[str]}
)


class GQAQuestionType(TypedDict):
    """Class wrapper for GQA question types.

    Attributes:
    -----------
    `structural`: Question structural type, e.g. query (open), verify (yes/no).

    `semantic`: Question subject's type, e.g. 'attribute' for questions about
    color or material.

    `detailed`: Question complete type specification, out of 20+ subtypes,
    e.g. twoSame.

    References:
    -----------
    https://cs.stanford.edu/people/dorarad/gqa/download.html
    """

    structural: Optional[str]
    semantic: Optional[str]
    detailed: Optional[str]


class GQAQuestionAnnotation(TypedDict):
    """Class wrapper for GQA question object annotations.

    Attributes:
    -----------
    `question`: Visual pointer from question words (e.g. slice "2:4", key) to
    object (objectId, value).

    `answer`: Visual pointer from answer word (e.g. index "0", key) to object
    (objectId, value).

    `full_answer`: Visual pointer from answer words (e.g: "0", "2:4", key) to
    object (objectId, value).

    References:
    -----------
    https://cs.stanford.edu/people/dorarad/gqa/download.html
    """

    question: Dict[slice, str]
    answer: Dict[slice, str]
    fullAnswer: Dict[slice, str]


class GQAQuestionSemanticStep(TypedDict):
    """Class wrapper for GQA question object annotations.

    Attributes:
    -----------
    `operation`: Reasoning operation. e.g. select, filter, relate.

    `argument`: Operation argument(s). Depends on the specific operation,
    usually an object id.

    `dependencies`: Prior steps the current one depends on.

    References:
    -----------
    https://cs.stanford.edu/people/dorarad/gqa/download.html
    """

    operation: str
    argument: str
    dependencies: List[int]


class _GQAQuestionBase(TypedDict):
    """Base class wraper for mandatory GQA question information."""

    imageId: str
    question: str
    isBalanced: bool


class GQAQuestion(_GQAQuestionBase, total=False):
    """Class wrapper for GQA question information."""

    questionId: str
    answer: Optional[str]
    fullAnswer: Optional[str]
    groups: Optional[GQAQuestionGroup]
    entailed: Optional[List[str]]
    equivalent: Optional[List[str]]
    types: Optional[GQAQuestionType]
    annotations: Optional[GQAQuestionAnnotation]
    semantic: Optional[List[GQAQuestionSemanticStep]]
    semanticStr: Optional[str]


# Alternative declaration to allow "object" as a key.
GQASceneGraphObjectRelation = TypedDict(
    "GQASceneGraphObjectRelation", {"name": str, "object": str}
)


class GQASceneGraphObject(TypedDict):
    """Class wrapper for required GQA scene graph object information."""

    name: str
    x: int
    y: int
    w: int
    h: int
    attributes: List[str]
    relations: List[GQASceneGraphObjectRelation]


class _GQASceneGraphBase(TypedDict):
    """Class wrapper for required GQA scene graph information."""

    imageId: str
    width: int
    height: int
    objects: Dict[str, GQASceneGraphObject]


class GQASceneGraph(_GQASceneGraphBase, total=False):
    """Class wrapper for GQA scene graph information."""

    location: str
    weather: str
