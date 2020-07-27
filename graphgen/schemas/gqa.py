"""Schema definitions for the GQA dataset."""

from schema import And, Optional, Or, Schema

from .common import SLICE_SCHEMA

GQA_IMAGE_ID_SCHEMA = Schema(And(str, len))
GQA_QUESTION_ID_SCHEMA = Schema(And(str, len))
GQA_OBJECT_ID_SCHEMA = Schema(And(str, len))

GQA_QUESTION_SCHEMA = Schema(
    {
        "imageId": GQA_IMAGE_ID_SCHEMA,
        "question": And(str, len),
        "isBalanced": bool,
        Optional("answer"): And(str, len),
        Optional("fullAnswer"): And(str, len),
        Optional("groups"): {"global": Or(And(str), None), "local": Or(And(str), None)},
        Optional("entailed"): [GQA_QUESTION_ID_SCHEMA],
        Optional("equivalent"): [GQA_QUESTION_ID_SCHEMA],
        Optional("types"): {"structural": str, "semantic": str, "detailed": str},
        Optional("annotations"): {
            "question": {Optional(SLICE_SCHEMA): GQA_OBJECT_ID_SCHEMA},
            "answer": {Optional(SLICE_SCHEMA): GQA_OBJECT_ID_SCHEMA},
            "fullAnswer": {Optional(SLICE_SCHEMA): GQA_OBJECT_ID_SCHEMA},
        },
        Optional("semantic"): [
            {
                "operation": str,
                "argument": str,
                "dependencies": [And(int, lambda x: x >= 0)],
            }
        ],
        Optional("semanticStr"): And(str),
    }
)

GQA_SCENE_GRAPH_SCHEMA = Schema(
    {
        "width": And(int, lambda x: x > 0),
        "height": And(int, lambda x: x > 0),
        Optional("location"): str,
        Optional("weather"): str,
        "objects": {
            Optional(GQA_OBJECT_ID_SCHEMA): {
                "name": str,
                "x": And(int, lambda x: x >= 0),
                "y": And(int, lambda x: x >= 0),
                "w": int,
                "h": int,
                "attributes": [str],
                "relations": [{"name": str, "object": GQA_OBJECT_ID_SCHEMA}],
            }
        },
    }
)

GQA_QUESTIONS_SCHEMA = Schema({GQA_QUESTION_ID_SCHEMA: GQA_QUESTION_SCHEMA})
GQA_SCENE_GRAPHS_SCHEMA = Schema({GQA_IMAGE_ID_SCHEMA: GQA_SCENE_GRAPH_SCHEMA})
