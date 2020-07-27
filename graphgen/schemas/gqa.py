"""Schema definitions for the GQA dataset."""

from schema import And, Optional, Or, Schema

from .common import SLICE

GQA_IMAGE_ID = Schema(And(str, len))
GQA_QUESTION_ID = Schema(And(str, len))
GQA_OBJECT_ID = Schema(And(str, len))

GQA_QUESTION = Schema(
    {
        "imageId": GQA_IMAGE_ID,
        "question": And(str, len),
        "isBalanced": bool,
        Optional("answer"): And(str, len),
        Optional("fullAnswer"): And(str, len),
        Optional("groups"): {"global": Or(And(str), None), "local": Or(And(str), None)},
        Optional("entailed"): [GQA_QUESTION_ID],
        Optional("equivalent"): [GQA_QUESTION_ID],
        Optional("types"): {"structural": str, "semantic": str, "detailed": str},
        Optional("annotations"): {
            "question": {Optional(SLICE): GQA_OBJECT_ID},
            "answer": {Optional(SLICE): GQA_OBJECT_ID},
            "fullAnswer": {Optional(SLICE): GQA_OBJECT_ID},
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

GQA_QUESTIONS = Schema({GQA_QUESTION_ID: GQA_QUESTION})
