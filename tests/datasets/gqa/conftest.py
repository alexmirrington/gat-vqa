"""Fixtures and configuration for GQA dataset tests."""

import json

import h5py
import numpy as np
import pytest

from graphgen.config.gqa import GQAFilemap, GQASplit, GQAVersion

_QUESTION_SAMPLE_FULL = {
    "imageId": "2407890",
    "question": "Is there a red apple on the table?",
    "answer": "no",
    "fullAnswer": "No, there is an apple but it is green.",
    "isBalanced": True,
    "groups": {"global": None, "local": "8r-binary-apple"},
    "entailed": ["1352631", "1245832", "842753"],
    "equivalent": ["1245832", "842753"],
    "types": {
        "structural": "verify",
        "semantic": "relation",
        "detailed": "existAttrRel",
    },
    "annotations": {
        "question": {"4": "271881", "7": "279472"},
        "answer": {},
        "fullAnswer": {"4": "271881"},
    },
    "semantic": [
        {"operation": "select", "argument": "table (279472)", "dependencies": []},
        {
            "operation": "relate",
            "argument": "on, subject, apple (271881)",
            "dependencies": [0],
        },
        {"operation": "filter", "argument": "red", "dependencies": [1]},
        {"operation": "exist", "argument": "?", "dependencies": [2]},
    ],
    "semanticStr": "select: table (279472) -> \
        relate: on, subject, apple (271881) -> exist: ?",
}

_QUESTION_SAMPLE_PARTIAL = {
    "imageId": "2407890",
    "question": "Is there a red apple on the table?",
    "isBalanced": True,
}

_SCENE_GRAPH_SAMPLE_FULL = {
    "width": 640,
    "height": 480,
    "location": "living room",
    "objects": {
        "271881": {
            "name": "chair",
            "x": 220,
            "y": 310,
            "w": 50,
            "h": 80,
            "attributes": ["brown", "wooden", "small"],
            "relations": [
                {"name": "on", "object": "275312"},
                {"name": "near", "object": "279472"},
            ],
        }
    },
}


@pytest.fixture(name="gqa")
def fixture_gqa(tmp_path_factory):
    """Create a fake GQA dataset in a temporary directory for use in tests."""
    # Create directory
    root = tmp_path_factory.mktemp("gqa")
    filemap = GQAFilemap(root)

    # Create question files
    for split in GQASplit:
        for version in GQAVersion:
            if split == GQASplit.TRAIN and version == GQAVersion.ALL:
                paths = [
                    filemap.question_path(split, version, chunked=True, chunk_id=idx)
                    for idx in range(10)
                ]
            else:
                paths = [filemap.question_path(split, version)]

            for idx, path in enumerate(paths):
                if not path.parent.exists():
                    path.parent.mkdir(parents=True)

                content = (
                    {str(idx): _QUESTION_SAMPLE_FULL}
                    if split in (GQASplit.TRAIN, GQASplit.VAL)
                    else {str(idx): _QUESTION_SAMPLE_PARTIAL}
                )
                with path.open("w") as json_file:
                    json.dump(content, json_file)

    # Create scene graph files
    for split in (GQASplit.TRAIN, GQASplit.VAL):
        path = filemap.scene_graph_path(split)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        content = {"0": _SCENE_GRAPH_SAMPLE_FULL}
        with path.open("w") as json_file:
            json.dump(content, json_file)

    # Create spatial features
    paths = [filemap.spatial_path(chunk_id=idx) for idx in range(16)]
    content = np.zeros((1, 2048, 7, 7))
    for path in paths:
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        with h5py.File(path, "w") as h5_file:
            h5_file.create_dataset("features", data=content)

    # Create spatial features meta file
    with open(filemap.spatial_meta_path(), "w") as json_file:
        json.dump({str(idx): {"idx": 0, "file": idx} for idx in range(16)}, json_file)

    # Create object features
    paths = [filemap.object_path(chunk_id=idx) for idx in range(16)]
    content_features = np.zeros((1, 100, 2048))
    content_bboxes = np.zeros((1, 100, 4))

    for path in paths:
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        with h5py.File(path, "w") as h5_file:
            h5_file.create_dataset("features", data=content_features)
            h5_file.create_dataset("bboxes", data=content_bboxes)

    # Create object features meta file
    with open(filemap.object_meta_path(), "w") as json_file:
        json.dump({str(idx): {"idx": 0, "file": idx} for idx in range(16)}, json_file)

    return root
