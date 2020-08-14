"""Fixtures and configuration for GQA dataset tests."""

import json
from pathlib import Path
from random import randint
from typing import Any, List, Mapping, Tuple

import h5py
import numpy as np
import pytest
from PIL import Image

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


def generate_image_files(
    filenames: List[Path],
    min_size: Tuple[int, int] = (16, 16),
    max_size: Tuple[int, int] = (32, 32),
) -> None:
    """Create multiple images with given filenames and random sizes."""
    for path in filenames:
        image = Image.new(
            mode="RGB",
            size=(randint(min_size[0], max_size[0]), randint(min_size[1], max_size[1])),
        )
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, "w") as img_file:
            image.save(img_file)


def generate_hdf5_files(
    filenames: List[Path], dataset_shapes: Mapping[str, Tuple[int, ...]]
) -> None:
    """Create multiple hdf5 files with given filenames and corresponding \
    datasets containing randomly generated data."""
    for path in filenames:
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        with h5py.File(path, "w") as h5_file:
            for dataset, shape in dataset_shapes.items():
                h5_file.create_dataset(dataset, data=np.random.rand(*shape))


def generate_json_files(filenames: List[Path], contents: List[Any]) -> None:
    """Create multiple JSON files with given filenames and corresponding contents."""
    for content, path in zip(contents, filenames):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with path.open("w") as json_file:
            json.dump(content, json_file)


@pytest.fixture(name="gqa")
def fixture_gqa(tmp_path_factory):
    """Create a fake GQA dataset in a temporary directory for use in tests."""
    # Create directory
    root = tmp_path_factory.mktemp("gqa")
    filemap = GQAFilemap(root)

    image_count = 8
    spatial_count = 16
    object_count = 16

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

            contents = [
                {str(idx): _QUESTION_SAMPLE_FULL}
                if split in (GQASplit.TRAIN, GQASplit.VAL)
                else {str(idx): _QUESTION_SAMPLE_PARTIAL}
                for idx in range(len(paths))
            ]

            generate_json_files(paths, contents)

    # Create scene graph files
    paths = [
        filemap.scene_graph_path(split) for split in (GQASplit.TRAIN, GQASplit.VAL)
    ]
    contents = [{"0": _SCENE_GRAPH_SAMPLE_FULL} for _ in range(len(paths))]
    generate_json_files(paths, contents)

    # Create spatial features
    paths = [filemap.spatial_path(chunk_id=idx) for idx in range(spatial_count)]
    spatial_datasets = {"features": (1, 2048, 7, 7)}
    generate_hdf5_files(paths, spatial_datasets)

    # Create spatial features meta file
    paths = [filemap.spatial_meta_path()]
    contents = [{str(idx): {"idx": 0, "file": idx} for idx in range(spatial_count)}]
    generate_json_files(paths, contents)

    # Create object features
    paths = [filemap.object_path(chunk_id=idx) for idx in range(object_count)]
    object_datasets = {"features": (1, 100, 2048), "bboxes": (1, 100, 4)}
    generate_hdf5_files(paths, object_datasets)

    # Create object features meta file
    paths = [filemap.object_meta_path()]
    contents = [{str(idx): {"idx": 0, "file": idx} for idx in range(object_count)}]
    generate_json_files(paths, contents)

    # Create image files
    paths = [filemap.image_path(str(idx)) for idx in range(image_count)]
    generate_image_files(paths)

    return root
