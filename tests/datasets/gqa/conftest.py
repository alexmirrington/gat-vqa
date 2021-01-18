"""Fixtures and configuration for GQA dataset tests."""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import h5py
import numpy as np
import pytest
from PIL import Image

from gat_vqa.config.gqa import GQAFilemap, GQASplit, GQAVersion

_QUESTIONS_FULL: Dict[str, Any] = {
    "0": {
        "imageId": "0",
        "question": "Is there a red apple on the table?",
        "answer": "no",
        "fullAnswer": "No, there is an apple but it is green.",
        "isBalanced": True,
        "groups": {"global": None, "local": "8r-binary-apple"},
        "entailed": [],
        "equivalent": [],
        "types": {
            "structural": "verify",
            "semantic": "relation",
            "detailed": "existAttrRel",
        },
        "annotations": {
            "question": {"4": "0", "7": "1"},
            "answer": {},
            "fullAnswer": {"4": "0"},
        },
        "semantic": [
            {"operation": "select", "argument": "table (1)", "dependencies": []},
            {
                "operation": "relate",
                "argument": "on, subject, apple (0)",
                "dependencies": [0],
            },
            {"operation": "filter", "argument": "red", "dependencies": [1]},
            {"operation": "exist", "argument": "?", "dependencies": [2]},
        ],
        "semanticStr": "select: table (1) -> relate: on, subject, \
apple (0) -> exist: ?",
    }
}

_QUESTIONS_PARTIAL: Dict[str, Any] = {
    "0": {
        "imageId": "0",
        "question": "Is there a red apple on the table?",
        "isBalanced": True,
    }
}

_SCENE_GRAPHS: Dict[str, Any] = {
    "0": {
        "width": 800,
        "height": 564,
        "location": "living room",
        "objects": {
            "0": {
                "name": "apple",
                "x": 386,
                "y": 174,
                "w": 264,
                "h": 260,
                "attributes": ["green", "round"],
                "relations": [{"name": "on", "object": "1"}],
            },
            "1": {
                "name": "table",
                "x": 4,
                "y": 100,
                "w": 791,
                "h": 457,
                "attributes": ["wooden"],
                "relations": [{"name": "under", "object": "0"}],
            },
        },
    }
}


def generate_image_files(
    filenames: List[Path],
    dimensions: List[Tuple[int, int]],
) -> None:
    """Create multiple images with given filenames and random sizes."""
    for path, dim in zip(filenames, dimensions):
        image = Image.new(mode="RGB", size=dim)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, "w") as img_file:
            image.save(img_file)


def generate_hdf5_files(
    filenames: List[Path], datasets: Mapping[str, np.ndarray]
) -> None:
    """Create multiple hdf5 files with given filenames and corresponding \
    datasets containing randomly generated data."""
    for path in filenames:
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        with h5py.File(path, "w") as h5_file:
            for dataset, data in datasets.items():
                h5_file.create_dataset(dataset, data=data)


def generate_json_files(filenames: List[Path], contents: List[Any]) -> None:
    """Create multiple JSON files with given filenames and corresponding contents."""
    for content, path in zip(contents, filenames):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with path.open("w") as json_file:
            json.dump(content, json_file)


@pytest.fixture(name="gqa", scope="session")
def fixture_gqa(tmp_path_factory):
    """Create a fake GQA dataset in a temporary directory for use in tests."""
    # Create directory
    root = tmp_path_factory.mktemp("gqa")
    filemap = GQAFilemap(root)

    question_count = len(_QUESTIONS_FULL)
    image_count = len(_SCENE_GRAPHS)

    # Create question files
    for split in GQASplit:
        for version in GQAVersion:
            if split == GQASplit.TRAIN and version == GQAVersion.ALL:
                paths = [
                    filemap.question_path(split, version, chunked=True, chunk_id=idx)
                    for idx in range(question_count)
                ]
                contents = [
                    {key: val}
                    for key, val in (
                        _QUESTIONS_FULL
                        if split in (GQASplit.TRAIN, GQASplit.VAL)
                        else _QUESTIONS_PARTIAL
                    ).items()
                ]
            else:
                paths = [filemap.question_path(split, version)]
                contents = [
                    _QUESTIONS_FULL
                    if split in (GQASplit.TRAIN, GQASplit.VAL)
                    else _QUESTIONS_PARTIAL
                ]

            generate_json_files(paths, contents)

    # Create scene graph files
    paths = [
        filemap.scene_graph_path(split) for split in (GQASplit.TRAIN, GQASplit.VAL)
    ]
    contents = [_SCENE_GRAPHS for _ in range(len(paths))]
    generate_json_files(paths, contents)

    # Create spatial features, one h5 file per image
    paths = [filemap.spatial_path(chunk_id=idx) for idx in range(image_count)]
    spatial_datasets: Dict[str, Any] = {
        "features": np.random.rand(len(_SCENE_GRAPHS), 2048, 7, 7)
    }
    generate_hdf5_files(paths, spatial_datasets)

    # Create spatial features meta file
    paths = [filemap.spatial_meta_path()]
    contents = [{str(idx): {"idx": 0, "file": idx} for idx in range(image_count)}]
    generate_json_files(paths, contents)

    # Create object features, one h5 file per image
    paths = [filemap.object_path(chunk_id=idx) for idx in range(image_count)]
    object_counts = {
        img_id: len(graph["objects"]) for img_id, graph in _SCENE_GRAPHS.items()
    }
    object_datasets: Dict[str, Any] = {
        "features": np.array(
            [
                np.concatenate(
                    (np.random.rand(n, 2048), np.zeros((100 - n, 2048))), axis=0
                )
                for n in object_counts.values()
            ]
        ),
        "bboxes": np.array(
            [
                np.concatenate(
                    (
                        np.array(
                            [
                                (val["x"], val["y"], val["w"], val["h"])
                                for val in _SCENE_GRAPHS[img_id]["objects"].values()
                            ]
                        ),
                        np.zeros((100 - n, 4)),
                    ),
                    axis=0,
                )
                for img_id, n in object_counts.items()
            ]
        ),
    }
    generate_hdf5_files(paths, object_datasets)

    # Create object features meta file
    paths = [filemap.object_meta_path()]
    contents = [{str(idx): {"idx": 0, "file": idx} for idx in range(image_count)}]
    generate_json_files(paths, contents)

    # Create image files
    paths = [filemap.image_path(str(idx)) for idx in range(image_count)]
    dimensions = [
        (int(graph["width"]), int(graph["height"])) for graph in _SCENE_GRAPHS.values()
    ]
    generate_image_files(paths, dimensions)

    return root
