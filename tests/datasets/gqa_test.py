"""Tests for the GQA dataset."""
import json
from pathlib import Path

import pytest

from graphgen.config.gqa import GQAFilemap, GQASplit, GQAVersion
from graphgen.datasets.gqa import GQAQuestions, GQASceneGraphs
from graphgen.schemas.gqa import GQA_QUESTION_SCHEMA, GQA_SCENE_GRAPH_SCHEMA

_SPLIT_VERSION_GRID = [(split, version) for split in GQASplit for version in GQAVersion]

_QUESTION_SAMPLE_FULL = {
    "1238592": {
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
}

_QUESTION_SAMPLE_PARTIAL = {
    "1238592": {
        "imageId": "2407890",
        "question": "Is there a red apple on the table?",
        "isBalanced": True,
    }
}

_SCENE_GRAPH_SAMPLE_FULL = {
    "2407890": {
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
}


@pytest.fixture(scope="session", name="gqa_data")
def fixture_gqa_data(tmp_path_factory):
    """Create a fake GQA dataset in a temporary directory for use in tests."""
    root = tmp_path_factory.mktemp("gqa")
    filemap = GQAFilemap(root)

    # Seed question files
    for split in GQASplit:
        for version in GQAVersion:
            if split == GQASplit.TRAIN and version == GQAVersion.ALL:
                paths = [
                    filemap.question_path(split, version, chunked=True, chunk_id=idx)
                    for idx in range(10)
                ]
            else:
                paths = [filemap.question_path(split, version)]

            for path in paths:
                if not path.parent.exists():
                    path.parent.mkdir(parents=True)

                content = (
                    _QUESTION_SAMPLE_FULL
                    if split in (GQASplit.TRAIN, GQASplit.VAL)
                    else _QUESTION_SAMPLE_PARTIAL
                )
                with path.open("w") as file:
                    json.dump(content, file)

    # Seed scene graph files
    for split in (GQASplit.TRAIN, GQASplit.VAL):
        path = filemap.scene_graph_path(split)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        content = _SCENE_GRAPH_SAMPLE_FULL
        with path.open("w") as file:
            json.dump(content, file)

    return root


# region GQAQuestions


def test_questions_nonexistent_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with a non-existent root."""
    root = tmp_path / "gqa"
    with pytest.raises(ValueError):
        GQAQuestions(GQAFilemap(root), GQASplit.TRAIN, GQAVersion.BALANCED)


def test_questions_valid_symlink_root(tmp_path: Path, gqa_data: Path) -> None:
    """Ensure a dataset instance can be created with a symlinked root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(gqa_data)
    GQAQuestions(GQAFilemap(ln_root), GQASplit.TRAIN, GQAVersion.BALANCED)


def test_questions_valid_directory_root(gqa_data: Path) -> None:
    """Ensure a dataset instance can be created with a regular directory root."""
    GQAQuestions(GQAFilemap(gqa_data), GQASplit.TRAIN, GQAVersion.BALANCED)


def test_questions_invalid_root_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid root type."""
    with pytest.raises(TypeError):
        GQAQuestions("gqa", GQASplit.TRAIN, GQAVersion.BALANCED)  # type: ignore


def test_questions_invalid_split_type(gqa_data: Path) -> None:
    """Ensure a dataset instance cannot be created with an invalid split type."""
    with pytest.raises(TypeError):
        GQAQuestions(GQAFilemap(gqa_data), "train", GQAVersion.BALANCED)  # type: ignore


def test_questions_invalid_version_type(gqa_data: Path) -> None:
    """Ensure a dataset instance cannot be created with an invalid version type."""
    with pytest.raises(TypeError):
        GQAQuestions(GQAFilemap(gqa_data), GQASplit.TRAIN, "balanced")  # type: ignore


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_nonexistent_question_json(
    tmp_path: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure a dataset instance cannot be created with a missing question file."""
    root = tmp_path / "gqa"
    root.mkdir()

    with pytest.raises(ValueError):
        GQAQuestions(GQAFilemap(root), split, version)


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_version_property(
    gqa_data: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `version` property returns a correct value."""
    dataset = GQAQuestions(GQAFilemap(gqa_data), split, version)
    assert dataset.version == version


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_split_property(
    gqa_data: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `split` property returns a correct value."""
    dataset = GQAQuestions(GQAFilemap(gqa_data), split, version)
    assert dataset.split == split


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_filemap_property(
    gqa_data: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `filemap` property returns a correct value."""
    filemap = GQAFilemap(gqa_data)
    dataset = GQAQuestions(filemap, split, version)
    assert dataset.filemap == filemap


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_getitem(
    gqa_data: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure an item is returned given valid GQA data."""
    dataset = GQAQuestions(GQAFilemap(gqa_data), split, version)
    question = dataset[0]
    GQA_QUESTION_SCHEMA.validate(question)


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_len(gqa_data: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the correct dataset length is returned given valid GQA data."""
    dataset = GQAQuestions(GQAFilemap(gqa_data), split, version)
    length = len(dataset)
    assert isinstance(length, int)
    assert length == 10 if split == GQASplit.TRAIN and version == GQAVersion.ALL else 1


# endregion GQAQuestions


# region GQASceneGraphs


def test_scene_graphs_nonexistent_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with a non-existent root."""
    root = tmp_path / "gqa"
    with pytest.raises(ValueError):
        GQASceneGraphs(GQAFilemap(root), GQASplit.TRAIN)


def test_scene_graphs_valid_symlink_root(tmp_path: Path, gqa_data: Path) -> None:
    """Ensure a dataset instance can be created with a symlinked root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(gqa_data)
    GQASceneGraphs(GQAFilemap(ln_root), GQASplit.TRAIN)


def test_scene_graphs_valid_directory_root(gqa_data: Path) -> None:
    """Ensure a dataset instance can be created with a regular directory root."""
    GQASceneGraphs(GQAFilemap(gqa_data), GQASplit.TRAIN)


def test_scene_graphs_invalid_root_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid root type."""
    with pytest.raises(TypeError):
        GQASceneGraphs("gqa", GQASplit.TRAIN)  # type: ignore


def test_scene_graphs_invalid_split_type(gqa_data: Path) -> None:
    """Ensure a dataset instance cannot be created with an invalid split type."""
    with pytest.raises(TypeError):
        GQASceneGraphs(GQAFilemap(gqa_data), "train")  # type: ignore


@pytest.mark.parametrize("split", iter(GQASplit))
def test_scene_graphs_nonexistent_scene_graph_json(
    tmp_path: Path, split: GQASplit
) -> None:
    """Ensure a dataset instance cannot be created with a missing sceneGraphs file."""
    root = tmp_path / "gqa"
    root.mkdir()

    with pytest.raises(ValueError):
        GQASceneGraphs(GQAFilemap(root), split)


@pytest.mark.parametrize("split", [GQASplit.TRAIN, GQASplit.VAL])
def test_scene_graphs_split_property(gqa_data: Path, split: GQASplit) -> None:
    """Ensure the `split` property returns a correct value.

    We can only assume valid GQA data exists for train and val given they are
    the only splits for which scene graphs are present in the real GQA dataset.
    """
    dataset = GQASceneGraphs(GQAFilemap(gqa_data), split)
    assert dataset.split == split


@pytest.mark.parametrize("split", [GQASplit.TRAIN, GQASplit.VAL])
def test_scene_graphs_filemap_property(gqa_data: Path, split: GQASplit) -> None:
    """Ensure the `filemap` property returns a correct value.

    We can only assume valid GQA data exists for train and val given they are
    the only splits for which scene graphs are present in the real GQA dataset.
    """
    filemap = GQAFilemap(gqa_data)
    dataset = GQASceneGraphs(filemap, split)
    assert dataset.filemap == filemap


@pytest.mark.parametrize("split", [GQASplit.TRAIN, GQASplit.VAL])
def test_scene_graphs_getitem(gqa_data: Path, split: GQASplit) -> None:
    """Ensure an item is returned given valid GQA data.

    We can only assume valid GQA data exists for train and val given they are
    the only splits for which scene graphs are present in the real GQA dataset.
    """
    dataset = GQASceneGraphs(GQAFilemap(gqa_data), split)
    scene_graph = dataset[0]
    GQA_SCENE_GRAPH_SCHEMA.validate(scene_graph)


@pytest.mark.parametrize("split", [GQASplit.TRAIN, GQASplit.VAL])
def test_scene_graphs_len(gqa_data: Path, split: GQASplit) -> None:
    """Ensure the correct dataset length is returned given valid GQA data.

    We can only assume valid GQA data exists for train and val given they are
    the only splits for which scene graphs are present in the real GQA dataset.
    """
    dataset = GQASceneGraphs(GQAFilemap(gqa_data), split)
    length = len(dataset)
    assert isinstance(length, int)
    assert length == 1


# endregion GQASceneGraphs
