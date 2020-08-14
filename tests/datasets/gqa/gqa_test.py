"""Tests for the GQA images dataset."""
from pathlib import Path

import pytest

from graphgen.config.gqa import GQAFilemap, GQASplit, GQAVersion
from graphgen.datasets.gqa import GQA
from graphgen.datasets.gqa.images import GQAImages
from graphgen.datasets.gqa.objects import GQAObjects
from graphgen.datasets.gqa.questions import GQAQuestions
from graphgen.datasets.gqa.scene_graphs import GQASceneGraphs
from graphgen.datasets.gqa.spatial import GQASpatial
from graphgen.schemas.gqa import GQA_QUESTION_SCHEMA, GQA_SCENE_GRAPH_SCHEMA

_SPLIT_VERSION_GRID = [(split, version) for split in GQASplit for version in GQAVersion]


def test_gqa_nonexistent_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with a non-existent root."""
    root = tmp_path / "gqa"
    with pytest.raises(ValueError):
        GQA(GQAFilemap(root), GQASplit.TRAIN, GQAVersion.BALANCED)


def test_gqa_valid_symlink_root(tmp_path: Path, gqa: Path) -> None:
    """Ensure a dataset instance can be created with a symlinked root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(gqa)
    GQA(GQAFilemap(ln_root), GQASplit.TRAIN, GQAVersion.BALANCED)


def test_gqa_valid_directory_root(gqa: Path) -> None:
    """Ensure a dataset instance can be created with a regular directory root."""
    GQA(GQAFilemap(gqa), GQASplit.TRAIN, GQAVersion.BALANCED)


def test_gqa_invalid_root_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid root type."""
    with pytest.raises(TypeError):
        GQA("gqa", GQASplit.TRAIN, GQAVersion.BALANCED)  # type: ignore


def test_gqa_invalid_split_type(gqa: Path) -> None:
    """Ensure a dataset instance cannot be created with an invalid split type."""
    with pytest.raises(TypeError):
        GQA(GQAFilemap(gqa), "train", GQAVersion.BALANCED)  # type: ignore


def test_gqa_invalid_version_type(gqa: Path) -> None:
    """Ensure a dataset instance cannot be created with an invalid version type."""
    with pytest.raises(TypeError):
        GQA(GQAFilemap(gqa), GQASplit.TRAIN, "balanced")  # type: ignore


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_nonexistent_question_json(
    tmp_path: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure a dataset instance cannot be created with a missing question file."""
    root = tmp_path / "gqa"
    root.mkdir()

    with pytest.raises(ValueError):
        GQA(GQAFilemap(root), split, version)


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_filemap_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `filemap` property returns a correct value."""
    filemap = GQAFilemap(gqa)
    dataset = GQA(filemap, split, version)
    assert dataset.filemap == filemap


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_version_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `version` property returns a correct value."""
    dataset = GQA(GQAFilemap(gqa), split, version)
    assert dataset.version == version


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_split_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `split` property returns a correct value."""
    dataset = GQA(GQAFilemap(gqa), split, version)
    assert dataset.split == split


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_images_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `images` property returns a correct value."""
    dataset = GQA(GQAFilemap(gqa), split, version)
    assert isinstance(dataset.images, GQAImages)


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_objects_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `objects` property returns a correct value."""
    dataset = GQA(GQAFilemap(gqa), split, version)
    assert isinstance(dataset.objects, GQAObjects)


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_scene_graphs_property(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `scene_graphs` property returns a correct value."""
    dataset = GQA(GQAFilemap(gqa), split, version)
    if split in (GQASplit.TRAIN, GQASplit.VAL):
        assert isinstance(dataset.scene_graphs, GQASceneGraphs)
    else:
        assert dataset.scene_graphs is None


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_spatial_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `spatial` property returns a correct value."""
    dataset = GQA(GQAFilemap(gqa), split, version)
    assert isinstance(dataset.spatial, GQASpatial)


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_questions_property(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `questions` property returns a correct value."""
    dataset = GQA(GQAFilemap(gqa), split, version)
    assert isinstance(dataset.questions, GQAQuestions)


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_getitem(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure an item is returned given valid GQA data."""
    dataset = GQA(GQAFilemap(gqa), split, version)
    question, image, spatial, objects, scene_graph = dataset[0]

    # Validate scene question data
    GQA_QUESTION_SCHEMA.validate(question)

    # Validate scene graph data
    if split in (GQASplit.TRAIN, GQASplit.VAL):
        GQA_SCENE_GRAPH_SCHEMA.validate(scene_graph)
    else:
        assert scene_graph is None

    # Validate spatial features
    assert isinstance(spatial, dict)
    assert "features" in spatial.keys()
    assert spatial["features"].shape == (2048, 7, 7)

    # Validate object features
    assert isinstance(objects, dict)
    assert "features" in objects.keys()
    assert objects["features"].shape == (100, 2048)
    assert "bboxes" in objects.keys()
    assert objects["bboxes"].shape == (100, 4)


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_len(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the correct dataset length is returned given valid GQA data."""
    dataset = GQA(GQAFilemap(gqa), split, version)
    length = len(dataset)
    assert isinstance(length, int)
    assert length == 10 if split == GQASplit.TRAIN and version == GQAVersion.ALL else 1


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_key_to_index(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure key_to_index returns the correct index given valid GQA data."""
    dataset = GQA(GQAFilemap(gqa), split, version)

    for idx in range(len(dataset)):
        assert dataset.key_to_index(str(idx)) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("abc")
