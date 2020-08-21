"""Tests for the GQA images dataset."""
from pathlib import Path

import pytest
from torch import Tensor

from graphgen.config.gqa import GQAFilemap, GQASplit, GQAVersion
from graphgen.datasets.gqa import GQA
from graphgen.datasets.gqa.images import GQAImages
from graphgen.datasets.gqa.objects import GQAObjects
from graphgen.datasets.gqa.questions import GQAQuestions
from graphgen.datasets.gqa.scene_graphs import GQASceneGraphs
from graphgen.datasets.gqa.spatial import GQASpatial

_SPLIT_VERSION_GRID = [(split, version) for split in GQASplit for version in GQAVersion]


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_version_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `version` property returns a correct value."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    dataset = GQA(questions)
    assert dataset.version == version


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_split_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `split` property returns a correct value."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    dataset = GQA(questions)
    assert dataset.split == split


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_images_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `images` property returns a correct value when am images \
    dataset is supplied."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    images = GQAImages(GQAFilemap(gqa))
    dataset = GQA(questions, images=images)
    assert dataset.images is images


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_images_property_none(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `images` property returns `None` when no image dataset is \
    supplied."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    dataset = GQA(questions)
    assert dataset.images is None


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_objects_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `objects` property returns a correct value when an objects \
    dataset is supplied."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    objects = GQAObjects(GQAFilemap(gqa))
    dataset = GQA(questions, objects=objects)
    assert dataset.objects is objects


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_objects_property_none(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `objects` property returns `None` when no objects dataset is \
    supplied."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    dataset = GQA(questions)
    assert dataset.objects is None


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_scene_graphs_property(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `scene_graphs` property returns a correct value when a scene \
    graphs dataset is supplied."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    if split not in (GQASplit.TRAIN, GQASplit.VAL):
        with pytest.raises(ValueError):
            scene_graphs = GQASceneGraphs(GQAFilemap(gqa), split)
            return
    else:
        scene_graphs = GQASceneGraphs(GQAFilemap(gqa), split)
        dataset = GQA(questions, scene_graphs=scene_graphs)
        assert dataset.scene_graphs is scene_graphs


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_scene_graphs_property_none(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `scene_graphs` property returns `None` when no scene graphs \
    dataset is supplied."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    dataset = GQA(questions)
    assert dataset.scene_graphs is None


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_spatial_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `spatial` property returns a correct value when a spatial \
    dataset is supplied."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    spatial = GQASpatial(GQAFilemap(gqa))
    dataset = GQA(questions, spatial=spatial)
    assert dataset.spatial is spatial


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_spatial_property_none(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `spatial` property returns `None` when no spatial dataset is \
    supplied."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    dataset = GQA(questions)
    assert dataset.spatial is None


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_questions_property(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `questions` property returns a correct value."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    dataset = GQA(questions)
    assert dataset.questions is questions


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_getitem_all(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure an item is returned given valid GQA data."""
    filemap = GQAFilemap(gqa)
    questions = GQAQuestions(filemap, split, version)
    images = GQAImages(filemap)
    objects = GQAObjects(filemap)
    spatials = GQASpatial(filemap)
    scene_graphs = (
        GQASceneGraphs(filemap, split)
        if split in (GQASplit.TRAIN, GQASplit.VAL)
        else None
    )

    dataset = GQA(
        questions,
        images=images,
        objects=objects,
        spatial=spatials,
        scene_graphs=scene_graphs,
    )
    sample = dataset[0]

    # Validate question data
    assert isinstance(sample["question"], dict)

    # Validate image data
    assert isinstance(sample["image"], Tensor)

    # Validate scene graph data
    if split in (GQASplit.TRAIN, GQASplit.VAL):
        assert isinstance(sample["scene_graph"], dict)
    else:
        assert "scene_graph" not in sample.keys()

    # Validate spatial features
    assert isinstance(sample["spatial"], Tensor)
    assert sample["spatial"].size() == (2048, 7, 7)

    # Validate object features
    assert isinstance(sample["objects"], Tensor)
    assert sample["objects"].size() == (100, 2048)
    assert isinstance(sample["boxes"], Tensor)
    assert sample["boxes"].size() == (100, 4)


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_getitem_questions_only(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure an item is returned given valid GQA data."""
    filemap = GQAFilemap(gqa)
    questions = GQAQuestions(filemap, split, version)
    dataset = GQA(questions)
    sample = dataset[0]

    # Validate scene question data
    assert isinstance(sample["question"], dict)

    # Validate all other data
    assert "image" not in sample.keys()
    assert "scene_graph" not in sample.keys()
    assert "spatial" not in sample.keys()
    assert "objects" not in sample.keys()
    assert "boxes" not in sample.keys()


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_len(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the correct dataset length is returned given valid GQA data."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    dataset = GQA(questions)
    length = len(dataset)
    assert isinstance(length, int)
    assert length == 1


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_key_to_index(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure key_to_index returns the correct index given valid GQA data."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    dataset = GQA(questions)

    for idx in range(len(dataset)):
        assert dataset.key_to_index(str(idx)) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("abc")
