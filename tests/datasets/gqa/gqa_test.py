"""Tests for the GQA images dataset."""
from pathlib import Path

import pytest
from torch import Tensor

from gat_vqa.config.gqa import GQAFilemap, GQASplit, GQAVersion
from gat_vqa.datasets.gqa import GQA
from gat_vqa.datasets.gqa.images import GQAImages
from gat_vqa.datasets.gqa.objects import GQAObjects
from gat_vqa.datasets.gqa.questions import GQAQuestions
from gat_vqa.datasets.gqa.scene_graphs import GQASceneGraphs
from gat_vqa.datasets.gqa.spatial import GQASpatial

_SPLIT_VERSION_GRID = [(split, version) for split in GQASplit for version in GQAVersion]


def test_gqa_invalid_split_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid split type."""
    with pytest.raises(TypeError):
        GQA("train", GQAVersion.BALANCED)  # type: ignore


def test_gqa_invalid_version_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid version type."""
    with pytest.raises(TypeError):
        GQA(GQASplit.TRAIN, "balanced")  # type: ignore


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_version_property(split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `version` property returns a correct value."""
    dataset = GQA(split, version)
    assert dataset.version == version


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_split_property(split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `split` property returns a correct value."""
    dataset = GQA(split, version)
    assert dataset.split == split


def test_gqa_mismatched_question_version(gqa: Path) -> None:
    """Ensure the `split` property returns a correct value."""
    questions = GQAQuestions(GQAFilemap(gqa), GQASplit.TRAIN, GQAVersion.BALANCED)
    with pytest.raises(ValueError):
        GQA(GQASplit.TRAIN, GQAVersion.ALL, questions=questions)


def test_gqa_mismatched_question_split(gqa: Path) -> None:
    """Ensure the `split` property returns a correct value."""
    questions = GQAQuestions(GQAFilemap(gqa), GQASplit.TRAIN, GQAVersion.BALANCED)
    with pytest.raises(ValueError):
        GQA(GQASplit.VAL, GQAVersion.BALANCED, questions=questions)


def test_gqa_mismatched_scene_graphs_split(gqa: Path) -> None:
    """Ensure the `split` property returns a correct value."""
    scene_graphs = GQASceneGraphs(GQAFilemap(gqa), GQASplit.TRAIN)
    with pytest.raises(ValueError):
        GQA(GQASplit.VAL, GQAVersion.BALANCED, scene_graphs=scene_graphs)


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_images_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `images` property returns a correct value when an images \
    dataset is supplied."""
    images = GQAImages(GQAFilemap(gqa))
    dataset = GQA(split, version, images=images)
    assert dataset.images is images


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_images_property_none(split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `images` property returns `None` when no image dataset is \
    supplied."""
    dataset = GQA(split, version)
    assert dataset.images is None


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_objects_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `objects` property returns a correct value when an objects \
    dataset is supplied."""
    objects = GQAObjects(GQAFilemap(gqa))
    dataset = GQA(split, version, objects=objects)
    assert dataset.objects is objects


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_objects_property_none(split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `objects` property returns `None` when no objects dataset is \
    supplied."""
    dataset = GQA(split, version)
    assert dataset.objects is None


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_scene_graphs_property(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `scene_graphs` property returns a correct value when a scene \
    graphs dataset is supplied."""
    if split not in (GQASplit.TRAIN, GQASplit.VAL):
        with pytest.raises(ValueError):
            scene_graphs = GQASceneGraphs(GQAFilemap(gqa), split)
            return
    else:
        scene_graphs = GQASceneGraphs(GQAFilemap(gqa), split)
        dataset = GQA(split, version, scene_graphs=scene_graphs)
        assert dataset.scene_graphs is scene_graphs


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_scene_graphs_property_none(split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `scene_graphs` property returns `None` when no scene graphs \
    dataset is supplied."""
    dataset = GQA(split, version)
    assert dataset.scene_graphs is None


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_spatial_property(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `spatial` property returns a correct value when a spatial \
    dataset is supplied."""
    spatial = GQASpatial(GQAFilemap(gqa))
    dataset = GQA(split, version, spatial=spatial)
    assert dataset.spatial is spatial


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_spatial_property_none(split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `spatial` property returns `None` when no spatial dataset is \
    supplied."""
    dataset = GQA(split, version)
    assert dataset.spatial is None


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_questions_property(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `questions` property returns a correct value."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    dataset = GQA(split, version, questions=questions)
    assert dataset.questions is questions


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_questions_property_none(split: GQASplit, version: GQAVersion) -> None:
    """Ensure the `questions` property returns `None` when no questions dataset \
    is supplied."""
    dataset = GQA(split, version)
    assert dataset.questions is None


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
        split,
        version,
        questions=questions,
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
def test_gqa_getitem_none(split: GQASplit, version: GQAVersion) -> None:
    """Ensure no items are returned if no datasets are given."""
    dataset = GQA(split, version)
    with pytest.raises(IndexError):
        _ = dataset[0]


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_len(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the correct dataset length is returned given valid GQA data."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    dataset = GQA(split, version, questions)
    length = len(dataset)
    assert isinstance(length, int)
    assert length == 1


def test_gqa_empty_dataset_len() -> None:
    """Ensure the correct dataset length is returned given an empty dataset."""
    dataset = GQA(GQASplit.TRAIN, GQAVersion.BALANCED)
    length = len(dataset)
    assert isinstance(length, int)
    assert length == 0


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_gqa_key_to_index(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure key_to_index returns the correct index given valid GQA data."""
    questions = GQAQuestions(GQAFilemap(gqa), split, version)
    dataset = GQA(split, version, questions)

    for idx in range(len(dataset)):
        assert dataset.key_to_index(str(idx)) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("abc")


def test_gqa_empty_dataset_key_to_index() -> None:
    """Ensure a KeyError is raised given an empty dataset when calling key_to_index."""
    dataset = GQA(GQASplit.TRAIN, GQAVersion.BALANCED)
    with pytest.raises(KeyError):
        dataset.key_to_index("0")
