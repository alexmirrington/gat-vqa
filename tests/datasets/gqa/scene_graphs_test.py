"""Tests for the GQA scene graphs dataset."""
from pathlib import Path
from typing import Any, Dict

import pytest

from gat_vqa.config.gqa import GQAFilemap, GQASplit
from gat_vqa.datasets.gqa.scene_graphs import GQASceneGraphs
from gat_vqa.schemas.gqa import GQASceneGraph


def test_scene_graphs_nonexistent_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with a non-existent root."""
    root = tmp_path / "gqa"
    with pytest.raises(ValueError):
        GQASceneGraphs(GQAFilemap(root), GQASplit.TRAIN)


def test_scene_graphs_valid_symlink_root(tmp_path: Path, gqa: Path) -> None:
    """Ensure a dataset instance can be created with a symlinked root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(gqa)
    GQASceneGraphs(GQAFilemap(ln_root), GQASplit.TRAIN)


def test_scene_graphs_valid_directory_root(gqa: Path) -> None:
    """Ensure a dataset instance can be created with a regular directory root."""
    GQASceneGraphs(GQAFilemap(gqa), GQASplit.TRAIN)


def test_scene_graphs_invalid_root_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid root type."""
    with pytest.raises(TypeError):
        GQASceneGraphs("gqa", GQASplit.TRAIN)  # type: ignore


def test_scene_graphs_invalid_split_type(gqa: Path) -> None:
    """Ensure a dataset instance cannot be created with an invalid split type."""
    with pytest.raises(TypeError):
        GQASceneGraphs(GQAFilemap(gqa), "train")  # type: ignore


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
def test_scene_graphs_split_property(gqa: Path, split: GQASplit) -> None:
    """Ensure the `split` property returns a correct value.

    We can only assume valid GQA data exists for train and val given they are
    the only splits for which scene graphs are present in the real GQA dataset.
    """
    dataset = GQASceneGraphs(GQAFilemap(gqa), split)
    assert dataset.split == split


@pytest.mark.parametrize("split", [GQASplit.TRAIN, GQASplit.VAL])
def test_scene_graphs_filemap_property(gqa: Path, split: GQASplit) -> None:
    """Ensure the `filemap` property returns a correct value.

    We can only assume valid GQA data exists for train and val given they are
    the only splits for which scene graphs are present in the real GQA dataset.
    """
    filemap = GQAFilemap(gqa)
    dataset = GQASceneGraphs(filemap, split)
    assert dataset.filemap == filemap


@pytest.mark.parametrize("split", [GQASplit.TRAIN, GQASplit.VAL])
def test_scene_graphs_getitem(gqa: Path, split: GQASplit) -> None:
    """Ensure an item is returned given valid GQA data.

    We can only assume valid GQA data exists for train and val given they are
    the only splits for which scene graphs are present in the real GQA dataset.
    """
    dataset = GQASceneGraphs(GQAFilemap(gqa), split)
    scene_graph = dataset[0]
    assert isinstance(scene_graph, dict)


@pytest.mark.parametrize("split", [GQASplit.TRAIN, GQASplit.VAL])
def test_scene_graphs_transformed_getitem(gqa: Path, split: GQASplit) -> None:
    """Ensure a transformed item is returned given valid GQA data and a transform.

    We can only assume valid GQA data exists for train and val given they are
    the only splits for which scene graphs are present in the real GQA dataset.
    """

    def transform(graph: GQASceneGraph) -> Dict[str, Any]:
        return {"width": graph["width"], "height": graph["height"]}

    dataset = GQASceneGraphs(GQAFilemap(gqa), split, transform=transform)
    scene_graph = dataset[0]
    assert isinstance(scene_graph, dict)
    assert "width" in scene_graph.keys()
    assert "height" in scene_graph.keys()
    assert "location" not in scene_graph.keys()
    assert "objects" not in scene_graph.keys()


@pytest.mark.parametrize("split", [GQASplit.TRAIN, GQASplit.VAL])
def test_scene_graphs_len(gqa: Path, split: GQASplit) -> None:
    """Ensure the correct dataset length is returned given valid GQA data.

    We can only assume valid GQA data exists for train and val given they are
    the only splits for which scene graphs are present in the real GQA dataset.
    """
    dataset = GQASceneGraphs(GQAFilemap(gqa), split)
    length = len(dataset)
    assert isinstance(length, int)
    assert length == 1


@pytest.mark.parametrize("split", [GQASplit.TRAIN, GQASplit.VAL])
def test_scene_graphs_key_to_index(gqa: Path, split: GQASplit) -> None:
    """Ensure key_to_index returns the correct index given valid GQA data."""
    dataset = GQASceneGraphs(GQAFilemap(gqa), split)

    for idx in range(len(dataset)):
        assert dataset.key_to_index(str(idx)) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("abc")
