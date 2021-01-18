"""Tests for the GQA object features dataset."""
from pathlib import Path

import pytest
from torch import Tensor

from gat_vqa.config.gqa import GQAFilemap
from gat_vqa.datasets.gqa.objects import GQAObjects


def test_objects_nonexistent_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with a non-existent root."""
    root = tmp_path / "gqa"
    with pytest.raises(ValueError):
        GQAObjects(GQAFilemap(root))


def test_objects_valid_symlink_root(tmp_path: Path, gqa: Path) -> None:
    """Ensure a dataset instance can be created with a symlinked root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(gqa)
    GQAObjects(GQAFilemap(ln_root))


def test_objects_valid_directory_root(gqa: Path) -> None:
    """Ensure a dataset instance can be created with a regular directory root."""
    GQAObjects(GQAFilemap(gqa))


def test_objects_invalid_root_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid root type."""
    with pytest.raises(TypeError):
        GQAObjects("gqa")  # type: ignore


def test_objects_empty_directory_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with an empty root directory."""
    root = tmp_path / "gqa"
    root.mkdir()

    with pytest.raises(ValueError):
        GQAObjects(GQAFilemap(root))


def test_objects_filemap_property(gqa: Path) -> None:
    """Ensure the `filemap` property returns a correct value."""
    filemap = GQAFilemap(gqa)
    dataset = GQAObjects(filemap)
    assert dataset.filemap == filemap


def test_objects_getitem(gqa: Path) -> None:
    """Ensure an item is returned given valid GQA data."""
    dataset = GQAObjects(GQAFilemap(gqa))
    objects, boxes = dataset[0]

    # Validate object features
    assert isinstance(objects, Tensor)
    assert objects.size() == (100, 2048)
    assert isinstance(boxes, Tensor)
    assert boxes.size() == (100, 4)


def test_objects_len(gqa: Path) -> None:
    """Ensure the correct dataset length is returned given valid GQA data."""
    dataset = GQAObjects(GQAFilemap(gqa))
    length = len(dataset)
    assert isinstance(length, int)
    assert length == 1


def test_objects_key_to_index(gqa: Path) -> None:
    """Ensure key_to_index returns the correct index given valid GQA data."""
    dataset = GQAObjects(GQAFilemap(gqa))
    for idx in range(len(dataset)):
        assert isinstance(dataset.key_to_index(str(idx)), int)
    with pytest.raises(KeyError):
        dataset.key_to_index("abc")
    with pytest.raises(KeyError):
        dataset.key_to_index(str(len(dataset)))
