"""Tests for the GQA spatial features dataset."""
from pathlib import Path

import pytest

from graphgen.config.gqa import GQAFilemap
from graphgen.datasets.gqa.spatial import GQASpatial


def test_spatial_nonexistent_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with a non-existent root."""
    root = tmp_path / "gqa"
    with pytest.raises(ValueError):
        GQASpatial(GQAFilemap(root))


def test_spatial_valid_symlink_root(tmp_path: Path, gqa: Path) -> None:
    """Ensure a dataset instance can be created with a symlinked root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(gqa)
    GQASpatial(GQAFilemap(ln_root))


def test_spatial_valid_directory_root(gqa: Path) -> None:
    """Ensure a dataset instance can be created with a regular directory root."""
    GQASpatial(GQAFilemap(gqa))


def test_spatial_invalid_root_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid root type."""
    with pytest.raises(TypeError):
        GQASpatial("gqa")  # type: ignore


def test_spatial_empty_directory_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with an empty root directory."""
    root = tmp_path / "gqa"
    root.mkdir()

    with pytest.raises(ValueError):
        GQASpatial(GQAFilemap(root))


def test_spatial_filemap_property(gqa: Path) -> None:
    """Ensure the `filemap` property returns a correct value."""
    filemap = GQAFilemap(gqa)
    dataset = GQASpatial(filemap)
    assert dataset.filemap == filemap


def test_spatial_getitem(gqa: Path) -> None:
    """Ensure an item is returned given valid GQA data."""
    dataset = GQASpatial(GQAFilemap(gqa))
    spatial = dataset[0]
    assert isinstance(spatial, dict)
    assert "features" in spatial.keys()
    assert spatial["features"].shape == (2048, 7, 7)


def test_spatial_len(gqa: Path) -> None:
    """Ensure the correct dataset length is returned given valid GQA data."""
    dataset = GQASpatial(GQAFilemap(gqa))
    length = len(dataset)
    assert isinstance(length, int)
    assert length == 16


def test_spatial_key_to_index(gqa: Path) -> None:
    """Ensure key_to_index returns the correct index given valid GQA data."""
    dataset = GQASpatial(GQAFilemap(gqa))
    for idx in range(len(dataset)):
        assert isinstance(dataset.key_to_index(str(idx)), int)
    with pytest.raises(KeyError):
        dataset.key_to_index("abc")
    with pytest.raises(KeyError):
        dataset.key_to_index(str(len(dataset)))
