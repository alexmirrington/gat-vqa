"""Tests for the GQA questions dataset."""
from pathlib import Path

import pytest

from graphgen.config.gqa import GQAFilemap, GQASplit, GQAVersion
from graphgen.datasets.gqa.questions import GQAQuestions

_SPLIT_VERSION_GRID = [(split, version) for split in GQASplit for version in GQAVersion]


def test_questions_nonexistent_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with a non-existent root."""
    root = tmp_path / "gqa"
    with pytest.raises(ValueError):
        GQAQuestions(GQAFilemap(root), GQASplit.TRAIN, GQAVersion.BALANCED)


def test_questions_valid_symlink_root(tmp_path: Path, gqa: Path) -> None:
    """Ensure a dataset instance can be created with a symlinked root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(gqa)
    GQAQuestions(GQAFilemap(ln_root), GQASplit.TRAIN, GQAVersion.BALANCED)


def test_questions_valid_directory_root(gqa: Path) -> None:
    """Ensure a dataset instance can be created with a regular directory root."""
    GQAQuestions(GQAFilemap(gqa), GQASplit.TRAIN, GQAVersion.BALANCED)


def test_questions_invalid_root_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid root type."""
    with pytest.raises(TypeError):
        GQAQuestions("gqa", GQASplit.TRAIN, GQAVersion.BALANCED)  # type: ignore


def test_questions_invalid_split_type(gqa: Path) -> None:
    """Ensure a dataset instance cannot be created with an invalid split type."""
    with pytest.raises(TypeError):
        GQAQuestions(GQAFilemap(gqa), "train", GQAVersion.BALANCED)  # type: ignore


def test_questions_invalid_version_type(gqa: Path) -> None:
    """Ensure a dataset instance cannot be created with an invalid version type."""
    with pytest.raises(TypeError):
        GQAQuestions(GQAFilemap(gqa), GQASplit.TRAIN, "balanced")  # type: ignore


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
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `version` property returns a correct value."""
    dataset = GQAQuestions(GQAFilemap(gqa), split, version)
    assert dataset.version == version


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_split_property(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `split` property returns a correct value."""
    dataset = GQAQuestions(GQAFilemap(gqa), split, version)
    assert dataset.split == split


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_filemap_property(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `filemap` property returns a correct value."""
    filemap = GQAFilemap(gqa)
    dataset = GQAQuestions(filemap, split, version)
    assert dataset.filemap == filemap


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_getitem(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure an item is returned given valid GQA data."""
    dataset = GQAQuestions(GQAFilemap(gqa), split, version)
    question = dataset[0]
    assert isinstance(question, dict)


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_len(gqa: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure the correct dataset length is returned given valid GQA data."""
    dataset = GQAQuestions(GQAFilemap(gqa), split, version)
    length = len(dataset)
    assert isinstance(length, int)
    assert length == 1


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_key_to_index(
    gqa: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure key_to_index returns the correct index given valid GQA data."""
    dataset = GQAQuestions(GQAFilemap(gqa), split, version)

    for idx in range(len(dataset)):
        assert dataset.key_to_index(str(idx)) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("abc")
