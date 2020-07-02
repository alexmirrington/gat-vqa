"""Tests for the GQA dataset."""
import json
from pathlib import Path

import pytest

from graphgen.config.gqa import GQAFilemap, GQASplit, GQAVersion
from graphgen.datasets.gqa import GQAQuestions

_SPLIT_VERSION_GRID = [(split, version) for split in GQASplit for version in GQAVersion]

_QUESTION_SAMPLE = {
    "02930152": {
        "semantic": [
            {"operation": "select", "dependencies": [], "argument": "sky (2486325)"},
            {"operation": "verify color", "dependencies": [0], "argument": "dark"},
        ],
        "entailed": [
            "02930160",
            "02930158",
            "02930159",
            "02930154",
            "02930155",
            "02930156",
            "02930153",
        ],
        "equivalent": ["02930152"],
        "question": "Is the sky dark?",
        "imageId": "2354786",
        "isBalanced": True,
        "groups": {"global": None, "local": "06-sky_dark"},
        "answer": "yes",
        "semanticStr": "select: sky (2486325)->verify color: dark [0]",
        "annotations": {
            "answer": {},
            "question": {"2": "2486325"},
            "fullAnswer": {"2": "2486325"},
        },
        "types": {"detailed": "verifyAttr", "semantic": "attr", "structural": "verify"},
        "fullAnswer": "Yes, the sky is dark.",
    }
}


@pytest.fixture(scope="session", name="gqa_data")
def fixture_gqa_data(tmp_path_factory):
    """Create a fake GQA dataset in a temporary directory for use in tests."""
    gqa_dir = tmp_path_factory.mktemp("gqa")
    (gqa_dir / "questions").mkdir()
    for split in GQASplit:
        for version in GQAVersion:
            basename = f"{split.value}_{version.value}_questions"
            if split == GQASplit.TRAIN and version == GQAVersion.ALL:
                (gqa_dir / "questions" / basename).mkdir()
                for i in range(10):
                    with (gqa_dir / "questions" / basename / f"{basename}.json").open(
                        "w"
                    ) as file:
                        json.dump(_QUESTION_SAMPLE, file)
            else:
                with (gqa_dir / "questions" / f"{basename}.json").open("w") as file:
                    json.dump(_QUESTION_SAMPLE, file)

    return gqa_dir


def test_questions_nonexistent_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with a non-existent root."""
    root = tmp_path / "gqa"
    with pytest.raises(ValueError):
        GQAQuestions(root, GQASplit.TRAIN, GQAVersion.BALANCED, GQAFilemap())


def test_questions_valid_symlink_root(tmp_path: Path, gqa_data: Path) -> None:
    """Ensure a dataset instance can be created with a symlinked root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(gqa_data)
    GQAQuestions(ln_root, GQASplit.TRAIN, GQAVersion.BALANCED, GQAFilemap())


def test_questions_valid_directory_root(gqa_data: Path) -> None:
    """Ensure a dataset instance can be created with a regular directory root."""
    GQAQuestions(gqa_data, GQASplit.TRAIN, GQAVersion.BALANCED, GQAFilemap())


def test_questions_invalid_root_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid root type."""
    with pytest.raises(TypeError):
        GQAQuestions(
            "gqa", GQASplit.TRAIN, GQAVersion.BALANCED, GQAFilemap()  # type: ignore
        )


def test_questions_invalid_split_type(gqa_data: Path) -> None:
    """Ensure a dataset instance cannot be created with an invalid split type."""
    with pytest.raises(TypeError):
        GQAQuestions(
            gqa_data, "train", GQAVersion.BALANCED, GQAFilemap()  # type: ignore
        )


def test_questions_invalid_version_type(gqa_data: Path) -> None:
    """Ensure a dataset instance cannot be created with an invalid version type."""
    with pytest.raises(TypeError):
        GQAQuestions(gqa_data, GQASplit.TRAIN, "balanced", GQAFilemap())  # type: ignore


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_nonexistent_question_json(
    tmp_path: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure a dataset instance cannot be created with a missing question file."""
    root = tmp_path / "gqa"
    root.mkdir()

    with pytest.raises(ValueError):
        GQAQuestions(root, split, version, GQAFilemap())


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_version_property(
    gqa_data: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `version` property is immutable and returns a correct value."""
    dataset = GQAQuestions(gqa_data, split, version, GQAFilemap())
    assert dataset.version == version


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_split_property(
    gqa_data: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `split` property is immutable and returns a correct value."""
    dataset = GQAQuestions(gqa_data, split, version, GQAFilemap())
    assert dataset.split == split


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_root_property(
    gqa_data: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure the `root` property is immutable and returns a correct value."""
    dataset = GQAQuestions(gqa_data, split, version, GQAFilemap())
    assert dataset.root == gqa_data


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_getitem(
    gqa_data: Path, split: GQASplit, version: GQAVersion
) -> None:
    """Ensure an item is returned given valid GQA data."""
    dataset = GQAQuestions(gqa_data, split, version, GQAFilemap())
    _ = dataset[0]


@pytest.mark.parametrize("split, version", _SPLIT_VERSION_GRID)
def test_questions_len(gqa_data: Path, split: GQASplit, version: GQAVersion) -> None:
    """Ensure an item is returned given valid GQA data."""
    dataset = GQAQuestions(gqa_data, split, version, GQAFilemap())
    length = len(dataset)
    assert isinstance(length, int)
    assert length > 0
