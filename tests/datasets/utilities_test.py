"""Tests for the utility datasets."""
import json
from pathlib import Path

import pytest

from graphgen.datasets.utilities import ChunkedJSONDataset


@pytest.fixture(name="chunked_json_data")
def fixture_chunked_json_data(tmp_path_factory, request):
    """Create a chunked json dataset in a temporary directory for use in tests."""
    # Make root dir
    root = tmp_path_factory.mktemp("data")

    # Set params
    num_chunks = request.param
    chunk_size = 1

    # Seed JSON data
    paths = [root / Path(f"{idx}.json") for idx in range(num_chunks)]
    for chunk_idx, path in enumerate(paths):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        content = {chunk_idx + idx: chunk_idx + idx for idx in range(chunk_size)}
        with path.open("w") as file:
            json.dump(content, file)

    return root


def test_chunkedjson_nonexistent_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with a non-existent root."""
    root = tmp_path / "data"
    with pytest.raises(ValueError):
        ChunkedJSONDataset(root)


def test_chunkedjson_empty_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with an empty root directory."""
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(ValueError):
        ChunkedJSONDataset(root)


def test_chunkedjson_invalid_root_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid root type."""
    with pytest.raises(TypeError):
        ChunkedJSONDataset("data")  # type: ignore


@pytest.mark.parametrize("chunked_json_data", [1, 8], indirect=["chunked_json_data"])
def test_chunkedjson_valid_directory_root(chunked_json_data: Path) -> None:
    """Ensure a dataset instance can be created with a regular directory root."""
    ChunkedJSONDataset(chunked_json_data)


@pytest.mark.parametrize("chunked_json_data", [1, 8], indirect=["chunked_json_data"])
def test_chunkedjson_valid_symlink_directory_root(
    tmp_path: Path, chunked_json_data: Path
) -> None:
    """Ensure a dataset instance can be created with a symlinked directory root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(chunked_json_data)
    ChunkedJSONDataset(ln_root)


@pytest.mark.parametrize(
    "chunked_json_data, num_chunks", [(1, 1)], indirect=["chunked_json_data"]
)
def test_chunkedjson_valid_file_root(chunked_json_data: Path, num_chunks: int) -> None:
    """Ensure a dataset instance can be created with a regular file root."""
    files = list(chunked_json_data.iterdir())
    assert len(files) == num_chunks
    assert files[0].is_file()
    ChunkedJSONDataset(files[0])


@pytest.mark.parametrize(
    "chunked_json_data, num_chunks", [(1, 1)], indirect=["chunked_json_data"]
)
def test_chunkedjson_valid_symlink_file_root(
    tmp_path: Path, chunked_json_data: Path, num_chunks: int
) -> None:
    """Ensure a dataset instance can be created with a symlinked file root."""
    files = list(chunked_json_data.iterdir())
    assert len(files) == num_chunks
    assert files[0].is_file()
    ln_file = tmp_path / "link"
    ln_file.symlink_to(files[0])
    ChunkedJSONDataset(ln_file)


@pytest.mark.parametrize(
    "chunked_json_data, num_chunks", [(1, 1), (8, 8)], indirect=["chunked_json_data"],
)
def test_chunkedjson_directory_root_getitem(
    chunked_json_data: Path, num_chunks: int
) -> None:
    """Ensure correct items are returned for each chunk given chunked json data."""
    # Create dataset
    dataset = ChunkedJSONDataset(chunked_json_data)
    chunk_size = 1

    # Test __getitem__ on chunk boundaries
    chunk_start = 0
    for chunk_idx in range(num_chunks):
        first = dataset[chunk_start]
        assert first == chunk_start
        last = dataset[chunk_start + chunk_size - 1]
        assert last == chunk_start + chunk_size - 1
        chunk_start += chunk_size


@pytest.mark.parametrize("chunked_json_data", [1], indirect=["chunked_json_data"])
def test_chunkedjson_file_root_getitem(chunked_json_data: Path) -> None:
    """Ensure correct items are returned for each chunk given a single JSON file."""
    # Get single file root
    files = list(chunked_json_data.iterdir())
    assert len(files) == 1
    assert files[0].is_file()

    # Create dataset
    dataset = ChunkedJSONDataset(files[0])
    chunk_size = 1

    # Test __getitem__ on chunk boundaries
    first = dataset[0]
    assert first == 0
    last = dataset[chunk_size - 1]
    assert last == chunk_size - 1


@pytest.mark.parametrize(
    "chunked_json_data, num_chunks", [(1, 1), (8, 8)], indirect=["chunked_json_data"],
)
def test_chunkedjson_directory_root_len(
    chunked_json_data: Path, num_chunks: int
) -> None:
    """Ensure dataset length is correct given chunked json data."""
    dataset = ChunkedJSONDataset(chunked_json_data)
    chunk_size = 1
    assert len(dataset) == num_chunks * chunk_size


@pytest.mark.parametrize("chunked_json_data", [1], indirect=["chunked_json_data"])
def test_chunkedjson_file_root_len(chunked_json_data: Path) -> None:
    """Ensure dataset length is correct given a single JSON file."""
    # Get single file root
    files = list(chunked_json_data.iterdir())
    assert len(files) == 1
    assert files[0].is_file()

    # Create dataset
    dataset = ChunkedJSONDataset(files[0])
    chunk_size = 1

    assert len(dataset) == chunk_size
