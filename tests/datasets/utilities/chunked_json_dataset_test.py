"""Tests for the chunked json datasets."""
from pathlib import Path

import pytest

from graphgen.datasets.utilities import ChunkedJSONDataset

from .conftest import ChunkedDataConfig


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


@pytest.mark.parametrize(
    "chunked_json_data",
    [ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=8)],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_valid_directory_root(chunked_json_data: Path) -> None:
    """Ensure a dataset instance can be created with a regular directory root."""
    ChunkedJSONDataset(chunked_json_data)


@pytest.mark.parametrize(
    "chunked_json_data",
    [ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=8)],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_valid_symlink_directory_root(
    tmp_path: Path, chunked_json_data: Path
) -> None:
    """Ensure a dataset instance can be created with a symlinked directory root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(chunked_json_data)
    ChunkedJSONDataset(ln_root)


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_valid_file_root(
    chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure a dataset instance can be created with a regular file root."""
    files = list(chunked_json_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()
    ChunkedJSONDataset(files[0])


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_valid_symlink_file_root(
    tmp_path: Path, chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure a dataset instance can be created with a symlinked file root."""
    files = list(chunked_json_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()
    ln_file = tmp_path / "link"
    ln_file.symlink_to(files[0])
    ChunkedJSONDataset(ln_file)


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [
        (ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1)),
        (ChunkedDataConfig(num_chunks=8), ChunkedDataConfig(num_chunks=8)),
    ],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_directory_root_getitem(
    chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure correct items are returned for each chunk given chunked json data."""
    # Create dataset
    dataset = ChunkedJSONDataset(chunked_json_data)
    chunk_size = chunk_config.chunk_size

    # Test __getitem__ on chunk boundaries
    chunk_start = 0
    for chunk_idx in range(chunk_config.num_chunks):
        first = dataset[chunk_start]
        assert first == chunk_start
        last = dataset[chunk_start + chunk_size - 1]
        assert last == chunk_start + chunk_size - 1
        chunk_start += chunk_size

    # Test invalid index in __getitem__
    with pytest.raises(IndexError):
        _ = dataset[chunk_start]


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_file_root_getitem(
    chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure correct items are returned for each chunk given a single JSON file."""
    # Get single file root
    files = list(chunked_json_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()

    # Create dataset
    dataset = ChunkedJSONDataset(files[0])
    chunk_size = chunk_config.chunk_size

    # Test __getitem__ on chunk boundaries
    first = dataset[0]
    assert first == 0
    last = dataset[chunk_size - 1]
    assert last == chunk_size - 1

    # Test invalid index in __getitem__
    with pytest.raises(IndexError):
        _ = dataset[chunk_size]


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [
        (ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1)),
        (ChunkedDataConfig(num_chunks=8), ChunkedDataConfig(num_chunks=8)),
    ],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_directory_root_len(
    chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure dataset length is correct given chunked JSON data."""
    dataset = ChunkedJSONDataset(chunked_json_data)
    assert len(dataset) == chunk_config.num_chunks * chunk_config.chunk_size


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_file_root_len(
    chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure dataset length is correct given a single JSON file."""
    # Get single file root
    files = list(chunked_json_data.iterdir())
    assert len(files) == chunk_config.chunk_size
    assert files[0].is_file()

    # Create dataset
    dataset = ChunkedJSONDataset(files[0])

    assert len(dataset) == chunk_config.chunk_size


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [
        (ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1)),
        (ChunkedDataConfig(num_chunks=8), ChunkedDataConfig(num_chunks=8)),
    ],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_directory_root_key_to_index(
    chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure key_to_index returns correct indices for chunked JSON data."""
    dataset = ChunkedJSONDataset(chunked_json_data)
    for idx in range(chunk_config.num_chunks * chunk_config.chunk_size):
        assert dataset.key_to_index(str(idx)) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("abc")
    with pytest.raises(KeyError):
        dataset.key_to_index("-1")
    with pytest.raises(KeyError):
        dataset.key_to_index(str(chunk_config.num_chunks * chunk_config.chunk_size))


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_file_root_key_to_index(
    chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure key_to_index returns correct indices given a single JSON file."""
    # Get single file root
    files = list(chunked_json_data.iterdir())
    assert len(files) == chunk_config.chunk_size
    assert files[0].is_file()

    # Create dataset
    dataset = ChunkedJSONDataset(files[0])
    for idx in range(chunk_config.num_chunks * chunk_config.chunk_size):
        assert dataset.key_to_index(str(idx)) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("abc")
    with pytest.raises(KeyError):
        dataset.key_to_index("-1")
    with pytest.raises(KeyError):
        dataset.key_to_index(str(chunk_config.num_chunks * chunk_config.chunk_size))


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [
        (ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1)),
        (ChunkedDataConfig(num_chunks=8), ChunkedDataConfig(num_chunks=8)),
    ],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_chunks_property_directory_root(
    chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure the `chunks` and `chunk_sizes` properties return correct values \
    when created with a regular directory root."""
    dataset = ChunkedJSONDataset(chunked_json_data)
    assert dataset.chunks == tuple(sorted(chunked_json_data.iterdir()))
    assert dataset.chunk_sizes == tuple([1] * chunk_config.num_chunks)


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [
        (ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1)),
        (ChunkedDataConfig(num_chunks=8), ChunkedDataConfig(num_chunks=8)),
    ],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_chunks_property_symlink_directory_root(
    tmp_path: Path, chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure the `chunks` and `chunk_sizes` properties return correct values \
    when created with a symlinked directory root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(chunked_json_data)
    dataset = ChunkedJSONDataset(ln_root)
    assert dataset.chunks == tuple(sorted(ln_root.iterdir()))
    assert tuple([p.resolve() for p in dataset.chunks]) == tuple(
        sorted(chunked_json_data.iterdir())
    )
    assert dataset.chunk_sizes == tuple([1] * chunk_config.num_chunks)


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_chunks_property_file_root(
    chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure the `chunks` and `chunk_sizes` properties return correct values \
    when created with a regular file root."""
    files = list(chunked_json_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()
    dataset = ChunkedJSONDataset(files[0])
    assert dataset.chunks == (files[0],)
    assert dataset.chunk_sizes == tuple([1] * chunk_config.num_chunks)


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_chunks_property_symlinked_file_root(
    tmp_path: Path, chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure the `chunks` and `chunk_sizes` properties return correct values \
    when created with a symlinked file root."""
    files = list(chunked_json_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()
    ln_file = tmp_path / "link"
    ln_file.symlink_to(files[0])
    dataset = ChunkedJSONDataset(ln_file)
    assert dataset.chunks == (ln_file,)
    assert tuple([p.resolve() for p in dataset.chunks]) == (files[0],)
    assert dataset.chunk_sizes == tuple([1] * chunk_config.num_chunks)


@pytest.mark.parametrize(
    "chunked_json_data",
    [ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=8)],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_root_property_directory_root(chunked_json_data: Path) -> None:
    """Ensure the root property returns a correct value when created with a \
    regular directory root."""
    dataset = ChunkedJSONDataset(chunked_json_data)
    assert dataset.root == chunked_json_data


@pytest.mark.parametrize(
    "chunked_json_data",
    [ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=8)],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_root_property_symlink_directory_root(
    tmp_path: Path, chunked_json_data: Path
) -> None:
    """Ensure the root property returns a correct value when created with a \
    symlinked directory root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(chunked_json_data)
    dataset = ChunkedJSONDataset(ln_root)
    assert dataset.root == ln_root
    assert dataset.root.resolve() == chunked_json_data


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_root_property_file_root(
    chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure the root property returns a correct value when created with a \
    regular file root."""
    files = list(chunked_json_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()
    dataset = ChunkedJSONDataset(files[0])
    assert dataset.root == files[0]


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_root_property_symlinked_file_root(
    tmp_path: Path, chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure the root property returns a correct value when created with a \
    symlinked file root."""
    files = list(chunked_json_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()
    ln_file = tmp_path / "link"
    ln_file.symlink_to(files[0])
    dataset = ChunkedJSONDataset(ln_file)
    assert dataset.root == ln_file
    assert dataset.root.resolve() == files[0]


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [
        (ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1)),
        (ChunkedDataConfig(num_chunks=8), ChunkedDataConfig(num_chunks=8)),
    ],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_directory_root_keys(
    chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure key_to_index returns correct indices for chunked JSON data."""
    dataset = ChunkedJSONDataset(chunked_json_data)
    for idx, key in enumerate(dataset.keys()):
        assert key == str(idx)
    assert (
        len(tuple(dataset.keys())) == chunk_config.num_chunks * chunk_config.chunk_size
    )


@pytest.mark.parametrize(
    "chunked_json_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_json_data"],
)
def test_chunkedjson_file_root_keys(
    chunked_json_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure key_to_index returns correct indices given a single JSON file."""
    # Get single file root
    files = list(chunked_json_data.iterdir())
    assert len(files) == chunk_config.chunk_size
    assert files[0].is_file()

    # Create dataset
    dataset = ChunkedJSONDataset(files[0])
    for idx, key in enumerate(dataset.keys()):
        assert key == str(idx)
    assert (
        len(tuple(dataset.keys())) == chunk_config.num_chunks * chunk_config.chunk_size
    )
