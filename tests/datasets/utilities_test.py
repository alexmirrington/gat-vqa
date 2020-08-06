"""Tests for the utility datasets."""
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest

from graphgen.datasets.utilities import ChunkedHDF5Dataset, ChunkedJSONDataset


@dataclass
class ChunkedDataConfig:
    """Class that stores information about chunks for use in fixture requests."""

    num_chunks: int = 1
    chunk_size: int = 1


@pytest.fixture(name="chunked_json_data")
def fixture_chunked_json_data(tmp_path_factory, request):
    """Create a chunked json dataset in a temporary directory for use in tests."""
    # Make root dir
    root = tmp_path_factory.mktemp("data")

    # Set params
    num_chunks = request.param.num_chunks
    chunk_size = request.param.chunk_size

    # Seed JSON data
    paths = [root / Path(f"{idx}.json") for idx in range(num_chunks)]
    for chunk_idx, path in enumerate(paths):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        content = {chunk_idx + idx: chunk_idx + idx for idx in range(chunk_size)}
        with path.open("w") as file:
            json.dump(content, file)

    return root


@pytest.fixture(name="chunked_hdf5_data")
def fixture_chunked_hdf5_data(tmp_path_factory, request):
    """Create a chunked hdf5 dataset in a temporary directory for use in tests."""
    # Make root dir
    root = tmp_path_factory.mktemp("data")

    # Set params
    num_chunks = request.param.num_chunks
    chunk_size = request.param.chunk_size
    data_shape = (1,)

    # Seed hdf5 data
    paths = [root / Path(f"{idx}.h5") for idx in range(num_chunks)]
    for chunk_idx, path in enumerate(paths):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        with h5py.File(path, "w") as file:
            file.create_dataset(
                "zeros", data=np.zeros((chunk_size,) + data_shape, dtype=np.int)
            )
            file.create_dataset(
                "ones", data=np.ones((chunk_size,) + data_shape, dtype=np.int)
            )

    return root


# region ChunkedJSONDataset


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
    """Ensure dataset length is correct given chunked json data."""
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


# endregion ChunkedJSONDataset


# region ChunkedHDF5Dataset


def test_chunkedhdf5_nonexistent_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with a non-existent root."""
    root = tmp_path / "data"
    with pytest.raises(ValueError):
        ChunkedHDF5Dataset(root)


def test_chunkedhdf5_empty_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with an empty root directory."""
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(ValueError):
        ChunkedHDF5Dataset(root)


def test_chunkedhdf5_invalid_root_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid root type."""
    with pytest.raises(TypeError):
        ChunkedHDF5Dataset("data")  # type: ignore


@pytest.mark.parametrize(
    "chunked_hdf5_data",
    [ChunkedDataConfig(1), ChunkedDataConfig(8)],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_valid_directory_root(chunked_hdf5_data: Path) -> None:
    """Ensure a dataset instance can be created with a regular directory root."""
    ChunkedHDF5Dataset(chunked_hdf5_data)


@pytest.mark.parametrize(
    "chunked_hdf5_data",
    [ChunkedDataConfig(1), ChunkedDataConfig(8)],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_valid_symlink_directory_root(
    tmp_path: Path, chunked_hdf5_data: Path
) -> None:
    """Ensure a dataset instance can be created with a symlinked directory root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(chunked_hdf5_data)
    ChunkedHDF5Dataset(ln_root)


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [(ChunkedDataConfig(1), ChunkedDataConfig(1))],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_valid_file_root(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure a dataset instance can be created with a regular file root."""
    files = list(chunked_hdf5_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()
    ChunkedHDF5Dataset(files[0])


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [(ChunkedDataConfig(1), ChunkedDataConfig(1))],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_valid_symlink_file_root(
    tmp_path: Path, chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure a dataset instance can be created with a symlinked file root."""
    files = list(chunked_hdf5_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()
    ln_file = tmp_path / "link"
    ln_file.symlink_to(files[0])
    ChunkedHDF5Dataset(ln_file)


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [
        (ChunkedDataConfig(1), ChunkedDataConfig(1)),
        (ChunkedDataConfig(8), ChunkedDataConfig(8)),
    ],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_directory_root_getitem(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure correct items are returned for each chunk given chunked HDF5 data."""
    # Create dataset
    dataset = ChunkedHDF5Dataset(chunked_hdf5_data)
    chunk_size = chunk_config.chunk_size

    # Test __getitem__ on chunk boundaries
    chunk_start = 0
    for chunk_idx in range(chunk_config.num_chunks):
        first = dataset[chunk_start]
        assert isinstance(first, dict)
        last = dataset[chunk_start + chunk_size - 1]
        assert isinstance(last, dict)
        for key in set(list(first.keys()) + list(last.keys())):
            assert key in first.keys()
            assert key in last.keys()
            assert first[key].shape == last[key].shape
        chunk_start += chunk_size


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [(ChunkedDataConfig(1), ChunkedDataConfig(1))],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_file_root_getitem(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure correct items are returned for each chunk given a single HDF5 file."""
    # Get single file root
    files = list(chunked_hdf5_data.iterdir())
    assert len(files) == 1
    assert files[0].is_file()

    # Create dataset
    dataset = ChunkedHDF5Dataset(files[0])
    chunk_size = chunk_config.chunk_size

    # Test __getitem__ on chunk boundaries
    first = dataset[0]
    assert isinstance(first, dict)
    last = dataset[chunk_size - 1]
    assert isinstance(last, dict)
    for key in set(list(first.keys()) + list(last.keys())):
        assert key in first.keys()
        assert key in last.keys()
        assert first[key].shape == last[key].shape


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [
        (ChunkedDataConfig(1), ChunkedDataConfig(1)),
        (ChunkedDataConfig(8), ChunkedDataConfig(8)),
    ],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_directory_root_len(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure dataset length is correct given chunked HDF5 data."""
    dataset = ChunkedHDF5Dataset(chunked_hdf5_data)
    assert len(dataset) == chunk_config.num_chunks * chunk_config.chunk_size


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [(ChunkedDataConfig(1), ChunkedDataConfig(1))],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_file_root_len(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure dataset length is correct given a single HDF5 file."""
    # Get single file root
    files = list(chunked_hdf5_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()

    # Create dataset
    dataset = ChunkedHDF5Dataset(files[0])

    assert len(dataset) == chunk_config.chunk_size


# endregion ChunkedHDF5Dataset
