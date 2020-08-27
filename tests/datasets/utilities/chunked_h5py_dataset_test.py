"""Tests for chunked h5py datasets."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from graphgen.datasets.utilities import ChunkedHDF5Dataset

from .conftest import ChunkedDataConfig


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


def test_chunkedhdf5_datasets_nonequal_length(tmp_path: Path) -> None:
    """Ensure dataset creation fails if a chunk has two datasets of nonequal length."""
    # Make root dir
    root = tmp_path / "data"
    root.mkdir()

    # Seed hdf5 data
    with h5py.File(root / Path("0.h5"), "w") as file:
        file.create_dataset("zeros", data=np.zeros((1, 2), dtype=np.int))
        file.create_dataset("ones", data=np.ones((2, 2), dtype=np.int))

    # Create dataset
    with pytest.raises(ValueError):
        ChunkedHDF5Dataset(root)


def test_chunkedhdf5_datasets_different_keys(tmp_path: Path) -> None:
    """Ensure dataset creation fails if a two chunks have datasets with \
    differing keys."""
    # Make root dir
    root = tmp_path / "data"
    root.mkdir()

    # Seed hdf5 data
    with h5py.File(root / Path("0.h5"), "w") as file:
        file.create_dataset("zeros", data=np.zeros((1, 1), dtype=np.int))

    with h5py.File(root / Path("1.h5"), "w") as file:
        file.create_dataset("ones", data=np.ones((1, 1), dtype=np.int))

    # Create dataset
    with pytest.raises(ValueError):
        ChunkedHDF5Dataset(root)


def test_chunkedhdf5_datasets_nonequal_shape(tmp_path: Path) -> None:
    """Ensure dataset creation fails if a two chunks have datasets with the \
    same key but differing shapes."""
    # Make root dir
    root = tmp_path / "data"
    root.mkdir()
    # Seed hdf5 data
    with h5py.File(root / Path("0.h5"), "w") as file:
        file.create_dataset("zeros", data=np.zeros((1, 1), dtype=np.int))

    with h5py.File(root / Path("1.h5"), "w") as file:
        file.create_dataset("zeros", data=np.zeros((1, 2), dtype=np.int))
    # Create dataset
    with pytest.raises(ValueError):
        ChunkedHDF5Dataset(root)


def test_chunkedhdf5_datasets_equal_shape_nonequal_length(tmp_path: Path) -> None:
    """Ensure dataset creation fails if two chunks have datasets with the same \
    key and shape but different lengths."""
    # Make root dir
    root = tmp_path / "data"
    root.mkdir()
    # Seed hdf5 data
    with h5py.File(root / Path("0.h5"), "w") as file:
        file.create_dataset("zeros", data=np.zeros((1, 1), dtype=np.int))

    with h5py.File(root / Path("1.h5"), "w") as file:
        file.create_dataset("zeros", data=np.zeros((2, 1), dtype=np.int))
    # Create dataset
    ChunkedHDF5Dataset(root)


@pytest.mark.parametrize(
    "chunked_json_data",
    [ChunkedDataConfig(1)],
    indirect=["chunked_json_data"],
)
def test_chunkedhdf5_directory_root_ignore_bad_files(chunked_json_data: Path) -> None:
    """Ensure a dataset can be created if a two chunks have datasets with the \
    same key and shape but different lengths."""
    # Create dataset
    with pytest.raises(ValueError):
        ChunkedHDF5Dataset(chunked_json_data)


@pytest.mark.parametrize(
    "chunked_json_data",
    [ChunkedDataConfig(1)],
    indirect=["chunked_json_data"],
)
def test_chunkedhdf5_file_root_ignore_bad_files(chunked_json_data: Path) -> None:
    """Ensure a dataset can be created if a two chunks have datasets with the \
    same key and shape but different lengths."""
    files = list(chunked_json_data.iterdir())
    assert len(files) == 1
    assert files[0].is_file()

    # Create dataset
    with pytest.raises(ValueError):
        ChunkedHDF5Dataset(files[0])


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [
        (ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1)),
        (ChunkedDataConfig(num_chunks=8), ChunkedDataConfig(num_chunks=8)),
    ],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_directory_root_key_to_index(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure key_to_index returns correct indices given chunked HDF5 data."""
    dataset = ChunkedHDF5Dataset(chunked_hdf5_data)
    for idx in range(chunk_config.num_chunks * chunk_config.chunk_size):
        assert dataset.key_to_index(str(idx)) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("abc")
    with pytest.raises(KeyError):
        dataset.key_to_index("-1")
    with pytest.raises(KeyError):
        dataset.key_to_index(str(chunk_config.num_chunks * chunk_config.chunk_size))


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_file_root_key_to_index(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure key_to_index returns correct indices given a single HDF5 file."""
    # Get single file root
    files = list(chunked_hdf5_data.iterdir())
    assert len(files) == chunk_config.chunk_size
    assert files[0].is_file()

    # Create dataset
    dataset = ChunkedHDF5Dataset(files[0])
    for idx in range(chunk_config.num_chunks * chunk_config.chunk_size):
        assert dataset.key_to_index(str(idx)) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("abc")
    with pytest.raises(KeyError):
        dataset.key_to_index("-1")
    with pytest.raises(KeyError):
        dataset.key_to_index(str(chunk_config.num_chunks * chunk_config.chunk_size))


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [
        (ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1)),
        (ChunkedDataConfig(num_chunks=8), ChunkedDataConfig(num_chunks=8)),
    ],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_directory_root_chunk_mapped_key_to_index(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure key_to_index returns correct indices given chunked HDF5 data."""
    chunk_paths = list(sorted(chunked_hdf5_data.iterdir()))
    chunk_map = {
        str(hex(idx)): (
            chunk_paths[idx // chunk_config.chunk_size],
            idx % chunk_config.chunk_size,
        )
        for idx in range(chunk_config.num_chunks * chunk_config.chunk_size)
    }
    dataset = ChunkedHDF5Dataset(chunked_hdf5_data, chunk_map)

    for idx in range(chunk_config.num_chunks * chunk_config.chunk_size):
        assert dataset.key_to_index(str(hex(idx))) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("www")
    with pytest.raises(KeyError):
        dataset.key_to_index("-1")
    with pytest.raises(KeyError):
        dataset.key_to_index(
            str(hex(chunk_config.num_chunks * chunk_config.chunk_size))
        )


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_file_root_chunk_mapped_key_to_index(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure key_to_index returns correct indices given a single HDF5 file."""
    # Get single file root
    files = list(chunked_hdf5_data.iterdir())
    assert len(files) == chunk_config.chunk_size
    assert files[0].is_file()
    chunk_map = {
        str(hex(idx)): (files[0], idx) for idx in range(chunk_config.chunk_size)
    }

    # Create dataset
    dataset = ChunkedHDF5Dataset(files[0], chunk_map)
    for idx in range(chunk_config.num_chunks * chunk_config.chunk_size):
        assert dataset.key_to_index(str(hex(idx))) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("www")
    with pytest.raises(KeyError):
        dataset.key_to_index("-1")
    with pytest.raises(KeyError):
        dataset.key_to_index(
            str(hex(chunk_config.num_chunks * chunk_config.chunk_size))
        )


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [
        (ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1)),
        (ChunkedDataConfig(num_chunks=8), ChunkedDataConfig(num_chunks=8)),
    ],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_chunks_property_directory_root(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure the `chunks` and `chunk_sizes` properties return correct values \
    when created with a regular directory root."""
    dataset = ChunkedHDF5Dataset(chunked_hdf5_data)
    assert dataset.chunks == tuple(sorted(chunked_hdf5_data.iterdir()))
    assert dataset.chunk_sizes == tuple([1] * chunk_config.num_chunks)


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [
        (ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1)),
        (ChunkedDataConfig(num_chunks=8), ChunkedDataConfig(num_chunks=8)),
    ],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_chunks_property_symlink_directory_root(
    tmp_path: Path, chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure the `chunks` and `chunk_sizes` properties return correct values \
    when created with a symlinked directory root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(chunked_hdf5_data)
    dataset = ChunkedHDF5Dataset(ln_root)
    assert dataset.chunks == tuple(sorted(ln_root.iterdir()))
    assert tuple([p.resolve() for p in dataset.chunks]) == tuple(
        sorted(chunked_hdf5_data.iterdir())
    )
    assert dataset.chunk_sizes == tuple([1] * chunk_config.num_chunks)


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_chunks_property_file_root(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure the `chunks` and `chunk_sizes` properties return correct values \
    when created with a regular file root."""
    files = list(chunked_hdf5_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()
    dataset = ChunkedHDF5Dataset(files[0])
    assert dataset.chunks == (files[0],)
    assert dataset.chunk_sizes == tuple([1] * chunk_config.num_chunks)


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_chunks_property_symlinked_file_root(
    tmp_path: Path, chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure the `chunks` and `chunk_sizes` properties return correct values \
    when created with a symlinked file root."""
    files = list(chunked_hdf5_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()
    ln_file = tmp_path / "link"
    ln_file.symlink_to(files[0])
    dataset = ChunkedHDF5Dataset(ln_file)
    assert dataset.chunks == (ln_file,)
    assert tuple([p.resolve() for p in dataset.chunks]) == (files[0],)
    assert dataset.chunk_sizes == tuple([1] * chunk_config.num_chunks)


@pytest.mark.parametrize(
    "chunked_hdf5_data",
    [ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=8)],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_root_property_directory_root(chunked_hdf5_data: Path) -> None:
    """Ensure the root property returns a correct value when created with a \
    regular directory root."""
    dataset = ChunkedHDF5Dataset(chunked_hdf5_data)
    assert dataset.root == chunked_hdf5_data


@pytest.mark.parametrize(
    "chunked_hdf5_data",
    [ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=8)],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_root_property_symlink_directory_root(
    tmp_path: Path, chunked_hdf5_data: Path
) -> None:
    """Ensure the root property returns a correct value when created with a \
    symlinked directory root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(chunked_hdf5_data)
    dataset = ChunkedHDF5Dataset(ln_root)
    assert dataset.root == ln_root
    assert dataset.root.resolve() == chunked_hdf5_data


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_root_property_file_root(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure the root property returns a correct value when created with a \
    regular file root."""
    files = list(chunked_hdf5_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()
    dataset = ChunkedHDF5Dataset(files[0])
    assert dataset.root == files[0]


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_root_property_symlinked_file_root(
    tmp_path: Path, chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure the root property returns a correct value when created with a \
    symlinked file root."""
    files = list(chunked_hdf5_data.iterdir())
    assert len(files) == chunk_config.num_chunks
    assert files[0].is_file()
    ln_file = tmp_path / "link"
    ln_file.symlink_to(files[0])
    dataset = ChunkedHDF5Dataset(ln_file)
    assert dataset.root == ln_file
    assert dataset.root.resolve() == files[0]


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [
        (ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1)),
        (ChunkedDataConfig(num_chunks=8), ChunkedDataConfig(num_chunks=8)),
    ],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_directory_root_keys(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure key_to_index returns correct indices given chunked HDF5 data."""
    dataset = ChunkedHDF5Dataset(chunked_hdf5_data)
    for idx, key in enumerate(dataset.keys()):
        assert key == str(idx)
    assert (
        len(tuple(dataset.keys())) == chunk_config.num_chunks * chunk_config.chunk_size
    )


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_file_root_keys(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure key_to_index returns correct indices given a single HDF5 file."""
    # Get single file root
    files = list(chunked_hdf5_data.iterdir())
    assert len(files) == chunk_config.chunk_size
    assert files[0].is_file()

    # Create dataset
    dataset = ChunkedHDF5Dataset(files[0])
    for idx, key in enumerate(dataset.keys()):
        assert key == str(idx)
    assert (
        len(tuple(dataset.keys())) == chunk_config.num_chunks * chunk_config.chunk_size
    )


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [
        (ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1)),
        (ChunkedDataConfig(num_chunks=8), ChunkedDataConfig(num_chunks=8)),
    ],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_directory_root_chunk_mapped_keys(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure key_to_index returns correct indices given chunked HDF5 data."""
    chunk_paths = list(sorted(chunked_hdf5_data.iterdir()))
    chunk_map = {
        str(hex(idx)): (
            chunk_paths[idx // chunk_config.chunk_size],
            idx % chunk_config.chunk_size,
        )
        for idx in range(chunk_config.num_chunks * chunk_config.chunk_size)
    }

    dataset = ChunkedHDF5Dataset(chunked_hdf5_data, chunk_map)
    for idx, key in enumerate(dataset.keys()):
        assert key == str(hex(idx))
    assert (
        len(tuple(dataset.keys())) == chunk_config.num_chunks * chunk_config.chunk_size
    )


@pytest.mark.parametrize(
    "chunked_hdf5_data, chunk_config",
    [(ChunkedDataConfig(num_chunks=1), ChunkedDataConfig(num_chunks=1))],
    indirect=["chunked_hdf5_data"],
)
def test_chunkedhdf5_file_root_chunk_mapped_keys(
    chunked_hdf5_data: Path, chunk_config: ChunkedDataConfig
) -> None:
    """Ensure key_to_index returns correct indices given a single HDF5 file."""
    # Get single file root
    files = list(chunked_hdf5_data.iterdir())
    assert len(files) == chunk_config.chunk_size
    assert files[0].is_file()
    chunk_map = {
        str(hex(idx)): (files[0], idx) for idx in range(chunk_config.chunk_size)
    }

    # Create dataset
    dataset = ChunkedHDF5Dataset(files[0], chunk_map)
    for idx, key in enumerate(dataset.keys()):
        assert key == str(hex(idx))
    assert (
        len(tuple(dataset.keys())) == chunk_config.num_chunks * chunk_config.chunk_size
    )
