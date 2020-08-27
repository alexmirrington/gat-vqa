"""Tests for image folder datasets."""
from pathlib import Path

import pytest
import torch
from torch import Tensor

from graphgen.datasets.utilities import ImageFolderDataset


def test_imagefolder_nonexistent_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with a non-existent root."""
    root = tmp_path / "data"
    with pytest.raises(ValueError):
        ImageFolderDataset(root)


def test_imagefolder_empty_root(tmp_path: Path) -> None:
    """Ensure a dataset instance cannot be created with an empty root directory."""
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(ValueError):
        ImageFolderDataset(root)


def test_imagefolder_invalid_root_type() -> None:
    """Ensure a dataset instance cannot be created with an invalid root type."""
    with pytest.raises(TypeError):
        ImageFolderDataset("data")  # type: ignore


@pytest.mark.parametrize(
    "image_data",
    [1, 8],
    indirect=["image_data"],
)
def test_imagefolder_valid_directory_root(image_data: Path) -> None:
    """Ensure a dataset instance can be created with a regular directory root."""
    ImageFolderDataset(image_data)


@pytest.mark.parametrize(
    "image_data",
    [1, 8],
    indirect=["image_data"],
)
def test_imagefolder_valid_symlink_directory_root(
    tmp_path: Path, image_data: Path
) -> None:
    """Ensure a dataset instance can be created with a symlinked directory root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(image_data)
    ImageFolderDataset(ln_root)


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1)],
    indirect=["image_data"],
)
def test_imagefolder_valid_file_root(image_data: Path, num_images: int) -> None:
    """Ensure a dataset instance can be created with a regular file root."""
    files = list(image_data.iterdir())
    assert len(files) == num_images
    assert files[0].is_file()
    ImageFolderDataset(files[0])


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1)],
    indirect=["image_data"],
)
def test_imagefolder_valid_symlink_file_root(
    tmp_path: Path, image_data: Path, num_images: int
) -> None:
    """Ensure a dataset instance can be created with a symlinked file root."""
    files = list(image_data.iterdir())
    assert len(files) == num_images
    assert files[0].is_file()
    ln_file = tmp_path / "link"
    ln_file.symlink_to(files[0])
    ImageFolderDataset(ln_file)


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1), (8, 8)],
    indirect=["image_data"],
)
def test_imagefolder_directory_root_getitem(image_data: Path, num_images: int) -> None:
    """Ensure correct items are returned for each chunk given chunked image data."""
    # Create dataset
    dataset = ImageFolderDataset(image_data)

    # Test __getitem__
    first = dataset[0]
    assert isinstance(first, Tensor)

    last = dataset[num_images - 1]
    assert isinstance(last, Tensor)

    # Test invalid index in __getitem__
    with pytest.raises(IndexError):
        _ = dataset[num_images]


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1)],
    indirect=["image_data"],
)
def test_imagefolder_file_root_getitem(image_data: Path, num_images: int) -> None:
    """Ensure correct items are returned for each chunk given a single image file."""
    # Get single file root
    files = list(image_data.iterdir())
    assert len(files) == num_images
    assert files[0].is_file()

    # Create dataset
    dataset = ImageFolderDataset(files[0])

    # Test __getitem__ on chunk boundaries
    first = dataset[0]
    assert isinstance(first, Tensor)

    last = dataset[num_images - 1]
    assert isinstance(last, Tensor)

    assert torch.all(first == last)

    # Test invalid index in __getitem__
    with pytest.raises(IndexError):
        _ = dataset[num_images]


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1), (8, 8)],
    indirect=["image_data"],
)
def test_imagefolder_directory_root_len(image_data: Path, num_images: int) -> None:
    """Ensure dataset length is correct given chunked image data."""
    dataset = ImageFolderDataset(image_data)
    assert len(dataset) == num_images


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1)],
    indirect=["image_data"],
)
def test_imagefolder_file_root_len(image_data: Path, num_images: int) -> None:
    """Ensure dataset length is correct given a single image file."""
    # Get single file root
    files = list(image_data.iterdir())
    assert len(files) == num_images
    assert files[0].is_file()

    # Create dataset
    dataset = ImageFolderDataset(files[0])

    assert len(dataset) == num_images


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1), (8, 8)],
    indirect=["image_data"],
)
def test_imagefolder_directory_root_key_to_index(
    image_data: Path, num_images: int
) -> None:
    """Ensure `key_to_index` returns correct indices for chunked image data."""
    dataset = ImageFolderDataset(image_data)
    for idx in range(num_images):
        assert dataset.key_to_index(str(idx)) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("abc")
    with pytest.raises(KeyError):
        dataset.key_to_index("-1")
    with pytest.raises(KeyError):
        dataset.key_to_index(str(num_images))


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1)],
    indirect=["image_data"],
)
def test_imagefolder_file_root_key_to_index(image_data: Path, num_images: int) -> None:
    """Ensure `key_to_index` returns correct indices given a single image file."""
    # Get single file root
    files = list(image_data.iterdir())
    assert len(files) == num_images
    assert files[0].is_file()

    # Create dataset
    dataset = ImageFolderDataset(files[0])
    for idx in range(num_images):
        assert dataset.key_to_index(str(idx)) == idx

    with pytest.raises(KeyError):
        dataset.key_to_index("abc")
    with pytest.raises(KeyError):
        dataset.key_to_index("-1")
    with pytest.raises(KeyError):
        dataset.key_to_index(str(image_data))


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1), (8, 8)],
    indirect=["image_data"],
)
def test_imagefolder_chunks_property_directory_root(
    image_data: Path, num_images: int
) -> None:
    """Ensure the `chunks` and `chunk_sizes` properties return correct values \
    when created with a regular directory root."""
    dataset = ImageFolderDataset(image_data)
    assert dataset.chunks == tuple(sorted(image_data.iterdir()))
    assert dataset.chunk_sizes == tuple([1] * num_images)


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1), (8, 8)],
    indirect=["image_data"],
)
def test_imagefolder_chunks_property_symlink_directory_root(
    tmp_path: Path, image_data: Path, num_images: int
) -> None:
    """Ensure the `chunks` and `chunk_sizes` properties return correct values \
    when created with a symlinked directory root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(image_data)
    dataset = ImageFolderDataset(ln_root)
    assert dataset.chunks == tuple(sorted(ln_root.iterdir()))
    assert tuple([p.resolve() for p in dataset.chunks]) == tuple(
        sorted(image_data.iterdir())
    )
    assert dataset.chunk_sizes == tuple([1] * num_images)


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1)],
    indirect=["image_data"],
)
def test_imagefolder_chunks_property_file_root(
    image_data: Path, num_images: int
) -> None:
    """Ensure the `chunks` and `chunk_sizes` properties return correct values \
    when created with a regular file root."""
    files = list(image_data.iterdir())
    assert len(files) == num_images
    assert files[0].is_file()
    dataset = ImageFolderDataset(files[0])
    assert dataset.chunks == (files[0],)
    assert dataset.chunk_sizes == tuple([1] * num_images)


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1)],
    indirect=["image_data"],
)
def test_imagefolder_chunks_property_symlinked_file_root(
    tmp_path: Path, image_data: Path, num_images: int
) -> None:
    """Ensure the `chunks` and `chunk_sizes` properties return correct values \
    when created with a symlinked file root."""
    files = list(image_data.iterdir())
    assert len(files) == num_images
    assert files[0].is_file()
    ln_file = tmp_path / "link"
    ln_file.symlink_to(files[0])
    dataset = ImageFolderDataset(ln_file)
    assert dataset.chunks == (ln_file,)
    assert tuple([p.resolve() for p in dataset.chunks]) == (files[0],)
    assert dataset.chunk_sizes == tuple([1] * num_images)


@pytest.mark.parametrize(
    "image_data",
    [1, 8],
    indirect=["image_data"],
)
def test_imagefolder_root_property_directory_root(image_data: Path) -> None:
    """Ensure the `root` property returns a correct value when created with a \
    regular directory root."""
    dataset = ImageFolderDataset(image_data)
    assert dataset.root == image_data


@pytest.mark.parametrize(
    "image_data",
    [1, 8],
    indirect=["image_data"],
)
def test_imagefolder_root_property_symlink_directory_root(
    tmp_path: Path, image_data: Path
) -> None:
    """Ensure the `root` property returns a correct value when created with a \
    symlinked directory root."""
    ln_root = tmp_path / "link"
    ln_root.symlink_to(image_data)
    dataset = ImageFolderDataset(ln_root)
    assert dataset.root == ln_root
    assert dataset.root.resolve() == image_data


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1)],
    indirect=["image_data"],
)
def test_imagefolder_root_property_file_root(image_data: Path, num_images: int) -> None:
    """Ensure the `root` property returns a correct value when created with a \
    regular file root."""
    files = list(image_data.iterdir())
    assert len(files) == num_images
    assert files[0].is_file()
    dataset = ImageFolderDataset(files[0])
    assert dataset.root == files[0]


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1)],
    indirect=["image_data"],
)
def test_imagefolder_root_property_symlinked_file_root(
    tmp_path: Path, image_data: Path, num_images: int
) -> None:
    """Ensure the `root` property returns a correct value when created with a \
    symlinked file root."""
    files = list(image_data.iterdir())
    assert len(files) == num_images
    assert files[0].is_file()
    ln_file = tmp_path / "link"
    ln_file.symlink_to(files[0])
    dataset = ImageFolderDataset(ln_file)
    assert dataset.root == ln_file
    assert dataset.root.resolve() == files[0]


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1), (8, 8)],
    indirect=["image_data"],
)
def test_imagefolder_directory_root_keys(image_data: Path, num_images: int) -> None:
    """Ensure `key_to_index` returns correct indices for multiple images."""
    dataset = ImageFolderDataset(image_data)
    for idx, key in enumerate(dataset.keys()):
        assert key == str(idx)
    assert len(tuple(dataset.keys())) == num_images


@pytest.mark.parametrize(
    "image_data, num_images",
    [(1, 1)],
    indirect=["image_data"],
)
def test_imagefolder_file_root_keys(image_data: Path, num_images: int) -> None:
    """Ensure `key_to_index` returns correct indices given a single image file."""
    # Get single file root
    files = list(image_data.iterdir())
    assert len(files) == num_images
    assert files[0].is_file()

    # Create dataset
    dataset = ImageFolderDataset(files[0])
    for idx, key in enumerate(dataset.keys()):
        assert key == str(idx)
    assert len(tuple(dataset.keys())) == num_images
