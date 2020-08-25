"""Fixtures and configuration for dataset utility tests."""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image


@dataclass
class ChunkedDataConfig:
    """Class that stores information about chunks for use in fixture requests."""

    num_chunks: int = 1
    chunk_size: int = 1


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

        content = {str(chunk_idx + idx): chunk_idx + idx for idx in range(chunk_size)}
        with path.open("w") as file:
            json.dump(content, file)

    return root


@pytest.fixture(name="image_data")
def fixture_image_data(tmp_path_factory, request):
    """Create a image folder dataset in a temporary directory for use in tests."""
    # Make root dir
    root = tmp_path_factory.mktemp("data")

    # Set params
    num_images = request.param

    # Create image files
    paths = [root / Path(f"{idx}.png") for idx in range(num_images)]
    dimensions = [(idx % 10 + 1, (10 - idx) % 10 + 1) for idx in range(num_images)]
    for path, dim in zip(paths, dimensions):
        image = Image.new(mode="RGB", size=dim)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, "wb") as img_file:
            image.save(img_file)
    return root
