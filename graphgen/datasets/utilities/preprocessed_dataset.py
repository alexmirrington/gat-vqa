"""Classes for preprocessing and caching datasets."""
import json
from pathlib import Path
from typing import Any, Callable, Dict

import torch.utils.data
import wandb

from .chunked_json_dataset import ChunkedJSONDataset


class PreprocessedJSONDataset(ChunkedJSONDataset):
    """Class that preprocesses and caches a dataset for later use."""

    def __init__(
        self,
        source: torch.utils.data.Dataset,
        preprocessor: Callable[[Dict[str, Any]], Dict[str, Any]],
        cache: Path,
        chunks: int = 1,
    ):
        """Create a `PreprocessedJSONDataset` instance.

        Params:
        -------
        `source`: dataset to process.

        `preprocessor`: function for preprocessing a collection of samples. The
        number of input samples to the preprocessor is not guaranteed to be the
        same number of samples as in `source` unless `chunks == 1`.

        `cache`: directory to save preprocessed data in. If `cache` is a nonempty
        directory or is a file, no preprocessing occurs, `source` is ignored and
        the data in `cache` is loaded. No attempt is made to determine if the
        loaded data originated from `source` or is compatible with the provided
        preprocessor.
        """
        if not isinstance(cache, Path):
            raise TypeError(f"Parameter {cache=} must be of type {Path.__name__}.")

        if not cache.exists():
            cache.mkdir(parents=True)

        if not cache.is_dir():
            raise TypeError(f"Parameter {cache=} must be a directory.")

        self._preprocessor = preprocessor

        # Preprocess from dataset if cache is empty
        cache_hit = cache.is_dir() and len(tuple(cache.iterdir())) > 0
        if not cache_hit:
            self.process(source, cache, chunks)

        super().__init__(cache)

    def process(
        self, source: torch.utils.data.Dataset, cache: Path, chunks: int = 1
    ) -> None:
        """Process a `source` dataset and save it at `cache`."""
        chunk_size = len(source) // chunks

        data = {}
        for idx, item in enumerate(source):
            data[str(idx)] = item  # TODO move to iterables for preprocessors
            if idx % chunk_size == chunk_size - 1 or idx == len(source) - 1:
                preprocessed_data = self._preprocessor(data)
                # Save to file
                with open(cache / f"{idx // chunk_size}.json", "w") as json_file:
                    json.dump(preprocessed_data, json_file)
                del preprocessed_data
                data = {}

        artifact = wandb.Artifact(
            "gqa-preprocessed-dataset", type="dataset"
        )  # TODO give proper name
        artifact.add_dir(cache)
        wandb.log_artifact(artifact)
