"""Classes for preprocessing and caching datasets."""
import json
import pickle
from pathlib import Path
from typing import Dict, Iterator

import jsons
import torch.utils.data
import wandb

from ...utilities.preprocessing import Preprocessor
from .chunked_json_dataset import ChunkedJSONDataset
from .keyed_dataset import KeyedDataset


class PreprocessedJSONDataset(ChunkedJSONDataset, KeyedDataset):
    """Class that preprocesses and caches a dataset for later use."""

    def __init__(
        self,
        source: KeyedDataset,
        preprocessor: Preprocessor,
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

        if not (cache / "chunks").exists():
            (cache / "chunks").mkdir(parents=True)

        if not cache.is_dir():
            raise TypeError(f"Parameter {cache=} must be a directory.")

        self._preprocessor = preprocessor
        self._key_to_index: Dict[str, int] = {}
        if not self.cache_hit(cache):
            self.process(source, cache, chunks)

        super().__init__(cache / "chunks")

    def cache_hit(self, cache: Path) -> bool:
        """Determine whether a cached preprocessed dataset alread exists."""
        if len(tuple((cache / "chunks").iterdir())) == 0:
            return False
        if not (cache / "preprocessor.pkl").exists():
            return False
        try:
            with open(cache / "preprocessor.pkl", "rb") as pkl_file:
                cached_preprocessor = pickle.load(pkl_file)
                result: bool = self._preprocessor == cached_preprocessor
            return result
        except OSError:
            return False
        return False

    def process(
        self, source: torch.utils.data.Dataset, cache: Path, chunks: int = 1
    ) -> None:
        """Process a `source` dataset and save it at `cache`."""
        chunk_size = len(source) // chunks

        # Preprocess
        keys = []
        data = []
        for idx, key in enumerate(source.keys()):
            self._key_to_index[key] = idx
            keys.append(key)
            data.append(source[source.key_to_index(key)])
            if idx % chunk_size == chunk_size - 1 or idx == len(source) - 1:
                preprocessed_data = self._preprocessor(data)
                # Save to file
                with open(
                    cache / "chunks" / f"{idx // chunk_size}.json", "w"
                ) as json_file:
                    json.dump(dict(zip(keys, preprocessed_data)), json_file)
                del preprocessed_data
                keys = []
                data = []

        # Dump preprocessor object public fields so we can inspect information
        # about the preprocessed dataset.
        preprocessor = jsons.dump(self._preprocessor, strip_privates=True)
        metadata = {"preprocessor": preprocessor}
        with open(cache / "meta.json", "w") as json_file:
            json.dump(metadata, json_file)

        # Pickle preprocessor so we can compare if preprocessors are equal later
        with open(cache / "preprocessor.pkl", "wb") as pkl_file:
            pickle.dump(self._preprocessor, pkl_file)

        artifact = wandb.Artifact(
            "gqa-preprocessed-dataset", type="dataset"
        )  # TODO give proper name
        artifact.add_dir(cache)
        wandb.log_artifact(artifact)

    def keys(self) -> Iterator[str]:
        """Get the dataset's keys."""
        return iter(self._key_to_index.keys())

    def key_to_index(self, key: str) -> int:
        """Get index of a given key in the dataset."""
        return self._key_to_index[key]
