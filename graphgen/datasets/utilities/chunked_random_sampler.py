"""Utilities for chunked data shuffling."""
from typing import Dict, Iterable, List

import torch
from tqdm import tqdm

from ...datasets.gqa import GQA
from .chunked_dataset import ChunkedDataset


class ChunkedRandomSampler:
    """Custom sampler that performs a chunked shuffle fo maximise cache hits."""

    def __init__(
        self, data_source: ChunkedDataset, generator: torch.Generator = None
    ) -> None:
        """Create a new ChunkedRandomSampler instance.

        Params:
        -------
        `data_source`: Dataset to sample from.
        `generator`: Generator used in sampling.
        """
        self.data_source = data_source
        self.generator = generator

    def __iter__(self) -> Iterable[int]:
        """Get an iterator for the sampler instance."""
        chunk_bounds = self.data_source.chunk_sizes
        # Permute items inside chunks
        perms = []
        start = 0
        for bound in chunk_bounds:
            perms.append(
                (torch.randperm(bound, generator=self.generator) + start).tolist()
            )
            start += bound
        # Permute chunks
        chunk_perm = torch.randperm(len(perms), generator=self.generator).tolist()
        result = []
        for cidx in chunk_perm:
            result += perms[cidx]
        return iter(result)

    def __len__(self) -> int:
        """Get the length of the sampler's data source."""
        return len(self.data_source)


class GQAObjectsChunkedRandomSampler:
    """Custom sampler that performs a chunked shuffle fo maximise cache hits \
    on GQA object features. Assumes data contains question and object feats."""

    def __init__(self, data: GQA, generator: torch.Generator = None) -> None:
        """Create a new ChunkedRandomSampler instance.

        Params:
        -------
        `data_source`: Dataset to sample from.
        `generator`: Generator used in sampling.
        """
        self.data = data
        self.generator = generator

    def __iter__(self) -> Iterable[int]:
        """Get an iterator for the sampler instance."""
        # Build index to chunk map.
        assert self.data.objects is not None
        assert self.data.questions is not None
        chunk_sizes = self.data.objects.chunk_sizes
        chunk_to_idx: Dict[int, List[int]] = {
            cidx: [] for cidx in range(len(chunk_sizes))
        }
        for idx in tqdm(range(len(self.data.questions)), desc="shuffling: "):
            question = self.data.questions[idx]
            image_key = question["imageId"]
            obj_index = self.data.objects.key_to_index(image_key)
            start = 0
            for cidx, size in enumerate(chunk_sizes):
                if start <= obj_index < start + size:
                    # we found correct chunk index
                    chunk_to_idx[cidx].append(idx)
                    break
                start += size

        print(f"{[(idx, len(val)) for idx, val in chunk_to_idx.items()]}")

        # Permute items inside chunks
        shuffled_idxs = []
        for cidx, indices in chunk_to_idx.items():
            perm = torch.randperm(len(indices), generator=self.generator)
            shuffled_idxs.append(
                torch.tensor(indices)[perm].tolist()  # pylint: disable=not-callable
            )

        print(f"{chunk_sizes=}")

        # Permute chunks
        chunk_perm = torch.randperm(
            len(shuffled_idxs), generator=self.generator
        ).tolist()
        print(f"{chunk_perm=}")
        result = []
        for cidx in chunk_perm:
            result += shuffled_idxs[cidx]
        return iter(result)

    def __len__(self) -> int:
        """Get the length of the sampler's data source."""
        return len(self.data)
