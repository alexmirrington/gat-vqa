"""Utilities for chunked data shuffling."""
from typing import Iterable

import torch

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
