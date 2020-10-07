"""Implementation of a collator that can handle batches containing lists of \
tensors with variable size."""

from typing import Any, List, Optional

import torch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data.dataloader import Collater


class VariableSizeTensorCollator(Collater):  # type: ignore
    """Extension of the default `torch-geometric` collator that can handle \
    batches containing lists of tensors with variable size."""

    def __init__(self, follow_batch: Optional[List[Any]] = None) -> None:
        """Create a new `VariableSizeTensorCollator`."""
        follow_batch_ = follow_batch if follow_batch is not None else []
        super().__init__(follow_batch_)

    def collate(self, batch: List[Any]) -> Any:
        """Recursively collate a batch."""
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            try:
                default_collate(batch)
            except RuntimeError:
                return batch
        return super().collate(batch)

    def __call__(self, batch: List[Any]) -> Any:
        """Collate a batch."""
        return self.collate(batch)
