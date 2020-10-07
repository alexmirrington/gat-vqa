"""Abstract GCN base implementation."""
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Batch


class AbstractGCN(torch.nn.Module, ABC):  # type: ignore
    """Abstract GCN base class."""

    def __init__(
        self,
        shape: Sequence[int],
        pooling: Optional[Callable[..., torch.Tensor]],
    ) -> None:
        """Create a GCN with layer sizes according to `shape`."""
        super().__init__()
        self.shape = shape
        self.pooling = pooling

    @abstractmethod
    def forward(self, graphs: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform a forward GCN pass.

        Params:
        -------
        `data`: A PyG batch containing coo-formatted edge indices and node
        embeddings.

        Returns:
        --------
        `features`: A tensor containing the processed node features for all
        graphs in the batch.
        `pooled_features`: An optional tensor containing globally-pooled node
        features for all graphs in the batch.
        """
