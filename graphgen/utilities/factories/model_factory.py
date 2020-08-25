"""Utilities for creating models from a config."""

import torch.nn

from ...config import Config


class ModelFactory:
    """Factory for creating models from a config."""

    def create(self, config: Config) -> torch.nn.Module:
        """Create a model from a config."""
        raise NotImplementedError()
