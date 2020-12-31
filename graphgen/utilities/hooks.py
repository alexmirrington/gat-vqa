"""Module containing various torch hooks."""
from typing import Any

import torch
from torch_geometric.nn.conv import GATConv

from ..modules.reasoning.mac.control import ControlUnit
from ..modules.reasoning.mac.read import ReadUnit


class GATConvAttentionHook:
    """Callable hook for retrieving attention maps from a `GATConv` layer."""

    def __init__(self) -> None:
        """Initialise the hook."""
        self.result = None

    def reset(self):
        """Reset the hook for reuse."""
        self.result = None

    def __call__(self, module: torch.nn.Module, ipt: Any, output: Any) -> None:
        """Get the attention weights from a `torch_geometric.nn.conv.GATConv` layer."""
        if not isinstance(module, GATConv):
            return
        if isinstance(output, tuple) and len(output) > 1:
            coos, attn = output[1]
            self.result = coos, attn
            return
        self.result = None


class ReadUnitAttentionHook:
    """Callable hook for retrieving attention maps from a `ReadUnit` layer."""

    def __init__(self) -> None:
        """Initialise the hook."""
        self.result = None

    def reset(self):
        """Reset the hook for reuse."""
        self.result = None

    def __call__(self, module: torch.nn.Module, ipt: Any, output: Any) -> None:
        """Get the attention weights from a `mac.read.ReadUnit` layer."""
        if not isinstance(module, ReadUnit):
            return
        if isinstance(output, tuple) and len(output) > 1:
            _, attn = output
            if self.result is None:
                self.result = []
            self.result.append(attn)
            return
        self.result = None


class ControlUnitAttentionHook:
    """Callable hook for retrieving attention maps from a `ControlUnit` layer."""

    def __init__(self) -> None:
        """Initialise the hook."""
        self.result = None

    def reset(self):
        """Reset the hook for reuse."""
        self.result = None

    def __call__(self, module: torch.nn.Module, ipt: Any, output: Any) -> None:
        """Get the attention weights from a `mac.control.ControlUnit` layer."""
        if not isinstance(module, ControlUnit):
            return
        if isinstance(output, tuple) and len(output) > 1:
            _, attn = output
            if self.result is None:
                self.result = []
            self.result.append(attn)
            return
        self.result = None
