"""Classes storing model configuration information."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RPNConfig:
    """Class for storing R-CNN modlue configuration information."""

    proposals: int


@dataclass(frozen=True)
class RCNNConfig:
    """Class for storing R-CNN modlue configuration information."""

    rpn: RPNConfig


@dataclass(frozen=True)
class ModelConfig:
    """Class for storing model configuration information."""

    rcnn: Any
