"""Classes for storing preprocessing configuration information."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PipelineStepConfig:
    """Class for storing preprocessor pipeline step information."""

    feature: str


@dataclass(frozen=True)
class PreprocessingConfig:
    """Class for storing preprocessoring configuration information."""

    pipeline: List[PipelineStepConfig]
