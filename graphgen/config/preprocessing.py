"""Classes for storing preprocessing configuration information."""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PipelineStepConfig:
    """Class for storing preprocessor pipeline step information."""

    feature: str


@dataclass(frozen=True)
class PreprocessingConfig:
    """Class for storing preprocessoing configuration information."""

    pipeline: List[PipelineStepConfig]
    cache: Path

    def __post_init__(self) -> None:
        """Validate fields after dataclass construction."""
        if not self.cache.exists() or not self.cache.is_dir():
            raise ValueError(f"Field {self.cache=} must point to a valid directory.")
