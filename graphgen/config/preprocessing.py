"""Classes for storing preprocessing configuration information."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PipelineStepConfig:
    """Class for storing preprocessor pipeline step information."""

    feature: str
    split: str
    version: Optional[str]


@dataclass
class PreprocessingCacheConfig:
    """Class for storing preprocessor caching information."""

    root: Path
    artifact: str

    def __post_init__(self) -> None:
        """Validate fields after dataclass construction."""
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"Field {self.root=} must point to a valid directory.")


@dataclass
class PreprocessingConfig:
    """Class for storing preprocessoing configuration information."""

    pipeline: List[PipelineStepConfig]
    cache: PreprocessingCacheConfig
