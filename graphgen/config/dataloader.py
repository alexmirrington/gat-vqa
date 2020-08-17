"""Classes for storing dataloader-related configuration information."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DataloaderConfig:
    """A class specifying the common fields used across all data loaders."""

    batch_size: int
    workers: int

    def __post_init__(self) -> None:
        """Perform post-init checks on fields."""
        if self.batch_size <= 0:
            raise ValueError(f"Field {self.batch_size=} must be greater than 0.")

        if self.workers < 0:
            raise ValueError(
                f"Field {self.workers=} must be greater than or equal to 0."
            )
