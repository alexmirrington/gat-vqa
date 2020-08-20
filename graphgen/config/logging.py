"""Classes for storing logging-related configuration information."""
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    """Configuration options related to logging."""

    step: int

    def __post_init__(self) -> None:
        """Validate fields after initialisation."""
        if self.step <= 0:
            raise ValueError(f"Field {self.step=} must be strictly positive.")
