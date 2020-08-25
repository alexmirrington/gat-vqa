"""Classes for storing training configuration information."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Class for storing training configuration information."""

    epochs: int
    log_step: int

    def __post_init__(self) -> None:
        """Validate fields after dataclass construction."""
        if self.epochs <= 0:
            raise ValueError(f"Field {self.epochs=} must be positive.")

        if self.log_step <= 0:
            raise ValueError(f"Field {self.log_step=} must be strictly positive.")
