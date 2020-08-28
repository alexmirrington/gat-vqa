"""Package containing various collator implementations."""
from .variable_sized_tensor_collator import (
    VariableSizeTensorCollator as VariableSizeTensorCollator,
)

__all__ = [VariableSizeTensorCollator.__name__]
