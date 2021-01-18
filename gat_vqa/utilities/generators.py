"""Generators for large scale data processing to help minimise memory overhead."""
from typing import Any, Dict, Generator, Sequence


def slice_dict(
    dictionary: Dict[Any, Any], step: int
) -> Generator[Dict[Any, Any], None, None]:
    """Slice a dictionary into multiple dictionaries with length `step`."""
    result = {}
    for idx, (key, val) in enumerate(dictionary.items()):
        result[key] = val
        if idx % step == step - 1:
            yield result
            result = {}
    yield result


def slice_sequence(
    sequence: Sequence[Any], step: int
) -> Generator[Sequence[Any], None, None]:
    """Slice a dictionary into multiple dictionaries with length `step`."""
    start = 0
    while start < len(sequence):
        yield sequence[start : min(start + step, len(sequence))]
        start += step
