"""Utilities for logigng metrics."""
from typing import Any, Mapping, Optional, Sequence

from termcolor import colored


def log_metrics_stdout(
    metrics: Mapping[str, Any],
    colors: Optional[Sequence[Optional[str]]] = None,
    newline: bool = True,
) -> None:
    """Log all cached metrics in the `metrics` collection to stdout."""
    output = ""

    for idx, (key, value) in enumerate(metrics.items()):
        color = None
        if colors is not None and idx < len(colors):
            color = colors[idx]

        if isinstance(value, float):
            value = f"{value:.4f}"
        valstr = colored(value, color=color) if color else value
        output += f"{key}: {valstr} "
    output = output.rstrip()
    if not newline:
        output = f"{output}\r"
    print(output, end="\n" if newline else "")
