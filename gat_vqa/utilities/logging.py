"""Utilities for logigng metrics."""
from typing import Any, Mapping

from termcolor import colored


def log_metrics_stdout(
    metrics: Mapping[str, Any],
    coloured: bool = True,
    newline: bool = True,
) -> None:
    """Log all cached metrics in the `metrics` collection to stdout."""
    output = ""

    for idx, (key, value) in enumerate(metrics.items()):
        color = None
        if coloured:
            color = "cyan"
            if "epoch" in key:
                color = None
            elif "loss" in key:
                color = "red"
            elif "accuracy" in key:
                color = "blue"
        if isinstance(value, float):
            value = f"{value:.4f}"
        valstr = colored(value, color=color) if color else value
        output += f"{key}: {valstr} "
    output = output.rstrip()
    if not newline:
        output = f"{output}\r"
    print(output, end="\n" if newline else "")
