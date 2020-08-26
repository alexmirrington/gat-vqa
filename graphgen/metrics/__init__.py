"""Utilities for metric calculation."""

from .metric_collection import Metric as Metric
from .metric_collection import MetricCollection as MetricCollection

__all__ = [
    MetricCollection.__name__,
    Metric.__name__,
]
