"""Utilities for efficient management of metric calculation."""
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .metrics import accuracy, confusion_matrix, consistency, f_1, precision, recall


class Metric(Enum):
    """Enum defining names of various metrics."""

    ACCURACY = "accuracy"
    RECALL = "recall"
    PRECISION = "precision"
    F1 = "f1"
    CONSISTENCY = "consistency"
    CONFUSION_MATRIX = "confusion_matrix"


class MetricCollection:
    """Wrapper class for storing a collection of metrics to evaluate."""

    def __init__(
        self, metrics: Union[Metric, Iterable[Metric]], labels: Sequence[str]
    ) -> None:
        """Create a new collection of metrics."""
        self._metrics: Dict[Metric, Optional[float]] = {}

        if isinstance(metrics, Metric):
            self._metrics[metrics] = None
        else:
            for key in metrics:
                self._metrics[key] = None
        self._labels = labels
        self._metric_funcs = {
            Metric.ACCURACY: accuracy,
            Metric.RECALL: recall,
            Metric.PRECISION: precision,
            Metric.F1: f_1,
            Metric.CONSISTENCY: consistency,
            Metric.CONFUSION_MATRIX: confusion_matrix,
        }
        self._ids: List[str] = []
        self._preds: List[str] = []
        self._targets: List[str] = []

    def append(
        self,
        ids: Iterable[str],
        preds: Iterable[str],
        targets: Iterable[str],
    ) -> None:
        """Add a prediction or batch of predictions to the collection to be \
        evaluated in the next `evaluate()` call."""
        self._ids += list(ids)
        self._preds += list(preds)
        self._targets += list(targets)

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate all metrics in the collection and return the results."""
        kwargs = {"ids": self._ids, "labels": self._labels}
        return {
            metric.value: self._metric_funcs[metric](
                self._targets, self._preds, **kwargs
            )
            for metric in self._metrics
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._ids = []
        self._preds = []
        self._targets = []
