"""Utilities for efficient management of metric calculation."""
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Union

from ..config import Config
from .metrics import accuracy, f_1, precision, recall


class Metric(Enum):
    """Enum defining names of various metrics."""

    ACCURACY = "accuracy"
    RECALL = "recall"
    PRECISION = "precision"
    F1 = "f1"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    PLAUSIBILITY = "plausibility"
    GROUNDING = "grounding"
    DISTRIBUTION = "distribution"


class MetricCollection:
    """Wrapper class for storing a collection of metrics to evaluate."""

    def __init__(
        self, config: Config, metrics: Union[Metric, Iterable[Metric]]
    ) -> None:
        """Create a new collection of metrics."""
        self._metrics: Dict[Metric, Optional[float]] = {}

        if isinstance(metrics, Metric):
            self._metrics[metrics] = None
        else:
            for key in metrics:
                self._metrics[key] = None
        self._metric_funcs = {
            Metric.ACCURACY: accuracy,
            Metric.RECALL: recall,
            Metric.PRECISION: precision,
            Metric.F1: f_1,
        }
        self._config = config
        self._ids: List[str] = []
        self._preds: List[str] = []
        self._targets: List[str] = []

    def append(
        self, ids: Iterable[str], preds: Iterable[str], targets: Iterable[str],
    ) -> None:
        """Add a prediction or batch of predictions to the collection to be \
        evaluated in the next `evaluate()` call."""
        self._ids += list(ids)
        self._preds += list(preds)
        self._targets += list(targets)

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate all metrics in the collection and return the results."""
        return {
            metric.value: self._metric_funcs[metric](self._targets, self._preds)
            for metric in self._metrics
        }
