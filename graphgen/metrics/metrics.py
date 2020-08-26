"""Functions for computing various metrics."""
from typing import Any, Iterable

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def accuracy(targets: Iterable[Any], preds: Iterable[Any]) -> Any:
    """Compute standard accuracy metric."""
    return float(accuracy_score(targets, preds))


def precision(targets: Iterable[Any], preds: Iterable[Any]) -> Any:
    """Compute standard precision metric."""
    return float(precision_score(targets, preds))


def recall(targets: Iterable[Any], preds: Iterable[Any]) -> Any:
    """Compute standard recall metric."""
    return float(recall_score(targets, preds))


def f_1(targets: Iterable[Any], preds: Iterable[Any]) -> Any:
    """Compute standard f1 metric."""
    return float(f1_score(targets, preds))
