"""Functions for computing various metrics."""
from typing import Any, Iterable

import sklearn.metrics
import wandb


def accuracy(targets: Iterable[Any], preds: Iterable[Any], **_: Any) -> Any:
    """Compute standard accuracy metric."""
    return float(sklearn.metrics.accuracy_score(targets, preds))


def precision(targets: Iterable[Any], preds: Iterable[Any], **_: Any) -> Any:
    """Compute standard precision metric."""
    return float(
        sklearn.metrics.precision_score(
            targets, preds, average="micro", zero_division=0
        )
    )


def recall(targets: Iterable[Any], preds: Iterable[Any], **_: Any) -> Any:
    """Compute standard recall metric."""
    return float(
        sklearn.metrics.recall_score(targets, preds, average="micro", zero_division=0)
    )


def f_1(targets: Iterable[Any], preds: Iterable[Any], **_: Any) -> Any:
    """Compute standard f1 metric."""
    return float(
        sklearn.metrics.f1_score(targets, preds, average="micro", zero_division=0)
    )


def confusion_matrix(
    targets: Iterable[Any], preds: Iterable[Any], **kwargs: Any
) -> Any:
    """Compute confusion matrix."""
    return wandb.plots.HeatMap(
        kwargs["labels"],
        kwargs["labels"],
        sklearn.metrics.confusion_matrix(
            targets, preds, labels=list(range(len(kwargs["labels"])))
        ),
    )


def consistency(targets: Iterable[Any], preds: Iterable[Any], **kwargs: Any) -> Any:
    """Compute GQA consistency.

    References:
    -----------
    Based on official GQA evaluation script:
    https://cs.stanford.edu/people/dorarad/gqa/evaluate.htm
    """
    questions = kwargs["questions"]
    qids = kwargs["ids"]
    scores = []
    qid_pred_map = dict(zip(qids, preds))

    for qid, pred, target in zip(qids, preds, targets):
        entailed = [
            eid
            for eid in questions[questions.key_to_index(qid)]["entailed"]
            if eid != qid
        ]

        if pred == target and len(entailed) > 0:
            consistency_scores = []
            for eid in entailed:
                entailed_question = questions[questions.key_to_index(eid)]
                # Filter out entailed questions that are not balanced. This is not
                # implemented in the original eval script, but good for val metrics.
                if entailed_question["isBalanced"]:
                    gold = entailed_question["answer"]
                    predicted = qid_pred_map[eid]
                    consistency_scores.append(1 if predicted == gold else 0)
            if len(consistency_scores) != 0:
                scores.append(sum(consistency_scores) / len(consistency_scores))
    return sum(scores) / len(scores)
