"""Functions for computing various metrics."""
from typing import Any, Iterable

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def accuracy(targets: Iterable[Any], preds: Iterable[Any], **_: Any) -> Any:
    """Compute standard accuracy metric."""
    return float(accuracy_score(targets, preds))


def precision(targets: Iterable[Any], preds: Iterable[Any], **_: Any) -> Any:
    """Compute standard precision metric."""
    return float(precision_score(targets, preds, average="macro", zero_division=0))


def recall(targets: Iterable[Any], preds: Iterable[Any], **_: Any) -> Any:
    """Compute standard recall metric."""
    return float(recall_score(targets, preds, average="macro", zero_division=0))


def f_1(targets: Iterable[Any], preds: Iterable[Any], **_: Any) -> Any:
    """Compute standard f1 metric."""
    return float(f1_score(targets, preds, average="macro", zero_division=0))


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
