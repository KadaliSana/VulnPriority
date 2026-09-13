"""Minority-class reporting (DESIGN.md 3.9, Gap 7).

Exploited findings are rare - a few percent of a scan at most - and they are the only
ones that matter. Aggregate accuracy is therefore actively misleading: the "not
exploited for everything" classifier scores 0.97 on a 3%-positive problem and is worth
nothing. This module produces the report that makes that impossible to hide: MCC,
balanced accuracy, and per-class precision, recall, F1 **with support**, so a class with
three examples cannot disappear inside an aggregate.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from vulnprio.core.models import MinorityClassReport
from vulnprio.eval.metrics import (
    balanced_accuracy,
    confusion_counts,
    f1_binary,
    mcc,
)

__all__ = ["minority_report", "per_class_scores"]

#: Default names for the two classes of the exploitation problem.
DEFAULT_CLASS_LABELS: tuple[str, str] = ("not_exploited", "exploited")


def per_class_scores(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    class_labels: Sequence[str] = DEFAULT_CLASS_LABELS,
) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 and support for each class.

    ::

        precision = TP / (TP + FP)      recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        support = number of true members of the class

    The negative class is scored by inverting both vectors, so the two rows are computed
    by identical code and no asymmetry can creep in.
    """
    truth = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    if truth.shape != predicted.shape:
        raise ValueError(f"length mismatch: {truth.shape} vs {predicted.shape}")
    negative_name, positive_name = (
        (class_labels[0], class_labels[1]) if len(class_labels) >= 2 else DEFAULT_CLASS_LABELS
    )
    truth_binary = (truth > 0).astype(float)
    predicted_binary = (predicted > 0).astype(float)

    out: dict[str, dict[str, float]] = {}
    for name, actual, guess in (
        (positive_name, truth_binary, predicted_binary),
        (negative_name, 1.0 - truth_binary, 1.0 - predicted_binary),
    ):
        tp, fp, _, fn = confusion_counts(actual, guess)
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        out[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_binary(actual, guess),
            "support": float(tp + fn),
        }
    return out


def minority_report(
    y_true: Iterable[float],
    y_score: Iterable[float],
    threshold: float = 0.5,
    class_labels: Sequence[str] = DEFAULT_CLASS_LABELS,
) -> MinorityClassReport:
    """Full minority-class report at a decision threshold.

    ``y_score`` is a probability (or any score in ``[0, 1]``); predictions are
    ``score >= threshold``. The report carries the positive rate itself so that every
    other number in it can be read against the base rate, which is the context that makes
    an F1 of 0.4 on a 3%-positive problem look like the substantial result it is.

    Empty input returns an all-zero report rather than raising.
    """
    truth = np.asarray(list(y_true), dtype=float)
    score = np.asarray(list(y_score), dtype=float)
    if truth.shape != score.shape:
        raise ValueError(f"length mismatch: {truth.shape} vs {score.shape}")
    threshold = float(min(1.0, max(0.0, threshold)))
    if len(truth) == 0:
        return MinorityClassReport(
            positive_rate=0.0, mcc=0.0, f1_positive=0.0, balanced_accuracy=0.0, threshold=threshold
        )

    truth_binary = (truth > 0).astype(float)
    predicted = (score >= threshold).astype(float)
    return MinorityClassReport(
        positive_rate=float(truth_binary.mean()),
        mcc=float(min(1.0, max(-1.0, mcc(truth_binary, predicted)))),
        f1_positive=f1_binary(truth_binary, predicted),
        balanced_accuracy=balanced_accuracy(truth_binary, predicted),
        threshold=threshold,
        per_class=per_class_scores(truth_binary, predicted, class_labels),
    )
