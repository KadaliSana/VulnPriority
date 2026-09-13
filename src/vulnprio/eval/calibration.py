"""Probability calibration (DESIGN.md 4).

A ranking only needs the *order* of ``P(exploit)`` to be right. The framework's priority
construct needs the *value* to be right, because expected loss multiplies that
probability by money: a model that says 0.9 when it means 0.3 misprices every finding
it touches by a factor of three, and the knapsack and the simulation both spend real
capacity on that mispricing. So calibration is reported as a first-class result, not as
a diagnostic.

Two numbers and a picture:

* **Brier score** - mean squared error of the probability. Proper scoring rule, so it
  cannot be improved by shading predictions toward the base rate.
* **Expected calibration error** - mean absolute gap between confidence and observed
  frequency, over equal-width bins.
* **Reliability bins** - the same decomposition as a curve, which is what the report
  plots; a well-calibrated model lies on the diagonal.

Equal-width binning (rather than equal-count) is chosen deliberately: with a 3% positive
rate almost every prediction sits near zero, and equal-count bins would hide a badly
overconfident high-probability tail inside one wide bin.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from vulnprio.core.models import CalibrationReport

__all__ = ["brier_score", "expected_calibration_error", "reliability_bins", "bin_indices"]


def _pair(y_true: Iterable[float], y_prob: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(list(y_true), dtype=float)
    probability = np.asarray(list(y_prob), dtype=float)
    if truth.shape != probability.shape:
        raise ValueError(f"length mismatch: {truth.shape} vs {probability.shape}")
    return (truth > 0).astype(float), np.clip(probability, 0.0, 1.0)


def brier_score(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    """Mean squared error of a probabilistic forecast.

    ::

        BS = (1 / N) * sum_i (p_i - y_i)^2

    Range ``[0, 1]``, lower is better. The base-rate forecaster scores ``p(1-p)``, which
    on a 3% positive rate is 0.029 - so a Brier score near 0.03 means "no better than
    predicting the base rate for everything", and the report says so rather than
    presenting a small number as a good one. Returns ``0.0`` for empty input.
    """
    truth, probability = _pair(y_true, y_prob)
    if len(truth) == 0:
        return 0.0
    return float(np.mean((probability - truth) ** 2))


def bin_indices(probability: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin index in ``[0, n_bins)`` for each probability.

    Bin ``b`` covers ``[b/n_bins, (b+1)/n_bins)``, with the top bin closed so that
    ``p = 1.0`` lands in it rather than overflowing.
    """
    raw = np.floor(np.asarray(probability, dtype=float) * n_bins).astype(int)
    return np.clip(raw, 0, n_bins - 1)


def expected_calibration_error(
    y_true: Iterable[float], y_prob: Iterable[float], n_bins: int = 10
) -> float:
    """Expected calibration error over equal-width bins.

    ::

        ECE = sum_b (n_b / N) * |acc(b) - conf(b)|

    where ``conf(b)`` is the mean predicted probability in bin ``b`` and ``acc(b)`` the
    observed positive frequency in it. Empty bins contribute nothing. Zero means the
    predicted probabilities match observed frequencies at every confidence level.
    Returns ``0.0`` for empty input; ``n_bins`` below 2 is raised to 2.
    """
    truth, probability = _pair(y_true, y_prob)
    if len(truth) == 0:
        return 0.0
    n_bins = max(2, int(n_bins))
    assignment = bin_indices(probability, n_bins)
    total = float(len(truth))
    error = 0.0
    for index in range(n_bins):
        mask = assignment == index
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probability[mask].mean())
        accuracy = float(truth[mask].mean())
        error += (count / total) * abs(accuracy - confidence)
    return float(error)


def reliability_bins(
    y_true: Iterable[float], y_prob: Iterable[float], n_bins: int = 10
) -> CalibrationReport:
    """Full calibration report: Brier, ECE and the per-bin reliability curve.

    Every bin appears in the output, including empty ones (confidence and accuracy
    ``0.0``, count ``0``), so the arrays are always ``n_bins`` long and the reliability
    diagram's x-axis is stable across rankers and folds.
    """
    truth, probability = _pair(y_true, y_prob)
    n_bins = max(2, int(n_bins))
    confidences: list[float] = []
    accuracies: list[float] = []
    counts: list[int] = []
    if len(truth):
        assignment = bin_indices(probability, n_bins)
        for index in range(n_bins):
            mask = assignment == index
            count = int(mask.sum())
            counts.append(count)
            confidences.append(float(probability[mask].mean()) if count else 0.0)
            accuracies.append(float(truth[mask].mean()) if count else 0.0)
    else:
        confidences = [0.0] * n_bins
        accuracies = [0.0] * n_bins
        counts = [0] * n_bins

    return CalibrationReport(
        brier=brier_score(truth, probability),
        ece=expected_calibration_error(truth, probability, n_bins),
        n_bins=n_bins,
        bin_confidence=tuple(confidences),
        bin_accuracy=tuple(accuracies),
        bin_count=tuple(counts),
    )
