"""Calibration (DESIGN.md 4).

A ranking needs the order of ``P(exploit)`` to be right; the framework's expected-loss
construct needs its *value* to be right, because it is multiplied by money. These tests
check the two numbers against hand computations and against scikit-learn, and check that
a deliberately overconfident model is caught.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from vulnprio.core.models import CalibrationReport
from vulnprio.eval.calibration import (
    bin_indices,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)


def test_brier_score_matches_the_formula_by_hand() -> None:
    truth = [1.0, 0.0, 1.0, 0.0]
    probability = [0.9, 0.1, 0.8, 0.3]
    expected = ((0.9 - 1) ** 2 + (0.1 - 0) ** 2 + (0.8 - 1) ** 2 + (0.3 - 0) ** 2) / 4
    assert expected == pytest.approx(0.0375)
    assert brier_score(truth, probability) == pytest.approx(expected)


def test_brier_score_matches_sklearn_on_random_data() -> None:
    rng = np.random.default_rng(3)
    truth = (rng.random(500) < 0.2).astype(float)
    probability = np.clip(0.2 + 0.3 * truth + rng.normal(0, 0.15, 500), 0.0, 1.0)
    assert brier_score(truth, probability) == pytest.approx(
        float(brier_score_loss(truth, probability)), abs=1e-12
    )


def test_a_perfect_forecast_scores_zero_and_the_worst_scores_one() -> None:
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier_score([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_brier_of_the_base_rate_forecaster_is_p_times_one_minus_p() -> None:
    """The number a reported Brier score has to be read against (see the report text)."""
    rate = 0.03
    truth = np.zeros(1000)
    truth[: int(rate * 1000)] = 1.0
    assert brier_score(truth, np.full(1000, rate)) == pytest.approx(rate * (1 - rate), abs=1e-9)


def test_bin_indices_are_equal_width_with_a_closed_top_bin() -> None:
    probability = np.array([0.0, 0.09, 0.1, 0.55, 0.999, 1.0])
    assert list(bin_indices(probability, 10)) == [0, 0, 1, 5, 9, 9]


def test_expected_calibration_error_worked_example() -> None:
    """Two bins, computed by hand.

    Predictions 0.1, 0.1 (bin 0) with one positive -> accuracy 0.5, confidence 0.1.
    Predictions 0.9, 0.9 (bin 1) with two positives -> accuracy 1.0, confidence 0.9.
    ECE = 0.5*|0.5-0.1| + 0.5*|1.0-0.9| = 0.25.
    """
    truth = [1.0, 0.0, 1.0, 1.0]
    probability = [0.1, 0.1, 0.9, 0.9]
    assert expected_calibration_error(truth, probability, n_bins=2) == pytest.approx(0.25)


def test_a_perfectly_calibrated_forecaster_has_zero_calibration_error() -> None:
    """Half the 0.5-predictions come true, all the 1.0-predictions do, none of the 0.0s."""
    truth = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    probability = [0.5, 0.5, 1.0, 1.0, 0.0, 0.0]
    assert expected_calibration_error(truth, probability, n_bins=10) == pytest.approx(0.0)


def test_an_overconfident_model_is_caught_by_ece_at_identical_discrimination() -> None:
    """The failure calibration exists to catch: right order, wrong prices.

    Two groups of 300: one is 80% exploited, the other 20%. The calibrated model says
    0.8 and 0.2; the overconfident one says 0.99 and 0.01. Their orderings - and so
    their ROC-AUC - are identical, but the overconfident model would misprice every
    expected-loss calculation it touches, and only ECE sees it.
    """
    from vulnprio.eval.metrics import roc_auc

    truth = np.concatenate([np.repeat([1.0, 0.0], [240, 60]), np.repeat([1.0, 0.0], [60, 240])])
    group_high = np.arange(600) < 300
    calibrated = np.where(group_high, 0.8, 0.2)
    overconfident = np.where(group_high, 0.99, 0.01)

    assert roc_auc(truth, calibrated) == pytest.approx(roc_auc(truth, overconfident))
    assert expected_calibration_error(truth, calibrated, 10) == pytest.approx(0.0, abs=1e-12)
    assert expected_calibration_error(truth, overconfident, 10) == pytest.approx(0.19, abs=1e-9)


def test_calibration_error_is_bounded_and_never_negative() -> None:
    rng = np.random.default_rng(5)
    for _ in range(20):
        n = int(rng.integers(5, 200))
        truth = (rng.random(n) < 0.3).astype(float)
        probability = rng.random(n)
        error = expected_calibration_error(truth, probability, 10)
        assert 0.0 <= error <= 1.0


def test_reliability_bins_returns_the_frozen_report_with_every_bin_present() -> None:
    rng = np.random.default_rng(9)
    truth = (rng.random(300) < 0.3).astype(float)
    probability = np.clip(0.3 * truth + rng.random(300) * 0.6, 0.0, 1.0)

    report = reliability_bins(truth, probability, n_bins=8)
    assert isinstance(report, CalibrationReport)
    assert report.n_bins == 8
    assert len(report.bin_confidence) == len(report.bin_accuracy) == len(report.bin_count) == 8
    assert sum(report.bin_count) == len(truth)
    assert report.brier == pytest.approx(brier_score(truth, probability))
    assert report.ece == pytest.approx(expected_calibration_error(truth, probability, 8))
    for confidence, accuracy in zip(report.bin_confidence, report.bin_accuracy):
        assert 0.0 <= confidence <= 1.0 and 0.0 <= accuracy <= 1.0


def test_empty_bins_are_reported_as_empty_rather_than_dropped() -> None:
    """A stable x-axis across rankers and folds is what makes the diagram comparable."""
    report = reliability_bins([1.0, 0.0], [0.95, 0.9], n_bins=10)
    assert report.bin_count == (0, 0, 0, 0, 0, 0, 0, 0, 0, 2)
    assert report.bin_confidence[0] == 0.0 and report.bin_accuracy[0] == 0.0


def test_degenerate_inputs_do_not_raise() -> None:
    empty = reliability_bins([], [], n_bins=5)
    assert empty.brier == 0.0 and empty.ece == 0.0 and empty.bin_count == (0,) * 5
    assert brier_score([], []) == 0.0
    assert expected_calibration_error([], [], 10) == 0.0
    # n_bins below the contract minimum is raised to it rather than rejected
    assert reliability_bins([1.0], [1.0], n_bins=1).n_bins == 2


def test_length_mismatch_is_a_programming_error_and_raises() -> None:
    with pytest.raises(ValueError):
        brier_score([1.0, 0.0], [0.5])
    with pytest.raises(ValueError):
        reliability_bins([1.0, 0.0], [0.5])


def test_probabilities_outside_the_unit_interval_are_clipped() -> None:
    assert brier_score([1.0], [1.4]) == pytest.approx(0.0)
    assert brier_score([0.0], [-0.3]) == pytest.approx(0.0)
