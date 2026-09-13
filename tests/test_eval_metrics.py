"""The metric mathematics (DESIGN.md 3.9 and 4).

This is the module where an error is most expensive: every claim the framework makes is
expressed through these numbers, so each one is checked either against a value computed
by hand from its published formula, or against an independent implementation
(scikit-learn, scipy) on random data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats
from sklearn import metrics as sk

from vulnpriority.core.enums import MetricName
from vulnpriority.core.interfaces import RankMetric
from vulnpriority.eval.metrics import (
    RANK_METRICS,
    average_precision_at_k,
    balanced_accuracy,
    coverage,
    dcg_at_k,
    efficiency,
    f1_binary,
    f1_minority,
    kendall_tau_vs_cvss,
    mcc,
    mean_average_precision,
    mean_rank_of_exploited,
    mean_reciprocal_rank,
    ndcg_at_k,
    pr_auc,
    precision_at_k,
    recall_at_k,
    risk_capture_at_k,
    roc_auc,
    workload_reduction,
)
from vulnpriority.eval.minority import minority_report, per_class_scores

IDS = ["f1", "f2", "f3", "f4", "f5", "f6"]
GRADES = [3, 2, 3, 0, 1, 2]
RELEVANCE = dict(zip(IDS, GRADES))


# ---------------------------------------------------------------------------
# NDCG against a hand-computed example
# ---------------------------------------------------------------------------


def test_dcg_matches_the_formula_written_out_by_hand() -> None:
    """DCG = sum (2^rel - 1) / log2(i + 1), i one-based."""
    expected = (
        (2**3 - 1) / math.log2(2)
        + (2**2 - 1) / math.log2(3)
        + (2**3 - 1) / math.log2(4)
        + (2**0 - 1) / math.log2(5)
        + (2**1 - 1) / math.log2(6)
        + (2**2 - 1) / math.log2(7)
    )
    assert expected == pytest.approx(13.848263629, abs=1e-9)
    assert dcg_at_k(IDS, RELEVANCE) == pytest.approx(expected)


def test_ndcg_matches_the_hand_computed_value() -> None:
    """The ideal ordering is the same relevance multiset sorted descending."""
    ideal = (
        7 / math.log2(2)
        + 7 / math.log2(3)
        + 3 / math.log2(4)
        + 3 / math.log2(5)
        + 1 / math.log2(6)
        + 0 / math.log2(7)
    )
    assert ideal == pytest.approx(14.595390756, abs=1e-9)
    assert ndcg_at_k(IDS, RELEVANCE) == pytest.approx(13.848263629 / 14.595390756, abs=1e-9)
    assert ndcg_at_k(IDS, RELEVANCE) == pytest.approx(0.9488107, abs=1e-6)


def test_ndcg_at_k_truncates_both_the_ranking_and_the_ideal() -> None:
    """NDCG@3 compares the top 3 achieved against the top 3 achievable."""
    achieved = 7 + 3 / math.log2(3) + 7 / math.log2(4)
    ideal = 7 + 7 / math.log2(3) + 3 / math.log2(4)
    assert ndcg_at_k(IDS, RELEVANCE, 3) == pytest.approx(achieved / ideal, abs=1e-9)
    assert ndcg_at_k(IDS, RELEVANCE, 3) == pytest.approx(0.9594535, abs=1e-6)


def test_perfect_ordering_scores_one_and_reversal_scores_less() -> None:
    ordered = sorted(IDS, key=lambda key: -RELEVANCE[key])
    assert ndcg_at_k(ordered, RELEVANCE) == pytest.approx(1.0)
    assert ndcg_at_k(list(reversed(ordered)), RELEVANCE) < ndcg_at_k(IDS, RELEVANCE)


def test_ndcg_tie_case_is_order_independent_among_equal_grades() -> None:
    """Permuting equally relevant findings cannot change the score."""
    tied = {"a": 2.0, "b": 2.0, "c": 0.0}
    assert ndcg_at_k(["a", "b", "c"], tied) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a", "c"], tied) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "b", "c"], tied) == pytest.approx(ndcg_at_k(["b", "a", "c"], tied))

    # A tie split by an irrelevant finding is genuinely worse, and still <= 1.
    assert ndcg_at_k(["a", "c", "b"], tied) < 1.0
    assert ndcg_at_k(["a", "c", "b"], tied) <= 1.0


def test_ndcg_all_zero_relevance_returns_zero_rather_than_raising() -> None:
    """A scan with no confirmed-exploited finding must not abort an evaluation run."""
    zeros = {key: 0.0 for key in IDS}
    assert ndcg_at_k(IDS, zeros) == 0.0
    assert ndcg_at_k(IDS, zeros, 5) == 0.0
    assert ndcg_at_k(IDS, {}) == 0.0


def test_ranking_metrics_survive_empty_input() -> None:
    for k in (None, 0, 5):
        assert ndcg_at_k([], RELEVANCE, k) == 0.0
        assert precision_at_k([], RELEVANCE, k) == 0.0
        assert recall_at_k([], RELEVANCE, k) == 0.0
        assert risk_capture_at_k([], RELEVANCE, k) == 0.0
        assert average_precision_at_k([], RELEVANCE, k) == 0.0
        assert mean_reciprocal_rank([], RELEVANCE, k) == 0.0
        assert kendall_tau_vs_cvss([], {}, k) == 0.0
    assert math.isnan(mean_rank_of_exploited([], RELEVANCE))


# ---------------------------------------------------------------------------
# Precision, recall, MAP, MRR, mean rank
# ---------------------------------------------------------------------------


def test_precision_and_recall_at_k_worked_example() -> None:
    # grades 3, 2, 3, 0, 1, 2 -> positives at ranks 1, 2, 3, 5, 6
    assert precision_at_k(IDS, RELEVANCE, 3) == pytest.approx(1.0)
    assert precision_at_k(IDS, RELEVANCE, 4) == pytest.approx(3 / 4)
    assert recall_at_k(IDS, RELEVANCE, 3) == pytest.approx(3 / 5)
    assert recall_at_k(IDS, RELEVANCE, 6) == pytest.approx(1.0)


def test_precision_denominator_is_the_shorter_of_k_and_the_queue() -> None:
    """A three-finding scan is not penalised on P@10 for having seven fewer findings."""
    short = {"a": 1.0, "b": 1.0, "c": 1.0}
    assert precision_at_k(["a", "b", "c"], short, 10) == pytest.approx(1.0)


def test_average_precision_worked_example() -> None:
    """AP = (1/R) sum over hits of P@i, with R the positives in the candidate set."""
    relevance = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
    # hits at rank 1 (P=1/1) and rank 3 (P=2/3), R=2
    assert average_precision_at_k(["a", "b", "c", "d"], relevance) == pytest.approx(
        (1.0 + 2.0 / 3.0) / 2.0
    )
    assert mean_average_precision(["a", "b", "c", "d"], relevance) == pytest.approx(
        (1.0 + 2.0 / 3.0) / 2.0
    )


def test_mean_average_precision_averages_over_queries() -> None:
    relevance = {"a": 1.0, "b": 0.0, "x": 1.0, "y": 0.0}
    first = average_precision_at_k(["a", "b"], relevance)
    second = average_precision_at_k(["y", "x"], relevance)
    assert mean_average_precision([["a", "b"], ["y", "x"]], relevance) == pytest.approx(
        (first + second) / 2.0
    )


def test_mean_reciprocal_rank_worked_example() -> None:
    relevance = {"a": 0.0, "b": 0.0, "c": 1.0}
    assert mean_reciprocal_rank(["a", "b", "c"], relevance) == pytest.approx(1 / 3)
    assert mean_reciprocal_rank([["a", "b", "c"], ["c", "a", "b"]], relevance) == pytest.approx(
        (1 / 3 + 1.0) / 2.0
    )
    # nothing relevant inside the cut contributes zero
    assert mean_reciprocal_rank(["a", "b", "c"], relevance, 2) == 0.0


def test_mean_rank_of_exploited_worked_example_and_undefined_case() -> None:
    relevance = {"a": 0.0, "b": 4.0, "c": 0.0, "d": 3.0}
    assert mean_rank_of_exploited(["a", "b", "c", "d"], relevance) == pytest.approx((2 + 4) / 2)
    # below the cut, an exploited finding is charged k + 1 rather than its true rank
    assert mean_rank_of_exploited(["a", "b", "c", "d"], relevance, 2) == pytest.approx((2 + 3) / 2)
    assert math.isnan(mean_rank_of_exploited(["a", "c"], relevance))


# ---------------------------------------------------------------------------
# Risk capture, efficiency, coverage, workload reduction: worked examples
# ---------------------------------------------------------------------------


def test_risk_capture_at_k_worked_example() -> None:
    """Fraction of total expected loss sitting in the top k."""
    loss = {"a": 900.0, "b": 50.0, "c": 30.0, "d": 20.0}  # total 1000
    order = ["a", "b", "c", "d"]
    assert risk_capture_at_k(order, loss, 1) == pytest.approx(0.90)
    assert risk_capture_at_k(order, loss, 2) == pytest.approx(0.95)
    assert risk_capture_at_k(order, loss, 4) == pytest.approx(1.0)
    # the same set ordered badly captures far less at the same k
    assert risk_capture_at_k(["d", "c", "b", "a"], loss, 1) == pytest.approx(0.02)


def test_risk_capture_rewards_one_expensive_finding_over_three_cheap_ones() -> None:
    """The decision-relevant complement to precision: money, not counts."""
    loss = {"big": 1_000_000.0, "s1": 10.0, "s2": 10.0, "s3": 10.0}
    relevance = {"big": 1.0, "s1": 1.0, "s2": 1.0, "s3": 1.0}
    money_first = ["big", "s1", "s2", "s3"]
    count_first = ["s1", "s2", "s3", "big"]
    assert precision_at_k(money_first, relevance, 3) == precision_at_k(count_first, relevance, 3)
    assert risk_capture_at_k(money_first, loss, 1) > risk_capture_at_k(count_first, loss, 3)


def test_risk_capture_degenerate_inputs() -> None:
    assert risk_capture_at_k(["a"], {"a": 0.0}) == 0.0
    assert risk_capture_at_k(["a", "b"], {}) == 0.0


def test_efficiency_and_coverage_worked_example() -> None:
    """Shimizu and Hashimoto's pair: precision of the selection, recall of the exploited."""
    selected = ["a", "b", "c", "d"]           # four items asked for
    exploited = ["a", "c", "e"]               # three items that mattered
    assert efficiency(selected, exploited) == pytest.approx(2 / 4)
    assert coverage(selected, exploited) == pytest.approx(2 / 3)

    # selecting everything: perfect coverage, poor efficiency, no workload saved
    everything = ["a", "b", "c", "d", "e"]
    assert coverage(everything, exploited) == pytest.approx(1.0)
    assert efficiency(everything, exploited) == pytest.approx(3 / 5)
    assert workload_reduction(everything, everything) == pytest.approx(0.0)


def test_workload_reduction_worked_example() -> None:
    assert workload_reduction(20, 400) == pytest.approx(0.95)
    assert workload_reduction(["a", "b"], ["a", "b", "c", "d"]) == pytest.approx(0.5)
    assert workload_reduction(5, 0) == 0.0


def test_efficiency_and_coverage_degenerate_inputs() -> None:
    assert efficiency([], ["a"]) == 0.0
    assert coverage(["a"], []) == 0.0


# ---------------------------------------------------------------------------
# Kendall tau against scipy
# ---------------------------------------------------------------------------


def test_kendall_tau_vs_cvss_matches_scipy_on_random_data() -> None:
    rng = np.random.default_rng(11)
    ids = [f"f{index}" for index in range(40)]
    # heavy ties on purpose: CVSS scores cluster on a handful of values
    cvss = {key: float(rng.choice([4.3, 5.3, 7.5, 9.8])) for key in ids}
    order = list(rng.permutation(ids))
    expected = scipy_stats.kendalltau(
        [-position for position in range(len(order))], [cvss[key] for key in order], variant="b"
    ).statistic
    assert kendall_tau_vs_cvss(order, cvss) == pytest.approx(float(expected), abs=1e-12)


def test_kendall_tau_is_one_when_the_ranking_reproduces_cvss() -> None:
    cvss = {"a": 9.8, "b": 7.5, "c": 4.3}
    assert kendall_tau_vs_cvss(["a", "b", "c"], cvss) == pytest.approx(1.0)
    assert kendall_tau_vs_cvss(["c", "b", "a"], cvss) == pytest.approx(-1.0)
    # findings with no CVSS score are excluded, not imputed as zero
    assert kendall_tau_vs_cvss(["a", "no_cve", "b", "c"], cvss) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Classification metrics against scikit-learn on random data
# ---------------------------------------------------------------------------


@pytest.fixture
def random_binary() -> tuple[np.ndarray, np.ndarray]:
    """An imbalanced, partially informative problem: 400 rows, ~12% positive."""
    rng = np.random.default_rng(20240601)
    truth = (rng.random(400) < 0.12).astype(float)
    score = np.clip(0.25 * truth + rng.normal(0.4, 0.2, size=400), 0.0, 1.0)
    return truth, score


def test_roc_auc_matches_sklearn(random_binary) -> None:
    truth, score = random_binary
    assert roc_auc(truth, score) == pytest.approx(float(sk.roc_auc_score(truth, score)), abs=1e-12)


def test_pr_auc_matches_sklearn_average_precision(random_binary) -> None:
    truth, score = random_binary
    assert pr_auc(truth, score) == pytest.approx(
        float(sk.average_precision_score(truth, score)), abs=1e-12
    )


def test_mcc_matches_sklearn(random_binary) -> None:
    truth, score = random_binary
    predicted = (score >= 0.5).astype(float)
    assert mcc(truth, predicted) == pytest.approx(float(sk.matthews_corrcoef(truth, predicted)), abs=1e-12)


def test_auc_metrics_match_sklearn_over_many_random_draws() -> None:
    """Repeated draws, including heavy score ties, which is where AP implementations differ."""
    rng = np.random.default_rng(7)
    for _ in range(25):
        n = int(rng.integers(20, 120))
        truth = (rng.random(n) < rng.uniform(0.05, 0.5)).astype(float)
        if truth.sum() == 0 or truth.sum() == n:
            continue
        score = rng.choice([0.1, 0.2, 0.5, 0.9], size=n)  # deliberate ties
        assert roc_auc(truth, score) == pytest.approx(float(sk.roc_auc_score(truth, score)), abs=1e-12)
        assert pr_auc(truth, score) == pytest.approx(
            float(sk.average_precision_score(truth, score)), abs=1e-12
        )
        predicted = (score >= 0.5).astype(float)
        assert mcc(truth, predicted) == pytest.approx(
            float(sk.matthews_corrcoef(truth, predicted)), abs=1e-12
        )


def test_f1_and_balanced_accuracy_match_sklearn(random_binary) -> None:
    truth, score = random_binary
    predicted = (score >= 0.5).astype(float)
    assert f1_binary(truth, predicted) == pytest.approx(float(sk.f1_score(truth, predicted)), abs=1e-12)
    assert balanced_accuracy(truth, predicted) == pytest.approx(
        float(sk.balanced_accuracy_score(truth, predicted)), abs=1e-12
    )


def test_classification_metrics_survive_a_single_class() -> None:
    """An all-negative fold is common and must not raise."""
    truth = np.zeros(10)
    score = np.linspace(0.0, 1.0, 10)
    assert roc_auc(truth, score) == 0.5
    assert pr_auc(truth, score) == 0.0
    assert mcc(truth, (score >= 0.5).astype(float)) == 0.0
    assert f1_minority(truth, (score >= 0.5).astype(float)) == 0.0
    assert balanced_accuracy(truth, np.zeros(10)) == pytest.approx(0.5)


def test_majority_class_predictor_is_exposed_by_mcc_not_by_accuracy() -> None:
    """Gap 7 in one assertion: the do-nothing classifier looks excellent on accuracy."""
    truth = np.array([1.0] * 3 + [0.0] * 97)
    predicted = np.zeros(100)
    accuracy = float(np.mean(truth == predicted))
    assert accuracy == pytest.approx(0.97)
    assert mcc(truth, predicted) == 0.0
    assert f1_minority(truth, predicted) == 0.0
    assert balanced_accuracy(truth, predicted) == pytest.approx(0.5)


def test_f1_minority_scores_the_rarer_class() -> None:
    truth = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])       # negatives are the majority
    predicted = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert f1_minority(truth, predicted) == pytest.approx(float(sk.f1_score(truth, predicted)))

    flipped_truth = 1.0 - truth                              # now positives are the majority
    flipped_predicted = 1.0 - predicted
    assert f1_minority(flipped_truth, flipped_predicted) == pytest.approx(
        float(sk.f1_score(flipped_truth, flipped_predicted, pos_label=0))
    )


# ---------------------------------------------------------------------------
# Minority-class report (Gap 7)
# ---------------------------------------------------------------------------


def test_minority_report_carries_support_per_class(random_binary) -> None:
    truth, score = random_binary
    report = minority_report(truth, score, threshold=0.5)
    assert report.positive_rate == pytest.approx(float(truth.mean()))
    assert report.threshold == 0.5
    assert set(report.per_class) == {"exploited", "not_exploited"}
    assert report.per_class["exploited"]["support"] == pytest.approx(float(truth.sum()))
    assert report.per_class["not_exploited"]["support"] == pytest.approx(float(len(truth) - truth.sum()))
    assert report.mcc == pytest.approx(
        float(sk.matthews_corrcoef(truth, (score >= 0.5).astype(float))), abs=1e-12
    )


def test_per_class_scores_match_sklearn(random_binary) -> None:
    truth, score = random_binary
    predicted = (score >= 0.5).astype(float)
    scores = per_class_scores(truth, predicted)
    assert scores["exploited"]["precision"] == pytest.approx(
        float(sk.precision_score(truth, predicted, zero_division=0))
    )
    assert scores["exploited"]["recall"] == pytest.approx(
        float(sk.recall_score(truth, predicted, zero_division=0))
    )
    assert scores["not_exploited"]["f1"] == pytest.approx(
        float(sk.f1_score(truth, predicted, pos_label=0))
    )


def test_minority_report_handles_empty_input() -> None:
    report = minority_report([], [], threshold=0.4)
    assert report.positive_rate == 0.0 and report.mcc == 0.0 and report.threshold == 0.4


# ---------------------------------------------------------------------------
# The registry conforms to the frozen RankMetric protocol
# ---------------------------------------------------------------------------


def test_rank_metric_registry_conforms_to_the_frozen_protocol() -> None:
    assert MetricName.NDCG_AT_K in RANK_METRICS
    for name, metric in RANK_METRICS.items():
        assert isinstance(metric, RankMetric)
        assert metric.name == name.value
        assert isinstance(metric.compute(IDS, RELEVANCE, 5), float)


def test_metric_value_key_spelling_matches_the_registry() -> None:
    """``MetricValue.key`` is what the report and the bootstrap index on."""
    from vulnpriority.core.models import MetricValue

    assert MetricValue(name=MetricName.NDCG_AT_K, k=10, value=0.5).key == "ndcg@10"
    assert MetricValue(name=MetricName.MCC, value=0.5).key == "mcc"
    assert MetricValue(name=MetricName.RISK_CAPTURE_AT_K, k=5, value=0.5).key == "risk_capture@5"
