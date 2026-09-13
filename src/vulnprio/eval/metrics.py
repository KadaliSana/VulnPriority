"""Every metric in the evaluation protocol, implemented from first principles.

DESIGN.md 3.9 and 4. This module is deliberately dependency-light: each function is
written out from its definition and carries that definition in its docstring, because
the correctness of the numbers here is the thing a reviewer will check first. Where an
established library computes the same quantity (scikit-learn's ``roc_auc_score``,
``average_precision_score``, ``matthews_corrcoef``, scipy's ``kendalltau``) the test
suite asserts agreement on random data rather than the implementation importing it.

Three conventions hold throughout:

1. **Degenerate input never raises.** An empty ranking, an all-zero relevance vector, a
   single-class label vector and tied scores all return a documented value. Evaluation
   runs over hundreds of scans and a scan with no exploited finding must not abort a run.
   Only genuinely malformed input (mismatched array lengths) raises ``ValueError``.
2. **Ranking metrics take ``(ranked_ids, values, k)``** where ``ranked_ids`` is the
   ordering under test (best first) and ``values`` maps a finding id to its relevance
   grade (or, for :func:`risk_capture_at_k`, to its expected loss). Ids absent from the
   mapping score zero.
3. **The candidate set is the ranked set.** Recall, NDCG's ideal ordering and risk
   capture all normalise against the findings that were actually ranked. A finding that
   was never presented to the analyst cannot be retrieved, and normalising against it
   would make a ranker look worse for having been given less to rank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Collection, Iterable, Mapping, Sequence

import numpy as np

from vulnprio.core.enums import MetricName

__all__ = [
    "dcg_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "risk_capture_at_k",
    "average_precision_at_k",
    "mean_average_precision",
    "mean_reciprocal_rank",
    "kendall_tau_vs_cvss",
    "mean_rank_of_exploited",
    "roc_auc",
    "pr_auc",
    "mcc",
    "f1_binary",
    "f1_minority",
    "balanced_accuracy",
    "confusion_counts",
    "efficiency",
    "coverage",
    "workload_reduction",
    "FunctionRankMetric",
    "RANK_METRICS",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _top_k(ranked_ids: Sequence[str], k: int | None) -> list[str]:
    """The first ``k`` ids. ``k`` of ``None`` or ``<= 0`` means the whole ranking."""
    ids = list(ranked_ids)
    if k is None or k <= 0:
        return ids
    return ids[:k]


def _values_of(ranked_ids: Iterable[str], values: Mapping[str, float]) -> list[float]:
    """Relevance (or loss) per ranked id; an id the mapping does not know scores zero."""
    return [float(values.get(finding_id, 0.0)) for finding_id in ranked_ids]


def _as_pair(y_true: Iterable[float], y_other: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(list(y_true), dtype=float)
    right = np.asarray(list(y_other), dtype=float)
    if left.shape != right.shape:
        raise ValueError(f"length mismatch: {left.shape} vs {right.shape}")
    return left, right


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ascending ranks starting at 1, with tied values sharing their mean rank."""
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(n, dtype=float)
    index = 0
    while index < n:
        end = index
        while end + 1 < n and ordered[end + 1] == ordered[index]:
            end += 1
        ranks[order[index : end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    return ranks


def _count(items: Collection[str] | int) -> int:
    return int(items) if isinstance(items, int) else len(set(items))


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def dcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int | None = None) -> float:
    """Discounted cumulative gain with exponential gain and logarithmic discount.

    ::

        DCG@k = sum_{i=1..k} (2^rel_i - 1) / log2(i + 1)

    with ``i`` the one-based rank, so the first position is discounted by
    ``log2(2) = 1``. The exponential gain is the Burges formulation: it is what makes a
    grade-4 finding worth more than five grade-1 findings, which is the behaviour the
    protocol wants when a single confirmed-exploited finding is buried under noise.
    """
    gains = _values_of(_top_k(ranked_ids, k), relevance)
    return float(
        sum((2.0**gain - 1.0) / math.log2(position + 2) for position, gain in enumerate(gains))
    )


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int | None = None) -> float:
    """Normalised DCG: ``NDCG@k = DCG@k / IDCG@k``.

    ``IDCG@k`` is the DCG of the *same relevance multiset* sorted in descending order,
    i.e. the best ordering achievable over exactly the findings that were ranked. This
    is what makes the score comparable across scans with different numbers of exploited
    findings.

    Returns ``0.0`` when the ideal gain is zero (no ranked finding has positive
    relevance) rather than raising: a scan with no confirmed-exploited finding carries
    no ranking signal and must not abort the run. Tied relevance grades are handled by
    construction - the ideal ordering sorts the multiset, so any permutation of equally
    relevant findings yields the same ideal and the same score.
    """
    top = _top_k(ranked_ids, k)
    if not top:
        return 0.0
    dcg = dcg_at_k(ranked_ids, relevance, k)
    ideal_order = sorted(_values_of(ranked_ids, relevance), reverse=True)[: len(top)]
    idcg = sum((2.0**gain - 1.0) / math.log2(position + 2) for position, gain in enumerate(ideal_order))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def precision_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int | None = None) -> float:
    """Share of the top ``k`` that is relevant.

    ::

        P@k = |{i <= k : rel_i > 0}| / min(k, n)

    The denominator is ``min(k, n)`` rather than ``k`` so that a scan holding fewer than
    ``k`` findings is not penalised for the findings it does not have. Empty ranking
    returns ``0.0``.
    """
    top = _top_k(ranked_ids, k)
    if not top:
        return 0.0
    hits = sum(1 for value in _values_of(top, relevance) if value > 0.0)
    return float(hits / len(top))


def recall_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int | None = None) -> float:
    """Share of the relevant findings that the top ``k`` retrieves.

    ::

        R@k = |{i <= k : rel_i > 0}| / |{i <= n : rel_i > 0}|

    The denominator counts positives inside the ranked candidate set. Returns ``0.0``
    when the candidate set holds no positive.
    """
    all_values = _values_of(ranked_ids, relevance)
    total_positive = sum(1 for value in all_values if value > 0.0)
    if total_positive == 0:
        return 0.0
    hits = sum(1 for value in _values_of(_top_k(ranked_ids, k), relevance) if value > 0.0)
    return float(hits / total_positive)


def risk_capture_at_k(
    ranked_ids: Sequence[str], expected_loss: Mapping[str, float], k: int | None = None
) -> float:
    """Share of the total expected loss that sits in the top ``k`` (Gap 1, Gap 10).

    ::

        RiskCapture@k = sum_{i <= k} loss_i / sum_{i <= n} loss_i

    This is the decision-relevant complement to precision: a ranking that puts one
    high-value finding at rank 1 and nineteen trivial ones after it has poor
    precision and excellent risk capture, and the second fact is the one that matters to
    the team spending the week. Negative losses are clipped to zero. Returns ``0.0``
    when the total is zero.
    """
    values = [max(0.0, value) for value in _values_of(ranked_ids, expected_loss)]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    captured = sum(max(0.0, value) for value in _values_of(_top_k(ranked_ids, k), expected_loss))
    return float(min(1.0, captured / total))


def average_precision_at_k(
    ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int | None = None
) -> float:
    """Average precision for one ranking.

    ::

        AP@k = (1 / min(R, k)) * sum_{i <= k} 1[rel_i > 0] * P@i

    where ``R`` is the number of relevant findings in the candidate set. Dividing by
    ``min(R, k)`` rather than ``R`` keeps ``AP@k = 1`` attainable when more positives
    exist than positions. Returns ``0.0`` when nothing is relevant.
    """
    top = _top_k(ranked_ids, k)
    total_positive = sum(1 for value in _values_of(ranked_ids, relevance) if value > 0.0)
    if total_positive == 0 or not top:
        return 0.0
    hits = 0
    running = 0.0
    for position, value in enumerate(_values_of(top, relevance), start=1):
        if value > 0.0:
            hits += 1
            running += hits / position
    denominator = min(total_positive, len(top))
    return float(running / denominator) if denominator else 0.0


def mean_average_precision(
    ranked: Sequence[str] | Sequence[Sequence[str]],
    relevance: Mapping[str, float] | Sequence[Mapping[str, float]],
    k: int | None = None,
) -> float:
    """Mean of :func:`average_precision_at_k` over queries (scans).

    Accepts either one ranking (a flat sequence of ids, in which case this is plain
    average precision) or a sequence of rankings. ``relevance`` may be a single mapping
 - finding ids are globally unique, so one map covers every scan - or one mapping per
    ranking. An empty input returns ``0.0``.
    """
    rankings, relevances = _normalise_queries(ranked, relevance)
    if not rankings:
        return 0.0
    scores = [
        average_precision_at_k(ranking, rel, k) for ranking, rel in zip(rankings, relevances)
    ]
    return float(sum(scores) / len(scores))


def mean_reciprocal_rank(
    ranked: Sequence[str] | Sequence[Sequence[str]],
    relevance: Mapping[str, float] | Sequence[Mapping[str, float]],
    k: int | None = None,
) -> float:
    """Mean reciprocal rank of the first relevant finding.

    ::

        MRR = (1 / Q) * sum_q 1 / rank of the first relevant item in q

    A query with no relevant item inside the top ``k`` contributes ``0``. Accepts one
    ranking or a sequence of rankings, like :func:`mean_average_precision`.
    """
    rankings, relevances = _normalise_queries(ranked, relevance)
    if not rankings:
        return 0.0
    total = 0.0
    for ranking, rel in zip(rankings, relevances):
        for position, value in enumerate(_values_of(_top_k(ranking, k), rel), start=1):
            if value > 0.0:
                total += 1.0 / position
                break
    return float(total / len(rankings))


def mean_rank_of_exploited(
    ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int | None = None
) -> float:
    """Mean one-based rank of the confirmed-exploited findings.

    Lower is better, and unlike NDCG it is expressed in positions an analyst can feel:
    "the exploited findings sat at rank 3.5 on average". When ``k`` is given only the
    top ``k`` positions are considered and exploited findings below the cut are charged
    ``k + 1``, so a ranker cannot improve the number by hiding positives past the cut.

    Returns ``nan`` when the candidate set holds no exploited finding: the quantity is
    undefined, and callers (``BenchmarkRunner``) omit it rather than inventing a value.
    """
    values = _values_of(ranked_ids, relevance)
    positions = [position for position, value in enumerate(values, start=1) if value > 0.0]
    if not positions:
        return math.nan
    if k is not None and k > 0:
        positions = [min(position, k + 1) for position in positions]
    return float(sum(positions) / len(positions))


def kendall_tau_vs_cvss(
    ranked_ids: Sequence[str], cvss_by_id: Mapping[str, float], k: int | None = None
) -> float:
    """Kendall's tau-b between the produced ordering and a CVSS ordering.

    ::

        tau_b = (C - D) / sqrt((n0 - n1) * (n0 - n2))

    with ``C``/``D`` the concordant/discordant pair counts, ``n0 = n(n-1)/2`` and
    ``n1``/``n2`` the tie corrections ``sum t(t-1)/2`` within each variable. The first
    variable is the negated rank position (higher means remediate sooner, never tied);
    the second is the CVSS base score (heavily tied, which is exactly why tau-b rather
    than tau-a is used).

    This is a *descriptive* number, not a target: tau near 1 means the framework has
    reproduced CVSS and added nothing, tau near 0 means it has reordered the queue
    substantially. Findings with no CVSS score are excluded rather than imputed, since
    most web application findings carry no CVE at all. Fewer than two comparable
    findings returns ``0.0``.
    """
    pairs = [
        (-position, float(cvss_by_id[finding_id]))
        for position, finding_id in enumerate(_top_k(ranked_ids, k))
        if finding_id in cvss_by_id
    ]
    n = len(pairs)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pairs[i][0] - pairs[j][0]
            dy = pairs[i][1] - pairs[j][1]
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    n0 = n * (n - 1) / 2.0
    denominator = math.sqrt((n0 - ties_x) * (n0 - ties_y))
    if denominator <= 0.0:
        return 0.0
    return float((concordant - discordant) / denominator)


def _normalise_queries(
    ranked: Sequence[str] | Sequence[Sequence[str]],
    relevance: Mapping[str, float] | Sequence[Mapping[str, float]],
) -> tuple[list[Sequence[str]], list[Mapping[str, float]]]:
    """Accept either one query or many, and broadcast a single relevance map over them."""
    items = list(ranked)
    if items and isinstance(items[0], str):
        rankings: list[Sequence[str]] = [items]  # type: ignore[list-item]
    else:
        rankings = [list(item) for item in items]  # type: ignore[arg-type]
    if isinstance(relevance, Mapping):
        relevances = [relevance] * len(rankings)
    else:
        relevances = list(relevance)
        if len(relevances) != len(rankings):
            raise ValueError("one relevance mapping per ranking is required")
    return rankings, relevances


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def roc_auc(y_true: Iterable[float], y_score: Iterable[float]) -> float:
    """Area under the ROC curve, via the Mann-Whitney U identity.

    ::

        AUC = (sum of ranks of the positives - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    Ranks are average ranks, which is what gives tied scores the correct 0.5 credit and
    makes this identical to the trapezoidal area under the ROC curve. Returns ``0.5``
    when only one class is present - the metric is undefined and 0.5 is the value of an
    uninformative model, which is the honest reading.
    """
    truth, scores = _as_pair(y_true, y_score)
    positive = truth > 0
    n_pos = int(positive.sum())
    n_neg = int(len(truth) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(scores)
    rank_sum = float(ranks[positive].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(y_true: Iterable[float], y_score: Iterable[float]) -> float:
    """Area under the precision-recall curve as *average precision*.

    ::

        AP = sum_n (R_n - R_{n-1}) * P_n

    over the distinct score thresholds, with ``R_0 = 0``. This step-wise estimator is
    the one scikit-learn's ``average_precision_score`` computes; the trapezoidal
    alternative interpolates between thresholds and is optimistically biased, so it is
    deliberately not used. Equal scores are grouped into a single threshold, so a ranker
    cannot profit from an arbitrary tie-break.

    PR-AUC rather than ROC-AUC is the headline classification number here because the
    positive class is rare (Gap 7): ROC-AUC is dominated by the abundant negatives.
    Returns ``0.0`` when no positive is present.
    """
    truth, scores = _as_pair(y_true, y_score)
    labels = (truth > 0).astype(float)
    n_pos = float(labels.sum())
    if n_pos == 0 or len(labels) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    ordered_scores = scores[order]
    # last index of each run of equal scores
    distinct = np.where(np.diff(ordered_scores))[0]
    threshold_index = np.r_[distinct, len(labels) - 1]
    true_positive = np.cumsum(labels)[threshold_index]
    false_positive = (1 + threshold_index) - true_positive
    precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)
    recall = true_positive / n_pos
    average = 0.0
    previous_recall = 0.0
    for step_precision, step_recall in zip(precision, recall):
        average += (step_recall - previous_recall) * step_precision
        previous_recall = step_recall
    return float(average)


def confusion_counts(
    y_true: Iterable[float], y_pred: Iterable[float], positive: float = 1.0
) -> tuple[int, int, int, int]:
    """``(tp, fp, tn, fn)`` for a binary prediction, treating ``>= positive`` as positive."""
    truth, predicted = _as_pair(y_true, y_pred)
    actual = truth >= positive
    guess = predicted >= positive
    tp = int(np.sum(actual & guess))
    fp = int(np.sum(~actual & guess))
    tn = int(np.sum(~actual & ~guess))
    fn = int(np.sum(actual & ~guess))
    return tp, fp, tn, fn


def mcc(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Matthews correlation coefficient (Gap 7's headline classification metric).

    ::

        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    MCC is reported because accuracy and even F1 flatter a model on a 3%-positive
    problem: a classifier that predicts "not exploited" everywhere scores 0.97 accuracy
    and 0.0 MCC. It is the correlation between prediction and truth over the full
    confusion matrix, so no cell can be ignored. Returns ``0.0`` when any marginal is
    zero (the degenerate all-one-class case), matching scikit-learn.
    """
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    numerator = float(tp) * tn - float(fp) * fn
    denominator = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def f1_binary(y_true: Iterable[float], y_pred: Iterable[float], positive: float = 1.0) -> float:
    """``F1 = 2 * precision * recall / (precision + recall)`` for one class.

    ``positive`` selects which class is scored: ``1.0`` for the exploited class, ``0.0``
    (with inverted inputs) for the negative class. Returns ``0.0`` when precision and
    recall are both zero.
    """
    tp, fp, _, fn = confusion_counts(y_true, y_pred, positive)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def f1_minority(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """F1 of whichever class is rarer in ``y_true`` (Gap 7).

    The rare class is the consequential one: an exploited finding missed costs far more
    than a benign one reviewed. When the classes are balanced the positive class is
    scored. Returns ``0.0`` for an empty or single-class truth vector.
    """
    truth, predicted = _as_pair(y_true, y_pred)
    if len(truth) == 0:
        return 0.0
    n_pos = int(np.sum(truth > 0))
    n_neg = int(len(truth) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    if n_pos <= n_neg:
        return f1_binary(truth, predicted)
    return f1_binary(1.0 - (truth > 0), 1.0 - (predicted > 0))


def balanced_accuracy(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """``(TPR + TNR) / 2``: the mean of per-class recall.

    Equal to accuracy on a balanced problem and to 0.5 for the majority-class predictor
    on any problem, which is why it appears alongside MCC. A class with no member
    contributes its recall as ``0.0``; with neither class present the result is ``0.0``.
    """
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    if (tp + fn) == 0 and (tn + fp) == 0:
        return 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float((tpr + tnr) / 2.0)


# ---------------------------------------------------------------------------
# Decision metrics (Shimizu and Hashimoto; Gap 10)
# ---------------------------------------------------------------------------


def efficiency(selected_ids: Collection[str], exploited_ids: Collection[str]) -> float:
    """Confirmed-exploited findings as a share of the selected set.

    ::

        efficiency = |selected AND exploited| / |selected|

    "Of the work we asked for, how much of it mattered." Returns ``0.0`` for an empty
    selection.
    """
    selected = set(selected_ids)
    if not selected:
        return 0.0
    return float(len(selected & set(exploited_ids)) / len(selected))


def coverage(selected_ids: Collection[str], exploited_ids: Collection[str]) -> float:
    """Share of all confirmed-exploited findings the selection retains.

    ::

        coverage = |selected AND exploited| / |exploited|

    "Of the things that mattered, how many did we ask for." Efficiency and coverage
    trade against each other and are always reported as a pair. Returns ``0.0`` when
    nothing is exploited.
    """
    exploited = set(exploited_ids)
    if not exploited:
        return 0.0
    return float(len(set(selected_ids) & exploited) / len(exploited))


def workload_reduction(selected: Collection[str] | int, total: Collection[str] | int) -> float:
    """Fraction of the queue the analyst does not have to look at.

    ::

        workload_reduction = 1 - |selected| / |total|

    Reported next to coverage because it is only meaningful as a pair: cutting 95% of
    the queue is a triumph at 100% coverage and negligence at 40%. Returns ``0.0`` for
    an empty queue.
    """
    total_count = _count(total)
    if total_count <= 0:
        return 0.0
    return float(max(0.0, 1.0 - _count(selected) / total_count))


# ---------------------------------------------------------------------------
# Registry: the ranking metrics as ``RankMetric`` implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionRankMetric:
    """Adapts a ranking function to ``vulnprio.core.interfaces.RankMetric``."""

    name: str
    fn: Callable[[Sequence[str], Mapping[str, float], int | None], float]

    def compute(self, ranked_ids: list[str], relevance: Mapping[str, float], k: int | None) -> float:
        return float(self.fn(ranked_ids, relevance, k))


#: Every ranking metric of the protocol, keyed by its ``MetricName``. ``RISK_CAPTURE_AT_K``
#: is fed expected loss rather than relevance; every other entry takes relevance grades.
RANK_METRICS: dict[MetricName, FunctionRankMetric] = {
    MetricName.NDCG_AT_K: FunctionRankMetric(MetricName.NDCG_AT_K.value, ndcg_at_k),
    MetricName.PRECISION_AT_K: FunctionRankMetric(MetricName.PRECISION_AT_K.value, precision_at_k),
    MetricName.RECALL_AT_K: FunctionRankMetric(MetricName.RECALL_AT_K.value, recall_at_k),
    MetricName.RISK_CAPTURE_AT_K: FunctionRankMetric(
        MetricName.RISK_CAPTURE_AT_K.value, risk_capture_at_k
    ),
    MetricName.MAP: FunctionRankMetric(MetricName.MAP.value, average_precision_at_k),
    MetricName.MRR: FunctionRankMetric(MetricName.MRR.value, mean_reciprocal_rank),
    MetricName.MEAN_RANK_OF_EXPLOITED: FunctionRankMetric(
        MetricName.MEAN_RANK_OF_EXPLOITED.value, mean_rank_of_exploited
    ),
    MetricName.KENDALL_TAU_VS_CVSS: FunctionRankMetric(
        MetricName.KENDALL_TAU_VS_CVSS.value, kendall_tau_vs_cvss
    ),
}
