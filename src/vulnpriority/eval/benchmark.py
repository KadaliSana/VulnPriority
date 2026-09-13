"""``BenchmarkRunner``: every ranker on identical data (DESIGN.md 3.9, Gap 4).

Gap 4 is that published comparisons are not comparisons: different studies use different
datasets, different preprocessing, different metrics and different splits, so the numbers
cannot be placed beside one another. The answer is not a better metric, it is a harness
that makes divergence impossible - one frame per split, built once; one label set; one
metric battery; one seed; every ranker, learned and baseline, fed exactly the same rows
in exactly the same order.

What the runner guarantees:

* **Identical data.** Every ranker sees the same ``FeatureFrame`` objects. A baseline is
  not privileged with extra columns and the learned ranker is not privileged with a
  different preprocessing path.
* **Identical metrics.** Metrics are computed once per scan from the ranker's output
  order, by the same code, for every ranker.
* **Per-scan retention.** Every metric is kept per test scan, not only as a fold mean,
  because the paired bootstrap and the Wilcoxon test in :mod:`vulnpriority.eval.bootstrap`
  need the pairs. This is what lets the report say "better on 24 of 30 scans" rather
  than "0.71 versus 0.68".

``vulnpriority.rank`` is imported **lazily, inside the resolution function**, so this module
imports cleanly while the ranking package is still being written and so that a caller who
passes ranker instances never pays for xgboost at all.
"""

from __future__ import annotations

import time
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vulnpriority.core.config import PipelineConfig
from vulnpriority.core.enums import MetricName, RankerName
from vulnpriority.core.interfaces import Ranker
from vulnpriority.core.models import (
    ComponentFlags,
    FeatureFrame,
    LabelSet,
    MetricBundle,
    MetricValue,
    Split,
)
from vulnpriority.eval.bootstrap import bootstrap_mean_ci
from vulnpriority.eval.calibration import reliability_bins
from vulnpriority.eval.metrics import (
    average_precision_at_k,
    balanced_accuracy,
    coverage,
    efficiency,
    f1_minority,
    kendall_tau_vs_cvss,
    mcc,
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
from vulnpriority.eval.minority import minority_report

__all__ = ["SplitFrames", "BenchmarkRunner", "scan_order", "rank_from_scores", "flags_of"]


@dataclass(frozen=True)
class SplitFrames:
    """The train and test matrices for one fold, built once and shared by every ranker."""

    split: Split
    train: FeatureFrame
    test: FeatureFrame


def scan_order(frame: FeatureFrame, scores: np.ndarray, scan_id: str) -> list[str]:
    """Finding ids of one scan, ordered by descending score.

    Ties break on the frame's own row order, which is deterministic and identical for
    every ranker, so no ranker gains or loses from an arbitrary tie-break. That matters
    for the KEV-first and CVSS-only baselines, whose scores are heavily tied by design.
    """
    rows = [
        (index, finding_id)
        for index, (finding_id, group) in enumerate(zip(frame.finding_ids, frame.group_ids))
        if group == scan_id
    ]
    rows.sort(key=lambda item: (-float(scores[item[0]]), item[0]))
    return [finding_id for _, finding_id in rows]


def rank_from_scores(frame: FeatureFrame, scores: np.ndarray) -> dict[str, list[str]]:
    """``{scan_id: ranked finding ids}`` for every scan in the frame."""
    return {scan_id: scan_order(frame, scores, scan_id) for scan_id in dict.fromkeys(frame.group_ids)}


@dataclass
class BenchmarkRunner:
    """Runs every ranker over every fold and produces one :class:`MetricBundle` each.

    ``expected_loss``, ``cvss_base`` and ``probabilities`` are optional side
    channels. Expected loss and CVSS are read from the feature matrix when not supplied
    (``b_expected_loss_log`` is ``log1p`` of the loss, ``cvss_base_max`` is the score),
    which keeps the runner usable in ablation cells where Component B's columns have been
    dropped. ``probabilities`` maps a ranker to calibrated ``P(exploit)`` per finding: it
    is the only input from which calibration metrics are computed, because a ranking
    score is not a probability and reporting its Brier score would be meaningless.
    """

    expected_loss: Mapping[str, float] | None = None
    cvss_base: Mapping[str, float] | None = None
    probabilities: Mapping[Any, Mapping[str, float]] | None = None
    #: ``(ranker, fold) -> metric key -> scan id -> value``, retained for the bootstrap.
    per_scan_values: dict[tuple[RankerName, int], dict[str, dict[str, float]]] = field(
        default_factory=dict
    )

    # -- public API --------------------------------------------------------

    def run(
        self,
        frames_by_split: Any,
        labels: LabelSet,
        rankers: Sequence[Any],
        config: PipelineConfig,
    ) -> list[MetricBundle]:
        """Fit and score every ranker on every fold; one bundle per (ranker, fold)."""
        folds = _normalise_folds(frames_by_split)
        relevance = {key: float(value) for key, value in labels.relevance().items()}
        positives = labels.positives()
        resolved = [_resolve_ranker(item, config) for item in rankers]
        self.per_scan_values = {}

        bundles: list[MetricBundle] = []
        for fold in folds:
            for name, ranker in resolved:
                bundles.append(
                    self._run_one(fold, name, ranker, relevance, positives, labels, config)
                )
        return bundles

    def paired_values(
        self,
        ranker_a: RankerName,
        ranker_b: RankerName,
        metric_key: str,
    ) -> tuple[list[float], list[float]]:
        """Per-scan values for two rankers, aligned on the scans both were tested on.

        This is the input :func:`vulnpriority.eval.bootstrap.paired_bootstrap_ci` expects.
        """
        left = self._collect(ranker_a, metric_key)
        right = self._collect(ranker_b, metric_key)
        shared = sorted(set(left) & set(right))
        return ([left[scan] for scan in shared], [right[scan] for scan in shared])

    def per_scan(self, ranker: RankerName, metric_key: str) -> dict[str, float]:
        """Every retained per-scan value for one ranker and metric, across folds."""
        return self._collect(ranker, metric_key)

    # -- internals ---------------------------------------------------------

    def _collect(self, ranker: RankerName, metric_key: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for (name, fold), metrics in sorted(self.per_scan_values.items(), key=lambda kv: kv[0][1]):
            if name != ranker:
                continue
            for scan_id, value in metrics.get(metric_key, {}).items():
                out[f"{scan_id}#f{fold}"] = value
        return out

    def _run_one(
        self,
        fold: SplitFrames,
        name: RankerName,
        ranker: Any,
        relevance: Mapping[str, float],
        positives: set[str],
        labels: LabelSet,
        config: PipelineConfig,
    ) -> MetricBundle:
        train, test = fold.train, fold.test
        loss = self._loss_lookup(test)
        cvss = self._cvss_lookup(test)

        started = time.perf_counter()
        if getattr(ranker, "requires_fit", lambda: True)():
            train_relevance = np.asarray(
                [relevance.get(finding_id, 0.0) for finding_id in train.finding_ids], dtype=float
            )
            weights = None
            if config.ranking.impact_weighted_pairs:
                train_loss = self._loss_lookup(train)
                weights = np.asarray(
                    [
                        1.0 + np.log1p(max(0.0, train_loss.get(finding_id, 0.0)) / 1000.0)
                        for finding_id in train.finding_ids
                    ],
                    dtype=float,
                )
            ranker.fit(train, train_relevance, weights, config.ranking.seed)
        scores = np.asarray(ranker.score(test), dtype=float)
        runtime = time.perf_counter() - started
        if scores.shape[0] != len(test.finding_ids):
            raise ValueError(
                f"{name.value} returned {scores.shape[0]} scores for {len(test.finding_ids)} rows"
            )

        orders = rank_from_scores(test, scores)
        per_scan = self._ranking_metrics(orders, relevance, loss, cvss, config)
        self.per_scan_values[(name, fold.split.fold)] = per_scan

        values = _aggregate(per_scan, config)
        truth = np.asarray(
            [1.0 if finding_id in positives else 0.0 for finding_id in test.finding_ids], dtype=float
        )
        probability = self._probability_vector(name, test, scores)
        values.extend(_classification_values(truth, scores, probability, config))

        calibration = None
        supplied = self._supplied_probabilities(name)
        if supplied is not None:
            calibration = reliability_bins(truth, probability, config.evaluation.calibration_bins)

        return MetricBundle(
            ranker=name,
            flags=test.flags,
            split=fold.split,
            seed=config.ranking.seed,
            values=tuple(values),
            calibration=calibration,
            minority=minority_report(
                truth, probability, config.evaluation.classification_threshold
            ),
            runtime_seconds=max(0.0, runtime),
        )

    def _ranking_metrics(
        self,
        orders: Mapping[str, list[str]],
        relevance: Mapping[str, float],
        loss: Mapping[str, float],
        cvss: Mapping[str, float],
        config: PipelineConfig,
    ) -> dict[str, dict[str, float]]:
        """Every ranking metric, per scan, keyed the way ``MetricValue.key`` spells it."""
        per_scan: dict[str, dict[str, float]] = {}

        def record(key: str, scan_id: str, value: float) -> None:
            per_scan.setdefault(key, {})[scan_id] = float(value)

        largest_k = max(config.evaluation.k_values) if config.evaluation.k_values else 20
        for scan_id, order in orders.items():
            for k in config.evaluation.k_values:
                record(f"ndcg@{k}", scan_id, ndcg_at_k(order, relevance, k))
                record(f"precision@{k}", scan_id, precision_at_k(order, relevance, k))
                record(f"recall@{k}", scan_id, recall_at_k(order, relevance, k))
                record(f"risk_capture@{k}", scan_id, risk_capture_at_k(order, loss, k))
            record(MetricName.MAP.value, scan_id, average_precision_at_k(order, relevance, None))
            record(MetricName.MRR.value, scan_id, mean_reciprocal_rank(order, relevance, None))
            record(
                MetricName.KENDALL_TAU_VS_CVSS.value, scan_id, kendall_tau_vs_cvss(order, cvss, None)
            )
            mean_rank = mean_rank_of_exploited(order, relevance, None)
            if np.isfinite(mean_rank):
                # Undefined for a scan with no exploited finding: omitted rather than
                # filled in, so the fold mean is over the scans where it means something.
                record(MetricName.MEAN_RANK_OF_EXPLOITED.value, scan_id, mean_rank)

            selected = order[:largest_k]
            exploited = [finding_id for finding_id in order if relevance.get(finding_id, 0.0) > 0.0]
            record(MetricName.EFFICIENCY.value, scan_id, efficiency(selected, exploited))
            record(MetricName.COVERAGE.value, scan_id, coverage(selected, exploited))
            record(
                MetricName.WORKLOAD_REDUCTION.value, scan_id, workload_reduction(selected, order)
            )
        return per_scan

    def _loss_lookup(self, frame: FeatureFrame) -> dict[str, float]:
        """Expected loss per finding: supplied, else recovered from the feature matrix.

        ``b_expected_loss_log`` is ``log1p(expected_loss)``, so ``expm1`` inverts it.
        In an ablation cell with Component B disabled the column is absent and every
        finding is given unit loss, which makes ``risk_capture@k`` degenerate to
        ``recall@k`` - the honest reading, since without Component B there is no monetary
        estimate to capture.
        """
        if self.expected_loss is not None:
            return {
                finding_id: float(self.expected_loss.get(finding_id, 0.0))
                for finding_id in frame.finding_ids
            }
        if "b_expected_loss_log" in frame.X.columns:
            values = np.expm1(np.clip(frame.X["b_expected_loss_log"].to_numpy(dtype=float), 0, 50))
            return dict(zip(frame.finding_ids, (float(value) for value in values)))
        return {finding_id: 1.0 for finding_id in frame.finding_ids}

    def _cvss_lookup(self, frame: FeatureFrame) -> dict[str, float]:
        """CVSS base score per finding, for the Kendall tau comparison.

        Findings scoring zero are dropped: ``cvss_base_max`` is zero exactly when the
        finding carries no CVE, and a missing score is not a score of zero.
        """
        if self.cvss_base is not None:
            return {
                finding_id: float(value)
                for finding_id, value in self.cvss_base.items()
                if finding_id in set(frame.finding_ids)
            }
        if "cvss_base_max" in frame.X.columns:
            values = frame.X["cvss_base_max"].to_numpy(dtype=float)
            return {
                finding_id: float(value)
                for finding_id, value in zip(frame.finding_ids, values)
                if value > 0.0
            }
        return {}

    def _supplied_probabilities(self, name: RankerName) -> Mapping[str, float] | None:
        if not self.probabilities:
            return None
        for key in (name, name.value):
            if key in self.probabilities:
                return self.probabilities[key]
        return None

    def _probability_vector(
        self, name: RankerName, frame: FeatureFrame, scores: np.ndarray
    ) -> np.ndarray:
        """A value in ``[0, 1]`` per row for the threshold-dependent metrics.

        When a calibrated head has supplied probabilities they are used directly.
        Otherwise ranking scores are min-max normalised over the fold. That normalisation
        is **not** a calibration - it is a monotone rescaling that lets MCC, minority F1
        and balanced accuracy be computed at a fixed threshold on identical terms for
        every ranker. Brier and ECE are deliberately *not* computed from it.
        """
        supplied = self._supplied_probabilities(name)
        if supplied is not None:
            return np.asarray(
                [float(supplied.get(finding_id, 0.0)) for finding_id in frame.finding_ids],
                dtype=float,
            ).clip(0.0, 1.0)
        if len(scores) == 0:
            return scores
        low, high = float(np.min(scores)), float(np.max(scores))
        if high - low <= 0.0:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - low) / (high - low)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

_SIMPLE_METRICS: dict[str, MetricName] = {
    MetricName.MAP.value: MetricName.MAP,
    MetricName.MRR.value: MetricName.MRR,
    MetricName.KENDALL_TAU_VS_CVSS.value: MetricName.KENDALL_TAU_VS_CVSS,
    MetricName.MEAN_RANK_OF_EXPLOITED.value: MetricName.MEAN_RANK_OF_EXPLOITED,
    MetricName.EFFICIENCY.value: MetricName.EFFICIENCY,
    MetricName.COVERAGE.value: MetricName.COVERAGE,
    MetricName.WORKLOAD_REDUCTION.value: MetricName.WORKLOAD_REDUCTION,
}

_AT_K_METRICS: dict[str, MetricName] = {
    "ndcg": MetricName.NDCG_AT_K,
    "precision": MetricName.PRECISION_AT_K,
    "recall": MetricName.RECALL_AT_K,
    "risk_capture": MetricName.RISK_CAPTURE_AT_K,
}


def _aggregate(per_scan: Mapping[str, Mapping[str, float]], config: PipelineConfig) -> list[MetricValue]:
    """Fold means with bootstrap intervals over the per-scan values."""
    values: list[MetricValue] = []
    for key, by_scan in per_scan.items():
        sample = [value for value in by_scan.values() if np.isfinite(value)]
        if not sample:
            continue
        if "@" in key:
            stem, _, raw_k = key.partition("@")
            name, k = _AT_K_METRICS[stem], int(raw_k)
        else:
            name, k = _SIMPLE_METRICS[key], None
        low, high = bootstrap_mean_ci(sample, iters=config.evaluation.bootstrap_iters, seed=config.seed)
        values.append(
            MetricValue(
                name=name,
                k=k,
                value=float(np.mean(sample)),
                ci_low=low,
                ci_high=high,
                n=len(sample),
            )
        )
    return values


def _classification_values(
    truth: np.ndarray, scores: np.ndarray, probability: np.ndarray, config: PipelineConfig
) -> list[MetricValue]:
    """Pooled classification metrics over the whole test fold.

    ROC-AUC and PR-AUC use the raw ranking scores because both are rank-based and so are
    invariant to any monotone rescaling. The threshold-dependent three use the
    probability vector.
    """
    predicted = (probability >= config.evaluation.classification_threshold).astype(float)
    n = int(len(truth))
    return [
        MetricValue(name=MetricName.ROC_AUC, value=roc_auc(truth, scores), n=n),
        MetricValue(name=MetricName.PR_AUC, value=pr_auc(truth, scores), n=n),
        MetricValue(name=MetricName.MCC, value=mcc(truth, predicted), n=n),
        MetricValue(name=MetricName.F1_MINORITY, value=f1_minority(truth, predicted), n=n),
        MetricValue(
            name=MetricName.BALANCED_ACCURACY, value=balanced_accuracy(truth, predicted), n=n
        ),
    ]


def _normalise_folds(frames_by_split: Any) -> list[SplitFrames]:
    """Accept a mapping of split to ``(train, test)`` or a sequence of :class:`SplitFrames`."""
    folds: list[SplitFrames] = []
    if isinstance(frames_by_split, MappingABC):
        items: Iterable[Any] = frames_by_split.items()
        for key, value in items:
            if isinstance(value, SplitFrames):
                folds.append(value)
            elif isinstance(key, Split):
                train, test = value
                folds.append(SplitFrames(split=key, train=train, test=test))
            else:
                raise TypeError(
                    "frames_by_split keys must be Split objects when values are (train, test) pairs"
                )
    else:
        for item in frames_by_split:
            if isinstance(item, SplitFrames):
                folds.append(item)
            else:
                split, train, test = item
                folds.append(SplitFrames(split=split, train=train, test=test))
    return sorted(folds, key=lambda item: item.split.fold)


def _resolve_ranker(item: Any, config: PipelineConfig) -> tuple[RankerName, Any]:
    """Turn a ranker name or instance into ``(name, instance)``.

    ``vulnpriority.rank`` is imported here and nowhere else in the module: importing it
    registers the built-in rankers with ``core.registry``, and deferring it to call time
    means this module stays importable (and cheap) when the ranking package is absent or
    only the caller's own ranker instances are used.
    """
    if isinstance(item, tuple) and len(item) == 2:
        name, instance = item
        return (RankerName(name), instance)
    if isinstance(item, Ranker) or (hasattr(item, "score") and hasattr(item, "fit")):
        return (RankerName(getattr(item, "name", RankerName.RANDOM)), item)

    name = RankerName(item)
    import vulnpriority.rank  # noqa: F401  (registers the built-in rankers)
    from vulnpriority.core.registry import get_ranker

    cls = get_ranker(name)
    for attempt in (lambda: cls(config), lambda: cls(config=config), lambda: cls()):
        try:
            return (name, attempt())
        except TypeError:
            continue
    raise TypeError(f"cannot construct ranker {name.value}: no supported constructor signature")


def flags_of(frames: Sequence[SplitFrames]) -> ComponentFlags:
    """The component flags the folds were built under (they must agree)."""
    distinct = {fold.test.flags for fold in frames}
    if len(distinct) > 1:
        raise ValueError("benchmark folds disagree about which components are enabled")
    return distinct.pop() if distinct else ComponentFlags()
