"""The evaluation stack (DESIGN.md 3.9 and 4).

This package is where the framework's claims are made falsifiable. It closes four of the
review's gaps directly:

* **Gap 3 - circular validation.** :mod:`vulnprio.eval.labels` accepts exploitation
  ground truth from KEV, exploit evidence at the maturity floor, recorded incidents and
  the synthetic oracle, and raises ``LabelPolicyError`` when asked for anything
  CVSS-derived. Labels are version-aware and source-aware.
* **Gap 4 - no standardised protocol.** :mod:`vulnprio.eval.splits` gives time-ordered
  folds with a gap buffer (and a clearly labelled random control),
  :mod:`vulnprio.eval.metrics` implements the fixed metric battery from first principles,
  :mod:`vulnprio.eval.benchmark` runs every ranker on identical data, and
  :mod:`vulnprio.eval.bootstrap` puts an interval on every comparison.
* **Gap 5 - missing component ablation.** :mod:`vulnprio.eval.ablation` runs the full
  ``2^3`` factorial with main effects, interactions and paired intervals.
* **Gap 7 - class imbalance.** :mod:`vulnprio.eval.minority` reports MCC, balanced
  accuracy and per-class F1 with support.
* **Gap 10 - evaluation stops at prediction.** :mod:`vulnprio.eval.simulation` spends a
  weekly remediation capacity and measures exposure days.

:mod:`vulnprio.eval.report` turns all of it into ``report.md``, ``report.json`` and
figures.
"""

from __future__ import annotations

from vulnprio.eval.ablation import (
    COMPONENT_KEYS,
    INTERACTION_KEYS,
    FullFactorialAblation,
    cell_means,
    contrast,
    interactions,
    main_effects,
)
from vulnprio.eval.benchmark import BenchmarkRunner, SplitFrames, rank_from_scores, scan_order
from vulnprio.eval.bootstrap import (
    BootstrapDiff,
    WilcoxonResult,
    bootstrap_mean_ci,
    paired_bootstrap,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)
from vulnprio.eval.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_bins,
)
from vulnprio.eval.labels import (
    CVSS_DERIVED_TOKENS,
    LabelAudit,
    LabelBuilder,
    assert_not_cvss_derived,
    resolve_label_source,
)
from vulnprio.eval.metrics import (
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
from vulnprio.eval.minority import minority_report, per_class_scores
from vulnprio.eval.report import ReportArtifacts, ReportBuilder
from vulnprio.eval.simulation import FindingState, LongitudinalSimulator, PolicyTrace
from vulnprio.eval.splits import (
    LeaveOneAppOutSplitter,
    RandomSplitter,
    TimeOrderedSplitter,
    assert_no_temporal_leakage,
    build_splitter,
)

__all__ = [
    # labels
    "LabelBuilder",
    "LabelAudit",
    "CVSS_DERIVED_TOKENS",
    "resolve_label_source",
    "assert_not_cvss_derived",
    # splits
    "TimeOrderedSplitter",
    "LeaveOneAppOutSplitter",
    "RandomSplitter",
    "build_splitter",
    "assert_no_temporal_leakage",
    # ranking metrics
    "dcg_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "risk_capture_at_k",
    "average_precision_at_k",
    "mean_average_precision",
    "mean_reciprocal_rank",
    "mean_rank_of_exploited",
    "kendall_tau_vs_cvss",
    "RANK_METRICS",
    # classification metrics
    "roc_auc",
    "pr_auc",
    "mcc",
    "f1_binary",
    "f1_minority",
    "balanced_accuracy",
    # decision metrics
    "efficiency",
    "coverage",
    "workload_reduction",
    # calibration
    "brier_score",
    "expected_calibration_error",
    "reliability_bins",
    # uncertainty
    "paired_bootstrap",
    "paired_bootstrap_ci",
    "bootstrap_mean_ci",
    "wilcoxon_signed_rank",
    "BootstrapDiff",
    "WilcoxonResult",
    # minority reporting
    "minority_report",
    "per_class_scores",
    # benchmark
    "BenchmarkRunner",
    "SplitFrames",
    "scan_order",
    "rank_from_scores",
    # ablation
    "FullFactorialAblation",
    "main_effects",
    "interactions",
    "cell_means",
    "contrast",
    "COMPONENT_KEYS",
    "INTERACTION_KEYS",
    # simulation
    "LongitudinalSimulator",
    "FindingState",
    "PolicyTrace",
    # report
    "ReportBuilder",
    "ReportArtifacts",
]
