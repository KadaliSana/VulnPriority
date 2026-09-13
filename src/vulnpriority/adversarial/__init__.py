"""Adversarial robustness evaluation (DESIGN.md 3.10, ADR-002, Architecture item 4).

The review behind this framework found that three of eighty-four surveyed studies tested
adversarial robustness at all. This package is the answer to that half of Gap 4: a versioned
corpus of attacks and benign controls, an injector that places each payload exactly where an
adversary could write it, two detectors that bracket the model call, and an evaluator that
turns the whole thing into an :class:`~vulnpriority.core.models.AdversarialReport` whose
``attack_success_rate`` and ``canary_leak_rate`` are assertions in the test suite rather than
sentences in a paper.

Typical use::

    from vulnpriority.adversarial import AdversarialEvaluator, load_corpus

    version, cases = load_corpus()
    report = AdversarialEvaluator(scan, intel).run(run_pipeline, cases, config.adversarial)
    assert not threshold_violations(report, config.adversarial)

Nothing here touches the sandbox package: the pattern library stays the single source of
truth for what an injection looks like, and this package only measures it.
"""

from __future__ import annotations

from vulnpriority.adversarial.corpus import (
    ATTACK_CATEGORIES,
    KNOWN_PATTERN_GAPS,
    MIN_ATTACK_CASES,
    MIN_BENIGN_CONTROLS,
    VALID_INJECTION_POINTS,
    CorpusStats,
    attack_cases,
    benign_controls,
    case_by_id,
    cases_in_category,
    corpus_stats,
    default_corpus_path,
    load_corpus,
    validate_corpus,
)
from vulnpriority.adversarial.detectors import (
    DEFAULT_HIGH_CONFIDENCE,
    DEFAULT_MAX_FEATURE_JUMP,
    NORMALISATION_PREFIX,
    AssessmentSnapshot,
    CoverageReport,
    DocumentScore,
    OutputAnomalyDetector,
    PreLLMPatternDetector,
    TrustedEvidence,
    measure_pattern_coverage,
)
from vulnpriority.adversarial.evaluator import (
    DEFAULT_GOAL_BY_CATEGORY,
    FEATURE_TOLERANCE,
    MOVEMENT_EPSILON,
    OVERALL_KEY,
    SYSTEM_PROMPT_MARKERS,
    AdversarialEvaluator,
    OutcomeVerdict,
    PipelineOutput,
    classify,
    coerce_output,
    containment_breach_rate,
    goal_directed_rank_success_rate,
    split_cases,
    threshold_violations,
)
from vulnpriority.adversarial.injector import (
    PAYLOAD_SEPARATOR,
    InjectionSite,
    inject,
    inject_many,
    plan_injection,
    select_endpoint,
    select_finding,
    select_intel_key,
    site_text,
    stable_index,
    target_finding_id,
)

__all__ = [
    # corpus
    "MIN_ATTACK_CASES",
    "MIN_BENIGN_CONTROLS",
    "ATTACK_CATEGORIES",
    "VALID_INJECTION_POINTS",
    "KNOWN_PATTERN_GAPS",
    "CorpusStats",
    "load_corpus",
    "validate_corpus",
    "corpus_stats",
    "default_corpus_path",
    "attack_cases",
    "benign_controls",
    "cases_in_category",
    "case_by_id",
    # injector
    "PAYLOAD_SEPARATOR",
    "InjectionSite",
    "inject",
    "inject_many",
    "plan_injection",
    "site_text",
    "stable_index",
    "select_finding",
    "select_endpoint",
    "select_intel_key",
    "target_finding_id",
    # detectors
    "DEFAULT_MAX_FEATURE_JUMP",
    "DEFAULT_HIGH_CONFIDENCE",
    "NORMALISATION_PREFIX",
    "DocumentScore",
    "TrustedEvidence",
    "AssessmentSnapshot",
    "CoverageReport",
    "PreLLMPatternDetector",
    "OutputAnomalyDetector",
    "measure_pattern_coverage",
    # evaluator
    "FEATURE_TOLERANCE",
    "MOVEMENT_EPSILON",
    "OVERALL_KEY",
    "SYSTEM_PROMPT_MARKERS",
    "DEFAULT_GOAL_BY_CATEGORY",
    "OutcomeVerdict",
    "classify",
    "PipelineOutput",
    "coerce_output",
    "AdversarialEvaluator",
    "threshold_violations",
    "containment_breach_rate",
    "goal_directed_rank_success_rate",
    "split_cases",
]
