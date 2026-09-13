"""The learning-to-rank stack: features, LambdaMART, baselines, explanation, rank guard.

Importing this package registers every ranker in
:data:`vulnprio.core.registry.RANKERS`, which is what makes
``get_ranker(RankerName.LAMBDAMART)`` and the benchmark runner's baseline sweep work. The
registration happens as a side effect of importing :mod:`vulnprio.rank.lambdamart` and
:mod:`vulnprio.rank.baselines`, so the names below are re-exported eagerly rather than
lazily.

Reading order, if you are meeting this package for the first time: ``features`` (what the
model sees, and what an ablation cell removes), ``lambdamart`` (how the ordering is
learned and constrained), ``baselines`` (what it has to beat), ``explain`` (why a finding
is where it is), ``compose`` (score vector to remediation queue) and ``rank_guard`` (what
happens when someone tries to move the queue from outside).
"""

from __future__ import annotations

from vulnprio.rank.baselines import (
    KEV_BAND,
    VMC_BAND_EXPLOITED,
    VMC_BAND_REST,
    VMC_BAND_SEVERE,
    VMC_CVSS_THRESHOLD,
    VMC_EPSS_THRESHOLD,
    Baseline,
    CvssOnlyRanker,
    EpssOnlyRanker,
    ExpectedLossRanker,
    KevFirstRanker,
    RandomRanker,
    ScannerSeverityRanker,
    VmcChainRanker,
)
from vulnprio.rank.compose import order_within_groups, rank_scan
from vulnprio.rank.explain import (
    DEFAULT_MAX_REASON_CODES,
    FEATURE_TIER,
    REASON_TEMPLATES,
    EvidenceExplainer,
    ShapExplainer,
    build_explainer,
    evidence_reason_codes,
    feature_tier,
    safe_token,
)
from vulnprio.rank.features import (
    FEATURE_DOC,
    NEUTRAL,
    OWASP_TOP10_CWES,
    SEVERITY_ORDER,
    VERSION_MATCH_ORDER,
    FeatureBuilder,
    neutral_row,
    severity_ordinal,
    usable_intel,
    version_match_ordinal,
)
from vulnprio.rank.lambdamart import FALLBACK_COLUMNS, LambdaMartRanker
from vulnprio.rank.likelihood_head import CALIBRATION_METHODS, CostSensitiveExploitHead
from vulnprio.rank.rank_guard import (
    DEFAULT_DISPLACEMENT_MAD_MULTIPLIER,
    DEFAULT_MIN_RANK_DISPLACEMENT,
    EVIDENCE_SOURCE_PHRASE,
    MAD_TO_SIGMA,
    FLOOR_CONFLICT_MARKER,
    NOT_APPLICABLE_P_THRESHOLD,
    RankManipulationDetector,
    neutralise_component_a,
    ordinal,
)

__all__ = [
    # features
    "FeatureBuilder",
    "FEATURE_DOC",
    "NEUTRAL",
    "OWASP_TOP10_CWES",
    "SEVERITY_ORDER",
    "VERSION_MATCH_ORDER",
    "neutral_row",
    "severity_ordinal",
    "usable_intel",
    "version_match_ordinal",
    # learned ranker and probability head
    "LambdaMartRanker",
    "FALLBACK_COLUMNS",
    "CostSensitiveExploitHead",
    "CALIBRATION_METHODS",
    # baselines
    "Baseline",
    "CvssOnlyRanker",
    "EpssOnlyRanker",
    "KevFirstRanker",
    "ScannerSeverityRanker",
    "ExpectedLossRanker",
    "VmcChainRanker",
    "RandomRanker",
    "KEV_BAND",
    "VMC_BAND_EXPLOITED",
    "VMC_BAND_SEVERE",
    "VMC_BAND_REST",
    "VMC_CVSS_THRESHOLD",
    "VMC_EPSS_THRESHOLD",
    # explanation
    "ShapExplainer",
    "EvidenceExplainer",
    "build_explainer",
    "evidence_reason_codes",
    "FEATURE_TIER",
    "REASON_TEMPLATES",
    "DEFAULT_MAX_REASON_CODES",
    "feature_tier",
    "safe_token",
    # composition and guard
    "rank_scan",
    "order_within_groups",
    "RankManipulationDetector",
    "neutralise_component_a",
    "ordinal",
    "DEFAULT_DISPLACEMENT_MAD_MULTIPLIER",
    "DEFAULT_MIN_RANK_DISPLACEMENT",
    "MAD_TO_SIGMA",
    "NOT_APPLICABLE_P_THRESHOLD",
    "FLOOR_CONFLICT_MARKER",
    "EVIDENCE_SOURCE_PHRASE",
]
