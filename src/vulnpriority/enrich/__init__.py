"""Component B assembly: trust accounting and the ``ContextualEnricher``."""

from __future__ import annotations

from vulnpriority.enrich.enricher import (
    DISABLED_IMPACT,
    DISABLED_P_EXPLOIT,
    INFLUENCED_FEATURES,
    SEVERITY_FEASIBILITY_PRIOR,
    ContextualEnricher,
    baseline_evidence,
)
from vulnpriority.enrich.trust import (
    KEV_FLOOR_P,
    KEV_RANSOMWARE_FLOOR_P,
    VERIFIED_FUNCTIONAL_FLOOR_P,
    VERIFIED_WEAPONIZED_FLOOR_P,
    InfluenceRecord,
    TrustLedger,
    compute_floor,
)

__all__ = [
    "ContextualEnricher",
    "baseline_evidence",
    "DISABLED_IMPACT",
    "DISABLED_P_EXPLOIT",
    "INFLUENCED_FEATURES",
    "SEVERITY_FEASIBILITY_PRIOR",
    "InfluenceRecord",
    "TrustLedger",
    "compute_floor",
    "KEV_FLOOR_P",
    "KEV_RANSOMWARE_FLOOR_P",
    "VERIFIED_FUNCTIONAL_FLOOR_P",
    "VERIFIED_WEAPONIZED_FLOOR_P",
]
