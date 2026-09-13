"""Component B, decision half: monetary impact, unequal remediation cost, expected loss.

:mod:`vulnpriority.decision.expected_loss` carries the construct definition of priority
(Gap 1); the other two modules estimate the quantities it multiplies and the budget the
selection layer spends.
"""

from __future__ import annotations

from vulnpriority.decision.expected_loss import chain_adjusted_loss, expected_loss
from vulnpriority.decision.impact import estimate_impact, impact_components
from vulnpriority.decision.remediation_cost import (
    CWE_CLASS_HOURS,
    CWE_CLASSES,
    DEFAULT_CWE_HOURS,
    base_hours,
    cwe_class,
    estimate_cost,
)

__all__ = [
    "chain_adjusted_loss",
    "expected_loss",
    "estimate_impact",
    "impact_components",
    "CWE_CLASS_HOURS",
    "CWE_CLASSES",
    "DEFAULT_CWE_HOURS",
    "base_hours",
    "cwe_class",
    "estimate_cost",
]
