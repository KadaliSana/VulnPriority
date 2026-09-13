"""Component A: agentic semantic assessment (Research Goals 1-3).

Three questions, answered per finding, each with a deterministic structural answer that a
sandboxed model may only adjust within its trust tier's influence budget:

* **How much does this asset matter?** (:mod:`~vulnpriority.semantic.criticality`, Goal 1) --
  inferred from path tokens, method, authentication, content type, cookies, response size,
  parameter names and PII/secret markers. Never from a manual asset tag.
* **How exploitable is this finding?** (:mod:`~vulnpriority.semantic.exploitability`, Goal 2)
  -- fused from CVSS submetrics, CWE class, exploit records and KEV, plus free-text
  maturity evidence that can raise maturity by at most one step.
* **Does it apply here at all?** (:mod:`~vulnpriority.semantic.applicability`, Goal 3) --
  decided by :mod:`~vulnpriority.semantic.cpe_match` first; a tier <= 1 version mismatch is
  authoritative and no model output can overturn it.

:class:`~vulnpriority.semantic.assessor.AgenticSemanticAssessor` wires all three together.
"""

from __future__ import annotations

from vulnpriority.semantic.applicability import assess_applicability, baseline_applicability
from vulnpriority.semantic.assessor import AgenticSemanticAssessor, AssessorStats
from vulnpriority.semantic.cpe_match import (
    cpe_product_matches,
    match_affected,
    parse_version,
    version_in_range,
)
from vulnpriority.semantic.criticality import (
    assess_asset_criticality,
    structural_criticality,
)
from vulnpriority.semantic.exploitability import (
    assess_exploitability,
    baseline_exploitability,
    derive_privilege_gained,
    extract_maturity,
)
from vulnpriority.semantic.lexicon import (
    PII_MARKERS,
    SECRET_MARKERS,
    TOKEN_LEXICON,
    classify_function,
    find_pii_markers,
    find_secret_markers,
    luhn_check,
)

__all__ = [
    "AgenticSemanticAssessor",
    "AssessorStats",
    "PII_MARKERS",
    "SECRET_MARKERS",
    "TOKEN_LEXICON",
    "assess_applicability",
    "assess_asset_criticality",
    "assess_exploitability",
    "baseline_applicability",
    "baseline_exploitability",
    "classify_function",
    "cpe_product_matches",
    "derive_privilege_gained",
    "extract_maturity",
    "find_pii_markers",
    "find_secret_markers",
    "luhn_check",
    "match_affected",
    "parse_version",
    "structural_criticality",
    "version_in_range",
]
