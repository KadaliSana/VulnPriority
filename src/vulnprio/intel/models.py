"""Data shapes for internet-sourced exploit intelligence.

**The contract types live in :mod:`vulnprio.core.models`, not here.** ``EnrichedFinding``
carries an ``IntelResult``, and ``EnrichedFinding`` lives in core, so the retrieval layer's
output has to be a core type or the layering inverts -- exactly as
``ExploitabilityAssessment`` and ``ChainScore`` already work. The feature builder, the SHAP
explainer, the rank guard and the report all reach the result through the enriched finding
rather than through a side map threaded into three call sites, which is what makes it
impossible for one of those call sites to be silently forgotten.

This module re-exports them so ``vulnprio.intel`` remains a coherent namespace to import
from, and adds only the shapes that are genuinely internal to retrieval:
:class:`IntelGather` (what one provider call returned, including its failures) and the
wire-format JSON Schema for phase two.

The invariants worth restating, because they are what the consumers assume:

* ``IntelDocument.snippet`` is always ``Provenance.REFERENCE_PAGE``. The core model
  enforces it. The explainer tiers the six claim-bearing intel features as reference-page
  evidence on that basis; if this ever admits another provenance, that tiering is wrong.
* ``IntelResult.feature_values()`` is authoritative. The ranking layer applies ``log1p`` to
  ``a_intel_documents`` and nothing else, and reimplements none of the derivation.
* An :class:`IntelSummary` cannot exist without at least one citation. Not "is rejected
  somewhere" -- cannot be constructed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict

from vulnprio.core.config import IntelConfig
from vulnprio.core.enums import (
    AgreementAxis,
    AsOfStatus,
    IntelQueryKind,
    IntelRemedy,
    IntelSourceKind,
)
from vulnprio.core.models import (
    INTEL_FEATURE_NAMES,
    MAX_CITED_TEXT_CHARS,
    MAX_CLAIMED_VERSIONS,
    MAX_EXPLOIT_URLS,
    MAX_INTEL_CITATIONS,
    MAX_INTEL_PRECONDITIONS,
    MAX_INTEL_SUMMARY_CHARS,
    MAX_INTEL_URL_CHARS,
    AsOfVerdict,
    ExploitIntelOut,
    FeedAgreement,
    IntelCitation,
    IntelDocument,
    IntelQuery,
    IntelResult,
    IntelSummary,
    IntelUsage,
    neutral_intel_features,
)

__all__ = [
    "MAX_CITATIONS",
    "MAX_CITED_TEXT_CHARS",
    "MAX_SUMMARY_CHARS",
    "MAX_CLAIMED_VERSIONS",
    "MAX_EXPLOIT_URLS",
    "INTEL_FEATURE_NAMES",
    "IntelSourceKind",
    "IntelQueryKind",
    "IntelQuery",
    "IntelDocument",
    "IntelCitation",
    "IntelUsage",
    "ExploitIntelOut",
    "EXPLOIT_INTEL_JSON_SCHEMA",
    "AgreementAxis",
    "FeedAgreement",
    "AsOfStatus",
    "IntelRemedy",
    "AsOfVerdict",
    "IntelSummary",
    "IntelGather",
    "IntelResult",
    "IntelConfig",
    "neutral_intel_features",
    "utc_now",
]

#: Kept as aliases of the core caps so existing call sites and tests keep reading
#: naturally. There is one definition of each number, in core.
MAX_CITATIONS = MAX_INTEL_CITATIONS
MAX_SUMMARY_CHARS = MAX_INTEL_SUMMARY_CHARS
MAX_URL_CHARS = MAX_INTEL_URL_CHARS


def utc_now() -> datetime:
    """Timezone-aware now. Injected in tests so ``retrieved_at`` is deterministic."""
    return datetime.now(timezone.utc)


#: JSON Schema sent as ``output_config.format`` in phase 2.
#:
#: Written out rather than generated from ``model_json_schema()`` for two reasons: the
#: provider's structured-output mode wants a flat, self-contained schema with
#: ``additionalProperties: false`` and every property required, and an explicit schema is
#: what an auditor actually wants to read. ``test_intel_summarize`` asserts it stays in
#: step with :class:`~vulnprio.core.models.ExploitIntelOut`, so drift fails the build
#: rather than a run.
EXPLOIT_INTEL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "exploit_maturity": {
            "type": "integer",
            "enum": [0, 1, 2, 3, 4],
            "description": "0 unknown, 1 unproven, 2 proof-of-concept, 3 functional, 4 weaponized",
        },
        "exploit_feasibility": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "attack_complexity": {"type": "string", "enum": ["low", "high", "unknown"]},
        "preconditions": {
            "type": "array",
            "maxItems": MAX_INTEL_PRECONDITIONS,
            "items": {"type": "string", "maxLength": 200},
        },
        "affected_versions_claimed": {
            "type": "array",
            "maxItems": MAX_CLAIMED_VERSIONS,
            "items": {"type": "string", "maxLength": 120},
        },
        "public_exploit_urls": {
            "type": "array",
            "maxItems": MAX_EXPLOIT_URLS,
            "items": {"type": "string", "maxLength": MAX_INTEL_URL_CHARS},
        },
        "active_exploitation_claimed": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string", "maxLength": 600},
        "evidence_spans": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string", "maxLength": 200},
            "description": "Each span must be copied character for character from the data shown.",
        },
    },
    "required": [
        "exploit_maturity",
        "exploit_feasibility",
        "attack_complexity",
        "preconditions",
        "affected_versions_claimed",
        "public_exploit_urls",
        "active_exploitation_claimed",
        "confidence",
        "rationale",
        "evidence_spans",
    ],
    "additionalProperties": False,
}


class IntelGather(BaseModel):
    """Everything one phase-1 provider call produced, including its failures.

    Internal to retrieval: a provider's return value, not part of the contract anything
    downstream consumes. What survives into :class:`~vulnprio.core.models.IntelResult` is
    the sanitized, deduplicated, budget-capped remainder.
    """

    model_config = ConfigDict(frozen=True)

    documents: tuple[IntelDocument, ...] = ()
    narrative: str = ""
    citations: tuple[IntelCitation, ...] = ()
    usage: IntelUsage = IntelUsage()
    errors: tuple[str, ...] = ()
    model: str = ""
    provider: str = ""
