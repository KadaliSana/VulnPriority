"""Bounded structured-output schemas for every model-facing task.

These are the only shapes a language model is ever allowed to emit. They exist so that
an out-of-range or free-form answer is *unrepresentable* rather than merely rejected
later: every numeric field is bounded to ``[0, 1]``, every categorical field is a closed
enumeration from :mod:`vulnpriority.core.enums`, free text is length-capped, and
``extra="forbid"`` makes schema smuggling (adding fields the pipeline might read) fail
validation instead of silently succeeding.

They deliberately hold **less** than the corresponding core models. A model may not set
``endpoint_id``/``finding_id`` (identity is the operator's), may not set
``evidence_features`` (those are structural measurements), and may not set
``version_match`` (DESIGN 3.5: a version verdict from tier <= 1 evidence is
authoritative and the model cannot overturn it). The ``to_*`` converters below take the
structural/heuristic baseline plus the model output and produce the core model, so no
caller ever hand-assembles one from raw model text.
"""

from __future__ import annotations

from typing import Annotated, Mapping

from pydantic import BaseModel, ConfigDict, Field

from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    AttackComplexity,
    EndpointFunction,
    ExploitMaturity,
    PrivilegeLevel,
    UserInteraction,
    VersionMatch,
)
from vulnpriority.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    ExploitabilityAssessment,
    LLMAudit,
)

__all__ = [
    "MAX_RATIONALE_CHARS",
    "MAX_EVIDENCE_SPANS",
    "MAX_EVIDENCE_SPAN_CHARS",
    "MAX_PRECONDITIONS",
    "BoundedOut",
    "AssetCriticalityOut",
    "ExploitabilityOut",
    "ApplicabilityOut",
    "SCHEMA_FOR_TASK",
    "TASK_FOR_SCHEMA",
    "schema_for_task",
    "task_for_schema",
    "to_asset_criticality",
    "to_exploitability",
    "to_applicability",
]

#: Free-text caps. A rationale is an explanation, not a channel: 600 characters is
#: enough to justify a score and far too little to smuggle a payload of substance.
MAX_RATIONALE_CHARS = 600
MAX_EVIDENCE_SPANS = 5
MAX_EVIDENCE_SPAN_CHARS = 200
MAX_PRECONDITIONS = 8
MAX_PRECONDITION_CHARS = 200

Rationale = Annotated[str, Field(max_length=MAX_RATIONALE_CHARS)]
EvidenceSpan = Annotated[str, Field(max_length=MAX_EVIDENCE_SPAN_CHARS)]
EvidenceSpans = Annotated[tuple[EvidenceSpan, ...], Field(max_length=MAX_EVIDENCE_SPANS)]
Precondition = Annotated[str, Field(max_length=MAX_PRECONDITION_CHARS)]
Preconditions = Annotated[tuple[Precondition, ...], Field(max_length=MAX_PRECONDITIONS)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class BoundedOut(BaseModel):
    """Common bounded envelope shared by every task schema.

    Frozen so a guard cannot be defeated by mutating the object after validation, and
    ``extra="forbid"`` so a reply that invents fields is rejected outright.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    confidence: Unit = 0.5
    rationale: Rationale = ""
    evidence_spans: EvidenceSpans = ()


class AssetCriticalityOut(BoundedOut):
    """What a model may decide about an endpoint's business criticality (Goal 1)."""

    function: EndpointFunction = EndpointFunction.UNKNOWN
    criticality: Unit = 0.5
    data_sensitivity: Unit = 0.5
    exposure: Unit = 0.5
    is_auth_boundary: bool = False
    is_admin_surface: bool = False


class ExploitabilityOut(BoundedOut):
    """What a model may decide about how exploitable a finding is (Goal 2)."""

    exploit_feasibility: Unit = 0.5
    exploit_maturity: ExploitMaturity = ExploitMaturity.UNKNOWN
    attack_complexity: AttackComplexity = AttackComplexity.UNKNOWN
    privileges_required: PrivilegeLevel = PrivilegeLevel.NONE
    user_interaction: UserInteraction = UserInteraction.UNKNOWN
    preconditions: Preconditions = ()
    impact_c: Unit = 0.0
    impact_i: Unit = 0.0
    impact_a: Unit = 0.0
    privilege_gained: PrivilegeLevel = PrivilegeLevel.NONE


class ApplicabilityOut(BoundedOut):
    """What a model may decide about whether a finding applies here (Goal 3).

    ``version_match`` is intentionally absent: version evidence is settled by
    ``semantic.cpe_match`` from tier <= 1 data and the model only rules on the
    preconditions that version data cannot settle.
    """

    verdict: ApplicabilityVerdict = ApplicabilityVerdict.UNCERTAIN
    p_applicable: Unit = 0.5
    preconditions_met: dict[str, bool] = Field(default_factory=dict, max_length=MAX_PRECONDITIONS)


#: Canonical task name -> schema. Used by prompts, cache keys and the backends.
SCHEMA_FOR_TASK: dict[str, type[BoundedOut]] = {
    "asset_criticality": AssetCriticalityOut,
    "exploitability": ExploitabilityOut,
    "applicability": ApplicabilityOut,
}

TASK_FOR_SCHEMA: dict[str, str] = {schema.__name__: task for task, schema in SCHEMA_FOR_TASK.items()}


def schema_for_task(task: str) -> type[BoundedOut]:
    """Schema a task must answer in. Raises ``KeyError`` for an unknown task."""
    return SCHEMA_FOR_TASK[task]


def task_for_schema(schema: type[BaseModel]) -> str:
    """Task name implied by an output schema. Raises ``KeyError`` for an unknown schema."""
    return TASK_FOR_SCHEMA[schema.__name__]


# ---------------------------------------------------------------------------
# Converters: baseline + model output -> core model
# ---------------------------------------------------------------------------


def _clip_unit(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


def _spans(out: BoundedOut) -> tuple[str, ...]:
    return tuple(span for span in out.evidence_spans if span)


def to_asset_criticality(
    baseline: AssetCriticality,
    out: AssetCriticalityOut | None = None,
    audit: LLMAudit | None = None,
    evidence_features: Mapping[str, float] | None = None,
) -> AssetCriticality:
    """Fuse a structural baseline with a model's asset judgement.

    Identity and ``evidence_features`` always come from the baseline because they are
    measurements, not opinions. When ``out`` is ``None`` the baseline is returned
    unchanged except for the audit, which is what a fallback path wants.
    """
    features = dict(evidence_features if evidence_features is not None else baseline.evidence_features)
    if out is None:
        return baseline.model_copy(update={"evidence_features": features, "audit": audit})
    return AssetCriticality(
        endpoint_id=baseline.endpoint_id,
        function=out.function,
        criticality=_clip_unit(out.criticality),
        data_sensitivity=_clip_unit(out.data_sensitivity),
        exposure=_clip_unit(out.exposure),
        is_auth_boundary=out.is_auth_boundary,
        is_admin_surface=out.is_admin_surface,
        confidence=_clip_unit(out.confidence),
        rationale=out.rationale,
        evidence_spans=_spans(out),
        evidence_features=features,
        audit=audit,
    )


def to_exploitability(
    baseline: ExploitabilityAssessment,
    out: ExploitabilityOut | None = None,
    audit: LLMAudit | None = None,
) -> ExploitabilityAssessment:
    """Fuse a heuristic baseline with a model's exploitability judgement.

    Preconditions fall back to the baseline's when the model lists none, so a terse
    reply cannot erase structurally derived preconditions.
    """
    if out is None:
        return baseline.model_copy(update={"audit": audit})
    return ExploitabilityAssessment(
        finding_id=baseline.finding_id,
        exploit_feasibility=_clip_unit(out.exploit_feasibility),
        exploit_maturity=out.exploit_maturity,
        attack_complexity=out.attack_complexity,
        privileges_required=out.privileges_required,
        user_interaction=out.user_interaction,
        preconditions=tuple(out.preconditions) or baseline.preconditions,
        impact_c=_clip_unit(out.impact_c),
        impact_i=_clip_unit(out.impact_i),
        impact_a=_clip_unit(out.impact_a),
        privilege_gained=out.privilege_gained,
        confidence=_clip_unit(out.confidence),
        rationale=out.rationale,
        evidence_spans=_spans(out),
        audit=audit,
    )


def to_applicability(
    baseline: ApplicabilityAssessment,
    out: ApplicabilityOut | None = None,
    audit: LLMAudit | None = None,
) -> ApplicabilityAssessment:
    """Fuse a version-matching baseline with a model's applicability judgement.

    A ``MISMATCH`` established from version evidence is authoritative (DESIGN 3.5): the
    model may only lower ``p_applicable`` from there and may never return an
    ``APPLICABLE`` verdict over it.
    """
    if out is None:
        return baseline.model_copy(update={"audit": audit})

    p_applicable = _clip_unit(out.p_applicable)
    verdict = out.verdict
    if baseline.version_match == VersionMatch.MISMATCH:
        p_applicable = min(p_applicable, baseline.p_applicable)
        if verdict == ApplicabilityVerdict.APPLICABLE:
            verdict = baseline.verdict
    preconditions_met = dict(baseline.preconditions_met)
    preconditions_met.update(out.preconditions_met)
    return ApplicabilityAssessment(
        finding_id=baseline.finding_id,
        verdict=verdict,
        p_applicable=p_applicable,
        version_match=baseline.version_match,
        preconditions_met=preconditions_met,
        confidence=_clip_unit(out.confidence),
        rationale=out.rationale,
        evidence_spans=_spans(out),
        audit=audit,
    )
