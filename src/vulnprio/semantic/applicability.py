"""Goal 3: does this vulnerability actually apply to the observed application?

The order of authority here is the whole point of the module. Version evidence decides
first, through :mod:`vulnprio.semantic.cpe_match`. When that evidence comes from tier <= 1
data -- an NVD affected-product range checked against a scanner fingerprint -- a
``MISMATCH`` is *authoritative*: the model is still consulted, because it may know
something about preconditions that version ranges cannot express, but its verdict and its
probability are discarded. A confident sentence in a reference page cannot resurrect a
finding that the version data has ruled out.

That is a security property, not a style preference: reference pages and response bodies
are attacker-writable, and "this CVE does apply, p=0.99" is exactly what an attacker
wanting to flood the queue with noise would write.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from vulnprio.core.config import PipelineConfig
from vulnprio.core.enums import ApplicabilityVerdict, TrustTier, VersionMatch
from vulnprio.core.interfaces import LLMBackend, Sanitizer
from vulnprio.core.models import (
    ApplicabilityAssessment,
    Finding,
    TechComponent,
    UntrustedText,
    VulnIntel,
)
from vulnprio.semantic.criticality import (
    ModelCall,
    apply_budget,
    clamp01,
    consult_model,
    heuristic_audit,
)
from vulnprio.semantic.cpe_match import match_affected

__all__ = [
    "P_MATCH",
    "P_MISMATCH",
    "P_UNKNOWN",
    "AUTHORITATIVE_MISMATCH_CAP",
    "APPLICABLE_THRESHOLD",
    "NOT_APPLICABLE_THRESHOLD",
    "ApplicabilityOut",
    "version_evidence",
    "baseline_applicability",
    "verdict_for",
    "assess_applicability",
]


try:  # pragma: no cover - exercised only once vulnprio.llm exists
    from vulnprio.llm.schemas import ApplicabilityOut  # type: ignore[no-redef]
except Exception:  # pragma: no cover - parallel-development fallback

    class ApplicabilityOut(BaseModel):
        """Bounded applicability output; out-of-range values are unrepresentable."""

        model_config = ConfigDict(frozen=True)

        verdict: ApplicabilityVerdict = ApplicabilityVerdict.UNCERTAIN
        p_applicable: float = Field(0.5, ge=0.0, le=1.0)
        preconditions_met: dict[str, bool] = Field(default_factory=dict)
        confidence: float = Field(0.5, ge=0.0, le=1.0)
        rationale: str = Field("", max_length=600)
        evidence_spans: tuple[str, ...] = ()


#: Probability priors per version verdict.
P_MATCH = 0.85
P_MISMATCH = 0.03
P_UNKNOWN = 0.50

#: Hard ceiling on ``p_applicable`` once an authoritative mismatch is established.
#: Nothing downstream -- model, reference page or response body -- may raise it past this.
AUTHORITATIVE_MISMATCH_CAP = 0.05

APPLICABLE_THRESHOLD = 0.65
NOT_APPLICABLE_THRESHOLD = 0.35


def verdict_for(p_applicable: float) -> ApplicabilityVerdict:
    """Map a probability onto the three-valued verdict the rest of the framework uses."""
    if p_applicable >= APPLICABLE_THRESHOLD:
        return ApplicabilityVerdict.APPLICABLE
    if p_applicable <= NOT_APPLICABLE_THRESHOLD:
        return ApplicabilityVerdict.NOT_APPLICABLE
    return ApplicabilityVerdict.UNCERTAIN


def _observed_components(
    finding: Finding, tech: Sequence[TechComponent]
) -> tuple[TechComponent, ...]:
    """Observed stack for the match, with the finding's own affected component first."""
    components: list[TechComponent] = []
    if finding.affected_component is not None:
        components.append(finding.affected_component)
    for component in tech:
        if component not in components:
            components.append(component)
    return tuple(components)


def version_evidence(
    finding: Finding,
    intel: Sequence[VulnIntel],
    tech: Sequence[TechComponent],
) -> tuple[VersionMatch, str, TrustTier]:
    """Version verdict, its reason, and the trust tier of the data that produced it.

    The tier is returned explicitly because authority depends on it: affected-product
    ranges come from a curated feed (tier 1), so a mismatch derived from them outranks
    anything a model says. If the ranges ever arrived from a less trusted place, the
    same code path would stop being authoritative without any other change.
    """
    products = tuple(product for item in intel for product in item.affected)
    if not products:
        return VersionMatch.UNKNOWN, "no affected-product data available for this CVE", TrustTier.SCANNER
    verdict, reason = match_affected(_observed_components(finding, tech), products)
    return verdict, reason, TrustTier.CURATED_FEED


def baseline_applicability(
    finding: Finding,
    intel: Sequence[VulnIntel] = (),
    tech: Sequence[TechComponent] = (),
) -> tuple[ApplicabilityAssessment, bool]:
    """Model-free applicability plus a flag saying whether the verdict is authoritative.

    A finding with no CVE at all was observed directly by the scanner, so applicability
    is the scanner's confidence rather than a version question; that path is never
    authoritative and is exactly where the model earns its keep.
    """
    intel = tuple(intel)
    tech = tuple(tech)
    match, reason, tier = version_evidence(finding, intel, tech)

    authoritative = match == VersionMatch.MISMATCH and tier <= TrustTier.CURATED_FEED
    preconditions_met: dict[str, bool] = {}

    if match == VersionMatch.MISMATCH:
        p_applicable = P_MISMATCH
        confidence = 0.85
    elif match == VersionMatch.MATCH:
        p_applicable = P_MATCH
        confidence = 0.80
    elif not finding.cve_ids:
        # Directly observed by the scanner: no version range can confirm or deny it.
        p_applicable = clamp01(0.50 + 0.40 * finding.scanner_confidence)
        confidence = clamp01(0.40 + 0.30 * finding.scanner_confidence)
        reason = (
            f"finding carries no CVE; applicability follows the scanner's own confidence "
            f"({finding.scanner_confidence:.2f})"
        )
    else:
        p_applicable = P_UNKNOWN
        confidence = 0.40
        preconditions_met["version_confirmed"] = False

    assessment = ApplicabilityAssessment(
        finding_id=finding.finding_id,
        verdict=verdict_for(p_applicable),
        p_applicable=p_applicable,
        version_match=match,
        preconditions_met=preconditions_met,
        confidence=confidence,
        rationale=reason[:600],
    )
    return assessment, authoritative


def _untrusted_segments(
    finding: Finding, intel: Sequence[VulnIntel], max_references: int
) -> tuple[UntrustedText, ...]:
    segments: list[UntrustedText] = [finding.description]
    for item in intel:
        if item.description is not None:
            segments.append(item.description)
    references = [reference.content for item in intel for reference in item.references]
    segments.extend(references[:max_references])
    return tuple(segments)


def _applicability_context(
    finding: Finding,
    intel: Sequence[VulnIntel],
    tech: Sequence[TechComponent],
    baseline: ApplicabilityAssessment,
    authoritative: bool,
) -> str:
    """Operator-tier structured facts, including the standing version verdict."""
    observed = [
        f"{component.vendor or '?'}/{component.product}@{component.version or '?'}"
        for component in _observed_components(finding, tech)
    ]
    ranges = [
        product.cpe
        for item in intel
        for product in item.affected
    ]
    lines = [
        f"finding_id: {finding.finding_id}",
        f"name: {finding.name}",
        f"cwe_id: {finding.cwe_id}",
        f"cve_ids: {list(finding.cve_ids)}",
        f"scanner_confidence: {finding.scanner_confidence:.2f}",
        f"observed_components: {observed}",
        f"affected_cpes: {ranges[:8]}",
        f"version_match: {baseline.version_match.value}",
        f"version_reason: {baseline.rationale}",
        f"version_verdict_is_authoritative: {authoritative}",
        "instruction: rule only on preconditions; the version verdict above is final.",
    ]
    return "\n".join(lines)


def assess_applicability(
    finding: Finding,
    intel: Sequence[VulnIntel] = (),
    tech: Sequence[TechComponent] = (),
    backend: LLMBackend | None = None,
    sandbox: Sanitizer | None = None,
    config: PipelineConfig | None = None,
) -> ApplicabilityAssessment:
    """Applicability with version evidence first and the model strictly second.

    When the version evidence is an authoritative mismatch the model is still asked --
    its precondition findings are kept and recorded -- but ``verdict``,
    ``p_applicable`` and ``version_match`` are taken from the version evidence alone and
    ``p_applicable`` is capped at :data:`AUTHORITATIVE_MISMATCH_CAP`.
    """
    config = config or PipelineConfig()
    intel = tuple(intel)
    tech = tuple(tech)
    baseline, authoritative = baseline_applicability(finding, intel, tech)
    if not config.component_a.enabled or not config.component_a.assess_applicability:
        return baseline.model_copy(update={"audit": heuristic_audit("applicability", reason="disabled")})

    segments = _untrusted_segments(finding, intel, config.component_a.max_references_in_prompt)
    call: ModelCall = consult_model(
        "applicability",
        ApplicabilityOut,
        _applicability_context(finding, intel, tech, baseline, authoritative),
        segments,
        backend,
        sandbox,
        config,
    )
    if not call.used_model:
        return baseline.model_copy(update={"audit": call.audit})

    output = call.output
    proposed_preconditions = getattr(output, "preconditions_met", None) or {}
    preconditions_met = dict(baseline.preconditions_met)
    for key, value in list(proposed_preconditions.items())[:12]:
        preconditions_met[str(key)[:80]] = bool(value)

    model_rationale = str(getattr(output, "rationale", "") or "")[:200]
    spans = tuple(dict.fromkeys(getattr(output, "evidence_spans", ()) or ()))[:12]

    if authoritative:
        # The model was heard; on the version question it is not listened to.
        preconditions_met["version_mismatch_authoritative"] = True
        return baseline.model_copy(
            update={
                "verdict": ApplicabilityVerdict.NOT_APPLICABLE,
                "p_applicable": min(baseline.p_applicable, AUTHORITATIVE_MISMATCH_CAP),
                "version_match": VersionMatch.MISMATCH,
                "preconditions_met": preconditions_met,
                "confidence": max(baseline.confidence, 0.85),
                "rationale": (
                    f"{baseline.rationale} | tier<=1 version mismatch is authoritative; "
                    f"model verdict discarded ({model_rationale})"
                )[:600],
                "evidence_spans": spans,
                "audit": call.audit,
            }
        )

    p_applicable, _ = apply_budget(
        baseline.p_applicable,
        clamp01(float(getattr(output, "p_applicable", baseline.p_applicable))),
        call.budget,
    )
    # An unmet precondition the model reported can only reduce applicability, never raise it.
    if proposed_preconditions and not all(bool(value) for value in proposed_preconditions.values()):
        p_applicable = min(p_applicable, baseline.p_applicable)

    return baseline.model_copy(
        update={
            "verdict": verdict_for(p_applicable),
            "p_applicable": p_applicable,
            "preconditions_met": preconditions_met,
            "confidence": clamp01(
                (baseline.confidence + clamp01(float(getattr(output, "confidence", baseline.confidence)))) / 2.0
            ),
            "rationale": f"{baseline.rationale} | model: {model_rationale}"[:600],
            "evidence_spans": spans,
            "audit": call.audit,
        }
    )
