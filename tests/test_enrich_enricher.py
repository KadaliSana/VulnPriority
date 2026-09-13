"""``ContextualEnricher``: Component B end to end (DESIGN.md 3.6).

Component A is faked here with plain core models rather than imported, because the
semantic package is written in parallel and these tests must not depend on it.
"""

from __future__ import annotations

from datetime import date

import pytest

from vulnpriority.attacker.likelihood import build_evidence
from vulnpriority.attacker.model import p_exploit
from vulnpriority.attacker.presets import load_preset
from vulnpriority.core.config import PipelineConfig, SandboxConfig
from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    AttackComplexity,
    EndpointFunction,
    ExploitMaturity,
    LLMBackendKind,
    PrivilegeLevel,
    ScannerSeverity,
    TrustTier,
    UserInteraction,
)
from vulnpriority.core.interfaces import Enricher
from vulnpriority.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    ComponentFlags,
    EnrichedFinding,
    ExploitabilityAssessment,
    ImpactModel,
    InjectionSignal,
    LLMAudit,
)
from vulnpriority.core.enums import InjectionCategory
from vulnpriority.decision.impact import estimate_impact
from vulnpriority.decision.remediation_cost import estimate_cost
from vulnpriority.enrich.enricher import (
    DISABLED_IMPACT,
    DISABLED_P_EXPLOIT,
    INFLUENCED_FEATURES,
    ContextualEnricher,
    baseline_evidence,
)
from vulnpriority.enrich.trust import KEV_FLOOR_P

AS_OF = date(2024, 6, 1)


# --------------------------------------------------------------------------
# Fake Component A outputs
# --------------------------------------------------------------------------


def audit(tier: TrustTier, signals: int = 0, canary: bool = False) -> LLMAudit:
    """An audit record claiming the assessment leaned on content of ``tier``."""
    return LLMAudit(
        backend=LLMBackendKind.HEURISTIC,
        model="heuristic",
        task="test",
        max_tier_used=tier,
        canary_leaked=canary,
        signals=tuple(
            InjectionSignal(
                pattern_id=f"p{index}",
                category=InjectionCategory.INSTRUCTION_OVERRIDE,
                snippet="ignore previous instructions",
                tier=tier,
            )
            for index in range(signals)
        ),
    )


def asset_of(
    criticality: float = 0.8,
    exposure: float = 1.0,
    function: EndpointFunction = EndpointFunction.AUTH,
    tier: TrustTier | None = None,
) -> AssetCriticality:
    return AssetCriticality(
        endpoint_id="ep_login",
        function=function,
        criticality=criticality,
        data_sensitivity=0.7,
        exposure=exposure,
        is_auth_boundary=True,
        audit=None if tier is None else audit(tier),
    )


def exploitability_of(
    finding_id: str,
    feasibility: float = 0.7,
    tier: TrustTier | None = None,
) -> ExploitabilityAssessment:
    return ExploitabilityAssessment(
        finding_id=finding_id,
        exploit_feasibility=feasibility,
        exploit_maturity=ExploitMaturity.POC,
        attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegeLevel.NONE,
        user_interaction=UserInteraction.NONE,
        impact_c=0.9,
        impact_i=0.6,
        impact_a=0.3,
        privilege_gained=PrivilegeLevel.ADMIN,
        audit=None if tier is None else audit(tier),
    )


def applicability_of(
    finding_id: str,
    p_applicable: float = 0.9,
    tier: TrustTier | None = None,
) -> ApplicabilityAssessment:
    return ApplicabilityAssessment(
        finding_id=finding_id,
        verdict=ApplicabilityVerdict.APPLICABLE,
        p_applicable=p_applicable,
        audit=None if tier is None else audit(tier),
    )


@pytest.fixture
def attacker():
    return load_preset("opportunistic")


@pytest.fixture
def impact_model() -> ImpactModel:
    return ImpactModel(
        name="round",
        cost_per_record=100.0,
        records_by_function={EndpointFunction.AUTH: 1000},
        downtime_cost_per_hour=1000.0,
        downtime_hours_by_privilege={PrivilegeLevel.ADMIN: 10.0},
        integrity_loss_by_function={EndpointFunction.AUTH: 20000.0},
        regulatory_multiplier=1.2,
        reputational_fraction=0.2,
    )


@pytest.fixture
def parts(sample_scan, sample_endpoints, sample_intel):
    """``(finding, endpoint, intel)`` for the KEV-listed SQL injection in the sample scan."""
    return sample_scan.findings[0], sample_endpoints[0], (sample_intel,)


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_enricher_implements_the_frozen_interface() -> None:
    assert isinstance(ContextualEnricher(), Enricher)


def test_enriched_finding_is_complete_and_consistent(parts, attacker, impact_model) -> None:
    finding, endpoint, intel = parts
    enricher = ContextualEnricher(PipelineConfig())

    enriched = enricher.enrich(
        finding,
        endpoint,
        intel,
        asset_of(),
        exploitability_of(finding.finding_id),
        applicability_of(finding.finding_id),
        attacker,
        impact_model,
        AS_OF,
    )

    assert isinstance(enriched, EnrichedFinding)
    assert enriched.finding_id == finding.finding_id
    assert enriched.scan_id == finding.scan_id
    assert enriched.as_of == AS_OF
    assert enriched.likelihood.attacker == "opportunistic"
    assert enriched.likelihood.horizon_days == attacker.horizon_days
    assert 0.0 < enriched.likelihood.p_exploit <= 1.0
    assert enriched.expected_loss == pytest.approx(
        enriched.likelihood.p_exploit * enriched.impact.total
    )
    assert enriched.impact.total == pytest.approx(
        estimate_impact(
            finding, endpoint, asset_of(), exploitability_of(finding.finding_id), impact_model
        ).total
    )
    assert enriched.remediation.hours == pytest.approx(
        estimate_cost(finding, enricher.component_b).hours
    )
    assert enriched.flags == ComponentFlags(a=True, b=True, c=True)


def test_log_odds_terms_survive_onto_the_enriched_finding(parts, attacker, impact_model) -> None:
    """The audit trail must reach the ranker, not stop at the attacker model."""
    finding, endpoint, intel = parts
    enriched = ContextualEnricher().enrich(
        finding,
        endpoint,
        intel,
        asset_of(),
        exploitability_of(finding.finding_id),
        applicability_of(finding.finding_id),
        attacker,
        impact_model,
        AS_OF,
    )
    assert enriched.likelihood.log_odds_terms["kev"] == pytest.approx(attacker.w_kev)
    assert "feasibility" in enriched.likelihood.log_odds_terms


def test_enrichment_is_deterministic(parts, attacker, impact_model) -> None:
    finding, endpoint, intel = parts
    enricher = ContextualEnricher()

    def call():
        return enricher.enrich(
            finding,
            endpoint,
            intel,
            asset_of(),
            exploitability_of(finding.finding_id),
            applicability_of(finding.finding_id),
            attacker,
            impact_model,
            AS_OF,
        )

    first, second = call(), call()
    assert first.likelihood.p_exploit == second.likelihood.p_exploit
    assert first.expected_loss == second.expected_loss


def test_baseline_evidence_uses_structure_only(parts, attacker) -> None:
    """The trusted reference ignores Component A entirely for the influenceable features."""
    finding, endpoint, intel = parts
    assessed = build_evidence(
        finding,
        intel,
        asset_of(criticality=0.95, exposure=1.0),
        exploitability_of(finding.finding_id, feasibility=0.95),
        applicability_of(finding.finding_id, p_applicable=1.0),
        endpoint,
        AS_OF,
        attacker,
    )
    baseline = baseline_evidence(assessed, finding, endpoint)

    assert baseline["feasibility"] == pytest.approx(0.6)     # HIGH scanner severity prior
    assert baseline["asset_criticality"] == pytest.approx(0.5)
    assert baseline["exposure"] == pytest.approx(1.0)        # internet facing, unauthenticated
    assert baseline["applicability"] == pytest.approx(0.0)
    # curated-feed facts are copied through untouched
    assert baseline["kev"] == assessed["kev"] == 1.0
    assert baseline["epss_logit"] == assessed["epss_logit"]


# --------------------------------------------------------------------------
# Untrusted influence is budgeted and floored
# --------------------------------------------------------------------------


def _deflating_call(enricher: ContextualEnricher, parts, attacker, impact_model):
    """Component A outputs that argue every influenceable feature to its minimum,
    on the authority of target-authored content (tier 4, the smallest budget)."""
    finding, endpoint, intel = parts
    return enricher.enrich(
        finding,
        endpoint,
        intel,
        asset_of(criticality=0.0, exposure=0.0, tier=TrustTier.TARGET_CONTENT),
        exploitability_of(finding.finding_id, feasibility=0.0, tier=TrustTier.TARGET_CONTENT),
        applicability_of(finding.finding_id, p_applicable=0.0, tier=TrustTier.TARGET_CONTENT),
        attacker,
        impact_model,
        AS_OF,
    )


def test_target_content_deflation_is_clamped_to_its_budget(parts, attacker, impact_model) -> None:
    enriched = _deflating_call(ContextualEnricher(), parts, attacker, impact_model)

    budget = SandboxConfig().influence_budget[TrustTier.TARGET_CONTENT]
    used = enriched.trust.influence_used
    assert set(used) == set(INFLUENCED_FEATURES)
    for feature in INFLUENCED_FEATURES:
        assert used[feature] == pytest.approx(budget)
        assert enriched.trust.caps_applied[feature] == pytest.approx(budget)
    assert enriched.trust.max_tier_used == TrustTier.TARGET_CONTENT


def test_kev_floor_cannot_be_breached_by_untrusted_deflation(parts, attacker, impact_model) -> None:
    """The headline security property: a blog post cannot argue KEV membership away."""
    finding, endpoint, intel = parts
    enricher = ContextualEnricher()
    enriched = _deflating_call(enricher, parts, attacker, impact_model)

    neutral = build_evidence(
        finding, intel, asset_of(), exploitability_of(finding.finding_id),
        applicability_of(finding.finding_id), endpoint, AS_OF, attacker,
    )
    trusted = p_exploit(attacker, baseline_evidence(neutral, finding, endpoint)).p_exploit

    assert enriched.trust.floor_p_exploit == pytest.approx(KEV_FLOOR_P)
    assert enriched.likelihood.p_exploit == pytest.approx(min(KEV_FLOOR_P, trusted))
    assert any("floor" in conflict for conflict in enriched.trust.conflicts)


def test_without_the_floor_the_same_deflation_succeeds(parts, attacker, impact_model) -> None:
    """Shows the floor is doing real work rather than being satisfied by accident."""
    config = PipelineConfig(sandbox=SandboxConfig(allow_downgrade_below_floor=True))
    unfloored = _deflating_call(ContextualEnricher(config), parts, attacker, impact_model)
    floored = _deflating_call(ContextualEnricher(), parts, attacker, impact_model)
    assert unfloored.likelihood.p_exploit < floored.likelihood.p_exploit


def test_a_reference_page_may_move_more_than_target_content(parts, attacker, impact_model) -> None:
    """Budgets are ordered by tier, and the enricher routes each feature to the right one."""
    finding, endpoint, intel = parts
    enricher = ContextualEnricher()

    def used(tier: TrustTier) -> float:
        enriched = enricher.enrich(
            finding,
            endpoint,
            intel,
            asset_of(criticality=0.0, exposure=0.0, tier=tier),
            exploitability_of(finding.finding_id, feasibility=0.0, tier=tier),
            applicability_of(finding.finding_id, p_applicable=0.0, tier=tier),
            attacker,
            impact_model,
            AS_OF,
        )
        return enriched.trust.influence_used["feasibility"]

    assert used(TrustTier.TARGET_CONTENT) < used(TrustTier.REFERENCE_PAGE)


def test_corroborated_inflation_is_allowed_the_wider_budget(parts, attacker, impact_model) -> None:
    """The CVE is in KEV, so an untrusted page arguing it upward is corroborating."""
    finding, endpoint, intel = parts
    enriched = ContextualEnricher().enrich(
        finding,
        endpoint,
        intel,
        asset_of(criticality=1.0, exposure=1.0, tier=TrustTier.TARGET_CONTENT),
        exploitability_of(finding.finding_id, feasibility=1.0, tier=TrustTier.TARGET_CONTENT),
        applicability_of(finding.finding_id, p_applicable=1.0, tier=TrustTier.TARGET_CONTENT),
        attacker,
        impact_model,
        AS_OF,
    )
    budget = SandboxConfig().influence_budget[TrustTier.TARGET_CONTENT]
    assert enriched.trust.corroborated is True
    assert enriched.trust.influence_used["feasibility"] > budget


def test_without_curated_corroboration_inflation_is_clamped(sample_scan, sample_endpoints, attacker, impact_model) -> None:
    """The same upward push with no KEV and no exploit evidence gets the tier budget only."""
    finding = sample_scan.findings[1]      # the XSS finding: no CVE, no intel
    endpoint = sample_endpoints[1]
    enriched = ContextualEnricher().enrich(
        finding,
        endpoint,
        (),
        asset_of(criticality=1.0, exposure=1.0, tier=TrustTier.TARGET_CONTENT),
        exploitability_of(finding.finding_id, feasibility=1.0, tier=TrustTier.TARGET_CONTENT),
        applicability_of(finding.finding_id, p_applicable=1.0, tier=TrustTier.TARGET_CONTENT),
        attacker,
        impact_model,
        AS_OF,
    )
    budget = SandboxConfig().influence_budget[TrustTier.TARGET_CONTENT]
    assert enriched.trust.corroborated is False
    assert enriched.trust.influence_used["feasibility"] == pytest.approx(budget)
    assert enriched.trust.floor_p_exploit == 0.0


def test_assessments_with_no_audit_are_treated_as_scanner_tier(parts, attacker, impact_model) -> None:
    """A deterministic heuristic assessment read no untrusted text, so it gets the scanner budget."""
    finding, endpoint, intel = parts
    enriched = ContextualEnricher().enrich(
        finding,
        endpoint,
        intel,
        asset_of(criticality=0.0, exposure=0.0),
        exploitability_of(finding.finding_id, feasibility=0.0),
        applicability_of(finding.finding_id, p_applicable=0.0),
        attacker,
        impact_model,
        AS_OF,
    )
    scanner_budget = SandboxConfig().influence_budget[TrustTier.SCANNER]
    assert enriched.trust.influence_used["feasibility"] == pytest.approx(0.6)  # inside 0.8
    assert enriched.trust.influence_used["exposure"] == pytest.approx(scanner_budget)


def test_injection_signals_and_canary_leaks_are_carried_through(parts, attacker, impact_model) -> None:
    finding, endpoint, intel = parts
    exploitability = exploitability_of(finding.finding_id)
    enriched = ContextualEnricher().enrich(
        finding,
        endpoint,
        intel,
        asset_of(tier=TrustTier.REFERENCE_PAGE),
        exploitability.model_copy(
            update={"audit": audit(TrustTier.REFERENCE_PAGE, signals=2, canary=True)}
        ),
        applicability_of(finding.finding_id),
        attacker,
        impact_model,
        AS_OF,
    )
    assert enriched.trust.injection_signal_count == 2
    assert enriched.trust.canary_leaked is True


def test_scanner_and_target_provenance_are_observed(parts, attacker, impact_model) -> None:
    """The sample SQL injection quotes a target response, so tier 4 is on the record."""
    finding, endpoint, intel = parts
    enriched = ContextualEnricher().enrich(
        finding,
        endpoint,
        intel,
        asset_of(),
        exploitability_of(finding.finding_id),
        applicability_of(finding.finding_id),
        attacker,
        impact_model,
        AS_OF,
    )
    assert enriched.trust.max_tier_used == TrustTier.TARGET_CONTENT


# --------------------------------------------------------------------------
# The disabled-component ablation cell
# --------------------------------------------------------------------------


def test_disabled_component_b_still_produces_a_valid_enriched_finding(
    parts, attacker, impact_model
) -> None:
    """The B-off cells of the 2^3 ablation must run through identical downstream code."""
    finding, endpoint, intel = parts
    config = PipelineConfig().with_flags(ComponentFlags(a=True, b=False, c=True))
    enriched = ContextualEnricher(config).enrich(
        finding,
        endpoint,
        intel,
        asset_of(),
        exploitability_of(finding.finding_id),
        applicability_of(finding.finding_id),
        attacker,
        impact_model,
        AS_OF,
    )

    assert isinstance(enriched, EnrichedFinding)
    assert enriched.flags == ComponentFlags(a=True, b=False, c=True)
    assert enriched.likelihood.p_exploit == pytest.approx(DISABLED_P_EXPLOIT)
    assert enriched.likelihood.attacker == "component_b_disabled"
    assert enriched.likelihood.log_odds_terms == {}
    assert enriched.impact.total == pytest.approx(DISABLED_IMPACT)
    assert enriched.expected_loss == pytest.approx(DISABLED_P_EXPLOIT * DISABLED_IMPACT)
    assert enriched.remediation.hours > 0.0
    assert enriched.as_of == AS_OF
    assert enriched.asset is not None and enriched.exploitability is not None


def test_disabled_component_b_is_flat_across_findings(sample_scan, sample_endpoints, attacker, impact_model) -> None:
    """Flat by construction: with B off, nothing distinguishes findings on loss."""
    config = PipelineConfig().with_flags(ComponentFlags(a=True, b=False, c=True))
    enricher = ContextualEnricher(config)

    losses = set()
    for finding, endpoint in zip(sample_scan.findings, sample_endpoints):
        enriched = enricher.enrich(
            finding,
            endpoint,
            (),
            asset_of(criticality=0.1 * len(losses)),
            exploitability_of(finding.finding_id, feasibility=0.1 * len(losses)),
            applicability_of(finding.finding_id),
            attacker,
            impact_model,
            AS_OF,
        )
        losses.add(round(enriched.expected_loss, 6))
    assert len(losses) == 1


def test_disabled_component_b_ignores_the_kev_floor(parts, attacker, impact_model) -> None:
    """No trust machinery runs in the disabled cell, so the summary is empty, not faked."""
    finding, endpoint, intel = parts
    config = PipelineConfig().with_flags(ComponentFlags(a=False, b=False, c=False))
    enriched = ContextualEnricher(config).enrich(
        finding,
        endpoint,
        intel,
        asset_of(),
        exploitability_of(finding.finding_id),
        applicability_of(finding.finding_id),
        attacker,
        impact_model,
        AS_OF,
    )
    assert enriched.trust.floor_p_exploit == 0.0
    assert enriched.trust.influence_used == {}
    assert enriched.trust.max_tier_used == TrustTier.OPERATOR


# --------------------------------------------------------------------------
# Cross-cutting sanity
# --------------------------------------------------------------------------


def test_a_high_value_kev_finding_outranks_an_informational_one(
    sample_scan, sample_endpoints, attacker, impact_model
) -> None:
    """End-to-end ordering check on the construct itself."""
    enricher = ContextualEnricher()
    sqli, info = sample_scan.findings[0], sample_scan.findings[2]

    high = enricher.enrich(
        sqli,
        sample_endpoints[0],
        (),
        asset_of(criticality=0.9, function=EndpointFunction.AUTH),
        exploitability_of(sqli.finding_id, feasibility=0.9),
        applicability_of(sqli.finding_id, p_applicable=0.95),
        attacker,
        impact_model,
        AS_OF,
    )
    low = enricher.enrich(
        info,
        sample_endpoints[2],
        (),
        asset_of(criticality=0.1, exposure=1.0, function=EndpointFunction.STATIC_CONTENT),
        exploitability_of(info.finding_id, feasibility=0.05).model_copy(
            update={"impact_c": 0.0, "impact_i": 0.0, "impact_a": 0.0,
                    "privilege_gained": PrivilegeLevel.NONE}
        ),
        applicability_of(info.finding_id, p_applicable=0.5),
        attacker,
        impact_model,
        AS_OF,
    )
    assert high.expected_loss > low.expected_loss
    assert high.likelihood.p_exploit > low.likelihood.p_exploit


def test_severity_prior_shifts_the_trusted_baseline(sample_scan, sample_endpoints, attacker, impact_model) -> None:
    """Two identical findings differing only in scanner severity get different baselines."""
    finding = sample_scan.findings[0]
    endpoint = sample_endpoints[0]
    enricher = ContextualEnricher()

    def loss(severity: ScannerSeverity) -> float:
        enriched = enricher.enrich(
            finding.model_copy(update={"scanner_severity": severity}),
            endpoint,
            (),
            asset_of(),
            exploitability_of(finding.finding_id, feasibility=0.0, tier=TrustTier.TARGET_CONTENT),
            applicability_of(finding.finding_id),
            attacker,
            impact_model,
            AS_OF,
        )
        return enriched.likelihood.p_exploit

    assert loss(ScannerSeverity.CRITICAL) > loss(ScannerSeverity.LOW)
