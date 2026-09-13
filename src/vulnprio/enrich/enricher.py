"""``ContextualEnricher``: Component B assembled (DESIGN.md 3.6).

This is the outermost layer of Component B and the only place in it with state. It takes
a finding plus everything Component A said about it, decides how much of that Component A
said on the authority of untrusted text, computes ``P(exploit)`` and monetary impact, and
returns the :class:`EnrichedFinding` the ranker and the attack graph both consume.

The untrusted-influence accounting works by difference. For each feature that untrusted
text can plausibly move - exploit feasibility, asset criticality, exposure, applicability -
the enricher computes a *structural baseline* from scanner-tier facts alone (severity,
whether the endpoint is internet-facing and unauthenticated) and treats the gap between
that baseline and the assessed value as the influence exercised by whatever tier produced
the assessment. That gap goes through :class:`TrustLedger`, which clamps it to the tier's
budget. The likelihood is then computed twice: once on the baseline (the trusted
reference) and once on the clamped-adjusted evidence, which is what lets the KEV floor be
applied relative to the trusted reference rather than absolutely.

When ``component_b.enabled`` is False the enricher still returns a valid, well-formed
``EnrichedFinding`` built from fixed neutral constants, because the 2^3 ablation (Gap 5)
has to be able to run the B-off cells through the identical downstream code path. The
Component B feature columns are *dropped* in that cell by the feature builder, so these
constants never leak into a model - they only keep the object graph intact.
"""

from __future__ import annotations

from datetime import date

from vulnprio.attacker.likelihood import build_evidence
from vulnprio.attacker.model import p_exploit
from vulnprio.core.config import ComponentBConfig, PipelineConfig, SandboxConfig
from vulnprio.core.enums import (
    ExploitMaturity,
    PrivilegeLevel,
    ScannerSeverity,
    TrustTier,
)
from vulnprio.core.interfaces import Enricher
from vulnprio.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    AttackerModel,
    BusinessImpact,
    ComponentFlags,
    Endpoint,
    EnrichedFinding,
    ExploitLikelihood,
    ExploitabilityAssessment,
    Finding,
    ImpactModel,
    LLMAudit,
    RemediationCost,
    TrustSummary,
    VulnIntel,
)
from vulnprio.decision.expected_loss import expected_loss
from vulnprio.decision.impact import estimate_impact
from vulnprio.decision.remediation_cost import estimate_cost
from vulnprio.enrich.trust import TrustLedger, compute_floor

__all__ = [
    "INFLUENCED_FEATURES",
    "SEVERITY_FEASIBILITY_PRIOR",
    "DISABLED_P_EXPLOIT",
    "DISABLED_IMPACT",
    "baseline_evidence",
    "ContextualEnricher",
]

#: Features an untrusted document could plausibly argue about, and which therefore go
#: through the influence budget. Everything else in the evidence vector is a curated-feed
#: fact (KEV, EPSS) or an attacker-model constant.
INFLUENCED_FEATURES: tuple[str, ...] = (
    "feasibility",
    "asset_criticality",
    "exposure",
    "applicability",
)

#: Scanner-tier prior for exploit feasibility, keyed by the scanner's own severity. This
#: is the "what would we have believed without Component A" number.
SEVERITY_FEASIBILITY_PRIOR: dict[ScannerSeverity, float] = {
    ScannerSeverity.INFO: 0.10,
    ScannerSeverity.LOW: 0.20,
    ScannerSeverity.MEDIUM: 0.40,
    ScannerSeverity.HIGH: 0.60,
    ScannerSeverity.CRITICAL: 0.80,
}

#: Structural prior for asset criticality: deliberately uninformative, since criticality
#: is precisely what Component A is for.
BASELINE_CRITICALITY: float = 0.5

#: Fixed prior used when Component B is disabled. Roughly the unconditional rate at which
#: web application findings are exploited; it is a constant, not an estimate, on purpose.
DISABLED_P_EXPLOIT: float = 0.05

#: Flat impact used when Component B is disabled: every finding costs the same, which is
#: exactly the assumption the ablation is testing. In the impact model's currency; a
#: proportional prior carried across from the previous $10,000 at the same 39.13x ratio
#: the impact presets use. Being flat, its absolute value cannot change any ordering -
#: it is re-denominated so that an ablation run's money figures read in the same units as
#: every other run's, not because the number itself carries information.
DISABLED_IMPACT: float = 391_000.0


def _structural_exposure(endpoint: Endpoint) -> float:
    """Exposure implied by scanner-observed structure alone."""
    if not endpoint.internet_facing:
        return 0.0
    return 1.0 if endpoint.auth_required == PrivilegeLevel.NONE else 0.5


def baseline_evidence(
    evidence: dict[str, float],
    finding: Finding,
    endpoint: Endpoint,
) -> dict[str, float]:
    """The evidence vector as it would look with no Component A influence at all.

    Curated-feed terms (EPSS, KEV, feed maturity) and attacker constants are copied
    through unchanged; only the four influenceable features fall back to structure.
    """
    baseline = dict(evidence)
    baseline["feasibility"] = SEVERITY_FEASIBILITY_PRIOR.get(finding.scanner_severity, 0.4)
    baseline["asset_criticality"] = BASELINE_CRITICALITY
    baseline["exposure"] = _structural_exposure(endpoint)
    baseline["applicability"] = 0.0  # p_applicable = 0.5, i.e. "we do not know"
    return baseline


def _audit_tier(audit: LLMAudit | None) -> TrustTier:
    """Tier that an assessment was produced on the authority of.

    No audit means the value came from deterministic scanner-tier structure (the
    heuristic backend with no untrusted text), so SCANNER is the honest default.
    """
    if audit is None:
        return TrustTier.SCANNER
    return audit.max_tier_used


def _trusted_supports_increase(intel: tuple[VulnIntel, ...], as_of: date) -> bool:
    """True when tier <= 1 evidence already argues this finding is exploitable.

    Used to decide corroboration: an untrusted page that says "this is easy to exploit"
    about a CVE that CISA already lists as exploited is adding detail, not inventing a
    claim, and earns the corroborated budget.
    """
    for record in intel:
        if record.as_of > as_of:
            continue
        kev = record.kev
        if kev is not None and kev.in_kev and not (kev.date_added is not None and kev.date_added > as_of):
            return True
        for exploit in record.exploits:
            if exploit.published is not None and exploit.published > as_of:
                continue
            if exploit.verified and exploit.maturity >= ExploitMaturity.FUNCTIONAL:
                return True
    return False


class ContextualEnricher(Enricher):
    """Component B implementation of :class:`vulnprio.core.interfaces.Enricher`."""

    def __init__(self, config: PipelineConfig | None = None, *, strict_budget: bool = False) -> None:
        """``strict_budget`` makes a budget breach raise rather than clamp.

        The full :class:`PipelineConfig` is taken rather than just
        :class:`ComponentBConfig` because the enricher needs the sandbox budgets too, and
        because ``config.flags()`` is what stamps the ablation cell onto every finding.
        """
        self.config: PipelineConfig = config if config is not None else PipelineConfig()
        self.strict_budget = strict_budget

    @property
    def component_b(self) -> ComponentBConfig:
        """Component B settings for this run."""
        return self.config.component_b

    @property
    def sandbox(self) -> SandboxConfig:
        """Sandbox settings, which own the influence budgets."""
        return self.config.sandbox

    @property
    def flags(self) -> ComponentFlags:
        """Ablation cell this enricher is running in."""
        return self.config.flags()

    # -- main entry point ---------------------------------------------------

    def enrich(
        self,
        finding: Finding,
        endpoint: Endpoint,
        intel: tuple[VulnIntel, ...],
        asset: AssetCriticality,
        exploitability: ExploitabilityAssessment,
        applicability: ApplicabilityAssessment,
        attacker: AttackerModel,
        impact_model: ImpactModel,
        as_of: date,
    ) -> EnrichedFinding:
        """Assemble the enriched finding, honouring the ablation flag for Component B."""
        if not self.component_b.enabled:
            return self._enrich_disabled(
                finding, endpoint, intel, asset, exploitability, applicability, attacker, as_of
            )

        ledger = TrustLedger(finding.finding_id, self.sandbox, strict=self.strict_budget)
        self._observe_provenance(ledger, finding, intel, asset, exploitability, applicability)

        floor, floor_reason = compute_floor(intel, as_of)
        ledger.set_floor(floor, floor_reason)

        assessed = build_evidence(
            finding, intel, asset, exploitability, applicability, endpoint, as_of, attacker
        )
        baseline = baseline_evidence(assessed, finding, endpoint)
        corroborating = _trusted_supports_increase(intel, as_of)

        adjusted = dict(baseline)
        for feature in INFLUENCED_FEATURES:
            delta = assessed[feature] - baseline[feature]
            tier = self._tier_for(feature, asset, exploitability, applicability)
            corroborated = corroborating and delta > 0.0
            adjusted[feature] = baseline[feature] + ledger.record(
                feature, tier, delta, corroborated=corroborated
            )

        trusted = p_exploit(attacker, baseline, finding_id=finding.finding_id)
        likelihood = p_exploit(attacker, adjusted, finding_id=finding.finding_id)
        floored = ledger.apply_floor(likelihood.p_exploit, trusted.p_exploit)
        if floored != likelihood.p_exploit:
            likelihood = likelihood.model_copy(update={"p_exploit": floored})

        impact = estimate_impact(finding, endpoint, asset, exploitability, impact_model)
        remediation = estimate_cost(finding, self.component_b)

        return EnrichedFinding(
            finding=finding,
            endpoint=endpoint,
            intel=intel,
            asset=asset,
            exploitability=exploitability,
            applicability=applicability,
            likelihood=likelihood,
            impact=impact,
            remediation=remediation,
            expected_loss=expected_loss(likelihood, impact),
            trust=ledger.summary(),
            flags=self.flags,
            as_of=as_of,
        )

    # -- helpers ------------------------------------------------------------

    def _tier_for(
        self,
        feature: str,
        asset: AssetCriticality,
        exploitability: ExploitabilityAssessment,
        applicability: ApplicabilityAssessment,
    ) -> TrustTier:
        """Which assessment authorised a feature, and therefore which budget applies."""
        if feature in ("asset_criticality", "exposure"):
            return _audit_tier(asset.audit)
        if feature == "applicability":
            return _audit_tier(applicability.audit)
        return _audit_tier(exploitability.audit)

    def _observe_provenance(
        self,
        ledger: TrustLedger,
        finding: Finding,
        intel: tuple[VulnIntel, ...],
        asset: AssetCriticality,
        exploitability: ExploitabilityAssessment,
        applicability: ApplicabilityAssessment,
    ) -> None:
        """Record every tier that touched this finding, plus sandbox signals and canaries."""
        ledger.note_tier(finding.description.tier)
        for item in finding.evidence:
            ledger.note_tier(item.tier)
        for record in intel:
            if record.description is not None:
                ledger.note_tier(record.description.tier)
            for reference in record.references:
                ledger.note_tier(reference.content.tier)
        for audit in (asset.audit, exploitability.audit, applicability.audit):
            if audit is None:
                continue
            ledger.note_tier(audit.max_tier_used)
            ledger.note_signals(len(audit.signals))
            ledger.note_canary(audit.canary_leaked)

    def _enrich_disabled(
        self,
        finding: Finding,
        endpoint: Endpoint,
        intel: tuple[VulnIntel, ...],
        asset: AssetCriticality,
        exploitability: ExploitabilityAssessment,
        applicability: ApplicabilityAssessment,
        attacker: AttackerModel,
        as_of: date,
    ) -> EnrichedFinding:
        """The B-off ablation cell: structurally valid, deliberately uninformative."""
        likelihood = ExploitLikelihood(
            finding_id=finding.finding_id,
            attacker="component_b_disabled",
            p_exploit=DISABLED_P_EXPLOIT,
            p_exploit_uncapped=DISABLED_P_EXPLOIT,
            log_odds_terms={},
            horizon_days=attacker.horizon_days,
        )
        impact = BusinessImpact(
            finding_id=finding.finding_id,
            total=DISABLED_IMPACT,
            rationale="component B disabled: flat impact constant",
        )
        hours = max(float(self.component_b.remediation_default_hours), 0.25)
        remediation = RemediationCost(
            finding_id=finding.finding_id,
            hours=hours,
            cost=hours * float(self.component_b.remediation_hourly_rate),
            basis="component B disabled: flat default hours",
        )
        return EnrichedFinding(
            finding=finding,
            endpoint=endpoint,
            intel=intel,
            asset=asset,
            exploitability=exploitability,
            applicability=applicability,
            likelihood=likelihood,
            impact=impact,
            remediation=remediation,
            expected_loss=expected_loss(likelihood, impact),
            trust=TrustSummary(),
            flags=self.flags,
            as_of=as_of,
        )
