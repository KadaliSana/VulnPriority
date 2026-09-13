"""Influence budgets, KEV floors and corroboration (DESIGN.md 3.3 item 7).

This is the security contract of Component B, so the tests are written adversarially:
each one asks what an attacker who controls the text would try, and asserts that it fails.
"""

from __future__ import annotations

from datetime import date

import pytest

from vulnpriority.core.config import SandboxConfig
from vulnpriority.core.enums import (
    ExploitMaturity,
    ExploitSource,
    Provenance,
    TrustTier,
)
from vulnpriority.core.errors import InfluenceBudgetExceeded
from vulnpriority.core.models import (
    ExploitEvidence,
    KevRecord,
    UntrustedText,
    VulnIntel,
)
from vulnpriority.enrich.trust import (
    KEV_FLOOR_P,
    KEV_RANSOMWARE_FLOOR_P,
    VERIFIED_FUNCTIONAL_FLOOR_P,
    VERIFIED_WEAPONIZED_FLOOR_P,
    TrustLedger,
    compute_floor,
)

AS_OF = date(2024, 6, 1)


def intel_with_kev(ransomware: bool = False, date_added: date = date(2024, 2, 1)) -> VulnIntel:
    return VulnIntel(
        cve_id="CVE-2024-0001",
        as_of=AS_OF,
        kev=KevRecord(
            cve_id="CVE-2024-0001",
            in_kev=True,
            date_added=date_added,
            known_ransomware_use=ransomware,
            as_of=AS_OF,
        ),
    )


def intel_with_exploit(maturity: ExploitMaturity, verified: bool) -> VulnIntel:
    return VulnIntel(
        cve_id="CVE-2024-0002",
        as_of=AS_OF,
        exploits=(
            ExploitEvidence(
                source=ExploitSource.EXPLOIT_DB,
                published=date(2024, 3, 1),
                maturity=maturity,
                verified=verified,
            ),
        ),
    )


# --------------------------------------------------------------------------
# Floors
# --------------------------------------------------------------------------


def test_kev_membership_sets_a_floor() -> None:
    floor, reason = compute_floor((intel_with_kev(),), AS_OF)
    assert floor == pytest.approx(KEV_FLOOR_P)
    assert "CISA KEV" in reason


def test_known_ransomware_use_sets_a_higher_floor() -> None:
    floor, reason = compute_floor((intel_with_kev(ransomware=True),), AS_OF)
    assert floor == pytest.approx(KEV_RANSOMWARE_FLOOR_P)
    assert "ransomware" in reason


def test_verified_exploit_evidence_sets_a_floor_by_maturity() -> None:
    functional, _ = compute_floor((intel_with_exploit(ExploitMaturity.FUNCTIONAL, True),), AS_OF)
    weaponized, _ = compute_floor((intel_with_exploit(ExploitMaturity.WEAPONIZED, True),), AS_OF)
    assert functional == pytest.approx(VERIFIED_FUNCTIONAL_FLOOR_P)
    assert weaponized == pytest.approx(VERIFIED_WEAPONIZED_FLOOR_P)


def test_unverified_or_immature_exploit_evidence_sets_no_floor() -> None:
    """A proof-of-concept nobody checked is not grounds for a floor."""
    assert compute_floor((intel_with_exploit(ExploitMaturity.FUNCTIONAL, False),), AS_OF)[0] == 0.0
    assert compute_floor((intel_with_exploit(ExploitMaturity.POC, True),), AS_OF)[0] == 0.0


def test_a_kev_listing_after_as_of_sets_no_floor() -> None:
    """Temporal honesty: a February listing must not raise a January floor."""
    floor, _ = compute_floor((intel_with_kev(),), date(2024, 1, 15))
    assert floor == 0.0


def test_no_intel_means_no_floor() -> None:
    assert compute_floor((), AS_OF) == (0.0, "")


def test_floors_never_fall() -> None:
    ledger = TrustLedger("f1")
    ledger.set_floor(0.7, "ransomware KEV")
    ledger.set_floor(0.2, "a blog post")
    assert ledger.floor_p_exploit == pytest.approx(0.7)
    assert ledger.floor_reason == "ransomware KEV"


# --------------------------------------------------------------------------
# The floor cannot be breached by untrusted evidence
# --------------------------------------------------------------------------


def test_untrusted_evidence_cannot_argue_below_the_kev_floor() -> None:
    """"This CVE is not really exploitable" from a reference page does not win."""
    ledger = TrustLedger("f1")
    ledger.set_floor(KEV_FLOOR_P, "CVE-2024-0001 is in CISA KEV")
    ledger.record("feasibility", TrustTier.REFERENCE_PAGE, -0.9)

    held = ledger.apply_floor(0.05, trusted_reference=0.62)
    assert held == pytest.approx(KEV_FLOOR_P)
    assert any("floor" in conflict for conflict in ledger.summary().conflicts)
    assert ledger.summary().floor_p_exploit == pytest.approx(KEV_FLOOR_P)


def test_the_floor_never_invents_confidence_above_the_trusted_baseline() -> None:
    """With a trusted baseline of 0.3 the floor can restore 0.3, never manufacture 0.5."""
    ledger = TrustLedger("f1")
    ledger.set_floor(KEV_FLOOR_P, "KEV")
    assert ledger.apply_floor(0.02, trusted_reference=0.30) == pytest.approx(0.30)


def test_the_floor_is_a_no_op_when_nothing_untrusted_moved_anything() -> None:
    ledger = TrustLedger("f1")
    ledger.set_floor(KEV_FLOOR_P, "KEV")
    assert ledger.apply_floor(0.42, trusted_reference=0.42) == pytest.approx(0.42)


def test_operator_may_explicitly_disable_the_floor() -> None:
    """``allow_downgrade_below_floor`` exists so the ablation can measure what the floor buys."""
    ledger = TrustLedger("f1", SandboxConfig(allow_downgrade_below_floor=True))
    ledger.set_floor(KEV_FLOOR_P, "KEV")
    assert ledger.apply_floor(0.05, trusted_reference=0.62) == pytest.approx(0.05)


# --------------------------------------------------------------------------
# Influence budget accounting
# --------------------------------------------------------------------------


def test_target_content_is_clamped_to_its_small_budget() -> None:
    """The application describing itself gets 0.15 of movement, not 0.9."""
    sandbox = SandboxConfig()
    ledger = TrustLedger("f1", sandbox)
    applied = ledger.record("feasibility", TrustTier.TARGET_CONTENT, 0.9)

    budget = sandbox.influence_budget[TrustTier.TARGET_CONTENT]
    assert applied == pytest.approx(budget)
    summary = ledger.summary()
    assert summary.influence_used["feasibility"] == pytest.approx(budget)
    assert summary.caps_applied["feasibility"] == pytest.approx(budget)
    assert summary.max_tier_used == TrustTier.TARGET_CONTENT


def test_clamping_is_symmetric_for_deflation_attacks() -> None:
    """Arguing a finding *down* is budgeted exactly as tightly as arguing it up."""
    sandbox = SandboxConfig()
    ledger = TrustLedger("f1", sandbox)
    applied = ledger.record("asset_criticality", TrustTier.REFERENCE_PAGE, -0.8)
    assert applied == pytest.approx(-sandbox.influence_budget[TrustTier.REFERENCE_PAGE])


def test_a_delta_inside_budget_passes_through_untouched() -> None:
    ledger = TrustLedger("f1")
    assert ledger.record("exposure", TrustTier.REFERENCE_PAGE, 0.2) == pytest.approx(0.2)
    summary = ledger.summary()
    assert summary.influence_used["exposure"] == pytest.approx(0.2)
    assert "exposure" not in summary.caps_applied
    assert summary.conflicts == ()


def test_curated_feeds_and_scanner_have_larger_budgets_than_untrusted_tiers() -> None:
    """The budget ordering *is* the trust hierarchy; assert it rather than assume it."""
    sandbox = SandboxConfig()
    budgets = [
        TrustLedger("f1", sandbox).budget_for(tier)
        for tier in (
            TrustTier.CURATED_FEED,
            TrustTier.SCANNER,
            TrustTier.REFERENCE_PAGE,
            TrustTier.TARGET_CONTENT,
        )
    ]
    assert budgets == sorted(budgets, reverse=True)
    assert budgets[-1] < budgets[0]


def test_influence_accumulates_across_repeated_attempts() -> None:
    """Ten small nudges are accounted as ten small nudges, not forgotten one by one."""
    ledger = TrustLedger("f1")
    for _ in range(4):
        ledger.record("feasibility", TrustTier.TARGET_CONTENT, 0.05)
    assert ledger.summary().influence_used["feasibility"] == pytest.approx(0.20)


def test_strict_mode_raises_on_a_budget_breach() -> None:
    """The adversarial evaluation runs strict: an over-reach is itself the finding."""
    ledger = TrustLedger("f1", SandboxConfig(), strict=True)
    with pytest.raises(InfluenceBudgetExceeded) as excinfo:
        ledger.record("feasibility", TrustTier.TARGET_CONTENT, 0.9)
    assert "feasibility" in str(excinfo.value)


def test_corroborated_untrusted_claims_get_the_corroborated_budget() -> None:
    """Agreeing with CISA is adding detail, not inventing a claim."""
    sandbox = SandboxConfig(corroborated_budget=1.0)
    ledger = TrustLedger("f1", sandbox)
    applied = ledger.record("feasibility", TrustTier.TARGET_CONTENT, 0.6, corroborated=True)
    assert applied == pytest.approx(0.6)
    assert ledger.summary().corroborated is True


def test_corroboration_does_not_widen_trusted_tiers_beyond_their_budget() -> None:
    """Corroboration is a concession to untrusted tiers only; it must not inflate others."""
    sandbox = SandboxConfig(corroborated_budget=1.0)
    ledger = TrustLedger("f1", sandbox)
    assert ledger.budget_for(TrustTier.SCANNER, corroborated=True) == pytest.approx(
        sandbox.influence_budget[TrustTier.SCANNER]
    )


def test_an_uncorroborated_finding_reports_corroborated_false() -> None:
    ledger = TrustLedger("f1")
    ledger.record("feasibility", TrustTier.REFERENCE_PAGE, 0.1)
    assert ledger.summary().corroborated is False


# --------------------------------------------------------------------------
# Summary bookkeeping
# --------------------------------------------------------------------------


def test_summary_records_tiers_signals_and_canaries() -> None:
    ledger = TrustLedger("f1")
    ledger.note_tier(TrustTier.CURATED_FEED)
    ledger.note_tier(TrustTier.REFERENCE_PAGE)
    ledger.note_signals(2)
    ledger.note_signals(1)
    ledger.note_canary(False)
    ledger.note_canary(True)
    ledger.note_conflict("something disagreed")
    ledger.note_conflict("something disagreed")     # deduplicated

    summary = ledger.summary()
    assert summary.max_tier_used == TrustTier.REFERENCE_PAGE
    assert summary.injection_signal_count == 3
    assert summary.canary_leaked is True
    assert summary.conflicts == ("something disagreed",)


def test_a_clean_ledger_summarises_as_operator_tier() -> None:
    summary = TrustLedger("f1").summary()
    assert summary.max_tier_used == TrustTier.OPERATOR
    assert summary.influence_used == {}
    assert summary.caps_applied == {}
    assert summary.floor_p_exploit == 0.0
    assert summary.canary_leaked is False


def test_records_expose_the_full_audit_trail() -> None:
    ledger = TrustLedger("f1")
    ledger.record("feasibility", TrustTier.TARGET_CONTENT, 0.9)
    (entry,) = ledger.records
    assert entry.feature == "feasibility"
    assert entry.tier == TrustTier.TARGET_CONTENT
    assert entry.requested == pytest.approx(0.9)
    assert entry.was_capped is True


def test_untrusted_text_tier_is_derived_not_declared(untrusted) -> None:
    """Sanity check on the contract the ledger depends on: provenance decides the tier."""
    page = untrusted("anything at all", Provenance.REFERENCE_PAGE)
    body = UntrustedText(text="echoed", provenance=Provenance.TARGET_RESPONSE)
    assert page.tier == TrustTier.REFERENCE_PAGE
    assert body.tier == TrustTier.TARGET_CONTENT
    assert TrustLedger("f1").budget_for(body.tier) < TrustLedger("f1").budget_for(page.tier)
