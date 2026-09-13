"""Output guard: what a model is allowed to say, and how far it is allowed to move a score.

Layers 5 to 7 of ADR-002. Even a model that has been fully compromised by an injection can
only return a schema-valid object, with spans that exist in the input, moving a feature no
further than its tier's influence budget.
"""

from __future__ import annotations

import pytest

from vulnprio.core.config import load_config
from vulnprio.core.enums import InjectionCategory, TrustTier
from vulnprio.core.errors import CanaryLeakError, SchemaRejectedError
from vulnprio.llm.schemas import AssetCriticalityOut, ExploitabilityOut
from vulnprio.sandbox.output_guard import OutputGuard, clamp_influence, span_matches

SANITIZED = [
    "A proof-of-concept exploit is published and the vendor confirms remote code execution.",
    "The affected versions are 2.5.0 through 2.5.13; 2.5.14 contains the fix.",
]


@pytest.fixture(scope="module")
def guard() -> OutputGuard:
    return OutputGuard(load_config("configs/default.yaml").sandbox)


def _valid_exploitability(**overrides: object) -> dict:
    payload = {
        "confidence": 0.8,
        "rationale": "A published proof of concept exists.",
        "evidence_spans": ["A proof-of-concept exploit is published"],
        "exploit_feasibility": 0.7,
        "exploit_maturity": 2,
        "attack_complexity": "low",
        "privileges_required": 0,
        "user_interaction": "none",
        "preconditions": [],
        "impact_c": 0.9,
        "impact_i": 0.8,
        "impact_a": 0.5,
        "privilege_gained": 3,
    }
    payload.update(overrides)
    return payload


def test_valid_output_passes(guard: OutputGuard) -> None:
    parsed, signals = guard.validate(_valid_exploitability(), ExploitabilityOut, SANITIZED, "CANARY-1")
    assert isinstance(parsed, ExploitabilityOut)
    assert parsed.exploit_feasibility == pytest.approx(0.7)
    assert not signals


def test_out_of_range_numbers_cannot_be_expressed(guard: OutputGuard) -> None:
    """A schema bound is the difference between 'unlikely' and 'impossible'."""
    parsed, signals = guard.validate(
        _valid_exploitability(exploit_feasibility=999.0), ExploitabilityOut, SANITIZED, "CANARY-1"
    )
    assert 0.0 <= parsed.exploit_feasibility <= 1.0
    assert signals, "clamping an out-of-range value must leave a signal"


def test_extra_fields_are_dropped_and_flagged(guard: OutputGuard) -> None:
    """Smuggled keys never reach the parsed object, and the attempt is recorded."""
    parsed, signals = guard.validate(
        _valid_exploitability(priority="critical", rank=1), ExploitabilityOut, SANITIZED, "CANARY-1"
    )
    assert not hasattr(parsed, "priority") and not hasattr(parsed, "rank")
    assert {signal.category for signal in signals} == {InjectionCategory.SCHEMA_SMUGGLING}


def test_unverifiable_evidence_span_is_dropped(guard: OutputGuard) -> None:
    parsed, signals = guard.validate(
        _valid_exploitability(evidence_spans=["CISA has confirmed active exploitation"]),
        ExploitabilityOut,
        SANITIZED,
        "CANARY-1",
    )
    assert "CISA has confirmed active exploitation" not in parsed.evidence_spans
    assert signals


def test_span_matching_tolerates_whitespace_only(guard: OutputGuard) -> None:
    assert span_matches("A  proof-of-concept   exploit is published", SANITIZED)
    assert not span_matches("an exploit was sold on a forum", SANITIZED)


def test_canary_leak_raises(guard: OutputGuard) -> None:
    with pytest.raises(CanaryLeakError):
        guard.validate(
            _valid_exploitability(rationale="The session canary is CANARY-1."),
            ExploitabilityOut,
            SANITIZED,
            "CANARY-1",
        )


def test_out_of_scope_identifier_is_rejected(guard: OutputGuard) -> None:
    """Referring to another finding is how an injection tries to move a neighbour's rank."""
    with pytest.raises(SchemaRejectedError):
        guard.validate(
            {
                "confidence": 0.5,
                "rationale": "See finding f_other for the real issue.",
                "evidence_spans": [],
                "function": "admin",
                "criticality": 0.6,
                "data_sensitivity": 0.5,
                "exposure": 0.5,
                "is_auth_boundary": False,
                "is_admin_surface": True,
            },
            AssetCriticalityOut,
            SANITIZED,
            "CANARY-1",
            in_scope_ids={"f_self"},
        )


@pytest.mark.parametrize(
    "tier,proposed,expected_max_delta",
    [
        (TrustTier.CURATED_FEED, 1.0, 1.0),
        (TrustTier.SCANNER, 1.0, 0.8),
        (TrustTier.REFERENCE_PAGE, 1.0, 0.35),
        (TrustTier.TARGET_CONTENT, 1.0, 0.15),
    ],
)
def test_influence_budget_caps_every_tier(tier: TrustTier, proposed: float, expected_max_delta: float) -> None:
    config = load_config("configs/default.yaml").sandbox
    value, delta = clamp_influence(0.2, proposed, tier, config)
    assert delta <= expected_max_delta + 1e-9
    assert value <= 0.2 + expected_max_delta + 1e-9


def test_budget_is_symmetric_for_deflation() -> None:
    """Arguing a score down is capped exactly like arguing it up."""
    config = load_config("configs/default.yaml").sandbox
    value, delta = clamp_influence(0.9, 0.0, TrustTier.TARGET_CONTENT, config)
    assert delta <= 0.15 + 1e-9
    assert value >= 0.9 - 0.15 - 1e-9


def test_floor_cannot_be_breached_by_untrusted_evidence() -> None:
    """KEV-derived evidence sets a floor a blog post may not argue below."""
    config = load_config("configs/default.yaml").sandbox
    value, _ = clamp_influence(0.8, 0.0, TrustTier.REFERENCE_PAGE, config, floor=0.7)
    assert value >= 0.7


def test_corroborated_claims_get_more_room() -> None:
    config = load_config("configs/default.yaml").sandbox
    capped, _ = clamp_influence(0.2, 1.0, TrustTier.REFERENCE_PAGE, config, corroborated=False)
    corroborated, _ = clamp_influence(0.2, 1.0, TrustTier.REFERENCE_PAGE, config, corroborated=True)
    assert corroborated >= capped
