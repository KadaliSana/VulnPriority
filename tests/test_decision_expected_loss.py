"""Gap 1: expected loss is the construct definition of priority (DESIGN.md 2).

These tests are short on purpose. The functions under test are two multiplications; what
matters is that they are *the definition*, that they have the properties a priority
ordering needs, and that nothing about them is negotiable downstream.
"""

from __future__ import annotations

import pytest

from vulnpriority.core.models import BusinessImpact, ExploitLikelihood
from vulnpriority.decision.expected_loss import chain_adjusted_loss, expected_loss


def likelihood(p: float) -> ExploitLikelihood:
    return ExploitLikelihood(
        finding_id="f1",
        attacker="test",
        p_exploit=p,
        p_exploit_uncapped=p,
        log_odds_terms={"intercept": 0.0},
        horizon_days=90,
    )


def impact(total: float) -> BusinessImpact:
    return BusinessImpact(finding_id="f1", total=total)


def test_expected_loss_is_probability_times_money() -> None:
    """The definition, verbatim: P(exploit | evidence, attacker) x impact."""
    assert expected_loss(likelihood(0.25), impact(400_000.0)) == pytest.approx(100_000.0)
    assert expected_loss(likelihood(0.0), impact(400_000.0)) == pytest.approx(0.0)
    assert expected_loss(likelihood(1.0), impact(400_000.0)) == pytest.approx(400_000.0)


def test_expected_loss_is_monotone_in_both_factors() -> None:
    """A more likely or more costly finding can never rank lower on the construct."""
    base = expected_loss(likelihood(0.3), impact(100_000.0))
    assert expected_loss(likelihood(0.6), impact(100_000.0)) >= base
    assert expected_loss(likelihood(0.3), impact(200_000.0)) >= base


def test_expected_loss_is_never_negative() -> None:
    assert expected_loss(likelihood(0.0), impact(0.0)) == 0.0


def test_a_certain_small_loss_can_outrank_an_unlikely_large_one() -> None:
    """This inversion is the whole argument for Gap 1 over severity-first ordering."""
    certain_small = expected_loss(likelihood(0.9), impact(50_000.0))      # 45_000
    unlikely_large = expected_loss(likelihood(0.01), impact(3_000_000.0))  # 30_000
    assert certain_small > unlikely_large


def test_chain_adjusted_loss_adds_the_weighted_reach_delta() -> None:
    assert chain_adjusted_loss(100_000.0, 40_000.0, 1.0) == pytest.approx(140_000.0)
    assert chain_adjusted_loss(100_000.0, 40_000.0, 0.5) == pytest.approx(120_000.0)
    assert chain_adjusted_loss(100_000.0, 40_000.0, 0.0) == pytest.approx(100_000.0)


def test_chain_adjustment_can_only_raise_priority() -> None:
    """reach_delta is non-negative by the monotonicity proof; a stepping stone gains, never loses."""
    for reach_delta in (0.0, 1.0, 1e6):
        assert chain_adjusted_loss(10_000.0, reach_delta, 1.0) >= 10_000.0


def test_chain_adjustment_defends_against_a_negative_reach_delta() -> None:
    """A negative delta would mean patching raised risk, which is impossible; clamp, do not propagate."""
    assert chain_adjusted_loss(10_000.0, -5_000.0, 1.0) == pytest.approx(10_000.0)


def test_a_stepping_stone_can_outrank_a_direct_finding() -> None:
    """Low direct loss, high unlocked value: the ordering Component C exists to produce."""
    direct = chain_adjusted_loss(expected_loss(likelihood(0.4), impact(60_000.0)), 0.0, 1.0)
    stepping_stone = chain_adjusted_loss(expected_loss(likelihood(0.4), impact(5_000.0)), 500_000.0, 1.0)
    assert stepping_stone > direct


def test_expected_loss_defaults_to_a_plain_baseline_ranking() -> None:
    """``RankerName.EXPECTED_LOSS`` orders by this function alone; the ordering must be total."""
    findings = [(0.9, 10_000.0), (0.1, 500_000.0), (0.5, 120_000.0)]
    losses = [expected_loss(likelihood(p), impact(value)) for p, value in findings]
    assert sorted(losses, reverse=True) == [60_000.0, 50_000.0, 9_000.0]
