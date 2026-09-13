"""The monotonicity property: remediation can never raise computed risk.

``ChainScore.reach_delta`` is a non-negative field on the frozen contract and
``chain_adjusted_loss`` adds it to expected loss, so if patching could ever raise ``R(G)``
the framework would be reporting negative contributions as zero and quietly lying about
every chain-adjusted figure downstream.

The proof is in :mod:`vulnpriority.graph.reachability`; this module is the evidence. A
randomised world of several hosts and a few dozen findings is patched under 200 random
remediation plans, seeded so a failure can be replayed exactly.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

from vulnpriority.core.config import ComponentCConfig
from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    EndpointFunction,
    HttpMethod,
    PrivilegeLevel,
    Provenance,
    ScannerSeverity,
)
from vulnpriority.core.errors import MonotonicityViolationError
from vulnpriority.core.interfaces import ChainScorer
from vulnpriority.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    AttackGraphSummary,
    BusinessImpact,
    ChainScore,
    Endpoint,
    EnrichedFinding,
    ExploitLikelihood,
    ExploitabilityAssessment,
    Finding,
    RemediationCost,
    Scan,
    UntrustedText,
)
from vulnpriority.graph.chain_scorer import ReachabilityChainScorer
from vulnpriority.graph.monotone import (
    assert_monotone_under_patching,
    random_patch_subsets,
)

SEED = 7
N_SUBSETS = 200
AS_OF = date(2024, 6, 1)
SCANNED_AT = datetime(2024, 5, 1, 9, 0, 0)
SCAN_ID = "scan_monotone"

#: CWEs spanning every row of the privilege table, so the random world exercises SYSTEM,
#: ADMIN, CIA-derived, victim-dependent and no-privilege transitions together.
CWE_POOL = (78, 89, 79, 287, 200, 502, 22, 863, 352, 611, 16, 306)


def random_world(seed: int = SEED, n_hosts: int = 4, n_findings: int = 40) -> tuple[Scan, list[EnrichedFinding]]:
    """A seeded multi-host application with chained privileges and lateral links."""
    rng = np.random.default_rng(seed)
    hosts = [f"h{index}.example.com" for index in range(n_hosts)]

    endpoints: list[Endpoint] = []
    for index in range(n_findings):
        host = hosts[int(rng.integers(0, n_hosts))]
        auth = PrivilegeLevel(int(rng.integers(0, 3)))
        # Host 0 is the perimeter; everything else is only reachable through a link.
        internet_facing = host == hosts[0]
        links: tuple[str, ...] = ()
        if index > 0 and rng.random() < 0.25:
            links = (f"ep_{int(rng.integers(0, index))}",)
        endpoints.append(
            Endpoint(
                endpoint_id=f"ep_{index}",
                app_id="app_m",
                host=host,
                url=f"https://{host}/p{index}",
                path=f"/p{index}",
                method=HttpMethod.GET,
                auth_required=auth,
                internet_facing=internet_facing,
                response_status=200,
                links_to=links,
            )
        )

    enriched: list[EnrichedFinding] = []
    for index, endpoint in enumerate(endpoints):
        finding_id = f"f_{index}"
        cwe_id = int(CWE_POOL[int(rng.integers(0, len(CWE_POOL)))])
        p_exploit = float(rng.uniform(0.01, 0.95))
        p_applicable = float(rng.uniform(0.05, 1.0))
        impact = float(rng.uniform(0.0, 250_000.0))
        criticality = float(rng.uniform(0.0, 1.0))
        finding = Finding(
            finding_id=finding_id,
            scan_id=SCAN_ID,
            app_id="app_m",
            endpoint_id=endpoint.endpoint_id,
            name=f"finding {index}",
            cwe_id=cwe_id,
            scanner="zap",
            scanner_severity=ScannerSeverity.MEDIUM,
            description=UntrustedText(text=f"finding {index}", provenance=Provenance.SCANNER_OUTPUT),
            observed_at=SCANNED_AT,
            dedup_key=f"dk_{index % 9}",
        )
        enriched.append(
            EnrichedFinding(
                finding=finding,
                endpoint=endpoint,
                asset=AssetCriticality(
                    endpoint_id=endpoint.endpoint_id,
                    function=EndpointFunction.API_DATA,
                    criticality=criticality,
                    data_sensitivity=criticality,
                    exposure=1.0 if endpoint.internet_facing else 0.2,
                ),
                exploitability=ExploitabilityAssessment(
                    finding_id=finding_id,
                    exploit_feasibility=p_exploit,
                    privileges_required=PrivilegeLevel(int(rng.integers(0, 3))),
                    privilege_gained=PrivilegeLevel(int(rng.integers(0, 4))),
                    impact_c=float(rng.uniform(0.0, 1.0)),
                    impact_i=float(rng.uniform(0.0, 1.0)),
                    impact_a=float(rng.uniform(0.0, 1.0)),
                ),
                applicability=ApplicabilityAssessment(
                    finding_id=finding_id,
                    verdict=ApplicabilityVerdict.UNCERTAIN,
                    p_applicable=p_applicable,
                ),
                likelihood=ExploitLikelihood(
                    finding_id=finding_id,
                    attacker="opportunistic",
                    p_exploit=p_exploit,
                    p_exploit_uncapped=p_exploit,
                    horizon_days=90,
                ),
                impact=BusinessImpact(finding_id=finding_id, total=impact),
                remediation=RemediationCost(finding_id=finding_id, hours=1.0 + index % 8, cost=120.0),
                expected_loss=p_exploit * impact,
                as_of=AS_OF,
            )
        )

    scan = Scan(
        scan_id=SCAN_ID,
        app_id="app_m",
        app_name="Monotonicity fixture",
        scanned_at=SCANNED_AT,
        scanner_name="zap",
        hosts=tuple(hosts),
        endpoints=tuple(endpoints),
        findings=tuple(item.finding for item in enriched),
    )
    return scan, enriched


@pytest.fixture(scope="module")
def scorer() -> ReachabilityChainScorer:
    scan, enriched = random_world()
    built = ReachabilityChainScorer(ComponentCConfig(verify_monotonicity=False))
    built.build(scan, enriched)
    return built


def test_the_random_world_is_actually_interesting(scorer: ReachabilityChainScorer) -> None:
    """Guard against the property test passing because the graph is trivial."""
    summary = scorer.summary_for(SCAN_ID)
    scores = scorer.score(SCAN_ID)

    assert summary.total_risk > 0.0
    assert len(summary.target_nodes) >= 3
    assert sum(1 for score in scores.values() if score.reach_delta > 0.0) >= 3
    assert any(score.privilege_gain >= 2 for score in scores.values())
    assert len({node.asset for node in summary.nodes}) >= 3


@pytest.mark.property
def test_risk_never_increases_under_two_hundred_random_patch_subsets(
    scorer: ReachabilityChainScorer,
) -> None:
    """The property, stated and checked exactly as DESIGN.md 3.7 states it."""
    rng = np.random.default_rng(SEED)
    finding_ids = scorer.finding_ids(SCAN_ID)
    base = scorer.total_risk_after_patching(SCAN_ID, set())
    subsets = random_patch_subsets(finding_ids, N_SUBSETS, rng)

    assert len(subsets) == N_SUBSETS
    risks = [scorer.total_risk_after_patching(SCAN_ID, set(subset)) for subset in subsets]

    assert all(risk <= base + 1e-6 for risk in risks)
    assert all(risk >= 0.0 for risk in risks)
    # Not vacuous: some of those plans must actually have bitten.
    assert any(risk < base - 1.0 for risk in risks)


@pytest.mark.property
def test_patching_more_never_helps_the_attacker(scorer: ReachabilityChainScorer) -> None:
    """Nested monotonicity: ``R(G \\ (S u {f})) <= R(G \\ S)`` for random nested pairs."""
    rng = np.random.default_rng(SEED + 1)
    finding_ids = list(scorer.finding_ids(SCAN_ID))

    for _ in range(N_SUBSETS):
        size = int(rng.integers(0, len(finding_ids)))
        subset = {finding_ids[int(index)] for index in rng.choice(len(finding_ids), size=size, replace=False)} if size else set()
        remaining = [item for item in finding_ids if item not in subset]
        extra = remaining[int(rng.integers(0, len(remaining)))]
        smaller = scorer.total_risk_after_patching(SCAN_ID, subset)
        larger = scorer.total_risk_after_patching(SCAN_ID, subset | {extra})
        assert larger <= smaller + 1e-6


@pytest.mark.property
def test_assert_monotone_under_patching_accepts_the_real_scorer(
    scorer: ReachabilityChainScorer,
) -> None:
    risks = assert_monotone_under_patching(scorer, SCAN_ID, subsets=N_SUBSETS, rng=SEED)
    base = scorer.total_risk_after_patching(SCAN_ID, set())
    assert risks[0] == pytest.approx(base)
    assert all(risk <= base + 1e-6 for risk in risks)


def test_every_reach_delta_is_non_negative(scorer: ReachabilityChainScorer) -> None:
    base = scorer.total_risk_after_patching(SCAN_ID, set())
    for finding_id, score in scorer.score(SCAN_ID).items():
        assert score.reach_delta >= 0.0
        assert score.reach_delta <= base + 1e-6
        recomputed = base - scorer.total_risk_after_patching(SCAN_ID, {finding_id})
        assert score.reach_delta == pytest.approx(max(0.0, recomputed), abs=1e-6)


def test_patching_everything_leaves_nothing_reachable(scorer: ReachabilityChainScorer) -> None:
    everything = set(scorer.finding_ids(SCAN_ID))
    assert scorer.total_risk_after_patching(SCAN_ID, everything) == pytest.approx(0.0)


def test_explicit_subsets_are_accepted_as_well_as_a_count(scorer: ReachabilityChainScorer) -> None:
    ids = scorer.finding_ids(SCAN_ID)
    plans = [set(), {ids[0]}, {ids[0], ids[1]}, set(ids)]
    risks = assert_monotone_under_patching(scorer, SCAN_ID, subsets=plans, rng=SEED)
    assert len(risks) >= len(plans)


def test_random_patch_subsets_are_reproducible_and_span_the_range() -> None:
    ids = [f"f_{index}" for index in range(12)]
    first = random_patch_subsets(ids, 50, 11)
    second = random_patch_subsets(ids, 50, 11)
    assert first == second
    assert random_patch_subsets(ids, 50, 12) != first
    sizes = {len(subset) for subset in first}
    assert min(sizes) == 0 and max(sizes) == len(ids)
    assert random_patch_subsets([], 3, SEED) == [frozenset(), frozenset(), frozenset()]
    assert random_patch_subsets(ids, 0, SEED) == []


# --------------------------------------------------------------------------
# The assertion has to be able to fail
# --------------------------------------------------------------------------


class _BrokenScorer(ChainScorer):
    """A scorer whose risk rises with the size of the remediation plan.

    Exactly the failure mode a sign error in the differencing would produce, and the one
    the runtime check exists to catch.
    """

    def __init__(self, finding_count: int = 6) -> None:
        self._ids = tuple(f"f_{index}" for index in range(finding_count))

    def build(self, scan: Scan, enriched: list[EnrichedFinding]) -> AttackGraphSummary:  # pragma: no cover
        return AttackGraphSummary(scan_id=scan.scan_id)

    def score(self, scan_id: str) -> dict[str, ChainScore]:
        return {finding_id: ChainScore(finding_id=finding_id) for finding_id in self._ids}

    def total_risk_after_patching(self, scan_id: str, patched_finding_ids: set[str]) -> float:
        return 1_000.0 + 500.0 * len(patched_finding_ids)


class _NegativeScorer(_BrokenScorer):
    """A scorer that reports negative unpatched risk, which ``R(G)`` cannot be."""

    def total_risk_after_patching(self, scan_id: str, patched_finding_ids: set[str]) -> float:
        return -1.0


def test_a_scorer_that_rewards_patching_the_attacker_is_rejected() -> None:
    with pytest.raises(MonotonicityViolationError, match="raised risk"):
        assert_monotone_under_patching(_BrokenScorer(), SCAN_ID, subsets=25, rng=SEED)


def test_negative_risk_is_rejected() -> None:
    with pytest.raises(MonotonicityViolationError, match="negative"):
        assert_monotone_under_patching(_NegativeScorer(), SCAN_ID, subsets=5, rng=SEED)


def test_runtime_verification_runs_on_build_and_stamps_the_summary() -> None:
    scan, enriched = random_world(seed=SEED + 3, n_hosts=3, n_findings=18)
    summary = ReachabilityChainScorer(ComponentCConfig(verify_monotonicity=True)).build(scan, enriched)
    assert summary.monotone_verified is True
    assert summary.total_risk >= 0.0


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_the_property_holds_on_independently_seeded_worlds(seed: int) -> None:
    scan, enriched = random_world(seed=seed, n_hosts=3, n_findings=25)
    built = ReachabilityChainScorer(ComponentCConfig(verify_monotonicity=False))
    built.build(scan, enriched)
    assert_monotone_under_patching(built, SCAN_ID, subsets=40, rng=seed)
