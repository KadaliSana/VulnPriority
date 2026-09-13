"""Attack-graph construction: shape, direction, value and the edge trust rule.

The headline test here is ``test_low_cvss_chokepoint_outranks_high_cvss_leaf``: it is the
result the literature review claims Component C delivers, and it is the reason the
framework models states rather than findings.

Components A and B are faked with plain ``vulnpriority.core.models`` objects rather than
imported, so these tests exercise the graph and nothing else.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from vulnpriority.core.config import ComponentCConfig
from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    AttackComplexity,
    EndpointFunction,
    HttpMethod,
    LLMBackendKind,
    PrivilegeLevel,
    Provenance,
    ScannerSeverity,
    TrustTier,
    UserInteraction,
)
from vulnpriority.core.errors import GraphError
from vulnpriority.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    AttackGraphSummary,
    BusinessImpact,
    Endpoint,
    EnrichedFinding,
    ExploitLikelihood,
    ExploitabilityAssessment,
    Finding,
    LLMAudit,
    RemediationCost,
    Scan,
    TrustSummary,
    UntrustedText,
)
from vulnpriority.decision.expected_loss import chain_adjusted_loss
from vulnpriority.graph.attack_graph import (
    AttackGraphBuilder,
    admits_edge,
    edge_evidence_tier,
    parse_state_node,
    state_node,
)
from vulnpriority.graph.chain_scorer import ReachabilityChainScorer
from vulnpriority.graph.privilege_map import privileges_for

AS_OF = date(2024, 6, 1)
SCANNED_AT = datetime(2024, 5, 1, 9, 0, 0)
HOST = "shop.example.com"
SCAN_ID = "scan_c"


# --------------------------------------------------------------------------
# Fixtures built directly from the frozen models
# --------------------------------------------------------------------------


def make_endpoint(
    endpoint_id: str,
    path: str,
    *,
    host: str = HOST,
    auth: PrivilegeLevel = PrivilegeLevel.NONE,
    internet_facing: bool = True,
    links_to: tuple[str, ...] = (),
) -> Endpoint:
    return Endpoint(
        endpoint_id=endpoint_id,
        app_id="app_c",
        host=host,
        url=f"https://{host}{path}",
        path=path,
        method=HttpMethod.GET,
        auth_required=auth,
        internet_facing=internet_facing,
        response_status=200,
        links_to=links_to,
    )


def make_enriched(
    finding_id: str,
    endpoint: Endpoint,
    *,
    cwe_id: int | None,
    severity: ScannerSeverity = ScannerSeverity.MEDIUM,
    p_exploit: float = 0.5,
    p_applicable: float = 1.0,
    impact: float = 1000.0,
    criticality: float = 0.5,
    impact_c: float = 0.5,
    impact_i: float = 0.5,
    impact_a: float = 0.0,
    privileges_required: PrivilegeLevel = PrivilegeLevel.NONE,
    privilege_gained: PrivilegeLevel = PrivilegeLevel.NONE,
    audit: LLMAudit | None = None,
    canary_leaked: bool = False,
    hours: float = 4.0,
    dedup_key: str | None = None,
    scan_id: str = SCAN_ID,
) -> EnrichedFinding:
    """One enriched finding with exactly the knobs the graph reads."""
    finding = Finding(
        finding_id=finding_id,
        scan_id=scan_id,
        app_id="app_c",
        endpoint_id=endpoint.endpoint_id,
        name=finding_id,
        cwe_id=cwe_id,
        scanner="zap",
        scanner_severity=severity,
        scanner_confidence=0.8,
        description=UntrustedText(text=f"{finding_id} on {endpoint.path}", provenance=Provenance.SCANNER_OUTPUT),
        observed_at=SCANNED_AT,
        dedup_key=dedup_key or f"dk_{finding_id}",
    )
    return EnrichedFinding(
        finding=finding,
        endpoint=endpoint,
        asset=AssetCriticality(
            endpoint_id=endpoint.endpoint_id,
            function=EndpointFunction.API_DATA,
            criticality=criticality,
            data_sensitivity=criticality,
            exposure=1.0 if endpoint.auth_required == PrivilegeLevel.NONE else 0.5,
        ),
        exploitability=ExploitabilityAssessment(
            finding_id=finding_id,
            exploit_feasibility=p_exploit,
            attack_complexity=AttackComplexity.LOW,
            privileges_required=privileges_required,
            user_interaction=UserInteraction.NONE,
            impact_c=impact_c,
            impact_i=impact_i,
            impact_a=impact_a,
            privilege_gained=privilege_gained,
            audit=audit,
        ),
        applicability=ApplicabilityAssessment(
            finding_id=finding_id,
            verdict=ApplicabilityVerdict.APPLICABLE,
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
        remediation=RemediationCost(finding_id=finding_id, hours=hours, cost=hours * 120.0),
        expected_loss=p_exploit * impact,
        trust=TrustSummary(canary_leaked=canary_leaked),
        as_of=AS_OF,
    )


def make_scan(endpoints: tuple[Endpoint, ...], enriched: list[EnrichedFinding], scan_id: str = SCAN_ID) -> Scan:
    return Scan(
        scan_id=scan_id,
        app_id="app_c",
        app_name="Component C fixture",
        scanned_at=SCANNED_AT,
        scanner_name="zap",
        hosts=tuple(sorted({endpoint.host for endpoint in endpoints})),
        endpoints=endpoints,
        findings=tuple(item.finding for item in enriched),
    )


def untrusted_audit(tier: TrustTier) -> LLMAudit:
    """An assessment that claims to have been authored on the authority of ``tier``."""
    return LLMAudit(backend=LLMBackendKind.HEURISTIC, model="heuristic", task="exploitability", max_tier_used=tier)


# --------------------------------------------------------------------------
# The four-node chokepoint scenario
# --------------------------------------------------------------------------


def chokepoint_world() -> tuple[Scan, list[EnrichedFinding]]:
    """Four states, three findings, one gate.

    ``state:internet:NONE`` -> ``state:shop:NONE`` -> ``state:shop:USER`` -> ``state:shop:ADMIN``

    * ``f_choke``  low severity, CVSS-4.3-shaped, worth $2,000 on its own, and the only
      way off the unauthenticated surface.
    * ``f_crown``  the escalation to administrator, where the money is.
    * ``f_leaf``   a CVSS-9.8-shaped SQL injection sitting *behind* the administrator
      boundary: devastating on paper, but it unlocks nothing that is not already unlocked.
    """
    login = make_endpoint("ep_login", "/login", auth=PrivilegeLevel.NONE)
    panel = make_endpoint("ep_panel", "/admin/panel", auth=PrivilegeLevel.USER)
    report = make_endpoint("ep_report", "/admin/search", auth=PrivilegeLevel.ADMIN)

    enriched = [
        make_enriched(
            "f_choke",
            login,
            cwe_id=79,  # reflected XSS: NONE -> USER
            severity=ScannerSeverity.LOW,
            p_exploit=0.6,
            impact=2_000.0,
            criticality=0.2,
        ),
        make_enriched(
            "f_crown",
            panel,
            cwe_id=863,  # incorrect authorization: USER -> ADMIN
            severity=ScannerSeverity.HIGH,
            p_exploit=0.7,
            impact=900_000.0,
            criticality=1.0,
        ),
        make_enriched(
            "f_leaf",
            report,
            cwe_id=89,  # SQL injection, but the endpoint already demands ADMIN
            severity=ScannerSeverity.CRITICAL,
            p_exploit=0.95,
            impact=100_000.0,
            criticality=0.5,
            impact_c=0.9,
            impact_i=0.2,
            impact_a=0.0,
        ),
    ]
    return make_scan((login, panel, report), enriched), enriched


@pytest.fixture
def chokepoint() -> tuple[Scan, list[EnrichedFinding]]:
    return chokepoint_world()


def test_graph_has_exactly_the_four_expected_states(chokepoint) -> None:
    scan, enriched = chokepoint
    graph, summary = AttackGraphBuilder().build(scan, enriched)

    assert set(graph.nodes) == {
        "state:internet:NONE",
        f"state:{HOST}:NONE",
        f"state:{HOST}:USER",
        f"state:{HOST}:ADMIN",
    }
    assert summary.entry_node == "state:internet:NONE"
    assert graph.nodes["state:internet:NONE"]["is_entry"] is True
    assert isinstance(summary, AttackGraphSummary)
    assert summary.scan_id == SCAN_ID


def test_node_ids_round_trip_through_the_naming_convention() -> None:
    node = state_node("shop.example.com:8443", PrivilegeLevel.ADMIN)
    assert node == "state:shop.example.com:8443:ADMIN"
    assert parse_state_node(node) == ("shop.example.com:8443", PrivilegeLevel.ADMIN)
    with pytest.raises(GraphError):
        parse_state_node("shop.example.com:ADMIN")


def test_exploit_edges_carry_p_exploit_times_p_applicable(chokepoint) -> None:
    scan, enriched = chokepoint
    graph, _ = AttackGraphBuilder().build(scan, enriched)

    choke = graph.edges[f"state:{HOST}:NONE", f"state:{HOST}:USER"]
    assert choke["kind"] == "exploit"
    assert choke["probability"] == pytest.approx(0.6)
    assert choke["finding_ids"] == ("f_choke",)

    crown = graph.edges[f"state:{HOST}:USER", f"state:{HOST}:ADMIN"]
    assert crown["probability"] == pytest.approx(0.7)

    # f_leaf sits on an ADMIN-only endpoint and confers no more than ADMIN, so it creates
    # no transition at all: nothing new is unlocked by exploiting it.
    assert all("f_leaf" not in data["finding_ids"] for _, _, data in graph.edges(data=True))


def test_p_applicable_scales_the_edge_probability() -> None:
    login = make_endpoint("ep_login", "/login")
    enriched = [make_enriched("f_a", login, cwe_id=79, p_exploit=0.6, p_applicable=0.5)]
    scan = make_scan((login,), enriched)
    graph, _ = AttackGraphBuilder().build(scan, enriched)
    assert graph.edges[f"state:{HOST}:NONE", f"state:{HOST}:USER"]["probability"] == pytest.approx(0.3)


def test_privilege_implication_edges_run_downward_at_certainty(chokepoint) -> None:
    scan, enriched = chokepoint
    graph, _ = AttackGraphBuilder().build(scan, enriched)

    for higher, lower in (
        (f"state:{HOST}:ADMIN", f"state:{HOST}:USER"),
        (f"state:{HOST}:ADMIN", f"state:{HOST}:NONE"),
        (f"state:{HOST}:USER", f"state:{HOST}:NONE"),
    ):
        data = graph.edges[higher, lower]
        assert data["kind"] == "privilege_implication"
        assert data["probability"] == pytest.approx(1.0)
        assert data["finding_ids"] == ()
        # ... and never the other way round: implication is not escalation, so any edge
        # climbing the lattice has to be an exploit somebody actually found.
        if graph.has_edge(lower, higher):
            assert graph.edges[lower, higher]["kind"] == "exploit"


def test_node_values_are_impact_weighted_by_criticality(chokepoint) -> None:
    scan, enriched = chokepoint
    graph, summary = AttackGraphBuilder().build(scan, enriched)

    # 2_000 * 0.2 on the USER state; (900_000 * 1.0) + (100_000 * 0.5) on ADMIN.
    assert graph.nodes[f"state:{HOST}:USER"]["value"] == pytest.approx(400.0)
    assert graph.nodes[f"state:{HOST}:ADMIN"]["value"] == pytest.approx(950_000.0)
    # Reaching a host is not itself worth anything.
    assert graph.nodes[f"state:{HOST}:NONE"]["value"] == pytest.approx(0.0)
    assert set(summary.target_nodes) == {f"state:{HOST}:USER", f"state:{HOST}:ADMIN"}


def test_criticality_weight_exponent_concentrates_value(chokepoint) -> None:
    scan, enriched = chokepoint
    flat = AttackGraphBuilder(ComponentCConfig(criticality_weight_exponent=0.0)).build(scan, enriched)[0]
    sharp = AttackGraphBuilder(ComponentCConfig(criticality_weight_exponent=2.0)).build(scan, enriched)[0]

    assert flat.nodes[f"state:{HOST}:USER"]["value"] == pytest.approx(2_000.0)
    assert sharp.nodes[f"state:{HOST}:USER"]["value"] == pytest.approx(2_000.0 * 0.04)


def test_total_risk_is_value_weighted_reachability(chokepoint) -> None:
    scan, enriched = chokepoint
    _, summary = AttackGraphBuilder().build(scan, enriched)
    # 400 * 0.6 + 950_000 * (0.6 * 0.7)
    assert summary.total_risk == pytest.approx(240.0 + 399_000.0)


def test_low_cvss_chokepoint_outranks_high_cvss_leaf(chokepoint) -> None:
    """The headline capability of Component C (Gap 9).

    ``f_choke`` is the finding every triage process built on severity throws away: low
    scanner severity, a CVSS-4.3-shaped weakness, two thousand of direct exposure.
    ``f_leaf`` is the one every such process fixes first: critical, CVSS-9.8-shaped, nearly
    certain to be exploitable. Ordered by chain-adjusted loss, the cheap one wins by more
    than four to one, because it is the only way in and the expensive one is behind a door
    the attacker has to already be through.
    """
    scan, enriched = chokepoint
    scorer = ReachabilityChainScorer()
    summary = scorer.build(scan, enriched)
    scores = scorer.score(SCAN_ID)

    base = summary.total_risk
    assert scores["f_choke"].reach_delta == pytest.approx(base)
    assert scores["f_crown"].reach_delta == pytest.approx(399_000.0)
    assert scores["f_leaf"].reach_delta == pytest.approx(0.0)

    by_id = {item.finding_id: item for item in enriched}
    adjusted = {
        finding_id: chain_adjusted_loss(
            by_id[finding_id].expected_loss, score.reach_delta, chain_weight=1.0
        )
        for finding_id, score in scores.items()
    }

    # Severity says the opposite of what reachable compromise says.
    assert by_id["f_choke"].finding.scanner_severity == ScannerSeverity.LOW
    assert by_id["f_leaf"].finding.scanner_severity == ScannerSeverity.CRITICAL
    assert by_id["f_choke"].expected_loss < by_id["f_leaf"].expected_loss
    assert adjusted["f_choke"] > adjusted["f_leaf"]
    assert adjusted["f_choke"] / adjusted["f_leaf"] > 4.0

    assert scores["f_choke"].is_chokepoint is True
    assert scores["f_leaf"].is_chokepoint is False


def test_chain_score_positional_fields(chokepoint) -> None:
    scan, enriched = chokepoint
    scorer = ReachabilityChainScorer()
    scorer.build(scan, enriched)
    scores = scorer.score(SCAN_ID)

    choke, crown, leaf = scores["f_choke"], scores["f_crown"], scores["f_leaf"]

    assert (choke.hops_from_entry, crown.hops_from_entry, leaf.hops_from_entry) == (2, 3, 0)
    assert (choke.privilege_gain, crown.privilege_gain, leaf.privilege_gain) == (1, 1, 0)
    assert choke.max_path_prob_to_target == pytest.approx(0.6)
    assert crown.max_path_prob_to_target == pytest.approx(0.42)
    assert crown.best_target == f"state:{HOST}:ADMIN"
    assert leaf.max_path_prob_to_target == pytest.approx(0.0)
    assert choke.n_paths_through == 2 and crown.n_paths_through == 1 and leaf.n_paths_through == 0
    assert choke.betweenness > 0.0 and leaf.betweenness == pytest.approx(0.0)


def test_top_paths_are_recorded_in_descending_expected_value(chokepoint) -> None:
    scan, enriched = chokepoint
    _, summary = AttackGraphBuilder().build(scan, enriched)

    assert summary.top_paths
    best = summary.top_paths[0]
    assert best.nodes == (
        "state:internet:NONE",
        f"state:{HOST}:NONE",
        f"state:{HOST}:USER",
        f"state:{HOST}:ADMIN",
    )
    assert best.probability == pytest.approx(0.42)
    assert best.expected_value == pytest.approx(0.42 * 950_000.0)
    values = [path.expected_value for path in summary.top_paths]
    assert values == sorted(values, reverse=True)


# --------------------------------------------------------------------------
# Direction
# --------------------------------------------------------------------------


def directional_world(deep_required: PrivilegeLevel) -> tuple[Scan, list[EnrichedFinding]]:
    """A gate that confers USER, and a deep finding whose requirement we vary.

    With ``deep_required == USER`` the two chain. With ``deep_required == ADMIN`` the deep
    edge points out of a state nothing in the graph confers, so it is an edge the wrong way
    round relative to the only route that exists.
    """
    front = make_endpoint("ep_front", "/portal")
    deep = make_endpoint("ep_deep", "/internal/exec")
    enriched = [
        make_enriched(
            "f_gate",
            front,
            cwe_id=None,
            p_exploit=0.8,
            impact=1_000.0,
            criticality=0.5,
            privileges_required=PrivilegeLevel.NONE,
            privilege_gained=PrivilegeLevel.USER,
        ),
        make_enriched(
            "f_deep",
            deep,
            cwe_id=None,
            p_exploit=0.9,
            impact=500_000.0,
            criticality=1.0,
            privileges_required=deep_required,
            privilege_gained=PrivilegeLevel.SYSTEM,
        ),
    ]
    return make_scan((front, deep), enriched), enriched


def test_edge_the_wrong_way_round_contributes_nothing() -> None:
    """Directionality is the property undirected chaining models discard.

    Same two findings, same probabilities, same money. Only the direction of the second
    transition relative to the first differs, and the contribution collapses to zero.
    """
    forward_scan, forward = directional_world(PrivilegeLevel.USER)
    forward_scorer = ReachabilityChainScorer()
    forward_scorer.build(forward_scan, forward)
    forward_scores = forward_scorer.score(SCAN_ID)

    wrong_scan, wrong = directional_world(PrivilegeLevel.ADMIN)
    wrong_scorer = ReachabilityChainScorer()
    wrong_summary = wrong_scorer.build(wrong_scan, wrong)
    wrong_scores = wrong_scorer.score(SCAN_ID)

    assert forward_scores["f_deep"].reach_delta > 100_000.0
    assert forward_scores["f_deep"].max_path_prob_to_target > 0.0

    assert wrong_scores["f_deep"].reach_delta == pytest.approx(0.0)
    assert wrong_scores["f_deep"].max_path_prob_to_target == pytest.approx(0.0)
    assert wrong_scores["f_deep"].n_paths_through == 0
    # The edge exists; it simply starts somewhere the attacker cannot stand.
    assert wrong_scorer.graph_for(SCAN_ID).has_edge(f"state:{HOST}:ADMIN", f"state:{HOST}:SYSTEM")
    assert f"state:{HOST}:SYSTEM" in wrong_summary.target_nodes
    assert wrong_summary.total_risk < forward_scorer.summary_for(SCAN_ID).total_risk


# --------------------------------------------------------------------------
# The edge trust rule (security property)
# --------------------------------------------------------------------------


def ghost_world(tier: TrustTier, *, canary_leaked: bool = False, cwe_id: int | None = None):
    """A trusted gate plus a finding claiming a spectacular escalation at ``tier``."""
    front = make_endpoint("ep_front", "/portal")
    ghost_endpoint = make_endpoint("ep_ghost", "/status")
    enriched = [
        make_enriched(
            "f_gate",
            front,
            cwe_id=None,
            p_exploit=0.8,
            impact=1_000.0,
            criticality=0.5,
            privileges_required=PrivilegeLevel.NONE,
            privilege_gained=PrivilegeLevel.USER,
        ),
        make_enriched(
            "f_ghost",
            ghost_endpoint,
            cwe_id=cwe_id,
            p_exploit=0.99,
            impact=5_000_000.0,
            criticality=1.0,
            impact_c=0.9,
            impact_i=0.9,
            impact_a=0.9,
            privileges_required=PrivilegeLevel.NONE,
            privilege_gained=PrivilegeLevel.SYSTEM,
            audit=untrusted_audit(tier),
            canary_leaked=canary_leaked,
        ),
    ]
    return make_scan((front, ghost_endpoint), enriched), enriched


@pytest.mark.parametrize("tier", [TrustTier.REFERENCE_PAGE, TrustTier.TARGET_CONTENT])
def test_untrusted_text_cannot_create_an_edge(tier: TrustTier) -> None:
    """A page on the internet, or the target itself, cannot invent a path into the graph.

    ``f_ghost`` has an uncovered CWE, so the *only* reason to believe it escalates to
    SYSTEM is an assessment written on the authority of tier-3 or tier-4 text. The edge is
    refused, the state it claimed is never created, the five million it claimed to
    unlock never enters ``R(G)``, and the refusal is counted.
    """
    scan, enriched = ghost_world(tier)
    graph, summary = AttackGraphBuilder().build(scan, enriched)

    assert summary.rejected_untrusted_edges == 1
    assert edge_evidence_tier(enriched[1]) == tier
    assert admits_edge(enriched[1]) is False
    assert f"state:{HOST}:SYSTEM" not in graph
    assert all(edge.finding_id != "f_ghost" for edge in summary.edges)
    assert summary.total_risk < 100_000.0


def test_scanner_tier_evidence_is_allowed_to_create_the_same_edge() -> None:
    """The rule is about provenance, not about the claim: at tier 2 the edge is admitted."""
    scan, enriched = ghost_world(TrustTier.SCANNER)
    graph, summary = AttackGraphBuilder().build(scan, enriched)

    assert summary.rejected_untrusted_edges == 0
    assert admits_edge(enriched[1]) is True
    assert graph.has_edge(f"state:{HOST}:NONE", f"state:{HOST}:SYSTEM")
    assert summary.total_risk > 1_000_000.0


def test_operator_cwe_table_is_not_hostage_to_an_untrusted_assessment() -> None:
    """A covered CWE does not need the model's word for its privilege transition.

    SQL injection escalates because :mod:`vulnpriority.graph.privilege_map` says so - operator
    code keyed by an identifier the scanner reported - so an assessment authored over
    target content does not make the edge untrusted. What untrusted text *can* still do is
    bounded elsewhere, by the enricher's influence budget on ``p_exploit``.
    """
    scan, enriched = ghost_world(TrustTier.TARGET_CONTENT, cwe_id=89)
    graph, summary = AttackGraphBuilder().build(scan, enriched)

    assert edge_evidence_tier(enriched[1]) == TrustTier.SCANNER
    assert summary.rejected_untrusted_edges == 0
    assert graph.has_edge(f"state:{HOST}:NONE", f"state:{HOST}:SYSTEM")


def test_canary_leak_rejects_the_edge_whatever_tier_is_claimed() -> None:
    """A leaked canary means the assessment is the attacker's output, not the model's."""
    scan, enriched = ghost_world(TrustTier.SCANNER, canary_leaked=True, cwe_id=89)
    graph, summary = AttackGraphBuilder().build(scan, enriched)

    assert admits_edge(enriched[1]) is False
    assert summary.rejected_untrusted_edges == 1
    assert f"state:{HOST}:SYSTEM" not in graph


def test_every_recorded_edge_is_at_or_below_scanner_tier() -> None:
    scan, enriched = ghost_world(TrustTier.TARGET_CONTENT)
    _, summary = AttackGraphBuilder().build(scan, enriched)
    assert all(edge.tier <= TrustTier.SCANNER for edge in summary.edges)


# --------------------------------------------------------------------------
# Lateral movement and multiple hosts
# --------------------------------------------------------------------------


def lateral_world(*, admit: bool = True) -> tuple[Scan, list[EnrichedFinding], ComponentCConfig]:
    """A public front host linking to an internal host that holds the money."""
    front = make_endpoint("ep_front", "/portal", host="www.example.com", links_to=("ep_back",))
    back = make_endpoint("ep_back", "/reports", host="db.example.com", internet_facing=False)
    enriched = [
        make_enriched(
            "f_front",
            front,
            cwe_id=None,
            p_exploit=0.8,
            impact=1_000.0,
            criticality=0.5,
            privileges_required=PrivilegeLevel.NONE,
            privilege_gained=PrivilegeLevel.USER,
        ),
        make_enriched(
            "f_back",
            back,
            cwe_id=78,  # OS command injection -> SYSTEM
            p_exploit=0.5,
            impact=400_000.0,
            criticality=1.0,
        ),
    ]
    return make_scan((front, back), enriched), enriched, ComponentCConfig(admit_lateral_edges=admit)


def test_lateral_edges_come_from_observed_links_between_hosts() -> None:
    scan, enriched, config = lateral_world()
    graph, summary = AttackGraphBuilder(config).build(scan, enriched)

    lateral = graph.edges["state:www.example.com:NONE", "state:db.example.com:NONE"]
    assert lateral["kind"] == "lateral"
    assert lateral["probability"] == pytest.approx(0.5)
    assert lateral["finding_ids"] == ()
    # The internal host is not attached to the entry state directly.
    assert not graph.has_edge("state:internet:NONE", "state:db.example.com:NONE")
    assert graph.has_edge("state:internet:NONE", "state:www.example.com:NONE")
    assert summary.total_risk > 0.0


def test_without_lateral_edges_the_internal_host_is_unreachable() -> None:
    scan, enriched, config = lateral_world(admit=False)
    graph, summary = AttackGraphBuilder(config).build(scan, enriched)

    assert not graph.has_edge("state:www.example.com:NONE", "state:db.example.com:NONE")
    # The internal host still holds its 400_000, it is simply unreachable, so all that
    # survives in R(G) is the front host's own 1_000 * 0.5 at probability 0.8.
    assert "state:db.example.com:SYSTEM" in summary.target_nodes
    assert summary.total_risk == pytest.approx(400.0)

    # Admitting the observed link puts it back: 400_000 at 0.5 (lateral) x 0.5 (f_back).
    _, connected = AttackGraphBuilder(ComponentCConfig(admit_lateral_edges=True)).build(scan, enriched)
    assert connected.total_risk == pytest.approx(400.0 + 400_000.0 * 0.5 * 0.5)


def test_alternative_exploits_share_one_transition_and_each_lose_value() -> None:
    """Two routes to the same escalation: patching either one leaves the other standing."""
    front = make_endpoint("ep_front", "/portal")
    alt = make_endpoint("ep_alt", "/portal/v2")
    enriched = [
        make_enriched(
            "f_a", front, cwe_id=None, p_exploit=0.8, impact=100_000.0, criticality=1.0,
            privileges_required=PrivilegeLevel.NONE, privilege_gained=PrivilegeLevel.USER,
        ),
        make_enriched(
            "f_b", alt, cwe_id=None, p_exploit=0.4, impact=100_000.0, criticality=1.0,
            privileges_required=PrivilegeLevel.NONE, privilege_gained=PrivilegeLevel.USER,
        ),
    ]
    scan = make_scan((front, alt), enriched)
    scorer = ReachabilityChainScorer()
    summary = scorer.build(scan, enriched)
    scores = scorer.score(SCAN_ID)

    # The graph keeps the better of the two on the single transition.
    assert scorer.graph_for(SCAN_ID).edges[f"state:{HOST}:NONE", f"state:{HOST}:USER"][
        "probability"
    ] == pytest.approx(0.8)
    # Patching the stronger exploit falls back to the weaker one rather than to nothing.
    assert scores["f_a"].reach_delta == pytest.approx(summary.total_risk * 0.5)
    assert scores["f_b"].reach_delta == pytest.approx(0.0)
    assert scorer.total_risk_after_patching(SCAN_ID, {"f_a", "f_b"}) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# The privilege table
# --------------------------------------------------------------------------


def assessment(**kwargs) -> ExploitabilityAssessment:
    base = dict(finding_id="f", exploit_feasibility=0.5, impact_c=0.3, impact_i=0.3, impact_a=0.3)
    base.update(kwargs)
    return ExploitabilityAssessment(**base)


@pytest.mark.parametrize("cwe_id", [78, 94, 502])
def test_command_and_code_execution_cwes_confer_system(cwe_id: int) -> None:
    assert privileges_for(cwe_id, assessment()) == (PrivilegeLevel.NONE, PrivilegeLevel.SYSTEM)


@pytest.mark.parametrize("cwe_id", [287, 306, 862])
def test_authentication_bypass_cwes_confer_admin_from_nothing(cwe_id: int) -> None:
    assert privileges_for(cwe_id, assessment()) == (PrivilegeLevel.NONE, PrivilegeLevel.ADMIN)


@pytest.mark.parametrize("cwe_id", [269, 863])
def test_privilege_escalation_cwes_presuppose_a_user(cwe_id: int) -> None:
    assert privileges_for(cwe_id, assessment()) == (PrivilegeLevel.USER, PrivilegeLevel.ADMIN)


@pytest.mark.parametrize("cwe_id", [89, 22, 98, 611])
def test_data_access_cwes_scale_with_cia_impact(cwe_id: int) -> None:
    low = assessment(impact_c=0.3, impact_i=0.1, impact_a=0.0)
    medium = assessment(impact_c=0.9, impact_i=0.7, impact_a=0.6)
    total = assessment(impact_c=1.0, impact_i=1.0, impact_a=0.9)

    assert privileges_for(cwe_id, low)[1] == PrivilegeLevel.USER
    assert privileges_for(cwe_id, medium)[1] == PrivilegeLevel.ADMIN
    assert privileges_for(cwe_id, total)[1] == PrivilegeLevel.SYSTEM
    assert privileges_for(cwe_id, low)[0] == PrivilegeLevel.NONE


@pytest.mark.parametrize("cwe_id", [79, 352])
def test_victim_dependent_cwes_confer_user(cwe_id: int) -> None:
    assert privileges_for(cwe_id, assessment()) == (PrivilegeLevel.NONE, PrivilegeLevel.USER)


@pytest.mark.parametrize("cwe_id", [200, 16])
def test_disclosure_and_configuration_cwes_confer_nothing(cwe_id: int) -> None:
    assert privileges_for(cwe_id, assessment()) == (PrivilegeLevel.NONE, PrivilegeLevel.NONE)


def test_unknown_cwe_falls_back_to_the_assessment() -> None:
    claimed = assessment(
        privileges_required=PrivilegeLevel.USER, privilege_gained=PrivilegeLevel.SYSTEM
    )
    assert privileges_for(4242, claimed) == (PrivilegeLevel.USER, PrivilegeLevel.SYSTEM)
    assert privileges_for(None, claimed) == (PrivilegeLevel.USER, PrivilegeLevel.SYSTEM)


def test_a_rule_can_never_describe_a_loss_of_privilege() -> None:
    """Downward movement is the privilege-implication edges' job, never an exploit's."""
    backwards = assessment(
        privileges_required=PrivilegeLevel.ADMIN, privilege_gained=PrivilegeLevel.NONE
    )
    required, gained = privileges_for(None, backwards)
    assert required == PrivilegeLevel.ADMIN and gained == PrivilegeLevel.ADMIN


def test_endpoint_authentication_raises_the_required_privilege() -> None:
    """A SQL injection behind an admin login cannot be run by an anonymous attacker."""
    public = make_endpoint("ep_pub", "/search", auth=PrivilegeLevel.NONE)
    gated = make_endpoint("ep_gated", "/admin/search", auth=PrivilegeLevel.ADMIN)
    enriched = [
        make_enriched("f_pub", public, cwe_id=89, impact=1_000.0),
        make_enriched("f_gated", gated, cwe_id=89, impact=1_000.0),
    ]
    builder = AttackGraphBuilder()
    transitions = {t.finding_id: t for t in builder.transitions(make_scan((public, gated), enriched), enriched)}

    assert transitions["f_pub"].required == PrivilegeLevel.NONE
    assert transitions["f_pub"].escalates is True
    assert transitions["f_gated"].required == PrivilegeLevel.ADMIN
    assert transitions["f_gated"].escalates is False


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_building_a_finding_from_another_scan_is_refused(chokepoint) -> None:
    scan, enriched = chokepoint
    stray = make_enriched("f_stray", make_endpoint("ep_x", "/x"), cwe_id=79, scan_id="other_scan")
    with pytest.raises(GraphError):
        AttackGraphBuilder().build(scan, [*enriched, stray])


def test_scoring_an_unbuilt_scan_is_refused() -> None:
    with pytest.raises(GraphError):
        ReachabilityChainScorer().score("never_built")


def test_monotonicity_is_verified_at_build_time_when_configured(chokepoint) -> None:
    scan, enriched = chokepoint
    verified = ReachabilityChainScorer(ComponentCConfig(verify_monotonicity=True)).build(scan, enriched)
    unverified = ReachabilityChainScorer(ComponentCConfig(verify_monotonicity=False)).build(scan, enriched)

    assert verified.monotone_verified is True
    assert unverified.monotone_verified is False
    assert verified.total_risk == pytest.approx(unverified.total_risk)


def test_scoring_hundreds_of_findings_stays_tractable() -> None:
    """The state graph is small however many findings there are; scoring must exploit that."""
    endpoints = []
    enriched = []
    for index in range(300):
        host = f"host{index % 6}.example.com"
        endpoint = make_endpoint(f"ep_{index}", f"/path/{index}", host=host)
        endpoints.append(endpoint)
        enriched.append(
            make_enriched(
                f"f_{index}",
                endpoint,
                cwe_id=[79, 89, 287, 78, 200][index % 5],
                p_exploit=0.1 + (index % 9) / 10.0,
                impact=1_000.0 * (index % 40 + 1),
                criticality=0.1 + (index % 9) / 10.0,
            )
        )
    scan = make_scan(tuple(endpoints), enriched)
    scorer = ReachabilityChainScorer(ComponentCConfig(verify_monotonicity=False))
    summary = scorer.build(scan, enriched)
    scores = scorer.score(SCAN_ID)

    assert len(scores) == 300
    assert len(summary.nodes) <= 1 + 6 * len(PrivilegeLevel)
    assert all(score.reach_delta >= 0.0 for score in scores.values())
    assert summary.total_risk > 0.0
