"""Budget-constrained remediation selection (Gap 10).

The claims under test: the dynamic programme is genuinely exact (checked against brute
force on small instances), the greedy fallback is a sane heuristic and is *labelled* as
one, remediation effort is charged once per root cause, and the rank-prefix control
behaves like a team working a queue rather than like an optimiser.
"""

from __future__ import annotations

import itertools
from datetime import date, datetime

import numpy as np
import pytest

from vulnpriority.core.config import PipelineConfig, SelectionConfig
from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    EndpointFunction,
    HttpMethod,
    LabelSource,
    PrivilegeLevel,
    Provenance,
    RankerName,
    ScannerSeverity,
    SelectionMethod,
)
from vulnpriority.core.errors import ConfigError
from vulnpriority.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    BusinessImpact,
    ChainScore,
    Endpoint,
    EnrichedFinding,
    ExploitLikelihood,
    ExploitabilityAssessment,
    Finding,
    GroundTruthLabel,
    LabelSet,
    RankedFinding,
    RemediationCost,
    SelectionResult,
    UntrustedText,
)
from vulnpriority.select.knapsack import (
    MIN_HOURS,
    SelectionItem,
    cluster_items,
    items_from_enriched,
    items_from_ranking,
    select_under_budget,
)

SEED = 7
AS_OF = date(2024, 6, 1)
SCANNED_AT = datetime(2024, 5, 1, 9, 0, 0)
GRANULARITY = 0.5


def item(
    finding_id: str,
    value: float,
    hours: float,
    *,
    dedup_key: str | None = None,
    rank: int | None = None,
    exploited: bool = False,
) -> SelectionItem:
    return SelectionItem(
        finding_id=finding_id,
        value=value,
        hours=hours,
        dedup_key=dedup_key,
        rank=rank,
        exploited=exploited,
    )


def brute_force(items: list[SelectionItem], budget_hours: float) -> float:
    """Best achievable value over every subset of *clusters*, by exhaustion."""
    clusters = cluster_items(items)
    best = 0.0
    for size in range(len(clusters) + 1):
        for combination in itertools.combinations(clusters, size):
            hours = sum(cluster.hours for cluster in combination)
            if hours > budget_hours + 1e-9:
                continue
            best = max(best, sum(cluster.value for cluster in combination))
    return best


# --------------------------------------------------------------------------
# Exactness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("instance", range(12))
def test_dp_matches_brute_force_on_small_instances(instance: int) -> None:
    """Exactness is the whole point of paying for a dynamic programme."""
    rng = np.random.default_rng(SEED + instance)
    n_items = int(rng.integers(4, 11))
    items = [
        item(
            f"f_{index}",
            value=float(rng.integers(0, 40)) * 1_000.0,
            hours=float(rng.integers(1, 13)) * GRANULARITY,
            rank=index + 1,
        )
        for index in range(n_items)
    ]
    budget = float(rng.integers(2, 16)) * GRANULARITY

    result = select_under_budget(
        items,
        budget_hours=budget,
        method=SelectionMethod.DP_EXACT,
        hour_granularity=GRANULARITY,
    )

    assert result.method == SelectionMethod.DP_EXACT
    assert result.risk_captured == pytest.approx(brute_force(items, budget))
    assert result.total_hours <= budget + 1e-9
    assert len(set(result.selected_ids)) == len(result.selected_ids)


def test_dp_selection_is_internally_consistent() -> None:
    """The reported hours, value and ids all describe the same chosen set."""
    items = [item(f"f_{index}", value=1_000.0 * (index + 1), hours=1.0 + index * 0.5, rank=index + 1)
             for index in range(8)]
    result = select_under_budget(items, budget_hours=6.0, hour_granularity=GRANULARITY)

    chosen = {entry.finding_id: entry for entry in items if entry.finding_id in set(result.selected_ids)}
    assert result.total_hours == pytest.approx(sum(entry.hours for entry in chosen.values()))
    assert result.risk_captured == pytest.approx(sum(entry.value for entry in chosen.values()))
    assert isinstance(result, SelectionResult)


def test_hours_are_rounded_up_onto_the_grid_so_a_plan_is_never_understated() -> None:
    """A 0.6-hour fix costs a full grid unit; two of them will not fit in one."""
    items = [item("f_a", 100.0, 0.6, rank=1), item("f_b", 100.0, 0.6, rank=2)]
    result = select_under_budget(items, budget_hours=1.0, hour_granularity=GRANULARITY)
    assert len(result.selected_ids) == 1


def test_an_item_heavier_than_the_budget_can_never_be_selected() -> None:
    items = [item("f_big", 1_000_000.0, 80.0, rank=1), item("f_small", 1.0, 1.0, rank=2)]
    result = select_under_budget(items, budget_hours=8.0, hour_granularity=GRANULARITY)
    assert result.selected_ids == ("f_small",)


def test_empty_input_produces_an_empty_plan() -> None:
    result = select_under_budget([], budget_hours=40.0)
    assert result.selected_ids == ()
    assert result.risk_captured == 0.0
    assert result.risk_capture_fraction == 0.0
    assert result.exploited_total == 0


# --------------------------------------------------------------------------
# Greedy fallback
# --------------------------------------------------------------------------


def test_greedy_fallback_takes_over_above_the_exact_limit_and_says_so() -> None:
    """An exact label on a heuristic answer would be the dishonest failure mode."""
    items = [item(f"f_{index}", 1_000.0 * (index + 1), 1.0, rank=index + 1) for index in range(20)]
    result = select_under_budget(
        items,
        budget_hours=10.0,
        method=SelectionMethod.DP_EXACT,
        max_items_for_exact=5,
        hour_granularity=GRANULARITY,
    )
    assert result.method == SelectionMethod.GREEDY_RATIO
    assert result.total_hours <= 10.0 + 1e-9


def test_greedy_orders_by_value_per_hour() -> None:
    items = [
        item("f_cheap_good", 1_000.0, 1.0, rank=3),   # 1000/hour
        item("f_dear_best", 9_000.0, 3.0, rank=2),    # 3000/hour
        item("f_dear_poor", 2_000.0, 4.0, rank=1),    # 500/hour
    ]
    result = select_under_budget(items, budget_hours=4.0, method=SelectionMethod.GREEDY_RATIO)
    assert set(result.selected_ids) == {"f_dear_best", "f_cheap_good"}
    assert result.risk_captured == pytest.approx(10_000.0)


def test_greedy_is_feasible_but_can_be_beaten_by_the_exact_optimum() -> None:
    """The classic knapsack trap, kept as evidence that ``dp_exact`` is worth having."""
    items = [
        item("f_x", 7_000.0, 3.0, rank=1),
        item("f_y", 4_000.0, 2.0, rank=2),
        item("f_z", 4_000.0, 2.0, rank=3),
    ]
    greedy = select_under_budget(items, budget_hours=4.0, method=SelectionMethod.GREEDY_RATIO)
    exact = select_under_budget(items, budget_hours=4.0, method=SelectionMethod.DP_EXACT)

    assert greedy.selected_ids == ("f_x",)
    assert greedy.risk_captured == pytest.approx(7_000.0)
    assert set(exact.selected_ids) == {"f_y", "f_z"}
    assert exact.risk_captured == pytest.approx(8_000.0)
    assert greedy.total_hours <= 4.0 and exact.total_hours <= 4.0


@pytest.mark.parametrize("instance", range(8))
def test_greedy_never_beats_the_exact_optimum(instance: int) -> None:
    rng = np.random.default_rng(100 + instance)
    items = [
        item(
            f"f_{index}",
            value=float(rng.integers(1, 30)) * 1_000.0,
            hours=float(rng.integers(1, 10)) * GRANULARITY,
            rank=index + 1,
        )
        for index in range(9)
    ]
    budget = float(rng.integers(4, 14)) * GRANULARITY
    greedy = select_under_budget(items, budget_hours=budget, method=SelectionMethod.GREEDY_RATIO)
    exact = select_under_budget(items, budget_hours=budget, method=SelectionMethod.DP_EXACT)
    assert greedy.risk_captured <= exact.risk_captured + 1e-9
    assert greedy.total_hours <= budget + 1e-9


# --------------------------------------------------------------------------
# Rank prefix control
# --------------------------------------------------------------------------


def test_rank_prefix_walks_the_ranking_and_stops_when_the_budget_runs_out() -> None:
    """The control condition: no hunting down the list for something that fits."""
    items = [
        item("f_1", 5_000.0, 2.0, rank=1),
        item("f_2", 4_000.0, 2.0, rank=2),
        item("f_3", 3_000.0, 8.0, rank=3),   # does not fit: everything after it is lost
        item("f_4", 2_500.0, 1.0, rank=4),
    ]
    control = select_under_budget(items, budget_hours=5.0, method=SelectionMethod.RANK_PREFIX)
    exact = select_under_budget(items, budget_hours=5.0, method=SelectionMethod.DP_EXACT)

    assert control.selected_ids == ("f_1", "f_2")
    assert control.total_hours == pytest.approx(4.0)
    # The optimiser spends the same budget better, which is the number Gap 10 wants.
    assert set(exact.selected_ids) == {"f_1", "f_2", "f_4"}
    assert exact.risk_captured > control.risk_captured


def test_rank_prefix_falls_back_to_value_order_without_ranks() -> None:
    items = [item("f_a", 1_000.0, 1.0), item("f_b", 9_000.0, 1.0)]
    result = select_under_budget(items, budget_hours=1.0, method=SelectionMethod.RANK_PREFIX)
    assert result.selected_ids == ("f_b",)


# --------------------------------------------------------------------------
# Clusters: remediation charged once per root cause
# --------------------------------------------------------------------------


def test_cluster_cost_is_charged_once_not_once_per_alert() -> None:
    """One root cause on five endpoints is one fix, and the budget must see it that way."""
    cluster = [
        item(f"f_dup{index}", 10_000.0, 8.0, dedup_key="dk_shared", rank=index + 1)
        for index in range(5)
    ]
    rival = item("f_solo", 30_000.0, 8.0, dedup_key="dk_solo", rank=6)

    result = select_under_budget([*cluster, rival], budget_hours=8.0, hour_granularity=GRANULARITY)

    # Charged per alert the cluster would cost 40 hours and be unaffordable.
    assert sum(entry.hours for entry in cluster) == pytest.approx(40.0)
    assert set(result.selected_ids) == {f"f_dup{index}" for index in range(5)}
    assert result.total_hours == pytest.approx(8.0)
    assert result.risk_captured == pytest.approx(50_000.0)


def test_clustering_sums_value_and_takes_the_hours_once() -> None:
    clusters = cluster_items(
        [
            item("f_a", 100.0, 3.0, dedup_key="dk", rank=4),
            item("f_b", 250.0, 3.0, dedup_key="dk", rank=2),
            item("f_c", 900.0, 5.0, dedup_key=None, rank=1),
        ]
    )
    by_key = {cluster.key: cluster for cluster in clusters}

    assert by_key["dk"].value == pytest.approx(350.0)
    assert by_key["dk"].hours == pytest.approx(3.0)
    assert sorted(by_key["dk"].finding_ids) == ["f_a", "f_b"]
    assert by_key["dk"].best_rank == 2
    # A finding with no dedup key stands alone under its own id.
    assert by_key["f_c"].finding_ids == ("f_c",)
    # Deterministic order: best rank first.
    assert [cluster.key for cluster in clusters] == ["f_c", "dk"]


def test_selecting_a_cluster_selects_every_member() -> None:
    items = [
        item("f_x1", 500.0, 2.0, dedup_key="dk_x", rank=1),
        item("f_x2", 500.0, 2.0, dedup_key="dk_x", rank=9),
    ]
    result = select_under_budget(items, budget_hours=2.0)
    assert result.selected_ids == ("f_x1", "f_x2")
    assert result.total_hours == pytest.approx(2.0)


def test_zero_hour_items_are_floored_so_the_knapsack_stays_bounded() -> None:
    clusters = cluster_items([item("f_free", 100.0, 0.0)])
    assert clusters[0].hours == pytest.approx(MIN_HOURS)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_risk_capture_fraction_is_the_share_of_available_value() -> None:
    items = [item("f_a", 750.0, 1.0, rank=1), item("f_b", 250.0, 1.0, rank=2)]
    result = select_under_budget(items, budget_hours=1.0, hour_granularity=GRANULARITY)
    assert result.risk_captured == pytest.approx(750.0)
    assert result.risk_capture_fraction == pytest.approx(0.75)

    everything = select_under_budget(items, budget_hours=2.0, hour_granularity=GRANULARITY)
    assert everything.risk_capture_fraction == pytest.approx(1.0)


def test_exploited_capture_is_reported_when_labels_are_supplied() -> None:
    items = [
        item("f_hit", 9_000.0, 1.0, rank=1),
        item("f_miss", 100.0, 1.0, rank=2),
        item("f_other", 50.0, 40.0, rank=3),
    ]
    labels = LabelSet(
        observation_cutoff=AS_OF,
        labels=(
            GroundTruthLabel(
                finding_id="f_hit", exploited=True, relevance_grade=4, sources=(LabelSource.KEV,)
            ),
            GroundTruthLabel(
                finding_id="f_other",
                exploited=True,
                relevance_grade=3,
                sources=(LabelSource.EXPLOIT_EVIDENCE,),
            ),
            GroundTruthLabel(finding_id="f_miss", exploited=False),
        ),
    )
    result = select_under_budget(items, budget_hours=1.0, labels=labels, ranker=RankerName.EXPECTED_LOSS)

    assert result.selected_ids == ("f_hit",)
    assert result.exploited_captured == 1
    assert result.exploited_total == 2
    assert result.ranker == RankerName.EXPECTED_LOSS


def test_labels_override_whatever_the_items_claimed() -> None:
    items = [item("f_a", 10.0, 1.0, rank=1, exploited=True)]
    labels = LabelSet(observation_cutoff=AS_OF, labels=(GroundTruthLabel(finding_id="f_a", exploited=False),))
    result = select_under_budget(items, budget_hours=1.0, labels=labels)
    assert result.exploited_total == 0 and result.exploited_captured == 0


def test_scan_id_and_method_are_carried_into_the_result() -> None:
    result = select_under_budget(
        [item("f_a", 1.0, 1.0, rank=1)],
        budget_hours=2.0,
        scan_id="scan_42",
        method=SelectionMethod.RANK_PREFIX,
    )
    assert result.scan_id == "scan_42"
    assert result.method == SelectionMethod.RANK_PREFIX
    assert result.budget_hours == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_configuration_supplies_the_defaults() -> None:
    config = SelectionConfig(budget_hours=3.0, hour_granularity=0.5, method=SelectionMethod.RANK_PREFIX)
    items = [item("f_a", 100.0, 2.0, rank=1), item("f_b", 900.0, 2.0, rank=2)]
    result = select_under_budget(items, config=config)
    assert result.budget_hours == pytest.approx(3.0)
    assert result.method == SelectionMethod.RANK_PREFIX
    assert result.selected_ids == ("f_a",)


def test_the_whole_pipeline_config_is_accepted_too() -> None:
    result = select_under_budget([item("f_a", 1.0, 1.0, rank=1)], config=PipelineConfig())
    assert result.budget_hours == pytest.approx(SelectionConfig().budget_hours)


@pytest.mark.parametrize("budget,granularity", [(0.0, 0.5), (-1.0, 0.5), (10.0, 0.0), (10.0, -0.5)])
def test_nonsense_budgets_are_refused(budget: float, granularity: float) -> None:
    with pytest.raises(ConfigError):
        select_under_budget([item("f_a", 1.0, 1.0)], budget_hours=budget, hour_granularity=granularity)


# --------------------------------------------------------------------------
# Building items from the pipeline's own objects
# --------------------------------------------------------------------------


def make_enriched(finding_id: str, *, expected_loss: float, hours: float, dedup_key: str) -> EnrichedFinding:
    endpoint = Endpoint(
        endpoint_id=f"ep_{finding_id}",
        app_id="app_s",
        host="shop.example.com",
        url=f"https://shop.example.com/{finding_id}",
        path=f"/{finding_id}",
        method=HttpMethod.GET,
        auth_required=PrivilegeLevel.NONE,
    )
    finding = Finding(
        finding_id=finding_id,
        scan_id="scan_s",
        app_id="app_s",
        endpoint_id=endpoint.endpoint_id,
        name=finding_id,
        cwe_id=89,
        scanner="zap",
        scanner_severity=ScannerSeverity.HIGH,
        description=UntrustedText(text=finding_id, provenance=Provenance.SCANNER_OUTPUT),
        observed_at=SCANNED_AT,
        dedup_key=dedup_key,
    )
    return EnrichedFinding(
        finding=finding,
        endpoint=endpoint,
        asset=AssetCriticality(
            endpoint_id=endpoint.endpoint_id,
            function=EndpointFunction.API_DATA,
            criticality=0.5,
            data_sensitivity=0.5,
            exposure=1.0,
        ),
        exploitability=ExploitabilityAssessment(finding_id=finding_id, exploit_feasibility=0.5),
        applicability=ApplicabilityAssessment(finding_id=finding_id, verdict=ApplicabilityVerdict.APPLICABLE),
        likelihood=ExploitLikelihood(
            finding_id=finding_id,
            attacker="opportunistic",
            p_exploit=0.5,
            p_exploit_uncapped=0.5,
            horizon_days=90,
        ),
        impact=BusinessImpact(finding_id=finding_id, total=expected_loss * 2.0),
        remediation=RemediationCost(finding_id=finding_id, hours=hours, cost=hours * 120.0),
        expected_loss=expected_loss,
        as_of=AS_OF,
    )


def test_items_from_enriched_uses_chain_adjusted_loss() -> None:
    enriched = [
        make_enriched("f_a", expected_loss=1_000.0, hours=4.0, dedup_key="dk_a"),
        make_enriched("f_b", expected_loss=500.0, hours=2.0, dedup_key="dk_b"),
    ]
    chain = {
        "f_a": ChainScore(finding_id="f_a", reach_delta=250.0),
        "f_b": ChainScore(finding_id="f_b", reach_delta=10_000.0),
    }
    items = items_from_enriched(enriched, chain, chain_weight=2.0, order=["f_b", "f_a"])
    by_id = {entry.finding_id: entry for entry in items}

    assert by_id["f_a"].value == pytest.approx(1_000.0 + 2.0 * 250.0)
    assert by_id["f_b"].value == pytest.approx(500.0 + 2.0 * 10_000.0)
    assert by_id["f_a"].hours == pytest.approx(4.0) and by_id["f_a"].dedup_key == "dk_a"
    assert (by_id["f_b"].rank, by_id["f_a"].rank) == (1, 2)

    # Without Component C the value is exactly the expected loss.
    plain = {entry.finding_id: entry for entry in items_from_enriched(enriched)}
    assert plain["f_b"].value == pytest.approx(500.0)


def test_items_from_ranking_joins_the_ranking_to_its_remediation_cost() -> None:
    enriched = [
        make_enriched("f_a", expected_loss=1_000.0, hours=4.0, dedup_key="dk_a"),
        make_enriched("f_b", expected_loss=500.0, hours=2.0, dedup_key="dk_a"),
    ]
    ranked = [
        RankedFinding(
            finding_id="f_b", scan_id="scan_s", rank=1, score=9.0, chain_adjusted_loss=20_500.0
        ),
        RankedFinding(
            finding_id="f_a", scan_id="scan_s", rank=2, score=1.0, chain_adjusted_loss=1_500.0
        ),
        RankedFinding(
            finding_id="f_missing", scan_id="scan_s", rank=3, score=0.0, chain_adjusted_loss=99.0
        ),
    ]
    items = items_from_ranking(ranked, enriched)

    assert [entry.finding_id for entry in items] == ["f_b", "f_a"]  # unknown ids are dropped
    assert items[0].value == pytest.approx(20_500.0) and items[0].rank == 1
    assert items[1].hours == pytest.approx(4.0)

    # Both share a dedup key, so the plan buys the pair for the larger of the two costs.
    result = select_under_budget(items, budget_hours=4.0, scan_id="scan_s")
    assert set(result.selected_ids) == {"f_a", "f_b"}
    assert result.total_hours == pytest.approx(4.0)
    assert result.risk_capture_fraction == pytest.approx(1.0)
