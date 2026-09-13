"""Longitudinal remediation simulation (DESIGN.md 3.9, Gap 10).

The central test is
:func:`test_a_policy_that_fixes_every_exploited_finding_first_reports_a_large_reduction`:
on a backlog far larger than the remediation budget - which is the realistic case - a
policy that fixes every confirmed-exploited finding first must report a decisive reduction
against one that fixes none of them.

That test exists because the first version of this module computed the reduction on
``exposure_days_total``, which a remediation ordering cannot move: the hundreds of findings
the budget never reaches accrue the full horizon under every policy and dominate the sum.
:func:`test_total_exposure_is_near_constant_when_the_backlog_exceeds_the_budget` pins that
as a property, and the test above then asserts the headline reports the difference anyway.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from vulnpriority.core.config import PipelineConfig, SimulationConfig
from vulnpriority.core.enums import (
    EndpointFunction,
    HttpMethod,
    LabelSource,
    PrivilegeLevel,
    Provenance,
    RankerName,
    ScannerSeverity,
)
from vulnpriority.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    BusinessImpact,
    Endpoint,
    EnrichedFinding,
    ExploitLikelihood,
    ExploitabilityAssessment,
    Finding,
    GroundTruthLabel,
    LabelSet,
    RankedFinding,
    RankingResult,
    RemediationCost,
    SimulationResult,
    UntrustedText,
)
from vulnpriority.eval.simulation import (
    DAYS_PER_WEEK,
    LongitudinalSimulator,
    prevention_rate,
    summarise,
)

START = datetime(2024, 3, 1, 9, 0, 0)
START_DATE = START.date()
AS_OF = date(2024, 3, 1)

ENDPOINT = Endpoint(
    endpoint_id="ep_1",
    app_id="app1",
    host="shop.example.com",
    url="https://shop.example.com/api/x",
    path="/api/x",
    method=HttpMethod.POST,
    auth_required=PrivilegeLevel.NONE,
)


def enriched(
    finding_id: str,
    hours: float,
    expected_loss: float,
    dedup: str | None = None,
) -> EnrichedFinding:
    finding = Finding(
        finding_id=finding_id,
        scan_id="scan_1",
        app_id="app1",
        endpoint_id="ep_1",
        name="Injection",
        scanner="zap",
        scanner_severity=ScannerSeverity.HIGH,
        description=UntrustedText(text="text", provenance=Provenance.SCANNER_OUTPUT),
        observed_at=START,
        dedup_key=dedup or finding_id,
    )
    return EnrichedFinding(
        finding=finding,
        endpoint=ENDPOINT,
        asset=AssetCriticality(
            endpoint_id="ep_1",
            function=EndpointFunction.API_DATA,
            criticality=0.5,
            data_sensitivity=0.5,
            exposure=1.0,
        ),
        exploitability=ExploitabilityAssessment(finding_id=finding_id, exploit_feasibility=0.5),
        applicability=ApplicabilityAssessment(finding_id=finding_id),
        likelihood=ExploitLikelihood(
            finding_id=finding_id,
            attacker="opportunistic",
            p_exploit=0.2,
            p_exploit_uncapped=0.2,
            horizon_days=90,
        ),
        impact=BusinessImpact(finding_id=finding_id, total=expected_loss * 5.0),
        remediation=RemediationCost(finding_id=finding_id, hours=hours, cost=hours * 120.0),
        expected_loss=expected_loss,
        as_of=AS_OF,
    )


def label(finding_id: str, exploited: bool, evidence_day: int | None = None) -> GroundTruthLabel:
    return GroundTruthLabel(
        finding_id=finding_id,
        exploited=exploited,
        relevance_grade=4 if exploited else 0,
        sources=(LabelSource.SYNTHETIC_ORACLE,) if exploited else (),
        first_evidence_date=(
            START_DATE + timedelta(days=evidence_day) if evidence_day is not None else None
        ),
    )


@pytest.fixture
def world():
    """Ten findings: two cheap exploited ones, two expensive irrelevant ones, six routine.

    Weekly capacity is 20 hours, so exactly how the queue is ordered decides how much of
    the estate is closed by the end of each week.
    """
    findings = [enriched("f_exp_a", 2.0, 100_000.0), enriched("f_exp_b", 2.0, 100_000.0)]
    findings += [enriched("f_big_0", 10.0, 100.0), enriched("f_big_1", 10.0, 100.0)]
    findings += [enriched(f"f_mid_{index}", 2.0, 100.0) for index in range(6)]
    labels = LabelSet(
        observation_cutoff=date(2024, 12, 31),
        labels=tuple(
            [label("f_exp_a", True, 10), label("f_exp_b", True, 10)]
            + [label(item.finding_id, False) for item in findings[2:]]
        ),
    )
    return findings, labels


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(
        simulation=SimulationConfig(
            weeks=4,
            capacity_hours_per_week=20.0,
            policies=(RankerName.LAMBDAMART, RankerName.CVSS_ONLY),
            reference_policy=RankerName.CVSS_ONLY,
        )
    )


def orders(findings) -> dict[RankerName, list[str]]:
    routine = [item.finding_id for item in findings[2:]]
    exploited = ["f_exp_a", "f_exp_b"]
    return {
        # exploited first, then the cheap routine work, then the two expensive items
        RankerName.LAMBDAMART: exploited + [key for key in routine if key.startswith("f_mid")]
        + ["f_big_0", "f_big_1"],
        # the reference policy spends the first week on two expensive low-value findings
        RankerName.CVSS_ONLY: ["f_big_0", "f_big_1"]
        + [key for key in routine if key.startswith("f_mid")]
        + exploited,
    }


# ---------------------------------------------------------------------------
# The headline result
# ---------------------------------------------------------------------------


def test_fixing_exploited_findings_first_reduces_exposure(world, config) -> None:
    findings, labels = world
    results = LongitudinalSimulator().run(findings, labels, orders(findings), config)
    by_policy = {item.policy: item for item in results}
    smart = by_policy[RankerName.LAMBDAMART]
    reference = by_policy[RankerName.CVSS_ONLY]

    # Hand-checked: 20 hours a week over ten findings.
    # reference  -> week 1 closes the two 10-hour findings; everything else waits to week 2
    #               exposure = 2*7 + 8*14 = 126 days
    # exploited-first -> week 1 closes eight 2-hour findings; the expensive pair waits
    #               exposure = 8*7 + 2*14 = 84 days
    assert reference.exposure_days_total == pytest.approx(126.0)
    assert smart.exposure_days_total == pytest.approx(84.0)
    assert smart.exposure_days_total < reference.exposure_days_total

    # the exploited findings specifically are closed a week earlier
    assert smart.exposure_days_exploited == pytest.approx(14.0)
    assert reference.exposure_days_exploited == pytest.approx(28.0)

    # and the money-weighted exposure, which is what the programme actually pays
    assert smart.expected_loss_days < reference.expected_loss_days
    assert smart.expected_loss_days == pytest.approx(
        100_000.0 * 14 + 100.0 * (6 * 7 + 2 * 14)
    )


def test_the_counterfactual_is_counted_prevented_versus_not(world, config) -> None:
    """Exploitation evidence lands on day 10: week-1 remediation beats it, week-2 does not."""
    findings, labels = world
    results = {
        item.policy: item
        for item in LongitudinalSimulator().run(findings, labels, orders(findings), config)
    }
    smart, reference = results[RankerName.LAMBDAMART], results[RankerName.CVSS_ONLY]

    assert smart.exploited_total == reference.exploited_total == 2
    assert smart.exploited_remediated_before_exploit == 2
    assert reference.exploited_remediated_before_exploit == 0


def test_reduction_is_measured_on_the_exploited_findings_not_the_total(world, config) -> None:
    """The headline reduction must track the part of the outcome an ordering controls."""
    findings, labels = world
    results = {
        item.policy: item
        for item in LongitudinalSimulator().run(findings, labels, orders(findings), config)
    }
    assert results[RankerName.CVSS_ONLY].reduction_vs_cvss == pytest.approx(0.0)
    # exploited exposure 14 against the reference's 28, not total exposure 84 against 126
    assert results[RankerName.LAMBDAMART].reduction_vs_cvss == pytest.approx((28 - 14) / 28)
    assert results[RankerName.LAMBDAMART].reduction_vs_cvss != pytest.approx((126 - 84) / 126)


def test_results_are_the_frozen_contract_type(world, config) -> None:
    findings, labels = world
    for item in LongitudinalSimulator().run(findings, labels, orders(findings), config):
        assert isinstance(item, SimulationResult)
        assert item.weeks == 4
        assert item.capacity_hours_per_week == 20.0
        assert len(item.weekly_cumulative_exposure) == 4


# ---------------------------------------------------------------------------
# The week loop
# ---------------------------------------------------------------------------


def test_cumulative_exposure_is_non_decreasing_and_flattens_once_work_is_done(world, config) -> None:
    findings, labels = world
    results = LongitudinalSimulator().run(findings, labels, orders(findings), config)
    for item in results:
        curve = item.weekly_cumulative_exposure
        assert list(curve) == sorted(curve)
        assert curve[-1] == pytest.approx(item.exposure_days_total)
        # everything is closed by week 2, so weeks 3 and 4 add nothing
        assert curve[2] == pytest.approx(curve[3])


def test_work_carries_over_between_weeks_so_nothing_stalls(config) -> None:
    """A finding costing more than a week of capacity is still eventually remediated."""
    findings = [enriched("f_huge", 50.0, 1000.0)]
    labels = LabelSet(observation_cutoff=date(2024, 12, 31), labels=(label("f_huge", False),))
    result = LongitudinalSimulator().run(
        findings, labels, {RankerName.LAMBDAMART: ["f_huge"]}, config
    )[0]
    # 50 hours at 20 a week completes during week 3, i.e. at day 21
    assert result.exposure_days_total == pytest.approx(3 * DAYS_PER_WEEK)


def test_a_finding_never_reached_accrues_the_whole_horizon(config) -> None:
    """"Never got to it" is an outcome, not a missing value."""
    findings = [enriched("f_a", 20.0, 10.0), enriched("f_b", 200.0, 10.0)]
    labels = LabelSet(
        observation_cutoff=date(2024, 12, 31),
        labels=(label("f_a", False), label("f_b", False)),
    )
    result = LongitudinalSimulator().run(
        findings, labels, {RankerName.LAMBDAMART: ["f_a", "f_b"]}, config
    )[0]
    horizon = config.simulation.weeks * DAYS_PER_WEEK
    assert result.exposure_days_total == pytest.approx(DAYS_PER_WEEK + horizon)


def test_remediating_a_finding_closes_its_root_cause_cluster(config) -> None:
    """The framework ranks root causes; the same fix must not be paid for twice."""
    clustered = [enriched(f"f_c{index}", 10.0, 10.0, dedup="dk_shared") for index in range(3)]
    separate = [enriched(f"f_s{index}", 10.0, 10.0, dedup=f"dk_{index}") for index in range(3)]
    labels = LabelSet(
        observation_cutoff=date(2024, 12, 31),
        labels=tuple(label(item.finding_id, False) for item in clustered + separate),
    )

    simulator = LongitudinalSimulator()
    clustered_result = simulator.run(
        clustered, labels, {RankerName.LAMBDAMART: [item.finding_id for item in clustered]}, config
    )[0]
    separate_result = simulator.run(
        separate, labels, {RankerName.LAMBDAMART: [item.finding_id for item in separate]}, config
    )[0]

    # one 10-hour fix closes all three clustered findings in week 1
    assert clustered_result.exposure_days_total == pytest.approx(3 * DAYS_PER_WEEK)
    # three independent 10-hour fixes need two weeks
    assert separate_result.exposure_days_total > clustered_result.exposure_days_total


def test_findings_a_policy_never_mentions_are_still_charged_for(world, config) -> None:
    """A policy cannot win by declining to rank the difficult findings."""
    findings, labels = world
    partial = LongitudinalSimulator().run(
        findings, labels, {RankerName.LAMBDAMART: ["f_exp_a"]}, config
    )[0]
    assert partial.exposure_days_total > 0
    assert partial.exploited_total == 2


def test_a_ranking_result_is_accepted_in_place_of_a_list(world, config) -> None:
    findings, labels = world
    order = orders(findings)[RankerName.LAMBDAMART]
    ranking = RankingResult(
        ranker=RankerName.LAMBDAMART,
        items=tuple(
            RankedFinding(finding_id=finding_id, scan_id="scan_1", rank=index + 1, score=-float(index))
            for index, finding_id in enumerate(order)
        ),
    )
    from_result = LongitudinalSimulator().run(
        findings, labels, {RankerName.LAMBDAMART: ranking}, config
    )[0]
    from_list = LongitudinalSimulator().run(
        findings, labels, {RankerName.LAMBDAMART: order}, config
    )[0]
    assert from_result.exposure_days_total == from_list.exposure_days_total


def test_an_exploited_finding_with_no_evidence_date_is_not_counted_as_prevented(config) -> None:
    """The counterfactual cannot be judged without a date, so it is not asserted."""
    findings = [enriched("f_x", 2.0, 10.0)]
    labels = LabelSet(
        observation_cutoff=date(2024, 12, 31),
        labels=(label("f_x", True, evidence_day=None),),
    )
    result = LongitudinalSimulator().run(
        findings, labels, {RankerName.LAMBDAMART: ["f_x"]}, config
    )[0]
    assert result.exploited_total == 1
    assert result.exploited_remediated_before_exploit == 0


def test_simulation_is_deterministic_and_traceable(world, config) -> None:
    findings, labels = world
    simulator = LongitudinalSimulator()
    first = simulator.run(findings, labels, orders(findings), config)
    trace = simulator.traces[RankerName.LAMBDAMART]
    second = simulator.run(findings, labels, orders(findings), config)

    assert [item.exposure_days_total for item in first] == [
        item.exposure_days_total for item in second
    ]
    assert trace.remediated_day["f_exp_a"] == DAYS_PER_WEEK
    assert trace.never_remediated() == []


# ---------------------------------------------------------------------------
# The regression the coordinator caught: a capacity-bound backlog
# ---------------------------------------------------------------------------


@pytest.fixture
def overloaded_backlog():
    """A realistic backlog: far more work than the budget can ever reach.

    400 findings at 4 hours each is 1,600 hours against 26 weeks x 20 = 520, so roughly
    three quarters of the estate is never touched under any ordering. 40 of them are
    confirmed-exploited, with exploitation evidence landing on day 60.
    """
    findings = [enriched(f"f_{index:03d}", 4.0, 1_000.0) for index in range(400)]
    exploited_ids = {f"f_{index:03d}" for index in range(0, 400, 10)}     # 40 of 400
    labels = LabelSet(
        observation_cutoff=date(2025, 12, 31),
        labels=tuple(
            label(item.finding_id, item.finding_id in exploited_ids, evidence_day=60)
            for item in findings
        ),
    )
    return findings, labels, exploited_ids


@pytest.fixture
def long_run_config() -> PipelineConfig:
    return PipelineConfig(
        simulation=SimulationConfig(
            weeks=26,
            capacity_hours_per_week=20.0,
            policies=(RankerName.LAMBDAMART, RankerName.CVSS_ONLY),
            reference_policy=RankerName.CVSS_ONLY,
        )
    )


def test_total_exposure_is_near_constant_when_the_backlog_exceeds_the_budget(
    overloaded_backlog, long_run_config
) -> None:
    """The defect itself, pinned as a property rather than a bug.

    Every finding the budget never reaches accrues the full horizon under every ordering,
    so the total is set by capacity. Two maximally different policies land within a couple
    of percent of each other, which is why the headline reduction is not computed on it.
    """
    findings, labels, exploited_ids = overloaded_backlog
    everything = [item.finding_id for item in findings]
    results = {
        item.policy: item
        for item in LongitudinalSimulator().run(
            findings,
            labels,
            {
                RankerName.LAMBDAMART: [key for key in everything if key in exploited_ids]
                + [key for key in everything if key not in exploited_ids],
                RankerName.CVSS_ONLY: [key for key in everything if key not in exploited_ids]
                + [key for key in everything if key in exploited_ids],
            },
            long_run_config,
        )
    }
    smart, reference = results[RankerName.LAMBDAMART], results[RankerName.CVSS_ONLY]
    spread = abs(smart.exposure_days_total - reference.exposure_days_total)
    assert spread / reference.exposure_days_total < 0.05


def test_a_policy_that_fixes_every_exploited_finding_first_reports_a_large_reduction(
    overloaded_backlog, long_run_config
) -> None:
    """The test that would have caught the defect.

    One policy fixes every confirmed-exploited finding first; the other fixes none of them
    inside the horizon. On a capacity-bound backlog their *total* exposure is nearly
    identical, so a reduction computed on the total would report ~0% for a policy that
    prevented everything. Measured where it belongs, the reduction is decisive.
    """
    findings, labels, exploited_ids = overloaded_backlog
    everything = [item.finding_id for item in findings]
    exploited_first = [key for key in everything if key in exploited_ids] + [
        key for key in everything if key not in exploited_ids
    ]
    exploited_last = [key for key in everything if key not in exploited_ids] + [
        key for key in everything if key in exploited_ids
    ]

    simulator = LongitudinalSimulator()
    results = {
        item.policy: item
        for item in simulator.run(
            findings,
            labels,
            {RankerName.LAMBDAMART: exploited_first, RankerName.CVSS_ONLY: exploited_last},
            long_run_config,
        )
    }
    smart, reference = results[RankerName.LAMBDAMART], results[RankerName.CVSS_ONLY]

    # the reference never reaches a single exploited finding: all 40 carry the full horizon
    horizon = long_run_config.simulation.weeks * DAYS_PER_WEEK
    assert reference.exposure_days_exploited == pytest.approx(40 * horizon)
    assert reference.exploited_remediated_before_exploit == 0

    # the smart policy clears all 40 in the first eight weeks
    assert smart.exposure_days_exploited < 0.2 * reference.exposure_days_exploited
    assert smart.exploited_remediated_before_exploit == 40
    assert smart.reduction_vs_cvss is not None and smart.reduction_vs_cvss > 0.8

    # ... and the total, computed the old way, would have reported almost nothing
    naive_reduction = (
        reference.exposure_days_total - smart.exposure_days_total
    ) / reference.exposure_days_total
    assert naive_reduction < 0.05
    assert smart.reduction_vs_cvss > 10 * naive_reduction


def test_the_capacity_context_explains_why_the_total_cannot_move(
    overloaded_backlog, long_run_config
) -> None:
    """Eleven of 145 only reads correctly once the budget's reach is on the page."""
    findings, labels, _ = overloaded_backlog
    simulator = LongitudinalSimulator()
    simulator.run(
        findings,
        labels,
        {RankerName.CVSS_ONLY: [item.finding_id for item in findings]},
        long_run_config,
    )
    capacity = simulator.capacity
    assert capacity is not None
    assert capacity.weeks == 26 and capacity.capacity_hours_per_week == 20.0
    assert capacity.total_hours_available == pytest.approx(520.0)
    assert capacity.backlog_hours == pytest.approx(400 * 4.0)
    assert capacity.reachable_fraction == pytest.approx(520.0 / 1600.0)
    assert capacity.capacity_bound is True
    assert capacity.n_findings == 400 and capacity.n_exploited == 40

    payload = capacity.as_dict()
    assert payload["reachable_fraction"] == pytest.approx(0.325)
    assert payload["capacity_bound"] is True


def test_capacity_is_not_flagged_as_bound_when_the_budget_clears_the_backlog(config) -> None:
    findings = [enriched("f_a", 2.0, 10.0), enriched("f_b", 2.0, 10.0)]
    labels = LabelSet(
        observation_cutoff=date(2024, 12, 31),
        labels=(label("f_a", False), label("f_b", False)),
    )
    simulator = LongitudinalSimulator()
    simulator.run(findings, labels, {RankerName.CVSS_ONLY: ["f_a", "f_b"]}, config)
    assert simulator.capacity is not None
    assert simulator.capacity.capacity_bound is False
    assert simulator.capacity.reachable_fraction == pytest.approx(1.0)


def test_the_backlog_charges_each_root_cause_cluster_once(config) -> None:
    """The backlog must match how the simulator spends, or the reachable share is wrong."""
    findings = [enriched(f"f_{index}", 6.0, 10.0, dedup="dk_one") for index in range(5)]
    labels = LabelSet(
        observation_cutoff=date(2024, 12, 31),
        labels=tuple(label(item.finding_id, False) for item in findings),
    )
    simulator = LongitudinalSimulator()
    simulator.run(
        findings, labels, {RankerName.CVSS_ONLY: [item.finding_id for item in findings]}, config
    )
    assert simulator.capacity is not None
    assert simulator.capacity.n_findings == 5
    assert simulator.capacity.n_clusters == 1
    assert simulator.capacity.backlog_hours == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Prevention as a first-class reported outcome
# ---------------------------------------------------------------------------


def test_prevention_rate_is_available_beside_every_policy(world, config) -> None:
    findings, labels = world
    results = LongitudinalSimulator().run(findings, labels, orders(findings), config)
    by_policy = {item.policy: item for item in results}
    assert prevention_rate(by_policy[RankerName.LAMBDAMART]) == pytest.approx(1.0)
    assert prevention_rate(by_policy[RankerName.CVSS_ONLY]) == pytest.approx(0.0)


def test_prevention_rate_is_zero_rather_than_one_when_nothing_was_exploited(config) -> None:
    findings = [enriched("f_a", 2.0, 10.0)]
    labels = LabelSet(observation_cutoff=date(2024, 12, 31), labels=(label("f_a", False),))
    result = LongitudinalSimulator().run(findings, labels, {RankerName.CVSS_ONLY: ["f_a"]}, config)[0]
    assert result.exploited_total == 0
    assert prevention_rate(result) == 0.0


def test_the_summary_carries_the_outcome_numbers_with_their_capacity(world, config) -> None:
    """The three outcome numbers must never travel without the budget that produced them."""
    findings, labels = world
    simulator = LongitudinalSimulator()
    simulator.run(findings, labels, orders(findings), config)
    summary = simulator.summary()

    assert summary["reduction_metric"] == "exposure_days_exploited"
    assert "capacity" in summary["reduction_metric_note"]
    assert summary["reference_policy"] == RankerName.CVSS_ONLY.value
    assert summary["capacity"]["total_hours_available"] == pytest.approx(80.0)
    assert summary["best_by_prevention"] == RankerName.LAMBDAMART.value

    rows = {row["policy"]: row for row in summary["policies"]}
    assert rows[RankerName.LAMBDAMART.value]["prevented"] == 2
    assert rows[RankerName.LAMBDAMART.value]["prevention_rate"] == pytest.approx(1.0)
    assert rows[RankerName.LAMBDAMART.value]["reduction_vs_reference"] == pytest.approx(0.5)
    assert rows[RankerName.CVSS_ONLY.value]["exposure_days_total"] == pytest.approx(126.0)


def test_summarise_without_a_capacity_context_still_works(world, config) -> None:
    findings, labels = world
    results = LongitudinalSimulator().run(findings, labels, orders(findings), config)
    summary = summarise(results)
    assert summary["capacity"] is None
    assert len(summary["policies"]) == 2


def test_no_reduction_is_claimed_when_nothing_is_known_to_be_exploited(config) -> None:
    """No exploited findings means the headline quantity is undefined, not zero."""
    findings = [enriched("f_a", 2.0, 10.0), enriched("f_b", 2.0, 10.0)]
    labels = LabelSet(
        observation_cutoff=date(2024, 12, 31),
        labels=(label("f_a", False), label("f_b", False)),
    )
    results = LongitudinalSimulator().run(
        findings,
        labels,
        {RankerName.LAMBDAMART: ["f_a", "f_b"], RankerName.CVSS_ONLY: ["f_b", "f_a"]},
        config,
    )
    assert all(item.reduction_vs_cvss is None for item in results)


def test_no_reduction_is_claimed_when_the_reference_policy_was_not_simulated(world, config) -> None:
    findings, labels = world
    results = LongitudinalSimulator().run(
        findings, labels, {RankerName.LAMBDAMART: orders(findings)[RankerName.LAMBDAMART]}, config
    )
    assert results[0].reduction_vs_cvss is None


def test_no_findings_produces_no_results(config) -> None:
    labels = LabelSet(observation_cutoff=date(2024, 12, 31), labels=())
    assert LongitudinalSimulator().run([], labels, {RankerName.CVSS_ONLY: []}, config) == []


def test_zero_capacity_is_rejected(world) -> None:
    findings, labels = world
    with pytest.raises(Exception):
        bad = PipelineConfig(simulation=SimulationConfig(capacity_hours_per_week=0.0))
        LongitudinalSimulator().run(findings, labels, {RankerName.CVSS_ONLY: []}, bad)
