"""Longitudinal deployment simulation (DESIGN.md 3.9, Gap 10).

Gap 10 is that evaluation in this field stops at prediction. A ranking with excellent
NDCG is not the deliverable; a team with twenty hours a week and four hundred findings
is. This module runs that team.

Each week the simulator spends ``capacity_hours_per_week`` following one policy's
ordering, and measures what the literature never measures:

* **exposure days** - the sum over findings of the days each one stayed unremediated.
  This is the quantity a security programme actually wants to minimise, and it punishes a
  policy for what it *defers* rather than only rewarding what it picks;
* **expected-loss days** - the same sum weighted by each finding's expected loss, so
  leaving a high-value finding open for a week costs more than leaving a trivial one
  open for a month;
* **exploited-before-evidence** - how many confirmed-exploited findings were closed
  *before* their first exploitation evidence date. This is the counterfactual the
  synthetic oracle exists to provide: it is the number that says the ordering would have
  prevented something.

Which of those is the headline, and why it is not the first one
---------------------------------------------------------------

``exposure_days_total`` is a **capacity measurement, not a prioritisation measurement**,
and the report says so in those words. On a realistic backlog almost nothing is reachable:
26 weeks at 20 hours is 520 engineer-hours against a backlog an order of magnitude larger,
so several hundred findings accrue the full horizon under *every* ordering. Their exposure
is fixed by the budget before any policy has a say, and it dominates the sum. Measured on
the synthetic world, every policy lands within 4% of every other on the total - including
a learned ranker that is decisively better on everything that can actually be moved. A
metric that is near-constant by construction cannot be a headline, and a *reduction*
computed on it reports noise.

So ``SimulationResult.reduction_vs_cvss`` is the reduction in
**``exposure_days_exploited``** against the reference policy: the days of exposure carried
by the findings that were really exploited. That is precisely the quantity an ordering
controls - the whole claim of prioritisation is that the dangerous work is reached sooner,
not that more work is done - and it is what Gap 10 asks about. ``exposure_days_exploited``
is preferred over ``expected_loss_days`` for the headline because it is ground truth
rather than the framework's own estimate: expected loss is a product of the model's
``p_exploit`` and its impact model, so reducing it can be achieved by believing one's own
estimates harder, whereas an exploited finding was exploited whatever the model thought.
``expected_loss_days`` is still reported, as the decision-theoretic complement.

The clearest single sentence the simulation can produce is the prevention count:
``exploited_remediated_before_exploit`` of ``exploited_total`` confirmed-exploited findings
closed before their first exploitation evidence date. It is reported as a rate beside every
policy, and always next to the capacity context (:class:`CapacityContext`) - "eleven of 145"
is only interpretable once a reader knows the budget could ever reach a few percent of the
backlog.

Modelling choices, all deliberate and all visible:

* Work carries over between weeks, so a finding costing more than a week's capacity is
  not skipped forever.
* Remediation completes at the **end** of the week in which its last hour is spent; a
  finding closed in week ``w`` (zero-based) accrues ``(w + 1) * 7`` exposure days.
* Fixing a finding fixes its whole root-cause cluster (``Finding.dedup_key``) at no extra
  cost, because that is what fixing the root cause does. The framework ranks root causes,
  not alerts, and charging the cluster twice would flatter whichever policy happened to
  scatter a cluster across its ordering.
* A finding never reached inside the horizon accrues the full ``weeks * 7`` days; it is
  not dropped, because "never got to it" is the outcome being measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

from vulnprio.core.config import PipelineConfig
from vulnprio.core.enums import RankerName
from vulnprio.core.models import (
    EnrichedFinding,
    LabelSet,
    RankingResult,
    SimulationResult,
)

__all__ = [
    "FindingState",
    "PolicyTrace",
    "CapacityContext",
    "LongitudinalSimulator",
    "prevention_rate",
    "summarise",
    "DAYS_PER_WEEK",
]

DAYS_PER_WEEK = 7


@dataclass(frozen=True)
class FindingState:
    """The per-finding facts the simulation needs, extracted once for every policy."""

    finding_id: str
    hours: float
    expected_loss: float
    cluster: str
    discovered_on: date
    exploited: bool = False
    first_evidence_date: date | None = None


@dataclass(frozen=True)
class CapacityContext:
    """What the remediation budget could ever have reached.

    Every exposure and prevention number in the report is meaningless without this.
    "Eleven of 145 confirmed-exploited findings prevented" reads as a poor result until
    the reader sees that 520 engineer-hours cover a few percent of the backlog - at which
    point the same number reads as the ordering having spent a tiny budget almost entirely
    on the things that mattered.

    ``backlog_hours`` charges each root-cause cluster once, matching how the simulator
    spends. Members of a cluster share a remediation estimate (``estimate_cost`` is a
    function of CWE and cluster size, not of the individual finding), so the mean over a
    cluster is exact in practice and the field is not sensitive to which member a policy
    happens to reach first.
    """

    weeks: int
    capacity_hours_per_week: float
    n_findings: int
    n_clusters: int
    n_exploited: int
    backlog_hours: float

    @property
    def total_hours_available(self) -> float:
        return float(self.weeks * self.capacity_hours_per_week)

    @property
    def reachable_fraction(self) -> float:
        """Share of the backlog the budget could cover even under a perfect ordering."""
        if self.backlog_hours <= 0.0:
            return 1.0
        return float(min(1.0, self.total_hours_available / self.backlog_hours))

    @property
    def capacity_bound(self) -> bool:
        """True when the budget cannot clear the backlog, which is the normal case.

        When this holds, ``exposure_days_total`` is bounded below by the capacity and is
        expected to be nearly identical across policies: the unreachable remainder accrues
        the full horizon whatever the ordering.
        """
        return self.reachable_fraction < 1.0

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "weeks": self.weeks,
            "capacity_hours_per_week": self.capacity_hours_per_week,
            "total_hours_available": self.total_hours_available,
            "backlog_hours": self.backlog_hours,
            "reachable_fraction": self.reachable_fraction,
            "capacity_bound": self.capacity_bound,
            "n_findings": self.n_findings,
            "n_clusters": self.n_clusters,
            "n_exploited": self.n_exploited,
        }


@dataclass
class PolicyTrace:
    """Per-finding outcome of one policy run, kept so the report can explain a result."""

    policy: RankerName
    remediated_day: dict[str, int] = field(default_factory=dict)
    exposure_days: dict[str, float] = field(default_factory=dict)
    weekly_cumulative: tuple[float, ...] = ()

    def never_remediated(self) -> list[str]:
        return [key for key, value in self.remediated_day.items() if value < 0]


@dataclass
class LongitudinalSimulator:
    """Runs one week-by-week remediation simulation per policy.

    ``reference_policy`` (``SimulationConfig.reference_policy``, CVSS-only by default) is
    the comparison point for ``SimulationResult.reduction_vs_cvss``: the fractional
    reduction in **exposure days on the confirmed-exploited findings** against the policy
    most teams actually run today. See the module docstring for why the reduction is not
    computed on total exposure - in short, total exposure is set by the budget rather than
    by the ordering, so a reduction computed on it reports noise.

    After :meth:`run`, ``capacity`` holds the :class:`CapacityContext` the run was
    executed under, and ``traces`` holds the per-finding remediation days per policy.
    """

    traces: dict[RankerName, PolicyTrace] = field(default_factory=dict)
    capacity: CapacityContext | None = None
    reference_policy: RankerName | None = None
    results: list[SimulationResult] = field(default_factory=list)

    def run(
        self,
        enriched: Sequence[EnrichedFinding],
        labels: LabelSet,
        ranking_by_policy: Mapping[Any, Sequence[str] | RankingResult],
        config: PipelineConfig,
    ) -> list[SimulationResult]:
        """One :class:`SimulationResult` per policy, in the order the policies were given."""
        weeks = max(1, int(config.simulation.weeks))
        capacity = float(config.simulation.capacity_hours_per_week)
        if capacity <= 0.0:
            raise ValueError("capacity_hours_per_week must be positive")

        states = _states(enriched, labels)
        if not states:
            self.traces = {}
            self.capacity = None
            self.results = []
            return []
        start = min(state.discovered_on for state in states.values())
        self.traces = {}
        self.capacity = _capacity_context(states, weeks, capacity)
        self.reference_policy = config.simulation.reference_policy

        results: list[SimulationResult] = []
        traces: dict[RankerName, PolicyTrace] = {}
        for raw_policy, ranking in ranking_by_policy.items():
            policy = RankerName(raw_policy)
            order = _order_of(ranking, states)
            traces[policy] = self._simulate(policy, order, states, weeks, capacity)

        reference = _reference_exploited_exposure(
            traces, config.simulation.reference_policy, states
        )
        for policy, trace in traces.items():
            results.append(_result(policy, trace, states, weeks, capacity, start, reference))
        self.traces = traces
        self.results = results
        return results

    def summary(self) -> dict[str, Any]:
        """The last run's outcome numbers with their capacity context attached."""
        return summarise(self.results, self.capacity, self.reference_policy)

    # -- the week loop -----------------------------------------------------

    def _simulate(
        self,
        policy: RankerName,
        order: Sequence[str],
        states: Mapping[str, FindingState],
        weeks: int,
        capacity: float,
    ) -> PolicyTrace:
        horizon_days = weeks * DAYS_PER_WEEK
        remaining = {key: max(0.0, state.hours) for key, state in states.items()}
        remediated_day: dict[str, int] = {key: -1 for key in states}
        open_ids = set(states)

        for week in range(weeks):
            budget = capacity
            end_of_week = (week + 1) * DAYS_PER_WEEK
            for finding_id in order:
                if budget <= 1e-9:
                    break
                if finding_id not in open_ids:
                    continue
                spend = min(budget, remaining[finding_id])
                remaining[finding_id] -= spend
                budget -= spend
                if remaining[finding_id] > 1e-9:
                    continue
                # Closing a finding closes its whole root-cause cluster.
                cluster = states[finding_id].cluster
                for sibling in [key for key in open_ids if states[key].cluster == cluster]:
                    remediated_day[sibling] = end_of_week
                    open_ids.discard(sibling)
                    remaining[sibling] = 0.0

        exposure = {
            key: float(remediated_day[key] if remediated_day[key] >= 0 else horizon_days)
            for key in states
        }
        weekly = tuple(
            float(sum(min(exposure[key], (week + 1) * DAYS_PER_WEEK) for key in states))
            for week in range(weeks)
        )
        return PolicyTrace(
            policy=policy,
            remediated_day=remediated_day,
            exposure_days=exposure,
            weekly_cumulative=weekly,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _states(enriched: Sequence[EnrichedFinding], labels: LabelSet) -> dict[str, FindingState]:
    """Collapse the enriched findings and the label set into the simulation's own view."""
    states: dict[str, FindingState] = {}
    for item in enriched:
        label = labels.by_id(item.finding_id)
        finding = item.finding
        states[item.finding_id] = FindingState(
            finding_id=item.finding_id,
            hours=float(item.remediation.hours),
            expected_loss=float(item.expected_loss),
            cluster=finding.dedup_key or item.finding_id,
            discovered_on=finding.observed_at.date(),
            exploited=bool(label.exploited) if label else False,
            first_evidence_date=label.first_evidence_date if label else None,
        )
    return states


def _order_of(
    ranking: Sequence[str] | RankingResult, states: Mapping[str, FindingState]
) -> list[str]:
    """A policy's ordering as a list of finding ids, restricted to the simulated set.

    Findings a policy does not mention are appended in a stable id order: every policy
    must eventually address every finding, or the exposure comparison would reward a
    policy for simply declining to rank the hard ones.
    """
    if isinstance(ranking, RankingResult):
        ids = [item.finding_id for item in sorted(ranking.items, key=lambda row: row.rank)]
    else:
        ids = list(ranking)
    seen: set[str] = set()
    order = []
    for finding_id in ids:
        if finding_id in states and finding_id not in seen:
            order.append(finding_id)
            seen.add(finding_id)
    order.extend(sorted(key for key in states if key not in seen))
    return order


def _capacity_context(
    states: Mapping[str, FindingState], weeks: int, capacity: float
) -> CapacityContext:
    """Backlog and budget, charging each root-cause cluster once (as the simulator does)."""
    by_cluster: dict[str, list[float]] = {}
    for state in states.values():
        by_cluster.setdefault(state.cluster, []).append(max(0.0, state.hours))
    backlog = sum(sum(hours) / len(hours) for hours in by_cluster.values())
    return CapacityContext(
        weeks=weeks,
        capacity_hours_per_week=capacity,
        n_findings=len(states),
        n_clusters=len(by_cluster),
        n_exploited=sum(1 for state in states.values() if state.exploited),
        backlog_hours=float(backlog),
    )


def _reference_exploited_exposure(
    traces: Mapping[RankerName, PolicyTrace],
    reference: RankerName,
    states: Mapping[str, FindingState],
) -> float | None:
    """Exposure days on the confirmed-exploited findings under the reference policy.

    The denominator of ``reduction_vs_cvss``. ``None`` when the reference policy was not
    simulated or when nothing is known to have been exploited, in which case no reduction
    is claimed rather than a zero being invented.
    """
    trace = traces.get(reference)
    if trace is None:
        return None
    exploited = [key for key, state in states.items() if state.exploited]
    if not exploited:
        return None
    return float(sum(trace.exposure_days[key] for key in exploited))


def _result(
    policy: RankerName,
    trace: PolicyTrace,
    states: Mapping[str, FindingState],
    weeks: int,
    capacity: float,
    start: date,
    reference_exploited_exposure: float | None,
) -> SimulationResult:
    exposure_total = float(sum(trace.exposure_days.values()))
    exploited = [state for state in states.values() if state.exploited]
    exposure_exploited = float(
        sum(trace.exposure_days[state.finding_id] for state in exploited)
    )
    loss_days = float(
        sum(
            max(0.0, states[key].expected_loss) * days
            for key, days in trace.exposure_days.items()
        )
    )

    prevented = 0
    for state in exploited:
        if state.first_evidence_date is None:
            # No recorded evidence date: the counterfactual cannot be judged, so this
            # finding is counted in the denominator and never in the numerator.
            continue
        day = trace.remediated_day.get(state.finding_id, -1)
        if day < 0:
            continue
        if start + timedelta(days=day) < state.first_evidence_date:
            prevented += 1

    # The field keeps its contract name; its meaning is the reduction in exposure days on
    # the confirmed-exploited findings, which is the part of the outcome an ordering can
    # actually move. Total exposure is bounded below by the remediation capacity and is
    # near-identical across policies, so a reduction computed on it would report noise.
    reduction: float | None = None
    if reference_exploited_exposure is not None and reference_exploited_exposure > 0.0:
        reduction = float(
            (reference_exploited_exposure - exposure_exploited) / reference_exploited_exposure
        )

    return SimulationResult(
        policy=policy,
        weeks=weeks,
        capacity_hours_per_week=capacity,
        exposure_days_total=exposure_total,
        exposure_days_exploited=exposure_exploited,
        expected_loss_days=loss_days,
        exploited_remediated_before_exploit=prevented,
        exploited_total=len(exploited),
        weekly_cumulative_exposure=trace.weekly_cumulative,
        reduction_vs_cvss=reduction,
    )


# ---------------------------------------------------------------------------
# Reporting the outcome
# ---------------------------------------------------------------------------


def prevention_rate(result: SimulationResult) -> float:
    """Share of confirmed-exploited findings closed before their first evidence date.

    The simulation's clearest single statement, and the one the report leads with. Returns
    ``0.0`` when nothing is known to have been exploited (the rate is undefined and
    claiming 1.0 for "prevented everything we knew about" would be a lie by construction).
    """
    if result.exploited_total <= 0:
        return 0.0
    return float(result.exploited_remediated_before_exploit / result.exploited_total)


def summarise(
    results: Sequence[SimulationResult],
    capacity: CapacityContext | None = None,
    reference: RankerName | None = None,
) -> dict[str, Any]:
    """Machine-readable summary of a simulation run, with its capacity context attached.

    Written for the report and the CLI so that the three outcome numbers - prevention,
    exploited exposure and its reduction against the reference - travel together with the
    budget that produced them, and so that ``exposure_days_total`` is never presented
    without the label that says it is a capacity measurement.
    """
    rows = [
        {
            "policy": item.policy.value,
            "exposure_days_total": item.exposure_days_total,
            "exposure_days_exploited": item.exposure_days_exploited,
            "expected_loss_days": item.expected_loss_days,
            "prevented": item.exploited_remediated_before_exploit,
            "exploited_total": item.exploited_total,
            "prevention_rate": prevention_rate(item),
            "reduction_vs_reference": item.reduction_vs_cvss,
        }
        for item in results
    ]
    best = max(results, key=prevention_rate, default=None)
    return {
        "reduction_metric": "exposure_days_exploited",
        "reduction_metric_note": (
            "Reduction is measured on exposure days carried by the confirmed-exploited "
            "findings, which is the part of the outcome a remediation ordering controls. "
            "Total exposure is bounded below by remediation capacity and is expected to be "
            "near-identical across policies."
        ),
        "reference_policy": reference.value if reference is not None else None,
        "capacity": capacity.as_dict() if capacity is not None else None,
        "policies": rows,
        "best_by_prevention": best.policy.value if best is not None else None,
    }
