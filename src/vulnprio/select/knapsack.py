"""Resource-constrained selection under a remediation budget (DESIGN.md 3.7, Gap 10).

A ranking is not a remediation plan. The team has a number of engineer-hours this sprint,
fixes cost wildly different amounts, and several "findings" are one root cause showing up
on twenty endpoints. Gap 10 is that the literature evaluates the ranking and stops; this
module evaluates the *decision*, which is the 0/1 knapsack

    maximise  sum of chain_adjusted_loss over the chosen set
    subject to  sum of remediation hours  <=  budget_hours

with two properties the cited work had to assume away:

**Cost is unequal.** ``RemediationCost.hours`` comes from
:mod:`vulnprio.decision.remediation_cost` and varies by the class of change the CWE
implies, from a configuration flip to redesigning a trust boundary.

**Cost is charged once per root cause.** Findings sharing a ``dedup_key`` are the same
defect seen through different endpoints. Fixing it fixes all of them, so the knapsack item
is the *cluster*: its value is the sum of its members' values, its weight is charged once,
and selecting it selects every member. Charging per alert instead would make an
endpoint-heavy root cause look unaffordable and is precisely the modelling error that
makes budget-aware comparison meaningless.

Three methods, all reported honestly in ``SelectionResult.method``:

``dp_exact``
    Exact 0/1 dynamic programming on a half-hour grid
    (``SelectionConfig.hour_granularity``). Hours are rounded *up* onto the grid so the
    plan is never cheaper than reality, and the optimum is exact for the discretised
    instance. Above ``max_items_for_exact`` clusters it hands over to the greedy method
    and says so.

``greedy_ratio``
    Take clusters in descending value per hour, skipping any that no longer fit. The
    classical knapsack heuristic, and the fallback at scale.

``rank_prefix``
    The control condition: walk the ranking in order and stop at the first item that does
    not fit. This is what a team actually does with a ranked queue, and comparing it
    against ``dp_exact`` is what measures the cost of ignoring effort.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from vulnprio.core.config import PipelineConfig, SelectionConfig
from vulnprio.core.enums import RankerName, SelectionMethod
from vulnprio.core.errors import ConfigError
from vulnprio.core.models import (
    ChainScore,
    EnrichedFinding,
    LabelSet,
    RankedFinding,
    SelectionResult,
)
from vulnprio.decision.expected_loss import chain_adjusted_loss

__all__ = [
    "MIN_HOURS",
    "SelectionItem",
    "Cluster",
    "cluster_items",
    "items_from_enriched",
    "items_from_ranking",
    "select_under_budget",
]

#: Floor on a cluster's charged hours. A fix that costs nothing is not a fix, and a zero
#: weight would let the knapsack take an unbounded number of free items.
MIN_HOURS = 0.25


@dataclass(frozen=True, slots=True)
class SelectionItem:
    """One finding as the selection layer sees it.

    ``value`` is ``chain_adjusted_loss`` (expected loss plus what the finding
    unlocks for the rest of the attack graph), ``hours`` is its remediation cost, and
    ``dedup_key`` names the root cause whose cost is charged once.
    """

    finding_id: str
    value: float
    hours: float
    dedup_key: str | None = None
    rank: int | None = None
    exploited: bool = False

    @property
    def cluster_key(self) -> str:
        """Root cause this finding belongs to; its own id when it stands alone."""
        return self.dedup_key or self.finding_id


@dataclass(frozen=True, slots=True)
class Cluster:
    """A root cause: the unit the knapsack actually chooses between."""

    key: str
    finding_ids: tuple[str, ...]
    value: float
    hours: float
    best_rank: int
    exploited: int

    def units(self, granularity: float) -> int:
        """Weight on the discrete grid, rounded up so a plan is never understated."""
        return max(1, int(math.ceil(self.hours / granularity - 1e-9)))


def cluster_items(items: Iterable[SelectionItem]) -> list[Cluster]:
    """Group findings by root cause, summing value and charging hours once.

    Hours for a cluster are the maximum over its members rather than the sum: members of a
    cluster are the same defect, and :mod:`vulnprio.decision.remediation_cost` already
    grows a single member's hours with ``cluster_size`` to cover rolling the fix out. The
    maximum is defensive - it keeps the charge correct if a caller supplies members priced
    inconsistently.

    Clusters come back in a deterministic order: best (lowest) rank first, then descending
    value, then key, so every method sees the same ordering and ties never depend on dict
    iteration order.
    """
    grouped: dict[str, list[SelectionItem]] = {}
    for item in items:
        grouped.setdefault(item.cluster_key, []).append(item)

    clusters: list[Cluster] = []
    for key, members in grouped.items():
        ranks = [member.rank for member in members if member.rank is not None]
        clusters.append(
            Cluster(
                key=key,
                finding_ids=tuple(member.finding_id for member in members),
                value=max(0.0, sum(max(0.0, float(member.value)) for member in members)),
                hours=max(MIN_HOURS, max(float(member.hours) for member in members)),
                best_rank=min(ranks) if ranks else len(grouped) + 1,
                exploited=sum(1 for member in members if member.exploited),
            )
        )
    clusters.sort(key=lambda cluster: (cluster.best_rank, -cluster.value, cluster.key))
    return clusters


def items_from_enriched(
    enriched: Sequence[EnrichedFinding],
    chain: Mapping[str, ChainScore] | None = None,
    *,
    chain_weight: float = 1.0,
    labels: LabelSet | None = None,
    order: Sequence[str] | None = None,
) -> list[SelectionItem]:
    """Build selection items straight from Component B and Component C output.

    Value is ``chain_adjusted_loss(expected_loss, reach_delta, chain_weight)`` -
    the construct DESIGN.md 2 defines - so the selection layer optimises the same quantity
    the ranker is trained to order. ``order`` supplies the ranking (used by
    ``rank_prefix``); without it, items carry no rank.
    """
    positives = labels.positives() if labels is not None else set()
    ranks = {finding_id: index + 1 for index, finding_id in enumerate(order or ())}
    out: list[SelectionItem] = []
    for item in enriched:
        score = (chain or {}).get(item.finding_id)
        reach = float(score.reach_delta) if score is not None else 0.0
        out.append(
            SelectionItem(
                finding_id=item.finding_id,
                value=chain_adjusted_loss(item.expected_loss, reach, chain_weight),
                hours=max(MIN_HOURS, float(item.remediation.hours)),
                dedup_key=item.finding.dedup_key,
                rank=ranks.get(item.finding_id),
                exploited=item.finding_id in positives,
            )
        )
    return out


def items_from_ranking(
    ranked: Sequence[RankedFinding],
    enriched: Sequence[EnrichedFinding],
    *,
    labels: LabelSet | None = None,
) -> list[SelectionItem]:
    """Build selection items from a produced ranking plus the enrichment behind it.

    ``RankedFinding`` already carries ``chain_adjusted_loss`` and ``rank``; the
    enriched findings supply the remediation hours and the dedup key. Findings absent from
    the ranking are absent from the plan.
    """
    by_id = {item.finding_id: item for item in enriched}
    positives = labels.positives() if labels is not None else set()
    out: list[SelectionItem] = []
    for entry in ranked:
        source = by_id.get(entry.finding_id)
        if source is None:
            continue
        out.append(
            SelectionItem(
                finding_id=entry.finding_id,
                value=max(0.0, float(entry.chain_adjusted_loss)),
                hours=max(MIN_HOURS, float(source.remediation.hours)),
                dedup_key=source.finding.dedup_key,
                rank=int(entry.rank),
                exploited=entry.finding_id in positives,
            )
        )
    return out


def _selection_config(config: PipelineConfig | SelectionConfig | None) -> SelectionConfig:
    if config is None:
        return SelectionConfig()
    if isinstance(config, SelectionConfig):
        return config
    return config.selection


def _dp_exact(clusters: Sequence[Cluster], capacity: int, granularity: float) -> list[Cluster]:
    """Exact 0/1 knapsack by dynamic programming over the discretised hour grid.

    ``best[c]`` is the greatest value achievable in exactly ``c`` grid units or fewer, and
    ``taken[i][c]`` records whether cluster ``i`` is in that optimum, which is what lets
    the chosen set be reconstructed rather than only its value. The table is
    ``len(clusters) x capacity``; with a 40-hour budget on a half-hour grid that is 80
    columns, so exactness costs nothing at realistic sizes.
    """
    if capacity <= 0:
        return []
    best = [0.0] * (capacity + 1)
    taken: list[list[bool]] = []
    for cluster in clusters:
        weight = cluster.units(granularity)
        row = [False] * (capacity + 1)
        if weight <= capacity and cluster.value > 0.0:
            for column in range(capacity, weight - 1, -1):
                candidate = best[column - weight] + cluster.value
                if candidate > best[column]:
                    best[column] = candidate
                    row[column] = True
        taken.append(row)

    chosen: list[Cluster] = []
    column = capacity
    for index in range(len(clusters) - 1, -1, -1):
        if taken[index][column]:
            cluster = clusters[index]
            chosen.append(cluster)
            column -= cluster.units(granularity)
    chosen.reverse()
    return chosen


def _greedy_ratio(clusters: Sequence[Cluster], budget_hours: float) -> list[Cluster]:
    """Descending value per hour, skipping clusters that no longer fit."""
    ordered = sorted(
        clusters,
        key=lambda cluster: (-(cluster.value / cluster.hours), -cluster.value, cluster.key),
    )
    chosen: list[Cluster] = []
    spent = 0.0
    for cluster in ordered:
        if cluster.value <= 0.0:
            continue
        if spent + cluster.hours > budget_hours + 1e-9:
            continue
        chosen.append(cluster)
        spent += cluster.hours
    return chosen


def _rank_prefix(clusters: Sequence[Cluster], budget_hours: float) -> list[Cluster]:
    """The control: walk the ranking in order and stop when the budget runs out.

    Deliberately stops rather than skipping ahead. A team working a ranked queue does not
    hunt down the list for something that fits, and the gap between this and ``dp_exact``
    is the measurable cost of a ranking that ignores effort.
    """
    ordered = sorted(clusters, key=lambda cluster: (cluster.best_rank, -cluster.value, cluster.key))
    chosen: list[Cluster] = []
    spent = 0.0
    for cluster in ordered:
        if spent + cluster.hours > budget_hours + 1e-9:
            break
        chosen.append(cluster)
        spent += cluster.hours
    return chosen


def select_under_budget(
    items: Sequence[SelectionItem],
    *,
    budget_hours: float | None = None,
    config: PipelineConfig | SelectionConfig | None = None,
    scan_id: str = "",
    ranker: RankerName = RankerName.LAMBDAMART,
    method: SelectionMethod | None = None,
    hour_granularity: float | None = None,
    max_items_for_exact: int | None = None,
    labels: LabelSet | None = None,
) -> SelectionResult:
    """Choose what to remediate this budget, and report what that captures.

    ``items`` are per finding; they are clustered by ``dedup_key`` first, so the budget is
    spent on root causes. ``budget_hours``, ``method``, ``hour_granularity`` and
    ``max_items_for_exact`` default to ``config`` (or to :class:`SelectionConfig`'s own
    defaults). ``labels`` overrides each item's ``exploited`` flag, which is how the
    evaluation layer measures efficiency and coverage against confirmed exploitation.

    ``SelectionResult.method`` is the method that actually ran: asking for ``dp_exact``
    with more than ``max_items_for_exact`` clusters returns a greedy result labelled
    ``greedy_ratio``, never an exact label on a heuristic answer.
    """
    settings = _selection_config(config)
    budget = float(settings.budget_hours if budget_hours is None else budget_hours)
    granularity = float(settings.hour_granularity if hour_granularity is None else hour_granularity)
    requested = settings.method if method is None else method
    exact_limit = int(settings.max_items_for_exact if max_items_for_exact is None else max_items_for_exact)

    if budget <= 0.0:
        raise ConfigError(f"selection budget must be positive, got {budget!r} hours")
    if granularity <= 0.0:
        raise ConfigError(f"hour granularity must be positive, got {granularity!r}")

    resolved = _apply_labels(items, labels)
    clusters = cluster_items(resolved)
    total_value = sum(cluster.value for cluster in clusters)
    exploited_total = sum(1 for item in resolved if item.exploited)

    used = requested
    if requested == SelectionMethod.DP_EXACT and len(clusters) > exact_limit:
        used = SelectionMethod.GREEDY_RATIO

    if used == SelectionMethod.DP_EXACT:
        capacity = int(math.floor(budget / granularity + 1e-9))
        chosen = _dp_exact(clusters, capacity, granularity)
    elif used == SelectionMethod.GREEDY_RATIO:
        chosen = _greedy_ratio(clusters, budget)
    elif used == SelectionMethod.RANK_PREFIX:
        chosen = _rank_prefix(clusters, budget)
    else:  # pragma: no cover - SelectionMethod is exhaustive
        raise ConfigError(f"unknown selection method: {used!r}")

    selected_ids = tuple(
        finding_id for cluster in chosen for finding_id in sorted(cluster.finding_ids)
    )
    captured = sum(cluster.value for cluster in chosen)
    hours = sum(cluster.hours for cluster in chosen)
    fraction = 0.0 if total_value <= 0.0 else min(1.0, max(0.0, captured / total_value))

    return SelectionResult(
        scan_id=scan_id,
        ranker=ranker,
        method=used,
        budget_hours=budget,
        selected_ids=selected_ids,
        total_hours=max(0.0, hours),
        risk_captured=max(0.0, captured),
        risk_capture_fraction=fraction,
        exploited_captured=sum(cluster.exploited for cluster in chosen),
        exploited_total=exploited_total,
    )


def _apply_labels(items: Sequence[SelectionItem], labels: LabelSet | None) -> list[SelectionItem]:
    """Stamp ground-truth exploitation onto the items when a label set is supplied."""
    if labels is None:
        return list(items)
    positives = labels.positives()
    return [
        item if item.exploited == (item.finding_id in positives)
        else SelectionItem(
            finding_id=item.finding_id,
            value=item.value,
            hours=item.hours,
            dedup_key=item.dedup_key,
            rank=item.rank,
            exploited=item.finding_id in positives,
        )
        for item in items
    ]
