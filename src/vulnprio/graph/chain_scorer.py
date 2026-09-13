"""``ReachabilityChainScorer``: Component C's per-finding contribution (DESIGN.md 3.7).

The scorer owns one built graph per scan and answers, for every finding:

    reach_delta(f) = R(G) - R(G \\ edges(f))

in money. This is the number Gap 9 asks for. It is a *counterfactual*: not "how bad is
this finding" but "how much reachable value disappears when it is fixed", which is the
only formulation under which a finding with negligible direct impact can outrank a
critical one - and it does, whenever the low-severity finding is the gate everything else
passes through.

**Why the differencing is cheap.** The state graph has one node per (host, privilege)
pair, so it is tiny even when the scan has hundreds of findings. The expensive object is
the *graph*, and it is built once: patching is expressed as a weight callable that
re-maximises each edge over its surviving contributors, so no graph is ever rebuilt,
copied or mutated. Scoring a scan costs one Dijkstra for the base risk, one per finding
for its counterfactual, one reversed Dijkstra per target, one betweenness pass and one
bounded path enumeration. Base risk, reverse distances, betweenness and paths are all
computed once and reused across every finding.

**Why ``reach_delta`` cannot be negative.** See :mod:`vulnprio.graph.reachability`:
removing edges can only lengthen ``-log p`` paths, node values are fixed at build time, so
``R`` can only fall. ``ChainScore.reach_delta`` is a non-negative field on the frozen
contract and :mod:`vulnprio.graph.monotone` verifies the property at runtime when
``ComponentCConfig.verify_monotonicity`` is set.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import networkx as nx

from vulnprio.core.config import ComponentCConfig, PipelineConfig
from vulnprio.core.errors import GraphError
from vulnprio.core.interfaces import ChainScorer
from vulnprio.core.models import (
    AttackGraphSummary,
    AttackerModel,
    AttackPath,
    ChainScore,
    EnrichedFinding,
    Scan,
)
from vulnprio.graph.attack_graph import (
    AttackGraphBuilder,
    FindingTransition,
    effective_probability,
)
from vulnprio.graph.reachability import (
    VALUE_ATTR,
    edge_betweenness_for_findings,
    enumerate_paths,
    hops_from_entry,
    max_path_probability_from,
    max_path_probability_to,
    probability_to_weight,
)

__all__ = [
    "PATH_COUNT_LIMIT",
    "RUNTIME_VERIFY_SUBSETS",
    "ScanGraph",
    "ReachabilityChainScorer",
    "chain_scores_for",
]

#: Simple paths enumerated per target when counting ``n_paths_through``. Generous, since
#: the state graph is small, but bounded so a pathological topology cannot hang a run.
PATH_COUNT_LIMIT = 512

#: Random patch subsets drawn when ``verify_monotonicity`` is on. Small enough to be free
#: on every build, large enough that a sign error in the differencing cannot survive it.
RUNTIME_VERIFY_SUBSETS = 16

_EMPTY: frozenset[str] = frozenset()


class ScanGraph:
    """One scan's built graph plus everything derived from it that is patch-invariant.

    Holding these together is what makes scoring linear rather than quadratic: the base
    reachability, the reversed distances to each target, the betweenness and the path
    enumeration do not depend on which finding is being differenced out, so they are
    computed exactly once.
    """

    def __init__(
        self,
        scan_id: str,
        graph: nx.DiGraph,
        summary: AttackGraphSummary,
        transitions: list[FindingTransition],
        max_hops: int,
    ) -> None:
        self.scan_id = scan_id
        self.graph = graph
        self.summary = summary
        self.transitions: dict[str, FindingTransition] = {t.finding_id: t for t in transitions}
        self.entry = summary.entry_node
        self.targets = tuple(summary.target_nodes)
        self.max_hops = max_hops

        self.base_reach: dict[str, float] = max_path_probability_from(graph, self.entry)
        self.base_risk: float = summary.total_risk
        self.to_target: dict[str, dict[str, float]] = {
            target: max_path_probability_to(graph, target) for target in self.targets
        }
        self.betweenness: dict[str, float] = edge_betweenness_for_findings(graph)
        self.hops: dict[str, int] = hops_from_entry(graph, self.entry)
        self.paths: tuple[AttackPath, ...] = self._enumerate()

    def _enumerate(self) -> tuple[AttackPath, ...]:
        found: list[AttackPath] = []
        for target in self.targets:
            found.extend(
                enumerate_paths(self.graph, self.entry, target, self.max_hops, limit=PATH_COUNT_LIMIT)
            )
        return tuple(found)

    def target_value(self, node: str) -> float:
        """Money realised in a state, zero for a state that holds none."""
        if node not in self.graph:
            return 0.0
        return max(0.0, float(self.graph.nodes[node].get(VALUE_ATTR, 0.0)))

    def risk_without(self, patched: frozenset[str]) -> float:
        """``R(G)`` with every edge contribution from ``patched`` removed.

        The graph is untouched: patching is a weight function. An edge whose surviving
        contributors are empty returns ``None`` and networkx treats it as absent.
        """
        if not patched:
            return self.base_risk
        if not self.targets:
            return 0.0

        def weight(u: str, v: str, data: Mapping[str, Any]) -> float | None:
            probability = effective_probability(dict(data), patched)
            if probability <= 0.0:
                return None
            return probability_to_weight(probability)

        reach = max_path_probability_from(self.graph, self.entry, weight=weight)
        total = 0.0
        for target in self.targets:
            value = self.target_value(target)
            if value <= 0.0:
                continue
            total += value * reach.get(target, 0.0)
        return max(0.0, total)


class ReachabilityChainScorer(ChainScorer):
    """Component C implementation of :class:`vulnprio.core.interfaces.ChainScorer`.

    One instance can hold several scans; ``build`` stores a graph per ``scan_id`` and
    ``score``/``total_risk_after_patching`` address it by that id.
    """

    def __init__(
        self,
        config: PipelineConfig | ComponentCConfig | None = None,
        attacker: AttackerModel | None = None,
    ) -> None:
        """``config`` may be the whole pipeline config or just its Component C section."""
        self.config: ComponentCConfig = (
            config
            if isinstance(config, ComponentCConfig)
            else (config.component_c if config is not None else ComponentCConfig())
        )
        self.attacker = attacker
        self.builder = AttackGraphBuilder(self.config, attacker)
        self._graphs: dict[str, ScanGraph] = {}

    # -- ChainScorer ---------------------------------------------------------

    def build(self, scan: Scan, enriched: list[EnrichedFinding]) -> AttackGraphSummary:
        """Build and store the graph for ``scan``, returning its summary.

        When ``ComponentCConfig.verify_monotonicity`` is set the property is checked here,
        on this scan's real graph, before any score is reported:
        ``AttackGraphSummary.monotone_verified`` is only ``True`` if that check ran and
        passed. A violation raises :class:`MonotonicityViolationError` rather than
        returning a number nobody should trust.
        """
        graph, summary = self.builder.build(scan, enriched)
        transitions = self.builder.transitions(scan, enriched)
        self._graphs[scan.scan_id] = ScanGraph(
            scan.scan_id, graph, summary, transitions, int(self.config.max_hops)
        )
        if self.config.verify_monotonicity:
            from vulnprio.graph.monotone import assert_monotone_under_patching

            assert_monotone_under_patching(self, scan.scan_id, subsets=RUNTIME_VERIFY_SUBSETS, rng=0)
            summary = summary.model_copy(update={"monotone_verified": True})
            self._graphs[scan.scan_id].summary = summary
        return summary

    def score(self, scan_id: str) -> dict[str, ChainScore]:
        """A :class:`ChainScore` per finding in a built scan.

        Findings that create no edge - information disclosure, or anything whose CWE
        confers no privilege the endpoint's authentication does not already demand - score
        zero on every chain field. That is the correct answer, not a gap: their priority
        is their expected loss, which is Component B's to report.
        """
        state = self._state(scan_id)
        finding_ids = tuple(state.transitions)
        if not finding_ids:
            return {}

        base = state.base_risk
        fraction = float(self.config.chokepoint_delta_fraction)
        path_counts = self._paths_through(state)

        scores: dict[str, ChainScore] = {}
        for finding_id in finding_ids:
            transition = state.transitions[finding_id]
            has_edge = transition.admitted and transition.escalates and state.graph.has_edge(
                transition.src, transition.dst
            )
            if has_edge:
                delta = max(0.0, base - state.risk_without(frozenset({finding_id})))
                probability, best_target = self._best_target(state, transition)
                hops = state.hops.get(transition.src)
                hops_from = 0 if hops is None else int(hops) + 1
            else:
                delta = 0.0
                probability, best_target = 0.0, None
                hops_from = 0
            scores[finding_id] = ChainScore(
                finding_id=finding_id,
                reach_delta=delta,
                max_path_prob_to_target=min(1.0, max(0.0, probability)),
                n_paths_through=path_counts.get(finding_id, 0),
                betweenness=max(0.0, state.betweenness.get(finding_id, 0.0)),
                hops_from_entry=hops_from,
                privilege_gain=transition.privilege_gain if has_edge else 0,
                is_chokepoint=bool(base > 0.0 and delta > 0.0 and delta >= fraction * base),
                best_target=best_target,
            )
        return scores

    def total_risk_after_patching(self, scan_id: str, patched_finding_ids: set[str]) -> float:
        """``R(G)`` once every listed finding has been remediated.

        The evaluation-facing form of the same counterfactual ``score`` uses per finding,
        and the quantity :mod:`vulnprio.graph.monotone` asserts is non-increasing in the
        patched set.
        """
        state = self._state(scan_id)
        return state.risk_without(frozenset(str(item) for item in patched_finding_ids))

    # -- accessors -----------------------------------------------------------

    def graph_for(self, scan_id: str) -> nx.DiGraph:
        """The built ``networkx.DiGraph`` for a scan."""
        return self._state(scan_id).graph

    def summary_for(self, scan_id: str) -> AttackGraphSummary:
        """The stored :class:`AttackGraphSummary` for a scan."""
        return self._state(scan_id).summary

    def base_risk(self, scan_id: str) -> float:
        """``R(G)`` with nothing patched."""
        return self._state(scan_id).base_risk

    def finding_ids(self, scan_id: str) -> tuple[str, ...]:
        """Every finding the scan's graph knows about, in build order."""
        return tuple(self._state(scan_id).transitions)

    def built_scans(self) -> tuple[str, ...]:
        """Scan ids with a stored graph."""
        return tuple(self._graphs)

    # -- internals -----------------------------------------------------------

    def _state(self, scan_id: str) -> ScanGraph:
        state = self._graphs.get(scan_id)
        if state is None:
            raise GraphError(f"no attack graph built for scan {scan_id!r}; call build() first")
        return state

    def _best_target(self, state: ScanGraph, transition: FindingTransition) -> tuple[float, str | None]:
        """Most probable ``entry -> this transition -> target`` route, and its target.

        Priced as ``P(entry -> required) x p(edge) x P(gained -> target)``: the attacker
        has to arrive at the state the exploit needs, run it, and carry on. All three
        factors come from cached single-source searches.
        """
        prefix = state.base_reach.get(transition.src, 0.0)
        if prefix <= 0.0:
            return 0.0, None
        data = state.graph.edges[transition.src, transition.dst]
        own = effective_probability(dict(data), _EMPTY)
        best_probability = 0.0
        best_target: str | None = None
        for target in state.targets:
            suffix = state.to_target.get(target, {}).get(transition.dst, 0.0)
            if suffix <= 0.0:
                continue
            probability = prefix * own * suffix
            key = (probability, state.target_value(target), target)
            if best_target is None or key > (best_probability, state.target_value(best_target), best_target):
                best_probability, best_target = probability, target
        return best_probability, best_target

    def _paths_through(self, state: ScanGraph) -> dict[str, int]:
        """How many enumerated entry-to-target paths traverse each finding's transition.

        Counted over node pairs rather than the paths' recorded ``finding_ids``, so that
        alternative exploits collapsed onto one edge are each credited with the paths that
        edge carries.
        """
        counts: dict[str, int] = {}
        if not state.paths:
            return counts
        pairs: dict[tuple[str, str], list[str]] = {}
        for finding_id, transition in state.transitions.items():
            if not (transition.admitted and transition.escalates):
                continue
            if not state.graph.has_edge(transition.src, transition.dst):
                continue
            pairs.setdefault((transition.src, transition.dst), []).append(finding_id)
        if not pairs:
            return counts
        for path in state.paths:
            nodes = path.nodes
            for src, dst in zip(nodes, nodes[1:]):
                for finding_id in pairs.get((src, dst), ()):
                    counts[finding_id] = counts.get(finding_id, 0) + 1
        return counts


def chain_scores_for(
    scan: Scan,
    enriched: Iterable[EnrichedFinding],
    config: PipelineConfig | ComponentCConfig | None = None,
    attacker: AttackerModel | None = None,
) -> tuple[AttackGraphSummary, dict[str, ChainScore]]:
    """Build and score one scan in a single call.

    Convenience for callers that hold a scan and its enriched findings and want Component
    C's output without managing a scorer's lifetime.
    """
    scorer = ReachabilityChainScorer(config, attacker)
    summary = scorer.build(scan, list(enriched))
    return summary, scorer.score(scan.scan_id)
