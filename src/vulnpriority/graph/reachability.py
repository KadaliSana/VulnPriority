"""Maximum-probability reachability over the attack graph (DESIGN.md 3.7, Gap 9).

Everything Component C claims about a finding reduces to one question: how likely is the
attacker to arrive at a state that holds money, and by how much does that likelihood fall
if this finding is fixed? This module answers the first half; ``chain_scorer`` differences
it to answer the second.

**Why ``-log p`` and why it matters.** The probability of a path is the product of its
edge probabilities. Taking negative logarithms turns that product into a sum and turns
"most probable path" into "shortest path":

    argmax_P prod_{e in P} p(e)  ==  argmin_P sum_{e in P} -log p(e)

Because every ``p(e)`` lies in ``(0, 1]`` every weight ``-log p(e)`` is non-negative,
which is exactly Dijkstra's precondition, so the maximum-probability path is computed
exactly rather than approximated. That is not a performance note, it is the reason the
framework's central non-negativity claim is *provable*:

* Removing edges from a graph can only shrink the set of paths between two nodes.
* Shrinking the set of candidates can only raise the minimum of ``sum -log p``.
* ``exp(-x)`` is decreasing, so the maximum path probability can only fall.
* Node values are fixed at build time and never depend on which findings are patched.
* Therefore ``R(G) = sum_t value(t) * maxpathprob(entry -> t)`` can only fall when edges
  are removed, so ``reach_delta(f) = R(G) - R(G \\ edges(f)) >= 0`` for every finding.

A "most likely path" formulation with an additive, non-negative cost is what buys that
argument. A formulation that summed probabilities over paths, or that counted paths,
would have neither the exactness nor the monotonicity, and ``ChainScore.reach_delta``
could not be a non-negative field on the frozen contract.

Edge probabilities live on the ``probability`` attribute and their logarithms are cached
on ``weight`` by the builder; callers that need to evaluate a *modified* graph (the
patching counterfactual) pass a weight callable instead of the attribute name, which
networkx honours and which lets the same immutable graph answer every query.
"""

from __future__ import annotations

import math
from itertools import islice
from typing import Any, Callable, Iterable, Mapping

import networkx as nx

from vulnpriority.core.errors import GraphError
from vulnpriority.core.models import AttackPath

__all__ = [
    "WeightSpec",
    "PROBABILITY_ATTR",
    "WEIGHT_ATTR",
    "VALUE_ATTR",
    "probability_to_weight",
    "weight_to_probability",
    "entry_node_of",
    "target_nodes_of",
    "max_path_probability_from",
    "max_path_probability",
    "max_path_probability_to",
    "total_risk",
    "enumerate_paths",
    "edge_betweenness_for_findings",
    "hops_from_entry",
]

#: Edge attribute holding the transition probability in ``(0, 1]``.
PROBABILITY_ATTR = "probability"

#: Edge attribute holding ``-log(probability)``, the additive Dijkstra cost.
WEIGHT_ATTR = "weight"

#: Node attribute holding the monetary value realised in that state.
VALUE_ATTR = "value"

#: A networkx weight specification: an attribute name, or ``f(u, v, data) -> float|None``
#: where ``None`` means "this edge does not exist in the graph being evaluated".
WeightSpec = str | Callable[[str, str, Mapping[str, Any]], float | None]

#: Paths enumerated per request before the top-``limit`` are selected. Simple-path
#: enumeration is exponential in general; the state graph is small (one node per
#: (host, privilege) pair) but the cap keeps a pathological topology bounded.
ENUMERATION_MULTIPLIER = 20
ENUMERATION_FLOOR = 200


def probability_to_weight(probability: float) -> float:
    """``-log p``, the additive cost of taking an edge of probability ``p``.

    Returns ``math.inf`` for a non-positive probability so that an impossible transition
    is unreachable rather than an arithmetic error.
    """
    if probability <= 0.0:
        return math.inf
    return -math.log(min(1.0, float(probability)))


def weight_to_probability(weight: float) -> float:
    """Inverse of :func:`probability_to_weight`, clamped into ``[0, 1]``."""
    if not math.isfinite(weight):
        return 0.0
    return min(1.0, max(0.0, math.exp(-weight)))


def entry_node_of(graph: nx.DiGraph) -> str:
    """The node flagged ``is_entry``. Raises when the graph has no entry state."""
    for node, data in graph.nodes(data=True):
        if data.get("is_entry"):
            return str(node)
    raise GraphError("attack graph has no entry node")


def target_nodes_of(graph: nx.DiGraph) -> tuple[str, ...]:
    """Every node flagged ``is_target``, in a deterministic order."""
    return tuple(sorted(str(node) for node, data in graph.nodes(data=True) if data.get("is_target")))


def max_path_probability_from(
    graph: nx.DiGraph,
    source: str,
    *,
    weight: WeightSpec = WEIGHT_ATTR,
) -> dict[str, float]:
    """Maximum path probability from ``source`` to every reachable node.

    One Dijkstra pass over ``-log p``; nodes that are unreachable are simply absent from
    the result, which callers read as probability zero.
    """
    if source not in graph:
        return {}
    distances = nx.single_source_dijkstra_path_length(graph, source, weight=weight)
    return {str(node): weight_to_probability(dist) for node, dist in distances.items()}


def max_path_probability(
    graph: nx.DiGraph,
    source: str,
    target: str,
    *,
    weight: WeightSpec = WEIGHT_ATTR,
) -> float:
    """Probability of the most likely directed path ``source -> target``.

    Returns ``1.0`` when source and target coincide and ``0.0`` when either node is
    absent or no directed path exists. Direction is respected: this is a ``DiGraph`` and
    an edge pointing the other way contributes nothing, which is precisely the property
    the incumbent undirected chaining models throw away.
    """
    if source not in graph or target not in graph:
        return 0.0
    if source == target:
        return 1.0
    try:
        distance = nx.dijkstra_path_length(graph, source, target, weight=weight)
    except nx.NetworkXNoPath:
        return 0.0
    return weight_to_probability(distance)


def max_path_probability_to(
    graph: nx.DiGraph,
    target: str,
    *,
    weight: WeightSpec = WEIGHT_ATTR,
) -> dict[str, float]:
    """Maximum path probability from every node *to* ``target``.

    Computed with a single Dijkstra pass on the reversed graph, which is what lets the
    chain scorer price "entry -> this edge -> target" for every finding without a
    quadratic number of searches.
    """
    if target not in graph:
        return {}
    reverse = graph.reverse(copy=False)
    if callable(weight):

        def reversed_weight(u: str, v: str, data: Mapping[str, Any]) -> float | None:
            return weight(v, u, data)

        distances = nx.single_source_dijkstra_path_length(reverse, target, weight=reversed_weight)
    else:
        distances = nx.single_source_dijkstra_path_length(reverse, target, weight=weight)
    return {str(node): weight_to_probability(dist) for node, dist in distances.items()}


def total_risk(
    graph: nx.DiGraph,
    targets: Iterable[str] | None = None,
    *,
    entry: str | None = None,
    weight: WeightSpec = WEIGHT_ATTR,
) -> float:
    """``R(G) = sum over targets t of value(t) * maxpathprob(entry -> t)``.

    The framework's definition of how much money is currently reachable. ``targets``
    defaults to every node flagged ``is_target`` and ``entry`` to the node flagged
    ``is_entry``. Node values are read from the graph and are never a function of which
    edges are present, which is what makes the difference of two of these numbers a valid
    non-negative contribution.
    """
    if graph.number_of_nodes() == 0:
        return 0.0
    entry_id = entry if entry is not None else entry_node_of(graph)
    chosen = tuple(targets) if targets is not None else target_nodes_of(graph)
    if not chosen:
        return 0.0
    reach = max_path_probability_from(graph, entry_id, weight=weight)
    total = 0.0
    for node in chosen:
        value = float(graph.nodes[node].get(VALUE_ATTR, 0.0)) if node in graph else 0.0
        if value <= 0.0:
            continue
        total += value * reach.get(str(node), 0.0)
    return max(0.0, total)


def _path_probability(graph: nx.DiGraph, nodes: list[str]) -> tuple[float, tuple[str, ...]]:
    """Probability of a concrete node sequence and the findings that carry it."""
    probability = 1.0
    finding_ids: list[str] = []
    for src, dst in zip(nodes, nodes[1:]):
        data = graph.edges[src, dst]
        probability *= float(data.get(PROBABILITY_ATTR, 0.0))
        best = data.get("best_finding_id")
        if best is not None and best not in finding_ids:
            finding_ids.append(str(best))
    return probability, tuple(finding_ids)


def enumerate_paths(
    graph: nx.DiGraph,
    source: str,
    target: str,
    max_hops: int = 6,
    limit: int = 10,
) -> tuple[AttackPath, ...]:
    """The most probable simple paths ``source -> target`` of at most ``max_hops`` edges.

    Simple paths only: a repeated state adds no capability, so revisiting one is never
    part of a most-probable route. Enumeration is bounded (see
    :data:`ENUMERATION_MULTIPLIER`) and the survivors are returned in descending
    probability order, which is what ``AttackGraphSummary.top_paths`` reports and what the
    chain scorer counts ``n_paths_through`` over.
    """
    if source not in graph or target not in graph or limit <= 0 or max_hops <= 0:
        return ()
    if source == target:
        return ()
    cap = max(limit * ENUMERATION_MULTIPLIER, ENUMERATION_FLOOR)
    target_value = float(graph.nodes[target].get(VALUE_ATTR, 0.0))
    candidates: list[AttackPath] = []
    raw = nx.all_simple_paths(graph, source, target, cutoff=max_hops)
    for nodes in islice(raw, cap):
        probability, finding_ids = _path_probability(graph, list(nodes))
        if probability <= 0.0:
            continue
        candidates.append(
            AttackPath(
                nodes=tuple(str(node) for node in nodes),
                finding_ids=finding_ids,
                probability=min(1.0, probability),
                target_value=target_value,
            )
        )
    candidates.sort(key=lambda path: (-path.probability, path.nodes))
    return tuple(candidates[:limit])


def edge_betweenness_for_findings(
    graph: nx.DiGraph,
    *,
    weight: str | None = WEIGHT_ATTR,
) -> dict[str, float]:
    """Edge betweenness of every finding's transition, keyed by ``finding_id``.

    Betweenness here is computed over ``-log p`` shortest paths, so it measures how often
    a finding's transition lies on the *most probable* route between states rather than
    on an arbitrary hop-count route. A finding whose edge carries many such routes is
    structurally load-bearing even when its own severity is unremarkable; this is the
    topological half of the chokepoint story that ``reach_delta`` prices in money.

    Findings sharing one state transition (alternative exploits for the same escalation)
    each receive that edge's score: they are individually on those paths, and the
    difference in what *removing* one of them achieves is what ``reach_delta``
    measures.
    """
    if graph.number_of_edges() == 0:
        return {}
    scores = nx.edge_betweenness_centrality(graph, weight=weight)
    out: dict[str, float] = {}
    for (src, dst), value in scores.items():
        data = graph.edges[src, dst]
        for finding_id in data.get("finding_ids", ()):  # structural edges carry none
            out[str(finding_id)] = max(out.get(str(finding_id), 0.0), max(0.0, float(value)))
    return out


def hops_from_entry(graph: nx.DiGraph, entry: str | None = None) -> dict[str, int]:
    """Unweighted hop count from the entry state to every reachable node.

    Deliberately unweighted: this answers "how deep in the chain is this", a structural
    question, while everything probabilistic is answered by the ``-log p`` searches above.
    Unreachable nodes are absent from the result.
    """
    if graph.number_of_nodes() == 0:
        return {}
    entry_id = entry if entry is not None else entry_node_of(graph)
    if entry_id not in graph:
        return {}
    return {str(node): int(depth) for node, depth in nx.single_source_shortest_path_length(graph, entry_id).items()}
