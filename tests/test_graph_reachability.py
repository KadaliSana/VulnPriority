"""Reachability over a hand-built graph, checked against hand-computed numbers.

Nothing here goes through the builder: the graph is four nodes wired by hand so that every
expected value can be written out longhand in the test. If these numbers are right, the
``-log p`` formulation is computing maximum-probability paths and not something that merely
correlates with them.

The graph::

    A ──0.5──► B ──0.40──► C ──0.25──► D
    │                      ▲
    └────────0.1───────────┘
              │
              └──0.05──────────────────► D   (from B)

with ``value(C) = 1_000`` and ``value(D) = 100_000``.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from vulnprio.core.errors import GraphError
from vulnprio.core.models import AttackPath
from vulnprio.graph.reachability import (
    PROBABILITY_ATTR,
    VALUE_ATTR,
    WEIGHT_ATTR,
    edge_betweenness_for_findings,
    entry_node_of,
    enumerate_paths,
    hops_from_entry,
    max_path_probability,
    max_path_probability_from,
    max_path_probability_to,
    probability_to_weight,
    target_nodes_of,
    total_risk,
    weight_to_probability,
)

#: ``(src, dst, probability, finding_id)``. ``None`` marks a structural edge.
EDGES = (
    ("A", "B", 0.5, "f_ab"),
    ("B", "C", 0.4, "f_bc"),
    ("A", "C", 0.1, "f_ac"),
    ("C", "D", 0.25, "f_cd"),
    ("B", "D", 0.05, None),
)


def build_graph() -> nx.DiGraph:
    """The four-node graph above, wired exactly the way the builder wires one."""
    graph = nx.DiGraph()
    for node, value, entry, target in (
        ("A", 0.0, True, False),
        ("B", 0.0, False, False),
        ("C", 1_000.0, False, True),
        ("D", 100_000.0, False, True),
    ):
        graph.add_node(node, asset=node, **{VALUE_ATTR: value}, is_entry=entry, is_target=target)
    for src, dst, probability, finding_id in EDGES:
        finding_probs = {} if finding_id is None else {finding_id: probability}
        graph.add_edge(
            src,
            dst,
            **{PROBABILITY_ATTR: probability, WEIGHT_ATTR: probability_to_weight(probability)},
            kind="exploit" if finding_id else "lateral",
            structural_probability=None if finding_id else probability,
            finding_probs=finding_probs,
            finding_ids=tuple(finding_probs),
            best_finding_id=finding_id,
        )
    return graph


@pytest.fixture
def graph() -> nx.DiGraph:
    return build_graph()


# --------------------------------------------------------------------------
# The -log p transform
# --------------------------------------------------------------------------


@pytest.mark.parametrize("probability", [1.0, 0.9, 0.5, 0.1, 1e-4])
def test_weight_transform_round_trips(probability: float) -> None:
    assert weight_to_probability(probability_to_weight(probability)) == pytest.approx(probability)


def test_certainty_costs_nothing_and_impossibility_costs_everything() -> None:
    assert probability_to_weight(1.0) == pytest.approx(0.0)
    assert probability_to_weight(0.0) == math.inf
    assert weight_to_probability(math.inf) == 0.0


def test_weights_are_non_negative_which_is_dijkstras_precondition(graph: nx.DiGraph) -> None:
    """The reason the maximum-probability path is exact rather than approximate."""
    assert all(data[WEIGHT_ATTR] >= 0.0 for _, _, data in graph.edges(data=True))


def test_summing_weights_is_multiplying_probabilities(graph: nx.DiGraph) -> None:
    chain = graph.edges["A", "B"][WEIGHT_ATTR] + graph.edges["B", "C"][WEIGHT_ATTR]
    assert weight_to_probability(chain) == pytest.approx(0.5 * 0.4)


# --------------------------------------------------------------------------
# Maximum path probability
# --------------------------------------------------------------------------


def test_max_path_probability_against_hand_computed_values(graph: nx.DiGraph) -> None:
    # A -> B: the single edge.
    assert max_path_probability(graph, "A", "B") == pytest.approx(0.5)
    # A -> C: max(0.5 * 0.4, 0.1) = 0.2, i.e. the two-hop route beats the direct one.
    assert max_path_probability(graph, "A", "C") == pytest.approx(0.2)
    # A -> D: max(0.5*0.4*0.25, 0.1*0.25, 0.5*0.05) = max(0.05, 0.025, 0.025) = 0.05
    assert max_path_probability(graph, "A", "D") == pytest.approx(0.05)
    assert max_path_probability(graph, "B", "D") == pytest.approx(0.4 * 0.25)


def test_a_node_reaches_itself_with_certainty(graph: nx.DiGraph) -> None:
    assert max_path_probability(graph, "A", "A") == 1.0


def test_direction_is_respected_so_the_reverse_route_is_impossible(graph: nx.DiGraph) -> None:
    """The property an undirected chaining model throws away."""
    assert max_path_probability(graph, "D", "A") == 0.0
    assert max_path_probability(graph, "C", "B") == 0.0
    assert max_path_probability(graph, "C", "A") == 0.0


def test_absent_nodes_are_simply_unreachable(graph: nx.DiGraph) -> None:
    assert max_path_probability(graph, "A", "Z") == 0.0
    assert max_path_probability(graph, "Z", "A") == 0.0
    assert max_path_probability_from(graph, "Z") == {}


def test_single_source_agrees_with_the_pairwise_query(graph: nx.DiGraph) -> None:
    reach = max_path_probability_from(graph, "A")
    assert reach == pytest.approx({"A": 1.0, "B": 0.5, "C": 0.2, "D": 0.05})
    for node in graph:
        assert reach.get(node, 0.0) == pytest.approx(max_path_probability(graph, "A", node))


def test_reverse_search_prices_the_suffix_of_a_route(graph: nx.DiGraph) -> None:
    to_d = max_path_probability_to(graph, "D")
    assert to_d["C"] == pytest.approx(0.25)
    assert to_d["B"] == pytest.approx(0.4 * 0.25)
    assert to_d["A"] == pytest.approx(0.05)
    assert "D" in to_d and to_d["D"] == pytest.approx(1.0)


def test_removing_an_edge_can_only_lower_the_probability(graph: nx.DiGraph) -> None:
    """The step the non-negativity proof turns on, checked directly."""
    before = max_path_probability(graph, "A", "D")
    pruned = graph.copy()
    pruned.remove_edge("B", "C")
    after = max_path_probability(pruned, "A", "D")
    assert after <= before
    assert after == pytest.approx(0.025)  # best surviving route is now A->C->D or A->B->D


def test_a_weight_callable_evaluates_a_modified_graph_without_touching_it(graph: nx.DiGraph) -> None:
    """How the chain scorer expresses patching: a weight function, never a graph copy."""

    def without_bc(u: str, v: str, data: dict) -> float | None:
        if (u, v) == ("B", "C"):
            return None
        return data[WEIGHT_ATTR]

    assert max_path_probability(graph, "A", "D", weight=without_bc) == pytest.approx(0.025)
    assert graph.has_edge("B", "C")  # unchanged
    assert max_path_probability(graph, "A", "D") == pytest.approx(0.05)


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------


def test_total_risk_is_value_times_reachability(graph: nx.DiGraph) -> None:
    # 1_000 * 0.2 + 100_000 * 0.05
    assert total_risk(graph) == pytest.approx(200.0 + 5_000.0)
    assert total_risk(graph, ["C"]) == pytest.approx(200.0)
    assert total_risk(graph, ["D"]) == pytest.approx(5_000.0)
    assert total_risk(graph, []) == pytest.approx(0.0)


def test_total_risk_ignores_valueless_and_absent_targets(graph: nx.DiGraph) -> None:
    assert total_risk(graph, ["B"]) == pytest.approx(0.0)
    assert total_risk(graph, ["Z"]) == pytest.approx(0.0)


def test_entry_and_target_discovery(graph: nx.DiGraph) -> None:
    assert entry_node_of(graph) == "A"
    assert target_nodes_of(graph) == ("C", "D")
    with pytest.raises(GraphError):
        entry_node_of(nx.DiGraph())


def test_total_risk_of_an_empty_graph_is_zero() -> None:
    assert total_risk(nx.DiGraph()) == 0.0


# --------------------------------------------------------------------------
# Path enumeration
# --------------------------------------------------------------------------


def test_enumerate_paths_finds_every_simple_route_in_probability_order(graph: nx.DiGraph) -> None:
    paths = enumerate_paths(graph, "A", "D", max_hops=3, limit=10)
    assert [path.nodes for path in paths] == [
        ("A", "B", "C", "D"),
        ("A", "B", "D"),
        ("A", "C", "D"),
    ]
    assert [path.probability for path in paths] == pytest.approx([0.05, 0.025, 0.025])
    assert all(isinstance(path, AttackPath) for path in paths)
    assert all(path.target_value == pytest.approx(100_000.0) for path in paths)
    assert paths[0].expected_value == pytest.approx(0.05 * 100_000.0)


def test_enumerate_paths_records_the_findings_that_carry_each_hop(graph: nx.DiGraph) -> None:
    best = enumerate_paths(graph, "A", "D", max_hops=3, limit=1)[0]
    assert best.finding_ids == ("f_ab", "f_bc", "f_cd")
    structural = enumerate_paths(graph, "A", "D", max_hops=2, limit=10)
    # The B->D hop is structural and names no finding.
    assert structural[0].finding_ids in (("f_ab",), ("f_ac", "f_cd"))


def test_enumerate_paths_honours_max_hops_and_limit(graph: nx.DiGraph) -> None:
    assert len(enumerate_paths(graph, "A", "D", max_hops=2, limit=10)) == 2
    assert len(enumerate_paths(graph, "A", "D", max_hops=3, limit=1)) == 1
    assert enumerate_paths(graph, "A", "D", max_hops=1, limit=10) == ()
    assert enumerate_paths(graph, "A", "D", max_hops=3, limit=0) == ()


def test_enumerate_paths_is_directional_and_tolerates_absent_nodes(graph: nx.DiGraph) -> None:
    assert enumerate_paths(graph, "D", "A", max_hops=5, limit=10) == ()
    assert enumerate_paths(graph, "A", "Z", max_hops=5, limit=10) == ()
    assert enumerate_paths(graph, "A", "A", max_hops=5, limit=10) == ()


# --------------------------------------------------------------------------
# Structural position
# --------------------------------------------------------------------------


def test_edge_betweenness_is_reported_per_finding(graph: nx.DiGraph) -> None:
    scores = edge_betweenness_for_findings(graph)

    assert set(scores) == {"f_ab", "f_bc", "f_ac", "f_cd"}  # the structural edge names none
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    # A->B and B->C lie on the most probable route from A to both valuable states; the
    # direct A->C shortcut lies on none of them, because going through B is more likely.
    assert scores["f_ab"] > scores["f_ac"]
    assert scores["f_bc"] > scores["f_ac"]
    assert scores["f_ac"] == pytest.approx(0.0)


def test_edge_betweenness_of_an_edgeless_graph_is_empty() -> None:
    assert edge_betweenness_for_findings(nx.DiGraph()) == {}


def test_hops_from_entry_is_unweighted_depth(graph: nx.DiGraph) -> None:
    assert hops_from_entry(graph, "A") == {"A": 0, "B": 1, "C": 1, "D": 2}
    # Default entry is the flagged node.
    assert hops_from_entry(graph) == hops_from_entry(graph, "A")
    # Unreachable nodes are absent rather than infinite.
    assert hops_from_entry(graph, "D") == {"D": 0}
    assert hops_from_entry(graph, "Z") == {}
    assert hops_from_entry(nx.DiGraph()) == {}
