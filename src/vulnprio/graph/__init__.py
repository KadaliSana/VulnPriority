"""Component C: the directed multi-hop monotone attack graph (DESIGN.md 3.7, Gap 9).

Findings become *edges* between ``(asset, privilege)`` states, a finding's priority
contribution becomes the money that stops being reachable when it is patched, and the
whole thing is arranged so that contribution is provably non-negative.
"""

from __future__ import annotations

from vulnprio.graph.attack_graph import (
    ENTRY_EDGE_PROBABILITY,
    INTERNET_ASSET,
    LATERAL_EDGE_PROBABILITY,
    MAX_EDGE_TIER,
    NODE_PREFIX,
    PRIVILEGE_IMPLICATION_PROBABILITY,
    AttackGraphBuilder,
    FindingTransition,
    admits_edge,
    edge_evidence_tier,
    effective_probability,
    parse_state_node,
    state_node,
)
from vulnprio.graph.chain_scorer import (
    PATH_COUNT_LIMIT,
    RUNTIME_VERIFY_SUBSETS,
    ReachabilityChainScorer,
    ScanGraph,
    chain_scores_for,
)
from vulnprio.graph.monotone import (
    ABS_TOLERANCE,
    REL_TOLERANCE,
    assert_monotone_under_patching,
    random_patch_subsets,
)
from vulnprio.graph.privilege_map import (
    ADMIN_CONTROL_CWES,
    CIA_ADMIN_THRESHOLD,
    CIA_SYSTEM_THRESHOLD,
    CWE_PRIVILEGE_RULES,
    DATA_ACCESS_CWES,
    NO_PRIVILEGE_CWES,
    SYSTEM_CONTROL_CWES,
    USER_INTERACTION_CWES,
    PrivilegeRule,
    cia_derived_privilege,
    is_known_cwe,
    privileges_for,
    rule_for,
)
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

__all__ = [
    # privilege_map
    "PrivilegeRule",
    "CWE_PRIVILEGE_RULES",
    "SYSTEM_CONTROL_CWES",
    "ADMIN_CONTROL_CWES",
    "DATA_ACCESS_CWES",
    "USER_INTERACTION_CWES",
    "NO_PRIVILEGE_CWES",
    "CIA_ADMIN_THRESHOLD",
    "CIA_SYSTEM_THRESHOLD",
    "cia_derived_privilege",
    "is_known_cwe",
    "rule_for",
    "privileges_for",
    # attack_graph
    "AttackGraphBuilder",
    "FindingTransition",
    "NODE_PREFIX",
    "INTERNET_ASSET",
    "LATERAL_EDGE_PROBABILITY",
    "ENTRY_EDGE_PROBABILITY",
    "PRIVILEGE_IMPLICATION_PROBABILITY",
    "MAX_EDGE_TIER",
    "state_node",
    "parse_state_node",
    "edge_evidence_tier",
    "admits_edge",
    "effective_probability",
    # reachability
    "PROBABILITY_ATTR",
    "WEIGHT_ATTR",
    "VALUE_ATTR",
    "probability_to_weight",
    "weight_to_probability",
    "entry_node_of",
    "target_nodes_of",
    "max_path_probability",
    "max_path_probability_from",
    "max_path_probability_to",
    "total_risk",
    "enumerate_paths",
    "edge_betweenness_for_findings",
    "hops_from_entry",
    # chain_scorer
    "ReachabilityChainScorer",
    "ScanGraph",
    "chain_scores_for",
    "PATH_COUNT_LIMIT",
    "RUNTIME_VERIFY_SUBSETS",
    # monotone
    "assert_monotone_under_patching",
    "random_patch_subsets",
    "ABS_TOLERANCE",
    "REL_TOLERANCE",
]
