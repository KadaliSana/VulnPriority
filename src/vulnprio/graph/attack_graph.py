"""Directed multi-hop attack graph construction (DESIGN.md 3.7, Gap 9).

The graph is over **(asset, privilege) states**, not over findings. A node is "the
attacker holds privilege P on asset A", written ``state:<asset>:<PRIVILEGE NAME>``, and a
finding is an *edge*: a transition from the privilege its exploitation requires to the
privilege its exploitation confers, carried at probability ``p_exploit x p_applicable``.
Modelling states rather than findings is what makes chaining expressible: two findings
chain exactly when the privilege one confers is the privilege the other requires, and no
list-of-findings representation can say that.

Four kinds of edge exist, and each carries a trust tier:

``exploit``
    One per finding that escalates privilege, ``state:<host>:<required>`` to
    ``state:<host>:<gained>``. Probability ``p_exploit x p_applicable``, floored at
    ``min_edge_probability``. Admitted only when the transition's supporting evidence is
    at tier <= ``SCANNER``; see :func:`edge_evidence_tier`.

``privilege_implication``
    Probability 1, from every privilege on an asset to every lower privilege on it.
    Holding ADMIN means holding USER; the graph should not make the attacker re-earn it.

``lateral``
    The internet-to-host exposure edges that attach every internet-facing host to the
    entry state, and the host-to-host edges implied by scanner-observed
    ``Endpoint.links_to`` that cross a host boundary. Both are tier ``SCANNER``: they
    exist because the scanner saw structure, not because anything asserted them in prose.

**Value.** A state holds money when an attacker standing in it has realised a finding's
business impact there. For every finding the value ``impact.total x criticality **
criticality_weight_exponent`` is placed on ``state:<host>:<gained>``, deduplicated per
endpoint so two findings on one endpoint do not sell the same asset twice. States at
privilege ``NONE`` are never targets: merely being able to reach a host is not worth
anything, and treating it as valuable would put an unpatchable constant into ``R(G)``.

**Direction is load-bearing.** Every edge points from the privilege required to the
privilege gained. A finding that requires a privilege nothing in the graph confers sits on
an unreachable node and contributes exactly zero, which is the correct answer and the one
an undirected model cannot give.

**The security property.** Untrusted text must not be able to invent a path. The privilege
transition is taken from the operator-tier CWE table in
:mod:`vulnprio.graph.privilege_map` whenever the CWE is covered, and only falls back to
the exploitability assessment otherwise. An edge that exists *only* because untrusted text
asserted it - an uncovered CWE whose assessment was authored on the authority of a
reference page or a target response - is rejected and counted in
``AttackGraphSummary.rejected_untrusted_edges``. A finding whose sandbox canary leaked is
rejected outright: the model that produced its assessment was demonstrably under someone
else's control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from vulnprio.core.config import ComponentCConfig, PipelineConfig
from vulnprio.core.enums import PrivilegeLevel, TrustTier
from vulnprio.core.errors import GraphError
from vulnprio.core.models import (
    AttackGraphSummary,
    AttackerModel,
    EnrichedFinding,
    GraphEdge,
    GraphNode,
    Scan,
)
from vulnprio.graph.privilege_map import is_known_cwe, privileges_for
from vulnprio.graph.reachability import (
    PROBABILITY_ATTR,
    VALUE_ATTR,
    WEIGHT_ATTR,
    enumerate_paths,
    probability_to_weight,
    total_risk,
)

__all__ = [
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
    "FindingTransition",
    "AttackGraphBuilder",
]

#: Node id prefix. Node ids are ``state:<asset>:<PRIVILEGE NAME>``.
NODE_PREFIX = "state"

#: Asset name of the entry state: the attacker starts on the internet, not on a host.
INTERNET_ASSET = "internet"

#: Probability of moving between hosts the scanner observed a link between. A constant
#: rather than an estimate, because ``links_to`` says the hosts talk, not how easily the
#: attacker follows; the number is floored at ``min_edge_probability``.
LATERAL_EDGE_PROBABILITY = 0.5

#: Reaching the unauthenticated surface of an internet-facing host is free.
ENTRY_EDGE_PROBABILITY = 1.0

#: Holding a privilege implies holding every lower one, with certainty.
PRIVILEGE_IMPLICATION_PROBABILITY = 1.0

#: The highest trust tier allowed to create an edge (DESIGN.md 3.7).
MAX_EDGE_TIER = TrustTier.SCANNER


def state_node(asset: str, privilege: PrivilegeLevel) -> str:
    """Canonical node id for holding ``privilege`` on ``asset``."""
    return f"{NODE_PREFIX}:{asset}:{PrivilegeLevel(privilege).name}"


def parse_state_node(node_id: str) -> tuple[str, PrivilegeLevel]:
    """Inverse of :func:`state_node`. Tolerates assets containing colons (``host:8443``)."""
    if not node_id.startswith(f"{NODE_PREFIX}:"):
        raise GraphError(f"not a state node id: {node_id!r}")
    body = node_id[len(NODE_PREFIX) + 1 :]
    asset, _, privilege_name = body.rpartition(":")
    if not asset or privilege_name not in PrivilegeLevel.__members__:
        raise GraphError(f"not a state node id: {node_id!r}")
    return asset, PrivilegeLevel[privilege_name]


def edge_evidence_tier(item: EnrichedFinding) -> TrustTier:
    """Trust tier of the evidence this finding's privilege transition rests on.

    ``SCANNER`` when the CWE is in the operator-tier table: the transition is curated
    knowledge indexed by an identifier the scanner reported, and no prose was consulted to
    obtain it. Otherwise the tier of whatever produced the exploitability assessment,
    because for an uncovered CWE the assessment is the *only* reason to believe the
    transition exists. An assessment with no audit record came from deterministic
    scanner-tier structure, so ``SCANNER`` is the honest default there too.
    """
    if is_known_cwe(item.finding.cwe_id):
        return MAX_EDGE_TIER
    audit = item.exploitability.audit
    if audit is None:
        return MAX_EDGE_TIER
    return TrustTier(audit.max_tier_used)


def admits_edge(item: EnrichedFinding) -> bool:
    """Whether this finding may create an edge at all.

    Two ways to fail: the transition rests on evidence above ``SCANNER``, or the sandbox
    canary leaked while assessing the finding, which means the assessment is attacker
    output rather than model output regardless of what tier it claims.
    """
    if item.trust.canary_leaked:
        return False
    return edge_evidence_tier(item) <= MAX_EDGE_TIER


def effective_probability(data: dict, patched: frozenset[str] | set[str] = frozenset()) -> float:
    """Probability of an edge once ``patched`` findings are removed from it.

    Several findings can express the same state transition - alternative routes to the
    same escalation - and the graph keeps one edge per transition carrying the best of
    them, because a most-probable-path search would use the best one anyway. Patching a
    finding therefore does not necessarily delete the edge: it re-maximises over whatever
    is left, and returns ``0.0`` only when nothing is. Structural edges have no
    contributors and keep their own probability forever.
    """
    best = float(data.get("structural_probability") or 0.0)
    for finding_id, probability in data.get("finding_probs", {}).items():
        if finding_id in patched:
            continue
        best = max(best, float(probability))
    return best


@dataclass(frozen=True, slots=True)
class FindingTransition:
    """What one finding does to the privilege lattice, and whether it was believed."""

    finding_id: str
    host: str
    endpoint_id: str
    required: PrivilegeLevel
    gained: PrivilegeLevel
    probability: float
    value: float
    tier: TrustTier
    admitted: bool
    rejection_reason: str = ""

    @property
    def escalates(self) -> bool:
        """True when the finding moves the attacker up the lattice and so carries an edge."""
        return int(self.gained) > int(self.required)

    @property
    def privilege_gain(self) -> int:
        """Levels climbed, zero for a non-escalating finding."""
        return max(0, int(self.gained) - int(self.required))

    @property
    def src(self) -> str:
        return state_node(self.host, self.required)

    @property
    def dst(self) -> str:
        return state_node(self.host, self.gained)


@dataclass
class _EdgeAccumulator:
    """Collects the findings that express one state transition before it becomes an edge."""

    kind: str = "exploit"
    tier: TrustTier = TrustTier.SCANNER
    structural_probability: float | None = None
    finding_probs: dict[str, float] = field(default_factory=dict)


class AttackGraphBuilder:
    """Builds the ``(asset, privilege)`` state graph for one scan.

    Stateless between calls: ``build`` takes a scan and its enriched findings and returns
    a fresh graph plus the :class:`AttackGraphSummary` that records what it did, including
    how many edges the trust rule rejected.
    """

    def __init__(
        self,
        config: PipelineConfig | ComponentCConfig | None = None,
        attacker: AttackerModel | None = None,
    ) -> None:
        """``config`` may be the whole pipeline config or just its Component C section."""
        self.config: ComponentCConfig = _component_c(config)
        self.attacker = attacker
        self.entry_privilege: PrivilegeLevel = (
            attacker.entry_privilege if attacker is not None else PrivilegeLevel.NONE
        )

    # -- public API ---------------------------------------------------------

    @property
    def entry_node(self) -> str:
        """The state the attacker starts in: ``state:internet:<entry privilege>``."""
        return state_node(INTERNET_ASSET, self.entry_privilege)

    def transitions(self, scan: Scan, enriched: list[EnrichedFinding]) -> list[FindingTransition]:
        """Resolve every finding to a privilege transition, admitted or rejected.

        Exposed because it is the auditable middle of graph construction: it says, per
        finding, which states it connects, at what probability, on whose authority, and -
        when the answer is "nobody trustworthy" - why it was dropped.
        """
        out: list[FindingTransition] = []
        exponent = float(self.config.criticality_weight_exponent)
        for item in enriched:
            endpoint = item.endpoint
            if endpoint.endpoint_id != item.finding.endpoint_id:
                raise GraphError(
                    f"enriched finding {item.finding_id} carries endpoint "
                    f"{endpoint.endpoint_id} but names {item.finding.endpoint_id}"
                )
            mapped_required, mapped_gained = privileges_for(item.finding.cwe_id, item.exploitability)
            # An endpoint the scanner could only reach with credentials cannot be attacked
            # from below them, whatever the CWE table says in general.
            required = PrivilegeLevel(max(int(mapped_required), int(endpoint.auth_required)))
            gained = PrivilegeLevel(max(int(mapped_gained), int(required)))

            probability = self._edge_probability(item)
            tier = edge_evidence_tier(item)
            admitted = admits_edge(item)
            reason = ""
            if not admitted:
                reason = (
                    "sandbox canary leaked during assessment"
                    if item.trust.canary_leaked
                    else f"privilege transition asserted only at tier {tier.name}"
                )
            out.append(
                FindingTransition(
                    finding_id=item.finding_id,
                    host=endpoint.host,
                    endpoint_id=endpoint.endpoint_id,
                    required=required,
                    gained=gained,
                    probability=probability,
                    value=_weighted_value(item, exponent),
                    tier=tier,
                    admitted=admitted,
                    rejection_reason=reason,
                )
            )
        return out

    def build(
        self,
        scan: Scan,
        enriched: list[EnrichedFinding],
    ) -> tuple[nx.DiGraph, AttackGraphSummary]:
        """Construct the graph and its summary for ``scan``.

        Returns the live ``networkx.DiGraph`` - the object every reachability query runs
        against - and the frozen :class:`AttackGraphSummary` that records the nodes,
        edges, entry, targets, total risk, most probable paths and rejection count.
        """
        for item in enriched:
            if item.scan_id != scan.scan_id:
                raise GraphError(
                    f"finding {item.finding_id} belongs to scan {item.scan_id}, not {scan.scan_id}"
                )

        graph = nx.DiGraph()
        transitions = self.transitions(scan, enriched)
        rejected = sum(1 for transition in transitions if not transition.admitted)

        hosts = self._hosts(scan)
        internet_facing = self._internet_facing_hosts(scan, hosts)
        accumulators: dict[tuple[str, str], _EdgeAccumulator] = {}

        self._add_node(graph, INTERNET_ASSET, self.entry_privilege, is_entry=True)
        for host in hosts:
            self._add_node(graph, host, PrivilegeLevel.NONE)

        # Exploit edges, and the value each finding puts on the lattice. A rejected
        # transition creates nothing at all - not the edge, not the states it claimed to
        # connect, and not the value it claimed to unlock - because all three rest on the
        # same untrusted assertion.
        values: dict[str, dict[str, float]] = {}
        for transition in transitions:
            if not transition.admitted:
                continue
            self._add_node(graph, transition.host, transition.required)
            self._add_node(graph, transition.host, transition.gained)
            if transition.escalates:
                accumulator = accumulators.setdefault(
                    (transition.src, transition.dst), _EdgeAccumulator(kind="exploit")
                )
                accumulator.finding_probs[transition.finding_id] = transition.probability
                accumulator.tier = TrustTier(max(int(accumulator.tier), int(transition.tier)))
            if transition.gained == PrivilegeLevel.NONE or transition.value <= 0.0:
                continue
            per_endpoint = values.setdefault(transition.dst, {})
            per_endpoint[transition.endpoint_id] = max(
                per_endpoint.get(transition.endpoint_id, 0.0), transition.value
            )

        # Entry exposure: the attacker can talk to every internet-facing host for free.
        for host in internet_facing:
            self._add_node(graph, host, self.entry_privilege)
            accumulators.setdefault(
                (self.entry_node, state_node(host, self.entry_privilege)),
                _EdgeAccumulator(kind="lateral", structural_probability=self._floor(ENTRY_EDGE_PROBABILITY)),
            )

        # Lateral edges between hosts the scanner observed linking to one another.
        if self.config.admit_lateral_edges:
            for src_host, dst_host in self._observed_host_links(scan):
                self._add_node(graph, src_host, PrivilegeLevel.NONE)
                self._add_node(graph, dst_host, PrivilegeLevel.NONE)
                key = (state_node(src_host, PrivilegeLevel.NONE), state_node(dst_host, PrivilegeLevel.NONE))
                accumulators.setdefault(
                    key,
                    _EdgeAccumulator(
                        kind="lateral", structural_probability=self._floor(LATERAL_EDGE_PROBABILITY)
                    ),
                )

        # Privilege implication: holding more implies holding less, with certainty.
        for src, dst in self._implication_pairs(graph):
            accumulators.setdefault(
                (src, dst),
                _EdgeAccumulator(
                    kind="privilege_implication",
                    structural_probability=PRIVILEGE_IMPLICATION_PROBABILITY,
                ),
            )

        for (src, dst), accumulator in accumulators.items():
            probability = effective_probability(
                {
                    "structural_probability": accumulator.structural_probability,
                    "finding_probs": accumulator.finding_probs,
                }
            )
            if probability <= 0.0:
                continue
            best_finding = _best_finding(accumulator.finding_probs)
            graph.add_edge(
                src,
                dst,
                **{
                    PROBABILITY_ATTR: probability,
                    WEIGHT_ATTR: probability_to_weight(probability),
                    "kind": accumulator.kind,
                    "tier": accumulator.tier,
                    "structural_probability": accumulator.structural_probability,
                    "finding_probs": dict(accumulator.finding_probs),
                    "finding_ids": tuple(sorted(accumulator.finding_probs)),
                    "best_finding_id": best_finding,
                },
            )

        for node, per_endpoint in values.items():
            graph.nodes[node][VALUE_ATTR] = float(sum(per_endpoint.values()))
            graph.nodes[node]["is_target"] = True

        summary = self._summarise(scan, graph, transitions, rejected)
        return graph, summary

    # -- internals ----------------------------------------------------------

    def _floor(self, probability: float) -> float:
        """Clamp a probability into ``[min_edge_probability, 1]``."""
        return min(1.0, max(float(self.config.min_edge_probability), float(probability)))

    def _edge_probability(self, item: EnrichedFinding) -> float:
        """``p_exploit x p_applicable``, floored so that ``-log p`` stays finite."""
        raw = float(item.likelihood.p_exploit) * float(item.applicability.p_applicable)
        return self._floor(raw)

    def _add_node(
        self,
        graph: nx.DiGraph,
        asset: str,
        privilege: PrivilegeLevel,
        *,
        is_entry: bool = False,
    ) -> str:
        node_id = state_node(asset, privilege)
        if node_id not in graph:
            graph.add_node(
                node_id,
                asset=asset,
                privilege=PrivilegeLevel(privilege),
                **{VALUE_ATTR: 0.0},
                is_entry=is_entry,
                is_target=False,
            )
        elif is_entry:
            graph.nodes[node_id]["is_entry"] = True
        return node_id

    def _hosts(self, scan: Scan) -> tuple[str, ...]:
        """Every host the scan mentions, deterministically ordered."""
        hosts = {endpoint.host for endpoint in scan.endpoints if endpoint.host}
        hosts.update(host for host in scan.hosts if host)
        return tuple(sorted(hosts))

    def _internet_facing_hosts(self, scan: Scan, hosts: tuple[str, ...]) -> tuple[str, ...]:
        """Hosts with at least one internet-facing endpoint.

        A host with no endpoints at all (named only in ``Scan.hosts``) is treated as
        internet-facing, because nothing observed says otherwise and refusing to attach it
        would silently hide whatever value sits on it.
        """
        facing: set[str] = set()
        seen: set[str] = set()
        for endpoint in scan.endpoints:
            seen.add(endpoint.host)
            if endpoint.internet_facing:
                facing.add(endpoint.host)
        facing.update(host for host in hosts if host not in seen)
        return tuple(sorted(facing))

    def _observed_host_links(self, scan: Scan) -> tuple[tuple[str, str], ...]:
        """Host pairs implied by ``Endpoint.links_to`` that cross a host boundary."""
        by_id = {endpoint.endpoint_id: endpoint for endpoint in scan.endpoints}
        pairs: set[tuple[str, str]] = set()
        for endpoint in scan.endpoints:
            for target_id in endpoint.links_to:
                target = by_id.get(target_id)
                if target is None or target.host == endpoint.host:
                    continue
                pairs.add((endpoint.host, target.host))
        return tuple(sorted(pairs))

    def _implication_pairs(self, graph: nx.DiGraph) -> tuple[tuple[str, str], ...]:
        """Every ``(higher, lower)`` state pair on a shared asset."""
        by_asset: dict[str, list[PrivilegeLevel]] = {}
        for _, data in graph.nodes(data=True):
            by_asset.setdefault(str(data["asset"]), []).append(PrivilegeLevel(data["privilege"]))
        pairs: list[tuple[str, str]] = []
        for asset, privileges in sorted(by_asset.items()):
            ordered = sorted(set(privileges), key=int, reverse=True)
            for index, higher in enumerate(ordered):
                for lower in ordered[index + 1 :]:
                    pairs.append((state_node(asset, higher), state_node(asset, lower)))
        return tuple(pairs)

    def _summarise(
        self,
        scan: Scan,
        graph: nx.DiGraph,
        transitions: list[FindingTransition],
        rejected: int,
    ) -> AttackGraphSummary:
        """Freeze the built graph into the contract type, including risk and top paths."""
        nodes = tuple(
            GraphNode(
                node_id=str(node_id),
                asset=str(data["asset"]),
                privilege=PrivilegeLevel(data["privilege"]),
                value=max(0.0, float(data.get(VALUE_ATTR, 0.0))),
                is_entry=bool(data.get("is_entry", False)),
                is_target=bool(data.get("is_target", False)),
            )
            for node_id, data in sorted(graph.nodes(data=True))
        )
        targets = tuple(node.node_id for node in nodes if node.is_target)

        edges: list[GraphEdge] = []
        for src, dst, data in sorted(graph.edges(data=True), key=lambda item: (item[0], item[1])):
            finding_probs: dict[str, float] = data.get("finding_probs", {})
            if finding_probs:
                # One record per finding, so the summary keeps per-finding provenance even
                # though the graph collapses alternatives onto a single transition.
                for finding_id in sorted(finding_probs):
                    edges.append(
                        GraphEdge(
                            src=str(src),
                            dst=str(dst),
                            probability=min(1.0, float(finding_probs[finding_id])),
                            finding_id=str(finding_id),
                            kind="exploit",
                            tier=TrustTier(data.get("tier", TrustTier.SCANNER)),
                        )
                    )
            else:
                edges.append(
                    GraphEdge(
                        src=str(src),
                        dst=str(dst),
                        probability=min(1.0, float(data[PROBABILITY_ATTR])),
                        finding_id=None,
                        kind=data.get("kind", "lateral"),
                        tier=TrustTier(data.get("tier", TrustTier.SCANNER)),
                    )
                )

        risk = total_risk(graph, targets, entry=self.entry_node) if targets else 0.0
        paths: list = []
        for target in targets:
            paths.extend(
                enumerate_paths(
                    graph,
                    self.entry_node,
                    target,
                    max_hops=int(self.config.max_hops),
                    limit=int(self.config.top_paths),
                )
            )
        paths.sort(key=lambda path: (-path.expected_value, -path.probability, path.nodes))

        return AttackGraphSummary(
            scan_id=scan.scan_id,
            nodes=nodes,
            edges=tuple(edges),
            entry_node=self.entry_node,
            target_nodes=targets,
            total_risk=max(0.0, risk),
            top_paths=tuple(paths[: int(self.config.top_paths)]),
            monotone_verified=False,
            rejected_untrusted_edges=rejected,
        )


def _component_c(config: PipelineConfig | ComponentCConfig | None) -> ComponentCConfig:
    """Accept the whole pipeline config, just Component C's section, or nothing."""
    if config is None:
        return ComponentCConfig()
    if isinstance(config, ComponentCConfig):
        return config
    return config.component_c


def _weighted_value(item: EnrichedFinding, exponent: float) -> float:
    """``impact.total x criticality ** exponent`` for one finding's endpoint.

    The exponent (``ComponentCConfig.criticality_weight_exponent``) is the operator's
    statement of how sharply criticality should concentrate value: 0 ignores criticality,
    1 scales linearly, larger values push value onto the few assets Component A judged
    genuinely critical.
    """
    value = max(0.0, float(item.impact.total))
    if exponent == 0.0:
        return value
    criticality = max(0.0, float(item.asset.criticality))
    return value * (criticality ** float(exponent))


def _best_finding(finding_probs: dict[str, float]) -> str | None:
    """The finding whose probability the collapsed edge is carrying (ties by id)."""
    if not finding_probs:
        return None
    return max(sorted(finding_probs), key=lambda finding_id: finding_probs[finding_id])
