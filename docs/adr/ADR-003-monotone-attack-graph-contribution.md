# ADR-003: Maximum-probability paths for monotone chain contribution

**Status:** Accepted
**Date:** 2026-09-12
**Deciders:** M. Srikar Bharadwaj, K. Navneet Sai, K. Shasanth Reddy

## Context

Per-vulnerability scoring cannot express chaining. The review identifies this as a first-order
deficiency of the incumbent standard, and the graph-based studies in the corpus demonstrate
what it costs: a buffer overflow ranked sixth by severity is actually the prerequisite
chokepoint for everything else on the network; assets carrying no CVEs at all are ranked
significant because of where they sit; a host with one of the lowest severity scores carries
the highest cost function on its network.

The corpus also shows the failure modes of the graph approaches themselves. One study uses an
undirected relaxation and loses attack directionality. Another models only single-hop
exploitation. A third notes that Bayesian attack-graph propagation does not guarantee that
patching reduces computed risk, whereas effective resistance does, and treats that guarantee as
the reason to prefer its formulation.

Component C needs a risk function `R(G)` and a per-finding contribution
`reach_delta(f) = R(G) - R(G without f)`. Three properties are non-negotiable:

- **Directed.** An edge that runs the wrong way must contribute nothing.
- **Multi-hop.** A finding that only matters as the second step of a chain must be scored for it.
- **Monotone.** `reach_delta(f) >= 0` always. If patching could increase computed risk, the
  ranking would sometimes recommend not fixing something, and the `ChainScore.reach_delta`
  field could not be a non-negative type.

The graph here is small by network standards: a web application scan yields tens of hosts and
hundreds of findings, but the state space is (asset × privilege), so a few hundred nodes.
Computational cost is not the binding constraint; provability is.

## Decision

Risk is the value-weighted maximum-probability reachability from the attacker's entry state:

```
R(G) = Σ over target nodes t:  value(t) × maxpathprob(entry → t)
```

`maxpathprob` is computed by Dijkstra over `-log p` edge weights, which is exact and runs in
`O(E log V)`. Edge probabilities are the exploitation probabilities Components A and B already
produce, `p_exploit × p_applicable`, so the graph inherits grounded numbers rather than
inventing its own.

Monotonicity follows directly. Removing a finding removes edges. Removing edges can only remove
paths, never create one, so the maximum over paths cannot rise and `reach_delta(f) >= 0` is a
property of the formulation rather than an empirical observation. `graph/monotone.py` asserts it
at runtime when configured, and a property test patches random subsets to confirm it.

Only evidence at trust tier SCANNER or better may create an edge. An adversary who controls a
referenced page cannot assert a topology into existence; rejected attempts are counted in
`AttackGraphSummary.rejected_untrusted_edges`.

Noisy-OR over enumerated paths is retained for reporting, not for ranking.

## Options Considered

### Option A: Noisy-OR over all simple paths

| Dimension | Assessment |
|---|---|
| Complexity | Moderate formulation, high implementation |
| Cost | Exponential in graph size without aggressive capping |
| Scalability | Poor |
| Team familiarity | Moderate |

**Pros:** The most faithful model of "any path will do". Multiple independent weak paths
correctly aggregate into meaningful risk, which maximum-probability paths understate.
**Cons:** Simple-path enumeration is exponential, and the practical fix, keeping the top N
paths, breaks the monotonicity guarantee: after removing an edge, a path that was outside the
top N can enter it, and a capped sum is no longer provably non-increasing. Recovering the
guarantee requires exhaustive enumeration, which is what the cap was avoiding.

### Option B: Bayesian attack-graph inference

| Dimension | Assessment |
|---|---|
| Complexity | High |
| Cost | High; inference per query |
| Scalability | Poor on cyclic graphs |
| Team familiarity | Low |

**Pros:** Principled treatment of dependence between steps, and a natural place to put priors.
**Cons:** No monotonicity guarantee, which the corpus explicitly notes. Cycles require special
handling, and privilege-implication edges create them naturally. Inference cost makes the
leave-one-finding-out contribution loop, which is the actual workload, expensive.

### Option C: Centrality measures on the graph

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | Low |
| Scalability | High |
| Team familiarity | High |

**Pros:** Cheap, well understood, and betweenness genuinely identifies chokepoints.
**Cons:** Dimensionless. Centrality cannot be combined with monetary impact into expected loss,
which is the framework's construct, so it cannot answer "how much reachable risk
does fixing this remove". It is kept as a feature, not as the risk function.

### Option D: Maximum-probability path reachability (chosen)

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | `O(E log V)` per target, once per contribution query |
| Scalability | High at web application scale |
| Team familiarity | High; it is Dijkstra |

**Pros:** Monotonicity is provable rather than tested. Directed and multi-hop by construction.
Produces a money-denominated contribution that composes with expected loss. Fast enough to
recompute per finding for a few hundred findings. The path that produced a score is a concrete
attack path an analyst can read.
**Cons:** Understates risk when many independent paths reach the same target, because only the
best one counts. Ignores dependence between steps, treating each edge probability as
independent evidence. Both are conservative directions, but they are real approximations.

## Trade-off Analysis

**Provability against fidelity.** Option A models reality better; Option D can be proved
correct in the one respect that matters for a remediation queue. The decision rests on what the
number is used for: it is subtracted from itself to produce a contribution, and that difference
must never come out negative. An approximation that is provably one-sided is worth more here
than a better point estimate with no guarantee.

**Conservatism direction.** The maximum-probability path is a lower bound on the noisy-OR value
over the same path set. The framework therefore under-states the risk of targets reachable by
many weak paths and never over-states it. Under-statement produces a missed promotion; the
alternative would produce a promotion no analyst can trace to a single attack path. The
reporting layer keeps the noisy-OR estimate so the gap is visible rather than hidden.

**Where the probabilities come from.** The graph is only as good as its edges, and that is
precisely why Component C sits downstream of A and B rather than beside them. Graph-only
approaches must invent edge probabilities from severity; here they are exploitation
probabilities under an explicit attacker, already bounded by the trust budgets.

**Cost of the contribution loop.** Naively, per-finding contribution is one full recomputation
per finding. At a few hundred findings and a few hundred nodes this is milliseconds, and the
implementation computes the base risk once and reuses the graph. Should real topologies grow,
the fallback is to restrict recomputation to findings whose edges lie on some best path, which
preserves exactness because off-path edges have zero contribution by definition.

## Consequences

**What becomes easier**

- `ChainScore.reach_delta` can be a non-negative field, and the type system carries the
  guarantee.
- Every chain score comes with a readable attack path, so an analyst can check the reasoning.
- Chain contribution is in money, so it adds directly to expected loss.
- Patch-set planning is well defined: risk after patching a set is another Dijkstra run.

**What becomes harder**

- Targets reachable by many independent weak paths are systematically under-ranked.
- Edge independence is assumed, so two findings exploiting the same underlying weakness are
  double-counted as separate steps; correlation of findings is not modelled.
- The graph depends on the scanner having observed the topology; unobserved lateral movement is
  invisible.

**What we will need to revisit**

- Whether exhaustive noisy-OR over a bounded hop count is affordable for real applications, in
  which case monotonicity can be recovered by enumerating over a fixed path set.
- Correlated findings, which the review flags as excluded from prior work and which this
  formulation also excludes.
- Whether lateral edges from observed links deserve a probability better than a constant floor.

## Action Items

1. [x] Implement maximum-probability reachability, per-finding contribution and the chokepoint
       flag.
2. [x] Enforce the trust-tier rule on edge creation and count rejected edges.
3. [x] Add the runtime monotonicity assertion and the randomised property test.
4. [ ] Measure the gap between maximum-probability and enumerated noisy-OR risk on the
       synthetic world and report it.
5. [ ] Model correlated findings so that two symptoms of one root cause are not counted as two
       independent steps.
6. [ ] Replace the constant lateral-edge probability with something estimated once real crawl
       data is available.
