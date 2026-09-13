"""Runtime verification that patching never raises risk (DESIGN.md 3.7, Gap 9).

``ChainScore.reach_delta`` is declared non-negative on the frozen contract, and the
whole of Component C's claim to being a *contribution* rather than a *score* rests on that
being true rather than merely intended. The proof is in
:mod:`vulnpriority.graph.reachability`: maximum-probability paths computed as shortest paths
over ``-log p`` can only get longer when edges are removed, and node values do not depend
on the edge set. This module is the empirical check that the implementation actually has
the property the proof describes, which is a different question from whether the
mathematics does.

Two invariants are asserted:

1. **Absolute.** ``R(G \\ S) <= R(G)`` for every patched subset ``S``. Fixing things never
   makes the estimate worse.
2. **Nested.** ``R(G \\ (S u {f})) <= R(G \\ S)``. Fixing one more thing on top of any
   patch set never makes the estimate worse either. This is the stronger statement and it
   is the one that catches an implementation that special-cases the empty set, or that
   caches a stale base risk.

A failure is a :class:`vulnpriority.core.errors.MonotonicityViolationError`, never a warning
and never a clamp, because a violation means the differencing is wrong and every money
figure downstream of it is wrong too. Comparisons carry an absolute *and* relative
tolerance: ``R`` is a sum of products of floats and re-associating that sum along a
different Dijkstra tree can legitimately move the last bits.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from vulnpriority.core.errors import MonotonicityViolationError
from vulnpriority.core.interfaces import ChainScorer

__all__ = [
    "ABS_TOLERANCE",
    "REL_TOLERANCE",
    "random_patch_subsets",
    "assert_monotone_under_patching",
]

#: Absolute slack, in the impact model's currency, allowed before a rise counts as a
#: violation. A tolerance rather than an estimate, so it was not re-denominated when the
#: shipped presets moved to rupees: ``REL_TOLERANCE`` is what scales with the numbers.
#: Leaving it at 1e-6 only makes the check stricter.
ABS_TOLERANCE = 1e-6

#: Relative slack, as a fraction of the base risk, allowed for float re-association.
REL_TOLERANCE = 1e-9


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Coerce a seed, a generator or nothing into a seeded generator.

    Every random source in the framework is seeded; a property test that cannot be
    replayed is not evidence of anything.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(0 if rng is None else int(rng))


def random_patch_subsets(
    finding_ids: Sequence[str],
    n_subsets: int,
    rng: np.random.Generator | int | None = None,
) -> list[frozenset[str]]:
    """Draw ``n_subsets`` random remediation plans over ``finding_ids``.

    Subset sizes are drawn uniformly from ``0..len(finding_ids)`` and the members
    uniformly without replacement, so the sample covers "fix nothing", "fix everything"
    and the whole range between rather than clustering around half.
    """
    generator = _rng(rng)
    ids = list(finding_ids)
    if not ids or n_subsets <= 0:
        return [frozenset()] * max(0, n_subsets)
    subsets: list[frozenset[str]] = []
    for _ in range(int(n_subsets)):
        size = int(generator.integers(0, len(ids) + 1))
        if size == 0:
            subsets.append(frozenset())
            continue
        chosen = generator.choice(len(ids), size=size, replace=False)
        subsets.append(frozenset(ids[int(index)] for index in chosen))
    return subsets


def _exceeds(higher: float, reference: float, scale: float) -> bool:
    """True when ``higher`` beats ``reference`` by more than float noise can explain."""
    slack = ABS_TOLERANCE + REL_TOLERANCE * max(abs(scale), abs(reference))
    return higher > reference + slack


def assert_monotone_under_patching(
    scorer: ChainScorer,
    scan_id: str,
    subsets: int | Iterable[Iterable[str]] = 200,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, ...]:
    """Assert that remediation never raises computed risk, and return the risks observed.

    ``subsets`` is either a number of random patch sets to draw or an explicit iterable of
    them. ``rng`` seeds the draw. The scan must already have been built on ``scorer``.

    Raises :class:`MonotonicityViolationError` on the first violation, naming the subset,
    the two risks and the size of the rise, so that a failure is debuggable rather than
    merely alarming.
    """
    base = float(scorer.total_risk_after_patching(scan_id, set()))
    if base < 0.0:
        raise MonotonicityViolationError(
            f"scan {scan_id}: unpatched risk is negative ({base!r}); R(G) is a sum of "
            "non-negative value x probability terms and cannot be"
        )

    if isinstance(subsets, int):
        ids = _finding_ids(scorer, scan_id)
        plans = random_patch_subsets(ids, subsets, rng)
    else:
        plans = [frozenset(str(item) for item in subset) for subset in subsets]

    generator = _rng(rng)
    risks: list[float] = [base]
    for plan in plans:
        risk = float(scorer.total_risk_after_patching(scan_id, set(plan)))
        if risk < -ABS_TOLERANCE:
            raise MonotonicityViolationError(
                f"scan {scan_id}: risk after patching {sorted(plan)} is negative ({risk!r})"
            )
        if _exceeds(risk, base, base):
            raise MonotonicityViolationError(
                f"scan {scan_id}: patching {len(plan)} finding(s) raised risk from "
                f"{base!r} to {risk!r} (rise {risk - base!r}); patched={sorted(plan)}"
            )
        risks.append(risk)

        # Nested check: patching one more finding on top of this plan must not help the
        # attacker either. Uses a finding outside the plan so the comparison is strict.
        remaining = [item for item in _finding_ids(scorer, scan_id) if item not in plan]
        if not remaining:
            continue
        extra = remaining[int(generator.integers(0, len(remaining)))]
        nested = float(scorer.total_risk_after_patching(scan_id, set(plan) | {extra}))
        if _exceeds(nested, risk, base):
            raise MonotonicityViolationError(
                f"scan {scan_id}: additionally patching {extra!r} raised risk from "
                f"{risk!r} to {nested!r} on top of {sorted(plan)}"
            )
        risks.append(nested)

    return tuple(risks)


def _finding_ids(scorer: ChainScorer, scan_id: str) -> tuple[str, ...]:
    """Findings the scorer knows about, preferring its own cheap accessor.

    ``ChainScorer`` does not put an id list on the interface, so a scorer that exposes one
    (as :class:`~vulnpriority.graph.chain_scorer.ReachabilityChainScorer` does) is asked
    directly and anything else falls back to scoring the scan.
    """
    accessor = getattr(scorer, "finding_ids", None)
    if callable(accessor):
        return tuple(str(item) for item in accessor(scan_id))
    return tuple(scorer.score(scan_id))
