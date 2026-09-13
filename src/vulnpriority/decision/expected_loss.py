"""The construct definition of priority (DESIGN.md 2, Gap 1).

**These two functions are the answer to Gap 1.** The literature's recurring problem is
that "priority" is never defined: papers optimise a proxy (CVSS, a severity label, an
analyst's ordering) and then evaluate against the same proxy. Here priority is a
decision-theoretic quantity with units:

    expected_loss = P(exploit | evidence, attacker) x impact

and, once Component C has measured what a finding unlocks:

    chain_adjusted_loss = expected_loss + chain_weight x reach_delta

Everything else in the framework is machinery for estimating the two factors. The
learned ranker does not replace this definition - ``expected_loss`` is retained as a
first-class baseline precisely so the learned ordering can always be compared against
the decision-theoretic one, and so a reviewer can ask "ranked higher than what, and why"
and get an answer in money.
"""

from __future__ import annotations

from vulnpriority.core.models import BusinessImpact, ExploitLikelihood

__all__ = ["expected_loss", "chain_adjusted_loss"]


def expected_loss(likelihood: ExploitLikelihood, impact: BusinessImpact) -> float:
    """Expected loss in currency units over the attacker's horizon.

    The definition of priority for Gap 1: the probability that *this* attacker exploits
    *this* finding within *their* horizon, multiplied by what it costs when they do.
    Both factors are independently auditable - ``likelihood.log_odds_terms`` names every
    contribution to the probability, ``impact`` names every contribution to the money.
    """
    return max(0.0, float(likelihood.p_exploit) * float(impact.total))


def chain_adjusted_loss(
    expected_loss: float,
    reach_delta: float,
    chain_weight: float = 1.0,
) -> float:
    """Expected loss plus what this finding unlocks for the rest of the attack graph.

    ``reach_delta`` is ``R(G) - R(G without this finding's edges)`` from Component C
    and is non-negative by construction, so this can only raise a finding's priority: a
    stepping stone with trivial direct impact is still worth fixing when it is the only
    way into the assets that matter. ``chain_weight`` (``ComponentCConfig.chain_weight``)
    is the operator's statement of how much they believe the graph.
    """
    return max(0.0, float(expected_loss) + float(chain_weight) * max(0.0, float(reach_delta)))
