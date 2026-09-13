"""``rank_scan``: from scores to a remediation queue (DESIGN.md 3.8).

The ranker returns a vector of numbers; a queue is what an engineering team can act on.
This module turns one into the other and, in doing so, carries the decision-theoretic
quantities through so that every position in the queue can be read in money as well as
in rank order:

* ``expected_loss`` - the Gap 1 construct, straight from Component B;
* ``chain_adjusted_loss`` - plus ``chain_weight x reach_delta`` from Component C,
  so a stepping stone with little direct impact still surfaces;
* ``p_exploit`` - the attacker model's probability, kept beside the learned score because
  the learned score is not one.

Ranks are assigned **per scan**, because a LambdaMART score is only meaningful inside its
query group: comparing a score from one scan against another's is a category error, and
the scan's own ordering is what a team works through. Ties break on finding id, which is
a stable hash, so two runs never disagree about an arbitrary ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from vulnprio.core.config import PipelineConfig
from vulnprio.core.interfaces import Ranker
from vulnprio.core.models import (
    ChainScore,
    EnrichedFinding,
    Explanation,
    FeatureFrame,
    ManipulationAlert,
    RankedFinding,
    RankingResult,
)
from vulnprio.core.money import DEFAULT_CURRENCY
from vulnprio.decision.expected_loss import chain_adjusted_loss
from vulnprio.rank.explain import EvidenceExplainer

__all__ = ["rank_scan", "order_within_groups"]


def order_within_groups(
    scores: np.ndarray, group_ids: Sequence[str], finding_ids: Sequence[str]
) -> dict[str, int]:
    """Rank 1..n inside every group, descending by score, ties broken on finding id.

    Returned as ``finding_id -> rank`` so the caller does not have to track row positions
    through the sort.
    """
    buckets: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        buckets.setdefault(group_id, []).append(index)

    ranks: dict[str, int] = {}
    for indices in buckets.values():
        ordered = sorted(indices, key=lambda index: (-float(scores[index]), finding_ids[index]))
        for position, index in enumerate(ordered, start=1):
            ranks[finding_ids[index]] = position
    return ranks


def rank_scan(
    frame: FeatureFrame,
    ranker: Ranker,
    enriched: Sequence[EnrichedFinding] | Mapping[str, EnrichedFinding],
    chain: Mapping[str, ChainScore] | None = None,
    config: PipelineConfig | None = None,
    explainer: object | None = None,
    alerts: Sequence[ManipulationAlert] | None = None,
    explain_evidence: bool = True,
) -> RankingResult:
    """Score a frame and assemble the ranked queue.

    ``explainer`` is anything with an
    ``explain(frame, enriched, chain) -> list[Explanation]`` method - in practice
    :class:`vulnprio.rank.explain.ShapExplainer`. It is optional because the baselines
    have no booster to attribute over and the benchmark runner explains only the model
    under study.

    When it is omitted, every ranked finding still gets an :class:`Explanation` carrying
    evidence-derived reason codes, because those depend on the finding rather than on the
    model. Pass ``explain_evidence=False`` to suppress even that - only worth doing for
    bulk policy sweeps where the explanations are never read.

    ``alerts`` are manipulation alerts raised elsewhere (typically by
    :class:`vulnprio.rank.rank_guard.RankManipulationDetector`); they are merged with any
    alerts already attached to the enriched findings and de-duplicated.
    """
    settings = config if config is not None else PipelineConfig()
    lookup = _as_lookup(enriched)
    chain_map: Mapping[str, ChainScore] = chain or {}

    scores = np.asarray(ranker.score(frame), dtype=float).reshape(-1)
    if scores.shape[0] != len(frame.finding_ids):
        raise ValueError(
            f"ranker returned {scores.shape[0]} scores for {len(frame.finding_ids)} rows"
        )

    ranks = order_within_groups(scores, frame.group_ids, frame.finding_ids)
    explanations = _explanations(
        explainer, frame, lookup, chain_map, explain_evidence, settings.currency()
    )
    alert_map = _alerts_by_finding(lookup, alerts)
    chain_weight = float(settings.component_c.chain_weight)

    items: list[RankedFinding] = []
    for index, finding_id in enumerate(frame.finding_ids):
        item = lookup.get(finding_id)
        expected = float(item.expected_loss) if item is not None else 0.0
        reach = float(chain_map[finding_id].reach_delta) if finding_id in chain_map else 0.0
        items.append(
            RankedFinding(
                finding_id=finding_id,
                scan_id=frame.group_ids[index],
                rank=ranks[finding_id],
                score=float(scores[index]),
                expected_loss=expected,
                chain_adjusted_loss=chain_adjusted_loss(expected, reach, chain_weight),
                p_exploit=float(item.likelihood.p_exploit) if item is not None else 0.0,
                explanation=explanations.get(finding_id),
                alerts=tuple(alert_map.get(finding_id, ())),
            )
        )

    # Read off the ranker rather than assumed: a ranker that does not learn reports neither
    # flag, and a learned one that quietly failed to fit reports the reason instead of
    # letting its name imply a model that was never there.
    fell_back = bool(getattr(ranker, "used_fallback", False))
    pretrained = bool(getattr(ranker, "used_pretrained", False))
    learns = bool(getattr(ranker, "requires_fit", lambda: False)())

    return RankingResult(
        ranker=ranker.name,
        flags=frame.flags,
        config_hash=settings.hash(),
        seed=int(settings.seed),
        items=tuple(items),
        model_fitted=learns and not fell_back and not pretrained,
        model_pretrained=pretrained,
        fallback_reason=str(getattr(ranker, "fallback_reason", "")) if fell_back else "",
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_lookup(
    enriched: Sequence[EnrichedFinding] | Mapping[str, EnrichedFinding],
) -> dict[str, EnrichedFinding]:
    """Accept either a sequence or an id-keyed mapping of enriched findings."""
    if isinstance(enriched, Mapping):
        return dict(enriched)
    return {item.finding_id: item for item in enriched}


def _explanations(
    explainer: object | None,
    frame: FeatureFrame,
    lookup: Mapping[str, EnrichedFinding],
    chain: Mapping[str, ChainScore],
    explain_evidence: bool,
    currency: str = DEFAULT_CURRENCY,
) -> dict[str, Explanation]:
    """Run the explainer, falling back to evidence-only reason codes when there is none.

    The fallback is the point. Without it a queue produced by a baseline, or by a
    LambdaMART that took its degenerate-input path, comes back with nothing to say - and
    that is precisely the single-scan interactive run, where "why is this here" matters
    most and where a trained model is impossible by definition.
    """
    if explainer is None:
        if not explain_evidence:
            return {}
        explainer = EvidenceExplainer(currency=currency)
    explain = getattr(explainer, "explain", None)
    if explain is None:
        raise TypeError("explainer must expose an explain(frame, enriched, chain) method")
    produced = explain(frame, lookup, chain)
    if not produced:
        return {}
    return {item.finding_id: item for item in produced}


def _alerts_by_finding(
    lookup: Mapping[str, EnrichedFinding],
    alerts: Sequence[ManipulationAlert] | None,
) -> dict[str, list[ManipulationAlert]]:
    """Merge alerts already on the findings with those passed in, preserving order."""
    merged: dict[str, list[ManipulationAlert]] = {}
    for finding_id, item in lookup.items():
        if item.alerts:
            merged[finding_id] = list(item.alerts)
    for alert in alerts or ():
        bucket = merged.setdefault(alert.finding_id, [])
        if alert not in bucket:
            bucket.append(alert)
    return merged
