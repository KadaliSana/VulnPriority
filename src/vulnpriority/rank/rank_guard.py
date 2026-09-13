"""``RankManipulationDetector``: watching the ranking itself (DESIGN.md 3.8, Gap 6).

The sandbox stops untrusted text from *instructing* the model and the influence budgets
stop it from moving a feature too far. Neither notices the attack that succeeds anyway:
many small, individually-legal nudges that together move a finding to the top or bury it
at the bottom. This detector watches the outcome rather than the inputs, and fires on four
independent conditions:

1. **Attribution share.** Component A features carry more than
   ``SandboxConfig.max_untrusted_shap_share`` of the absolute SHAP mass. The finding's
   position is being decided by what the target application and the internet said about
   it rather than by curated facts.
2. **Displacement under neutralisation.** Re-score one finding with its own Component A
   columns reset to their documented neutrals, leaving the rest of the scan exactly as it
   is, and see how far it moves - compared against how far the *other* findings in that
   same scan move. A finding that shifts far beyond its scan's own distribution was, in
   effect, ranked by the semantic assessment alone. This is a counterfactual, not a
   correlation: it is exactly the attack's own metric.
3. **Contradiction of trusted evidence.** Either the assessment asserts the finding does
   not apply while curated feeds say the CVE is being exploited *and* the observed version
   matches, or the trust ledger recorded that it had to hold ``p_exploit`` up against an
   untrusted argument. Untrusted text is allowed to add detail; it is not allowed to
   overrule CISA.
4. **Canary leak.** The sandbox's canary reached model output, so the prompt boundary was
   crossed and everything downstream of it is suspect.

**The floor check reads the ledger; it does not re-derive one.** ``TrustSummary``
deliberately reports the *nominal* floor the curated evidence implies, while
``TrustLedger.apply_floor`` enforces ``min(nominal floor, trusted reference)`` - the floor
exists to undo an untrusted argument, never to raise a finding above what the feeds and
the attacker model already justified. An opportunistic attacker on a short horizon can
legitimately score a KEV-listed finding below the nominal floor with no untrusted text
involved anywhere, so comparing ``p_exploit`` against ``TrustSummary.floor_p_exploit``
flags a large fraction of an ordinary run and every one of those alerts is wrong. The
component that enforces the property is the one that knows whether it was breached, and it
says so in ``TrustSummary.conflicts``; this detector keys off that record.

**Alert text is written for an operator.** ``ManipulationAlert.message`` is rendered
verbatim in the assessment report, so it says what happened in plain English and keeps
every number. The precise internal form - feature names, tiers, thresholds - is available
from :meth:`RankManipulationDetector.detect_detailed` for the research view.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vulnpriority.core.config import PipelineConfig, SandboxConfig
from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    Component,
    DetectorName,
    ExploitMaturity,
    TrustTier,
    VersionMatch,
)
from vulnpriority.core.interfaces import ManipulationDetector, Ranker
from vulnpriority.core.models import (
    FEATURE_GROUPS,
    EnrichedFinding,
    Explanation,
    FeatureFrame,
    ManipulationAlert,
)
from vulnpriority.rank.compose import order_within_groups
from vulnpriority.rank.features import NEUTRAL, usable_intel

__all__ = [
    "DEFAULT_DISPLACEMENT_MAD_MULTIPLIER",
    "DEFAULT_MIN_RANK_DISPLACEMENT",
    "MAD_TO_SIGMA",
    "NOT_APPLICABLE_P_THRESHOLD",
    "FLOOR_CONFLICT_MARKER",
    "EVIDENCE_SOURCE_PHRASE",
    "ordinal",
    "neutralise_component_a",
    "RankManipulationDetector",
]

_LOG = logging.getLogger(__name__)

#: Multiple of the scaled median absolute deviation past a scan's median displacement at
#: which a finding is called anomalous. Three is the conventional robust outlier cut,
#: roughly three sigma for a normal sample, and it is a multiple of a *robust* dispersion
#: precisely because these distributions are small, skewed and full of the outliers being
#: looked for - a mean and a standard deviation would be dragged by them.
#:
#: ``RankingConfig`` has no field for this yet; when one is added it is read from there
#: automatically (see :meth:`RankManipulationDetector.__init__`).
DEFAULT_DISPLACEMENT_MAD_MULTIPLIER: float = 3.0

#: Converts a median absolute deviation into a standard-deviation-equivalent for a normal
#: sample, which is what makes the multiplier above readable as "about three sigma".
MAD_TO_SIGMA: float = 1.4826

#: Displacement no finding is flagged below, however quiet its scan. Without it, a scan
#: where nothing moves has a median and a deviation of zero, and a finding that shifted two
#: places would be a towering outlier. Aligned with ``AdversarialExpectation.max_rank_shift``
#: (2) plus one: below three places, nobody is looking at a different queue.
DEFAULT_MIN_RANK_DISPLACEMENT: int = 3

#: ``p_applicable`` at or below which an assessment is read as asserting "not applicable"
#: even when the verdict field says ``UNCERTAIN``.
NOT_APPLICABLE_P_THRESHOLD: float = 0.2

#: Substring that identifies a floor conflict in ``TrustSummary.conflicts``.
#:
#: ``TrustLedger`` writes exactly two kinds of conflict: an influence-budget cap, whose
#: text names a tier, a feature and a budget, and the floor record written by
#: ``apply_floor`` when it actually held a probability up. Only the second mentions a
#: floor, and it is written only on a real breach - which is precisely why this detector
#: reads it instead of re-deriving the comparison from the nominal floor.
FLOOR_CONFLICT_MARKER: str = "floor"

#: How each trust tier is described to an operator. The report prints these, so they name
#: the kind of source a person can go and look at rather than the framework's own tiers.
EVIDENCE_SOURCE_PHRASE: dict[TrustTier, str] = {
    TrustTier.OPERATOR: "your own configuration",
    TrustTier.CURATED_FEED: "the vulnerability catalogues",
    TrustTier.SCANNER: "the scanner's own output",
    TrustTier.REFERENCE_PAGE: "an advisory or blog page",
    TrustTier.TARGET_CONTENT: "text the scanned application itself returned",
}

#: One alert and the technical note that belongs with it, as returned by
#: :meth:`RankManipulationDetector.detect_detailed`.
_Detailed = tuple[ManipulationAlert, str]


def ordinal(value: int) -> str:
    """``1`` to ``"1st"``, ``4`` to ``"4th"``, ``11`` to ``"11th"`` - for queue positions."""
    number = int(value)
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def neutralise_component_a(frame: FeatureFrame) -> FeatureFrame:
    """A copy of ``frame`` with every Component A column reset to its documented neutral.

    The counterfactual behind detector 2. Columns are reset rather than dropped so the
    frame still matches the model's fitted feature list and the two scores stay
    comparable; the ablation drops columns, this does not, and the difference is
    deliberate.
    """
    X = frame.X.copy()
    for name in X.columns:
        if FEATURE_GROUPS.get(name) is Component.A:
            X[name] = float(NEUTRAL[name])
    return FeatureFrame(
        X=X,
        finding_ids=list(frame.finding_ids),
        group_ids=list(frame.group_ids),
        flags=frame.flags,
    )


class RankManipulationDetector(ManipulationDetector):
    """Post-ranking manipulation detection over explanations and counterfactual re-scoring."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        mad_multiplier: float | None = None,
        min_rank_displacement: int = DEFAULT_MIN_RANK_DISPLACEMENT,
    ) -> None:
        """``config`` supplies ``SandboxConfig.max_untrusted_shap_share``.

        ``mad_multiplier`` defaults to ``RankingConfig.displacement_mad_multiplier`` when
        that field exists and to :data:`DEFAULT_DISPLACEMENT_MAD_MULTIPLIER` otherwise, so
        the detector picks the setting up the moment it is added to the frozen config
        without a second change here.
        """
        self.config: PipelineConfig = config if config is not None else PipelineConfig()
        self.mad_multiplier = float(
            mad_multiplier
            if mad_multiplier is not None
            else getattr(
                self.config.ranking,
                "displacement_mad_multiplier",
                DEFAULT_DISPLACEMENT_MAD_MULTIPLIER,
            )
        )
        self.min_rank_displacement = int(min_rank_displacement)

    @property
    def sandbox(self) -> SandboxConfig:
        """Sandbox settings, which own the untrusted attribution share limit."""
        return self.config.sandbox

    # -- main entry point ---------------------------------------------------

    def detect(
        self,
        enriched: list[EnrichedFinding],
        context: dict[str, Any] | None = None,
    ) -> list[ManipulationAlert]:
        """Return every alert raised for these findings, in a deterministic order.

        Recognised ``context`` keys, all optional - each enables the detectors that need
        it, and detectors with no data stay silent rather than guessing:

        ``explanations``
            ``Sequence[Explanation]`` or ``{finding_id: Explanation}``, for detector 1.
        ``frame``
            the :class:`FeatureFrame` the ranking was produced from, for detector 2.
        ``ranker``
            the fitted :class:`Ranker` to re-score with, for detector 2.
        """
        return [alert for alert, _ in self.detect_detailed(enriched, context)]

    def detect_detailed(
        self,
        enriched: Sequence[EnrichedFinding],
        context: dict[str, Any] | None = None,
    ) -> list[_Detailed]:
        """Every alert paired with the technical note behind it.

        The alert's own ``message`` is the operator-facing sentence the report prints; the
        second element of each pair is the precise internal form - feature names, tiers,
        thresholds and raw ledger text - for the research view and for debugging.
        """
        ctx = context or {}
        explanations = self._as_lookup(ctx.get("explanations"))
        displacement = self._displacements(ctx.get("frame"), ctx.get("ranker"))

        found: list[_Detailed] = []
        for item in enriched:
            found.extend(self._canary_alerts(item))
            found.extend(self._share_alerts(item, explanations.get(item.finding_id)))
            found.extend(self._displacement_alerts(item, displacement.get(item.finding_id)))
            found.extend(self._contradiction_alerts(item))
        return found

    # -- detector 4: canary -------------------------------------------------

    @staticmethod
    def _canary_alerts(item: EnrichedFinding) -> list[_Detailed]:
        """A canary in model output means the prompt boundary was crossed."""
        on_summary = item.trust.canary_leaked
        audits = (item.asset.audit, item.exploitability.audit, item.applicability.audit)
        on_audit = any(audit is not None and audit.canary_leaked for audit in audits)
        if not (on_summary or on_audit):
            return []
        alert = ManipulationAlert(
            finding_id=item.finding_id,
            detector=DetectorName.CANARY,
            severity=1.0,
            message=(
                "A hidden marker placed in the analysis request came back in the reply, so "
                "text from outside reached the automated assessment. Treat everything this "
                "assessment says about this finding as unreliable."
            ),
        )
        note = (
            f"canary leak: trust.canary_leaked={on_summary}, any LLMAudit.canary_leaked="
            f"{on_audit}; max tier used {item.trust.max_tier_used.name}"
        )
        return [(alert, note)]

    # -- detector 1: untrusted attribution share ----------------------------

    def _share_alerts(
        self, item: EnrichedFinding, explanation: Explanation | None
    ) -> list[_Detailed]:
        """Component A attributions dominating the score beyond the configured share."""
        if explanation is None:
            return []
        limit = float(self.sandbox.max_untrusted_shap_share)
        share = float(explanation.untrusted_influence_share)
        if share <= limit:
            return []
        source = self._source_phrase(item)
        alert = ManipulationAlert(
            finding_id=item.finding_id,
            detector=DetectorName.INFLUENCE_BUDGET,
            severity=min(1.0, share),
            message=(
                f"This finding's position in the queue is {share:.0%} driven by the "
                f"automated reading of {source}, above the {limit:.0%} share allowed for "
                "evidence of that kind."
            ),
        )
        note = (
            f"untrusted SHAP share {share:.4f} exceeds max_untrusted_shap_share {limit:.4f}; "
            f"attribution mass over Component A features; max tier used "
            f"{item.trust.max_tier_used.name}"
        )
        return [(alert, note)]

    # -- detector 2: displacement under neutralisation ----------------------

    def _displacements(
        self, frame: object | None, ranker: object | None
    ) -> dict[str, tuple[int, int, int]]:
        """``finding_id -> (actual rank, counterfactual rank, delta)`` or an empty map.

        The counterfactual is **per finding**: this finding's Component A columns are
        neutralised while every other finding in the scan keeps the evidence it really
        has. That answers the question the detector is actually asking - "if the semantic
        assessment had said nothing about *this* finding, where would it sit among its
        peers?" - and it is the only form that attributes the movement to the finding's
        own untrusted evidence.

        Neutralising the whole column at once instead answers "what if Component A did not
        exist", which is a different question and a much noisier one: every finding then
        moves, including findings whose own assessment contributed nothing, purely because
        the rest of the queue collapsed around them. On a real run that reported six out of
        ten findings as manipulated.

        The per-finding form costs nothing extra. A tree ensemble scores each row
        independently, so neutralising one finding's columns changes only that one row's
        score: the single all-neutralised pass already contains every counterfactual score
        needed, and the ranks are recovered by counting comparisons.

        ``delta`` is ``counterfactual - actual``: positive means the semantic assessment
        moved the finding up the queue, negative means it moved it down.
        """
        if not isinstance(frame, FeatureFrame) or not isinstance(ranker, Ranker):
            return {}
        if not any(FEATURE_GROUPS.get(name) is Component.A for name in frame.feature_names):
            return {}
        try:
            actual = np.asarray(ranker.score(frame), dtype=float)
            without_a = np.asarray(ranker.score(neutralise_component_a(frame)), dtype=float)
        except Exception as error:  # pragma: no cover - defensive: a guard must not crash a run
            _LOG.warning("displacement detector could not re-score the frame: %s", error)
            return {}

        actual_ranks = order_within_groups(actual, frame.group_ids, frame.finding_ids)
        movements: dict[str, tuple[int, int, int, float]] = {}
        for rows in self._group_rows(frame.group_ids).values():
            peers = [(-float(actual[index]), frame.finding_ids[index]) for index in rows]
            deltas: dict[str, tuple[int, int, int]] = {}
            for index in rows:
                finding_id = frame.finding_ids[index]
                key = (-float(without_a[index]), finding_id)
                ahead = sum(
                    1
                    for position, peer in zip(rows, peers)
                    if position != index and peer < key
                )
                counterfactual = ahead + 1
                deltas[finding_id] = (
                    actual_ranks[finding_id],
                    counterfactual,
                    counterfactual - actual_ranks[finding_id],
                )
            threshold = self._scan_threshold([abs(entry[2]) for entry in deltas.values()])
            movements.update(
                {finding_id: (*entry, threshold) for finding_id, entry in deltas.items()}
            )
        return movements

    def _scan_threshold(self, displacements: Sequence[int]) -> float:
        """The displacement a finding must exceed to be anomalous *within its own scan*.

        A constant cannot do this job. Queues here run from twenty findings to sixty, and a
        fixed three places encodes the assumption that the semantic assessment should
        barely matter - which contradicts the thesis the framework is built on. On a real
        run the median displacement among flagged findings was eight, meaning the threshold
        was measuring the feature working rather than anything going wrong.

        So the cut is relative: ``median + multiplier x 1.4826 x MAD``, floored at
        :attr:`min_rank_displacement`. Median and MAD rather than mean and standard
        deviation because the sample is small, skewed, and contains the very outliers being
        searched for - a mean would be dragged towards them and hide them. A scan where
        everything moves eight places flags nobody; a scan with a median of three and one
        finding at thirty flags that one.
        """
        if not displacements:
            return float(self.min_rank_displacement)
        values = np.asarray(displacements, dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        cut = median + self.mad_multiplier * MAD_TO_SIGMA * mad
        return max(float(self.min_rank_displacement), cut)

    @staticmethod
    def _group_rows(group_ids: Sequence[str]) -> dict[str, list[int]]:
        """Row indices per query group, in row order."""
        rows: dict[str, list[int]] = {}
        for index, group_id in enumerate(group_ids):
            rows.setdefault(group_id, []).append(index)
        return rows

    def _displacement_alerts(
        self, item: EnrichedFinding, movement: tuple[int, int, int, float] | None
    ) -> list[_Detailed]:
        """Fire when this finding moved further than its own scan's findings typically do."""
        if movement is None:
            return []
        actual, neutral, delta, threshold = movement
        if abs(delta) <= threshold:
            return []
        direction = "up" if delta > 0 else "down"
        source = self._source_phrase(item)
        alert = ManipulationAlert(
            finding_id=item.finding_id,
            detector=DetectorName.DISPLACEMENT,
            severity=min(1.0, abs(delta) / max(1.0, 2.0 * threshold)),
            message=(
                f"Reading {source} moved this finding {abs(delta)} places {direction} the "
                f"queue, much further than anything else in this scan; without it the "
                f"finding would rank {ordinal(neutral)} rather than {ordinal(actual)}."
            ),
            rank_delta=int(delta),
        )
        note = (
            f"Component A displacement: rank {actual} with Component A, {neutral} with it "
            f"neutralised, delta {delta:+d}, scan threshold {threshold:.2f} "
            f"(median + {self.mad_multiplier} x scaled MAD, floor {self.min_rank_displacement})"
        )
        return [(alert, note)]

    # -- detector 3: contradiction of trusted evidence ----------------------

    def _contradiction_alerts(self, item: EnrichedFinding) -> list[_Detailed]:
        """Assessments that overrule curated feeds rather than adding detail to them."""
        found: list[_Detailed] = []
        applicability = item.applicability
        denies = (
            applicability.verdict == ApplicabilityVerdict.NOT_APPLICABLE
            or applicability.p_applicable <= NOT_APPLICABLE_P_THRESHOLD
        )
        kev, verified = self._tier1_evidence(item)

        if denies and (kev or verified) and applicability.version_match == VersionMatch.MATCH:
            evidence = (
                "the catalogue of known exploited vulnerabilities lists it as actively exploited"
                if kev
                else "working exploit code for it is published in a curated index"
            )
            found.append(
                (
                    ManipulationAlert(
                        finding_id=item.finding_id,
                        detector=DetectorName.CONSISTENCY,
                        severity=0.9,
                        message=(
                            "The automated assessment concluded this vulnerability does not "
                            f"apply here (it put the chance at {applicability.p_applicable:.0%}), "
                            f"but {evidence} and the software version seen on the target "
                            "matches the affected range."
                        ),
                    ),
                    (
                        f"applicability verdict {applicability.verdict.value} with "
                        f"p_applicable={applicability.p_applicable:.3f} contradicts tier<=1 "
                        f"evidence (kev={kev}, verified_functional={verified}) while "
                        f"version_match={applicability.version_match.value}"
                    ),
                )
            )

        found.extend(self._floor_alerts(item))
        return found

    def _floor_alerts(self, item: EnrichedFinding) -> list[_Detailed]:
        """Fire only when the trust ledger recorded that it actually held a probability up.

        This is a read of ``TrustSummary.conflicts``, not a fresh comparison. See the
        module docstring: ``TrustSummary.floor_p_exploit`` is the *nominal* floor, and a
        finding whose attacker-model probability sits below it with no untrusted movement
        anywhere has breached nothing.
        """
        records = [
            conflict
            for conflict in item.trust.conflicts
            if FLOOR_CONFLICT_MARKER in conflict.lower()
        ]
        if not records:
            return []
        source = self._source_phrase(item)
        alert = ManipulationAlert(
            finding_id=item.finding_id,
            detector=DetectorName.CONSISTENCY,
            severity=0.8,
            message=(
                f"Evidence read from {source} argued this finding's chance of being "
                f"exploited downwards; it was held at {item.likelihood.p_exploit:.0%}, the "
                "level the vulnerability catalogues already support."
            ),
        )
        return [(alert, "; ".join(records))]

    @staticmethod
    def _tier1_evidence(item: EnrichedFinding) -> tuple[bool, bool]:
        """``(in KEV, verified functional exploit)`` from curated feeds as of the cut-off."""
        kev = False
        verified = False
        for record in usable_intel(item):
            entry = record.kev
            if entry is not None and entry.in_kev:
                if entry.date_added is None or entry.date_added <= item.as_of:
                    kev = True
            for exploit in record.exploits:
                if exploit.published is not None and exploit.published > item.as_of:
                    continue
                if exploit.verified and exploit.maturity >= ExploitMaturity.FUNCTIONAL:
                    verified = True
        return kev, verified

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _source_phrase(item: EnrichedFinding) -> str:
        """Plain-English name for the least trusted source that touched this finding."""
        return EVIDENCE_SOURCE_PHRASE.get(
            item.trust.max_tier_used, EVIDENCE_SOURCE_PHRASE[TrustTier.TARGET_CONTENT]
        )

    @staticmethod
    def _as_lookup(
        explanations: Sequence[Explanation] | Mapping[str, Explanation] | None,
    ) -> dict[str, Explanation]:
        """Accept a sequence or an id-keyed mapping of explanations."""
        if explanations is None:
            return {}
        if isinstance(explanations, Mapping):
            return dict(explanations)
        return {item.finding_id: item for item in explanations}
