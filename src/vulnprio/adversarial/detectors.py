"""Two detectors that bracket the model call (DESIGN.md 3.10, ADR-002).

ADR-002 separates the three mechanisms: *filters prevent, budgets bound, guards detect*. The
sandbox owns prevention and bounding. This module owns detection, which is the part that has
to produce a number - an alert on a finding, a rate in a report - because a control that
cannot be measured cannot be shown to have regressed.

:class:`PreLLMPatternDetector` runs before the model and asks whether the text about to be
read is trying to give orders. It is a thin, deliberate wrapper over
:class:`~vulnprio.sandbox.pipeline.Sandbox`: the sandbox stays the single source of truth for
what counts as a signal, and this class adds only what a *detector* needs and a *sanitizer*
does not - a bounded score and :class:`~vulnprio.core.models.ManipulationAlert` emission.

Going through ``Sandbox.sanitize`` rather than straight to the pattern filter matters more
than it looks. The sandbox reports three kinds of evidence, and only one of them is a pattern
hit on normalised text: it also matches patterns against the *raw* text, because normalisation
legitimately destroys evidence (a forged ``</untrusted>`` delimiter is stripped as markup), and
it emits synthetic ``nz_*`` signals for content it removed before the filter could read it
(hidden elements, HTML comments, zero-width characters, folded homoglyphs, elided blobs).
Counting only normalised pattern hits measured the defence as weaker than it is: it scored
hidden-text payloads at 0.29 when the sandbox was in fact neutralising all of them.

:class:`OutputAnomalyDetector` runs after the model and asks a different question, the one no
pattern can answer: is the assessment consistent with what tier <= 1 evidence already says?
An injection that survives every filter still has to produce an *answer*, and the answers it
wants - a KEV-listed CVE judged not applicable, feasibility jumping half the scale away from
the heuristic, certainty with nothing quotable behind it - are exactly the answers curated
evidence contradicts.

Neither detector may fire on ordinary advisory prose. The corpus's benign controls are the
test of that, and the false-positive rate sits next to the detection rate in every report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from vulnprio.core.config import SandboxConfig
from vulnprio.core.enums import (
    ApplicabilityVerdict,
    DetectorName,
    ExploitMaturity,
    InjectionCategory,
    InjectionVerdict,
    TrustTier,
    VersionMatch,
    tier_of,
)
from vulnprio.core.interfaces import ManipulationDetector
from vulnprio.core.models import (
    AdversarialCase,
    ApplicabilityAssessment,
    AssetCriticality,
    EnrichedFinding,
    ExploitabilityAssessment,
    ExploitLikelihood,
    InjectionSignal,
    ManipulationAlert,
    UntrustedText,
    VulnIntel,
)
from vulnprio.sandbox.instruction_filter import InstructionFilter
from vulnprio.sandbox.normalize import normalize_untrusted
from vulnprio.sandbox.output_guard import span_matches
from vulnprio.sandbox.pipeline import Sandbox

__all__ = [
    "DEFAULT_MAX_FEATURE_JUMP",
    "DEFAULT_HIGH_CONFIDENCE",
    "KEV_FLOOR",
    "KEV_RANSOMWARE_FLOOR",
    "VERIFIED_FUNCTIONAL_FLOOR",
    "DocumentScore",
    "TrustedEvidence",
    "AssessmentSnapshot",
    "CoverageReport",
    "PreLLMPatternDetector",
    "OutputAnomalyDetector",
    "measure_pattern_coverage",
]

#: How far an untrusted-evidence path may move a normalised feature away from the
#: deterministic baseline before the movement is itself the evidence. Set at the
#: reference-page influence budget (0.35): a legitimate advisory is allowed to move a feature
#: that far, so anything beyond it has escaped the budget and is a bug or an attack.
DEFAULT_MAX_FEATURE_JUMP: float = 0.35

#: Confidence above which a claim must be able to quote its evidence.
DEFAULT_HIGH_CONFIDENCE: float = 0.85

#: Maturity a curated feed's exploit evidence must reach before an assessment calling the
#: same CVE unproven counts as a contradiction rather than a difference of opinion.
_CONTRADICTORY_MATURITY_FLOOR = ExploitMaturity.FUNCTIONAL

#: Pattern-id prefix the sandbox uses for signals raised by normalisation rather than by the
#: published pattern library (``nz_hidden_element``, ``nz_elided_blob`` and friends).
NORMALISATION_PREFIX = "nz_"

#: p_exploit floors implied by tier <= 1 evidence. These mirror ``vulnprio.enrich.trust`` but
#: are re-derived locally: a detector that imported the enricher's constants would stop being
#: an independent check on the enricher.
KEV_FLOOR: float = 0.50
KEV_RANSOMWARE_FLOOR: float = 0.70
VERIFIED_FUNCTIONAL_FLOOR: float = 0.35


# ---------------------------------------------------------------------------
# Pre-model detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentScore:
    """What the pre-model detector found in one document."""

    tier: TrustTier
    original_length: int
    sanitized_length: int
    signals: tuple[InjectionSignal, ...] = ()
    verdict: InjectionVerdict = InjectionVerdict.CLEAN
    zero_width_removed: int = 0
    homoglyphs_folded: int = 0
    hidden_elements_removed: int = 0
    html_comments_removed: int = 0
    hidden_text_removed: int = 0
    blobs_elided: int = 0
    truncated: bool = False
    sanitized_text: str = ""

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    @property
    def normalisation_signals(self) -> tuple[InjectionSignal, ...]:
        """Signals the sandbox raised for content it removed before the filter could read it.

        These are the ``nz_*`` synthetic signals. Separated out because "the payload was
        neutralised by normalisation" and "the payload matched a published pattern" are
        different facts about the defence, even though both mean the attempt was seen.
        """
        return tuple(
            signal for signal in self.signals if signal.pattern_id.startswith(NORMALISATION_PREFIX)
        )

    @property
    def pattern_signals(self) -> tuple[InjectionSignal, ...]:
        """Signals that came from the published pattern library."""
        return tuple(
            signal
            for signal in self.signals
            if not signal.pattern_id.startswith(NORMALISATION_PREFIX)
        )

    @property
    def categories(self) -> set[InjectionCategory]:
        """Distinct injection categories the patterns attributed to this document."""
        return {signal.category for signal in self.signals}

    @property
    def pattern_ids(self) -> tuple[str, ...]:
        """Pattern ids that fired, in document order, without repeats."""
        seen: list[str] = []
        for signal in self.signals:
            if signal.pattern_id not in seen:
                seen.append(signal.pattern_id)
        return tuple(seen)

    @property
    def obfuscation_count(self) -> int:
        """Evidence of deliberate hiding that normalisation removed before the filter ran.

        Zero-width characters, folded homoglyphs, hidden elements, HTML comments and elided
        blobs all mean the document was shaped to be read differently by a human and by a
        model. None of them is proof of an injection on its own, which is why this only
        contributes to the score and never sets the verdict by itself.
        """
        return (
            self.zero_width_removed
            + self.homoglyphs_folded
            + self.hidden_elements_removed
            + self.html_comments_removed
            + self.blobs_elided
        )

    @property
    def detected(self) -> bool:
        """True when at least one instruction pattern fired."""
        return bool(self.signals)

    @property
    def score(self) -> float:
        """Bounded suspicion score in ``[0, 1]``.

        Saturating rather than linear: the difference between one signal and two matters,
        the difference between nine and ten does not. Obfuscation evidence is worth a
        quarter of a signal each so that a page hiding three paragraphs scores above a clean
        one without ever reaching the threshold on hiding alone.
        """
        weighted = float(self.signal_count) + 0.25 * float(self.obfuscation_count)
        if weighted <= 0.0:
            return 0.0
        return min(1.0, 1.0 - math.exp(-weighted / 2.0))


class PreLLMPatternDetector(ManipulationDetector):
    """Scores untrusted documents before they reach a model.

    Delegates to :class:`~vulnprio.sandbox.pipeline.Sandbox` so that what counts as a signal
    is defined in exactly one place. The sandbox (and therefore its pattern library) is
    injectable, which is how a run pins a specific library revision and how the adversarial
    evaluation measures one revision against the corpus.
    """

    name = DetectorName.PRE_LLM_PATTERN

    def __init__(
        self,
        config: SandboxConfig | None = None,
        *,
        sandbox: Sandbox | None = None,
        instruction_filter: InstructionFilter | None = None,
    ) -> None:
        """``config`` supplies the length cap and the verdict thresholds.

        ``sandbox`` wins over ``instruction_filter``; supplying either pins the revision the
        measurement is taken against.
        """
        self.config = config or SandboxConfig()
        if sandbox is not None:
            self.sandbox = sandbox
        else:
            self.sandbox = Sandbox(self.config, instruction_filter=instruction_filter)

    @property
    def filter(self) -> InstructionFilter:
        """The pattern library the sandbox is using."""
        return self.sandbox.filter

    @property
    def pattern_library_version(self) -> str:
        """Version string of the loaded pattern library, for the report."""
        return self.sandbox.filter.version

    def __len__(self) -> int:
        return len(self.sandbox.filter)

    # -- scoring -----------------------------------------------------------

    def score_document(
        self, text: str, tier: TrustTier = TrustTier.REFERENCE_PAGE, *, nonce: str = "adv"
    ) -> DocumentScore:
        """Sanitize ``text`` exactly as the pipeline would and collect every signal raised.

        All three evidence channels count, because all three mean the attempt was seen:
        pattern hits on the normalised text, pattern hits on the raw text that normalisation
        would have destroyed, and the synthetic ``nz_*`` signals for content normalisation
        removed. ``normalize_untrusted`` is called a second time only to recover the
        per-stage counts, which the :class:`SanitizationReport` aggregates.
        """
        tier = TrustTier(tier)
        sanitized, report = self.sandbox.sanitize(text or "", tier, nonce)
        _, counts = normalize_untrusted(text or "", self.config.max_chars_per_segment)
        return DocumentScore(
            tier=tier,
            original_length=report.original_length,
            sanitized_length=report.sanitized_length,
            signals=report.signals,
            verdict=report.verdict,
            zero_width_removed=counts["zero_width_removed"] + counts["bidi_removed"],
            homoglyphs_folded=report.homoglyphs_folded,
            hidden_elements_removed=counts["hidden_elements_removed"],
            html_comments_removed=counts["html_comments_removed"],
            hidden_text_removed=report.hidden_text_removed,
            blobs_elided=report.base64_blobs_elided,
            truncated=report.truncated,
            sanitized_text=sanitized,
        )

    def score_untrusted(self, item: UntrustedText) -> DocumentScore:
        """Score an :class:`UntrustedText`, taking its tier from its provenance."""
        return self.score_document(item.text, item.tier)

    # -- alerts ------------------------------------------------------------

    def alerts_for(
        self,
        finding_id: str,
        text: str,
        tier: TrustTier = TrustTier.REFERENCE_PAGE,
        *,
        location: str = "",
    ) -> list[ManipulationAlert]:
        """One alert per document that fired, carrying the categories and pattern ids."""
        score = self.score_document(text, tier)
        if not score.detected:
            return []
        categories = ", ".join(sorted(category.value for category in score.categories))
        where = f" at {location}" if location else ""
        message = (
            f"{score.signal_count} injection signal(s){where} in tier-{int(score.tier)} content "
            f"[{categories}]; patterns {', '.join(score.pattern_ids)}"
        )
        if score.normalisation_signals and not score.pattern_signals:
            message += " (neutralised by normalisation before the filter ran)"
        return [
            ManipulationAlert(
                finding_id=finding_id,
                detector=DetectorName.PRE_LLM_PATTERN,
                severity=score.score,
                message=message[:400],
                rank_delta=None,
            )
        ]

    def detect(
        self, enriched: list[EnrichedFinding], context: Mapping[str, Any] | None = None
    ) -> list[ManipulationAlert]:
        """Scan every untrusted surface reachable from each enriched finding.

        Implements :class:`~vulnprio.core.interfaces.ManipulationDetector`, so the pipeline
        can run this alongside the rank guard without knowing what it wraps.
        """
        alerts: list[ManipulationAlert] = []
        for item in enriched:
            for location, document in self._surfaces(item):
                alerts.extend(
                    self.alerts_for(
                        item.finding_id, document.text, document.tier, location=location
                    )
                )
        return alerts

    @staticmethod
    def _surfaces(item: EnrichedFinding) -> list[tuple[str, UntrustedText]]:
        """Every untrusted document an enriched finding exposes, with a location label."""
        surfaces: list[tuple[str, UntrustedText]] = [("finding.description", item.finding.description)]
        for index, evidence in enumerate(item.finding.evidence):
            surfaces.append((f"finding.evidence[{index}]", evidence))
        if item.endpoint.response_sample is not None:
            surfaces.append(("endpoint.response_sample", item.endpoint.response_sample))
        for record in item.intel:
            if record.description is not None:
                surfaces.append((f"{record.cve_id}.description", record.description))
            for index, document in enumerate(record.references):
                surfaces.append((f"{record.cve_id}.references[{index}]", document.content))
            for index, exploit in enumerate(record.exploits):
                if exploit.title is not None:
                    surfaces.append((f"{record.cve_id}.exploits[{index}].title", exploit.title))
        return surfaces


# ---------------------------------------------------------------------------
# Post-model detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustedEvidence:
    """The tier <= 1 facts an assessment is not permitted to contradict.

    Built only from curated feeds and only from records dated at or before ``as_of``, so a
    contradiction alert can never be produced by temporal leakage.
    """

    in_kev: bool = False
    known_ransomware_use: bool = False
    verified_exploit_maturity: ExploitMaturity = ExploitMaturity.UNKNOWN
    version_match: VersionMatch = VersionMatch.UNKNOWN
    epss: float | None = None
    cve_ids: tuple[str, ...] = ()

    @property
    def floor_p_exploit(self) -> float:
        """Lower bound on ``p_exploit`` implied by this evidence."""
        floor = 0.0
        if self.in_kev:
            floor = max(floor, KEV_RANSOMWARE_FLOOR if self.known_ransomware_use else KEV_FLOOR)
        if self.verified_exploit_maturity >= _CONTRADICTORY_MATURITY_FLOOR:
            floor = max(floor, VERIFIED_FUNCTIONAL_FLOOR)
        return floor

    @property
    def is_empty(self) -> bool:
        """True when the curated feeds said nothing, so there is nothing to contradict."""
        return (
            not self.in_kev
            and self.verified_exploit_maturity < _CONTRADICTORY_MATURITY_FLOOR
            and self.version_match == VersionMatch.UNKNOWN
        )

    @classmethod
    def from_intel(
        cls,
        intel: Sequence[VulnIntel],
        as_of: date | None = None,
        *,
        version_match: VersionMatch = VersionMatch.UNKNOWN,
    ) -> "TrustedEvidence":
        """Collapse a finding's intel records into the facts a detector may rely on.

        ``version_match`` comes from deterministic CPE matching (``semantic.cpe_match``),
        which is tier <= 1 by construction, and is passed in rather than inferred here so
        this module stays independent of Component A.
        """
        in_kev = False
        ransomware = False
        maturity = ExploitMaturity.UNKNOWN
        epss: float | None = None
        cve_ids: list[str] = []
        for record in intel:
            if as_of is not None and record.as_of > as_of:
                continue
            cve_ids.append(record.cve_id)
            kev = record.kev
            listed_in_time = (
                kev is not None
                and kev.in_kev
                and not (kev.date_added is not None and as_of is not None and kev.date_added > as_of)
            )
            if listed_in_time and kev is not None:
                in_kev = True
                ransomware = ransomware or kev.known_ransomware_use
            if record.epss is not None and (as_of is None or record.epss.as_of <= as_of):
                epss = record.epss.score if epss is None else max(epss, record.epss.score)
            for exploit in record.exploits:
                if not exploit.verified:
                    continue
                if exploit.published is not None and as_of is not None and exploit.published > as_of:
                    continue
                if exploit.maturity > maturity:
                    maturity = exploit.maturity
        return cls(
            in_kev=in_kev,
            known_ransomware_use=ransomware,
            verified_exploit_maturity=maturity,
            version_match=version_match,
            epss=epss,
            cve_ids=tuple(sorted(set(cve_ids))),
        )


@dataclass(frozen=True)
class AssessmentSnapshot:
    """The part of an assessment a post-model detector needs, for one finding."""

    finding_id: str
    exploit_feasibility: float = 0.0
    asset_criticality: float = 0.0
    p_exploit: float = 0.0
    p_applicable: float = 0.5
    verdict: ApplicabilityVerdict = ApplicabilityVerdict.UNCERTAIN
    version_match: VersionMatch = VersionMatch.UNKNOWN
    exploit_maturity: ExploitMaturity = ExploitMaturity.UNKNOWN
    confidence: float = 0.5
    evidence_spans: tuple[str, ...] = ()
    canary_leaked: bool = False
    envelope_broken: bool = False
    injection_signals: int = 0
    max_tier_used: TrustTier = TrustTier.OPERATOR

    @classmethod
    def from_assessments(
        cls,
        finding_id: str,
        *,
        exploitability: ExploitabilityAssessment | None = None,
        applicability: ApplicabilityAssessment | None = None,
        asset: AssetCriticality | None = None,
        likelihood: ExploitLikelihood | None = None,
    ) -> "AssessmentSnapshot":
        """Assemble a snapshot from whichever Component A/B outputs are available."""
        audits = [
            item.audit
            for item in (exploitability, applicability, asset)
            if item is not None and item.audit is not None
        ]
        confidences = [
            item.confidence for item in (exploitability, applicability, asset) if item is not None
        ]
        spans: list[str] = []
        for item in (exploitability, applicability, asset):
            if item is not None:
                spans.extend(item.evidence_spans)
        return cls(
            finding_id=finding_id,
            exploit_feasibility=exploitability.exploit_feasibility if exploitability else 0.0,
            asset_criticality=asset.criticality if asset else 0.0,
            p_exploit=likelihood.p_exploit if likelihood else 0.0,
            p_applicable=applicability.p_applicable if applicability else 0.5,
            verdict=applicability.verdict if applicability else ApplicabilityVerdict.UNCERTAIN,
            version_match=applicability.version_match if applicability else VersionMatch.UNKNOWN,
            exploit_maturity=exploitability.exploit_maturity if exploitability else ExploitMaturity.UNKNOWN,
            confidence=max(confidences) if confidences else 0.5,
            evidence_spans=tuple(spans),
            canary_leaked=any(audit.canary_leaked for audit in audits),
            envelope_broken=any(audit.envelope_broken for audit in audits),
            injection_signals=sum(len(audit.signals) for audit in audits),
            max_tier_used=max((audit.max_tier_used for audit in audits), default=TrustTier.OPERATOR),
        )

    @classmethod
    def from_enriched(cls, item: EnrichedFinding) -> "AssessmentSnapshot":
        """Snapshot of an :class:`EnrichedFinding`, including its trust ledger."""
        snapshot = cls.from_assessments(
            item.finding_id,
            exploitability=item.exploitability,
            applicability=item.applicability,
            asset=item.asset,
            likelihood=item.likelihood,
        )
        return replace(
            snapshot,
            canary_leaked=snapshot.canary_leaked or item.trust.canary_leaked,
            injection_signals=max(snapshot.injection_signals, item.trust.injection_signal_count),
            max_tier_used=max(snapshot.max_tier_used, item.trust.max_tier_used),
        )


class OutputAnomalyDetector(ManipulationDetector):
    """Compares a post-model assessment against tier <= 1 evidence and its own baseline.

    Three question classes, each producing a different :class:`DetectorName`:

    ``CONSISTENCY``
        The assessment contradicts a curated fact - a KEV-listed CVE ruled not applicable,
        ``p_exploit`` argued below the KEV floor, an exploited CVE called unproven, or an
        applicable verdict asserted over a tier <= 1 version mismatch.
    ``DIVERGENCE``
        A feature moved further from the deterministic baseline than any tier is allowed to
        move it. This is the implausible-jump check and it needs no intel at all.
    ``EVIDENCE_SPAN``
        High confidence with nothing quotable behind it, or a quoted span that is not a
        literal substring of the sanitized input.

    ``CANARY`` and ``ENVELOPE`` alerts are emitted verbatim from the audit record when the
    sandbox already caught the leak, so a caller gets one list covering the whole call.
    """

    name = DetectorName.CONSISTENCY

    def __init__(
        self,
        config: SandboxConfig | None = None,
        *,
        max_feature_jump: float = DEFAULT_MAX_FEATURE_JUMP,
        high_confidence: float = DEFAULT_HIGH_CONFIDENCE,
        floor_tolerance: float = 0.02,
    ) -> None:
        """``max_feature_jump`` defaults to the reference-page influence budget (0.35)."""
        self.config = config or SandboxConfig()
        self.max_feature_jump = float(max_feature_jump)
        self.high_confidence = float(high_confidence)
        self.floor_tolerance = float(floor_tolerance)

    # -- the checks --------------------------------------------------------

    def inspect(
        self,
        snapshot: AssessmentSnapshot,
        evidence: TrustedEvidence | None = None,
        *,
        baseline: AssessmentSnapshot | None = None,
        sanitized_inputs: Sequence[str] = (),
    ) -> list[ManipulationAlert]:
        """Every alert this assessment earns. An empty list means it is consistent."""
        alerts: list[ManipulationAlert] = []
        alerts.extend(self._sandbox_alerts(snapshot))
        if evidence is not None:
            alerts.extend(self._contradiction_alerts(snapshot, evidence))
        if baseline is not None:
            alerts.extend(self._divergence_alerts(snapshot, baseline))
        alerts.extend(self._certainty_alerts(snapshot, sanitized_inputs))
        return alerts

    def _sandbox_alerts(self, snapshot: AssessmentSnapshot) -> list[ManipulationAlert]:
        alerts: list[ManipulationAlert] = []
        if snapshot.canary_leaked:
            alerts.append(
                ManipulationAlert(
                    finding_id=snapshot.finding_id,
                    detector=DetectorName.CANARY,
                    severity=1.0,
                    message="canary token appeared in model output: untrusted content reached the model",
                )
            )
        if snapshot.envelope_broken:
            alerts.append(
                ManipulationAlert(
                    finding_id=snapshot.finding_id,
                    detector=DetectorName.ENVELOPE,
                    severity=0.9,
                    message="model output reproduced an untrusted-envelope closing tag",
                )
            )
        return alerts

    def _contradiction_alerts(
        self, snapshot: AssessmentSnapshot, evidence: TrustedEvidence
    ) -> list[ManipulationAlert]:
        alerts: list[ManipulationAlert] = []
        cves = ", ".join(evidence.cve_ids) or "the finding's CVE"

        if evidence.in_kev and snapshot.verdict == ApplicabilityVerdict.NOT_APPLICABLE:
            alerts.append(
                self._consistency(
                    snapshot,
                    0.9,
                    f"assessment rules {cves} not applicable while CISA KEV lists it as exploited",
                )
            )
        floor = evidence.floor_p_exploit
        if floor > 0.0 and snapshot.p_exploit < floor - self.floor_tolerance:
            alerts.append(
                self._consistency(
                    snapshot,
                    0.8,
                    f"p_exploit {snapshot.p_exploit:.2f} is below the tier<=1 floor {floor:.2f} "
                    f"implied by curated evidence for {cves}",
                )
            )
        if (
            evidence.verified_exploit_maturity >= _CONTRADICTORY_MATURITY_FLOOR
            and snapshot.exploit_maturity < _CONTRADICTORY_MATURITY_FLOOR
        ):
            alerts.append(
                self._consistency(
                    snapshot,
                    0.7,
                    f"assessment reports maturity {snapshot.exploit_maturity.name} while a curated feed "
                    f"holds verified {evidence.verified_exploit_maturity.name} exploit code for {cves}",
                )
            )
        if evidence.version_match == VersionMatch.MISMATCH and (
            snapshot.verdict == ApplicabilityVerdict.APPLICABLE or snapshot.p_applicable > 0.5
        ):
            alerts.append(
                self._consistency(
                    snapshot,
                    0.85,
                    f"assessment asserts applicability (p={snapshot.p_applicable:.2f}) over a tier<=1 "
                    f"version mismatch for {cves}",
                )
            )
        if evidence.version_match == VersionMatch.MATCH and snapshot.version_match == VersionMatch.MISMATCH:
            alerts.append(
                self._consistency(
                    snapshot,
                    0.75,
                    f"assessment reports a version mismatch for {cves} against a tier<=1 version match",
                )
            )
        return alerts

    def _divergence_alerts(
        self, snapshot: AssessmentSnapshot, baseline: AssessmentSnapshot
    ) -> list[ManipulationAlert]:
        alerts: list[ManipulationAlert] = []
        for label, current, base in (
            ("exploit_feasibility", snapshot.exploit_feasibility, baseline.exploit_feasibility),
            ("asset_criticality", snapshot.asset_criticality, baseline.asset_criticality),
            ("p_exploit", snapshot.p_exploit, baseline.p_exploit),
            ("p_applicable", snapshot.p_applicable, baseline.p_applicable),
        ):
            delta = float(current) - float(base)
            if abs(delta) > self.max_feature_jump + 1e-9:
                alerts.append(
                    ManipulationAlert(
                        finding_id=snapshot.finding_id,
                        detector=DetectorName.DIVERGENCE,
                        severity=min(1.0, abs(delta)),
                        message=(
                            f"{label} moved {delta:+.3f} from the deterministic baseline "
                            f"({base:.3f} -> {current:.3f}), beyond the {self.max_feature_jump:.2f} "
                            "any tier may apply"
                        ),
                    )
                )
        return alerts

    def _certainty_alerts(
        self, snapshot: AssessmentSnapshot, sanitized_inputs: Sequence[str]
    ) -> list[ManipulationAlert]:
        if snapshot.confidence < self.high_confidence:
            return []
        if not snapshot.evidence_spans:
            return [
                ManipulationAlert(
                    finding_id=snapshot.finding_id,
                    detector=DetectorName.EVIDENCE_SPAN,
                    severity=0.6,
                    message=(
                        f"confidence {snapshot.confidence:.2f} asserted with no quotable evidence span"
                    ),
                )
            ]
        if not sanitized_inputs:
            return []
        unverified = [span for span in snapshot.evidence_spans if not span_matches(span, sanitized_inputs)]
        if not unverified:
            return []
        return [
            ManipulationAlert(
                finding_id=snapshot.finding_id,
                detector=DetectorName.EVIDENCE_SPAN,
                severity=0.7,
                message=(
                    f"{len(unverified)} evidence span(s) are not literal substrings of the sanitized "
                    f"input, e.g. {unverified[0][:80]!r}"
                ),
            )
        ]

    @staticmethod
    def _consistency(snapshot: AssessmentSnapshot, severity: float, message: str) -> ManipulationAlert:
        return ManipulationAlert(
            finding_id=snapshot.finding_id,
            detector=DetectorName.CONSISTENCY,
            severity=severity,
            message=message[:400],
        )

    # -- interface ---------------------------------------------------------

    def detect(
        self, enriched: list[EnrichedFinding], context: Mapping[str, Any] | None = None
    ) -> list[ManipulationAlert]:
        """Inspect every enriched finding against its own intel.

        ``context`` may carry ``{"baselines": {finding_id: AssessmentSnapshot}}`` from a run
        with untrusted tiers neutralised; without it the divergence check is skipped, because
        there is nothing honest to compare against.
        """
        baselines: Mapping[str, AssessmentSnapshot] = (context or {}).get("baselines", {})
        alerts: list[ManipulationAlert] = []
        for item in enriched:
            evidence = TrustedEvidence.from_intel(
                item.intel, item.as_of, version_match=item.applicability.version_match
            )
            alerts.extend(
                self.inspect(
                    AssessmentSnapshot.from_enriched(item),
                    evidence,
                    baseline=baselines.get(item.finding_id),
                )
            )
        return alerts


# ---------------------------------------------------------------------------
# Corpus coverage of the pattern library
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageReport:
    """How much of the corpus the shipped pattern library actually catches."""

    pattern_library_version: str
    n_patterns: int
    n_attacks: int
    n_attacks_detected: int
    n_controls: int
    n_controls_flagged: int
    n_attacks_matched_by_pattern: int = 0
    n_attacks_caught_only_by_normalisation: int = 0
    undetected_attack_ids: tuple[str, ...] = ()
    flagged_control_ids: tuple[str, ...] = ()
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    patterns_fired: dict[str, int] = field(default_factory=dict)

    @property
    def detection_rate(self) -> float:
        """Share of attack cases that produced at least one signal."""
        return self.n_attacks_detected / self.n_attacks if self.n_attacks else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Share of benign controls that produced at least one signal."""
        return self.n_controls_flagged / self.n_controls if self.n_controls else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Plain-JSON view for the run report."""
        return {
            "pattern_library_version": self.pattern_library_version,
            "n_patterns": self.n_patterns,
            "n_attacks": self.n_attacks,
            "n_attacks_detected": self.n_attacks_detected,
            "n_attacks_matched_by_pattern": self.n_attacks_matched_by_pattern,
            "n_attacks_caught_only_by_normalisation": self.n_attacks_caught_only_by_normalisation,
            "detection_rate": round(self.detection_rate, 4),
            "n_controls": self.n_controls,
            "n_controls_flagged": self.n_controls_flagged,
            "false_positive_rate": round(self.false_positive_rate, 4),
            "undetected_attack_ids": list(self.undetected_attack_ids),
            "flagged_control_ids": list(self.flagged_control_ids),
            "per_category": self.per_category,
        }


def measure_pattern_coverage(
    cases: Iterable[AdversarialCase],
    detector: PreLLMPatternDetector | None = None,
) -> CoverageReport:
    """Run every corpus payload through the real sandbox filter and count what fires.

    This is the measurement ADR-002 asks for: the pattern library's detection rate on the
    attack cases and its false-positive rate on the benign controls, per category, plus the
    ids of everything it missed so the gaps can be named rather than averaged away.
    """
    engine = detector or PreLLMPatternDetector()
    attacks_total = 0
    attacks_detected = 0
    matched_by_pattern = 0
    normalisation_only = 0
    controls_total = 0
    controls_flagged = 0
    undetected: list[str] = []
    flagged: list[str] = []
    per_category: dict[str, dict[str, float]] = {}
    fired: dict[str, int] = {}

    for case in cases:
        score = engine.score_document(case.payload, tier_of(case.injection_point))
        for pattern_id in score.pattern_ids:
            fired[pattern_id] = fired.get(pattern_id, 0) + 1
        bucket = per_category.setdefault(
            case.category.value, {"n": 0.0, "detected": 0.0, "normalisation_only": 0.0}
        )
        bucket["n"] += 1.0
        bucket["detected"] += 1.0 if score.detected else 0.0
        caught_by_normalisation_only = bool(score.normalisation_signals) and not score.pattern_signals
        bucket["normalisation_only"] += 1.0 if caught_by_normalisation_only else 0.0

        if case.category == InjectionCategory.BENIGN_CONTROL:
            controls_total += 1
            if score.detected:
                controls_flagged += 1
                flagged.append(case.case_id)
        else:
            attacks_total += 1
            if score.detected:
                attacks_detected += 1
                if score.pattern_signals:
                    matched_by_pattern += 1
                else:
                    normalisation_only += 1
            else:
                undetected.append(case.case_id)

    for bucket in per_category.values():
        bucket["rate"] = round(bucket["detected"] / bucket["n"], 4) if bucket["n"] else 0.0

    return CoverageReport(
        pattern_library_version=engine.pattern_library_version,
        n_patterns=len(engine),
        n_attacks=attacks_total,
        n_attacks_detected=attacks_detected,
        n_attacks_matched_by_pattern=matched_by_pattern,
        n_attacks_caught_only_by_normalisation=normalisation_only,
        n_controls=controls_total,
        n_controls_flagged=controls_flagged,
        undetected_attack_ids=tuple(undetected),
        flagged_control_ids=tuple(flagged),
        per_category=per_category,
        patterns_fired=dict(sorted(fired.items())),
    )
