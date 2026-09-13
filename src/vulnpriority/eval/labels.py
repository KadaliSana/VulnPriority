"""Exploitation ground truth (DESIGN.md 3.9, Gap 3).

The single most common methodological failure in the vulnerability-prioritisation
literature is circular validation: a model is trained on CVSS-derived labels and then
evaluated against CVSS-derived labels, so the reported accuracy measures agreement with
the instrument rather than with reality. This module is the framework's answer, and it
enforces the answer rather than documenting it.

A finding is labelled **exploited** only when one of four things is true:

============================  ==============================================================
``LabelSource.KEV``           the CVE was in CISA's Known Exploited Vulnerabilities catalogue
                              on or before the observation cutoff
``LabelSource.EXPLOIT_EVIDENCE``  exploit evidence exists at maturity >= the policy floor
                              (``FUNCTIONAL`` by default: a proof of concept is not proof
                              of exploitation)
``LabelSource.INCIDENT``      a recorded incident names the CVE or the finding
``LabelSource.SYNTHETIC_ORACLE``  the synthetic world's oracle recorded an exploitation
                              event, which is the only counterfactually complete source
                              that exists anywhere
============================  ==============================================================

CVSS is not on the list and cannot be added. The prohibition is enforced at three
levels: ``LabelSource`` has no CVSS member, ``GroundTruthLabel.cvss_used_as_label`` is
typed ``Literal[False]``, and :class:`LabelBuilder` raises
:class:`~vulnpriority.core.errors.LabelPolicyError` when a caller names a CVSS- or
severity-derived source.

Two further properties make the labels honest:

* **Version-aware.** A CVE exploited somewhere in the world says nothing about an
  installation running a version outside the affected range. A finding whose version
  evidence is ``VersionMatch.MISMATCH`` is *dropped* rather than labelled positive, so
  it pollutes neither the positive class nor the negative one.
* **Source-aware.** Every label records ``source_agreement``: the share of the sources
  that had an opinion which agreed with the verdict. A finding that is in KEV but has no
  exploit evidence carries an agreement below 1, and ``LabelPolicy.min_source_agreement``
  can require corroboration before a positive is accepted.

Everything is evaluated as of ``observation_cutoff``: evidence dated after it is ignored,
which is what allows a time-ordered split to be leak-free.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vulnpriority.core.enums import LabelSource, VersionMatch
from vulnpriority.core.errors import LabelPolicyError
from vulnpriority.core.models import (
    EnrichedFinding,
    Finding,
    GroundTruthLabel,
    LabelPolicy,
    LabelSet,
    VulnIntel,
)

__all__ = [
    "CVSS_DERIVED_TOKENS",
    "ORACLE_DATE_FIELDS",
    "LabelAudit",
    "LabelBuilder",
    "resolve_label_source",
    "assert_not_cvss_derived",
]

#: Names a caller might reach for that would reintroduce the circularity of Gap 3.
#: Matching is case-insensitive and substring-based on the normalised token, so
#: ``"CVSS v3 base"``, ``"cvss_score"`` and ``"scanner severity"`` are all refused.
CVSS_DERIVED_TOKENS: frozenset[str] = frozenset(
    {
        "cvss",
        "base_score",
        "basescore",
        "severity",
        "criticality_score",
        "risk_score",
        "vector",
        "epss",
    }
)

#: Attribute / key names an oracle event may use for the date exploitation was observed.
ORACLE_DATE_FIELDS: tuple[str, ...] = (
    "first_evidence_date",
    "exploited_at",
    "event_date",
    "occurred_at",
    "date",
)


def assert_not_cvss_derived(name: str) -> None:
    """Raise :class:`LabelPolicyError` when ``name`` denotes a CVSS-derived label source.

    EPSS is refused for the same reason CVSS is: it is a *prediction* of exploitation,
    and training on it teaches a model to reproduce another model rather than reality.
    """
    normalised = "".join(character if character.isalnum() else "_" for character in name.lower())
    for token in CVSS_DERIVED_TOKENS:
        if token in normalised:
            raise LabelPolicyError(
                f"{name!r} is a CVSS/score-derived label source. Ground truth may only come from "
                f"{[source.value for source in LabelSource]} (DESIGN.md 3.9, Gap 3): labelling "
                "with a severity score makes the evaluation circular."
            )


def resolve_label_source(name: str | LabelSource) -> LabelSource:
    """Coerce a source name to a :class:`LabelSource`, refusing CVSS-derived ones."""
    if isinstance(name, LabelSource):
        return name
    text = str(name)
    assert_not_cvss_derived(text)
    try:
        return LabelSource(text.strip().lower())
    except ValueError as exc:
        raise LabelPolicyError(
            f"unknown label source {name!r}; accepted sources are "
            f"{[source.value for source in LabelSource]}"
        ) from exc


@dataclass
class LabelAudit:
    """What the builder did and, more importantly, what it refused to do.

    ``LabelSet`` is a frozen contract with no room for bookkeeping, so the counts a
    reviewer needs ("how many findings did you drop, and why") live here on the builder.
    """

    n_findings: int = 0
    n_labelled: int = 0
    n_positive: int = 0
    dropped_version_mismatch: tuple[str, ...] = ()
    dropped_low_agreement: tuple[str, ...] = ()
    positives_by_source: dict[str, int] = field(default_factory=dict)
    evidence_after_cutoff: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_findings": self.n_findings,
            "n_labelled": self.n_labelled,
            "n_positive": self.n_positive,
            "n_dropped_version_mismatch": len(self.dropped_version_mismatch),
            "n_dropped_low_agreement": len(self.dropped_low_agreement),
            "dropped_version_mismatch": list(self.dropped_version_mismatch),
            "dropped_low_agreement": list(self.dropped_low_agreement),
            "positives_by_source": dict(self.positives_by_source),
            "evidence_after_cutoff": self.evidence_after_cutoff,
        }


@dataclass(frozen=True)
class _Vote:
    """One source's opinion about one finding."""

    source: LabelSource
    positive: bool
    evidence_date: date | None = None


class LabelBuilder:
    """Builds a :class:`LabelSet` from confirmed exploitation evidence only.

    ``sources`` overrides ``policy.accepted_sources`` and is the hook a caller would use
    to ask for something else; every entry passes through :func:`resolve_label_source`,
    which is where a CVSS-derived request is refused::

        LabelBuilder(policy, sources=["kev", "cvss"])   # -> LabelPolicyError
    """

    def __init__(
        self,
        policy: LabelPolicy | None = None,
        *,
        sources: Sequence[str | LabelSource] | None = None,
    ) -> None:
        policy = policy or LabelPolicy()
        accepted = tuple(
            resolve_label_source(source)
            for source in (sources if sources is not None else policy.accepted_sources)
        )
        if not accepted:
            raise LabelPolicyError("at least one label source is required")
        # The resolved tuple is written back onto the policy either way, so the policy
        # recorded on the LabelSet is exactly the one that was enforced.
        self.policy = policy.model_copy(update={"accepted_sources": accepted})
        self.accepted: frozenset[LabelSource] = frozenset(accepted)
        self.audit = LabelAudit()

    # -- public API --------------------------------------------------------

    def build(
        self,
        findings: Sequence[Finding | EnrichedFinding],
        intel_by_cve: Mapping[str, VulnIntel] | None = None,
        oracle_events: Any = None,
        *,
        observation_cutoff: date,
        incidents: Any = None,
        version_match_by_finding: Mapping[str, VersionMatch] | None = None,
        impact_by_finding: Mapping[str, float] | None = None,
    ) -> LabelSet:
        """Label every finding as of ``observation_cutoff``.

        ``findings`` may hold :class:`Finding` or :class:`EnrichedFinding` objects; the
        enriched form additionally supplies version evidence (from Component A's
        applicability assessment) and monetary impact (for the graded relevance), both of
        which can also be passed explicitly.

        ``oracle_events`` and ``incidents`` are accepted in whatever shape the synthetic
        generator produces: a mapping of finding id (or CVE id) to date, a sequence of
        mappings, or a sequence of objects carrying ``finding_id``/``cve_id`` and one of
        :data:`ORACLE_DATE_FIELDS`.
        """
        intel_by_cve = dict(intel_by_cve or {})
        oracle = _index_events(oracle_events)
        incident = _index_events(incidents)
        version_hint = dict(version_match_by_finding or {})
        impact_hint = dict(impact_by_finding or {})

        self.audit = LabelAudit(n_findings=len(findings))
        candidates: list[tuple[GroundTruthLabel, float]] = []

        for item in findings:
            finding = item.finding if isinstance(item, EnrichedFinding) else item
            version_match = version_hint.get(
                finding.finding_id,
                item.applicability.version_match
                if isinstance(item, EnrichedFinding)
                else VersionMatch.UNKNOWN,
            )
            impact = impact_hint.get(
                finding.finding_id,
                float(item.impact.total) if isinstance(item, EnrichedFinding) else 0.0,
            )

            votes = self._collect_votes(finding, intel_by_cve, oracle, incident, observation_cutoff)
            positives = [vote for vote in votes if vote.positive]

            if positives and self.policy.require_version_match and version_match == VersionMatch.MISMATCH:
                # The CVE was exploited somewhere; this installation runs a version
                # outside the affected range, so the evidence says nothing about it.
                if self.policy.drop_version_mismatch:
                    self.audit.dropped_version_mismatch += (finding.finding_id,)
                    continue
                positives = []

            agreement = _source_agreement(votes)
            if positives and agreement < self.policy.min_source_agreement:
                self.audit.dropped_low_agreement += (finding.finding_id,)
                continue

            grade = self._base_grade(positives)
            dates = [vote.evidence_date for vote in positives if vote.evidence_date is not None]
            label = GroundTruthLabel(
                finding_id=finding.finding_id,
                cve_id=finding.cve_ids[0] if finding.cve_ids else None,
                exploited=bool(positives),
                relevance_grade=grade,
                sources=tuple(sorted({vote.source for vote in positives}, key=lambda s: s.value)),
                first_evidence_date=min(dates) if dates else None,
                version_confirmed=version_match,
                source_agreement=agreement,
            )
            candidates.append((label, impact))
            for vote in positives:
                key = vote.source.value
                self.audit.positives_by_source[key] = self.audit.positives_by_source.get(key, 0) + 1

        labels = self._apply_impact_weighting(candidates)
        self.audit.n_labelled = len(labels)
        self.audit.n_positive = sum(1 for label in labels if label.exploited)
        return LabelSet(
            policy=self.policy,
            observation_cutoff=observation_cutoff,
            labels=tuple(labels),
        )

    # -- internals ---------------------------------------------------------

    def _collect_votes(
        self,
        finding: Finding,
        intel_by_cve: Mapping[str, VulnIntel],
        oracle: Mapping[str, date | None],
        incidents: Mapping[str, date | None],
        cutoff: date,
    ) -> list[_Vote]:
        """One vote per accepted source that had anything to say about this finding.

        A source that holds no data for the finding abstains and does not appear, which
        is what keeps ``source_agreement`` meaningful: it measures disagreement among
        informed sources, not silence.
        """
        votes: list[_Vote] = []
        intel = [intel_by_cve[cve] for cve in finding.cve_ids if cve in intel_by_cve]

        if LabelSource.KEV in self.accepted:
            kev_records = [item.kev for item in intel if item.kev is not None]
            if kev_records:
                listed = [
                    record
                    for record in kev_records
                    if record.in_kev and (record.date_added is None or record.date_added <= cutoff)
                ]
                self.audit.evidence_after_cutoff += sum(
                    1
                    for record in kev_records
                    if record.in_kev and record.date_added is not None and record.date_added > cutoff
                )
                if listed:
                    dates = [record.date_added for record in listed if record.date_added]
                    votes.append(_Vote(LabelSource.KEV, True, min(dates) if dates else None))
                else:
                    votes.append(_Vote(LabelSource.KEV, False))

        if LabelSource.EXPLOIT_EVIDENCE in self.accepted:
            evidence = [exploit for item in intel for exploit in item.exploits]
            if intel:
                floor = self.policy.min_exploit_maturity_for_evidence
                qualifying = [
                    exploit
                    for exploit in evidence
                    if exploit.maturity >= floor
                    and (exploit.published is None or exploit.published <= cutoff)
                ]
                self.audit.evidence_after_cutoff += sum(
                    1
                    for exploit in evidence
                    if exploit.maturity >= floor
                    and exploit.published is not None
                    and exploit.published > cutoff
                )
                if qualifying:
                    dates = [item.published for item in qualifying if item.published]
                    votes.append(
                        _Vote(LabelSource.EXPLOIT_EVIDENCE, True, min(dates) if dates else None)
                    )
                else:
                    votes.append(_Vote(LabelSource.EXPLOIT_EVIDENCE, False))

        if LabelSource.INCIDENT in self.accepted:
            vote = _lookup_event(incidents, finding, cutoff, LabelSource.INCIDENT)
            if vote is not None:
                votes.append(vote)

        if LabelSource.SYNTHETIC_ORACLE in self.accepted:
            vote = _lookup_event(oracle, finding, cutoff, LabelSource.SYNTHETIC_ORACLE)
            if vote is not None:
                votes.append(vote)

        return votes

    def _base_grade(self, positives: Sequence[_Vote]) -> int:
        """Graded relevance before impact weighting: the strongest firing source wins.

        KEV membership and a recorded incident are direct observations of exploitation
        (grade 4 by default); exploit evidence at the maturity floor is strong but
        indirect (grade 3). The synthetic oracle records an actual exploitation event in
        a world whose ground truth is complete, so it is graded like an incident.
        """
        if not positives:
            return 0
        grades = {
            LabelSource.KEV: self.policy.kev_grade,
            LabelSource.EXPLOIT_EVIDENCE: self.policy.exploit_evidence_grade,
            LabelSource.INCIDENT: self.policy.incident_grade,
            LabelSource.SYNTHETIC_ORACLE: self.policy.incident_grade,
        }
        return int(max(grades[vote.source] for vote in positives))

    def _apply_impact_weighting(
        self, candidates: Sequence[tuple[GroundTruthLabel, float]]
    ) -> list[GroundTruthLabel]:
        """Optional impact-quartile weighting of the graded relevance.

        With ``LabelPolicy.weight_by_impact`` set, a positive's grade is nudged by the
        quartile of its monetary impact among the positives: bottom quartile loses a
        grade, top quartile gains one, the middle two keep the source grade, and the
        result is clipped to ``[1, 4]``. This is what makes NDCG's exponential gain
        express the protocol's actual preference - of two confirmed-exploited findings,
        the one on the payment endpoint should be ranked first - without ever letting
        impact create or destroy a positive label.
        """
        labels = [label for label, _ in candidates]
        if not self.policy.weight_by_impact:
            return labels
        impacts = [impact for label, impact in candidates if label.exploited]
        if len(set(impacts)) < 2:
            return labels
        cuts = np.quantile(np.asarray(impacts, dtype=float), [0.25, 0.5, 0.75])
        out: list[GroundTruthLabel] = []
        for label, impact in candidates:
            if not label.exploited:
                out.append(label)
                continue
            quartile = int(np.searchsorted(cuts, impact, side="right"))  # 0..3
            adjustment = {0: -1, 1: 0, 2: 0, 3: 1}[quartile]
            grade = int(min(4, max(1, label.relevance_grade + adjustment)))
            out.append(label.model_copy(update={"relevance_grade": grade}))
        return out


# ---------------------------------------------------------------------------
# Event indexing: accept whatever shape the oracle / incident log arrives in
# ---------------------------------------------------------------------------


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _event_fields(event: Any) -> tuple[list[str], date | None]:
    """``(keys this event is about, its date)`` for a mapping or an attribute object."""
    keys: list[str] = []
    when: date | None = None
    if isinstance(event, MappingABC):
        getter = event.get
    else:
        def getter(name: str, default: Any = None) -> Any:  # type: ignore[misc]
            return getattr(event, name, default)
    for name in ("finding_id", "cve_id", "id"):
        value = getter(name)
        if isinstance(value, str) and value:
            keys.append(value)
    for name in ORACLE_DATE_FIELDS:
        when = _to_date(getter(name))
        if when is not None:
            break
    return keys, when


def _index_events(events: Any) -> dict[str, date | None]:
    """Normalise an oracle / incident log into ``{finding id or CVE id: date}``.

    ``None`` input means "this source was not consulted" and yields an empty index, which
    makes the source abstain rather than vote negative.
    """
    index: dict[str, date | None] = {}
    if events is None:
        return index
    if isinstance(events, MappingABC):
        for key, value in events.items():
            if isinstance(value, (date, datetime, str, type(None))):
                index[str(key)] = _to_date(value)
            else:
                keys, when = _event_fields(value)
                for name in keys or [str(key)]:
                    index[name] = when
        return index
    if isinstance(events, (str, bytes)):
        return index
    for event in events:
        if isinstance(event, str):
            index.setdefault(event, None)
            continue
        keys, when = _event_fields(event)
        for name in keys:
            index[name] = when
    return index


def _lookup_event(
    index: Mapping[str, date | None], finding: Finding, cutoff: date, source: LabelSource
) -> _Vote | None:
    """A vote from an event log, or ``None`` when the log was not consulted at all."""
    if not index:
        return None
    keys: Iterable[str] = (finding.finding_id, *finding.cve_ids)
    hits = [(key, index[key]) for key in keys if key in index]
    if not hits:
        return _Vote(source, False)
    in_window = [when for _, when in hits if when is None or when <= cutoff]
    if not in_window:
        return _Vote(source, False)
    dated = [when for when in in_window if when is not None]
    return _Vote(source, True, min(dated) if dated else None)


def _source_agreement(votes: Sequence[_Vote]) -> float:
    """Share of the informed sources that agree with the majority verdict.

    One informed source, or unanimity, gives ``1.0``; a two-to-one split gives ``0.667``;
    an even split gives ``0.5``. No informed source gives ``1.0``, because a finding
    nobody has any evidence about is unanimously unevidenced.
    """
    if not votes:
        return 1.0
    positive = sum(1 for vote in votes if vote.positive)
    negative = len(votes) - positive
    return float(max(positive, negative) / len(votes))
