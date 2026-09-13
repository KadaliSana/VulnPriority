"""The comparison, and the verdict it supports.

Everything in this module is computed from :mod:`vulnprio.novelty.corpus` and
:mod:`vulnprio.novelty.capabilities`. There are no hard-coded conclusions: the claim level for
each capability follows from counting how many reviewed studies satisfy that capability's
test, and the threats to each claim are assembled from caveats recorded in the corpus data
plus conditions the numbers themselves trigger.

The claim rules, stated once so a reviewer can disagree with the rule rather than guess at it:

======================  ==========================================================
``not_novel``           two or more reviewed studies satisfy the dimension's test.
``novel_combination``   exactly one does. The capability exists in the literature;
                        what is new here is combining it with the rest, not the
                        capability.
``incremental``         none fully satisfies it but at least one partly does. The
                        framework completes something prior work approached.
``novel``               none fully or partly satisfies it.
======================  ==========================================================

A ``novel`` claim additionally carries an evidence strength. If the review is silent about the
dimension for more than half the corpus, the claim is marked ``weak``: a wall of ``unknown``
is not evidence of absence, and saying so is the difference between an analysis and an
advertisement.

Two further rules keep the output honest:

* No claim is made for a dimension the framework only partly satisfies. Those are reported
  with the prior work that does them better.
* ``nature`` is carried through to the verdict, so a dimension where this framework is alone
  in the corpus because nobody else built the control - rather than because nobody else had
  the idea - is labelled as engineering.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vulnprio.novelty.capabilities import (
    CAPABILITIES,
    CAPABILITY_KEYS,
    Capability,
    CapabilityLevel,
    CapabilityNature,
    capability,
    framework_capabilities,
    framework_partial_capabilities,
)
from vulnprio.novelty.corpus import Corpus, PriorStudy, load_corpus

__all__ = [
    "ClaimLevel",
    "EvidenceStrength",
    "WEAK_EVIDENCE_UNKNOWN_FRACTION",
    "DEFAULT_RARE_THRESHOLD",
    "MatrixRow",
    "UniqueCapability",
    "RareCapability",
    "SharedCapability",
    "Neighbour",
    "CapabilityClaim",
    "NoveltyVerdict",
    "capability_matrix",
    "unique_capabilities",
    "rare_capabilities",
    "shared_capabilities",
    "nearest_neighbours",
    "novelty_verdict",
    "novelty_payload",
]


class ClaimLevel(str, Enum):
    NOVEL = "novel"                          # no reviewed study has or partly has it
    NOVEL_COMBINATION = "novel_combination"  # exactly one reviewed study has it
    INCREMENTAL = "incremental"              # approached but not reached by prior work
    NOT_NOVEL = "not_novel"                  # two or more reviewed studies have it


class EvidenceStrength(str, Enum):
    STRONG = "strong"
    WEAK = "weak"


#: A claim of absence is weak when the review is silent for more than this share of the corpus.
WEAK_EVIDENCE_UNKNOWN_FRACTION: float = 0.5

#: "Rare" by default means at most two of forty-five studies.
DEFAULT_RARE_THRESHOLD: int = 2


# --------------------------------------------------------------------------- result models


class MatrixRow(BaseModel):
    """One dimension of the comparison matrix, with the study keys behind every count."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    title: str
    nature: CapabilityNature
    framework_position: CapabilityLevel
    total: int
    has_count: int
    partial_count: int
    lacks_count: int
    unknown_count: int
    has_studies: tuple[str, ...]
    partial_studies: tuple[str, ...]
    lacks_studies: tuple[str, ...]
    unknown_studies: tuple[str, ...]

    def counts_sum(self) -> int:
        return self.has_count + self.partial_count + self.lacks_count + self.unknown_count


class UniqueCapability(BaseModel):
    """A dimension no reviewed study satisfies, even partly."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    title: str
    nature: CapabilityNature
    checked_against: tuple[str, ...]   # studies positively coded as lacking it
    unknown_studies: tuple[str, ...]   # studies the review is silent about: possible counterexamples
    evidence_strength: EvidenceStrength
    framework_evidence: tuple[str, ...]


class RareCapability(BaseModel):
    """A dimension at most ``threshold`` reviewed studies satisfy."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    title: str
    has_count: int
    has_studies: tuple[str, ...]
    partial_count: int
    framework_position: CapabilityLevel


class SharedCapability(BaseModel):
    """A dimension this framework has and prior work already had. Explicitly not novel."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    title: str
    has_count: int
    has_studies: tuple[str, ...]
    note: str


class Neighbour(BaseModel):
    """A reviewed study ranked by capability overlap with this framework."""

    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    paradigm: str
    jaccard: float
    shared_count: int
    shared: tuple[str, ...]
    framework_only: tuple[str, ...]   # what this framework adds over the study
    study_only: tuple[str, ...]       # what the study has and this framework does not


class CapabilityClaim(BaseModel):
    """The novelty claim for one capability, with the evidence it was checked against."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    title: str
    nature: CapabilityNature
    framework_position: CapabilityLevel
    level: ClaimLevel
    evidence_strength: EvidenceStrength
    has_count: int
    partial_count: int
    unknown_count: int
    prior_work_with: tuple[str, ...]        # studies that satisfy the test
    prior_work_partial: tuple[str, ...]     # studies that partly satisfy it
    statement: str


class NoveltyVerdict(BaseModel):
    """The structured answer to "is any of this new?"."""

    model_config = ConfigDict(frozen=True)

    corpus_size: int
    dimensions: int
    framework_has: int
    framework_partial: int
    claims: tuple[CapabilityClaim, ...]
    novel: tuple[str, ...]
    novel_combination: tuple[str, ...]
    incremental: tuple[str, ...]
    not_novel: tuple[str, ...]
    not_claimed: tuple[str, ...]            # dimensions the framework only partly satisfies
    strongest_prior_work: str
    max_shared_capabilities: int
    overall_statement: str
    threats_to_novelty: tuple[str, ...] = Field(min_length=3)


# ------------------------------------------------------------------------------ internals


def _corpus() -> Corpus:
    return load_corpus()


def _keys_at(corpus: Corpus, dimension: str, level: CapabilityLevel) -> tuple[str, ...]:
    return tuple(sorted(s.key for s in corpus.studies if s.level(dimension) is level))


def _evidence_strength(unknown_count: int, total: int) -> EvidenceStrength:
    if total == 0:
        return EvidenceStrength.WEAK
    silent = unknown_count / total
    return EvidenceStrength.WEAK if silent > WEAK_EVIDENCE_UNKNOWN_FRACTION else EvidenceStrength.STRONG


def _labels(corpus: Corpus, keys: tuple[str, ...]) -> str:
    """Render study keys as citations for prose, e.g. ``Tita et al. (2026) [66]``."""
    return "; ".join(corpus.get(k).label() for k in keys)


# ----------------------------------------------------------------------------- public API


@lru_cache(maxsize=1)
def capability_matrix() -> dict[str, MatrixRow]:
    """Per capability: how many studies have, partly have, lack or are unrecorded on it.

    The four counts always sum to the corpus size, because every study is coded on every
    dimension at load time. That is what makes the matrix auditable: a reader can take any
    row, open the YAML, and check the named studies.
    """
    corpus = _corpus()
    total = len(corpus)
    rows: dict[str, MatrixRow] = {}
    for cap in CAPABILITIES:
        has = _keys_at(corpus, cap.key, CapabilityLevel.HAS)
        partial = _keys_at(corpus, cap.key, CapabilityLevel.PARTIAL)
        lacks = _keys_at(corpus, cap.key, CapabilityLevel.LACKS)
        unknown = _keys_at(corpus, cap.key, CapabilityLevel.UNKNOWN)
        rows[cap.key] = MatrixRow(
            dimension=cap.key,
            title=cap.title,
            nature=cap.nature,
            framework_position=cap.framework_position,
            total=total,
            has_count=len(has),
            partial_count=len(partial),
            lacks_count=len(lacks),
            unknown_count=len(unknown),
            has_studies=has,
            partial_studies=partial,
            lacks_studies=lacks,
            unknown_studies=unknown,
        )
    return rows


@lru_cache(maxsize=1)
def unique_capabilities() -> tuple[UniqueCapability, ...]:
    """Capabilities this framework fully has that no reviewed study has, even partly.

    Restricted to dimensions the framework itself satisfies fully: claiming uniqueness on a
    dimension the framework only half satisfies would be the same overclaiming this analysis
    exists to avoid.
    """
    matrix = capability_matrix()
    out: list[UniqueCapability] = []
    for key in framework_capabilities():
        row = matrix[key]
        if row.has_count or row.partial_count:
            continue
        cap = capability(key)
        out.append(
            UniqueCapability(
                dimension=key,
                title=cap.title,
                nature=cap.nature,
                checked_against=row.lacks_studies,
                unknown_studies=row.unknown_studies,
                evidence_strength=_evidence_strength(row.unknown_count, row.total),
                framework_evidence=cap.framework_modules + cap.framework_tests,
            )
        )
    return tuple(out)


def rare_capabilities(threshold: int = DEFAULT_RARE_THRESHOLD) -> tuple[RareCapability, ...]:
    """Capabilities at most ``threshold`` reviewed studies fully satisfy.

    Rarity is reported across all dimensions, not only this framework's, because a dimension
    that is rare in the corpus and absent from this framework is the more interesting finding.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    matrix = capability_matrix()
    out = [
        RareCapability(
            dimension=key,
            title=row.title,
            has_count=row.has_count,
            has_studies=row.has_studies,
            partial_count=row.partial_count,
            framework_position=row.framework_position,
        )
        for key, row in matrix.items()
        if row.has_count <= threshold
    ]
    return tuple(sorted(out, key=lambda r: (r.has_count, r.dimension)))


@lru_cache(maxsize=1)
def shared_capabilities() -> tuple[SharedCapability, ...]:
    """What this framework does that prior work already did. Named, with the studies.

    This is the half of the answer a novelty analysis usually omits. Learning to rank over
    combined evidence, EPSS/KEV grounding, asset context, attack-graph position and SHAP
    explanation are all in this list, because they are all in the corpus.
    """
    corpus = _corpus()
    matrix = capability_matrix()
    out: list[SharedCapability] = []
    for key in framework_capabilities():
        row = matrix[key]
        if row.has_count < 2:
            continue
        out.append(
            SharedCapability(
                dimension=key,
                title=row.title,
                has_count=row.has_count,
                has_studies=row.has_studies,
                note=(
                    f"{row.has_count} of {row.total} reviewed studies already satisfy this "
                    f"test, including {_labels(corpus, row.has_studies[:3])}. "
                    "This framework claims no novelty here."
                ),
            )
        )
    return tuple(sorted(out, key=lambda s: (-s.has_count, s.dimension)))


def nearest_neighbours(k: int = 5) -> tuple[Neighbour, ...]:
    """The ``k`` reviewed studies closest to this framework by capability overlap.

    Overlap is Jaccard similarity over the sets of fully satisfied dimensions. Surveys, which
    have no method and are coded ``unknown`` throughout, fall to the bottom automatically -
    which is the point: the comparison should be against the strongest prior work, not the
    most convenient.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    corpus = _corpus()
    mine = frozenset(framework_capabilities())
    scored: list[Neighbour] = []
    for entry in corpus.studies:
        theirs = entry.capability_set()
        union = mine | theirs
        shared = mine & theirs
        jaccard = round(len(shared) / len(union), 4) if union else 0.0
        scored.append(
            Neighbour(
                key=entry.key,
                label=entry.label(),
                paradigm=entry.paradigm.value,
                jaccard=jaccard,
                shared_count=len(shared),
                shared=tuple(sorted(shared)),
                framework_only=tuple(sorted(mine - theirs)),
                study_only=tuple(sorted(theirs - mine)),
            )
        )
    scored.sort(key=lambda n: (-n.jaccard, -n.shared_count, n.key))
    return tuple(scored[:k])


def _statement(corpus: Corpus, cap: Capability, row: MatrixRow, level: ClaimLevel) -> str:
    """One sentence per claim, naming the studies it was checked against."""
    if level is ClaimLevel.NOT_NOVEL:
        return (
            f"Not novel. {row.has_count} of {row.total} reviewed studies satisfy this test, "
            f"including {_labels(corpus, row.has_studies[:3])}."
        )
    if level is ClaimLevel.NOVEL_COMBINATION:
        return (
            f"Not a new capability. {_labels(corpus, row.has_studies)} already does this; the "
            "claim is only that combining it with the rest of the framework is new. A single "
            "precedent also means a single re-reading of that paper could overturn the claim."
        )
    if level is ClaimLevel.INCREMENTAL:
        return (
            f"Incremental. No reviewed study fully satisfies this test, but {row.partial_count} "
            f"approach it, including {_labels(corpus, row.partial_studies[:3])}."
        )
    nature = (
        "The advantage is engineering, not science: the mechanisms come from outside this "
        "literature and what is new is building and measuring them here."
        if cap.nature is CapabilityNature.ENGINEERING
        else "This is a methodological claim about how priority should be modelled or measured."
    )
    return (
        f"Novel within this corpus: none of the {row.total} reviewed studies satisfies this "
        f"test, {row.lacks_count} are positively coded as lacking it and the review is silent "
        f"for {row.unknown_count}. {nature}"
    )


@lru_cache(maxsize=1)
def novelty_verdict() -> NoveltyVerdict:
    """The structured verdict: a claim level per capability, and what threatens each."""
    corpus = _corpus()
    matrix = capability_matrix()
    mine = framework_capabilities()
    partials = framework_partial_capabilities()

    claims: list[CapabilityClaim] = []
    for cap in CAPABILITIES:
        row = matrix[cap.key]
        if row.has_count >= 2:
            level = ClaimLevel.NOT_NOVEL
        elif row.has_count == 1:
            level = ClaimLevel.NOVEL_COMBINATION
        elif row.partial_count >= 1:
            level = ClaimLevel.INCREMENTAL
        else:
            level = ClaimLevel.NOVEL
        claims.append(
            CapabilityClaim(
                dimension=cap.key,
                title=cap.title,
                nature=cap.nature,
                framework_position=cap.framework_position,
                level=level,
                evidence_strength=_evidence_strength(row.unknown_count, row.total),
                has_count=row.has_count,
                partial_count=row.partial_count,
                unknown_count=row.unknown_count,
                prior_work_with=row.has_studies,
                prior_work_partial=row.partial_studies,
                statement=_statement(corpus, cap, row, level),
            )
        )

    def _at(level: ClaimLevel) -> tuple[str, ...]:
        # Claims are only made for dimensions the framework fully satisfies.
        return tuple(c.dimension for c in claims if c.level is level and c.dimension in mine)

    neighbours = nearest_neighbours(k=len(corpus))
    strongest = neighbours[0]
    max_shared = max(n.shared_count for n in neighbours)

    novel = _at(ClaimLevel.NOVEL)
    combination = _at(ClaimLevel.NOVEL_COMBINATION)
    incremental = _at(ClaimLevel.INCREMENTAL)
    not_novel = _at(ClaimLevel.NOT_NOVEL)

    engineering_novel = tuple(
        d for d in novel if capability(d).nature is CapabilityNature.ENGINEERING
    )
    quantifier = {1: "The one", 2: "Both"}.get(len(engineering_novel), f"All {len(engineering_novel)}")
    overall = (
        f"Of the {len(mine)} capabilities this framework fully satisfies, {len(novel)} are "
        f"unprecedented in the {len(corpus)} reviewed studies, {len(combination)} exist in "
        f"exactly one study each, {len(incremental)} are approached but not reached by prior "
        f"work, and {len(not_novel)} are already established. "
        + (
            f"{quantifier} unprecedented capabilities are engineering controls rather than "
            "research results. "
            if engineering_novel and len(engineering_novel) == len(novel)
            else ""
        )
        + f"The claim to novelty is therefore combinatorial rather than per-capability: the "
        f"closest prior work, {strongest.label}, shares {strongest.shared_count} of these "
        f"{len(mine)} capabilities and no reviewed study shares more than {max_shared}. "
        "Whether that combination is worth anything is an empirical question this framework has "
        "not yet answered, because it has never been run against a real deployment."
    )

    threats = _threats(corpus, matrix, claims, novel, combination, partials, strongest, max_shared)

    return NoveltyVerdict(
        corpus_size=len(corpus),
        dimensions=len(CAPABILITY_KEYS),
        framework_has=len(mine),
        framework_partial=len(partials),
        claims=tuple(claims),
        novel=novel,
        novel_combination=combination,
        incremental=incremental,
        not_novel=not_novel,
        not_claimed=partials,
        strongest_prior_work=strongest.label,
        max_shared_capabilities=max_shared,
        overall_statement=overall,
        threats_to_novelty=threats,
    )


def _threats(
    corpus: Corpus,
    matrix: dict[str, MatrixRow],
    claims: list[CapabilityClaim],
    novel: tuple[str, ...],
    combination: tuple[str, ...],
    partials: tuple[str, ...],
    strongest: Neighbour,
    max_shared: int,
) -> tuple[str, ...]:
    """Assemble the reasons a reviewer could push back, from data and from the numbers."""
    threats: list[str] = list(corpus.caveat_texts())

    claimed = set(novel) | set(combination)

    # Study-level caveats recorded against a dimension currently claimed as new.
    for entry in corpus.studies:
        for caveat in entry.caveats:
            if caveat.dimension in claimed:
                threats.append(
                    f"On '{capability(caveat.dimension).title}' - {entry.label()}: {caveat.note}"
                )

    # A claim of absence resting on silence rather than on evidence.
    for claim in claims:
        if claim.level is ClaimLevel.NOVEL and claim.evidence_strength is EvidenceStrength.WEAK:
            threats.append(
                f"The '{claim.title}' claim is weak evidence: the review says nothing about "
                f"{claim.unknown_count} of {len(corpus)} studies on this dimension, so absence "
                "here is largely unrecorded rather than established."
            )

    # Single-precedent claims.
    for dimension in combination:
        row = matrix[dimension]
        threats.append(
            f"'{capability(dimension).title}' rests on a single precedent "
            f"({_labels(corpus, row.has_studies)}). If that study does more than the review "
            "reports, the claim collapses from novel_combination to not_novel."
        )

    # Dimensions where prior work is ahead of this framework.
    for dimension in partials:
        row = matrix[dimension]
        if row.has_count:
            plural = "study does" if row.has_count == 1 else "studies do"
            threats.append(
                f"On '{capability(dimension).title}' this framework is only partial while "
                f"{row.has_count} reviewed {plural} satisfy the test in full, including "
                f"{_labels(corpus, row.has_studies[:2])}. "
                f"{capability(dimension).framework_note}"
            )

    # The combinatorial claim itself.
    threats.append(
        f"The combination argument is the whole claim, and it is the weakest kind: the closest "
        f"prior work ({strongest.label}) already shares {max_shared} capabilities, and "
        "'combines more things' is not a result. It becomes one only if the ablation shows the "
        "added components carry their weight, on data somebody else would accept."
    )
    return tuple(threats)


def _narrative(verdict: NoveltyVerdict, matrix: dict[str, MatrixRow]) -> tuple[str, ...]:
    """Computed prose for the site: the answer, the caveat, and the not-novel half."""
    corpus = _corpus()
    lines = [verdict.overall_statement]

    if verdict.novel:
        titles = "; ".join(capability(d).title for d in verdict.novel)
        lines.append(
            f"Genuinely unprecedented in this corpus: {titles}. Each was checked against all "
            f"{verdict.corpus_size} reviewed studies."
        )
    else:  # pragma: no cover - defensive; the corpus currently yields two
        lines.append("No capability of this framework is unprecedented in the reviewed corpus.")

    if verdict.novel_combination:
        pairs = "; ".join(
            f"{capability(d).title} ({_labels(corpus, matrix[d].has_studies)})"
            for d in verdict.novel_combination
        )
        lines.append(f"Present in exactly one reviewed study each, hence a combination claim only: {pairs}.")

    if verdict.not_novel:
        titles = "; ".join(capability(d).title for d in verdict.not_novel)
        lines.append(f"Established prior work, claimed by this framework as novel in no sense: {titles}.")

    if verdict.not_claimed:
        titles = "; ".join(capability(d).title for d in verdict.not_claimed)
        lines.append(
            f"Capabilities this framework only partly satisfies, where prior work is at least "
            f"as strong: {titles}."
        )
    return tuple(lines)


def novelty_payload() -> dict[str, Any]:
    """A JSON-serialisable view of the whole analysis, for the web layer.

    Stable across runs: every collection is sorted or follows the declared capability order,
    and every float is rounded at construction.
    """
    matrix = capability_matrix()
    verdict = novelty_verdict()
    corpus = _corpus()
    return {
        "schema_version": 1,
        "review_title": corpus.review_title,
        "corpus_note": corpus.review_corpus_note,
        "corpus_size": len(corpus),
        "capabilities": [c.model_dump(mode="json") for c in CAPABILITIES],
        "matrix": {key: row.model_dump(mode="json") for key, row in matrix.items()},
        "studies": [s.model_dump(mode="json") for s in corpus.studies],
        "verdict": verdict.model_dump(mode="json"),
        "unique_capabilities": [u.model_dump(mode="json") for u in unique_capabilities()],
        "rare_capabilities": [r.model_dump(mode="json") for r in rare_capabilities()],
        "shared_capabilities": [s.model_dump(mode="json") for s in shared_capabilities()],
        "nearest_neighbours": [n.model_dump(mode="json") for n in nearest_neighbours(5)],
        "narrative": list(_narrative(verdict, matrix)),
        "caveats": [c.model_dump(mode="json") for c in corpus.caveats],
        "claim_rules": {
            "not_novel": "two or more reviewed studies satisfy the dimension's test",
            "novel_combination": "exactly one reviewed study satisfies it",
            "incremental": "none fully satisfies it, at least one partly does",
            "novel": "none fully or partly satisfies it",
            "weak_evidence": (
                "a claim of absence is marked weak when the review is silent for more than "
                f"{int(WEAK_EVIDENCE_UNKNOWN_FRACTION * 100)}% of the corpus"
            ),
        },
    }


def _study_keys() -> tuple[str, ...]:  # pragma: no cover - convenience for the REPL
    return _corpus().keys()


def _study(key: str) -> PriorStudy:  # pragma: no cover - convenience for the REPL
    return _corpus().get(key)
