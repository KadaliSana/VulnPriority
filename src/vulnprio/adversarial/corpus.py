"""Loading and integrity-checking the adversarial corpus (DESIGN.md 3.10).

The corpus is a measurement instrument, so it is loaded the way a measurement instrument
should be: every structural defect is a hard :class:`ConfigError` rather than a silent skip.
A corpus that quietly lost its canary-exfiltration cases, or whose benign controls fell
below twenty, would make the framework report a robustness number it has not earned - which
is exactly the failure mode the review found in the literature (only three of eighty-four
surveyed studies tested adversarial robustness at all).

The checks enforced here are the ones DESIGN.md 3.10 and ADR-002 state as requirements:

* at least :data:`MIN_ATTACK_CASES` attack cases and :data:`MIN_BENIGN_CONTROLS` controls;
* every :class:`~vulnprio.core.enums.InjectionCategory` represented at least once;
* unique, non-empty case ids;
* no case injecting at the operator tier (also enforced by the frozen model, re-raised here
  as a configuration error so a bad corpus fails at load rather than at first use);
* benign controls carry ``goal: none`` and attacks never do, so the negative class cannot be
  polluted by an attack that merely forgot its goal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from pydantic import ValidationError

from vulnprio.core.config import PROJECT_ROOT, AdversarialConfig
from vulnprio.core.enums import InjectionCategory, Provenance, TrustTier, tier_of
from vulnprio.core.errors import ConfigError
from vulnprio.core.models import AdversarialCase

__all__ = [
    "MIN_ATTACK_CASES",
    "MIN_BENIGN_CONTROLS",
    "ATTACK_CATEGORIES",
    "VALID_INJECTION_POINTS",
    "KNOWN_PATTERN_GAPS",
    "CorpusStats",
    "default_corpus_path",
    "load_corpus",
    "corpus_stats",
    "attack_cases",
    "benign_controls",
    "case_by_id",
    "cases_in_category",
    "validate_corpus",
]

#: ADR-002 and DESIGN.md 3.10 both name these floors. They are minimums, not targets.
MIN_ATTACK_CASES: int = 60
MIN_BENIGN_CONTROLS: int = 20

#: Every category except the negative class is an attack category.
ATTACK_CATEGORIES: tuple[InjectionCategory, ...] = tuple(
    category for category in InjectionCategory if category != InjectionCategory.BENIGN_CONTROL
)

#: The four surfaces an adversary can actually write on, as listed in the frozen contract's
#: comment on ``AdversarialCase.injection_point``.
VALID_INJECTION_POINTS: tuple[Provenance, ...] = (
    Provenance.REFERENCE_PAGE,
    Provenance.TARGET_RESPONSE,
    Provenance.SCANNER_OUTPUT,
    Provenance.EXPLOIT_DB,
)

#: Case ids the sandbox is known to raise no signal for, with the remedy for each. These are
#: reported, not silently tolerated: the adversarial suite asserts that the set of undetected
#: attacks is a *subset* of this map, so a new evasion fails the build while a fix to the
#: sandbox does not.
#:
#: Seven entries were removed when ``Sandbox.sanitize`` began matching patterns against the
#: raw text as well as the normalised text and emitting synthetic ``nz_*`` signals for content
#: normalisation had removed (hidden elements, HTML comments, elided blobs). Those payloads
#: had always been neutralised; what was missing was the evidence that they had been. Detection
#: over the corpus rose from 0.891 to 0.978 on that change alone, with no new false positives.
KNOWN_PATTERN_GAPS: dict[str, str] = {
    # Plausible lies with no imperative anywhere in them. ADR-002 says explicitly that no
    # filter catches these - "the interesting attacks are not imperative sentences but
    # plausible false assertions" - and that influence budgets, not patterns, are the control.
    # Listed so the report can state the residual with a number rather than a footnote.
    "fi_009": "not catchable by pattern matching; bounded by the tier-3 influence budget",
    "fd_009": "not catchable by pattern matching; bounded by the tier-3 influence budget",
}


@dataclass(frozen=True)
class CorpusStats:
    """Counts over a loaded corpus, for the adversarial report and for integrity tests."""

    version: str
    n_cases: int
    n_attacks: int
    n_controls: int
    by_category: dict[InjectionCategory, int] = field(default_factory=dict)
    by_injection_point: dict[Provenance, int] = field(default_factory=dict)
    by_tier: dict[TrustTier, int] = field(default_factory=dict)
    by_language: dict[str, int] = field(default_factory=dict)
    by_goal: dict[str, int] = field(default_factory=dict)
    n_must_be_detected: int = 0

    @property
    def categories_covered(self) -> set[InjectionCategory]:
        """Categories with at least one case."""
        return {category for category, count in self.by_category.items() if count > 0}

    @property
    def missing_categories(self) -> set[InjectionCategory]:
        """Categories the corpus does not exercise at all."""
        return set(InjectionCategory) - self.categories_covered

    @property
    def languages_covered(self) -> set[str]:
        """Language tags present in the corpus."""
        return {language for language, count in self.by_language.items() if count > 0}

    def as_dict(self) -> dict[str, object]:
        """Plain-JSON view for the run report."""
        return {
            "version": self.version,
            "n_cases": self.n_cases,
            "n_attacks": self.n_attacks,
            "n_controls": self.n_controls,
            "n_must_be_detected": self.n_must_be_detected,
            "by_category": {key.value: value for key, value in sorted(self.by_category.items(), key=_category_key)},
            "by_injection_point": {
                key.value: value for key, value in sorted(self.by_injection_point.items(), key=_provenance_key)
            },
            "by_tier": {key.name: value for key, value in sorted(self.by_tier.items())},
            "by_language": dict(sorted(self.by_language.items())),
            "by_goal": dict(sorted(self.by_goal.items())),
        }


def _category_key(item: tuple[InjectionCategory, int]) -> str:
    return item[0].value


def _provenance_key(item: tuple[Provenance, int]) -> str:
    return item[0].value


def default_corpus_path(config: AdversarialConfig | None = None) -> Path:
    """Absolute path to the shipped corpus, resolved against the repository root."""
    relative = (config or AdversarialConfig()).corpus_path
    path = Path(relative)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _read_document(resolved: Path) -> dict[str, object]:
    if not resolved.exists():
        raise ConfigError(f"adversarial corpus not found: {resolved}")
    try:
        document = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"adversarial corpus is not valid YAML: {resolved}") from exc
    if not isinstance(document, dict):
        raise ConfigError(f"adversarial corpus must be a mapping: {resolved}")
    return document


def load_corpus(
    path: str | Path | None = None,
    *,
    strict: bool = True,
) -> tuple[str, list[AdversarialCase]]:
    """Load the corpus and return ``(version, cases)``.

    ``path`` defaults to :func:`default_corpus_path`. Each entry is validated against the
    frozen :class:`~vulnprio.core.models.AdversarialCase` model, so a corpus that drifts from
    the contract fails here rather than halfway through an evaluation run.

    ``strict`` (the default) additionally runs :func:`validate_corpus`, which enforces the
    size, coverage and uniqueness requirements. Pass ``strict=False`` only to inspect a
    deliberately partial corpus - never in the pipeline.
    """
    resolved = Path(path) if path is not None else default_corpus_path()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    document = _read_document(resolved)

    version = str(document.get("version", "")).strip()
    if not version:
        raise ConfigError(f"adversarial corpus has no version: {resolved}")

    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ConfigError(f"adversarial corpus has no cases: {resolved}")

    cases: list[AdversarialCase] = []
    for index, entry in enumerate(raw_cases):
        if not isinstance(entry, dict):
            raise ConfigError(f"corpus case #{index} is not a mapping in {resolved}")
        payload = entry.get("payload")
        if isinstance(payload, str):
            entry = {**entry, "payload": payload.strip()}
        try:
            cases.append(AdversarialCase.model_validate(entry))
        except ValidationError as exc:
            identifier = entry.get("case_id", f"#{index}")
            raise ConfigError(f"corpus case {identifier!r} is invalid in {resolved}: {exc}") from exc

    if strict:
        validate_corpus(cases, source=str(resolved))
    return version, cases


def validate_corpus(cases: Sequence[AdversarialCase], *, source: str = "<memory>") -> None:
    """Raise :class:`ConfigError` unless the corpus satisfies every stated requirement.

    Separated from :func:`load_corpus` so a caller assembling cases programmatically (the
    synthetic generator, a regression fixture) can run the same checks.
    """
    if not cases:
        raise ConfigError(f"adversarial corpus is empty: {source}")

    seen: set[str] = set()
    duplicates: list[str] = []
    for case in cases:
        identifier = case.case_id.strip()
        if not identifier:
            raise ConfigError(f"adversarial corpus contains a case with an empty case_id: {source}")
        if identifier in seen:
            duplicates.append(identifier)
        seen.add(identifier)
    if duplicates:
        raise ConfigError(f"duplicate adversarial case ids {sorted(set(duplicates))!r} in {source}")

    for case in cases:
        if not case.payload.strip():
            raise ConfigError(f"adversarial case {case.case_id!r} has an empty payload in {source}")
        if case.injection_point == Provenance.OPERATOR:
            raise ConfigError(
                f"adversarial case {case.case_id!r} injects at the operator tier in {source}"
            )
        if case.injection_point not in VALID_INJECTION_POINTS:
            raise ConfigError(
                f"adversarial case {case.case_id!r} uses injection point "
                f"{case.injection_point.value!r}, which is not an attacker-writable surface "
                f"(expected one of {[point.value for point in VALID_INJECTION_POINTS]}) in {source}"
            )
        is_control = case.category == InjectionCategory.BENIGN_CONTROL
        if is_control and case.goal != "none":
            raise ConfigError(
                f"benign control {case.case_id!r} declares goal {case.goal!r}; controls must use 'none' "
                f"in {source}"
            )
        if not is_control and case.goal == "none":
            raise ConfigError(
                f"attack case {case.case_id!r} declares goal 'none'; only benign controls may in {source}"
            )
        if is_control and case.expectation.must_be_detected:
            raise ConfigError(
                f"benign control {case.case_id!r} sets must_be_detected; controls must never be detected "
                f"in {source}"
            )

    attacks = attack_cases(cases)
    controls = benign_controls(cases)
    if len(attacks) < MIN_ATTACK_CASES:
        raise ConfigError(
            f"adversarial corpus has {len(attacks)} attack cases, at least {MIN_ATTACK_CASES} required "
            f"in {source}"
        )
    if len(controls) < MIN_BENIGN_CONTROLS:
        raise ConfigError(
            f"adversarial corpus has {len(controls)} benign controls, at least {MIN_BENIGN_CONTROLS} "
            f"required in {source}"
        )

    present = {case.category for case in cases}
    missing = [category.value for category in InjectionCategory if category not in present]
    if missing:
        raise ConfigError(f"adversarial corpus does not cover categories {sorted(missing)!r} in {source}")


def corpus_stats(version: str, cases: Sequence[AdversarialCase]) -> CorpusStats:
    """Summarise a loaded corpus for reporting."""
    by_category: Counter[InjectionCategory] = Counter(case.category for case in cases)
    by_point: Counter[Provenance] = Counter(case.injection_point for case in cases)
    by_tier: Counter[TrustTier] = Counter(tier_of(case.injection_point) for case in cases)
    by_language: Counter[str] = Counter(case.language for case in cases)
    by_goal: Counter[str] = Counter(case.goal for case in cases)
    return CorpusStats(
        version=version,
        n_cases=len(cases),
        n_attacks=len(attack_cases(cases)),
        n_controls=len(benign_controls(cases)),
        by_category=dict(by_category),
        by_injection_point=dict(by_point),
        by_tier=dict(by_tier),
        by_language=dict(by_language),
        by_goal=dict(by_goal),
        n_must_be_detected=sum(1 for case in cases if case.expectation.must_be_detected),
    )


def attack_cases(cases: Iterable[AdversarialCase]) -> list[AdversarialCase]:
    """Every case except the benign controls."""
    return [case for case in cases if case.category != InjectionCategory.BENIGN_CONTROL]


def benign_controls(cases: Iterable[AdversarialCase]) -> list[AdversarialCase]:
    """The negative class: cases that must never be flagged and must never move a feature."""
    return [case for case in cases if case.category == InjectionCategory.BENIGN_CONTROL]


def cases_in_category(cases: Iterable[AdversarialCase], category: InjectionCategory) -> list[AdversarialCase]:
    """All cases of one category, in corpus order."""
    return [case for case in cases if case.category == category]


def case_by_id(cases: Iterable[AdversarialCase], case_id: str) -> AdversarialCase | None:
    """Look one case up by id, or ``None``."""
    for case in cases:
        if case.case_id == case_id:
            return case
    return None
