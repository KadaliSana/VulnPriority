"""Loading and integrity-checking the prior-work corpus.

The corpus in ``data/novelty/prior_work.yaml`` is the evidence base for every novelty claim
this package makes. If it silently lost a study, or coded a dimension with a value the
analysis does not understand, the resulting verdict would be wrong in a way nobody would
notice - the matrix would still add up, the narrative would still read well. So every
structural defect here is a hard :class:`~vulnpriority.core.errors.ConfigError` at load time
rather than a skipped row.

What is checked:

* at least :data:`MIN_STUDIES` entries, and unique non-empty keys;
* every capability vector covers **every** dimension in
  :data:`~vulnpriority.novelty.capabilities.CAPABILITY_KEYS` exactly - no missing dimension
  defaulting quietly to ``unknown``, no dimension the capability module has never heard of;
* only the four allowed coding values;
* every study carries a description, dataset, headline result and limitation, so a reader can
  check the coding against the review;
* caveats attach to real dimensions.

The corpus is deliberately *not* the whole literature. It is one review's Table 1, and the
``caveats`` block in the YAML says so in the file itself. :func:`corpus_caveats` carries that
text through to the published verdict rather than leaving it in a comment nobody reads.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vulnpriority.core.config import PROJECT_ROOT
from vulnpriority.core.errors import ConfigError

from vulnpriority.novelty.capabilities import CAPABILITY_KEYS, CapabilityLevel

__all__ = [
    "MIN_STUDIES",
    "Paradigm",
    "StudyCaveat",
    "CorpusCaveat",
    "PriorStudy",
    "Corpus",
    "default_corpus_path",
    "load_corpus",
    "corpus_caveats",
    "studies_by_paradigm",
    "study",
]

#: The review's Table 1 lists 45 distinct studies. The floor is set below that so a
#: deliberate correction to the corpus does not break the suite, but a truncated or
#: half-written file does.
MIN_STUDIES: int = 40


class Paradigm(str, Enum):
    """The review's own methodological classification (Table 1)."""

    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    HYBRID = "hybrid"
    GRAPH = "graph"
    SURVEY = "survey"


class StudyCaveat(BaseModel):
    """A note on why one dimension was coded the way it was, or what it threatens.

    These are the sentences that keep the analysis honest: the reason a study was coded
    ``partial`` rather than ``has``, or the reason a study is a threat to a novelty claim
    even though it does not satisfy the dimension's test.
    """

    model_config = ConfigDict(frozen=True)

    dimension: str
    note: str


class CorpusCaveat(BaseModel):
    """A reason to distrust conclusions drawn from the corpus as a whole.

    Identified rather than anonymous, so a reviewer can refer to one by name and so the web
    layer can link to it.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class PriorStudy(BaseModel):
    """One reviewed study, as the literature review describes it."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(min_length=1)
    authors: str = Field(min_length=1)
    year: int = Field(ge=1990, le=2100)
    venue: str = Field(min_length=1)
    reference: str = Field(min_length=1)           # the review's own bibliography number
    paradigm: Paradigm
    description: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    headline_result: str = Field(min_length=1)
    limitation: str = Field(min_length=1)
    survey: bool = False
    capabilities: dict[str, CapabilityLevel]
    caveats: tuple[StudyCaveat, ...] = ()

    def level(self, dimension: str) -> CapabilityLevel:
        """This study's coding on one dimension."""
        try:
            return self.capabilities[dimension]
        except KeyError:  # pragma: no cover - load-time validation makes this unreachable
            raise KeyError(f"{self.key} has no coding for {dimension!r}") from None

    def has(self, dimension: str) -> bool:
        return self.level(dimension) is CapabilityLevel.HAS

    def capability_set(self) -> frozenset[str]:
        """Dimensions this study fully satisfies. The unit of the overlap comparison."""
        return frozenset(k for k, v in self.capabilities.items() if v is CapabilityLevel.HAS)

    def label(self) -> str:
        """Human-readable citation, e.g. ``Tita et al. (2026) [66]``."""
        return f"{self.authors} ({self.year}) {self.reference}"


class Corpus(BaseModel):
    """The reviewed prior work, plus the review-level caveats on reading it."""

    model_config = ConfigDict(frozen=True)

    version: int
    review_title: str
    review_corpus_note: str
    caveats: tuple[CorpusCaveat, ...]
    studies: tuple[PriorStudy, ...]

    def caveat_texts(self) -> tuple[str, ...]:
        return tuple(c.text for c in self.caveats)

    def __len__(self) -> int:
        return len(self.studies)

    def keys(self) -> tuple[str, ...]:
        return tuple(s.key for s in self.studies)

    def get(self, key: str) -> PriorStudy:
        for candidate in self.studies:
            if candidate.key == key:
                return candidate
        raise KeyError(f"no study with key {key!r}")

    def non_surveys(self) -> tuple[PriorStudy, ...]:
        """Studies with a method of their own. Surveys have nothing to code."""
        return tuple(s for s in self.studies if not s.survey)

    def counts(self, dimension: str) -> Counter[CapabilityLevel]:
        """Level counts for one dimension. Always sums to ``len(self)``."""
        return Counter(s.level(dimension) for s in self.studies)


def default_corpus_path() -> Path:
    return PROJECT_ROOT / "data" / "novelty" / "prior_work.yaml"


def _validate(corpus: Corpus, path: Path) -> None:
    """Every failure mode that would silently corrupt a novelty claim."""
    where = f"novelty corpus {path}"

    if len(corpus.studies) < MIN_STUDIES:
        raise ConfigError(
            f"{where}: {len(corpus.studies)} studies, expected at least {MIN_STUDIES}. "
            "The review's Table 1 lists 45."
        )

    duplicates = [key for key, n in Counter(corpus.keys()).items() if n > 1]
    if duplicates:
        raise ConfigError(f"{where}: duplicate study keys {sorted(duplicates)}")

    if not corpus.caveats:
        raise ConfigError(
            f"{where}: no corpus-level caveats. A novelty analysis with no stated limits on "
            "its own evidence base is not an honest one."
        )

    known = set(CAPABILITY_KEYS)
    for entry in corpus.studies:
        coded = set(entry.capabilities)
        unknown_dimensions = sorted(coded - known)
        if unknown_dimensions:
            raise ConfigError(
                f"{where}: study {entry.key!r} codes unknown dimensions {unknown_dimensions}"
            )
        missing = sorted(known - coded)
        if missing:
            raise ConfigError(
                f"{where}: study {entry.key!r} is missing codings for {missing}. Every study "
                "must be coded on every dimension, explicitly as 'unknown' where the review "
                "does not say, so the matrix counts are complete."
            )
        for caveat in entry.caveats:
            if caveat.dimension not in known:
                raise ConfigError(
                    f"{where}: study {entry.key!r} has a caveat on unknown dimension "
                    f"{caveat.dimension!r}"
                )


@lru_cache(maxsize=4)
def _load_cached(path_str: str) -> Corpus:
    path = Path(path_str)
    if not path.is_file():
        raise ConfigError(f"novelty corpus not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"novelty corpus {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"novelty corpus {path} must be a mapping at the top level")
    try:
        corpus = Corpus.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"novelty corpus {path} failed validation: {exc}") from exc
    _validate(corpus, path)
    return corpus


def load_corpus(path: Path | None = None) -> Corpus:
    """Load and integrity-check the prior-work corpus.

    Raises :class:`~vulnpriority.core.errors.ConfigError` on any structural defect.
    """
    return _load_cached(str(path or default_corpus_path()))


def corpus_caveats(path: Path | None = None) -> tuple[str, ...]:
    """The review-level reasons to distrust conclusions drawn from this corpus."""
    return load_corpus(path).caveat_texts()


def studies_by_paradigm(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Study keys grouped by the review's methodological classification."""
    corpus = load_corpus(path)
    out: dict[str, list[str]] = {p.value: [] for p in Paradigm}
    for entry in corpus.studies:
        out[entry.paradigm.value].append(entry.key)
    return {k: tuple(sorted(v)) for k, v in out.items()}


def study(key: str, path: Path | None = None) -> PriorStudy:
    """Look up one study by key."""
    return load_corpus(path).get(key)
