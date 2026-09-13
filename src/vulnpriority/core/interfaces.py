"""Abstract interfaces. Implementations live in their own subpackages.

Frozen contract: parallel implementers code against these signatures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any, Generic, Iterable, Protocol, TypeVar, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict

from vulnpriority.core.enums import (
    Component,
    FeedMode,
    LLMBackendKind,
    Provenance,
    RankerName,
    TrustTier,
)
from vulnpriority.core.models import (
    AssetCriticality,
    ApplicabilityAssessment,
    AttackGraphSummary,
    AttackerModel,
    ChainScore,
    Endpoint,
    EnrichedFinding,
    ExploitEvidence,
    ExploitabilityAssessment,
    Explanation,
    EpssRecord,
    FeatureFrame,
    Finding,
    GroundTruthLabel,
    ImpactModel,
    KevRecord,
    LLMAudit,
    LabelSet,
    ManipulationAlert,
    ReferenceDoc,
    SanitizationReport,
    Scan,
    Split,
    TechComponent,
    VulnIntel,
)

__all__ = [
    "ScannerParser",
    "FeedClient",
    "NvdFeed",
    "EpssFeed",
    "KevFeed",
    "ExploitFeed",
    "ReferenceFetcher",
    "FeedBundle",
    "IntelAssembler",
    "SandboxedPrompt",
    "Sanitizer",
    "LLMResult",
    "LLMBackend",
    "SemanticAssessor",
    "Enricher",
    "ChainScorer",
    "Ranker",
    "ProbabilityModel",
    "Splitter",
    "RankMetric",
    "ManipulationDetector",
]

T = TypeVar("T")
TModel = TypeVar("TModel", bound=BaseModel)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class ScannerParser(ABC):
    """Parses one scanner's native output into a :class:`Scan`."""

    name: str = "generic"

    @abstractmethod
    def parse(self, path: str | Path, app_id: str | None = None) -> Scan:
        ...

    @abstractmethod
    def sniff(self, path: str | Path) -> bool:
        """True when this parser recognises the file."""


# ---------------------------------------------------------------------------
# Feeds
# ---------------------------------------------------------------------------


class FeedClient(ABC, Generic[T]):
    """One external data feed.

    ``as_of`` is mandatory and implementations must never return information dated
    after it. Offline implementations must never open a socket.
    """

    name: str = "feed"
    mode: FeedMode = FeedMode.OFFLINE
    tier: TrustTier = TrustTier.CURATED_FEED

    @abstractmethod
    def get(self, key: str, as_of: date) -> T | None:
        ...

    def get_many(self, keys: Iterable[str], as_of: date) -> dict[str, T | None]:
        return {key: self.get(key, as_of) for key in keys}


class NvdFeed(FeedClient[VulnIntel]):
    """Returns a VulnIntel carrying description, CVSS records, affected products and reference URLs."""

    name = "nvd"


class EpssFeed(FeedClient[EpssRecord]):
    name = "epss"


class KevFeed(FeedClient[KevRecord]):
    name = "kev"


class ExploitFeed(FeedClient[tuple]):
    """Returns ``tuple[ExploitEvidence, ...]`` for a CVE id."""

    name = "exploitdb"


class ReferenceFetcher(FeedClient[ReferenceDoc]):
    """Key is a URL. Content is always untrusted (tier REFERENCE_PAGE)."""

    name = "references"
    tier = TrustTier.REFERENCE_PAGE


class FeedBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nvd: NvdFeed
    epss: EpssFeed
    kev: KevFeed
    exploits: ExploitFeed
    references: ReferenceFetcher
    mode: FeedMode = FeedMode.OFFLINE


class IntelAssembler(ABC):
    """Composes one VulnIntel per CVE from the whole feed bundle."""

    @abstractmethod
    def assemble(self, cve_id: str, as_of: date, max_references: int = 8) -> VulnIntel:
        ...


# ---------------------------------------------------------------------------
# Sandbox and LLM
# ---------------------------------------------------------------------------


class SandboxedPrompt(BaseModel):
    """Output of the sandbox. ``system`` is operator text and is hash-pinned."""

    model_config = ConfigDict(frozen=True)

    task: str
    system: str
    operator_context: str
    untrusted_blocks: tuple[tuple[str, str, Provenance], ...] = ()   # (segment_id, sanitized_text, provenance)
    reports: tuple[SanitizationReport, ...] = ()
    canary: str = ""
    nonce: str = ""
    prompt_hash: str = ""
    schema_name: str = ""

    @property
    def max_tier_used(self) -> TrustTier:
        tiers = [report.source_tier for report in self.reports]
        return max(tiers) if tiers else TrustTier.OPERATOR


@runtime_checkable
class Sanitizer(Protocol):
    """Normalises and neutralises untrusted text before it may enter a prompt."""

    def sanitize(self, text: str, tier: TrustTier, nonce: str) -> tuple[str, SanitizationReport]:
        ...

    def envelope(self, sanitized: str, tier: TrustTier, nonce: str, segment_id: str) -> str:
        ...


class LLMResult(BaseModel, Generic[TModel]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    parsed: TModel
    raw_text: str = ""
    audit: LLMAudit


class LLMBackend(ABC):
    """A structured-output text model. Implementations must be side-effect free and retryable."""

    kind: LLMBackendKind = LLMBackendKind.HEURISTIC
    model_id: str = "heuristic"

    @abstractmethod
    def complete_structured(self, prompt: SandboxedPrompt, schema: type[TModel]) -> LLMResult[TModel]:
        ...

    def available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Components A, B, C
# ---------------------------------------------------------------------------


class SemanticAssessor(ABC):
    """Component A."""

    @abstractmethod
    def assess_asset(self, endpoint: Endpoint, scan: Scan) -> AssetCriticality:
        ...

    @abstractmethod
    def assess_exploitability(
        self, finding: Finding, intel: tuple[VulnIntel, ...], endpoint: Endpoint
    ) -> ExploitabilityAssessment:
        ...

    @abstractmethod
    def assess_applicability(
        self, finding: Finding, intel: tuple[VulnIntel, ...], tech: tuple[TechComponent, ...]
    ) -> ApplicabilityAssessment:
        ...


class Enricher(ABC):
    """Component B."""

    @abstractmethod
    def enrich(
        self,
        finding: Finding,
        endpoint: Endpoint,
        intel: tuple[VulnIntel, ...],
        asset: AssetCriticality,
        exploitability: ExploitabilityAssessment,
        applicability: ApplicabilityAssessment,
        attacker: AttackerModel,
        impact_model: ImpactModel,
        as_of: date,
    ) -> EnrichedFinding:
        ...


class ChainScorer(ABC):
    """Component C."""

    @abstractmethod
    def build(self, scan: Scan, enriched: list[EnrichedFinding]) -> AttackGraphSummary:
        ...

    @abstractmethod
    def score(self, scan_id: str) -> dict[str, ChainScore]:
        ...

    @abstractmethod
    def total_risk_after_patching(self, scan_id: str, patched_finding_ids: set[str]) -> float:
        ...


# ---------------------------------------------------------------------------
# Ranking and evaluation
# ---------------------------------------------------------------------------


class Ranker(ABC):
    """Produces a score per row; higher means remediate sooner."""

    name: RankerName = RankerName.RANDOM

    @abstractmethod
    def fit(
        self,
        frame: FeatureFrame,
        relevance: np.ndarray,
        sample_weight: np.ndarray | None = None,
        seed: int = 42,
    ) -> "Ranker":
        ...

    @abstractmethod
    def score(self, frame: FeatureFrame) -> np.ndarray:
        ...

    def explain(self, frame: FeatureFrame, top_n: int = 5) -> list[Explanation] | None:
        return None

    def requires_fit(self) -> bool:
        return True

    def save(self, path: str | Path) -> None:  # pragma: no cover - optional capability
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "Ranker":  # pragma: no cover - optional capability
        raise NotImplementedError


class ProbabilityModel(ABC):
    """Calibrated P(exploit) head used for Brier/ECE and expected-loss ordering."""

    @abstractmethod
    def fit(self, frame: FeatureFrame, y: np.ndarray, seed: int = 42) -> "ProbabilityModel":
        ...

    @abstractmethod
    def predict_proba(self, frame: FeatureFrame) -> np.ndarray:
        ...


class Splitter(ABC):
    @abstractmethod
    def split(self, scans: list[Scan], labels: LabelSet) -> list[Split]:
        ...


@runtime_checkable
class RankMetric(Protocol):
    name: str

    def compute(self, ranked_ids: list[str], relevance: dict[str, float], k: int | None) -> float:
        ...


class ManipulationDetector(ABC):
    """Flags evidence that the ranking is being manipulated by untrusted content."""

    @abstractmethod
    def detect(self, enriched: list[EnrichedFinding], context: dict[str, Any] | None = None) -> list[ManipulationAlert]:
        ...
