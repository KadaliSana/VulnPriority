"""All shared data models.

Frozen contract. Every module in the package exchanges data using these types and
must not redefine them. Numeric fields that can be influenced by untrusted content are
bounded at the type level so that a prompt injection cannot express an out-of-range value.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vulnpriority.core.enums import (
    AgreementAxis,
    AsOfStatus,
    ApplicabilityVerdict,
    AttackComplexity,
    Component,
    CvssVersion,
    DetectorName,
    EndpointFunction,
    ExploitMaturity,
    ExploitSource,
    FeedMode,
    HttpMethod,
    InjectionCategory,
    InjectionVerdict,
    IntelQueryKind,
    IntelRemedy,
    IntelSourceKind,
    LLMBackendKind,
    LabelSource,
    MetricName,
    PrivilegeLevel,
    Provenance,
    RankerName,
    ScannerSeverity,
    ScoreSource,
    SelectionMethod,
    SplitKind,
    TrustTier,
    UserInteraction,
    VersionMatch,
    tier_of,
)
from vulnpriority.core.hashing import sha256_text
from vulnpriority.core.money import DEFAULT_CURRENCY

__all__ = [
    "Frozen",
    "UntrustedText",
    "TechComponent",
    "Endpoint",
    "Finding",
    "Scan",
    "CvssRecord",
    "EpssRecord",
    "KevRecord",
    "ExploitEvidence",
    "AffectedProduct",
    "ReferenceDoc",
    "VulnIntel",
    "InjectionSignal",
    "SanitizationReport",
    "LLMAudit",
    "AssetCriticality",
    "ExploitabilityAssessment",
    "ApplicabilityAssessment",
    "IntelQuery",
    "IntelDocument",
    "IntelCitation",
    "IntelUsage",
    "ExploitIntelOut",
    "IntelSummary",
    "FeedAgreement",
    "AsOfVerdict",
    "IntelResult",
    "INTEL_FEATURE_NAMES",
    "neutral_intel_features",
    "AttackerModel",
    "ImpactModel",
    "BusinessImpact",
    "RemediationCost",
    "ExploitLikelihood",
    "TrustSummary",
    "ComponentFlags",
    "EnrichedFinding",
    "GraphNode",
    "GraphEdge",
    "AttackPath",
    "ChainScore",
    "AttackGraphSummary",
    "FEATURE_SPECS",
    "FEATURE_NAMES",
    "FEATURE_GROUPS",
    "feature_names_for",
    "FeatureFrame",
    "FeatureContribution",
    "Explanation",
    "ManipulationAlert",
    "RankedFinding",
    "RankingResult",
    "GroundTruthLabel",
    "LabelPolicy",
    "LabelSet",
    "Split",
    "MetricValue",
    "CalibrationReport",
    "MinorityClassReport",
    "MetricBundle",
    "AblationCell",
    "AblationTable",
    "SelectionResult",
    "SimulationResult",
    "AdversarialExpectation",
    "AdversarialCase",
    "AdversarialOutcome",
    "AdversarialReport",
    "RunManifest",
]


class Frozen(BaseModel):
    """Immutable, strict base model."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Untrusted content
# ---------------------------------------------------------------------------


class UntrustedText(Frozen):
    """Text originating from the target application or the internet.

    Never rendered into a prompt except through ``vulnpriority.sandbox.pipeline.Sandbox``.
    ``sha256`` is filled on construction and ``tier`` is derived from provenance.
    """

    text: str
    provenance: Provenance
    source_url: str | None = None
    fetched_at: datetime | None = None
    language: str = "und"
    sha256: str = ""

    @model_validator(mode="after")
    def _fill_hash(self) -> "UntrustedText":
        if self.provenance == Provenance.OPERATOR:
            raise ValueError("OPERATOR text must not be wrapped as UntrustedText")
        if not self.sha256:
            object.__setattr__(self, "sha256", sha256_text(self.text))
        return self

    @property
    def tier(self) -> TrustTier:
        return tier_of(self.provenance)


# ---------------------------------------------------------------------------
# Ingest layer
# ---------------------------------------------------------------------------


class TechComponent(Frozen):
    vendor: str | None = None
    product: str
    version: str | None = None
    cpe: str | None = None


class Endpoint(Frozen):
    endpoint_id: str
    app_id: str
    host: str
    url: str
    path: str                                   # templated path, e.g. /users/{id}
    method: HttpMethod
    auth_required: PrivilegeLevel = PrivilegeLevel.NONE
    internet_facing: bool = True
    response_status: int | None = None
    response_content_type: str | None = None
    response_size_bytes: int | None = None
    sets_cookie: bool = False
    parameters: tuple[str, ...] = ()
    links_to: tuple[str, ...] = ()              # endpoint_ids observed as outgoing links (tier SCANNER)
    response_sample: UntrustedText | None = None
    observed_tech: tuple[TechComponent, ...] = ()


class Finding(Frozen):
    finding_id: str
    scan_id: str
    app_id: str
    endpoint_id: str
    name: str
    cwe_id: int | None = None
    cve_ids: tuple[str, ...] = ()
    scanner: str
    scanner_plugin_id: str | None = None
    scanner_severity: ScannerSeverity
    scanner_confidence: float = Field(0.5, ge=0.0, le=1.0)
    description: UntrustedText
    evidence: tuple[UntrustedText, ...] = ()    # request / response artefacts
    affected_component: TechComponent | None = None
    observed_at: datetime
    dedup_key: str | None = None                # set by ingest.correlate; same root cause across endpoints
    cluster_size: int = Field(1, ge=1)          # number of findings sharing dedup_key in this scan


class Scan(Frozen):
    scan_id: str
    app_id: str
    app_name: str
    sector: str = "generic"                     # ecommerce / healthcare / saas / fintech (Gap 8 transfer axis)
    scanned_at: datetime
    scanner_name: str
    scanner_version: str | None = None
    hosts: tuple[str, ...] = ()
    tech_stack: tuple[TechComponent, ...] = ()
    endpoints: tuple[Endpoint, ...] = ()
    findings: tuple[Finding, ...] = ()

    def endpoint_by_id(self, endpoint_id: str) -> Endpoint | None:
        for endpoint in self.endpoints:
            if endpoint.endpoint_id == endpoint_id:
                return endpoint
        return None


# ---------------------------------------------------------------------------
# Feed / intelligence layer
# ---------------------------------------------------------------------------


class CvssRecord(Frozen):
    version: CvssVersion
    source: ScoreSource
    base_score: float = Field(ge=0.0, le=10.0)
    vector: str | None = None
    severity: str | None = None
    submetrics: dict[str, str] = Field(default_factory=dict)   # AV, AC, PR, UI, C, I, A


class EpssRecord(Frozen):
    cve_id: str
    score: float = Field(ge=0.0, le=1.0)
    percentile: float = Field(ge=0.0, le=1.0)
    as_of: date


class KevRecord(Frozen):
    cve_id: str
    in_kev: bool
    date_added: date | None = None
    due_date: date | None = None
    known_ransomware_use: bool = False
    as_of: date


class ExploitEvidence(Frozen):
    source: ExploitSource
    url: str | None = None
    published: date | None = None
    maturity: ExploitMaturity = ExploitMaturity.UNKNOWN
    verified: bool = False
    language: str = "en"
    title: UntrustedText | None = None


class AffectedProduct(Frozen):
    cpe: str
    version_start_including: str | None = None
    version_start_excluding: str | None = None
    version_end_including: str | None = None
    version_end_excluding: str | None = None


class ReferenceDoc(Frozen):
    url: str
    title: str | None = None
    tags: tuple[str, ...] = ()
    content: UntrustedText
    language: str = "und"


class VulnIntel(Frozen):
    """Everything known about one CVE as of ``as_of``. No later information may be present."""

    cve_id: str
    as_of: date
    description: UntrustedText | None = None
    published: date | None = None
    last_modified: date | None = None
    cvss: tuple[CvssRecord, ...] = ()
    epss: EpssRecord | None = None
    kev: KevRecord | None = None
    exploits: tuple[ExploitEvidence, ...] = ()
    affected: tuple[AffectedProduct, ...] = ()
    references: tuple[ReferenceDoc, ...] = ()

    @model_validator(mode="after")
    def _no_future_leakage(self) -> "VulnIntel":
        if self.epss is not None and self.epss.as_of > self.as_of:
            raise ValueError("EPSS snapshot is dated after the intel as_of date")
        if self.kev is not None and self.kev.in_kev and self.kev.date_added and self.kev.date_added > self.as_of:
            raise ValueError("KEV membership leaks information from after as_of")
        if self.published is not None and self.published > self.as_of:
            raise ValueError("CVE publication date is after as_of")
        for exploit in self.exploits:
            if exploit.published is not None and exploit.published > self.as_of:
                raise ValueError("exploit evidence is dated after as_of")
        return self


# ---------------------------------------------------------------------------
# Sandbox / LLM audit
# ---------------------------------------------------------------------------


class InjectionSignal(Frozen):
    pattern_id: str
    category: InjectionCategory
    snippet: str = Field(max_length=200)
    tier: TrustTier


class SanitizationReport(Frozen):
    source_tier: TrustTier
    nonce: str
    original_length: int = Field(ge=0)
    sanitized_length: int = Field(ge=0)
    signals: tuple[InjectionSignal, ...] = ()
    stripped_patterns: tuple[str, ...] = ()
    hidden_text_removed: int = Field(0, ge=0)
    homoglyphs_folded: int = Field(0, ge=0)
    base64_blobs_elided: int = Field(0, ge=0)
    truncated: bool = False
    verdict: InjectionVerdict = InjectionVerdict.CLEAN

    @property
    def signal_count(self) -> int:
        return len(self.signals)


class LLMAudit(Frozen):
    """One record per model call. Always attached to whatever the call produced."""

    backend: LLMBackendKind
    model: str
    task: str = ""
    prompt_hash: str = ""
    canary_leaked: bool = False
    envelope_broken: bool = False
    schema_retries: int = Field(0, ge=0)
    evidence_span_failures: int = Field(0, ge=0)
    signals: tuple[InjectionSignal, ...] = ()
    max_tier_used: TrustTier = TrustTier.OPERATOR
    consistency_shrunk: bool = False
    divergence: float = Field(0.0, ge=0.0)
    fell_back_to_heuristic: bool = False
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    latency_ms: float = Field(0.0, ge=0.0)
    cached: bool = False


# ---------------------------------------------------------------------------
# Component A outputs
# ---------------------------------------------------------------------------


class AssetCriticality(Frozen):
    """Goal 1: endpoint criticality inferred from structure, never from manual asset tags."""

    endpoint_id: str
    function: EndpointFunction
    criticality: float = Field(ge=0.0, le=1.0)
    data_sensitivity: float = Field(ge=0.0, le=1.0)
    exposure: float = Field(ge=0.0, le=1.0)          # 1.0 = reachable anonymously from the internet
    is_auth_boundary: bool = False
    is_admin_surface: bool = False
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    rationale: str = Field("", max_length=600)
    evidence_spans: tuple[str, ...] = ()
    evidence_features: dict[str, float] = Field(default_factory=dict)
    audit: LLMAudit | None = None


class ExploitabilityAssessment(Frozen):
    """Goal 2 / Architecture item 1: exploit feasibility, maturity, complexity and impact."""

    finding_id: str
    exploit_feasibility: float = Field(ge=0.0, le=1.0)
    exploit_maturity: ExploitMaturity = ExploitMaturity.UNKNOWN
    attack_complexity: AttackComplexity = AttackComplexity.UNKNOWN
    privileges_required: PrivilegeLevel = PrivilegeLevel.NONE
    user_interaction: UserInteraction = UserInteraction.UNKNOWN
    preconditions: tuple[str, ...] = ()
    impact_c: float = Field(0.0, ge=0.0, le=1.0)
    impact_i: float = Field(0.0, ge=0.0, le=1.0)
    impact_a: float = Field(0.0, ge=0.0, le=1.0)
    privilege_gained: PrivilegeLevel = PrivilegeLevel.NONE   # post-condition for the attack graph
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    rationale: str = Field("", max_length=600)
    evidence_spans: tuple[str, ...] = ()
    audit: LLMAudit | None = None

    @property
    def impact_cia_mean(self) -> float:
        return (self.impact_c + self.impact_i + self.impact_a) / 3.0


class ApplicabilityAssessment(Frozen):
    """Goal 3 / Architecture item 2: is this finding actually applicable to the observed app?"""

    finding_id: str
    verdict: ApplicabilityVerdict = ApplicabilityVerdict.UNCERTAIN
    p_applicable: float = Field(0.5, ge=0.0, le=1.0)
    version_match: VersionMatch = VersionMatch.UNKNOWN
    preconditions_met: dict[str, bool] = Field(default_factory=dict)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    rationale: str = Field("", max_length=600)
    evidence_spans: tuple[str, ...] = ()
    audit: LLMAudit | None = None


# ---------------------------------------------------------------------------
# Internet exploit intelligence (Component A; produced by vulnpriority.intel)
# ---------------------------------------------------------------------------
#
# These types are the contract between the retrieval layer and everything that consumes
# it -- the feature builder, the SHAP explainer, the rank guard and the report. They live
# here rather than in ``vulnpriority.intel`` for the same reason every other inter-package
# type does: ``EnrichedFinding`` carries an ``IntelResult``, ``EnrichedFinding`` lives in
# core, and core cannot import a sibling package without inverting the layering. Nothing
# below imports ``vulnpriority.intel``, ``anthropic`` or ``httpx``; core stays dependency-light.

#: Caps on everything a fetched page or a model may contribute. Each exists so that
#: retrieved content cannot turn a summary or an extraction into a smuggling channel.
MAX_INTEL_CITATIONS = 12
MAX_CITED_TEXT_CHARS = 400
MAX_INTEL_SUMMARY_CHARS = 1400
MAX_CLAIMED_VERSIONS = 8
MAX_EXPLOIT_URLS = 8
MAX_INTEL_URL_CHARS = 500
MAX_INTEL_PRECONDITIONS = 8

IntelUnit = Annotated[float, Field(ge=0.0, le=1.0)]


class IntelQuery(Frozen):
    """One search the retrieval layer intends to run.

    Queries are operator tier: they are assembled from structured facts (the CVE id, the
    observed product and version, the weakness class), never from fetched text. A query
    built from untrusted text would let a page choose what the framework reads next.
    """

    text: str = Field(max_length=300)
    kind: IntelQueryKind = IntelQueryKind.CVE
    cve_id: str | None = None
    finding_id: str | None = None
    max_results: int = Field(5, ge=1, le=20)

    @property
    def key(self) -> str:
        """Normalised text, used for deduplication and for the cache key."""
        return " ".join(self.text.lower().split())


class IntelDocument(Frozen):
    """One page retrieved from the internet, as untrusted text plus its provenance.

    ``snippet`` must carry ``Provenance.REFERENCE_PAGE``. That is enforced rather than
    documented: a retrieved page must not be able to enter the pipeline wearing a more
    trusted badge than it earned, and the explanation layer's tiering of the intel
    features depends on this invariant holding.

    ``retrieved_at`` is mandatory. A live search returns today's internet; without a
    retrieval timestamp on every document there is no way to tell afterwards whether a
    historical scan was scored against evidence that did not exist at the time.
    """

    url: str = Field(max_length=MAX_INTEL_URL_CHARS)
    title: str | None = Field(None, max_length=300)
    snippet: UntrustedText
    retrieved_at: datetime
    source_kind: IntelSourceKind = IntelSourceKind.UNKNOWN
    relevance: IntelUnit = 0.5
    query_text: str = Field("", max_length=300)

    @model_validator(mode="after")
    def _snippet_is_a_reference_page(self) -> "IntelDocument":
        if self.snippet.provenance != Provenance.REFERENCE_PAGE:
            raise ValueError(
                "an IntelDocument snippet is text fetched from the internet and must carry "
                f"Provenance.REFERENCE_PAGE, not {self.snippet.provenance.value}"
            )
        return self

    @property
    def tier(self) -> TrustTier:
        return self.snippet.tier


class IntelCitation(Frozen):
    """A span of a retrieved page that a model quoted in its prose."""

    url: str = Field(max_length=MAX_INTEL_URL_CHARS)
    cited_text: str = Field("", max_length=MAX_CITED_TEXT_CHARS)
    title: str | None = Field(None, max_length=300)


class IntelUsage(Frozen):
    """Token and server-tool accounting for one or more model calls.

    Retrieval is the first part of the framework that spends money, so usage is recorded
    per call and summed rather than estimated afterwards from a log.
    """

    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    cache_read_tokens: int = Field(0, ge=0)
    web_search_requests: int = Field(0, ge=0)
    calls: int = Field(0, ge=0)

    def __add__(self, other: "IntelUsage") -> "IntelUsage":
        if not isinstance(other, IntelUsage):
            return NotImplemented
        return IntelUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            web_search_requests=self.web_search_requests + other.web_search_requests,
            calls=self.calls + other.calls,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens

    def cost_usd(self, input_per_mtok: float = 5.0, output_per_mtok: float = 25.0) -> float:
        """Rough dollar cost. Defaults are the published Claude Opus 5 rates.

        Cache reads are billed at a fraction of the input rate; charging them at the full
        input rate here over-states rather than under-states the bill.
        """
        return (
            (self.input_tokens + self.cache_read_tokens) * input_per_mtok
            + self.output_tokens * output_per_mtok
        ) / 1_000_000.0


class ExploitIntelOut(Frozen):
    """The bounded schema a model may emit when turning retrieved prose into numbers.

    Deliberately narrower than :class:`ExploitabilityAssessment`. There is no
    ``privilege_gained``, because that becomes an attack-graph edge and untrusted text does
    not create graph structure. There are no CIA impacts, because CVSS submetrics are
    curated-feed facts and this schema describes what a blog post said.

    ``active_exploitation_claimed`` is named as a *claim* on purpose: the authoritative
    answer to "is this exploited" is CISA KEV, and a page asserting otherwise must never be
    able to masquerade as one.
    """

    exploit_maturity: ExploitMaturity = ExploitMaturity.UNKNOWN
    exploit_feasibility: IntelUnit = 0.5
    attack_complexity: AttackComplexity = AttackComplexity.UNKNOWN
    preconditions: Annotated[
        tuple[Annotated[str, Field(max_length=200)], ...],
        Field(max_length=MAX_INTEL_PRECONDITIONS),
    ] = ()
    affected_versions_claimed: Annotated[
        tuple[Annotated[str, Field(max_length=120)], ...],
        Field(max_length=MAX_CLAIMED_VERSIONS),
    ] = ()
    public_exploit_urls: Annotated[
        tuple[Annotated[str, Field(max_length=MAX_INTEL_URL_CHARS)], ...],
        Field(max_length=MAX_EXPLOIT_URLS),
    ] = ()
    active_exploitation_claimed: bool = False
    confidence: IntelUnit = 0.5
    rationale: str = Field("", max_length=600)
    evidence_spans: Annotated[
        tuple[Annotated[str, Field(max_length=200)], ...], Field(max_length=5)
    ] = ()


class IntelSummary(Frozen):
    """Model-written prose about what is publicly known, with its citations.

    ``is_model_written`` is ``Literal[True]`` so a renderer cannot present this as
    framework prose; DESIGN 6.2 says no language model writes the report, and this is the
    one labelled exception.

    ``citations`` has ``min_length=1``. A summary of "what is publicly known" with no
    citation is the model's prior rather than intelligence, and the illegal state is made
    unrepresentable rather than validated somewhere and hoped for.
    """

    text: str = Field(max_length=MAX_INTEL_SUMMARY_CHARS)
    citations: Annotated[
        tuple[IntelCitation, ...], Field(min_length=1, max_length=MAX_INTEL_CITATIONS)
    ]
    model: str = ""
    prompt_hash: str = ""
    usage: IntelUsage = IntelUsage()
    generated_at: datetime | None = None
    #: True when the prose was replayed from a recorded corpus rather than written during
    #: this run. It was still written by a model -- at recording time -- which is why
    #: ``is_model_written`` stays True; a report should say both.
    recorded: bool = False
    is_model_written: Literal[True] = True

    @property
    def cited_urls(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(citation.url for citation in self.citations))


class FeedAgreement(Frozen):
    """How retrieved internet material compares with tier <= 1 evidence.

    The most valuable thing the retrieval layer produces. A page claiming active
    exploitation of something CISA also lists is corroboration; the same sentence about
    something no feed has flagged is an unverified claim; a page insisting a KEV entry is a
    false positive is an attempt at deflation. All three are identical to a ranker handed
    only the claim, so the comparison is computed and handed over as its own signal.

    ``corroborates`` and ``contradicts`` are not mutually exclusive: material can agree
    about exploitation and disagree about affected versions at once, and flattening that
    into one axis would discard the case most worth seeing.
    """

    kev: AgreementAxis = AgreementAxis.UNCHECKED
    exploit_records: AgreementAxis = AgreementAxis.UNCHECKED
    affected_versions: AgreementAxis = AgreementAxis.UNCHECKED
    corroborates: bool = False
    contradicts: bool = False
    reason: str = Field("", max_length=400)

    @property
    def axes(self) -> dict[str, AgreementAxis]:
        return {
            "kev": self.kev,
            "exploit_records": self.exploit_records,
            "affected_versions": self.affected_versions,
        }


class AsOfVerdict(Frozen):
    """The as-of decision for one gathering, and the action that fixes it.

    Two different runs need two different answers. An operational run scans a target now
    and gathers intelligence now: the two agree and live search is the point. A research
    run replays a historical scan for a time-ordered evaluation, and searching today's
    internet imports knowledge that did not exist at the time, silently inflating every
    metric in the protocol.

    The discriminator is the age of the scan, not a flag someone must remember to set;
    ``research_mode`` decides only what happens once a scan is found stale -- offer a
    re-scan, or refuse outright because the correct action there is a recorded corpus.
    """

    status: AsOfStatus = AsOfStatus.NOT_APPLICABLE
    remedy: IntelRemedy = IntelRemedy.NONE
    search_allowed: bool = True
    as_of: date | None = None
    evaluated_at: date | None = None
    age_days: int = Field(0, ge=0)
    max_age_days: int = Field(0, ge=0)
    research_mode: bool = False
    message: str = Field("", max_length=600)

    @property
    def evidence_is_current(self) -> bool:
        """True when the intelligence and the scan describe the same moment."""
        return self.status in (AsOfStatus.CURRENT, AsOfStatus.NOT_APPLICABLE)


#: The Component A feature columns the retrieval layer contributes, in emission order.
#: They belong to Component A's group in ``FEATURE_SPECS`` so an ablation cell that
#: disables A drops them entirely rather than zeroing them.
#:
#: The architecture's central claim is here: the model does not decide priority, it
#: extracts features, and the learned ranker decides what each is worth alongside CVSS,
#: EPSS and KEV. A summary in a report that no number depends on would be decoration.
INTEL_FEATURE_NAMES: tuple[str, ...] = (
    "a_intel_documents",
    "a_intel_public_exploit_urls",
    "a_intel_active_exploitation",
    "a_intel_confidence",
    "a_intel_corroborates_feeds",
    "a_intel_contradicts_feeds",
    "a_intel_injection_signals",
)


def neutral_intel_features() -> dict[str, float]:
    """Feature values meaning "no evidence either way".

    Used when retrieval is disabled, unavailable, or found nothing. All three are the same
    state -- the framework learned nothing from the internet about this finding -- and must
    produce identical numbers. In particular a finding for which retrieval never ran must
    not score differently from one where it ran and returned nothing: neither is evidence
    of absence, and a ranker handed a distinction that does not exist will learn it.
    """
    return {name: 0.0 for name in INTEL_FEATURE_NAMES}


class IntelResult(Frozen):
    """Everything the retrieval layer knows about one finding.

    ``errors`` is for failures that were survived -- no key, no network, a server-tool
    error, a schema rejection, a canary leak. ``skipped_reason`` is for the ordinary case
    of the feature being switched off, or of a stale scan offering a re-scan, neither of
    which is a failure and neither of which should read like one.

    Every field the feature layer needs is stored rather than derived on demand, so a
    result read back from a cache or a run artefact yields the same numbers without
    re-running the reasoning that produced them.
    """

    finding_id: str
    cve_id: str | None = None
    as_of: date
    queries: tuple[IntelQuery, ...] = ()
    documents: tuple[IntelDocument, ...] = ()
    extraction: ExploitIntelOut | None = None
    summary: IntelSummary | None = None
    agreement: FeedAgreement = FeedAgreement()
    #: The as-of decision and, when there was a problem, the action that fixes it. Always
    #: present, so a report read months later shows whether the intelligence was current
    #: when it was gathered or was collected against an already-stale scan.
    as_of_verdict: AsOfVerdict = AsOfVerdict()
    #: Distinct proof-of-concept / exploit URLs found, from the extraction and from the
    #: retrieved documents' kinds. Stored because it is defined even when nothing was
    #: extracted.
    public_exploit_urls: tuple[str, ...] = ()
    active_exploitation_claimed: bool = False
    usage: IntelUsage = IntelUsage()
    errors: tuple[str, ...] = ()
    skipped_reason: str = ""
    provider: str = ""
    model: str = ""
    injection_signals: int = Field(0, ge=0)
    max_tier_used: TrustTier = TrustTier.OPERATOR
    influence_used: dict[str, float] = Field(default_factory=dict)
    canary_leaked: bool = False
    fell_back_to_baseline: bool = False
    cache_hit: bool = False

    @property
    def found_anything(self) -> bool:
        return bool(self.documents)

    @property
    def confidence(self) -> float:
        """Extraction confidence, or 0.0 when nothing was extracted."""
        return float(self.extraction.confidence) if self.extraction is not None else 0.0

    @property
    def evidence_is_current(self) -> bool:
        """True when the intelligence describes the same moment the scan does."""
        return self.as_of_verdict.evidence_is_current

    @property
    def remedy(self) -> IntelRemedy:
        """What the caller should do about this result, if anything."""
        return self.as_of_verdict.remedy

    def feature_values(self) -> dict[str, float]:
        """The Component A intel features for this finding. Authoritative.

        Counts are returned raw; the feature layer applies its own ``log1p`` to
        ``a_intel_documents`` and nothing else. ``a_intel_injection_signals`` is a feature
        on purpose: the ranker is told how much attempted manipulation arrived attached to
        the evidence, so it can learn to discount such evidence rather than the framework
        either trusting it silently or dropping it silently.
        """
        if not self.found_anything and self.extraction is None:
            return neutral_intel_features()
        return {
            "a_intel_documents": float(len(self.documents)),
            "a_intel_public_exploit_urls": float(len(self.public_exploit_urls)),
            "a_intel_active_exploitation": 1.0 if self.active_exploitation_claimed else 0.0,
            "a_intel_confidence": self.confidence,
            "a_intel_corroborates_feeds": 1.0 if self.agreement.corroborates else 0.0,
            "a_intel_contradicts_feeds": 1.0 if self.agreement.contradicts else 0.0,
            "a_intel_injection_signals": float(self.injection_signals),
        }

    @staticmethod
    def features_for(result: "IntelResult | None") -> dict[str, float]:
        """Feature values for a finding, whether or not retrieval ran for it.

        Lets the call site in the feature layer be unconditional -- no ``if result is not
        None`` branch to get subtly wrong, and no way to emit a different set of columns
        for a finding the layer skipped.
        """
        return neutral_intel_features() if result is None else result.feature_values()

    def canonical_json(self) -> str:
        """Identity of the result, excluding whether this copy came from a cache.

        A cache hit is a property of the retrieval, not of the intelligence; excluding it
        is what lets a warm result be byte-identical to the cold one it reproduces.
        """
        return self.model_dump_json(exclude={"cache_hit"})


# ---------------------------------------------------------------------------
# Component B: attacker model, impact, decision-theoretic priority
# ---------------------------------------------------------------------------


class AttackerModel(Frozen):
    """The adversary, as code rather than prose (Gap 2).

    ``p_exploit`` is a logistic model over named evidence terms so that every
    contribution is auditable: ``P = sigmoid(w_intercept + sum_i w_i * x_i)``.
    """

    name: str
    description: str = ""
    skill: float = Field(0.5, ge=0.0, le=1.0)
    resources: float = Field(0.5, ge=0.0, le=1.0)
    entry_privilege: PrivilegeLevel = PrivilegeLevel.NONE
    horizon_days: int = Field(90, ge=1)
    max_chain_length: int = Field(4, ge=1)
    controls_target_content: bool = False        # red-team preset: the adversary authors target text
    target_preference: dict[EndpointFunction, float] = Field(default_factory=dict)
    w_intercept: float = -4.0
    w_epss_logit: float = 1.0
    w_kev: float = 2.5
    w_kev_ransomware: float = 0.5
    w_exploit_maturity: dict[ExploitMaturity, float] = Field(
        default_factory=lambda: {
            ExploitMaturity.UNKNOWN: 0.0,
            ExploitMaturity.UNPROVEN: 0.0,
            ExploitMaturity.POC: 0.8,
            ExploitMaturity.FUNCTIONAL: 1.6,
            ExploitMaturity.WEAPONIZED: 2.4,
        }
    )
    w_feasibility: float = 1.5
    w_applicability: float = 2.0
    w_exposure: float = 1.0
    w_asset_criticality: float = 0.8
    w_complexity_high: float = -0.8
    w_user_interaction: float = -0.5
    w_privileges_required: float = -0.6
    w_skill: float = 0.5
    w_resources: float = 0.3
    min_p: float = Field(0.0, ge=0.0, le=0.5)
    max_p: float = Field(0.99, ge=0.5, le=1.0)


class ImpactModel(Frozen):
    """Monetary impact parameters (Gap 9: business impact is quantified, not assumed).

    No field here names a currency, because only one of them decides: ``currency`` is the
    authority, and every formatter downstream asks it rather than a field name. That is
    the whole reason the money fields are called ``cost_per_record`` and ``max_impact``
    and not ``cost_per_record_usd`` -- a field whose name asserts a denomination is a
    field that can lie about one.

    The defaults mirror ``configs/impact_models/default_ecommerce.yaml``, where the
    provenance of each number is recorded: ``cost_per_record`` is sourced (IBM Cost of a
    Data Breach Report, India 2026: Rs 25.5 crore over 39,500 records), ``max_impact`` is
    one average Indian breach from the same report, and the rest are proportional priors
    anchored on that per-record figure rather than independent measurements.
    """

    name: str = "default"
    currency: str = DEFAULT_CURRENCY
    cost_per_record: float = Field(6_456.0, ge=0.0)
    records_by_function: dict[EndpointFunction, int] = Field(default_factory=dict)
    downtime_cost_per_hour: float = Field(196_000.0, ge=0.0)
    downtime_hours_by_privilege: dict[PrivilegeLevel, float] = Field(default_factory=dict)
    integrity_loss_by_function: dict[EndpointFunction, float] = Field(default_factory=dict)
    regulatory_multiplier: float = Field(1.0, ge=0.0)
    reputational_fraction: float = Field(0.2, ge=0.0, le=5.0)
    max_impact: float = Field(255_000_000.0, gt=0.0)
    asset_overrides: dict[str, float] = Field(default_factory=dict)   # endpoint_id -> operator value


class BusinessImpact(Frozen):
    finding_id: str
    confidentiality: float = Field(0.0, ge=0.0)
    integrity: float = Field(0.0, ge=0.0)
    availability: float = Field(0.0, ge=0.0)
    reputational: float = Field(0.0, ge=0.0)
    total: float = Field(0.0, ge=0.0)
    rationale: str = Field("", max_length=400)


class RemediationCost(Frozen):
    finding_id: str
    hours: float = Field(gt=0.0)
    cost: float = Field(ge=0.0)
    basis: str = ""


class ExploitLikelihood(Frozen):
    finding_id: str
    attacker: str
    p_exploit: float = Field(ge=0.0, le=1.0)
    p_exploit_uncapped: float = Field(ge=0.0, le=1.0)
    log_odds_terms: dict[str, float] = Field(default_factory=dict)   # audit trail, one entry per weighted term
    horizon_days: int = Field(ge=1)


class TrustSummary(Frozen):
    """How much untrusted evidence was allowed to move this finding."""

    max_tier_used: TrustTier = TrustTier.OPERATOR
    injection_signal_count: int = Field(0, ge=0)
    conflicts: tuple[str, ...] = ()
    influence_used: dict[str, float] = Field(default_factory=dict)   # feature name -> applied |delta|
    caps_applied: dict[str, float] = Field(default_factory=dict)     # feature name -> cap that bound it
    floor_p_exploit: float = Field(0.0, ge=0.0, le=1.0)              # tier<=1 floor, untrusted text cannot go below
    corroborated: bool = False
    canary_leaked: bool = False


class ComponentFlags(Frozen):
    a: bool = True
    b: bool = True
    c: bool = True

    def label(self) -> str:
        return "".join(name for name, on in (("A", self.a), ("B", self.b), ("C", self.c)) if on) or "none"

    def enabled(self) -> set[Component]:
        out: set[Component] = set()
        if self.a:
            out.add(Component.A)
        if self.b:
            out.add(Component.B)
        if self.c:
            out.add(Component.C)
        return out

    @classmethod
    def all_cells(cls) -> list["ComponentFlags"]:
        """The eight cells of the 2^3 factorial ablation (Gap 5)."""
        return [cls(a=a, b=b, c=c) for a in (True, False) for b in (True, False) for c in (True, False)]


class EnrichedFinding(Frozen):
    """One finding with everything Components A and B produced for it."""

    finding: Finding
    endpoint: Endpoint
    intel: tuple[VulnIntel, ...] = ()
    asset: AssetCriticality
    exploitability: ExploitabilityAssessment
    applicability: ApplicabilityAssessment
    likelihood: ExploitLikelihood
    impact: BusinessImpact
    remediation: RemediationCost
    #: Internet exploit intelligence for this finding, when the retrieval layer ran.
    #: ``None`` means it did not; consumers must call ``IntelResult.features_for`` rather
    #: than branching, so "disabled" and "found nothing" produce identical numbers.
    intel_result: IntelResult | None = None
    expected_loss: float = Field(0.0, ge=0.0)      # p_exploit * impact.total (the priority construct)
    trust: TrustSummary = TrustSummary()
    alerts: tuple["ManipulationAlert", ...] = ()
    flags: ComponentFlags = ComponentFlags()
    as_of: date

    @property
    def finding_id(self) -> str:
        return self.finding.finding_id

    @property
    def scan_id(self) -> str:
        return self.finding.scan_id


# ---------------------------------------------------------------------------
# Component C: attack graph
# ---------------------------------------------------------------------------


class GraphNode(Frozen):
    node_id: str                                  # "state:<asset>:<privilege name>"
    asset: str
    privilege: PrivilegeLevel
    value: float = Field(0.0, ge=0.0)
    is_entry: bool = False
    is_target: bool = False


class GraphEdge(Frozen):
    src: str
    dst: str
    probability: float = Field(ge=0.0, le=1.0)
    finding_id: str | None = None                 # None for structural edges (privilege implication, links)
    kind: Literal["exploit", "privilege_implication", "lateral"] = "exploit"
    tier: TrustTier = TrustTier.SCANNER           # builder rejects edges from tiers above SCANNER


class AttackPath(Frozen):
    nodes: tuple[str, ...]
    finding_ids: tuple[str, ...]
    probability: float = Field(ge=0.0, le=1.0)
    target_value: float = Field(0.0, ge=0.0)

    @property
    def expected_value(self) -> float:
        return self.probability * self.target_value


class ChainScore(Frozen):
    """Per-finding contribution to reachable compromise (Gap 9)."""

    finding_id: str
    reach_delta: float = Field(0.0, ge=0.0)   # R(G) - R(G without this finding); non-negative by monotonicity
    max_path_prob_to_target: float = Field(0.0, ge=0.0, le=1.0)
    n_paths_through: int = Field(0, ge=0)
    betweenness: float = Field(0.0, ge=0.0)
    hops_from_entry: int = Field(0, ge=0)
    privilege_gain: int = Field(0, ge=0)
    is_chokepoint: bool = False
    best_target: str | None = None


class AttackGraphSummary(Frozen):
    scan_id: str
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    entry_node: str = "state:internet:NONE"
    target_nodes: tuple[str, ...] = ()
    total_risk: float = Field(0.0, ge=0.0)
    top_paths: tuple[AttackPath, ...] = ()
    monotone_verified: bool = False
    rejected_untrusted_edges: int = Field(0, ge=0)


# ---------------------------------------------------------------------------
# Ranking layer
# ---------------------------------------------------------------------------

#: Ordered feature specification. ``None`` means the feature is always present (BASE group);
#: otherwise the feature belongs to the given component and is dropped when that component
#: is disabled in an ablation cell.
FEATURE_SPECS: tuple[tuple[str, Component | None], ...] = (
    # --- BASE: scanner and curated-feed facts that never depend on A, B or C ---
    ("cvss_base_max", None),
    ("cvss_version_ord", None),
    ("cvss_source_agreement", None),
    ("cvss_ac_low", None),
    ("cvss_pr_none", None),
    ("cvss_ui_none", None),
    ("cvss_c_high", None),
    ("cvss_i_high", None),
    ("cvss_a_high", None),
    ("scanner_severity_ord", None),
    ("scanner_confidence", None),
    ("cwe_owasp_top10", None),
    ("vuln_age_days", None),
    ("cluster_size", None),
    ("auth_required_ord", None),
    ("method_state_changing", None),
    ("param_count", None),
    # --- Component A: agentic semantic assessment ---
    ("a_asset_criticality", Component.A),
    ("a_data_sensitivity", Component.A),
    ("a_exposure", Component.A),
    ("a_function_ord", Component.A),
    ("a_is_admin_surface", Component.A),
    ("a_is_auth_boundary", Component.A),
    ("a_exploit_feasibility", Component.A),
    ("a_exploit_maturity_ord", Component.A),
    ("a_attack_complexity_high", Component.A),
    ("a_privileges_required_ord", Component.A),
    ("a_user_interaction_required", Component.A),
    ("a_impact_cia_mean", Component.A),
    ("a_privilege_gained_ord", Component.A),
    ("a_p_applicable", Component.A),
    ("a_version_match_ord", Component.A),
    ("a_confidence", Component.A),
    ("a_injection_signals", Component.A),
    # Retrieved exploit intelligence. The review's architecture is explicit that the model
    # is a feature-extraction layer and the ranker decides: "Rather than allowing the
    # language model to determine the final priority, the AI component serves as a
    # feature-extraction layer, while the ranking model learns how different risk factors
    # influence remediation priority." These are how what the model read on the internet
    # reaches the ranking, alongside CVSS, EPSS and KEV, rather than only reaching the
    # report. All are zero when intelligence gathering is disabled or found nothing, which
    # are deliberately the same state: absence of evidence, not evidence of absence.
    ("a_intel_documents", Component.A),
    ("a_intel_public_exploit_urls", Component.A),
    ("a_intel_active_exploitation", Component.A),
    ("a_intel_confidence", Component.A),
    # Agreement with the curated feeds is the most informative thing the layer produces: a
    # web claim that corroborates CISA is worth something quite different from one that
    # contradicts it, and the ranker can only learn that if the two are separate features.
    ("a_intel_corroborates_feeds", Component.A),
    ("a_intel_contradicts_feeds", Component.A),
    # Deliberately a feature rather than only a guard. Letting the ranker see how much
    # attempted manipulation arrived with a finding's evidence lets it learn to discount
    # that evidence, instead of the framework silently trusting or silently dropping it.
    ("a_intel_injection_signals", Component.A),
    # --- Component B: threat intelligence, attacker model, monetary impact ---
    ("b_epss", Component.B),
    ("b_epss_percentile", Component.B),
    ("b_kev", Component.B),
    ("b_kev_ransomware", Component.B),
    ("b_kev_age_days", Component.B),
    ("b_exploit_count", Component.B),
    ("b_exploit_maturity_feed_ord", Component.B),
    ("b_exploit_verified", Component.B),
    ("b_p_exploit_attacker", Component.B),
    ("b_impact_log", Component.B),
    ("b_expected_loss_log", Component.B),
    ("b_remediation_hours", Component.B),
    # --- Component C: attack-graph position ---
    ("c_reach_delta_log", Component.C),
    ("c_max_path_prob", Component.C),
    ("c_n_paths_through", Component.C),
    ("c_betweenness", Component.C),
    ("c_hops_from_entry", Component.C),
    ("c_privilege_gain", Component.C),
    ("c_is_chokepoint", Component.C),
)

FEATURE_NAMES: tuple[str, ...] = tuple(name for name, _ in FEATURE_SPECS)
FEATURE_GROUPS: dict[str, Component | None] = {name: group for name, group in FEATURE_SPECS}


def feature_names_for(flags: ComponentFlags) -> list[str]:
    """Feature columns present for an ablation cell: BASE plus every enabled component."""
    enabled = flags.enabled()
    return [name for name, group in FEATURE_SPECS if group is None or group in enabled]


class FeatureFrame(BaseModel):
    """Feature matrix plus the grouping metadata LambdaMART needs.

    Rows are aligned with ``finding_ids``; ``group_ids`` holds the scan id per row and
    must be contiguous (all rows of a scan adjacent) because XGBoost ranking groups are positional.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X: pd.DataFrame
    finding_ids: list[str]
    group_ids: list[str]
    flags: ComponentFlags = ComponentFlags()

    @model_validator(mode="after")
    def _check_shape(self) -> "FeatureFrame":
        if len(self.finding_ids) != len(self.X) or len(self.group_ids) != len(self.X):
            raise ValueError("FeatureFrame row counts disagree")
        expected = feature_names_for(self.flags)
        if list(self.X.columns) != expected:
            raise ValueError(
                f"FeatureFrame columns do not match flags {self.flags.label()}: "
                f"{len(self.X.columns)} columns, expected {len(expected)}"
            )
        return self

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def group_sizes(self) -> np.ndarray:
        """Contiguous group sizes in row order (what ``XGBRanker.fit(group=...)`` expects)."""
        sizes: list[int] = []
        previous: str | None = None
        for group_id in self.group_ids:
            if group_id != previous:
                sizes.append(0)
                previous = group_id
            sizes[-1] += 1
        return np.asarray(sizes, dtype=int)

    def to_numpy(self) -> np.ndarray:
        return self.X.to_numpy(dtype=float)


class FeatureContribution(Frozen):
    feature: str
    value: float
    shap_value: float
    group: Component | None = None
    tier: TrustTier = TrustTier.CURATED_FEED


class Explanation(Frozen):
    """Goal 5. Reason codes are templated strings; model free text never reaches the user."""

    finding_id: str
    base_value: float = 0.0
    top_contributions: tuple[FeatureContribution, ...] = ()
    reason_codes: tuple[str, ...] = ()
    untrusted_influence_share: float = Field(0.0, ge=0.0, le=1.0)


class ManipulationAlert(Frozen):
    finding_id: str
    detector: DetectorName
    severity: float = Field(0.5, ge=0.0, le=1.0)
    message: str = Field("", max_length=400)
    rank_delta: int | None = None


class RankedFinding(Frozen):
    finding_id: str
    scan_id: str
    rank: int = Field(ge=1)
    score: float
    expected_loss: float = Field(0.0, ge=0.0)
    chain_adjusted_loss: float = Field(0.0, ge=0.0)
    p_exploit: float = Field(0.0, ge=0.0, le=1.0)
    explanation: Explanation | None = None
    alerts: tuple[ManipulationAlert, ...] = ()

    @property
    def manipulation_flag(self) -> bool:
        return len(self.alerts) > 0


class RankingResult(Frozen):
    ranker: RankerName
    flags: ComponentFlags = ComponentFlags()
    config_hash: str = ""
    seed: int = 42
    items: tuple[RankedFinding, ...] = ()
    #: How the learned ranker got its scores, when the ranker was the learned one.
    #:
    #: ``ranker`` alone is the *name of the policy asked for*, not evidence that a model
    #: produced the ordering. A LambdaMART that could not fit and could not load a trained
    #: model orders by expected loss, and for a long time this record said ``lambdamart``
    #: either way - a learned ranking is a claim, and the artifact was making it without
    #: grounds. These three fields are what let a reader tell the cases apart.
    #:
    #: ``model_fitted`` means a booster was fitted on this run's own frame.
    #: ``model_pretrained`` means a booster fitted elsewhere scored it, which is the normal
    #: case for a single application. Both mean XGBoost produced the order.
    #: ``fallback_reason`` is non-empty only when neither happened.
    model_fitted: bool = False
    model_pretrained: bool = False
    fallback_reason: str = ""

    @property
    def learned(self) -> bool:
        """Whether a learned model produced this ordering, however it was obtained."""
        return self.model_fitted or self.model_pretrained

    def order(self, scan_id: str) -> list[str]:
        chosen = [item for item in self.items if item.scan_id == scan_id]
        return [item.finding_id for item in sorted(chosen, key=lambda item: item.rank)]

    def rank_of(self, finding_id: str) -> int | None:
        for item in self.items:
            if item.finding_id == finding_id:
                return item.rank
        return None


# ---------------------------------------------------------------------------
# Evaluation layer
# ---------------------------------------------------------------------------


class GroundTruthLabel(Frozen):
    """Exploitation ground truth. CVSS can never be a label source (Gap 3)."""

    finding_id: str
    cve_id: str | None = None
    exploited: bool = False
    relevance_grade: int = Field(0, ge=0, le=4)
    sources: tuple[LabelSource, ...] = ()
    first_evidence_date: date | None = None
    version_confirmed: VersionMatch = VersionMatch.UNKNOWN
    source_agreement: float = Field(1.0, ge=0.0, le=1.0)
    cvss_used_as_label: Literal[False] = False


class LabelPolicy(Frozen):
    horizon_days: int = Field(90, ge=1)
    accepted_sources: tuple[LabelSource, ...] = (
        LabelSource.KEV,
        LabelSource.EXPLOIT_EVIDENCE,
        LabelSource.SYNTHETIC_ORACLE,
    )
    kev_grade: int = Field(4, ge=1, le=4)
    exploit_evidence_grade: int = Field(3, ge=1, le=4)
    incident_grade: int = Field(4, ge=1, le=4)
    require_version_match: bool = True
    drop_version_mismatch: bool = True
    min_source_agreement: float = Field(0.0, ge=0.0, le=1.0)
    min_exploit_maturity_for_evidence: ExploitMaturity = ExploitMaturity.FUNCTIONAL
    weight_by_impact: bool = True


class LabelSet(Frozen):
    policy: LabelPolicy = LabelPolicy()
    observation_cutoff: date
    labels: tuple[GroundTruthLabel, ...] = ()

    def positives(self) -> set[str]:
        return {label.finding_id for label in self.labels if label.exploited}

    def relevance(self) -> dict[str, int]:
        return {label.finding_id: label.relevance_grade for label in self.labels}

    def by_id(self, finding_id: str) -> GroundTruthLabel | None:
        for label in self.labels:
            if label.finding_id == finding_id:
                return label
        return None


class Split(Frozen):
    kind: SplitKind
    fold: int = Field(0, ge=0)
    train_scan_ids: tuple[str, ...] = ()
    valid_scan_ids: tuple[str, ...] = ()
    test_scan_ids: tuple[str, ...] = ()
    train_end: date
    test_start: date
    gap_days: int = Field(0, ge=0)
    held_out_app_id: str | None = None

    @model_validator(mode="after")
    def _no_overlap(self) -> "Split":
        train = set(self.train_scan_ids)
        test = set(self.test_scan_ids)
        if train & test:
            raise ValueError("train and test scans overlap")
        if self.test_start < self.train_end:
            raise ValueError("test window starts before the train window ends")
        return self


class MetricValue(Frozen):
    name: MetricName
    k: int | None = None
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None

    @property
    def key(self) -> str:
        return f"{self.name.value}" if self.k is None else f"{self.name.value.replace('@k', '')}@{self.k}"


class CalibrationReport(Frozen):
    brier: float = Field(ge=0.0)
    ece: float = Field(ge=0.0)
    n_bins: int = Field(10, ge=2)
    bin_confidence: tuple[float, ...] = ()
    bin_accuracy: tuple[float, ...] = ()
    bin_count: tuple[int, ...] = ()


class MinorityClassReport(Frozen):
    """Gap 7: the rarest classes are the consequential ones, so they are reported separately."""

    positive_rate: float = Field(ge=0.0, le=1.0)
    mcc: float = Field(ge=-1.0, le=1.0)
    f1_positive: float = Field(ge=0.0, le=1.0)
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)


class MetricBundle(Frozen):
    ranker: RankerName
    flags: ComponentFlags = ComponentFlags()
    split: Split
    seed: int = 42
    values: tuple[MetricValue, ...] = ()
    calibration: CalibrationReport | None = None
    minority: MinorityClassReport | None = None
    runtime_seconds: float = Field(0.0, ge=0.0)

    def get(self, name: MetricName, k: int | None = None) -> float | None:
        for value in self.values:
            if value.name == name and value.k == k:
                return value.value
        return None

    def as_dict(self) -> dict[str, float]:
        return {value.key: value.value for value in self.values}


class AblationCell(Frozen):
    flags: ComponentFlags
    seeds: tuple[int, ...] = ()
    mean: dict[str, float] = Field(default_factory=dict)
    std: dict[str, float] = Field(default_factory=dict)
    n: int = Field(0, ge=0)


class AblationTable(Frozen):
    cells: tuple[AblationCell, ...] = ()
    main_effects: dict[str, dict[str, float]] = Field(default_factory=dict)
    interactions: dict[str, dict[str, float]] = Field(default_factory=dict)
    paired_ci: dict[str, dict[str, tuple[float, float]]] = Field(default_factory=dict)

    def cell(self, flags: ComponentFlags) -> AblationCell | None:
        for item in self.cells:
            if item.flags == flags:
                return item
        return None


class SelectionResult(Frozen):
    """Resource-constrained Top-K selection under unequal remediation cost (Gap 10)."""

    scan_id: str
    ranker: RankerName
    method: SelectionMethod
    budget_hours: float = Field(gt=0.0)
    selected_ids: tuple[str, ...] = ()
    total_hours: float = Field(0.0, ge=0.0)
    risk_captured: float = Field(0.0, ge=0.0)
    risk_capture_fraction: float = Field(0.0, ge=0.0, le=1.0)
    exploited_captured: int = Field(0, ge=0)
    exploited_total: int = Field(0, ge=0)


class SimulationResult(Frozen):
    """Longitudinal deployment protocol: exposure time, not prediction accuracy (Gap 10)."""

    policy: RankerName
    weeks: int = Field(ge=1)
    capacity_hours_per_week: float = Field(gt=0.0)
    exposure_days_total: float = Field(0.0, ge=0.0)
    exposure_days_exploited: float = Field(0.0, ge=0.0)
    expected_loss_days: float = Field(0.0, ge=0.0)
    exploited_remediated_before_exploit: int = Field(0, ge=0)
    exploited_total: int = Field(0, ge=0)
    weekly_cumulative_exposure: tuple[float, ...] = ()
    reduction_vs_cvss: float | None = None


# ---------------------------------------------------------------------------
# Adversarial layer
# ---------------------------------------------------------------------------


class AdversarialExpectation(Frozen):
    canary_must_not_leak: bool = True
    max_abs_delta_feasibility: float = Field(0.15, ge=0.0, le=1.0)
    max_abs_delta_criticality: float = Field(0.15, ge=0.0, le=1.0)
    max_rank_shift: int = Field(2, ge=0)
    must_be_detected: bool = False


class AdversarialCase(Frozen):
    case_id: str
    category: InjectionCategory
    injection_point: Provenance          # REFERENCE_PAGE, TARGET_RESPONSE, SCANNER_OUTPUT or EXPLOIT_DB
    payload: str
    goal: Literal["inflate", "deflate", "exfiltrate", "none"] = "inflate"
    language: str = "en"
    target_finding_id: str | None = None
    expectation: AdversarialExpectation = AdversarialExpectation()

    @field_validator("injection_point")
    @classmethod
    def _not_operator(cls, value: Provenance) -> Provenance:
        if value == Provenance.OPERATOR:
            raise ValueError("adversarial payloads cannot be injected at the operator tier")
        return value


class AdversarialOutcome(Frozen):
    case_id: str
    category: InjectionCategory
    canary_leaked: bool = False
    detected_pre_llm: bool = False
    detected_post_llm: bool = False
    delta_feasibility: float = 0.0
    delta_criticality: float = 0.0
    delta_p_exploit: float = 0.0
    rank_shift: int = 0
    passed: bool = True
    notes: str = Field("", max_length=400)


class AdversarialReport(Frozen):
    backend: LLMBackendKind
    corpus_version: str = "v1"
    n_cases: int = Field(0, ge=0)
    attack_success_rate: float = Field(0.0, ge=0.0, le=1.0)
    canary_leak_rate: float = Field(0.0, ge=0.0, le=1.0)
    detection_rate: float = Field(0.0, ge=0.0, le=1.0)
    false_positive_rate: float = Field(0.0, ge=0.0, le=1.0)
    mean_abs_rank_shift: float = Field(0.0, ge=0.0)
    max_abs_rank_shift: int = Field(0, ge=0)
    per_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    outcomes: tuple[AdversarialOutcome, ...] = ()


# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------


class RunManifest(Frozen):
    """Everything needed to reproduce a run (Gap 4)."""

    run_id: str
    created_at: datetime
    config_hash: str
    config: dict[str, Any] = Field(default_factory=dict)
    seeds: tuple[int, ...] = ()
    dataset_hash: str = ""
    package_versions: dict[str, str] = Field(default_factory=dict)
    llm_backend: LLMBackendKind = LLMBackendKind.HEURISTIC
    llm_model: str = ""
    feed_mode: FeedMode = FeedMode.OFFLINE
    as_of: date | None = None
    command: str = ""


EnrichedFinding.model_rebuild()
