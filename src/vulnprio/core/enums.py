"""All enumerations used across vulnprio.

This module is part of the frozen shared contract: every other module imports from
here and must not define competing enumerations.
"""

from __future__ import annotations

from enum import Enum, IntEnum

__all__ = [
    "TrustTier",
    "UNTRUSTED_TIERS",
    "Provenance",
    "PROVENANCE_TIER",
    "tier_of",
    "HttpMethod",
    "PrivilegeLevel",
    "EndpointFunction",
    "ScannerSeverity",
    "ExploitMaturity",
    "AttackComplexity",
    "UserInteraction",
    "ApplicabilityVerdict",
    "VersionMatch",
    "CvssVersion",
    "ScoreSource",
    "ExploitSource",
    "LabelSource",
    "Component",
    "LLMBackendKind",
    "FeedMode",
    "RankerName",
    "SplitKind",
    "InjectionCategory",
    "InjectionVerdict",
    "DetectorName",
    "MetricName",
    "SelectionMethod",
    "IntelSourceKind",
    "IntelQueryKind",
    "AgreementAxis",
    "AsOfStatus",
    "IntelRemedy",
]


class TrustTier(IntEnum):
    """Provenance tier. Lower is more trusted.

    Drives influence budgets (how far a tier may move a feature), attack-graph edge
    admissibility (only tiers <= SCANNER may create edges) and explanation labelling.
    """

    OPERATOR = 0          # config, attacker model, impact model, manual overrides
    CURATED_FEED = 1      # NVD, EPSS, CISA KEV, Exploit-DB index
    SCANNER = 2           # scanner-observed structure: endpoints, alerts, status codes, headers
    REFERENCE_PAGE = 3    # advisories, blog posts, PoC repositories fetched from the internet
    TARGET_CONTENT = 4    # response bodies / self-descriptions authored by the target application


UNTRUSTED_TIERS: tuple[TrustTier, ...] = (TrustTier.REFERENCE_PAGE, TrustTier.TARGET_CONTENT)


class Provenance(str, Enum):
    """Where a piece of text came from. Everything except OPERATOR is untrusted input."""

    OPERATOR = "operator"
    NVD = "nvd"
    EPSS = "epss"
    KEV = "kev"
    EXPLOIT_DB = "exploit_db"
    SCANNER_OUTPUT = "scanner_output"
    REFERENCE_PAGE = "reference_page"
    TARGET_RESPONSE = "target_response"
    SYNTHETIC = "synthetic"


PROVENANCE_TIER: dict[Provenance, TrustTier] = {
    Provenance.OPERATOR: TrustTier.OPERATOR,
    Provenance.NVD: TrustTier.CURATED_FEED,
    Provenance.EPSS: TrustTier.CURATED_FEED,
    Provenance.KEV: TrustTier.CURATED_FEED,
    Provenance.EXPLOIT_DB: TrustTier.CURATED_FEED,
    Provenance.SCANNER_OUTPUT: TrustTier.SCANNER,
    Provenance.REFERENCE_PAGE: TrustTier.REFERENCE_PAGE,
    Provenance.TARGET_RESPONSE: TrustTier.TARGET_CONTENT,
    Provenance.SYNTHETIC: TrustTier.SCANNER,
}


def tier_of(provenance: Provenance) -> TrustTier:
    """Trust tier for a provenance. Unknown provenance is treated as maximally untrusted."""
    return PROVENANCE_TIER.get(provenance, TrustTier.TARGET_CONTENT)


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class PrivilegeLevel(IntEnum):
    """Attack-graph privilege states; higher means more control."""

    NONE = 0      # unauthenticated internet user
    USER = 1      # authenticated application user
    ADMIN = 2     # application administrator
    SYSTEM = 3    # host / OS level (remote code execution)


class EndpointFunction(str, Enum):
    AUTH = "auth"
    PAYMENT = "payment"
    ADMIN = "admin"
    PII_DATA = "pii_data"
    FILE_IO = "file_io"
    API_DATA = "api_data"
    SEARCH = "search"
    STATIC_CONTENT = "static_content"
    UNKNOWN = "unknown"


class ScannerSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExploitMaturity(IntEnum):
    UNKNOWN = 0
    UNPROVEN = 1
    POC = 2
    FUNCTIONAL = 3
    WEAPONIZED = 4


class AttackComplexity(str, Enum):
    LOW = "low"
    HIGH = "high"
    UNKNOWN = "unknown"


class UserInteraction(str, Enum):
    NONE = "none"
    REQUIRED = "required"
    UNKNOWN = "unknown"


class ApplicabilityVerdict(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    UNCERTAIN = "uncertain"


class VersionMatch(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class CvssVersion(str, Enum):
    V2 = "2.0"
    V30 = "3.0"
    V31 = "3.1"
    V40 = "4.0"


class ScoreSource(str, Enum):
    NVD = "nvd"
    CNA = "cna"
    SCANNER = "scanner"
    OTHER = "other"


class ExploitSource(str, Enum):
    EXPLOIT_DB = "exploit_db"
    METASPLOIT = "metasploit"
    GITHUB_POC = "github_poc"
    NUCLEI = "nuclei"
    VENDOR_ADVISORY = "vendor_advisory"
    SYNTHETIC = "synthetic"


class LabelSource(str, Enum):
    """Accepted sources of exploitation ground truth. CVSS is deliberately absent (Gap 3)."""

    KEV = "kev"
    EXPLOIT_EVIDENCE = "exploit_evidence"
    INCIDENT = "incident"
    SYNTHETIC_ORACLE = "synthetic_oracle"


class Component(str, Enum):
    A = "A"  # agentic semantic assessment
    B = "B"  # contextual / threat-intelligence enrichment and decision theory
    C = "C"  # topology- and chain-aware ranking


class LLMBackendKind(str, Enum):
    """Which model backend Component A talks to.

    Two of these cost money and two do not, which is the axis that matters to someone
    running the framework out of their own pocket:

    * ``HEURISTIC`` is the default and spends nothing. It is a real deterministic
      assessor, not a stub, and the whole pipeline runs on it offline with no key.
    * ``ANTHROPIC`` is the reference paid path.
    * ``GEMINI`` is the free-tier path. ``gemini-3.5-flash-lite`` has a no-cost quota, and it
      is the only free backend that also brings Google Search grounding, which is what
      :mod:`vulnprio.intel` needs.
    * ``OPENAI_COMPATIBLE`` is *one* backend that reaches many providers, because Groq,
      OpenRouter, Together, Cerebras, a local Ollama, LM Studio and vLLM all speak the
      same ``POST {base_url}/chat/completions`` shape. Point ``llm.base_url`` at one of:

      ==================  ===========================================================
      Provider            ``base_url``
      ==================  ===========================================================
      Groq                ``https://api.groq.com/openai/v1``
      OpenRouter          ``https://openrouter.ai/api/v1``
      Together            ``https://api.together.xyz/v1``
      Cerebras            ``https://api.cerebras.ai/v1``
      Ollama (local)      ``http://localhost:11434/v1``   (no key)
      LM Studio (local)   ``http://localhost:1234/v1``    (no key)
      vLLM (local)        ``http://localhost:8000/v1``    (no key)
      ==================  ===========================================================

      There is deliberately no default endpoint: a backend that silently picked a host
      would be a backend that silently picked whose servers your scan data lands on.

    Adding a member here adds no path into the score. Every backend is wrapped by
    :class:`~vulnprio.llm.guarded.GuardedBackend` in
    :func:`~vulnprio.llm.factory.build_llm_backend`, so the sandbox, canary, envelope,
    schema and evidence-span checks apply identically whichever model answered.
    """

    #: Resolve at run time: take the first backend whose credentials are actually present,
    #: in the order documented by :mod:`vulnprio.core.resolve`, and settle on ``HEURISTIC``
    #: when none is. Never the *result* of resolution and never recorded as one -- a run
    #: manifest says which backend ran, never "auto".
    AUTO = "auto"

    HEURISTIC = "heuristic"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai_compatible"


class FeedMode(str, Enum):
    """Where feed data comes from.

    ``AUTO`` resolves at run time to ``LIVE_WITH_CACHE`` when the network is reachable and
    ``OFFLINE`` when it is not, so a machine with connectivity gets real NVD, EPSS and KEV
    data and a machine without still runs. Like :attr:`LLMBackendKind.AUTO` it is an input
    only: what a run records is the mode that actually ran.
    """

    AUTO = "auto"
    OFFLINE = "offline"
    LIVE = "live"
    LIVE_WITH_CACHE = "live_with_cache"


class RankerName(str, Enum):
    LAMBDAMART = "lambdamart"
    EXPECTED_LOSS = "expected_loss"
    CVSS_ONLY = "cvss_only"
    EPSS_ONLY = "epss_only"
    KEV_FIRST = "kev_first"
    SCANNER_SEVERITY = "scanner_severity"
    VMC_CHAIN = "vmc_chain"
    RANDOM = "random"


class SplitKind(str, Enum):
    TIME_ORDERED = "time_ordered"
    LEAVE_ONE_APP_OUT = "leave_one_app_out"
    RANDOM = "random"  # provided only as the unrealistic control condition (Gap 4)


class InjectionCategory(str, Enum):
    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_HIJACK = "role_hijack"
    SCHEMA_SMUGGLING = "schema_smuggling"
    CANARY_EXFIL = "canary_exfil"
    ENCODED_PAYLOAD = "encoded_payload"
    HIDDEN_TEXT = "hidden_text"
    MULTILINGUAL = "multilingual"
    FAKE_EVIDENCE_INFLATE = "fake_evidence_inflate"
    FAKE_EVIDENCE_DEFLATE = "fake_evidence_deflate"
    NUMERIC_OVERFLOW = "numeric_overflow"
    DELIMITER_ESCAPE = "delimiter_escape"
    BENIGN_CONTROL = "benign_control"


class InjectionVerdict(str, Enum):
    CLEAN = "clean"
    SUSPICIOUS = "suspicious"
    INJECTED = "injected"


class DetectorName(str, Enum):
    PRE_LLM_PATTERN = "pre_llm_pattern"
    CANARY = "canary"
    ENVELOPE = "envelope"
    SCHEMA = "schema"
    EVIDENCE_SPAN = "evidence_span"
    CONSISTENCY = "consistency"
    DIVERGENCE = "divergence"
    DISPLACEMENT = "displacement"
    INFLUENCE_BUDGET = "influence_budget"


class MetricName(str, Enum):
    NDCG_AT_K = "ndcg@k"
    PRECISION_AT_K = "precision@k"
    RECALL_AT_K = "recall@k"
    RISK_CAPTURE_AT_K = "risk_capture@k"
    MAP = "map"
    MRR = "mrr"
    KENDALL_TAU_VS_CVSS = "kendall_tau_vs_cvss"
    MEAN_RANK_OF_EXPLOITED = "mean_rank_of_exploited"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    MCC = "mcc"
    F1_MINORITY = "f1_minority"
    BALANCED_ACCURACY = "balanced_accuracy"
    BRIER = "brier"
    ECE = "ece"
    EFFICIENCY = "efficiency"
    COVERAGE = "coverage"
    WORKLOAD_REDUCTION = "workload_reduction"
    EXPOSURE_DAYS = "exposure_days"
    RUNTIME_SECONDS = "runtime_seconds"


class SelectionMethod(str, Enum):
    DP_EXACT = "dp_exact"
    GREEDY_RATIO = "greedy_ratio"
    RANK_PREFIX = "rank_prefix"


# ---------------------------------------------------------------------------
# Internet exploit intelligence (produced by vulnprio.intel)
# ---------------------------------------------------------------------------


class IntelSourceKind(str, Enum):
    """What kind of page a retrieved document is.

    Not a trust tier -- every retrieved document is tier ``REFERENCE_PAGE``. This is a
    coarse description used for ordering, for the report, and for telling an analyst
    where a claim came from.
    """

    ADVISORY = "advisory"      # vendor or CERT advisory, NVD, CISA
    POC = "poc"                # proof-of-concept code or an exploit database entry
    WRITEUP = "writeup"        # technical analysis, blog post, conference material
    VENDOR = "vendor"          # vendor documentation, release notes, patch notes
    SOCIAL = "social"          # forum, mailing list, social post
    UNKNOWN = "unknown"


class IntelQueryKind(str, Enum):
    """Why a search was issued, so a search plan can be audited and ablated."""

    CVE = "cve"                      # the identifier itself
    POC = "poc"                      # proof-of-concept / exploit code hunting
    EXPLOITATION = "exploitation"    # in-the-wild exploitation reporting
    PRODUCT = "product"              # observed product and version
    WEAKNESS = "weakness"            # CWE class, for findings with no CVE


class AgreementAxis(str, Enum):
    """Whether retrieved material agrees with one axis of curated-feed evidence.

    ``SILENT`` and ``UNCHECKED`` are kept apart because they mean different things. The
    retrieved pages saying nothing is a fact about the pages; the curated feeds asserting
    nothing is a fact about the feeds. Collapsing them would turn "we could not check"
    into "we checked and found agreement".
    """

    AGREE = "agree"
    CONTRADICT = "contradict"
    SILENT = "silent"        # the retrieved material makes no claim on this axis
    UNCHECKED = "unchecked"  # the curated feeds assert nothing to compare against


class AsOfStatus(str, Enum):
    """Whether live search can honestly answer a question about a scan's date."""

    NOT_APPLICABLE = "not_applicable"  # offline: fixtures carry their own retrieval dates
    CURRENT = "current"                # the scan is recent; today's internet is its internet
    STALE = "stale"                    # the scan is old; searching now answers a different question
    REFUSED = "refused"                # stale under the research protocol: not negotiable
    OVERRIDDEN = "overridden"          # stale, searched anyway because an operator said to


class IntelRemedy(str, Enum):
    """What a caller can do about an as-of problem.

    Machine-readable on purpose: a web application turns ``RESCAN_TARGET`` into a
    "Re-scan this target" button rather than parsing a sentence out of a log, which is the
    difference between telling an operator no and offering them the fix.
    """

    NONE = "none"
    RESCAN_TARGET = "rescan_target"    # re-run the assessment so scan and evidence agree
    USE_FIXTURES = "use_fixtures"      # research protocol: replay a recorded corpus instead
