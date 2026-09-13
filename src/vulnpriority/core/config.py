"""Configuration schema and loaders.

One pydantic tree describes an entire run. ``PipelineConfig.hash()`` is recorded in the
run manifest so any number the framework prints can be traced back to its configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vulnpriority.core.enums import (
    ExploitMaturity,
    FeedMode,
    LLMBackendKind,
    RankerName,
    SelectionMethod,
    SplitKind,
    TrustTier,
)
from vulnpriority.core.errors import ConfigError
from vulnpriority.core.hashing import config_hash
from vulnpriority.core.models import AttackerModel, ComponentFlags, ImpactModel, LabelPolicy
from vulnpriority.core.money import DEFAULT_CURRENCY

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from vulnpriority.core.resolve import RunResolution

__all__ = [
    "FeedsConfig",
    "LLMConfig",
    "SandboxConfig",
    "ComponentAConfig",
    "ComponentBConfig",
    "ComponentCConfig",
    "RankingConfig",
    "EvaluationConfig",
    "SelectionConfig",
    "SimulationConfig",
    "AdversarialConfig",
    "IntelConfig",
    "SyntheticConfig",
    "PipelineConfig",
    "load_config",
    "load_dotenv",
    "DOTENV_PATH",
    "load_attacker_preset",
    "load_impact_preset",
    "PROJECT_ROOT",
]

#: Repository root (``D:/Steg``); default relative paths resolve against it.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: The package directory itself, for data that ships with the code rather than with the
#: repository. ``PROJECT_ROOT`` is wrong for those: it does not exist in an installed wheel.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Where ``vulnpriority train-ranker`` writes, and where a run looks for a model to score with.
#: Inside the package so that an install has a learned ranker without having to train one.
DEFAULT_RANKER_MODEL: Path = PACKAGE_ROOT / "rank" / "models" / "lambdamart.ubj"


class FeedsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: ``auto`` prefers live-with-cache and degrades to the fixture corpus when the network
    #: is unreachable; see :mod:`vulnpriority.core.resolve`. Writing ``offline`` explicitly
    #: still means offline, and is what ``configs/offline.yaml`` and the test suite use.
    mode: FeedMode = FeedMode.AUTO
    fixture_dir: Path = Path("data/fixtures/feeds")
    cache_dir: Path = Path(".cache/feeds")
    cache_ttl_days: int = 7
    http_timeout_s: float = 20.0
    nvd_api_key_env: str = "NVD_API_KEY"
    max_references_per_cve: int = 8
    reference_char_budget: int = 6000
    reference_host_allowlist: tuple[str, ...] = (
        "nvd.nist.gov",
        "cve.mitre.org",
        "www.cisa.gov",
        "github.com",
        "www.exploit-db.com",
        "owasp.org",
        "portswigger.net",
    )
    max_reference_bytes: int = 400_000


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: ``auto`` takes the first backend whose credentials are present, cost-ascending, and
    #: settles on ``heuristic`` when none is; see :data:`vulnpriority.core.resolve
    #: .BACKEND_PROBE_ORDER`. Naming a backend explicitly skips resolution entirely -- and
    #: a named backend whose key is missing raises rather than quietly degrading, because a
    #: broken deployment must not hide behind a plausible number.
    backend: LLMBackendKind = LLMBackendKind.AUTO
    model: str = "claude-sonnet-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 2048
    temperature: float = 0.0
    max_retries: int = 2
    timeout_s: float = 120.0
    cache_dir: Path = Path(".cache/llm")
    use_cache: bool = True
    consistency_shrinkage: float = Field(0.5, ge=0.0, le=1.0)
    max_divergence: float = Field(0.35, ge=0.0, le=1.0)
    fallback_to_heuristic: bool = True

    # -- free-tier backends ------------------------------------------------
    # ``model`` and ``api_key_env`` above are Anthropic-shaped defaults. A backend that
    # is not Anthropic treats an untouched default as "not chosen for me" and substitutes
    # its own (``gemini-3.5-flash-lite`` / ``GEMINI_API_KEY``), so pointing a run at Gemini is
    # one line of YAML rather than three. Setting either explicitly always wins.

    #: OpenAI-compatible chat-completions root, e.g. ``https://api.groq.com/openai/v1``
    #: or ``http://localhost:11434/v1``. Required by ``openai_compatible`` and ignored by
    #: every other backend. There is no default on purpose: naming the endpoint is the
    #: operator's decision about where scan-derived text is sent.
    base_url: str | None = None

    #: Client-side pacing, in requests per minute. 0 disables it, which is the default and
    #: what the paid paths use. Free tiers rate-limit hard and a scan issues three model
    #: calls per finding, so the cheapest way to stay inside a quota is not to exceed it in
    #: the first place: a 429 costs a round trip and a backoff, a spaced request costs only
    #: the wait that was going to happen anyway.
    #:
    #: Set it from the quota you actually have. Google no longer publishes per-model
    #: free-tier request rates in its public documentation -- they are shown in AI Studio
    #: for your own project -- so there is no honest default to bake in here, and a guess
    #: would be a number someone later cites as fact. Groq's published free-tier ceiling is
    #: 30 requests per minute, OpenRouter's is 20.
    requests_per_minute: int = Field(0, ge=0)

    #: First backoff after a 429, doubling per attempt and capped at ``max_backoff_s``.
    #: A ``Retry-After`` header or an SDK-reported retry delay always wins over both.
    retry_backoff_s: float = Field(2.0, ge=0.0)
    max_backoff_s: float = Field(60.0, ge=0.0)

    #: How an OpenAI-compatible provider is asked for JSON.
    #:
    #: * ``auto`` -- ``response_format={"type": "json_object"}`` *and* the schema spelled
    #:   out in the prompt. The widest-supported combination: every provider that honours
    #:   JSON mode gets it, and one that ignores it still sees the schema.
    #: * ``json_schema`` -- provider-enforced schema. Groq and OpenRouter support it;
    #:   many local servers do not.
    #: * ``json_object`` -- JSON mode only, no schema in the prompt.
    #: * ``prompt`` -- no ``response_format`` at all, schema in the prompt only. The
    #:   documented fallback for a provider that 400s on ``response_format``.
    #:
    #: A weaker provider degrades rather than corrupts: whatever comes back still has to
    #: pass ``model_validate`` here and the output guard afterwards, and a reply that does
    #: not validate becomes a heuristic answer with an audit that says so.
    json_mode: Literal["auto", "json_schema", "json_object", "prompt"] = "auto"


class SandboxConfig(BaseModel):
    """Untrusted-content handling. These numbers are the security contract (Architecture item 4)."""

    model_config = ConfigDict(extra="forbid")

    max_chars_per_segment: int = 6000
    max_segments: int = 12
    canary_enabled: bool = True
    canary_length: int = 24
    nonce_length: int = 16
    patterns_file: Path = Path("configs/sandbox/instruction_patterns.yaml")
    suspicious_signal_threshold: int = 1
    injected_signal_threshold: int = 3
    require_evidence_spans: bool = True
    #: maximum absolute change a tier may apply to any feature normalised to [0, 1]
    influence_budget: dict[TrustTier, float] = Field(
        default_factory=lambda: {
            TrustTier.OPERATOR: 1.0,
            TrustTier.CURATED_FEED: 1.0,
            TrustTier.SCANNER: 0.8,
            TrustTier.REFERENCE_PAGE: 0.35,
            TrustTier.TARGET_CONTENT: 0.15,
        }
    )
    corroborated_budget: float = Field(1.0, ge=0.0, le=1.0)
    #: Untrusted evidence gets less room to argue a score DOWN than to argue it up. The two
    #: directions are not equally dangerous: talking a real vulnerability down leaves it
    #: unpatched, while talking a harmless one up only wastes remediation effort. The
    #: adversarial corpus measures both directions, and deflation is where the surviving
    #: attacks were. 1.0 restores symmetric budgets.
    deflation_budget_factor: float = Field(0.4, ge=0.0, le=1.0)
    allow_downgrade_below_floor: bool = False
    max_untrusted_shap_share: float = Field(0.35, ge=0.0, le=1.0)


class ComponentAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    lexicon_path: Path | None = None
    assess_endpoints: bool = True
    assess_exploitability: bool = True
    assess_applicability: bool = True
    max_references_in_prompt: int = 4


class ComponentBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    attacker_preset: str = "opportunistic"
    attacker: AttackerModel | None = None          # inline override wins over the preset
    impact_preset: str = "default_ecommerce"
    impact: ImpactModel | None = None
    remediation_hours_by_cwe: dict[int, float] = Field(default_factory=dict)
    remediation_default_hours: float = 4.0
    remediation_hours_per_extra_endpoint: float = 0.5
    #: Fully-loaded engineering cost per hour, in the impact model's currency (INR in the
    #: shipped presets; ``ImpactModel.currency`` is the authority). A proportional prior,
    #: not a measured rate: the previous $120/h figure carried across at the same 39.13x
    #: ratio the impact presets use, because no India-specific loaded-labour rate was
    #: sourced. Of every money figure in the framework this is the one an operator is
    #: most likely to know better than the default does.
    remediation_hourly_rate: float = 4_700.0


class ComponentCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    criticality_weight_exponent: float = 1.0
    max_hops: int = 6
    top_paths: int = 10
    min_edge_probability: float = 1e-4
    chokepoint_delta_fraction: float = Field(0.25, ge=0.0, le=1.0)
    chain_weight: float = Field(1.0, ge=0.0)       # weight of reach_delta in the chain-adjusted loss
    admit_lateral_edges: bool = True
    verify_monotonicity: bool = True


class RankingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranker: RankerName = RankerName.LAMBDAMART
    #: A LambdaMART booster fitted on a corpus that could teach a ranking, used to score
    #: runs that cannot fit one of their own.
    #:
    #: Assessing a single application is such a run, and it is the ordinary case: one scan
    #: is one query group, and a pairwise ranking objective has no pair to learn from inside
    #: a single group however many findings it holds. Without this the queue was ordered by
    #: expected loss while still being labelled ``lambdamart``. With it the ordering is the
    #: learned model's, because scoring needs neither labels nor groups.
    #:
    #: ``vulnpriority train-ranker`` writes one. The default points at the model that ships with
    #: the package; set it to ``null`` to refuse a pretrained model and take the
    #: expected-loss fallback instead, which is what an evaluation run wants so that it
    #: measures the model it just fitted.
    model_path: Path | None = DEFAULT_RANKER_MODEL
    objective: str = "rank:ndcg"
    eval_metric: str = "ndcg@10"
    lambdarank_pair_method: str = "topk"
    lambdarank_num_pair_per_sample: int = 8
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 2.0
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    n_jobs: int = 1
    seed: int = 42
    impact_weighted_pairs: bool = True
    monotone: dict[str, int] = Field(
        default_factory=lambda: {"b_kev": 1, "b_epss": 1, "c_reach_delta_log": 1, "b_expected_loss_log": 1}
    )
    head_scale_pos_weight: float | None = None      # None -> computed from the training label balance
    head_calibration: str = "isotonic"              # isotonic | sigmoid | none
    explain_top_n: int = 5
    #: How far a finding must sit outside its OWN scan's displacement distribution before the
    #: rank guard calls it manipulation, in robust standard deviations (median plus this
    #: multiple of a scaled median absolute deviation). A fixed count cannot work here: it
    #: would encode an assumption that semantic assessment should barely move anything, which
    #: is the opposite of what the framework claims. 3.0 is the conventional robust cut.
    displacement_mad_multiplier: float = Field(3.0, ge=0.0)
    #: Absolute floor, so a scan where nothing moves cannot flag a finding that moved two places.
    min_rank_displacement: int = Field(3, ge=1)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k_values: tuple[int, ...] = (5, 10, 20)
    split_kind: SplitKind = SplitKind.TIME_ORDERED
    n_folds: int = 3
    gap_days: int = 30
    min_train_scans: int = 4
    label_policy: LabelPolicy = LabelPolicy()
    #: Refuse to score the ranker against labels it could have read off its own features.
    #:
    #: ``KEV`` and ``EXPLOIT_EVIDENCE`` are accepted ground truth *and* feature columns
    #: (``b_kev``, ``b_exploit_count``). A positive justified by nothing else is a lookup
    #: dressed as a prediction. On the shipped 10,000-scan corpus 6,311 of 31,589 positives
    #: are in that state and are not scored; the 25,278 the oracle confirms are. Gap 3 made
    #: this argument about CVSS and stopped one column short.
    #:
    #: This governs only what the model is *graded* on, not what it may be trained on. The
    #: corpus builder happens to train on the filtered set too, but for a different reason
    #: (see scripts/build_training_corpus.py): a model taught that "in KEV" means
    #: "exploited" learns to restate its own b_kev column. Turn this off to reproduce a
    #: number computed the old way, and expect it to be optimistic.
    exclude_circular_labels: bool = True
    seeds: tuple[int, ...] = (42, 43, 44)
    bootstrap_iters: int = 500
    baselines: tuple[RankerName, ...] = (
        RankerName.CVSS_ONLY,
        RankerName.EPSS_ONLY,
        RankerName.KEV_FIRST,
        RankerName.SCANNER_SEVERITY,
        RankerName.EXPECTED_LOSS,
        RankerName.VMC_CHAIN,
        RankerName.RANDOM,
    )
    vmc_epss_threshold: float = 0.088
    vmc_cvss_threshold: float = 7.0
    calibration_bins: int = 10
    classification_threshold: float = 0.5
    minority_maturity_floor: ExploitMaturity = ExploitMaturity.FUNCTIONAL


class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget_hours: float = 40.0
    hour_granularity: float = 0.5
    method: SelectionMethod = SelectionMethod.DP_EXACT
    max_items_for_exact: int = 400


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weeks: int = 26
    capacity_hours_per_week: float = 20.0
    policies: tuple[RankerName, ...] = (
        RankerName.LAMBDAMART,
        RankerName.EXPECTED_LOSS,
        RankerName.CVSS_ONLY,
        RankerName.EPSS_ONLY,
        RankerName.KEV_FIRST,
    )
    reference_policy: RankerName = RankerName.CVSS_ONLY


class AdversarialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_path: Path = Path("data/adversarial/corpus_v1.yaml")

    # Containment is absolute. No amount of injected text may move a bounded feature past
    # its influence budget, and the canary must never appear in model output. These are the
    # properties ADR-002 claims, and a non-zero value here would be the claim being withdrawn.
    max_containment_breach_rate: float = 0.0
    max_canary_leak_rate: float = 0.0

    # Rank displacement is a weaker and different property. Untrusted evidence is *allowed*
    # to move a finding within its tier's budget, so a queue where injected text changes no
    # position at all would be one where legitimate advisory evidence counts for nothing.
    # What must not happen is displacement beyond that budget, or in the attacker's chosen
    # direction at scale. The measured rate on the shipped corpus is 2.2%, all of it
    # deflation within budget and none of it a containment breach; this ceiling is set just
    # above that so a regression fails the build while the designed behaviour does not.
    max_goal_directed_rank_success_rate: float = 0.05

    #: Overall ceiling, retained for callers that want one number. It is the union of the
    #: two conditions above, so it cannot be tighter than the rank allowance.
    max_attack_success_rate: float = 0.05

    min_detection_rate: float = 0.8
    max_false_positive_rate: float = 0.1


class IntelConfig(BaseModel):
    """Internet exploit-intelligence retrieval (produced by :mod:`vulnpriority.intel`).

    ``enabled`` and ``mode`` both default to ``auto``, which resolves to live only when a
    backend that can actually search is available and the run is not under the research
    protocol -- see :func:`vulnpriority.core.resolve.resolve_intel`. What has not changed is
    what happens with nothing configured and no key: that still resolves to offline, still
    serves the recorded corpus, and still cannot open a socket. ``auto`` prefers the real
    thing; it does not assume it.

    Writing ``enabled: false`` or ``mode: offline`` is final and is never resolved past.
    ``research_mode`` overrides everything: an evaluation must not quietly acquire today's
    internet because a key happened to be exported.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool | Literal["auto"] = "auto"
    mode: Literal["offline", "live", "auto"] = "auto"

    model: str = "claude-opus-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = Field(8000, ge=256)
    api_key_env: str = "ANTHROPIC_API_KEY"
    timeout_s: float = Field(180.0, gt=0.0)
    max_retries: int = Field(1, ge=0, le=5)

    #: Which live provider does the searching. ``anthropic`` uses the web-search and
    #: web-fetch server tools; ``gemini`` uses Google Search grounding, which is the
    #: closest free equivalent and the reason this package is reachable without a bill;
    #: ``parallel`` uses the Parallel Search API, which is a pure retrieval API with no
    #: model attached and therefore the only one of the three where phase 1 is search and
    #: nothing else. Ignored entirely when ``mode`` is ``offline``.
    #:
    #: As with :class:`LLMConfig`, ``model`` and ``api_key_env`` above are
    #: Anthropic-shaped defaults; the Gemini provider substitutes ``gemini-3.5-flash-lite``
    #: and ``GEMINI_API_KEY`` when they are left untouched. The Parallel provider ignores
    #: ``model`` entirely -- there is no model -- and reads its key from
    #: ``parallel_api_key_env`` below rather than from ``api_key_env``, because a run may
    #: legitimately search with Parallel and extract with Anthropic or Gemini, which means
    #: two keys are in play at once and one field cannot name both.
    search_provider: Literal["anthropic", "gemini", "parallel"] = "anthropic"

    #: Parallel Search API settings. ``turbo`` and ``fast`` are $1 per 1,000 requests,
    #: ``basic`` and ``advanced`` (the API's own default) are $5 per 1,000. The framework
    #: defaults to ``turbo`` because one request per finding at $0.001 is what makes live
    #: search affordable enough to leave switched on. Note that ``turbo`` accepts English
    #: and Japanese queries only, which the framework's operator-built queries always are.
    parallel_mode: Literal["turbo", "fast", "basic", "advanced"] = "turbo"
    parallel_api_key_env: str = "PARALLEL_API_KEY"
    parallel_endpoint: str = "https://api.parallel.ai/v1/search"

    #: Measured rather than advertised. The docs say ~200ms for ``turbo``; a live key
    #: returned in 1578ms cold and ~1030ms warm, so this is headroom over the real number
    #: rather than over the published one. Separate from ``timeout_s`` above, which is the
    #: extraction model's budget and is two orders of magnitude longer for good reason.
    parallel_timeout_s: float = Field(30.0, gt=0.0)

    #: Per-excerpt character cap, applied to the response rather than in the request: the
    #: Search API documents no max-chars parameter, so this is a client-side truncation and
    #: nothing else. It is load-bearing, not decorative -- a live request returns ten
    #: results with no way to ask for fewer, and one observed excerpt ran to 2,777
    #: characters. With ``max_documents`` (8) documents each held to ``snippet_char_budget``
    #: (4,000) characters, one finding contributes at most 32,000 characters against a
    #: sandbox ceiling of ``max_segments`` x ``max_chars_per_segment`` = 12 x 6,000.
    parallel_max_excerpt_chars: int = Field(1200, ge=100)

    #: Client-side pacing and 429 backoff for the live provider. See the identically
    #: named fields on :class:`LLMConfig`; free-tier grounded search is the tightest
    #: quota in the framework, so pacing matters more here than anywhere else.
    requests_per_minute: int = Field(0, ge=0)
    retry_backoff_s: float = Field(2.0, ge=0.0)
    max_backoff_s: float = Field(60.0, ge=0.0)

    #: Search-plan bounds. Every one is a cost ceiling as much as a safety one.
    max_queries: int = Field(5, ge=1, le=20)
    max_documents: int = Field(8, ge=1, le=40)
    max_search_uses: int = Field(5, ge=1, le=20)
    max_fetch_uses: int = Field(3, ge=0, le=20)
    max_content_tokens: int = Field(8000, ge=500, le=100_000)
    snippet_char_budget: int = Field(4000, ge=200)

    #: Where a live search may look. A short list of places that publish exploit
    #: information and are worth reading: the point of an allowlist is that adding a host
    #: is a decision someone made, not a default. The web tools accept one list or the
    #: other, never both.
    allowed_domains: tuple[str, ...] = (
        "nvd.nist.gov",
        "cve.mitre.org",
        "www.cve.org",
        "www.cisa.gov",
        "github.com",
        "www.exploit-db.com",
        "packetstormsecurity.com",
        "www.rapid7.com",
        "msrc.microsoft.com",
        "security.apache.org",
        "lists.apache.org",
        "portswigger.net",
        "owasp.org",
        "blog.projectdiscovery.io",
        "unit42.paloaltonetworks.com",
        "www.mandiant.com",
        "googleprojectzero.blogspot.com",
        # Vendor and distribution security trackers. Added after a live run dropped
        # access.redhat.com, whose advisories routinely carry the affected-version detail
        # and exploitation status that the CVE record itself omits. A allowlist that
        # excludes the vendor's own advisory is filtering out the best source it has.
        "access.redhat.com",
        "ubuntu.com",
        "security-tracker.debian.org",
        "security.gentoo.org",
        "lists.debian.org",
        "www.suse.com",
        "security.netapp.com",
        "www.oracle.com",
        "support.apple.com",
        "chromereleases.googleblog.com",
        "www.vmware.com",
        "www.ibm.com",
        "helpx.adobe.com",
        "cert-portal.siemens.com",
        "www.zerodayinitiative.com",
        "attackerkb.com",
    )
    blocked_domains: tuple[str, ...] = ()

    fixture_path: Path = Path("data/fixtures/intel/searches.json")
    cache_dir: Path = Path(".cache/intel")
    use_cache: bool = True
    record_path: Path | None = None

    #: As-of discipline. ``max_scan_age_days`` is how old a scan may be before live search
    #: stops describing the same moment the scan does. Generous on purpose: an assessment
    #: run this morning is answered honestly by today's internet, and an operational run
    #: must never need a flag to say so.
    max_scan_age_days: int = Field(7, ge=0)
    #: Set by the evaluation pipeline, never by the interactive path. It decides only what
    #: happens once a scan is found stale: offer a re-scan, or refuse outright because
    #: under the research protocol the correct action is a recorded corpus.
    research_mode: bool = False
    #: Operator override for a stale scan outside research mode. Searching anyway is a
    #: defensible choice for someone who wants today's threat picture against an old
    #: report; it is recorded on the result so nobody later mistakes it for a current one.
    allow_anachronistic_search: bool = False

    summarize: bool = True
    min_summary_sentences: int = Field(3, ge=1, le=10)
    max_summary_sentences: int = Field(5, ge=1, le=10)

    @model_validator(mode="after")
    def _check(self) -> "IntelConfig":
        if self.allowed_domains and self.blocked_domains:
            raise ValueError(
                "the web tools accept allowed_domains OR blocked_domains, never both"
            )
        if self.max_summary_sentences < self.min_summary_sentences:
            raise ValueError("max_summary_sentences is below min_summary_sentences")
        return self


class SyntheticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    n_apps: int = 8
    endpoints_per_app: tuple[int, int] = (12, 30)
    findings_per_app: tuple[int, int] = (30, 90)
    scans_per_app: int = 3
    scan_interval_days: int = 30
    start_date: str = "2024-01-15"
    base_exploit_rate: float = Field(0.06, ge=0.0, le=1.0)
    label_lag_days: tuple[int, int] = (3, 45)
    injection_fraction: float = Field(0.0, ge=0.0, le=1.0)
    non_english_fraction: float = Field(0.2, ge=0.0, le=1.0)
    sectors: tuple[str, ...] = ("ecommerce", "healthcare", "saas", "fintech")


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Path("runs")
    run_id: str | None = None
    data_dir: Path = Path("data")
    as_of: str | None = None                       # ISO date; None means "the scan date"
    seed: int = 42
    feeds: FeedsConfig = FeedsConfig()
    llm: LLMConfig = LLMConfig()
    sandbox: SandboxConfig = SandboxConfig()
    component_a: ComponentAConfig = ComponentAConfig()
    component_b: ComponentBConfig = ComponentBConfig()
    component_c: ComponentCConfig = ComponentCConfig()
    intel: IntelConfig = IntelConfig()
    ranking: RankingConfig = RankingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    selection: SelectionConfig = SelectionConfig()
    simulation: SimulationConfig = SimulationConfig()
    adversarial: AdversarialConfig = AdversarialConfig()
    synthetic: SyntheticConfig = SyntheticConfig()

    def flags(self) -> ComponentFlags:
        return ComponentFlags(
            a=self.component_a.enabled,
            b=self.component_b.enabled,
            c=self.component_c.enabled,
        )

    def with_flags(self, flags: ComponentFlags) -> "PipelineConfig":
        return self.model_copy(
            update={
                "component_a": self.component_a.model_copy(update={"enabled": flags.a}),
                "component_b": self.component_b.model_copy(update={"enabled": flags.b}),
                "component_c": self.component_c.model_copy(update={"enabled": flags.c}),
            }
        )

    def hash(self) -> str:
        return config_hash(self.model_dump(mode="json"))

    def impact_model(self) -> ImpactModel:
        """The monetary impact model in force: the inline override, else the named preset.

        A method rather than a field on purpose: ``hash()`` above dumps the *fields*, so
        resolving a preset here cannot change the run id.
        """
        if self.component_b.impact is not None:
            return self.component_b.impact
        try:
            return load_impact_preset(self.component_b.impact_preset)
        except ConfigError:
            return ImpactModel(name=self.component_b.impact_preset)

    def currency(self) -> str:
        """What this run's money figures are denominated in.

        The single authority, asked by every formatter in the framework. Nothing infers a
        currency from a field name, because after the ``_usd`` suffixes were dropped no
        field name carries one -- which is the point: a name cannot then disagree with the
        number under it.
        """
        return str(self.impact_model().currency) or DEFAULT_CURRENCY

    def resolution(self, probe: bool = True) -> "RunResolution":
        """What this run's ``auto`` switches resolve to, and why.

        Deliberately *not* applied to ``self``: the hash above feeds the run id, so a
        configuration rewritten by whatever keys happened to be exported would make run
        directories environment-dependent and reproducibility a claim the framework could
        not support. Callers keep the config they hashed and carry this beside it.
        """
        from vulnpriority.core.resolve import resolve_run  # noqa: PLC0415 - avoids a cycle

        return resolve_run(self, probe=probe)

    def resolve(self, path: Path) -> Path:
        """Absolute path for a possibly relative config path, rooted at the project directory."""
        path = Path(path)
        return path if path.is_absolute() else (PROJECT_ROOT / path)


def _apply_overrides(data: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    for dotted, value in (overrides or {}).items():
        cursor = data
        *keys, last = dotted.split(".")
        for key in keys:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[last] = value
    return data



#: Where secrets are read from, if the file exists. Deliberately not committed: see
#: ``.env.example`` for the variables and ``.gitignore`` for the exclusion.
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Read ``KEY=value`` lines from a ``.env`` file into the process environment.

    Written here rather than taken as a dependency, because the framework's default path
    runs offline with no key at all and a secrets loader is not worth a package.

    A real environment variable WINS over the file unless ``override`` is set. That is the
    precedence people expect: a key exported for one command, or injected by a CI secret
    store, must not be silently replaced by a stale file on disk.

    Values are never logged and never enter the config hash, so a key cannot leak into a
    run manifest. Returns the names that were set, never the values.
    """
    target = Path(path) if path is not None else DOTENV_PATH
    applied: dict[str, str] = {}
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return applied

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if not name:
            continue
        value = value.strip()
        # Strip one matching pair of surrounding quotes, which is how a value with spaces
        # is normally written. Anything else is taken literally.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not value:
            # An empty assignment is a placeholder, not a value. Setting it would make the
            # variable *present but empty*, and backend resolution asks "is a key present"
            # to decide whether to go live - so a blank line in the template would resolve
            # to a provider that cannot authenticate.
            continue
        if override or name not in os.environ:
            os.environ[name] = value
            applied[name] = "set"
    return applied


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    """Load a YAML config (defaults when ``path`` is None) and apply dotted-key overrides.

    Reads ``.env`` first, so an API key placed there is available to every backend without
    the caller having to export it. Real environment variables still win over the file.
    """
    load_dotenv()
    data: dict[str, Any] = {}
    if path is not None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        if not resolved.exists():
            raise ConfigError(f"config file not found: {resolved}")
        data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    _apply_overrides(data, overrides)
    return PipelineConfig.model_validate(data)


def load_attacker_preset(name: str, root: Path | None = None) -> AttackerModel:
    root = root or (PROJECT_ROOT / "configs" / "attacker_models")
    path = Path(root) / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"unknown attacker preset: {name} ({path})")
    return AttackerModel.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_impact_preset(name: str, root: Path | None = None) -> ImpactModel:
    root = root or (PROJECT_ROOT / "configs" / "impact_models")
    path = Path(root) / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"unknown impact preset: {name} ({path})")
    return ImpactModel.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
