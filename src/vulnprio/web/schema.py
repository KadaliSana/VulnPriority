"""The dashboard payload contract, and the interactive API's request/response contract.

The website is a static page plus one JSON document. This module defines that document,
so the exporter and the front end agree without either reading the other's source.

Every section is optional: a run that only ranked a scan produces a payload with findings
and no evaluation, and the page hides what is absent. That keeps the site useful during a
single-scan triage as well as after a full research run.

The second half of the file describes the *interactive* contract used by
:mod:`vulnprio.web.app`: what the browser may ask for (`AnalyzeRequest`), what a running
job looks like (`WebJobState`), and what the server advertises about itself
(`WebHealth`, `WebConfigOptions`). Those types are additions; nothing above them changed,
so an exported static site written by an older build still validates.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vulnprio.core.money import DEFAULT_CURRENCY

__all__ = [
    "WebMeta",
    "WebSummary",
    "WebScan",
    "WebContribution",
    "WebFinding",
    "WebGraphNode",
    "WebGraphEdge",
    "WebPath",
    "WebGraph",
    "WebMetric",
    "WebCalibration",
    "WebAblationCell",
    "WebAblation",
    "WebSelection",
    "WebSimulation",
    "WebAdversarial",
    "WebGapRow",
    "DashboardData",
    "DASHBOARD_SCHEMA_VERSION",
    # --- interactive API (additions; the static contract above is unchanged) ---
    "INTERACTIVE_API_VERSION",
    "SCAN_PROFILES",
    "JOB_STATUSES",
    "WebCapabilities",
    "WebHealth",
    "WebPreset",
    "WebScanProfile",
    "WebDefaults",
    "WebConfigOptions",
    "WebAsOf",
    "WebReportInspection",
    "WebScannerTool",
    "WebScannerEnvironment",
    "AnalyzeUpload",
    "AnalyzeComponents",
    "AnalyzeRequest",
    "WebJobState",
    "WebJobCreated",
    "WebError",
    "WebNovelty",
]

DASHBOARD_SCHEMA_VERSION = "1.0"

#: Version of the request/response contract in the second half of this module.
INTERACTIVE_API_VERSION = "1.0"

#: The two scanning postures the UI may ask for. The scan package is the authority on what
#: each one actually sends; these names are the vocabulary the two sides agree on.
SCAN_PROFILES: tuple[str, ...] = ("passive", "active")

#: ``queued`` and ``running`` are live; the other three are terminal. ``cancelled`` is its
#: own outcome rather than a flavour of ``failed`` because the two mean different things to
#: whoever is reading the page: one was asked for, the other was not.
JOB_STATUSES: tuple[str, ...] = ("queued", "running", "done", "failed", "cancelled")


class WebModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebMeta(WebModel):
    schema_version: str = DASHBOARD_SCHEMA_VERSION
    run_id: str = ""
    generated_at: datetime | None = None
    config_hash: str = ""
    dataset_hash: str = ""
    package_version: str = ""
    llm_backend: str = ""
    llm_model: str = ""
    feed_mode: str = ""
    attacker: str = ""
    impact_model: str = ""
    #: What every money figure in this payload is denominated in, as an ISO 4217 code
    #: ("INR", "USD", ...). Straight from ``ImpactModel.currency``, which is the run's only
    #: authority on the question. No money field in this document names a currency in its
    #: own name; this is where the front end reads it, and the front end must format from
    #: it rather than assume a symbol -- ``INR`` is grouped ``##,##,###`` and scaled in
    #: lakh and crore, which is not a decoration on top of western grouping but a different
    #: convention.
    currency: str = DEFAULT_CURRENCY
    as_of: str = ""
    seeds: list[int] = Field(default_factory=list)
    components: dict[str, bool] = Field(default_factory=dict)
    command: str = ""


class WebSummary(WebModel):
    n_apps: int = 0
    n_scans: int = 0
    n_endpoints: int = 0
    n_findings: int = 0
    n_clusters: int = 0
    n_kev: int = 0
    n_exploited: int = 0
    total_expected_loss: float = 0.0
    total_impact: float = 0.0
    total_remediation_hours: float = 0.0
    top_decile_loss_share: float = 0.0


class WebScan(WebModel):
    scan_id: str
    app_id: str = ""
    app_name: str = ""
    sector: str = ""
    scanned_at: str = ""
    scanner: str = ""
    n_endpoints: int = 0
    n_findings: int = 0
    total_expected_loss: float = 0.0


class WebContribution(WebModel):
    feature: str
    value: float = 0.0
    shap: float = 0.0
    group: str = "BASE"
    tier: int = 1


class WebFinding(WebModel):
    finding_id: str
    scan_id: str = ""
    app_id: str = ""
    name: str = ""
    #: The scanner's own prose for this finding. It is ``UntrustedText`` upstream and stays
    #: untrusted here: render it as text, never as markup.
    description: str = ""
    cwe_id: int | None = None
    cve_ids: list[str] = Field(default_factory=list)
    endpoint_path: str = ""
    endpoint_method: str = ""
    endpoint_function: str = ""
    auth_required: int = 0
    cluster_size: int = 1

    scanner_severity: str = ""
    cvss_base: float | None = None
    cvss_version: str = ""
    cvss_source_agreement: float | None = None
    epss: float | None = None
    epss_percentile: float | None = None
    kev: bool = False
    kev_ransomware: bool = False
    exploit_maturity: str = ""
    exploit_count: int = 0

    criticality: float = 0.0
    data_sensitivity: float = 0.0
    exposure: float = 0.0
    exploit_feasibility: float = 0.0
    applicability: str = ""
    p_applicable: float = 0.0
    version_match: str = ""

    p_exploit: float = 0.0
    impact: float = 0.0
    #: What ``impact`` is made of. A reader told "1.4 crore at risk" reasonably asks which
    #: part of that is stolen data and which is downtime, so the split travels with the total.
    impact_confidentiality: float = 0.0
    impact_integrity: float = 0.0
    impact_availability: float = 0.0
    impact_reputational: float = 0.0
    expected_loss: float = 0.0
    #: The named log-odds terms behind ``p_exploit``, straight from
    #: ``ExploitLikelihood.log_odds_terms``: one entry per weighted piece of evidence, which
    #: is what lets the page answer "why is this one likely to be exploited" in words.
    likelihood_terms: dict[str, float] = Field(default_factory=dict)
    chain_delta: float = 0.0
    chain_adjusted: float = 0.0
    is_chokepoint: bool = False
    hops_from_entry: int = 0
    remediation_hours: float = 0.0
    #: One line of fix guidance from ``vulnprio.report``, when that package is installed.
    remediation_summary: str = ""

    rank: int = 0
    ranks_by_policy: dict[str, int] = Field(default_factory=dict)
    score: float = 0.0

    reason_codes: list[str] = Field(default_factory=list)
    contributions: list[WebContribution] = Field(default_factory=list)
    untrusted_influence_share: float = 0.0
    max_tier_used: int = 0
    injection_signals: int = 0
    alerts: list[str] = Field(default_factory=list)

    exploited: bool | None = None
    relevance: int = 0
    selected_in_budget: bool = False


class WebGraphNode(WebModel):
    id: str
    asset: str = ""
    privilege: int = 0
    value: float = 0.0
    is_entry: bool = False
    is_target: bool = False
    reach_probability: float = 0.0


class WebGraphEdge(WebModel):
    src: str
    dst: str
    probability: float = 0.0
    finding_id: str | None = None
    kind: str = "exploit"
    tier: int = 2


class WebPath(WebModel):
    nodes: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    probability: float = 0.0
    target_value: float = 0.0
    expected_value: float = 0.0


class WebGraph(WebModel):
    scan_id: str
    nodes: list[WebGraphNode] = Field(default_factory=list)
    edges: list[WebGraphEdge] = Field(default_factory=list)
    entry_node: str = ""
    target_nodes: list[str] = Field(default_factory=list)
    total_risk: float = 0.0
    top_paths: list[WebPath] = Field(default_factory=list)
    monotone_verified: bool = False
    rejected_untrusted_edges: int = 0


class WebMetric(WebModel):
    ranker: str
    metric: str
    k: int | None = None
    value: float = 0.0
    ci_low: float | None = None
    ci_high: float | None = None
    components: str = "ABC"
    split: str = ""


class WebCalibration(WebModel):
    ranker: str = ""
    brier: float = 0.0
    ece: float = 0.0
    bin_confidence: list[float] = Field(default_factory=list)
    bin_accuracy: list[float] = Field(default_factory=list)
    bin_count: list[int] = Field(default_factory=list)
    mcc: float | None = None
    f1_positive: float | None = None
    balanced_accuracy: float | None = None
    positive_rate: float | None = None
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)


class WebAblationCell(WebModel):
    label: str
    a: bool = True
    b: bool = True
    c: bool = True
    mean: dict[str, float] = Field(default_factory=dict)
    std: dict[str, float] = Field(default_factory=dict)


class WebAblation(WebModel):
    cells: list[WebAblationCell] = Field(default_factory=list)
    main_effects: dict[str, dict[str, float]] = Field(default_factory=dict)
    interactions: dict[str, dict[str, float]] = Field(default_factory=dict)
    paired_ci: dict[str, dict[str, list[float]]] = Field(default_factory=dict)


class WebSelection(WebModel):
    scan_id: str = ""
    ranker: str = ""
    method: str = ""
    budget_hours: float = 0.0
    n_selected: int = 0
    total_hours: float = 0.0
    risk_captured: float = 0.0
    risk_capture_fraction: float = 0.0
    exploited_captured: int = 0
    exploited_total: int = 0
    selected_ids: list[str] = Field(default_factory=list)


class WebSimulation(WebModel):
    policy: str
    weeks: int = 0
    capacity_hours_per_week: float = 0.0
    exposure_days_total: float = 0.0
    exposure_days_exploited: float = 0.0
    expected_loss_days: float = 0.0
    exploited_remediated_before_exploit: int = 0
    exploited_total: int = 0
    weekly_cumulative_exposure: list[float] = Field(default_factory=list)
    reduction_vs_reference: float | None = None


class WebAdversarial(WebModel):
    backend: str = ""
    corpus_version: str = ""
    n_cases: int = 0
    attack_success_rate: float = 0.0
    canary_leak_rate: float = 0.0
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    mean_abs_rank_shift: float = 0.0
    max_abs_rank_shift: int = 0
    per_category: dict[str, dict[str, float]] = Field(default_factory=dict)


class WebGapRow(WebModel):
    gap_id: str
    title: str = ""
    mitigation: str = ""
    modules: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    evidence: str = ""


class DashboardData(WebModel):
    """The single JSON document the website consumes."""

    meta: WebMeta = WebMeta()
    summary: WebSummary = WebSummary()
    scans: list[WebScan] = Field(default_factory=list)
    findings: list[WebFinding] = Field(default_factory=list)
    graphs: list[WebGraph] = Field(default_factory=list)
    metrics: list[WebMetric] = Field(default_factory=list)
    calibration: list[WebCalibration] = Field(default_factory=list)
    ablation: WebAblation = WebAblation()
    selections: list[WebSelection] = Field(default_factory=list)
    simulations: list[WebSimulation] = Field(default_factory=list)
    adversarial: WebAdversarial | None = None
    gaps: list[WebGapRow] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)


# ===========================================================================
# The interactive contract
#
# Everything below is served by ``vulnprio.web.app``. None of it is written into
# the static export, so a site exported to ``file://`` never sees these types and
# a page that cannot reach a server simply never asks for them.
# ===========================================================================


class WebCapabilities(WebModel):
    """Which sibling packages this build can actually import.

    Reported honestly: a capability is false when the package is absent, and ``detail``
    carries the import error so the page can say why rather than only that.
    """

    scan: bool = False
    report: bool = False
    novelty: bool = False
    detail: dict[str, str] = Field(default_factory=dict)


class WebHealth(WebModel):
    ok: bool = True
    version: str = ""
    schema_version: str = DASHBOARD_SCHEMA_VERSION
    api_version: str = INTERACTIVE_API_VERSION
    capabilities: WebCapabilities = WebCapabilities()
    has_bundled_run: bool = False


class WebPreset(WebModel):
    """One attacker or impact-model preset, summarised for a dropdown."""

    name: str
    label: str = ""
    description: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class WebScanProfile(WebModel):
    """A scanning posture, described in the words the operator has to agree to."""

    name: str
    label: str = ""
    description: str = ""
    sends: list[str] = Field(default_factory=list)
    does_not_send: list[str] = Field(default_factory=list)
    authoritative: bool = False   # True when vulnprio.scan supplied this description itself


class WebDefaults(WebModel):
    attacker: str = ""
    impact_model: str = ""
    profile: str = "passive"
    budget_hours: float = 40.0
    components: dict[str, bool] = Field(default_factory=lambda: {"a": True, "b": True, "c": True})
    llm_backend: str = ""
    feed_mode: str = ""


class WebConfigOptions(WebModel):
    attackers: list[WebPreset] = Field(default_factory=list)
    impact_models: list[WebPreset] = Field(default_factory=list)
    profiles: list[WebScanProfile] = Field(default_factory=list)
    defaults: WebDefaults = WebDefaults()
    accepted_report_suffixes: list[str] = Field(default_factory=lambda: [".json", ".xml", ".jsonl"])
    max_upload_bytes: int = 0
    capabilities: WebCapabilities = WebCapabilities()


class WebAsOf(WebModel):
    """Whether live exploit intelligence can honestly describe this scan's moment.

    A straight projection of ``vulnprio.intel.models.AsOfVerdict``. It travels to the
    browser as data rather than prose because the remedy is a button: a report from three
    months ago and an internet read today describe different worlds, and the useful answer
    is "assess the target now", not an error.
    """

    status: str = "not_applicable"       # not_applicable | current | stale | refused | overridden
    remedy: str = "none"                 # none | rescan_target | use_fixtures
    search_allowed: bool = True
    evidence_is_current: bool = True
    as_of: str = ""                      # the scan's own date
    evaluated_at: str = ""               # the day the judgement was made
    age_days: int = 0
    max_age_days: int = 0
    message: str = ""


class WebScannerTool(WebModel):
    """One scanner the machine could run, and whether it can serve the requested profile.

    ``supports_requested_profile`` is the field that matters in the interface: Nuclei and
    Nikto request paths nobody linked to, which is exactly what the passive profile promises
    not to do, so they are registered active-only. A passive request on a Nuclei-only
    machine therefore falls back to the built-in scanner, and the operator should learn that
    while choosing rather than afterwards.
    """

    name: str
    installed: bool = False
    version: str = ""
    executable: str = ""
    image: str = ""
    profiles: list[str] = Field(default_factory=list)
    supports_requested_profile: bool = False
    summary: str = ""
    install_hint: str = ""
    preference_rank: int = 0
    #: True when this is the tool that would actually run for the requested profile.
    would_run: bool = False


class WebScannerEnvironment(WebModel):
    """What would actually scan, for one profile, on this machine right now."""

    profile: str = "passive"
    tools: list[WebScannerTool] = Field(default_factory=list)
    #: The external tool that would be chosen, or "" when the built-in scanner would run.
    preferred: str = ""
    builtin_fallback: bool = True
    skipped: list[str] = Field(default_factory=list)
    notice: str = ""
    available: bool = False       # whether vulnprio.scan could be asked at all


class WebReportInspection(WebModel):
    """What a posted report turns out to be, before anything is run on it.

    The Analyze view needs the scan's date and host before it can decide whether to warn
    about staleness or offer to re-scan, and it should not be reimplementing date arithmetic
    or scanner sniffing in the browser to find out.
    """

    app_name: str = ""
    scanner: str = ""
    host: str = ""
    scanned_at: str = ""
    n_findings: int = 0
    n_endpoints: int = 0
    #: A URL the operator could assess right now to get scan and intelligence from the same
    #: moment. Derived from the report's own host; empty when it names none.
    suggested_target_url: str = ""
    as_of: WebAsOf = WebAsOf()


class AnalyzeUpload(WebModel):
    filename: str = ""
    content_base64: str = ""


class AnalyzeComponents(WebModel):
    a: bool = True
    b: bool = True
    c: bool = True


class AnalyzeRequest(WebModel):
    """What the browser asks the server to do.

    ``authorized`` and ``authorization_note`` exist only for ``mode="scan"`` and are
    checked before a job is created: an unauthorised request is refused at the door, so
    nothing is ever sent to a target the operator did not explicitly claim.
    """

    mode: Literal["upload", "scan", "demo"] = "demo"
    report: AnalyzeUpload | None = None
    target_url: str = ""
    authorized: bool = False
    authorization_note: str = ""
    profile: Literal["passive", "active"] = "passive"
    attacker: str = ""
    impact_model: str = ""
    budget_hours: float = Field(40.0, gt=0.0, le=100_000.0)
    components: AnalyzeComponents = AnalyzeComponents()
    #: Whether to search the web for exploit intelligence about this scan's CVEs. The page
    #: turns it off when the operator chooses to proceed with a stale report rather than
    #: re-scan, so that the evidence and the scan still describe the same moment.
    use_intel: bool = True
    #: Consent to assess a loopback, private or link-local target: a container on localhost,
    #: a staging box on an internal range. Refused by default and kept separate from
    #: ``authorized`` on purpose - the two answer different questions. Authorisation is "am I
    #: allowed to test this system"; this is "did I mean to point a scanner at an internal
    #: address at all", which is the check that stops a cloud metadata endpoint being scanned
    #: by a typo. Someone can be fully authorised and still not have meant it.
    allow_private_target: bool = False
    #: Which scanner to run for ``mode="scan"``. Empty means "let the scan package choose",
    #: which prefers a real scanner over the built-in crawler; ``"builtin"`` pins the
    #: built-in one; any other value names an installed tool.
    scanner: str = ""


class WebJobState(WebModel):
    job_id: str
    mode: str = ""
    status: str = "queued"
    phase: str = ""
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str = ""
    log: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""
    finished_at: str | None = None
    has_result: bool = False


class WebJobCreated(WebModel):
    job_id: str
    status: str = "queued"


class WebError(WebModel):
    """Every non-2xx JSON body has this shape. A stack trace never appears in ``detail``."""

    error: str
    detail: str = ""


class WebNovelty(WebModel):
    """The shape returned when ``vulnprio.novelty`` is absent.

    When the package is present its own payload is passed through untouched; this model
    only pins down the honest "not in this build" answer.
    """

    available: bool = False
    detail: str = ""
