"""Data models for the built-in target assessment scanner.

These types describe what the operator asked for (:class:`ScanRequest`), what the scanner
observed (:class:`Page`, :class:`ProbeResult`, :class:`CheckFinding`) and what it produced
(:class:`ScanOutcome`). The *product* of a scan is a
:class:`vulnprio.core.models.Scan` - byte-for-byte the same shape a parsed ZAP report
produces - so every downstream stage (enrichment, attack graph, ranking, evaluation) works
on a self-assessed target without knowing that no external scanner was involved.

Two safety properties are enforced here, at construction time, rather than left to the
caller to remember:

* ``authorized`` defaults to ``False`` and an attestation without a written
  ``authorization_note`` is rejected. There is no bypass flag anywhere in this package.
* The request cannot carry credentials: ``Cookie``, ``Authorization`` and friends are
  refused, and the ``User-Agent`` must identify this tool honestly.

Every bound is expressed as a pydantic constraint, so an out-of-range limit is
unrepresentable rather than merely discouraged.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from vulnprio import __version__
from vulnprio.core.enums import HttpMethod, ScannerSeverity
from vulnprio.core.models import Frozen, Scan

__all__ = [
    "SCANNER_NAME",
    "DEFAULT_USER_AGENT",
    "DEFAULT_REQUESTS_PER_SECOND",
    "DEFAULT_MAX_PAGES",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_TIME_BUDGET_S",
    "DEFAULT_EXTERNAL_TIME_BUDGET_S",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "CREDENTIAL_HEADERS",
    "ScanProfile",
    "ScanPhase",
    "ProbeKind",
    "ScanRequest",
    "FormField",
    "FormInfo",
    "Page",
    "ProbeResult",
    "CheckFinding",
    "ScanProgress",
    "ScanOutcome",
    "ToolStatus",
    "ScannerEnvironment",
]

#: ``Scan.scanner_name`` written by this package. Downstream code treats it like any
#: other scanner name, which is the point: the pipeline must not special-case us.
SCANNER_NAME = "vulnprio-scan"

#: Honest, identifying User-Agent. A target operator reading their access log must be able
#: to tell what hit them and that it was a consented assessment, not a stealth crawler.
DEFAULT_USER_AGENT = (
    f"vulnprio-scan/{__version__} "
    "(authorised web application security assessment; contact the operator who ran this scan)"
)

DEFAULT_REQUESTS_PER_SECOND = 2.0
#: Default rate against a loopback target. See ``ScanRequest._loopback_may_go_faster``.
LOOPBACK_REQUESTS_PER_SECOND = 10.0
DEFAULT_MAX_PAGES = 200
DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_REQUESTS = 600
DEFAULT_TIME_BUDGET_S = 150.0
#: Wall clock for an external scanner. See ``ScanRequest.external_time_budget_s``.
DEFAULT_EXTERNAL_TIME_BUDGET_S = 1200.0
#: Per-response read cap. Four megabytes rather than one because a truncated
#: JavaScript bundle is a silently incomplete scan: the endpoint list recovered by
#: :mod:`vulnprio.scan.spa` comes out of that file, and a single-page application's
#: main bundle routinely exceeds a megabyte. The total download stays bounded by
#: ``max_bytes`` below, so raising this raises fidelity, not appetite.
DEFAULT_MAX_RESPONSE_BYTES = 4_000_000

#: Request headers this scanner refuses to send. It never authenticates to a target: an
#: authenticated scan is a different activity with a different consent conversation, and
#: leaking a session cookie into a crawl is how "read-only" assessments delete things.
CREDENTIAL_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "cookie2",
        "x-api-key",
        "x-auth-token",
        "x-csrf-token",
        "authentication",
    }
)


class ScanProfile(str, Enum):
    """How much the scanner is allowed to do.

    ``PASSIVE`` observes only: it fetches pages and reads what comes back. ``ACTIVE`` adds
    the three benign probes in :mod:`vulnprio.scan.active` and nothing else, ever.
    """

    PASSIVE = "passive"
    ACTIVE = "active"


class ScanPhase(str, Enum):
    """Coarse phase of a scan, reported through :class:`ScanProgress`."""

    AUTHORIZE = "authorize"
    ROBOTS = "robots"
    CRAWL = "crawl"
    PROBE = "probe"
    CHECK = "check"
    FINGERPRINT = "fingerprint"
    ASSEMBLE = "assemble"
    EXTERNAL = "external"
    DONE = "done"
    ABORTED = "aborted"


class ProbeKind(str, Enum):
    """The complete set of active behaviours this scanner can perform.

    The allowlist that gates these lives in :mod:`vulnprio.scan.safety`. Adding a member
    here is not enough to make a probe runnable; it must also be admitted there, and the
    test suite asserts that the admitted set is exactly these three.
    """

    #: Send a random alphanumeric marker as a parameter value and look for it in the
    #: response. This locates a *reflection point*. It is not an XSS payload: the marker
    #: contains no markup, no quotes and no script.
    REFLECTED_MARKER = "reflected_marker"
    #: A single ``OPTIONS`` request, to read the advertised ``Allow`` set.
    OPTIONS_METHOD = "options_method"
    #: A single ``GET`` for a well-known, publicly documented path
    #: (``/.well-known/security.txt``).
    WELL_KNOWN_PATH = "well_known_path"


class ScanRequest(Frozen):
    """One authorised assessment of one target.

    ``authorized`` is the attestation: the operator states, in code, that they are
    permitted to test ``target_url``. It defaults to ``False`` and
    :func:`vulnprio.scan.safety.require_authorization` refuses every scan without it.
    """

    target_url: str
    authorized: bool = False
    authorization_note: str = Field("", max_length=2000)
    profile: ScanProfile = ScanProfile.PASSIVE

    # --- hard caps: configurable, but bounded at the type level -------------------
    max_pages: int = Field(DEFAULT_MAX_PAGES, ge=1, le=1000)
    max_depth: int = Field(DEFAULT_MAX_DEPTH, ge=0, le=8)
    max_requests: int = Field(DEFAULT_MAX_REQUESTS, ge=1, le=5000)
    requests_per_second: float = Field(DEFAULT_REQUESTS_PER_SECOND, gt=0.0, le=20.0)
    time_budget_s: float = Field(DEFAULT_TIME_BUDGET_S, gt=0.0, le=3600.0)
    #: Wall clock an *external* scanner is allowed, in seconds. Separate from
    #: ``time_budget_s`` because the two measure different things for different tools.
    #:
    #: ``time_budget_s`` bounds the built-in crawler, which fetches pages at a polite rate
    #: and is done in a minute. Handing that same number to ZAP as its ``-T`` cap gave a
    #: full active scan two minutes, which is nowhere near enough for any real application:
    #: ZAP was killed part-way through, at a *different* part each time depending on machine
    #: load and how far its concurrent spider had got, so the same target produced different
    #: findings on every run. That looked like non-determinism in the framework and was
    #: actually a stopwatch set to the wrong tool's budget.
    #:
    #: Twenty minutes is enough for ZAP to finish a mid-sized application. A scan that still
    #: hits the cap is reported as truncated rather than presented as complete.
    external_time_budget_s: float = Field(
        DEFAULT_EXTERNAL_TIME_BUDGET_S, gt=0.0, le=21_600.0
    )
    max_response_bytes: int = Field(DEFAULT_MAX_RESPONSE_BYTES, ge=1024, le=20_000_000)
    max_total_bytes: int | None = Field(None, ge=1024)
    timeout_s: float = Field(10.0, gt=0.0, le=120.0)
    max_redirects: int = Field(5, ge=0, le=10)

    # --- scope and courtesy ------------------------------------------------------
    #: Whether a ``Disallow`` entry in ``robots.txt`` prevents a request. Off by default.
    #:
    #: ``robots.txt`` is a crawler convention, not an access control: it keeps search
    #: engines out of a directory and keeps nobody else out of anything. An attacker reads
    #: it as a list of the places the operator thought worth hiding, and every dedicated
    #: scanner - ZAP, Burp, Nuclei - ignores it for exactly that reason. An authorised
    #: assessment that honours it reports those paths as clean when it never looked at
    #: them, which is the most misleading result this scanner can produce.
    #:
    #: What makes this safe is not this flag. It is the authorisation attestation with a
    #: written note, the scope locked to the named host and re-checked on every redirect,
    #: the rate limit, the refusal to follow destructive-looking links, and a probe
    #: allowlist of three benign behaviours. Those decide what the scanner may do; this
    #: only decides where it looks. ``Crawl-delay`` is honoured either way, because that
    #: one is a statement about what the server can take.
    #:
    #: Set it to ``True`` for a scan that must be indistinguishable from a well-behaved
    #: crawler. The file is fetched and reported in both cases.
    respect_robots: bool = False
    extra_hosts: tuple[str, ...] = ()
    allow_private_targets: bool = False
    headers: dict[str, str] = Field(default_factory=dict)
    user_agent: str = DEFAULT_USER_AGENT
    verify_tls: bool = True

    # --- output metadata ---------------------------------------------------------
    app_name: str | None = None
    sector: str = "generic"
    #: Fixed observation timestamp. ``None`` means "now"; pass a value when reproducible
    #: identifiers matter, because ``scan_id`` is derived from it (DESIGN 3.1).
    scanned_at: datetime | None = None

    @field_validator("target_url")
    @classmethod
    def _usable_target(cls, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            raise ValueError("target_url is required")
        parts = urlsplit(raw)
        if parts.scheme.lower() not in {"http", "https"}:
            raise ValueError(f"target_url must be http(s), got {parts.scheme!r}")
        if not parts.hostname:
            raise ValueError("target_url has no host")
        return raw

    @field_validator("extra_hosts")
    @classmethod
    def _clean_hosts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(sorted({str(host).strip().lower() for host in value if str(host).strip()}))
        for host in cleaned:
            if "/" in host or ":" in host:
                raise ValueError(
                    f"extra_hosts entries are bare hostnames without scheme or port, got {host!r}"
                )
        return cleaned

    @field_validator("headers")
    @classmethod
    def _no_credentials(cls, value: dict[str, str]) -> dict[str, str]:
        for name in value:
            if str(name).strip().lower() in CREDENTIAL_HEADERS:
                raise ValueError(
                    f"{name!r} is a credential header: this scanner never authenticates to a target"
                )
        return {str(key): str(item) for key, item in value.items()}

    @field_validator("user_agent")
    @classmethod
    def _honest_agent(cls, value: str) -> str:
        if "vulnprio" not in (value or "").lower():
            raise ValueError("user_agent must identify this tool (it has to contain 'vulnprio')")
        return value

    @model_validator(mode="before")
    @classmethod
    def _loopback_may_go_faster(cls, data: object) -> object:
        """Raise the default request rate when the target is the operator's own machine.

        Rate limiting is a courtesy owed to *someone else's* service: a scan that saturates
        a production host is a denial of service whatever its intent. A loopback target is
        not someone else's service, and two requests a second there buys nothing but a scan
        that runs a hundred seconds longer and gets cut off by its own time budget before
        it reaches the end of its queue - which turns politeness into missing findings.

        Only the *default* moves. An explicit ``requests_per_second`` is always honoured,
        including an explicitly slow one, and every other budget is untouched.
        """
        if not isinstance(data, dict) or "requests_per_second" in data:
            return data
        target = str(data.get("target_url") or "")
        host = (urlsplit(target).hostname or "").lower()
        if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".localhost"):
            return {**data, "requests_per_second": LOOPBACK_REQUESTS_PER_SECOND}
        return data

    @model_validator(mode="after")
    def _attestation_is_written_down(self) -> "ScanRequest":
        if self.authorized and not self.authorization_note.strip():
            raise ValueError(
                "authorized=True requires an authorization_note recording who permitted this scan"
            )
        return self

    @property
    def target_host(self) -> str:
        return (urlsplit(self.target_url).hostname or "").lower()

    @property
    def target_scheme(self) -> str:
        return urlsplit(self.target_url).scheme.lower()

    @property
    def target_origin(self) -> str:
        parts = urlsplit(self.target_url)
        return f"{parts.scheme.lower()}://{parts.netloc.lower()}"

    @property
    def scope_hosts(self) -> frozenset[str]:
        """Every host this scan may touch: the target plus explicitly listed extras."""
        return frozenset({self.target_host, *self.extra_hosts}) - {""}

    @property
    def total_byte_budget(self) -> int:
        """Whole-scan download cap; derived from the per-response cap when unset."""
        if self.max_total_bytes is not None:
            return self.max_total_bytes
        return min(self.max_response_bytes * self.max_pages, 200_000_000)


class FormField(Frozen):
    """One input of an HTML form. Recorded for analysis; never filled in, never sent."""

    name: str = ""
    type: str = "text"
    value: str = ""
    autocomplete: str | None = None


class FormInfo(Frozen):
    """An HTML form the crawler saw.

    Forms are evidence, not actions: this scanner records the shape of a form (its method,
    its action, its fields) and never submits one. Submitting is how a "read-only" scan
    ends up creating, changing or deleting data.
    """

    action: str
    method: HttpMethod = HttpMethod.GET
    fields: tuple[FormField, ...] = ()
    enctype: str | None = None
    autocomplete: str | None = None
    source_url: str = ""

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields if field.name)

    @property
    def has_password_field(self) -> bool:
        return any(field.type.lower() == "password" for field in self.fields)

    @property
    def is_state_changing(self) -> bool:
        return self.method in (HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH, HttpMethod.DELETE)


class Page(Frozen):
    """One fetched resource, with everything the checks need to reason about it."""

    url: str
    final_url: str = ""
    method: HttpMethod = HttpMethod.GET
    status: int | None = None
    headers: dict[str, str] = Field(default_factory=dict)   # lower-cased names
    set_cookies: tuple[str, ...] = ()                       # raw Set-Cookie values
    body: str = ""
    body_bytes_len: int = Field(0, ge=0)
    content_type: str | None = None
    elapsed_ms: float = Field(0.0, ge=0.0)
    depth: int = Field(0, ge=0)
    links: tuple[str, ...] = ()
    forms: tuple[FormInfo, ...] = ()
    truncated: bool = False
    error: str | None = None
    redirects: tuple[str, ...] = ()

    @property
    def effective_url(self) -> str:
        return self.final_url or self.url

    @property
    def is_html(self) -> bool:
        return (self.content_type or "").lower().startswith("text/html")

    @property
    def scheme(self) -> str:
        return urlsplit(self.effective_url).scheme.lower()

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())


class ProbeResult(Frozen):
    """The outcome of one allowlisted active probe."""

    kind: ProbeKind
    url: str
    method: HttpMethod = HttpMethod.GET
    param: str | None = None
    marker: str | None = None
    status: int | None = None
    reflected: bool = False
    allow_methods: tuple[str, ...] = ()
    headers: dict[str, str] = Field(default_factory=dict)
    evidence: str = ""
    error: str | None = None


class CheckFinding(Frozen):
    """What one check concluded about one page.

    ``detail`` is operator-authored prose owned by this package. ``evidence`` is text
    echoed from the target and is wrapped as ``TARGET_RESPONSE`` when it becomes a
    :class:`~vulnprio.core.models.Finding`, so the sandbox sees it for what it is.
    """

    check_id: str
    name: str
    cwe_id: int
    severity: ScannerSeverity
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    url: str
    method: HttpMethod = HttpMethod.GET
    param: str | None = None
    detail: str = ""
    evidence: str = ""
    profile: ScanProfile = ScanProfile.PASSIVE
    #: What makes two findings of this check "the same condition", for
    #: :func:`vulnprio.scan.checks.collapse_site_wide`. Empty means "compare the evidence",
    #: which is right for a header or a cookie policy where the evidence *is* the
    #: condition. A check whose evidence necessarily differs per route - an error page
    #: quotes the route it failed on - sets this to the class of the condition instead, so
    #: one leaky error handler on forty routes collapses while a database error stays
    #: separate from a stack trace.
    signature: str = ""


class ScanProgress(Frozen):
    """One progress event. A scan emits a stream of these so a UI can follow along."""

    phase: ScanPhase
    pages_fetched: int = Field(0, ge=0)
    findings: int = Field(0, ge=0)
    elapsed_s: float = Field(0.0, ge=0.0)
    message: str = ""


class ToolStatus(Frozen):
    """One external scanner, installed or not, as the web UI should present it."""

    name: str
    installed: bool = False
    executable: str | None = None
    version: str | None = None
    image: str | None = None
    profiles: tuple[ScanProfile, ...] = ()
    supports_requested_profile: bool = False
    summary: str = ""
    install_hint: str = ""
    report_parser: str = ""
    preference_rank: int = Field(0, ge=0)


class ScannerEnvironment(Frozen):
    """What is installed on this machine and what will therefore run.

    Returned by :func:`vulnprio.scan.adapters.scanner_environment` so the web application
    can show the operator their real options - including how to install a proper scanner
    when the answer is "nothing is here, you are about to get the weaker built-in one".
    """

    profile: ScanProfile = ScanProfile.PASSIVE
    tools: tuple[ToolStatus, ...] = ()
    preferred: str | None = None
    builtin_fallback: bool = True
    skipped: tuple[str, ...] = ()
    notice: str = ""

    @property
    def installed(self) -> tuple[ToolStatus, ...]:
        return tuple(tool for tool in self.tools if tool.installed)

    @property
    def missing(self) -> tuple[ToolStatus, ...]:
        return tuple(tool for tool in self.tools if not tool.installed)


class ScanOutcome(Frozen):
    """Everything one assessment produced, with the safety ledger attached.

    ``scan`` is an ordinary :class:`~vulnprio.core.models.Scan`. The counters exist so the
    operator can see what the scanner *refused* to do, which is the interesting half of a
    defensive tool's behaviour, and ``tool_selection`` records why this particular scanner
    ran rather than another.
    """

    scan: Scan
    profile: ScanProfile = ScanProfile.PASSIVE
    tool: str = SCANNER_NAME
    tool_version: str | None = None
    #: Plain-English record of which scanner ran and why it, rather than another one.
    tool_selection: str = ""
    #: Names of the external scanners detected on this machine, in preference order.
    available_tools: tuple[str, ...] = ()
    #: External tools that were tried and failed, or skipped, with the reason.
    external_attempts: tuple[str, ...] = ()
    progress: tuple[ScanProgress, ...] = ()
    pages_fetched: int = Field(0, ge=0)
    requests_made: int = Field(0, ge=0)
    bytes_downloaded: int = Field(0, ge=0)
    skipped_out_of_scope: int = Field(0, ge=0)
    #: Requests ``robots.txt`` actually prevented. Zero unless ``respect_robots`` was on.
    robots_blocked: int = Field(0, ge=0)
    #: Requests ``robots.txt`` named as off-limits, whether or not they were prevented. The
    #: gap between this and ``robots_blocked`` is the ground an advisory scan covered *because*
    #: the operator had flagged it, which is usually the ground worth reading first.
    robots_named: int = Field(0, ge=0)
    skipped_dangerous_links: int = Field(0, ge=0)
    forms_recorded: int = Field(0, ge=0)
    probes_sent: int = Field(0, ge=0)
    errors: tuple[str, ...] = ()
    #: What the scanner could *not* see, in plain English. An error is something that went
    #: wrong; a coverage note is something that went right and still left a gap - a
    #: single-page application whose routes live in JavaScript, say. Without these a thin
    #: scan is indistinguishable from a clean application, which is the more dangerous of
    #: the two readings.
    coverage_notes: tuple[str, ...] = ()
    stopped_reason: str | None = None
    duration_s: float = Field(0.0, ge=0.0)
    report_path: Path | None = None

    @property
    def findings(self) -> int:
        return len(self.scan.findings)
