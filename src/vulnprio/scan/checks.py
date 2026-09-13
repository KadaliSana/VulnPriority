"""The check registry: what the scanner actually concludes about a target.

A :class:`Check` is a small, pure function over one :class:`~vulnprio.scan.models.Page`
and a :class:`CheckContext`. It returns :class:`~vulnprio.scan.models.CheckFinding`
objects, which :mod:`vulnprio.scan.runner` turns into ordinary
:class:`~vulnprio.core.models.Finding` records. Checks never touch the network: the
passive ones read a response that has already been fetched, and the active ones read the
result of a probe that :mod:`vulnprio.scan.active` already sent through the allowlist.

Every check carries a CWE and a docstring saying why the condition matters in practice,
because a finding that cannot explain itself is noise, and because the CWE is what joins
this scanner's output to the intelligence, impact and attack-graph layers downstream.

Detection philosophy: **structural, never exploitative.** An open redirect is reported
because a parameter *looks like* a redirect target, not because the scanner followed one
to an attacker-controlled host. An exposed ``.git`` directory is reported only when the
application linked to it - paths are never guessed, because guessing is how a scanner ends
up making thousands of requests for files that were never there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit

from vulnprio.core.enums import HttpMethod, ScannerSeverity
from vulnprio.core.models import TechComponent
from vulnprio.ingest.normalize import query_parameters, url_path
from vulnprio.ingest.tech_fingerprint import fingerprint_response
from vulnprio.scan.models import CheckFinding, FormInfo, Page, ProbeKind, ProbeResult, ScanProfile
from vulnprio.semantic.cpe_match import parse_version

__all__ = [
    "Check",
    "CheckContext",
    "CHECKS",
    "register_check",
    "checks_for_profile",
    "run_checks",
    "collapse_site_wide",
    "LIBRARY_ADVISORIES",
    "LibraryAdvisory",
    "SENSITIVE_PARAM",
    "REDIRECT_PARAM",
    "SENSITIVE_PATHS",
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One detection rule.

    ``cwe_id`` and ``severity`` are the check's defaults; an individual finding may carry
    a different pair when one rule legitimately covers several weaknesses (missing
    ``Secure`` is CWE-614 while missing ``HttpOnly`` on the same cookie is CWE-1004).
    """

    id: str
    name: str
    cwe_id: int
    severity: ScannerSeverity
    profile: ScanProfile
    run: Callable[[Page, "CheckContext"], list[CheckFinding]]
    rationale: str = ""
    #: Whether this check's subject is the origin rather than the endpoint. A missing
    #: ``Strict-Transport-Security`` header is one misconfiguration of one server, not a
    #: hundred separate defects because a hundred pages were fetched - and reporting it a
    #: hundred times does real harm, because it buries the endpoint-specific findings
    #: under repetition and makes a queue's length a function of crawl budget rather than
    #: of risk. Findings from a site-wide check are collapsed by identical evidence in
    #: :func:`collapse_site_wide`, so two genuinely different CORS policies on the same
    #: host still surface as two findings.
    site_wide: bool = False

    def __call__(self, page: Page, context: "CheckContext") -> list[CheckFinding]:
        return list(self.run(page, context))


#: Registered checks, in registration order (which is the order findings are emitted in).
CHECKS: dict[str, Check] = {}


def register_check(
    check_id: str,
    name: str,
    cwe_id: int,
    severity: ScannerSeverity,
    profile: ScanProfile = ScanProfile.PASSIVE,
    site_wide: bool = False,
) -> Callable[[Callable[[Page, "CheckContext"], list[CheckFinding]]], Check]:
    """Decorator registering a check function, returning the :class:`Check` itself."""

    def decorator(function: Callable[[Page, "CheckContext"], list[CheckFinding]]) -> Check:
        check = Check(
            id=check_id,
            name=name,
            cwe_id=cwe_id,
            severity=severity,
            profile=profile,
            run=function,
            rationale=(function.__doc__ or "").strip(),
            site_wide=site_wide,
        )
        if check_id in CHECKS:
            raise ValueError(f"duplicate check id {check_id!r}")
        CHECKS[check_id] = check
        return check

    return decorator


@dataclass
class CheckContext:
    """Everything a check may look at besides the page in front of it."""

    target_url: str = ""
    pages: tuple[Page, ...] = ()
    probes: tuple[ProbeResult, ...] = ()
    as_of: date | None = None
    #: Confirms candidate CVE ids against the intelligence feed. Takes a CVE id and the
    #: as-of date, returns True when the feed knows it. ``None`` means "no feed available",
    #: in which case library findings are still reported but with lower confidence.
    cve_lookup: Callable[[str, date | None], bool] | None = None
    crawled_urls: frozenset[str] = frozenset()
    extra: dict[str, object] = field(default_factory=dict)

    def probes_for(self, url: str, kind: ProbeKind | None = None) -> tuple[ProbeResult, ...]:
        return tuple(
            probe
            for probe in self.probes
            if probe.url == url and (kind is None or probe.kind == kind)
        )

    def confirm_cves(self, candidates: Sequence[str]) -> tuple[tuple[str, ...], bool]:
        """Cross-reference candidate CVE ids against the feed.

        Returns the confirmed ids and whether a feed was actually consulted, so the caller
        can lower its confidence when it was not.
        """
        if self.cve_lookup is None:
            return tuple(candidates), False
        confirmed = tuple(cve for cve in candidates if self.cve_lookup(cve, self.as_of))
        return confirmed, True


def checks_for_profile(profile: ScanProfile) -> tuple[Check, ...]:
    """Checks that may run under a profile. Active includes every passive check."""
    if profile == ScanProfile.PASSIVE:
        return tuple(check for check in CHECKS.values() if check.profile == ScanProfile.PASSIVE)
    return tuple(CHECKS.values())


def run_checks(
    pages: Iterable[Page],
    context: CheckContext,
    profile: ScanProfile = ScanProfile.PASSIVE,
    *,
    checks: Sequence[Check] | None = None,
    collapse: bool = True,
) -> list[CheckFinding]:
    """Run every check of a profile over every page, in a deterministic order.

    Site-wide checks are collapsed afterwards unless ``collapse`` is off, which exists
    for tests that want to see the raw per-page emissions.
    """
    selected = tuple(checks) if checks is not None else checks_for_profile(profile)
    findings: list[CheckFinding] = []
    for page in pages:
        for check in selected:
            findings.extend(check(page, context))
    return collapse_site_wide(findings) if collapse else findings


def collapse_site_wide(findings: Sequence[CheckFinding]) -> list[CheckFinding]:
    """Report one origin-wide misconfiguration once, with the reach it actually has.

    A finding survives collapse if its check is endpoint-scoped, or if its evidence
    differs from every other finding of the same check on the same host. The survivor
    keeps the first URL it was seen on - lowest depth, so nearest the entry point - and
    its detail gains the instance count and the other paths it held on. Nothing is
    discarded silently, and nothing an operator would act on is lost: a SQL error on three
    routes still names all three, because "which endpoints" is the whole question for an
    endpoint-specific weakness that happens to recur.
    """
    kept: dict[tuple[str, str, str | None, str], CheckFinding] = {}
    instances: dict[tuple[str, str, str | None, str], list[str]] = {}
    out: list[CheckFinding] = []

    for item in findings:
        check = CHECKS.get(item.check_id)
        if check is None or not check.site_wide:
            out.append(item)
            continue
        key = (
            item.check_id,
            urlsplit(item.url).netloc.lower(),
            item.param,
            item.signature or item.evidence,
        )
        seen = instances.setdefault(key, [])
        if key not in kept:
            kept[key] = item
            out.append(item)
        seen.append(item.url)

    if not kept:
        return out

    # Rewrite the survivors in place so the emission order above is preserved.
    counted = {
        id(finding): instances[key]
        for key, finding in kept.items()
        if len(instances[key]) > 1
    }
    if not counted:
        return out
    return [
        (
            item.model_copy(update={"detail": _with_reach(item.detail, counted[id(item)])})
            if id(item) in counted
            else item
        )
        for item in out
    ]


#: How many of the affected paths a collapsed finding names before it starts counting
#: instead. Enough to act on, short enough to stay readable in a queue row.
_REACH_SAMPLE = 6


def _with_reach(detail: str, urls: Sequence[str]) -> str:
    """Append where else the same condition held, naming paths rather than only counting."""
    paths: list[str] = []
    for url in urls:
        path = urlsplit(url).path or "/"
        if path not in paths:
            paths.append(path)
    shown = ", ".join(paths[:_REACH_SAMPLE])
    if len(paths) > _REACH_SAMPLE:
        shown += f" and {len(paths) - _REACH_SAMPLE} more"
    return (
        f"{detail} The same condition holds on {len(paths)} endpoint(s) reached by this "
        f"scan ({shown}), so it is reported once rather than once per page."
    )


def _finding(
    check: Check,
    page: Page,
    *,
    detail: str,
    evidence: str = "",
    param: str | None = None,
    cwe_id: int | None = None,
    severity: ScannerSeverity | None = None,
    confidence: float = 0.7,
    url: str | None = None,
    method: HttpMethod | None = None,
    signature: str = "",
) -> CheckFinding:
    return CheckFinding(
        check_id=check.id,
        name=check.name,
        cwe_id=cwe_id if cwe_id is not None else check.cwe_id,
        severity=severity if severity is not None else check.severity,
        confidence=confidence,
        url=url or page.effective_url,
        method=method or page.method,
        param=param,
        detail=detail,
        evidence=evidence[:600],
        profile=check.profile,
        signature=signature,
    )


def _analysable(page: Page) -> bool:
    """A page worth analysing: it answered, and it is not an error stub."""
    return page.status is not None and page.error is None


def _is_document(page: Page) -> bool:
    """HTML documents only. Header and cookie policy is a property of documents."""
    return _analysable(page) and page.is_html and page.status is not None and page.status < 400


# ---------------------------------------------------------------------------
# Passive: transport and headers
# ---------------------------------------------------------------------------

#: Header -> (why it matters). Used for the missing-header check's prose.
_SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    (
        "content-security-policy",
        "no Content-Security-Policy, so any injected markup executes with the page's full "
        "privileges and there is no second line of defence behind output encoding",
    ),
    (
        "x-content-type-options",
        "no 'X-Content-Type-Options: nosniff', so a browser may re-interpret an uploaded or "
        "user-controlled response as HTML or script",
    ),
    (
        "x-frame-options",
        "no framing control (X-Frame-Options or a CSP frame-ancestors directive), so the page "
        "can be embedded invisibly and clickjacked",
    ),
    (
        "referrer-policy",
        "no Referrer-Policy, so full URLs - including any identifiers or tokens in them - leak "
        "to every third party the page links to or loads from",
    ),
)

_HSTS_MIN_AGE = 15_552_000  # 180 days, the level the major preload lists require
_MAX_AGE = re.compile(r"max-age\s*=\s*(\d+)", re.IGNORECASE)


@register_check(
    "missing-security-headers",
    "Missing or weak security header",
    693,
    ScannerSeverity.MEDIUM,
    site_wide=True,
)
def check_security_headers(page: Page, context: CheckContext) -> list[CheckFinding]:
    """Protection mechanisms the application declined to turn on (CWE-693).

    Security headers are the cheapest controls in web security and the ones most often
    missing. Their absence rarely causes a breach on its own; it removes the mitigation
    that would have contained one, which is why they matter to a prioritisation framework:
    they multiply the impact of the injection, framing and content-type bugs around them.
    """
    if not _is_document(page):
        return []
    check = CHECKS["missing-security-headers"]
    out: list[CheckFinding] = []
    csp = page.header("content-security-policy")

    for header, why in _SECURITY_HEADERS:
        value = page.header(header)
        if header == "x-frame-options" and not value and csp and "frame-ancestors" in csp.lower():
            continue
        if not value:
            out.append(
                _finding(check, page, detail=why, param=header, confidence=0.9)
            )
        elif header == "x-content-type-options" and value.strip().lower() != "nosniff":
            out.append(
                _finding(
                    check,
                    page,
                    detail="X-Content-Type-Options is set to something other than 'nosniff', "
                    "which disables the protection entirely",
                    evidence=f"x-content-type-options: {value}",
                    param=header,
                    confidence=0.9,
                )
            )

    if csp:
        lowered = csp.lower()
        weaknesses = [token for token in ("'unsafe-inline'", "'unsafe-eval'") if token in lowered]
        if re.search(r"(?:default|script)-src[^;]*(?:^|\s)\*(?:\s|;|$)", lowered):
            weaknesses.append("wildcard source")
        if weaknesses:
            out.append(
                _finding(
                    check,
                    page,
                    detail="the Content-Security-Policy permits "
                    + ", ".join(weaknesses)
                    + ", which leaves script injection exploitable despite the policy being present",
                    evidence=f"content-security-policy: {csp}",
                    param="content-security-policy-weak",
                    severity=ScannerSeverity.LOW,
                    confidence=0.8,
                )
            )

    if page.scheme == "https":
        hsts = page.header("strict-transport-security")
        if not hsts:
            out.append(
                _finding(
                    check,
                    page,
                    detail="no Strict-Transport-Security on an HTTPS response, so a first visit "
                    "over plaintext can still be intercepted and downgraded",
                    param="strict-transport-security",
                    confidence=0.9,
                )
            )
        else:
            match = _MAX_AGE.search(hsts)
            age = int(match.group(1)) if match else 0
            if age < _HSTS_MIN_AGE:
                out.append(
                    _finding(
                        check,
                        page,
                        detail=f"Strict-Transport-Security max-age is {age}s, below the "
                        f"{_HSTS_MIN_AGE}s that makes the pin meaningful",
                        evidence=f"strict-transport-security: {hsts}",
                        param="strict-transport-security-weak",
                        severity=ScannerSeverity.LOW,
                        confidence=0.8,
                    )
                )
    return out


@register_check(
    "cookie-flags", "Cookie missing a security flag", 614, ScannerSeverity.MEDIUM,
    site_wide=True,
)
def check_cookie_flags(page: Page, context: CheckContext) -> list[CheckFinding]:
    """Session cookies without Secure, HttpOnly or SameSite (CWE-614/1004/1275).

    A session cookie without ``HttpOnly`` turns every cross-site scripting bug into full
    account takeover; without ``Secure`` it is sent over any plaintext request the attacker
    can provoke; without ``SameSite`` it is attached to cross-site requests, which is the
    precondition for CSRF. These flags are why the same XSS is a nuisance on one
    application and a breach on another.
    """
    if not _analysable(page) or not page.set_cookies:
        return []
    check = CHECKS["cookie-flags"]
    out: list[CheckFinding] = []
    https = page.scheme == "https"
    for raw in page.set_cookies:
        name = raw.split("=", 1)[0].strip()
        if not name:
            continue
        attributes = {part.strip().lower() for part in raw.split(";")[1:]}
        flags = {attribute.split("=", 1)[0].strip() for attribute in attributes}
        same_site = ""
        for attribute in attributes:
            if attribute.startswith("samesite"):
                same_site = attribute.split("=", 1)[-1].strip()
        evidence = raw[:300]

        if https and "secure" not in flags:
            out.append(
                _finding(
                    check,
                    page,
                    detail=f"cookie {name!r} is set without the Secure flag, so the browser will "
                    "also send it over plaintext HTTP",
                    evidence=evidence,
                    param=f"{name}:secure",
                    cwe_id=614,
                    confidence=0.9,
                )
            )
        if "httponly" not in flags:
            out.append(
                _finding(
                    check,
                    page,
                    detail=f"cookie {name!r} is readable from JavaScript (no HttpOnly), so any "
                    "script injection on this origin can steal it",
                    evidence=evidence,
                    param=f"{name}:httponly",
                    cwe_id=1004,
                    confidence=0.9,
                )
            )
        if not same_site:
            out.append(
                _finding(
                    check,
                    page,
                    detail=f"cookie {name!r} declares no SameSite attribute, so it is a candidate "
                    "for cross-site request forgery",
                    evidence=evidence,
                    param=f"{name}:samesite",
                    cwe_id=1275,
                    severity=ScannerSeverity.LOW,
                    confidence=0.8,
                )
            )
        elif same_site == "none" and "secure" not in flags:
            out.append(
                _finding(
                    check,
                    page,
                    detail=f"cookie {name!r} is SameSite=None without Secure, which browsers "
                    "reject and which exposes it cross-site where they do not",
                    evidence=evidence,
                    param=f"{name}:samesite",
                    cwe_id=1275,
                    confidence=0.85,
                )
            )
    return out


_VERSION_IN_HEADER = re.compile(r"\d+(?:\.\d+)+")
_VERSION_HEADERS: tuple[str, ...] = (
    "server",
    "x-powered-by",
    "x-aspnet-version",
    "x-aspnetmvc-version",
    "x-generator",
    "x-drupal-cache",
    "x-runtime",
)


@register_check(
    "version-disclosure", "Server or framework version disclosed", 200, ScannerSeverity.LOW,
    site_wide=True,
)
def check_version_disclosure(page: Page, context: CheckContext) -> list[CheckFinding]:
    """Exact product versions handed out in response headers (CWE-200).

    On its own this is informational. In a prioritisation framework it is a multiplier: a
    precise version is what lets an attacker skip reconnaissance and go straight to a
    known exploit, and it is what lets this framework's applicability layer decide whether
    a CVE actually applies. Both effects are real, which is why the finding is recorded
    rather than suppressed.
    """
    if not _analysable(page):
        return []
    check = CHECKS["version-disclosure"]
    out: list[CheckFinding] = []
    for header in _VERSION_HEADERS:
        value = page.header(header)
        if value and _VERSION_IN_HEADER.search(value):
            out.append(
                _finding(
                    check,
                    page,
                    detail=f"the {header} response header names an exact product version, which "
                    "removes the reconnaissance step from an attack on a known vulnerability",
                    evidence=f"{header}: {value}",
                    param=header,
                    confidence=0.95,
                )
            )
    generator = re.search(
        r"""<meta[^>]+name=["']generator["'][^>]+content=["']([^"']+)["']""", page.body or "", re.I
    )
    if generator and _VERSION_IN_HEADER.search(generator.group(1)):
        out.append(
            _finding(
                check,
                page,
                detail="a generator meta tag names the exact product version of the application",
                evidence=generator.group(0)[:300],
                param="meta-generator",
                confidence=0.9,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Passive: response body
# ---------------------------------------------------------------------------

_ERROR_SIGNATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Python traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("Werkzeug debugger", re.compile(r"(?i)werkzeug\s+debugger|/console\?__debugger__")),
    ("Java stack trace", re.compile(r"\bat\s+[\w.$]+\([\w.$]+\.java:\d+\)")),
    ("Java exception", re.compile(r"\b(?:java\.lang|javax?\.servlet)\.[A-Za-z]+Exception\b")),
    (".NET stack trace", re.compile(r"(?i)(?:System\.[A-Za-z.]+Exception|\[HttpException)")),
    ("PHP error", re.compile(r"(?i)(?:Fatal error|Parse error|Warning):.{0,80}?\bon line\b")),
    ("PHP stack trace", re.compile(r"(?i)#\d+\s+/[^\s]+\.php\(\d+\)")),
    ("Ruby stack trace", re.compile(r"(?i)\.rb:\d+:in\s+[`']")),
    # Two frame shapes, because V8 prints both: named frames carry the function and the
    # location in parentheses, anonymous ones carry only a bare path. Matching just the
    # first form misses every trace whose top frames are arrow functions or module-level
    # code, which in practice is most of them.
    ("Node stack trace", re.compile(r"(?m)(?:^|>)\s*at\s+[\w.<>\[\]]+\s+\([^\n]{0,200}?:\d+:\d+\)")),
    ("Node stack trace", re.compile(r"(?m)(?:^|>)[\s&nbsp;]*at\s+/[^\s<]{0,200}?:\d+:\d+")),
    ("SQL error", re.compile(r"(?i)(?:SQLSTATE\[|You have an error in your SQL syntax|ORA-\d{5}|"
                             r"Unclosed quotation mark|PG::[A-Za-z]+Error|psycopg2\.|"
                             # SQLite, and the Node ORMs that surface it. A query generator
                             # named in a trace means attacker-shaped input reached the
                             # engine, which is the half of an injection proof a passive
                             # scan can honestly observe.
                             r"SQLITE_(?:ERROR|CONSTRAINT|MISUSE)\b|\bno such table:|"
                             r"SQLiteQueryGenerator|Sequelize(?:Database|Validation|Unique\w*)Error|"
                             r"SequelizeConnectionError|QueryFailedError)")),
    ("Go panic", re.compile(r"(?m)^panic: .+\n\ngoroutine \d+")),
    # Express' default error handler. Its ``stacktrace`` list is empty in production mode
    # and full in development, so this fires on both - correctly, because even the empty
    # form names the exception class and its message to an anonymous requester, which is
    # what tells an attacker whether a route exists, whether it is authenticated, and
    # which library rejected them.
    ("Express error page", re.compile(r"""(?i)<ul\s+id=['"]?stacktrace['"]?""")),
)

#: How much each signature actually gives away. A Werkzeug console is remote code
#: execution and a database error is half of an injection proof, so both outrank a stack
#: trace; an Express error page in production mode leaks the exception class and nothing
#: else, so it ranks below one. Anything unlisted stays at the check's own severity.
_ERROR_SEVERITY: dict[str, ScannerSeverity] = {
    "Werkzeug debugger": ScannerSeverity.HIGH,
    "SQL error": ScannerSeverity.HIGH,
    "Express error page": ScannerSeverity.LOW,
}

#: Ordering over severities, for picking the most revealing of several matches.
_SEVERITY_RANK: dict[ScannerSeverity, int] = {
    ScannerSeverity.INFO: 0,
    ScannerSeverity.LOW: 1,
    ScannerSeverity.MEDIUM: 2,
    ScannerSeverity.HIGH: 3,
    ScannerSeverity.CRITICAL: 4,
}


@register_check(
    "verbose-error", "Verbose error or stack trace", 209, ScannerSeverity.MEDIUM,
    # Collapsed by evidence, not by check: one leaky error handler answering fifty routes
    # is one defect, while a SQL error on one route and a Python traceback on another are
    # two, and :func:`collapse_site_wide` names every affected path on the survivor.
    site_wide=True,
)
def check_verbose_error(page: Page, context: CheckContext) -> list[CheckFinding]:
    """A stack trace or debug page returned to an anonymous user (CWE-209).

    A framework stack trace hands over file paths, library versions, SQL fragments and
    sometimes credentials, and a debug console hands over remote code execution. The
    condition is also a reliable signal that error handling was never exercised for this
    input, which is exactly where injection bugs live.
    """
    if not _analysable(page) or not page.body:
        return []
    check = CHECKS["verbose-error"]

    # Every signature is evaluated and the most revealing match wins, rather than the
    # first one in the table. A page that carries both a Node stack trace and the database
    # error underneath it is a database error: the stack trace says which file threw, the
    # SQL error says the query reached the engine with attacker-shaped input in it. Taking
    # whichever pattern happened to be listed first would have downgraded exactly the
    # findings worth reading.
    matches = [
        (label, match)
        for label, pattern in _ERROR_SIGNATURES
        if (match := pattern.search(page.body)) is not None
    ]
    if not matches:
        return []
    label, match = max(
        matches, key=lambda pair: _SEVERITY_RANK[_ERROR_SEVERITY.get(pair[0], ScannerSeverity.MEDIUM)]
    )
    start = max(0, match.start() - 60)
    return [
        _finding(
            check,
            page,
            detail=f"the response contains a {label}, disclosing internal paths, component "
            "versions and application structure to an unauthenticated requester",
            evidence=page.body[start : match.end() + 160],
            param=None,
            severity=_ERROR_SEVERITY.get(label, ScannerSeverity.MEDIUM),
            confidence=0.85,
            # Two routes leaking the same *kind* of error are one leaky error handler, so
            # they collapse together; a SQL error stays its own finding because its kind
            # differs. Without this the raw snippets differ on every route - each quotes
            # its own path - and nothing would ever collapse.
            signature=label,
        )
    ]


_DIRECTORY_LISTING = (
    re.compile(r"(?i)<title>\s*Index of /"),           # Apache, nginx autoindex
    re.compile(r"(?i)<h1>\s*Index of /"),
    re.compile(r"(?i)\[To Parent Directory\]"),         # IIS
    re.compile(r"(?i)Directory Listing For /"),         # Tomcat
    re.compile(r"(?i)<title>\s*listing directory /"),   # Node serve-index
)


@register_check("directory-listing", "Directory listing enabled", 548, ScannerSeverity.MEDIUM)
def check_directory_listing(page: Page, context: CheckContext) -> list[CheckFinding]:
    """An auto-generated index of a directory's contents (CWE-548).

    Directory listing turns "you would have to know the filename" into "here are the
    filenames", which is how backups, editor swap files and forgotten admin scripts get
    found. It also indicates a web server serving a path it was never configured for.
    """
    if not _analysable(page) or not page.body:
        return []
    check = CHECKS["directory-listing"]
    for pattern in _DIRECTORY_LISTING:
        match = pattern.search(page.body)
        if match:
            return [
                _finding(
                    check,
                    page,
                    detail="the server returns an automatically generated directory index, "
                    "enumerating files that were never meant to be discoverable",
                    evidence=page.body[max(0, match.start() - 40) : match.end() + 200],
                    confidence=0.9,
                )
            ]
    return []


_HTTP_SUBRESOURCE = re.compile(
    r"""(?is)<(?:script|img|iframe|link|source|embed|audio|video)\b[^>]*?\b(?:src|href)\s*=\s*["']?(http://[^"'\s>]+)"""
)


@register_check(
    "insecure-transport", "Cleartext transport", 319, ScannerSeverity.MEDIUM,
    site_wide=True,
)
def check_insecure_transport(page: Page, context: CheckContext) -> list[CheckFinding]:
    """Plaintext HTTP, and HTTPS pages that load HTTP subresources (CWE-319).

    Anything served over HTTP is readable and rewritable by every network between the user
    and the server. Mixed content is the same problem with a padlock drawn over it: one
    HTTP script tag on an HTTPS page hands script execution to any attacker on the path,
    which defeats every other control on the origin.
    """
    if not _analysable(page):
        return []
    check = CHECKS["insecure-transport"]
    out: list[CheckFinding] = []
    if page.scheme == "http":
        out.append(
            _finding(
                check,
                page,
                detail="this resource is served over plaintext HTTP, so its content and any "
                "credentials or session cookies sent with it are readable and modifiable in transit",
                param="scheme",
                confidence=0.95,
            )
        )
    elif page.scheme == "https" and page.body:
        matches = _HTTP_SUBRESOURCE.findall(page.body)
        if matches:
            out.append(
                _finding(
                    check,
                    page,
                    detail=f"an HTTPS page loads {len(matches)} subresource(s) over plaintext HTTP; "
                    "an attacker on the network can replace them and execute script on this origin",
                    evidence="; ".join(matches[:5]),
                    param="mixed-content",
                    confidence=0.9,
                )
            )
    return out


@register_check(
    "permissive-cors", "Permissive CORS policy", 942, ScannerSeverity.MEDIUM,
    site_wide=True,
)
def check_permissive_cors(page: Page, context: CheckContext) -> list[CheckFinding]:
    """Cross-origin access granted too widely (CWE-942).

    ``Access-Control-Allow-Origin: *`` alongside ``Access-Control-Allow-Credentials: true``
    is the combination that lets any website on the internet read this origin's
    authenticated responses; browsers reject that exact pair, but the same intent
    expressed by reflecting the request's Origin header works, and is the real-world shape
    of this bug. A reflected origin with credentials is authentication bypass by design.
    """
    if not _analysable(page):
        return []
    check = CHECKS["permissive-cors"]
    origin = (page.header("access-control-allow-origin") or "").strip()
    if not origin:
        return []
    credentials = (page.header("access-control-allow-credentials") or "").strip().lower() == "true"
    evidence = f"access-control-allow-origin: {origin}"
    if credentials:
        evidence += "; access-control-allow-credentials: true"

    if origin == "*" and credentials:
        return [
            _finding(
                check,
                page,
                detail="the response allows any origin AND credentialed cross-origin requests, so "
                "any site a victim visits can read this origin's authenticated responses",
                evidence=evidence,
                severity=ScannerSeverity.HIGH,
                confidence=0.9,
            )
        ]
    if origin == "*":
        return [
            _finding(
                check,
                page,
                detail="the response is readable by any origin; safe only if nothing here is "
                "private, which is a property that tends not to survive the next release",
                evidence=evidence,
                severity=ScannerSeverity.LOW,
                confidence=0.85,
            )
        ]
    if origin.lower() == "null":
        return [
            _finding(
                check,
                page,
                detail="the response allows the 'null' origin, which any sandboxed iframe or "
                "data: document can present",
                evidence=evidence,
                severity=ScannerSeverity.MEDIUM,
                confidence=0.8,
            )
        ]
    if credentials and origin not in ("", "*"):
        return [
            _finding(
                check,
                page,
                detail=f"credentialed cross-origin access is granted to {origin}; if that value is "
                "reflected from the request's Origin header, every origin is allowed",
                evidence=evidence,
                severity=ScannerSeverity.MEDIUM,
                confidence=0.5,
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Passive: forms and parameters
# ---------------------------------------------------------------------------

_CSRF_FIELD = re.compile(
    r"(?i)(?:csrf|xsrf|authenticity_token|__requestverificationtoken|_token\b|nonce|"
    r"csrfmiddlewaretoken|anti[-_]?forgery)"
)


@register_check("missing-csrf-token", "State-changing form without a CSRF token", 352,
                ScannerSeverity.MEDIUM)
def check_missing_csrf_token(page: Page, context: CheckContext) -> list[CheckFinding]:
    """A form that changes state and carries no anti-forgery token (CWE-352).

    Without a token, any page on the internet can make a logged-in victim's browser submit
    this form. The classic consequences are password change, email change and funds
    transfer - all performed by the legitimate user's own session, which is why server
    logs show nothing unusual afterwards.
    """
    if not _is_document(page) or not page.forms:
        return []
    check = CHECKS["missing-csrf-token"]
    out: list[CheckFinding] = []
    for form in page.forms:
        if not form.is_state_changing:
            continue
        names = " ".join(form.field_names)
        values = " ".join(field.value for field in form.fields if field.type.lower() == "hidden")
        if _CSRF_FIELD.search(names) or _CSRF_FIELD.search(values):
            continue
        out.append(
            _finding(
                check,
                page,
                detail=f"the {form.method.value} form to {form.action} has no anti-forgery token "
                "among its fields, so a third-party site can submit it using the victim's session",
                evidence=f"{form.method.value} {form.action} fields=({', '.join(form.field_names)})",
                param=form.action,
                confidence=0.7,
            )
        )
    return out


#: Parameter names that should never appear in a URL, because URLs are logged everywhere.
SENSITIVE_PARAM = re.compile(
    r"(?i)^(?:password|passwd|pwd|pass|secret|token|access[_-]?token|id[_-]?token|refresh[_-]?token|"
    r"api[_-]?key|apikey|auth|authorization|session|sessionid|sid|jwt|signature|sig|otp|code|"
    r"credit[_-]?card|card[_-]?number|cvv|ssn|national[_-]?id)$"
)


@register_check(
    "sensitive-data-in-url", "Sensitive data in URL query string", 598, ScannerSeverity.MEDIUM
)
def check_sensitive_data_in_url(page: Page, context: CheckContext) -> list[CheckFinding]:
    """Credentials or tokens carried as query parameters (CWE-598).

    A query string is not a private channel. It reaches server access logs, proxy logs,
    browser history, bookmarks and - through the Referer header - every third party the
    page links to. A token that travels in a URL should be assumed compromised, which
    makes this a real finding rather than a style complaint.
    """
    if not _analysable(page):
        return []
    check = CHECKS["sensitive-data-in-url"]
    out: list[CheckFinding] = []
    seen: set[tuple[str, str]] = set()
    for url in (page.effective_url, *page.links):
        for name in query_parameters(url):
            if not SENSITIVE_PARAM.match(name):
                continue
            key = (url, name.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                _finding(
                    check,
                    page,
                    detail=f"the parameter {name!r} carries a secret in the URL, where it is "
                    "recorded by proxies, server logs, browser history and the Referer header",
                    evidence=url[:300],
                    param=name,
                    url=url if url != page.effective_url else None,
                    confidence=0.75,
                )
            )
    for form in page.forms:
        if form.method == HttpMethod.GET and any(
            SENSITIVE_PARAM.match(name) for name in form.field_names
        ):
            names = [name for name in form.field_names if SENSITIVE_PARAM.match(name)]
            out.append(
                _finding(
                    check,
                    page,
                    detail=f"a GET form submits {', '.join(names)} in the query string, putting "
                    "the value into logs and browser history on every submission",
                    evidence=f"GET {form.action} fields=({', '.join(form.field_names)})",
                    param=names[0],
                    confidence=0.8,
                )
            )
    return out


#: Parameter names whose value is, structurally, somewhere to send the user next.
REDIRECT_PARAM = re.compile(
    r"(?i)^(?:url|uri|next|redirect|redirect_uri|redirect_url|redirectto|return|returnto|"
    r"return_url|returnurl|dest|destination|continue|goto|target|forward|callback|checkout_url|"
    r"image_url|out|to|r|u)$"
)
_URLISH_VALUE = re.compile(r"(?i)^(?:https?://|//|/[^/])")


@register_check("open-redirect-param", "Possible open redirect parameter", 601, ScannerSeverity.LOW)
def check_open_redirect_param(page: Page, context: CheckContext) -> list[CheckFinding]:
    """A parameter that structurally looks like a redirect target (CWE-601).

    Detection here is deliberately structural: the scanner reports that a parameter *is
    shaped like* a redirect destination, and never sends a request pointing at another
    host to see whether the application follows it. Open redirects matter because they
    lend the target's own domain to phishing and because OAuth flows treat the redirect
    parameter as a security boundary; confirming one is a human's job.
    """
    if not _analysable(page):
        return []
    check = CHECKS["open-redirect-param"]
    out: list[CheckFinding] = []
    seen: set[str] = set()
    for url in (page.effective_url, *page.links):
        query = urlsplit(url).query
        for name, value in parse_qsl(query, keep_blank_values=True):
            if name.lower() in seen or not REDIRECT_PARAM.match(name):
                continue
            if not _URLISH_VALUE.match(value or ""):
                continue
            seen.add(name.lower())
            out.append(
                _finding(
                    check,
                    page,
                    detail=f"the parameter {name!r} carries a URL or path that the application "
                    "appears to redirect to; if it is not validated against an allowlist this is "
                    "an open redirect usable for phishing and for OAuth token theft",
                    evidence=f"{name}={value}"[:300],
                    param=name,
                    url=url if url != page.effective_url else None,
                    confidence=0.4,
                )
            )
    return out


@register_check("insecure-form", "Insecure form configuration", 319, ScannerSeverity.MEDIUM)
def check_insecure_form(page: Page, context: CheckContext) -> list[CheckFinding]:
    """Forms that post over HTTP, and password fields left autocompleting (CWE-319/CWE-525).

    A login form posting to ``http://`` sends the password in clear text however the page
    itself was loaded - the padlock in the address bar is about the page, not the
    submission. A password field without ``autocomplete="off"`` leaves the credential in
    the browser's store, which matters on shared and kiosk machines and is the reason the
    control exists at all.
    """
    if not _is_document(page) or not page.forms:
        return []
    check = CHECKS["insecure-form"]
    out: list[CheckFinding] = []
    for form in page.forms:
        if urlsplit(form.action).scheme.lower() == "http":
            out.append(
                _finding(
                    check,
                    page,
                    detail=f"the form submits to {form.action} over plaintext HTTP, so every field "
                    "it carries - including any password - crosses the network in clear text",
                    evidence=f"{form.method.value} {form.action}",
                    param=form.action,
                    cwe_id=319,
                    severity=ScannerSeverity.HIGH if form.has_password_field else ScannerSeverity.MEDIUM,
                    confidence=0.9,
                )
            )
        for item in form.fields:
            if item.type.lower() != "password":
                continue
            form_off = (form.autocomplete or "").lower() == "off"
            field_off = (item.autocomplete or "").lower() in ("off", "new-password", "current-password")
            if not form_off and not field_off:
                out.append(
                    _finding(
                        check,
                        page,
                        detail=f"the password field {item.name or '(unnamed)'!r} does not disable "
                        "autocomplete, so the browser may store the credential on a shared machine",
                        evidence=f"form {form.action} password field {item.name!r}",
                        param=f"{form.action}:{item.name or 'password'}",
                        cwe_id=525,
                        severity=ScannerSeverity.LOW,
                        confidence=0.7,
                    )
                )
    return out


# ---------------------------------------------------------------------------
# Passive: exposed files and outdated libraries
# ---------------------------------------------------------------------------

#: Paths that should never be reachable. Matched against links the application itself
#: published - this scanner does not guess filenames.
SENSITIVE_PATHS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)/\.git(?:/|$)"), "a .git directory exposes the full source history, "
                                       "including secrets in past commits"),
    (re.compile(r"(?i)/\.svn(?:/|$)"), "a .svn directory exposes source and repository metadata"),
    (re.compile(r"(?i)/\.hg(?:/|$)"), "a .hg directory exposes the full source history"),
    (re.compile(r"(?i)/\.env(?:\.[\w-]+)?$"), "a .env file typically holds database credentials "
                                              "and API keys in plain text"),
    (re.compile(r"(?i)/\.htpasswd$"), "an .htpasswd file holds password hashes"),
    (re.compile(r"(?i)/\.DS_Store$"), "a .DS_Store file enumerates the directory's filenames"),
    (re.compile(r"(?i)/(?:wp-config|config|configuration|settings|database|db)\."
                r"(?:php|ya?ml|json|ini|xml)\.(?:bak|old|orig|save|swp|txt|dist|_)$"),
     "a configuration backup is served as a static file, in clear text"),
    (re.compile(r"(?i)\.(?:bak|old|orig|save|swp|swo|tmp|rej)$"),
     "an editor or deployment backup file is reachable over the web"),
    (re.compile(r"(?i)/[\w.-]*(?:backup|dump|database|db)[\w.-]*\."
                r"(?:zip|tar|tar\.gz|tgz|gz|sql|bz2|7z|rar)$"),
     "a database or site backup archive is downloadable"),
    (re.compile(r"(?i)/(?:\.aws/credentials|\.ssh/id_[a-z0-9]+|id_rsa)$"),
     "a private key or cloud credential file is reachable"),
)


@register_check("exposed-sensitive-file", "Sensitive file exposed", 538, ScannerSeverity.HIGH)
def check_exposed_sensitive_file(page: Page, context: CheckContext) -> list[CheckFinding]:
    """Version-control directories, dotfiles and backups that the site links to (CWE-538).

    Reported **only** when the application itself references the path, or when the crawl
    reached it through such a reference. Guessing hundreds of filenames is what makes
    scanners hostile to the systems they assess, and a guessed 404 proves nothing anyway.
    A linked ``.git/config`` or ``.env``, by contrast, is a confirmed disclosure of source
    code or credentials.
    """
    if not _analysable(page):
        return []
    check = CHECKS["exposed-sensitive-file"]
    out: list[CheckFinding] = []
    seen: set[str] = set()

    def _consider(url: str, *, fetched_status: int | None) -> None:
        path = url_path(url)
        for pattern, why in SENSITIVE_PATHS:
            if not pattern.search(path):
                continue
            if path in seen:
                return
            seen.add(path)
            reachable = fetched_status is not None and 200 <= fetched_status < 300
            out.append(
                _finding(
                    check,
                    page,
                    detail=f"{why}; the path {path} is referenced by the application"
                    + (" and returned a successful response" if reachable else ""),
                    evidence=url[:300],
                    param=path,
                    url=url,
                    severity=ScannerSeverity.CRITICAL if reachable else ScannerSeverity.HIGH,
                    confidence=0.95 if reachable else 0.6,
                )
            )
            return

    fetched: dict[str, int | None] = {
        other.effective_url: other.status for other in context.pages
    }
    for link in page.links:
        _consider(link, fetched_status=fetched.get(link))
    return out


@dataclass(frozen=True)
class LibraryAdvisory:
    """The version at which a front-end library stopped being known-vulnerable."""

    product: str
    fixed_in: str
    cve_ids: tuple[str, ...]
    summary: str


#: Minimum safe versions for the libraries :mod:`vulnprio.ingest.tech_fingerprint` can
#: identify from a filename. The CVE ids here are *candidates*: they are confirmed against
#: the configured intelligence feed before a finding claims them (see
#: :meth:`CheckContext.confirm_cves`), so the scanner never asserts a CVE the feed does
#: not know about as of the scan date.
LIBRARY_ADVISORIES: tuple[LibraryAdvisory, ...] = (
    LibraryAdvisory("jquery", "3.5.0", ("CVE-2020-11022", "CVE-2020-11023"),
                    "jQuery below 3.5.0 executes script in HTML passed to DOM manipulation methods"),
    LibraryAdvisory("angular.js", "1.8.0", ("CVE-2020-7676",),
                    "AngularJS below 1.8.0 has known sanitisation bypasses and is end-of-life"),
    LibraryAdvisory("bootstrap", "4.3.1", ("CVE-2019-8331",),
                    "Bootstrap below 4.3.1 allows XSS through tooltip and popover data attributes"),
    LibraryAdvisory("lodash", "4.17.21", ("CVE-2021-23337", "CVE-2020-8203"),
                    "lodash below 4.17.21 is vulnerable to prototype pollution and command injection in template"),
    LibraryAdvisory("moment.js", "2.29.4", ("CVE-2022-31129",),
                    "moment below 2.29.4 has a regular-expression denial of service in string parsing"),
    LibraryAdvisory("handlebars", "4.7.7", ("CVE-2021-23383",),
                    "Handlebars below 4.7.7 allows prototype pollution leading to remote code execution"),
)

_ADVISORY_BY_PRODUCT: Mapping[str, LibraryAdvisory] = {
    advisory.product: advisory for advisory in LIBRARY_ADVISORIES
}


def observed_libraries(page: Page) -> tuple[TechComponent, ...]:
    """Versioned components visible in this page's markup, links and headers."""
    return fingerprint_response(
        headers=page.headers,
        cookies=page.set_cookies,
        body=" ".join((page.body or "", *page.links)),
        url=page.effective_url,
    )


@register_check(
    "outdated-js-library", "Outdated JavaScript library", 1104, ScannerSeverity.MEDIUM,
    site_wide=True,
)
def check_outdated_js_library(page: Page, context: CheckContext) -> list[CheckFinding]:
    """A front-end library pinned to a version with known vulnerabilities (CWE-1104).

    Versioned library filenames (``jquery-1.12.4.min.js``) are the most reliable version
    evidence a black-box scan ever gets, and unmaintained third-party components are
    consistently among the most exploited weaknesses in web applications. The candidate
    CVEs are cross-referenced against the configured intelligence feed as of the scan
    date, so the finding carries real identifiers that the enrichment layer can price.
    """
    if not _analysable(page):
        return []
    check = CHECKS["outdated-js-library"]
    out: list[CheckFinding] = []
    seen: set[str] = set()
    for component in observed_libraries(page):
        advisory = _ADVISORY_BY_PRODUCT.get(component.product)
        if advisory is None or not component.version:
            continue
        if parse_version(component.version) >= parse_version(advisory.fixed_in):
            continue
        key = f"{component.product}@{component.version}"
        if key in seen:
            continue
        seen.add(key)
        confirmed, consulted = context.confirm_cves(advisory.cve_ids)
        cve_text = ", ".join(confirmed) if confirmed else "no CVE confirmed by the feed as of this date"
        out.append(
            _finding(
                check,
                page,
                detail=f"{component.product} {component.version} is below {advisory.fixed_in}: "
                f"{advisory.summary} ({cve_text})",
                evidence=key + ("" if consulted else " [no intelligence feed consulted]"),
                param=key,
                confidence=0.85 if confirmed and consulted else 0.6,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Active: results of the three allowlisted probes
# ---------------------------------------------------------------------------


@register_check(
    "reflected-input", "Unencoded input reflection point", 79, ScannerSeverity.MEDIUM,
    profile=ScanProfile.ACTIVE,
)
def check_reflected_marker(page: Page, context: CheckContext) -> list[CheckFinding]:
    """A parameter value echoed into the response without encoding (CWE-79).

    The probe that feeds this check sends a random alphanumeric marker - no tags, no
    quotes, no script - and reports whether it comes back unmodified. That establishes a
    *reflection point*: user input reaching the response body. Whether it is exploitable
    as cross-site scripting depends on the context it lands in and on the encoding applied
    around it, which this scanner deliberately does not test, because testing it means
    sending an exploit.
    """
    check = CHECKS["reflected-input"]
    out: list[CheckFinding] = []
    for probe in context.probes_for(page.effective_url, ProbeKind.REFLECTED_MARKER):
        if not probe.reflected:
            continue
        out.append(
            _finding(
                check,
                page,
                detail=f"a benign marker sent in the {probe.param!r} parameter was returned "
                "unencoded in the response body: this parameter reaches the output. Whether it is "
                "exploitable depends on the surrounding context and requires manual review",
                evidence=probe.evidence,
                param=probe.param,
                url=probe.url,
                confidence=0.6,
            )
        )
    return out


_RISKY_METHODS = ("PUT", "DELETE", "PATCH", "TRACE", "CONNECT")


@register_check(
    "options-methods", "HTTP methods advertised by OPTIONS", 650, ScannerSeverity.INFO,
    profile=ScanProfile.ACTIVE,
)
def check_options_methods(page: Page, context: CheckContext) -> list[CheckFinding]:
    """The verb set the server admits to supporting (CWE-650).

    One ``OPTIONS`` request, read for its ``Allow`` header. This is enumeration, not
    exploitation: the scanner records what the server says it accepts and never sends any
    of those verbs.
    """
    check = CHECKS["options-methods"]
    out: list[CheckFinding] = []
    for probe in context.probes_for(page.effective_url, ProbeKind.OPTIONS_METHOD):
        if not probe.allow_methods:
            continue
        out.append(
            _finding(
                check,
                page,
                detail="the server advertises the HTTP methods "
                f"{', '.join(probe.allow_methods)} on this path",
                evidence=f"Allow: {', '.join(probe.allow_methods)}",
                confidence=0.9,
            )
        )
    return out


@register_check(
    "state-changing-methods", "State-changing HTTP methods advertised", 650,
    ScannerSeverity.MEDIUM, profile=ScanProfile.ACTIVE,
)
def check_state_changing_methods(page: Page, context: CheckContext) -> list[CheckFinding]:
    """Write and trace verbs offered on a resource (CWE-650).

    ``PUT`` and ``DELETE`` reachable without authentication is arbitrary file write and
    deletion; ``TRACE`` enables cross-site tracing. The scanner reports only what
    ``OPTIONS`` advertised. It never sends ``PUT``, ``DELETE`` or ``PATCH``, and never
    sends a request body to any of them: verifying this class of issue safely requires a
    human with a rollback plan.
    """
    check = CHECKS["state-changing-methods"]
    out: list[CheckFinding] = []
    for probe in context.probes_for(page.effective_url, ProbeKind.OPTIONS_METHOD):
        risky = tuple(method for method in probe.allow_methods if method in _RISKY_METHODS)
        if not risky:
            continue
        out.append(
            _finding(
                check,
                page,
                detail=f"the server advertises {', '.join(risky)} on this path. If these are "
                "reachable without authentication they permit arbitrary write, deletion or "
                "cross-site tracing; this scanner reports the advertisement and does not send them",
                evidence=f"Allow: {', '.join(probe.allow_methods)}",
                param=",".join(risky),
                severity=ScannerSeverity.HIGH if {"PUT", "DELETE"} & set(risky) else ScannerSeverity.MEDIUM,
                confidence=0.6,
            )
        )
    return out


def forms_of(pages: Iterable[Page]) -> tuple[FormInfo, ...]:
    """Every form across a crawl - used by the runner to record them as endpoints."""
    seen: dict[tuple[str, str], FormInfo] = {}
    for page in pages:
        for form in page.forms:
            seen.setdefault((form.action, form.method.value), form)
    return tuple(seen.values())
