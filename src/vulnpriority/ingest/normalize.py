"""URL canonicalisation, path templating, identifier construction and CWE/CVE extraction.

Why this module exists: a scanner reports the same underlying defect under dozens of
concrete URLs (``/users/1``, ``/users/2``, ...). Templating the path and deriving every
identifier from the *templated* form is what collapses forty alerts back into the one
vulnerability an engineer actually has to fix. Doing it in exactly one place is what makes
identifiers stable across scanners, across files and across runs, which the whole
evaluation protocol depends on.

Nothing here touches the network, the clock or any random source.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from vulnpriority.core.enums import HttpMethod, PrivilegeLevel, Provenance, ScannerSeverity
from vulnpriority.core.hashing import stable_id
from vulnpriority.core.models import Endpoint, TechComponent, UntrustedText

from vulnpriority.ingest.tech_fingerprint import merge_tech

__all__ = [
    "ID_PLACEHOLDER",
    "MAX_RESPONSE_SAMPLE_CHARS",
    "DEFAULT_SCANNED_AT",
    "HttpMessage",
    "canonical_url",
    "url_host",
    "url_path",
    "query_parameters",
    "template_path",
    "templated_path_of",
    "is_identifier_segment",
    "make_app_id",
    "make_endpoint_id",
    "make_finding_id",
    "make_scan_id",
    "extract_cwe",
    "extract_cves",
    "infer_auth_level",
    "severity_from_string",
    "severity_from_riskcode",
    "confidence_from_code",
    "method_is_state_changing",
    "http_method",
    "method_from_request",
    "parse_http_message",
    "parse_headers",
    "decode_maybe_base64",
    "strip_html_tags",
    "collapse_whitespace",
    "parse_timestamp",
    "scanner_text",
    "target_text",
    "dedupe_untrusted",
    "header_mapping",
    "EndpointAccumulator",
]

#: Replacement written into a templated path for any segment that looks like an identifier.
ID_PLACEHOLDER = "{id}"

#: Response bodies are kept only as a bounded sample: they are untrusted and can be huge.
MAX_RESPONSE_SAMPLE_CHARS = 4000

#: Used when a report carries no usable timestamp. A constant (never ``now()``) so that
#: parsing the same file twice produces the same scan id.
DEFAULT_SCANNED_AT = datetime(1970, 1, 1, 0, 0, 0)

_NUMERIC_SEGMENT = re.compile(r"^[0-9]+$")
_UUID_SEGMENT = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_HEX_SEGMENT = re.compile(r"^[0-9a-fA-F]{32,}$")
_BASE64_SEGMENT = re.compile(r"^[A-Za-z0-9+/\-_]{16,}={0,2}$")

_CWE_RE = re.compile(r"cwe[\s._-]*(\d{1,5})", re.IGNORECASE)
_CVE_RE = re.compile(r"CVE[-_](\d{4})[-_](\d{4,7})", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")

#: Path tokens that mean "administrative surface" (DESIGN 3.1 auth inference).
_ADMIN_TOKENS = (
    "admin",
    "administrator",
    "administration",
    "wp-admin",
    "sysadmin",
    "backoffice",
    "back-office",
    "phpmyadmin",
    "adminer",
    "console",
    "manage",
    "management",
    "moderator",
    "superuser",
)

#: Path tokens that mean "area only an authenticated user reaches".
_AUTH_TOKENS = (
    "account",
    "accounts",
    "billing",
    "cart",
    "checkout",
    "dashboard",
    "internal",
    "invoice",
    "invoices",
    "me",
    "myaccount",
    "order",
    "orders",
    "payment",
    "payments",
    "private",
    "profile",
    "secure",
    "session",
    "settings",
    "subscription",
    "user",
    "users",
    "wallet",
)

#: Actions inside an authenticated area that an anonymous user must still be able to reach.
_PUBLIC_AUTH_ACTIONS = (
    "login",
    "signin",
    "sign-in",
    "log-in",
    "register",
    "signup",
    "sign-up",
    "password-reset",
    "password_reset",
    "reset-password",
    "forgot-password",
    "recover",
)

_ADMIN_RE = re.compile(r"(?:^|[^a-z0-9])(?:%s)(?:[^a-z0-9]|$)" % "|".join(_ADMIN_TOKENS))
_AUTH_RE = re.compile(r"(?:^|[^a-z0-9])(?:%s)(?:[^a-z0-9]|$)" % "|".join(_AUTH_TOKENS))
_PUBLIC_AUTH_RE = re.compile(r"(?:^|[^a-z0-9])(?:%s)(?:[^a-z0-9]|$)" % "|".join(_PUBLIC_AUTH_ACTIONS))

_SEVERITY_WORDS: dict[str, ScannerSeverity] = {
    "info": ScannerSeverity.INFO,
    "informational": ScannerSeverity.INFO,
    "information": ScannerSeverity.INFO,
    "none": ScannerSeverity.INFO,
    "note": ScannerSeverity.INFO,
    "unknown": ScannerSeverity.INFO,
    "low": ScannerSeverity.LOW,
    "minor": ScannerSeverity.LOW,
    "warning": ScannerSeverity.LOW,
    "medium": ScannerSeverity.MEDIUM,
    "moderate": ScannerSeverity.MEDIUM,
    "high": ScannerSeverity.HIGH,
    "important": ScannerSeverity.HIGH,
    "major": ScannerSeverity.HIGH,
    "severe": ScannerSeverity.HIGH,
    "critical": ScannerSeverity.CRITICAL,
    "crit": ScannerSeverity.CRITICAL,
    "blocker": ScannerSeverity.CRITICAL,
    "emergency": ScannerSeverity.CRITICAL,
    "certain": ScannerSeverity.CRITICAL,
}

#: ZAP ``riskcode`` / Burp-style ordinal severities.
_RISKCODE_SEVERITY: dict[int, ScannerSeverity] = {
    0: ScannerSeverity.INFO,
    1: ScannerSeverity.LOW,
    2: ScannerSeverity.MEDIUM,
    3: ScannerSeverity.HIGH,
    4: ScannerSeverity.CRITICAL,
}

#: ZAP ``confidence`` code to a probability in [0, 1].
_CONFIDENCE_CODE: dict[int, float] = {0: 0.05, 1: 0.25, 2: 0.5, 3: 0.8, 4: 0.95}

_STATE_CHANGING = frozenset(
    {HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH, HttpMethod.DELETE}
)

#: ZAP spells September "Sept"; ``%b`` only accepts "Sep". Word-bounded so it cannot
#: touch anything else in the string.
_MONTH_SEPT = re.compile(r"\bSept\b", re.IGNORECASE)

_TIMESTAMP_FORMATS = (
    "%a, %d %b %Y %H:%M:%S",
    "%a, %d %b %Y %H:%M:%S %Z",
    "%a %b %d %H:%M:%S %Z %Y",
    "%a %b %d %H:%M:%S %Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d",
)


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------


def _normalize_path(path: str) -> str:
    """Collapse ``//``, resolve ``.``/``..`` and drop a trailing slash (root excepted)."""
    if not path:
        return "/"
    out: list[str] = []
    for segment in path.replace("\\", "/").split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if out:
                out.pop()
            continue
        out.append(segment)
    return "/" + "/".join(out)


def _base_origin(base: str | None) -> str:
    """``scheme://host[:port]`` for a possibly bare host string."""
    if not base:
        return ""
    raw = base.strip()
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlsplit(raw)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def canonical_url(url: str, base: str | None = None, default_scheme: str = "https") -> str:
    """Canonical form of a scanner-reported URL.

    Lowercases scheme and host, drops the default port, the fragment and any userinfo,
    normalises the path and sorts query parameters. Two scanners describing the same
    request must produce the same string here or every downstream identifier diverges.
    """
    raw = (url or "").strip()
    if not raw:
        origin = _base_origin(base)
        return f"{origin}/" if origin else ""
    if raw.startswith("//"):
        raw = f"{default_scheme}:{raw}"
    elif "://" not in raw:
        origin = _base_origin(base)
        if raw.startswith("/"):
            raw = f"{origin}{raw}" if origin else f"{default_scheme}://{raw.lstrip('/')}"
        else:
            raw = f"{default_scheme}://{raw}"
    parsed = urlsplit(raw)
    scheme = (parsed.scheme or default_scheme).lower()
    host = (parsed.hostname or "").lower()
    if not host:
        host = urlsplit(_base_origin(base) or f"{default_scheme}://").hostname or ""
    try:
        port = parsed.port
    except ValueError:
        port = None
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if (port is None or default_port) else f"{host}:{port}"
    path = _normalize_path(parsed.path)
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query = urlencode(sorted(pairs), doseq=False)
    return urlunsplit((scheme, netloc, path, query, ""))


def url_host(url: str, base: str | None = None) -> str:
    """Lower-cased host (no port) of a URL."""
    return urlsplit(canonical_url(url, base=base)).hostname or ""


def url_path(url: str, base: str | None = None) -> str:
    """Normalised (not yet templated) path of a URL."""
    return urlsplit(canonical_url(url, base=base)).path or "/"


def query_parameters(url: str, base: str | None = None) -> tuple[str, ...]:
    """Sorted, de-duplicated query parameter names of a URL."""
    query = urlsplit(canonical_url(url, base=base)).query
    return tuple(sorted({key for key, _ in parse_qsl(query, keep_blank_values=True)}))


# ---------------------------------------------------------------------------
# Path templating
# ---------------------------------------------------------------------------


def _looks_base64(segment: str) -> bool:
    """Opaque base64/base64url-looking token (session ids, encoded object references)."""
    if len(segment) < 16 or not _BASE64_SEGMENT.match(segment):
        return False
    if segment.endswith("="):
        return True
    has_digit = any(character.isdigit() for character in segment)
    has_upper = any(character.isupper() for character in segment)
    has_lower = any(character.islower() for character in segment)
    return has_digit and has_upper and has_lower


def is_identifier_segment(segment: str) -> bool:
    """True when a path segment is an instance identifier rather than a route name.

    Numeric ids, UUIDs, 32+ character hex digests and opaque base64 tokens all identify
    one row; the route is the thing a vulnerability belongs to.
    """
    if not segment or segment == ID_PLACEHOLDER:
        return False
    return bool(
        _NUMERIC_SEGMENT.match(segment)
        or _UUID_SEGMENT.match(segment)
        or _HEX_SEGMENT.match(segment)
        or _looks_base64(segment)
    )


def template_path(path: str) -> str:
    """Replace every identifier-looking segment with ``{id}``.

    ``/users/123/orders/9`` becomes ``/users/{id}/orders/{id}``. This is what stops one
    vulnerability appearing as forty findings.
    """
    normalized = _normalize_path(path)
    if normalized == "/":
        return "/"
    segments = normalized.split("/")[1:]
    return "/" + "/".join(
        ID_PLACEHOLDER if is_identifier_segment(segment) else segment for segment in segments
    )


def templated_path_of(url: str, base: str | None = None) -> str:
    """Canonicalise a URL and return its templated path."""
    return template_path(url_path(url, base=base))


# ---------------------------------------------------------------------------
# Identifiers (DESIGN 3.1)
# ---------------------------------------------------------------------------


def make_app_id(host_or_name: str) -> str:
    """Deterministic application id when the report does not name the application."""
    return stable_id("app", (host_or_name or "").strip().lower())


def make_endpoint_id(app_id: str, host: str, method: HttpMethod | str, templated_path: str) -> str:
    """``stable_id("ep", app_id, host, method, templated_path)`` exactly as DESIGN 3.1 states."""
    method_value = method.value if isinstance(method, HttpMethod) else str(method).upper()
    return stable_id("ep", app_id, host.lower(), method_value, templated_path)


def make_finding_id(
    scan_id: str, endpoint_id: str, plugin_id_or_name: str, param: str | None = None
) -> str:
    """``stable_id("f", scan_id, endpoint_id, plugin_id or name, param or "")``."""
    return stable_id("f", scan_id, endpoint_id, plugin_id_or_name, param or "")


def make_scan_id(app_id: str, scanned_at: datetime, scanner_name: str) -> str:
    """``stable_id("scan", app_id, scanned_at.isoformat(), scanner_name)``."""
    return stable_id("scan", app_id, scanned_at.isoformat(), scanner_name)


# ---------------------------------------------------------------------------
# CWE / CVE
# ---------------------------------------------------------------------------


def extract_cwe(*values: object) -> int | None:
    """First plausible CWE id found in the given values, in order.

    Accepts both bare scanner fields (``"89"``) and free text (``"... see CWE-89 ..."``).
    ``0`` and ``-1`` are the usual scanner sentinels for "no CWE" and are rejected.
    """
    for value in values:
        if value is None:
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            if value > 0:
                return value
            continue
        text = str(value).strip()
        if not text:
            continue
        if text.isdigit():
            number = int(text)
            if number > 0:
                return number
            continue
        match = _CWE_RE.search(text)
        if match:
            number = int(match.group(1))
            if number > 0:
                return number
    return None


def extract_cves(*values: object) -> tuple[str, ...]:
    """Every CVE id mentioned in the given values, upper-cased, de-duplicated and sorted."""
    found: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            found.update(extract_cves(*value))
            continue
        for match in _CVE_RE.finditer(str(value)):
            found.add(f"CVE-{match.group(1)}-{match.group(2)}")
    return tuple(sorted(found))


# ---------------------------------------------------------------------------
# Small scanner-field normalisers
# ---------------------------------------------------------------------------


def infer_auth_level(path: str, status: int | None = None) -> PrivilegeLevel:
    """Privilege an endpoint appears to require (DESIGN 3.1 auth inference).

    ``ADMIN`` when the path matches the admin lexicon, ``USER`` when an unauthenticated
    probe was refused (401/403) or the path names an authenticated area, ``NONE`` otherwise.
    Structure only: there are no manual asset tags anywhere in this framework.

    One refinement over the bare lexicon: ``/accounts/login`` sits inside an authenticated
    area but is by construction reachable anonymously, so the public auth actions are
    exempted from the USER branch. The ADMIN branch is not exempted, because an admin
    login page is still admin surface and the criticality model should see it as such.
    """
    lowered = (path or "").lower()
    if _ADMIN_RE.search(lowered):
        return PrivilegeLevel.ADMIN
    if status in (401, 403, 407):
        return PrivilegeLevel.USER
    if _AUTH_RE.search(lowered) and not _PUBLIC_AUTH_RE.search(lowered):
        return PrivilegeLevel.USER
    return PrivilegeLevel.NONE


def severity_from_string(value: object, default: ScannerSeverity = ScannerSeverity.INFO) -> ScannerSeverity:
    """Map any scanner severity spelling to :class:`ScannerSeverity`.

    Handles ZAP's ``"High (Medium)"`` riskdesc, Burp's ``"Information"``, Nuclei's
    ``"critical"`` and bare ordinal codes.
    """
    if isinstance(value, ScannerSeverity):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text.lstrip("-").isdigit():
        return severity_from_riskcode(text, default=default)
    word = re.split(r"[^a-z]+", text)[0]
    return _SEVERITY_WORDS.get(word, default)


def severity_from_riskcode(value: object, default: ScannerSeverity = ScannerSeverity.INFO) -> ScannerSeverity:
    """ZAP ``riskcode`` (0..4) to :class:`ScannerSeverity`."""
    try:
        code = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return _RISKCODE_SEVERITY.get(code, default)


def confidence_from_code(value: object, default: float = 0.5) -> float:
    """ZAP ``confidence`` code (0..4) to a probability in [0, 1]."""
    try:
        code = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return _CONFIDENCE_CODE.get(code, default)


def http_method(value: object, default: HttpMethod = HttpMethod.GET) -> HttpMethod:
    """Parse an HTTP method defensively; unknown verbs fall back to ``default``."""
    if isinstance(value, HttpMethod):
        return value
    try:
        return HttpMethod(str(value).strip().upper())
    except (AttributeError, ValueError):
        return default


def method_is_state_changing(method: HttpMethod | str) -> bool:
    """True for verbs that mutate server state, a base ranking feature."""
    return http_method(method) in _STATE_CHANGING


def method_from_request(raw: str | None, default: HttpMethod = HttpMethod.GET) -> HttpMethod:
    """Method from a raw HTTP request blob (``POST /api/login HTTP/1.1``)."""
    if not raw:
        return default
    first_line = raw.lstrip().splitlines()[0] if raw.strip() else ""
    token = first_line.split(" ", 1)[0] if first_line else ""
    return http_method(token, default=default)


# ---------------------------------------------------------------------------
# Raw HTTP messages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HttpMessage:
    """A split raw HTTP message. ``headers`` keys are lower-cased."""

    start_line: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""

    @property
    def status(self) -> int | None:
        """Status code from a response start line, or ``None`` for a request."""
        parts = self.start_line.split()
        if len(parts) >= 2 and parts[0].upper().startswith("HTTP/"):
            try:
                return int(parts[1])
            except ValueError:
                return None
        return None

    @property
    def content_type(self) -> str | None:
        """``Content-Type`` without its parameters."""
        raw = self.headers.get("content-type")
        return raw.split(";")[0].strip().lower() if raw else None

    @property
    def sets_cookie(self) -> bool:
        return "set-cookie" in self.headers

    @property
    def cookie_names(self) -> tuple[str, ...]:
        """Cookie names the response sets, in order of appearance."""
        raw = self.headers.get("set-cookie", "")
        names: list[str] = []
        for chunk in raw.split("\n"):
            name = chunk.split("=", 1)[0].strip()
            if name and name not in names:
                names.append(name)
        return tuple(names)


def parse_headers(raw: str | None) -> dict[str, str]:
    """Header block to a lower-cased dict; repeated headers are joined with ``\\n``."""
    headers: dict[str, str] = {}
    if not raw:
        return headers
    for line in raw.replace("\r\n", "\n").split("\n"):
        if not line.strip() or ":" not in line:
            continue
        name, _, value = line.partition(":")
        key = name.strip().lower()
        value = value.strip()
        headers[key] = f"{headers[key]}\n{value}" if key in headers else value
    return headers


def parse_http_message(raw: str | None) -> HttpMessage:
    """Split a raw HTTP request or response into start line, headers and body."""
    if not raw:
        return HttpMessage()
    text = raw.replace("\r\n", "\n")
    head, separator, body = text.partition("\n\n")
    if not separator:
        head, body = text, ""
    lines = head.split("\n")
    start_line = lines[0].strip() if lines else ""
    return HttpMessage(start_line=start_line, headers=parse_headers("\n".join(lines[1:])), body=body)


def decode_maybe_base64(value: str | None, is_base64: bool = True) -> str:
    """Decode a base64 blob from a scanner export, or return ``""``.

    Scanner exports embed attacker-influenced request and response bodies as base64. The
    bytes are decoded and lossily converted to text for inspection only: nothing here
    interprets, evaluates or executes the result, and a malformed blob is dropped rather
    than raised so one corrupt issue cannot fail a whole report.
    """
    if not value:
        return ""
    if not is_base64:
        return value
    candidate = "".join(value.split())
    padding = (-len(candidate)) % 4
    try:
        data = base64.b64decode(candidate + ("=" * padding), validate=False)
    except (binascii.Error, ValueError):
        return ""
    return data.decode("utf-8", errors="replace")


def strip_html_tags(text: str | None) -> str:
    """Remove HTML tags from scanner description fields (ZAP writes HTML in ``desc``)."""
    if not text:
        return ""
    without_tags = _TAG_RE.sub(" ", text)
    unescaped = (
        without_tags.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
    )
    return collapse_whitespace(unescaped)


def collapse_whitespace(text: str | None) -> str:
    """Collapse runs of spaces/tabs and trim, keeping paragraph structure readable."""
    if not text:
        return ""
    lines = [_WS_RE.sub(" ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    return " ".join(line for line in lines if line).strip()


def parse_timestamp(value: object) -> datetime | None:
    """Parse the assorted timestamp spellings scanners emit into a naive UTC datetime.

    Naive UTC everywhere so that ``scanned_at.isoformat()`` - which feeds the scan id -
    cannot change because a report happened to carry an offset.
    """
    if isinstance(value, datetime):
        return _as_naive_utc(value)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    iso_candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return _as_naive_utc(datetime.fromisoformat(iso_candidate))
    except ValueError:
        pass
    # ``%b`` accepts the three-letter abbreviation and nothing else, but real scanners do
    # not all agree with the C library about September: ZAP writes "Sept". Normalising it
    # here rather than adding a format keeps the fix in one place and out of every caller.
    # Getting this wrong is expensive, not cosmetic: an unparsed timestamp becomes
    # ``DEFAULT_SCANNED_AT``, which becomes the run's intelligence as-of date, and every
    # feed lookup is then asked what was known in 1970.
    text = _MONTH_SEPT.sub("Sep", text)
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return _as_naive_utc(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return None


def _as_naive_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Untrusted text wrappers
# ---------------------------------------------------------------------------


def scanner_text(text: str | None, source_url: str | None = None) -> UntrustedText | None:
    """Wrap scanner-authored prose as ``SCANNER_OUTPUT`` (tier ``SCANNER``)."""
    cleaned = strip_html_tags(text) if text and "<" in text else collapse_whitespace(text)
    if not cleaned:
        return None
    return UntrustedText(
        text=cleaned, provenance=Provenance.SCANNER_OUTPUT, source_url=source_url
    )


def target_text(text: str | None, source_url: str | None = None) -> UntrustedText | None:
    """Wrap anything echoed from the application as ``TARGET_RESPONSE`` (tier ``TARGET_CONTENT``).

    Deliberately unmodified: the sandbox, not ingest, decides how to neutralise it, and
    normalising here would destroy the very artefacts the adversarial evaluation studies.
    """
    if text is None:
        return None
    body = text[:MAX_RESPONSE_SAMPLE_CHARS]
    if not body.strip():
        return None
    return UntrustedText(text=body, provenance=Provenance.TARGET_RESPONSE, source_url=source_url)


# ---------------------------------------------------------------------------
# Endpoint construction
# ---------------------------------------------------------------------------


@dataclass
class _EndpointDraft:
    """Mutable accumulator for one endpoint while a report is being read."""

    endpoint_id: str
    app_id: str
    host: str
    url: str
    path: str
    method: HttpMethod
    auth_required: PrivilegeLevel = PrivilegeLevel.NONE
    response_status: int | None = None
    response_content_type: str | None = None
    response_size_bytes: int | None = None
    sets_cookie: bool = False
    parameters: list[str] = field(default_factory=list)
    links_to: list[str] = field(default_factory=list)
    response_sample: UntrustedText | None = None
    observed_tech: list[TechComponent] = field(default_factory=list)


class EndpointAccumulator:
    """Builds the de-duplicated :class:`Endpoint` set of one scan.

    Every parser funnels its URL observations through here so that two alerts on
    ``/users/1`` and ``/users/2`` become one endpoint with one identifier, and so that
    auth inference happens in exactly one place.
    """

    def __init__(self, app_id: str, *, infer_auth: bool = True) -> None:
        self._app_id = app_id
        self._infer_auth = infer_auth
        self._drafts: dict[str, _EndpointDraft] = {}
        self._hosts: list[str] = []
        self._scan_tech: list[TechComponent] = []

    @property
    def app_id(self) -> str:
        return self._app_id

    def add(
        self,
        url: str,
        method: HttpMethod | str = HttpMethod.GET,
        *,
        base: str | None = None,
        parameters: Iterable[str] = (),
        auth_required: PrivilegeLevel | None = None,
        response_status: int | None = None,
        response_content_type: str | None = None,
        response_size_bytes: int | None = None,
        sets_cookie: bool = False,
        response_sample: UntrustedText | None = None,
        observed_tech: Iterable[TechComponent] = (),
        links_to: Iterable[str] = (),
    ) -> str:
        """Record one observation of an endpoint and return its stable id."""
        canonical = canonical_url(url, base=base)
        host = url_host(canonical) or url_host(base or "")
        verb = http_method(method)
        templated = template_path(url_path(canonical))
        endpoint_id = make_endpoint_id(self._app_id, host, verb, templated)

        draft = self._drafts.get(endpoint_id)
        if draft is None:
            origin = urlsplit(canonical)
            draft = _EndpointDraft(
                endpoint_id=endpoint_id,
                app_id=self._app_id,
                host=host,
                url=urlunsplit((origin.scheme, origin.netloc, templated, "", "")),
                path=templated,
                method=verb,
            )
            self._drafts[endpoint_id] = draft
        if host and host not in self._hosts:
            self._hosts.append(host)

        for name in list(query_parameters(canonical)) + [p for p in parameters if p]:
            cleaned = str(name).strip()
            if cleaned and cleaned not in draft.parameters:
                draft.parameters.append(cleaned)
        for target in links_to:
            if target and target not in draft.links_to:
                draft.links_to.append(target)
        if draft.response_status is None and response_status is not None:
            draft.response_status = response_status
        if draft.response_content_type is None and response_content_type:
            draft.response_content_type = response_content_type
        if draft.response_size_bytes is None and response_size_bytes is not None:
            draft.response_size_bytes = response_size_bytes
        draft.sets_cookie = draft.sets_cookie or sets_cookie
        if draft.response_sample is None and response_sample is not None:
            draft.response_sample = response_sample
        for component in observed_tech:
            if component not in draft.observed_tech:
                draft.observed_tech.append(component)
            if component not in self._scan_tech:
                self._scan_tech.append(component)

        explicit = auth_required if auth_required is not None else PrivilegeLevel.NONE
        inferred = (
            infer_auth_level(templated, draft.response_status) if self._infer_auth else PrivilegeLevel.NONE
        )
        draft.auth_required = PrivilegeLevel(max(draft.auth_required, explicit, inferred))
        return endpoint_id

    def add_tech(self, components: Iterable[TechComponent]) -> None:
        """Record scan-level technology observed outside any single endpoint."""
        for component in components:
            if component not in self._scan_tech:
                self._scan_tech.append(component)

    def endpoints(self) -> tuple[Endpoint, ...]:
        """Frozen endpoints in first-observation order."""
        return tuple(
            Endpoint(
                endpoint_id=draft.endpoint_id,
                app_id=draft.app_id,
                host=draft.host,
                url=draft.url,
                path=draft.path,
                method=draft.method,
                auth_required=draft.auth_required,
                internet_facing=True,
                response_status=draft.response_status,
                response_content_type=draft.response_content_type,
                response_size_bytes=draft.response_size_bytes,
                sets_cookie=draft.sets_cookie,
                parameters=tuple(sorted(draft.parameters)),
                links_to=tuple(draft.links_to),
                response_sample=draft.response_sample,
                observed_tech=merge_tech(draft.observed_tech),
            )
            for draft in self._drafts.values()
        )

    def hosts(self) -> tuple[str, ...]:
        """Hosts seen, in first-observation order."""
        return tuple(self._hosts)

    def tech_stack(self) -> tuple[TechComponent, ...]:
        """Merged technology stack across every observation."""
        return merge_tech(self._scan_tech)

    def primary_host(self) -> str:
        return self._hosts[0] if self._hosts else ""


def dedupe_untrusted(items: Sequence[UntrustedText]) -> tuple[UntrustedText, ...]:
    """Drop duplicate untrusted blobs by content hash, preserving order."""
    seen: set[str] = set()
    out: list[UntrustedText] = []
    for item in items:
        if item.sha256 in seen:
            continue
        seen.add(item.sha256)
        out.append(item)
    return tuple(out)


def header_mapping(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Lower-case a header mapping without mutating the caller's dict."""
    return {str(key).lower(): str(value) for key, value in (headers or {}).items()}
