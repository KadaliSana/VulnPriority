"""Every limit this scanner obeys, in one place.

Nothing in :mod:`vulnprio.scan` decides for itself whether a request is permitted. The
HTTP client, the crawler and the active probes all ask this module, so the safety
properties of the scanner can be read, reviewed and tested as a single artefact rather
than reconstructed from behaviour scattered across five files.

What is enforced here:

* **Authorisation.** :func:`require_authorization` refuses unless
  ``ScanRequest.authorized is True``. There is no environment variable, no configuration
  key and no keyword argument anywhere in this package that skips it.
* **Scope.** :func:`host_in_scope` admits the authorised host and the explicitly listed
  extra hosts, and nothing else - not subdomains, not sibling domains, not redirect
  targets. Suffix matching is deliberately not used: ``example.com.attacker.net`` would
  pass a naive ``endswith`` check.
* **Private and metadata targets.** :func:`is_private_host` rejects loopback, RFC1918,
  carrier-grade NAT, link-local, multicast and reserved literals, ``.local``-style
  suffixes and the cloud metadata endpoints, so the tool cannot be aimed at
  ``169.254.169.254`` by accident.
* **Rate.** :class:`RateLimiter` is a token bucket over a monotonic clock, with the clock
  and the sleep function injectable so spacing is testable without real time.
* **robots.txt.** :class:`RobotsPolicy` honours ``Disallow`` by default. It fails open
  only when the file could not be fetched at all, and records that it did.
* **Budgets.** :class:`Budget` holds the request, page, byte and wall-clock caps and
  reports which one stopped the scan.
* **Active probes.** :data:`ALLOWED_PROBE_KINDS` is the complete allowlist, and
  :func:`require_allowed_probe` is the only door. :func:`assert_benign_marker` keeps the
  reflection marker to characters that cannot express a payload.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from vulnprio.core.errors import VulnprioError
from vulnprio.scan.models import ProbeKind, ScanRequest

__all__ = [
    "NotAuthorizedError",
    "OutOfScopeError",
    "ProbeNotAllowedError",
    "ALLOWED_PROBE_KINDS",
    "METADATA_HOSTS",
    "PRIVATE_HOST_SUFFIXES",
    "require_authorization",
    "require_target_allowed",
    "require_allowed_probe",
    "assert_benign_marker",
    "host_in_scope",
    "url_in_scope",
    "is_private_host",
    "normalise_target",
    "is_http_url",
    "RateLimiter",
    "RobotsPolicy",
    "Budget",
    "normalise_hosts",
]


class NotAuthorizedError(VulnprioError):
    """A scan was attempted without an explicit authorisation attestation."""


class OutOfScopeError(VulnprioError):
    """A URL outside the authorised scope was about to be requested."""


class ProbeNotAllowedError(VulnprioError):
    """A probe kind outside the benign allowlist was attempted."""


#: The complete set of active behaviours this scanner may perform. Anything not in this
#: frozenset is refused by :func:`require_allowed_probe`, which every probe calls before
#: it touches the network. Exploitation payloads, authentication bypass, brute forcing,
#: fuzzing and any request that writes, modifies or deletes data are absent by
#: construction and must stay that way.
ALLOWED_PROBE_KINDS: frozenset[ProbeKind] = frozenset(
    {
        ProbeKind.REFLECTED_MARKER,
        ProbeKind.OPTIONS_METHOD,
        ProbeKind.WELL_KNOWN_PATH,
    }
)

#: Cloud instance metadata endpoints. Reachable from inside a VPC, unauthenticated, and
#: full of credentials: the single most damaging thing a scanner can be pointed at.
METADATA_HOSTS: frozenset[str] = frozenset(
    {
        "169.254.169.254",          # AWS / Azure / GCP / DigitalOcean IMDS
        "fd00:ec2::254",            # AWS IMDS over IPv6
        "100.100.100.200",          # Alibaba Cloud
        "192.0.0.192",              # Oracle Cloud
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "metadata",
    }
)

#: Hostname suffixes that name a private or link-local namespace by convention.
PRIVATE_HOST_SUFFIXES: tuple[str, ...] = (
    ".local",
    ".localhost",
    ".localdomain",
    ".internal",
    ".intranet",
    ".lan",
    ".home",
    ".home.arpa",
    ".corp",
    ".private",
)

#: A marker is allowed to be letters and digits and nothing else. No angle brackets, no
#: quotes, no parentheses, no semicolons: it cannot express markup or a script.
_BENIGN_MARKER = re.compile(r"^[A-Za-z0-9]{4,64}$")


# ---------------------------------------------------------------------------
# Authorisation
# ---------------------------------------------------------------------------


def require_authorization(request: ScanRequest) -> None:
    """Raise :class:`NotAuthorizedError` unless the operator attested to authorisation.

    The identity comparison is deliberate: a truthy value that is not ``True`` (``1``, a
    non-empty string, an object) does not authorise a scan.
    """
    if request.authorized is not True:
        raise NotAuthorizedError(
            "refusing to scan "
            f"{request.target_url}: ScanRequest.authorized must be True and must record, in "
            "authorization_note, who permitted this assessment. This tool only tests systems "
            "its operator is authorised to test."
        )
    if not request.authorization_note.strip():
        raise NotAuthorizedError(
            "refusing to scan: authorization_note is empty, so there is no record of who "
            "permitted this assessment"
        )


def require_target_allowed(request: ScanRequest, *, resolve: bool = False) -> None:
    """Refuse a target that is private, loopback or a metadata endpoint.

    ``allow_private_targets`` is the only way past this, and it must be set deliberately
    on the request. ``resolve`` additionally resolves the hostname through DNS; it is off
    by default so that the check never depends on the network, and because a hostname that
    deliberately points at a private address is not the accident this guard exists to
    prevent.
    """
    require_authorization(request)
    host = request.target_host
    if not host:
        raise OutOfScopeError("target_url has no host")
    if is_private_host(host, resolve=resolve) and not request.allow_private_targets:
        raise OutOfScopeError(
            f"{host!r} is a private, loopback, link-local or cloud-metadata address. "
            "Set allow_private_targets=True on the ScanRequest if you really mean to "
            "assess an internal target you are authorised to test."
        )


def require_allowed_probe(kind: ProbeKind) -> ProbeKind:
    """The only door through which an active probe may be sent."""
    if kind not in ALLOWED_PROBE_KINDS:
        raise ProbeNotAllowedError(
            f"probe kind {kind!r} is not in the benign allowlist "
            f"{sorted(item.value for item in ALLOWED_PROBE_KINDS)}"
        )
    return kind


def assert_benign_marker(marker: str) -> str:
    """Refuse any reflection marker that is not purely alphanumeric.

    The reflected-marker probe exists to find a *reflection point*. Letting anything else
    into that parameter value would turn a detection probe into an exploitation attempt.
    """
    if not _BENIGN_MARKER.match(marker or ""):
        raise ProbeNotAllowedError(
            "a reflection marker must be 4-64 alphanumeric characters and nothing else; "
            f"refusing {marker!r}"
        )
    return marker


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def is_http_url(url: str) -> bool:
    """True for an absolute ``http``/``https`` URL with a host."""
    try:
        parts = urlsplit(str(url))
    except ValueError:
        return False
    return parts.scheme.lower() in {"http", "https"} and bool(parts.hostname)


def host_in_scope(url: str, request: ScanRequest) -> bool:
    """True when ``url`` is on the authorised host or an explicitly listed extra host.

    Exact hostname match only. A scan authorised for ``shop.example.com`` does not cover
    ``api.shop.example.com``: the operator has to say so.
    """
    if not is_http_url(url):
        return False
    host = (urlsplit(str(url)).hostname or "").lower()
    if host not in request.scope_hosts:
        return False
    if is_private_host(host) and not request.allow_private_targets:
        return False
    return True


#: Kept as an alias because "is this URL in scope" reads better at some call sites.
url_in_scope = host_in_scope


def is_private_host(host: str, *, resolve: bool = False) -> bool:
    """True for loopback, RFC1918, link-local, reserved or metadata hosts.

    Literal addresses are classified with :mod:`ipaddress`; hostnames are matched against
    the metadata names and the private suffix conventions. With ``resolve=True`` the name
    is also resolved and every returned address is classified, which costs a DNS lookup.
    """
    name = (host or "").strip().lower().rstrip(".")
    if not name:
        return True
    if name in METADATA_HOSTS:
        return True
    if name == "localhost" or any(name.endswith(suffix) for suffix in PRIVATE_HOST_SUFFIXES):
        return True

    literal = name
    if literal.startswith("[") and literal.endswith("]"):
        literal = literal[1:-1]
    try:
        return _address_is_private(ipaddress.ip_address(literal))
    except ValueError:
        pass

    if resolve:
        try:
            infos = socket.getaddrinfo(name, None)
        except (socket.gaierror, UnicodeError, OSError):
            return False
        for info in infos:
            candidate = info[4][0]
            try:
                if _address_is_private(ipaddress.ip_address(str(candidate).split("%")[0])):
                    return True
            except ValueError:
                continue
    return False


#: Ranges :mod:`ipaddress` does not classify as private on every supported Python, but
#: which are never a legitimate internet scan target.
_EXTRA_PRIVATE_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("100.64.0.0/10"),     # carrier-grade NAT (RFC 6598)
    ipaddress.ip_network("192.0.0.0/24"),      # IETF protocol assignments
    ipaddress.ip_network("198.18.0.0/15"),     # benchmarking (RFC 2544)
    ipaddress.ip_network("64:ff9b::/96"),      # NAT64
)


def _address_is_private(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if str(address) in METADATA_HOSTS:
        return True
    for network in _EXTRA_PRIVATE_NETWORKS:
        if address.version == network.version and address in network:
            return True
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
        or address.is_multicast
    )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """Token bucket over a monotonic clock.

    Politeness is a safety property: a scan that hammers a production application is a
    denial of service whatever its intent. The clock and sleep function are injectable so
    that spacing can be asserted in tests without waiting for real seconds to pass.
    """

    #: Tolerance for "a whole token has accrued". Guards against floating-point shortfall.
    _EPSILON = 1e-9
    #: Never sleep for less than this, so an injected or coarse clock always advances.
    _MIN_DELAY_S = 1e-4

    def __init__(
        self,
        rate_per_second: float,
        *,
        burst: float = 1.0,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self.rate = float(rate_per_second)
        self.capacity = max(1.0, float(burst))
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._tokens = self.capacity
        self._last = self._clock()
        self.total_wait_s = 0.0
        self.acquired = 0

    @classmethod
    def from_request(
        cls,
        request: ScanRequest,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> "RateLimiter":
        return cls(request.requests_per_second, clock=clock, sleep=sleep)

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available. Returns the seconds spent waiting.

        The epsilon and the minimum delay are not decoration: without them a bucket that
        lands a few ulps short of a whole token computes a delay so small that adding it
        to the clock is a no-op in floating point, and the loop never terminates.
        """
        waited = 0.0
        while True:
            self._refill()
            if self._tokens >= tokens - self._EPSILON:
                self._tokens = max(0.0, self._tokens - tokens)
                self.acquired += 1
                self.total_wait_s += waited
                return waited
            delay = max((tokens - self._tokens) / self.rate, self._MIN_DELAY_S)
            self._sleep(delay)
            waited += delay


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------


class RobotsPolicy:
    """``robots.txt``, read either as a rule or as reconnaissance.

    ``robots.txt`` is a crawler convention, not an access control. It keeps search engines
    out of a directory; it keeps nobody else out of anything, and an attacker reads it as a
    list of places the operator considers worth hiding. An authorised assessment that
    honours it therefore reports a directory as clean when it was never looked at - the
    single most misleading outcome this scanner can produce. So ``Disallow`` is **advisory
    by default**: the file is fetched and parsed either way, and
    :class:`~vulnprio.scan.models.ScanRequest` decides whether its rules bind.

    The other controls are the ones doing the work, and none of them is optional: an
    authorisation attestation with a written note, a scope locked to the named host and
    re-checked on every redirect, a rate limit, the dangerous-link refusal, and a probe
    allowlist that admits three benign behaviours and nothing else. Those constrain what
    the scanner may do. ``robots.txt`` only constrains where it looks.

    In either mode the paths named are recorded, because "the operator asked crawlers to
    stay out of ``/ftp``" is a finding-shaped fact whether or not this scan complied.
    """

    def __init__(
        self,
        *,
        parser: RobotFileParser | None = None,
        enabled: bool = True,
        source_url: str | None = None,
        fetch_error: str | None = None,
        advisory: bool = False,
    ) -> None:
        self._parser = parser
        self.enabled = bool(enabled)
        #: Parsed, but not binding: ``allows`` returns True for everything while the paths
        #: are still recorded. This is the default, and it is why the file is fetched at all
        #: when its rules do not apply.
        self.advisory = bool(advisory)
        self.source_url = source_url
        self.fetch_error = fetch_error
        self.blocked = 0
        #: Paths ``robots.txt`` names as off-limits, in the order they came up. Kept in both
        #: modes, because "``Disallow: /ftp``" is a disclosure either way - the operator is
        #: naming the directory they would rather nobody looked in. When the rules bind,
        #: this is the ground the scan did not cover; when they do not, it is the ground
        #: worth reading first.
        self.blocked_paths: list[str] = []

    @property
    def binding(self) -> bool:
        """Whether a ``Disallow`` entry actually prevents a request."""
        return self.enabled and not self.advisory and self._parser is not None

    @classmethod
    def disabled(cls) -> "RobotsPolicy":
        """A policy with no file behind it at all, which allows everything."""
        return cls(enabled=False)

    @classmethod
    def from_text(
        cls, text: str, *, source_url: str | None = None, advisory: bool = False
    ) -> "RobotsPolicy":
        parser = RobotFileParser()
        parser.parse((text or "").splitlines())
        return cls(parser=parser, source_url=source_url, advisory=advisory)

    @classmethod
    def fetch_failed(cls, error: str, *, source_url: str | None = None) -> "RobotsPolicy":
        """robots.txt was unreachable: allow, but say so."""
        return cls(parser=None, source_url=source_url, fetch_error=str(error))

    def allows(self, url: str, user_agent: str = "*") -> bool:
        """True when ``user_agent`` may fetch ``url``.

        In advisory mode the answer is always True and the path is still recorded, so the
        scan can report what the file named without treating it as a boundary.
        """
        if not self.enabled or self._parser is None:
            return True
        try:
            permitted = self._parser.can_fetch(user_agent, str(url))
        except Exception:  # pragma: no cover - robotparser is lenient, but never fail closed on a bug
            return True
        if not permitted:
            self.blocked += 1
            path = urlsplit(str(url)).path or "/"
            if path not in self.blocked_paths and len(self.blocked_paths) < 50:
                self.blocked_paths.append(path)
        return True if self.advisory else bool(permitted)

    def crawl_delay(self, user_agent: str = "*") -> float | None:
        """Crawl-delay in seconds, when the file states one.

        Honoured in both modes. Unlike ``Disallow``, a crawl delay is a statement about what
        the server can take, and ignoring it is how an assessment becomes an outage.
        """
        if not self.enabled or self._parser is None:
            return None
        try:
            delay = self._parser.crawl_delay(user_agent)
        except Exception:  # pragma: no cover
            return None
        return float(delay) if delay is not None else None


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    """Requests, pages, bytes and wall-clock, with the reason the scan stopped.

    One object holds every quantitative cap so that "why did the scan stop" has exactly
    one answer, and so that a caller cannot accidentally enforce three of the four.
    """

    max_requests: int = 240
    max_pages: int = 60
    max_bytes: int = 60_000_000
    time_budget_s: float = 60.0
    clock: Callable[[], float] = time.monotonic
    started_at: float | None = field(default=None, init=False)
    requests: int = field(default=0, init=False)
    pages: int = field(default=0, init=False)
    bytes_downloaded: int = field(default=0, init=False)
    _reason: str | None = field(default=None, init=False)

    @classmethod
    def from_request(
        cls, request: ScanRequest, *, clock: Callable[[], float] | None = None
    ) -> "Budget":
        return cls(
            max_requests=request.max_requests,
            max_pages=request.max_pages,
            max_bytes=request.total_byte_budget,
            time_budget_s=request.time_budget_s,
            clock=clock or time.monotonic,
        )

    def start(self) -> "Budget":
        if self.started_at is None:
            self.started_at = self.clock()
        return self

    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, self.clock() - self.started_at)

    def remaining_s(self) -> float:
        return max(0.0, self.time_budget_s - self.elapsed_s())

    def note_request(self, count: int = 1) -> None:
        self.start()
        self.requests += int(count)

    def note_page(self, count: int = 1) -> None:
        self.pages += int(count)

    def note_bytes(self, count: int) -> None:
        self.bytes_downloaded += max(0, int(count))

    def exhausted(self) -> bool:
        return self.reason() is not None

    def reason(self) -> str | None:
        """The first cap that is spent, or ``None`` while the scan may continue."""
        if self._reason is not None:
            return self._reason
        if self.requests >= self.max_requests:
            return f"request budget spent ({self.requests}/{self.max_requests} requests)"
        if self.pages >= self.max_pages:
            return f"page budget spent ({self.pages}/{self.max_pages} pages)"
        if self.bytes_downloaded >= self.max_bytes:
            return f"byte budget spent ({self.bytes_downloaded}/{self.max_bytes} bytes)"
        if self.started_at is not None and self.elapsed_s() >= self.time_budget_s:
            return f"time budget spent ({self.elapsed_s():.1f}s/{self.time_budget_s:.1f}s)"
        return None

    def stop(self, reason: str) -> None:
        """Latch a stop reason that is not one of the counted caps (an abort, an error)."""
        if self._reason is None:
            self._reason = str(reason)

    def can_request(self) -> bool:
        return not self.exhausted()

    def snapshot(self) -> dict[str, float]:
        return {
            "requests": float(self.requests),
            "pages": float(self.pages),
            "bytes": float(self.bytes_downloaded),
            "elapsed_s": self.elapsed_s(),
        }


def normalise_hosts(hosts: Iterable[str]) -> frozenset[str]:
    """Lower-cased, whitespace-trimmed host set, ignoring blanks."""
    return frozenset({str(host).strip().lower() for host in hosts if str(host).strip()})

#: A hostname or dotted IP: letters, digits, hyphens and dots, no leading or trailing dot.
_HOSTNAME = re.compile(r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)(?:\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*")


def normalise_target(target: str) -> str:
    """Turn what a person types into a URL, but only when it is plainly a host.

    ``localhost:3000`` is the obvious thing to write for a container, and it is also a valid
    URL whose scheme is ``localhost``, so untouched it reaches the request model as a scheme
    error rather than as the host and port it means.

    The rule is deliberately narrow: a scheme is added only when the text parses as
    ``host[:port][/path]`` with a hostname-shaped first segment and, if a colon is present, a
    numeric port. That distinction is load-bearing. A looser "prepend https:// to anything
    without //" turns ``javascript:alert(1)`` and ``data:text/html,x`` into strings that pass
    an ``http(s)`` scheme check, which would take a hostile value straight past the gate that
    exists to stop it. Anything that is not clearly a host is returned unchanged so the
    caller's scheme validation rejects it.

    The scheme follows the host rather than defaulting: a loopback or private address is
    almost always a local service on plain HTTP, and forcing HTTPS there fails the handshake
    instead of scanning. Callers should echo the result so the choice is never silent.
    """
    text = str(target).strip()
    if not text or "//" in text.split("?", 1)[0]:
        return text

    authority = text.split("/", 1)[0].split("?", 1)[0]
    host, _, port = authority.rpartition(":") if ":" in authority else (authority, "", "")
    if ":" in authority and not port.isdigit():
        return text                       # "javascript:alert(1)", "data:text/html,x"
    hostname = host or authority
    if not _HOSTNAME.fullmatch(hostname):
        return text                       # "../etc", "not a host", ""
    return f"{'http' if is_private_host(hostname) else 'https'}://{text}"
