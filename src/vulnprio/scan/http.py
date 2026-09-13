"""The scanner's only door to the network.

Every request this package makes goes through :class:`HttpClient`, which is where the
limits in :mod:`vulnprio.scan.safety` become behaviour:

* the rate limiter is acquired before the socket is touched, so politeness cannot be
  skipped by a caller that forgot;
* the response body is streamed and abandoned once the per-response byte cap is reached,
  so a hostile or merely enormous resource cannot exhaust memory;
* redirects are followed manually, one hop at a time, and **scope is re-checked on every
  hop** - a 302 to another host stops the chain and is counted rather than followed;
* the byte, request and wall-clock budgets are consulted before each request.

``Accept-Encoding: identity`` is sent deliberately: with no transport compression the byte
cap applies to the bytes that actually become text, so a compression bomb cannot expand
past the cap after the check.

The transport is injectable (``httpx.MockTransport``), which is how the whole test suite
runs a realistic fake site with no network at all.
"""

from __future__ import annotations

import time
from typing import Callable, Mapping
from urllib.parse import urljoin, urlsplit

import httpx

from vulnprio.core.enums import HttpMethod
from vulnprio.core.models import Frozen
from vulnprio.ingest.normalize import canonical_url
from vulnprio.scan.models import Page, ScanRequest
from vulnprio.scan.safety import (
    Budget,
    OutOfScopeError,
    RateLimiter,
    RobotsPolicy,
    host_in_scope,
    is_http_url,
)

__all__ = ["FetchResult", "HttpClient", "fetch_robots_policy", "TEXTUAL_CONTENT_TYPES"]

#: Content types whose body is decoded to text. Anything else is measured but not read as
#: text: there is nothing for a markup or header check to say about a JPEG.
TEXTUAL_CONTENT_TYPES: tuple[str, ...] = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml",
    "application/javascript",
    "application/ecmascript",
    "application/x-javascript",
    "+json",
    "+xml",
)


class FetchResult(Frozen):
    """One HTTP exchange, including the ones that were refused before being sent."""

    url: str
    final_url: str = ""
    method: HttpMethod = HttpMethod.GET
    status: int | None = None
    headers: dict[str, str] = {}
    set_cookies: tuple[str, ...] = ()
    body_text: str = ""
    body_bytes_len: int = 0
    elapsed_ms: float = 0.0
    truncated: bool = False
    redirects: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status is not None

    @property
    def content_type(self) -> str | None:
        raw = self.headers.get("content-type")
        return raw.split(";")[0].strip().lower() if raw else None

    def to_page(self, depth: int = 0) -> Page:
        """The crawler's view of this exchange (links and forms are filled in later)."""
        return Page(
            url=self.url,
            final_url=self.final_url or self.url,
            method=self.method,
            status=self.status,
            headers=dict(self.headers),
            set_cookies=self.set_cookies,
            body=self.body_text,
            body_bytes_len=self.body_bytes_len,
            content_type=self.content_type,
            elapsed_ms=self.elapsed_ms,
            depth=depth,
            truncated=self.truncated,
            error=self.error,
            redirects=self.redirects,
        )


def _is_textual(content_type: str | None) -> bool:
    lowered = (content_type or "").lower()
    return any(marker in lowered for marker in TEXTUAL_CONTENT_TYPES)


def _charset_of(content_type: str | None) -> str:
    for part in (content_type or "").split(";")[1:]:
        name, _, value = part.strip().partition("=")
        if name.strip().lower() == "charset" and value:
            return value.strip().strip('"') or "utf-8"
    return "utf-8"


def _split_set_cookie(response: httpx.Response) -> tuple[str, ...]:
    """Every ``Set-Cookie`` value, unmerged - flags are per-cookie, so merging loses them."""
    try:
        return tuple(value for key, value in response.headers.multi_items() if key.lower() == "set-cookie")
    except AttributeError:  # pragma: no cover - defensive, httpx always has multi_items
        raw = response.headers.get("set-cookie")
        return (raw,) if raw else ()


class HttpClient:
    """Rate-limited, scope-locked, budget-bounded HTTP.

    The client keeps a log of every request it actually issued (:attr:`requests`) and of
    everything it refused (:attr:`out_of_scope_blocked`). The tests assert against both:
    "the scanner never requested X" is only credible if the record of what it did request
    is complete.
    """

    def __init__(
        self,
        request: ScanRequest,
        *,
        budget: Budget | None = None,
        limiter: RateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.request = request
        self.clock = clock or time.monotonic
        self.budget = (budget or Budget.from_request(request, clock=self.clock)).start()
        self.limiter = limiter or RateLimiter.from_request(request, clock=self.clock, sleep=sleep)
        self.requests: list[tuple[str, str]] = []
        self.out_of_scope_blocked = 0
        self.errors: list[str] = []
        self._owns_client = client is None
        self._client = client or httpx.Client(
            transport=transport,
            follow_redirects=False,          # every hop is re-checked against scope by hand
            timeout=request.timeout_s,
            verify=request.verify_tls,
            headers=self.default_headers(),
        )

    # -- setup ---------------------------------------------------------------

    def default_headers(self) -> dict[str, str]:
        """Honest User-Agent, no credentials, no transport compression.

        The User-Agent is applied *after* the caller's headers, so the identifying string
        validated on :class:`~vulnprio.scan.models.ScanRequest` cannot be overridden by
        smuggling a ``User-Agent`` entry into ``headers``.
        """
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.5",
            "Accept-Encoding": "identity",
        }
        headers.update(self.request.headers)
        for name in list(headers):
            if name.lower() == "user-agent":
                headers.pop(name)
        headers["User-Agent"] = self.request.user_agent
        return headers

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- the one request primitive -------------------------------------------

    def fetch(
        self,
        url: str,
        method: HttpMethod | str = HttpMethod.GET,
        *,
        follow_redirects: bool = True,
        extra_headers: Mapping[str, str] | None = None,
    ) -> FetchResult:
        """Issue one request, honouring scope, rate, redirects and every budget.

        Raises :class:`OutOfScopeError` when asked directly for a URL outside scope: that
        is a programming error in the caller, not a routine skip. Redirects *to* an
        out-of-scope URL are not an error - the chain simply stops and is counted.
        """
        verb = method if isinstance(method, HttpMethod) else HttpMethod(str(method).upper())
        target = str(url)
        if not is_http_url(target):
            raise OutOfScopeError(f"refusing a non-http(s) URL: {target!r}")
        if not host_in_scope(target, self.request):
            self.out_of_scope_blocked += 1
            raise OutOfScopeError(
                f"{target} is outside the authorised scope {sorted(self.request.scope_hosts)}"
            )

        redirects: list[str] = []
        current = target
        started = self.clock()
        hops = 0
        while True:
            reason = self.budget.reason()
            if reason is not None:
                return FetchResult(url=target, method=verb, error=f"budget exhausted: {reason}")

            result = self._send(current, verb, extra_headers, started, redirects)
            if result.error is not None or result.status is None:
                return result
            location = result.headers.get("location")
            if not (follow_redirects and 300 <= result.status < 400 and location):
                return result
            if hops >= self.request.max_redirects:
                return result.model_copy(update={"error": "too many redirects"})

            nxt = canonical_url(urljoin(current, location))
            if not is_http_url(nxt) or not host_in_scope(nxt, self.request):
                # Never follow a redirect off the authorised host. Count it and stop.
                self.out_of_scope_blocked += 1
                return result.model_copy(
                    update={
                        "error": f"redirect to out-of-scope host stopped: {nxt}",
                        "redirects": tuple(redirects),
                    }
                )
            redirects.append(nxt)
            current = nxt
            hops += 1

    def _send(
        self,
        url: str,
        method: HttpMethod,
        extra_headers: Mapping[str, str] | None,
        started: float,
        redirects: list[str],
    ) -> FetchResult:
        """One actual wire request: rate-limited, byte-capped, never raising to the caller."""
        self.limiter.acquire()
        self.budget.note_request()
        self.requests.append((method.value, url))
        headers = dict(extra_headers or {})
        cap = self.request.max_response_bytes
        try:
            with self._client.stream(method.value, url, headers=headers or None) as response:
                content_type = response.headers.get("content-type")
                textual = _is_textual(content_type)
                buffer = bytearray()
                truncated = False
                for chunk in response.iter_bytes():
                    if len(buffer) >= cap:
                        truncated = True
                        break
                    buffer.extend(chunk)
                if len(buffer) > cap:
                    truncated = True
                    del buffer[cap:]
                raw = bytes(buffer)
                self.budget.note_bytes(len(raw))
                body = raw.decode(_charset_of(content_type), errors="replace") if textual else ""
                return FetchResult(
                    url=url,
                    final_url=str(response.url),
                    method=method,
                    status=response.status_code,
                    headers={key.lower(): value for key, value in response.headers.items()},
                    set_cookies=_split_set_cookie(response),
                    body_text=body,
                    body_bytes_len=len(raw),
                    elapsed_ms=max(0.0, (self.clock() - started) * 1000.0),
                    truncated=truncated,
                    redirects=tuple(redirects),
                )
        except httpx.HTTPError as error:
            message = f"{type(error).__name__}: {error}"
            self.errors.append(f"{url}: {message}")
            return FetchResult(
                url=url,
                method=method,
                elapsed_ms=max(0.0, (self.clock() - started) * 1000.0),
                error=message,
                redirects=tuple(redirects),
            )

    # -- convenience ----------------------------------------------------------

    def get(self, url: str, **kwargs: object) -> FetchResult:
        return self.fetch(url, HttpMethod.GET, **kwargs)  # type: ignore[arg-type]

    def options(self, url: str, **kwargs: object) -> FetchResult:
        return self.fetch(url, HttpMethod.OPTIONS, **kwargs)  # type: ignore[arg-type]


def fetch_robots_policy(client: HttpClient, request: ScanRequest) -> RobotsPolicy:
    """Fetch and parse ``/robots.txt`` for the target origin.

    The file is read whether or not its rules bind. When
    :attr:`~vulnprio.scan.models.ScanRequest.respect_robots` is off - the default - the
    policy comes back *advisory*: nothing is forbidden, and the paths it names are still
    recorded, because a ``Disallow`` list is the operator's own inventory of what they
    would rather nobody found. Discarding the file would throw that away to save one
    request.

    ``robots.txt`` is itself always fetchable (a robots file cannot forbid reading itself).
    A missing or unreachable file allows everything, which is the documented convention,
    but the failure is recorded on the policy so it can be surfaced to the operator.
    """
    parts = urlsplit(request.target_url)
    url = f"{parts.scheme.lower()}://{parts.netloc.lower()}/robots.txt"
    try:
        result = client.fetch(url, HttpMethod.GET)
    except OutOfScopeError as error:  # pragma: no cover - target is in scope by construction
        return RobotsPolicy.fetch_failed(str(error), source_url=url)
    if result.error is not None:
        return RobotsPolicy.fetch_failed(result.error, source_url=url)
    if result.status is None or result.status >= 400 or not result.body_text.strip():
        return RobotsPolicy.fetch_failed(f"no usable robots.txt (status {result.status})", source_url=url)
    return RobotsPolicy.from_text(
        result.body_text, source_url=url, advisory=not request.respect_robots
    )
