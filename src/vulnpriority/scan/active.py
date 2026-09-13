"""The active profile: three benign probes, and nothing else, ever.

The active profile adds exactly three behaviours to the passive one, listed in
:data:`vulnpriority.scan.safety.ALLOWED_PROBE_KINDS` and enforced by
:func:`~vulnpriority.scan.safety.require_allowed_probe`, which every probe in this module
calls before it touches the network:

1. **Reflected marker.** A random alphanumeric string is sent as the value of one existing
   parameter, and the response is searched for it. This locates a point where input
   reaches the output. It is *not* an XSS payload: the marker is validated by
   :func:`~vulnpriority.scan.safety.assert_benign_marker` to contain only letters and digits,
   so it cannot express a tag, an attribute, a quote or a script.
2. **OPTIONS.** One ``OPTIONS`` request per sampled path, read for its ``Allow`` header.
3. **Well-known path.** One ``GET`` for ``/.well-known/security.txt``, a path that exists
   by public convention (RFC 9116) precisely so that security tooling can read it.

This module MUST NOT ever:

* send SQL, command, XSS, XXE, SSTI, deserialisation, traversal or any other
  exploitation payload;
* attempt authentication bypass, credential stuffing, or any brute force;
* fuzz, flood, or otherwise probe for denial of service;
* write, modify or delete data - it never sends ``POST``, ``PUT``, ``PATCH`` or ``DELETE``,
  never submits a form, and never sends a request body;
* guess paths or enumerate files.

These are not conventions. :data:`vulnpriority.scan.safety.ALLOWED_PROBE_KINDS` is a closed
allowlist, ``tests/test_scan_checks.py`` asserts that its members are exactly the three
above, and ``tests/test_scan_crawler.py`` asserts against a recording of every request an
active scan issued that no other verb and no payload-shaped value ever left the process.
"""

from __future__ import annotations

import secrets
from typing import Callable, Iterable, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from vulnpriority.core.enums import HttpMethod
from vulnpriority.ingest.normalize import canonical_url
from vulnpriority.scan.checks import CHECKS, Check, CheckContext, run_checks
from vulnpriority.scan.http import HttpClient
from vulnpriority.scan.models import (
    CheckFinding,
    Page,
    ProbeKind,
    ProbeResult,
    ScanProfile,
    ScanRequest,
)
from vulnpriority.scan.safety import (
    OutOfScopeError,
    assert_benign_marker,
    host_in_scope,
    require_allowed_probe,
)

__all__ = [
    "ACTIVE_CHECKS",
    "MARKER_PREFIX",
    "WELL_KNOWN_PATH",
    "DEFAULT_MAX_REFLECTION_PROBES",
    "DEFAULT_MAX_OPTIONS_PROBES",
    "active_check_ids",
    "make_marker",
    "probe_reflected_marker",
    "probe_options",
    "probe_well_known_path",
    "run_active_probes",
    "run_active_checks",
]

#: Markers start with a fixed prefix so a target operator grepping their logs can see what
#: put the value there.
MARKER_PREFIX = "vulnpriority"

#: RFC 9116. The one path this scanner requests without the application linking to it.
WELL_KNOWN_PATH = "/.well-known/security.txt"

DEFAULT_MAX_REFLECTION_PROBES = 8
DEFAULT_MAX_OPTIONS_PROBES = 5


def _active() -> tuple[Check, ...]:
    return tuple(check for check in CHECKS.values() if check.profile == ScanProfile.ACTIVE)


#: Checks that only run in the active profile (they consume probe results).
ACTIVE_CHECKS: tuple[Check, ...] = _active()


def active_check_ids() -> tuple[str, ...]:
    return tuple(check.id for check in _active())


def make_marker(rng: Callable[[int], str] | None = None) -> str:
    """A random, harmless, alphanumeric marker.

    Validated before it is returned: if a future change ever let a non-alphanumeric
    character in, construction fails here rather than a payload leaving the process.
    """
    token = (rng or secrets.token_hex)(6)
    marker = f"{MARKER_PREFIX}{''.join(ch for ch in token if ch.isalnum())}"
    return assert_benign_marker(marker)


def _with_param(url: str, name: str, value: str) -> str:
    """Replace one query parameter's value, leaving the rest of the URL untouched."""
    parts = urlsplit(url)
    pairs = [(key, value if key == name else item) for key, item in parse_qsl(parts.query, keep_blank_values=True)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(pairs), ""))


def probe_reflected_marker(
    request: ScanRequest,
    client: HttpClient,
    page: Page,
    *,
    marker: str | None = None,
) -> ProbeResult | None:
    """Send a benign marker in one existing parameter and report whether it comes back.

    Only parameters the application already exposes are used, and only their value is
    changed. No parameter is invented, no value contains anything but letters and digits,
    and the request is a plain ``GET``.
    """
    require_allowed_probe(ProbeKind.REFLECTED_MARKER)
    url = page.effective_url
    pairs = parse_qsl(urlsplit(url).query, keep_blank_values=True)
    if not pairs:
        return None
    name = pairs[0][0]
    token = assert_benign_marker(marker or make_marker())
    probe_url = _with_param(url, name, token)
    if not host_in_scope(probe_url, request):  # pragma: no cover - same origin by construction
        return None
    try:
        result = client.fetch(probe_url, HttpMethod.GET)
    except OutOfScopeError as error:  # pragma: no cover
        return ProbeResult(
            kind=ProbeKind.REFLECTED_MARKER, url=url, param=name, marker=token, error=str(error)
        )
    if result.error is not None and result.status is None:
        return ProbeResult(
            kind=ProbeKind.REFLECTED_MARKER,
            url=url,
            param=name,
            marker=token,
            error=result.error,
        )
    body = result.body_text or ""
    index = body.find(token)
    reflected = index >= 0
    evidence = body[max(0, index - 80) : index + len(token) + 80] if reflected else ""
    return ProbeResult(
        kind=ProbeKind.REFLECTED_MARKER,
        url=url,
        method=HttpMethod.GET,
        param=name,
        marker=token,
        status=result.status,
        reflected=reflected,
        evidence=evidence,
    )


def probe_options(request: ScanRequest, client: HttpClient, page: Page) -> ProbeResult | None:
    """One ``OPTIONS`` request, read for the ``Allow`` header. No body is sent."""
    require_allowed_probe(ProbeKind.OPTIONS_METHOD)
    url = page.effective_url
    try:
        result = client.fetch(url, HttpMethod.OPTIONS, follow_redirects=False)
    except OutOfScopeError as error:  # pragma: no cover
        return ProbeResult(kind=ProbeKind.OPTIONS_METHOD, url=url, method=HttpMethod.OPTIONS, error=str(error))
    if result.error is not None and result.status is None:
        return ProbeResult(
            kind=ProbeKind.OPTIONS_METHOD, url=url, method=HttpMethod.OPTIONS, error=result.error
        )
    allow = result.headers.get("allow") or result.headers.get("access-control-allow-methods") or ""
    methods = tuple(
        sorted({token.strip().upper() for token in allow.split(",") if token.strip()})
    )
    return ProbeResult(
        kind=ProbeKind.OPTIONS_METHOD,
        url=url,
        method=HttpMethod.OPTIONS,
        status=result.status,
        allow_methods=methods,
        headers={"allow": allow} if allow else {},
        evidence=f"Allow: {allow}" if allow else "",
    )


def probe_well_known_path(request: ScanRequest, client: HttpClient) -> ProbeResult | None:
    """Request ``/.well-known/security.txt`` once (RFC 9116)."""
    require_allowed_probe(ProbeKind.WELL_KNOWN_PATH)
    url = canonical_url(f"{request.target_origin}{WELL_KNOWN_PATH}")
    try:
        result = client.fetch(url, HttpMethod.GET)
    except OutOfScopeError as error:  # pragma: no cover
        return ProbeResult(kind=ProbeKind.WELL_KNOWN_PATH, url=url, error=str(error))
    if result.error is not None and result.status is None:
        return ProbeResult(kind=ProbeKind.WELL_KNOWN_PATH, url=url, error=result.error)
    return ProbeResult(
        kind=ProbeKind.WELL_KNOWN_PATH,
        url=url,
        status=result.status,
        evidence=(result.body_text or "")[:300] if result.status == 200 else "",
    )


def run_active_probes(
    request: ScanRequest,
    client: HttpClient,
    pages: Sequence[Page],
    *,
    max_reflection_probes: int = DEFAULT_MAX_REFLECTION_PROBES,
    max_options_probes: int = DEFAULT_MAX_OPTIONS_PROBES,
    marker_factory: Callable[[], str] | None = None,
    on_probe: Callable[[ProbeResult], None] | None = None,
) -> tuple[ProbeResult, ...]:
    """Run the allowlisted probes over a completed crawl, under the shared budget.

    Probing is capped independently of the crawl so that an application with hundreds of
    parameterised pages does not turn an active scan into a flood.
    """
    if request.profile != ScanProfile.ACTIVE:
        return ()
    results: list[ProbeResult] = []

    def _record(probe: ProbeResult | None) -> None:
        if probe is None:
            return
        results.append(probe)
        if on_probe is not None:
            on_probe(probe)

    if not client.budget.exhausted():
        _record(probe_well_known_path(request, client))

    reflected = 0
    for page in pages:
        if client.budget.exhausted() or reflected >= max_reflection_probes:
            break
        if not page.is_html and page.content_type not in (None, "application/json"):
            continue
        probe = probe_reflected_marker(
            request, client, page, marker=(marker_factory() if marker_factory else None)
        )
        if probe is not None:
            reflected += 1
            _record(probe)

    options = 0
    seen_paths: set[str] = set()
    for page in pages:
        if client.budget.exhausted() or options >= max_options_probes:
            break
        path = urlsplit(page.effective_url).path
        if path in seen_paths or not page.is_html:
            continue
        seen_paths.add(path)
        options += 1
        _record(probe_options(request, client, page))

    return tuple(results)


def run_active_checks(
    pages: Iterable[Page],
    context: CheckContext,
    *,
    checks: Sequence[Check] | None = None,
) -> list[CheckFinding]:
    """Run the full active check set (passive checks included) over fetched pages."""
    return run_checks(pages, context, ScanProfile.ACTIVE, checks=checks)
