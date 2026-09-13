"""Safety invariants of the built-in scanner.

These are the tests that matter most in this package. Everything else decides how good the
findings are; these decide whether the tool can be misused or can cause harm by accident.
They are all offline: an ``httpx.MockTransport`` records every request, so "the scanner
never requested X" is asserted against a complete log rather than assumed.
"""

from __future__ import annotations

import httpx
import pytest

from vulnpriority.core.enums import HttpMethod
from vulnpriority.scan import (
    ALLOWED_PROBE_KINDS,
    Budget,
    HttpClient,
    NotAuthorizedError,
    OutOfScopeError,
    ProbeKind,
    ProbeNotAllowedError,
    RateLimiter,
    RobotsPolicy,
    ScanProfile,
    ScanRequest,
    assess_target,
    host_in_scope,
    is_private_host,
    require_allowed_probe,
    require_authorization,
    require_target_allowed,
    run_scan,
)
from vulnpriority.scan.active import make_marker, probe_options, probe_reflected_marker
from vulnpriority.scan.http import fetch_robots_policy
from vulnpriority.scan.safety import assert_benign_marker

TARGET = "https://shop.example.com/"


class FakeClock:
    """Monotonic clock that only advances when something sleeps (or the test says so)."""

    def __init__(self, start: float = 0.0, step: float = 0.0) -> None:
        self.now = float(start)
        self.step = float(step)
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.now += float(seconds)

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def recording_transport(handler) -> tuple[httpx.MockTransport, list[str]]:
    """A mock transport plus the list of every URL it was ever asked for."""
    log: list[str] = []

    def _record(request: httpx.Request) -> httpx.Response:
        log.append(str(request.url))
        return handler(request)

    return httpx.MockTransport(_record), log


def make_request(**overrides) -> ScanRequest:
    base = {
        "target_url": TARGET,
        "authorized": True,
        "authorization_note": "Engagement PT-2024-114, authorised by the application owner",
        "requests_per_second": 20.0,
    }
    base.update(overrides)
    return ScanRequest(**base)


def make_client(handler, request: ScanRequest | None = None, clock: FakeClock | None = None):
    request = request or make_request()
    clock = clock or FakeClock()
    transport, log = recording_transport(handler)
    client = HttpClient(request, transport=transport, clock=clock, sleep=clock.sleep)
    return request, client, log, clock


# ---------------------------------------------------------------------------
# Authorisation: the single most important behaviour in this package
# ---------------------------------------------------------------------------


def test_scan_without_authorisation_is_refused_and_sends_nothing():
    """A ScanRequest defaults to unauthorised and no scan may proceed from it."""
    request = ScanRequest(target_url=TARGET)
    assert request.authorized is False

    transport, log = recording_transport(lambda _: httpx.Response(200, html="<html></html>"))
    client = HttpClient(request, transport=transport)

    with pytest.raises(NotAuthorizedError):
        run_scan(request, client=client)
    # ``tools=()`` keeps this hermetic: an unauthorised request must be refused before
    # anything looks at PATH, let alone launches a scanner.
    with pytest.raises(NotAuthorizedError):
        assess_target(request, client=client, tools=())
    with pytest.raises(NotAuthorizedError):
        require_authorization(request)

    assert log == [], "an unauthorised scan must not touch the network at all"


def test_truthy_is_not_authorisation():
    """Only the value ``True`` authorises. A truthy stand-in does not."""
    request = ScanRequest(target_url=TARGET).model_copy(
        update={"authorized": 1, "authorization_note": "looks authorised"}
    )
    with pytest.raises(NotAuthorizedError):
        require_authorization(request)


def test_authorisation_requires_a_written_note():
    """An attestation with nobody's name on it is not an attestation."""
    with pytest.raises(ValueError, match="authorization_note"):
        ScanRequest(target_url=TARGET, authorized=True)
    with pytest.raises(ValueError):
        ScanRequest(target_url=TARGET, authorized=True, authorization_note="   ")

    bypass = ScanRequest(target_url=TARGET).model_copy(
        update={"authorized": True, "authorization_note": ""}
    )
    with pytest.raises(NotAuthorizedError, match="authorization_note"):
        require_authorization(bypass)


def test_authorised_request_is_accepted():
    require_authorization(make_request())
    require_target_allowed(make_request())


def test_credential_headers_are_refused_at_construction():
    """The scanner never authenticates to a target, so it cannot be handed a session."""
    for header in ("Cookie", "authorization", "X-Api-Key"):
        with pytest.raises(ValueError, match="credential header"):
            make_request(headers={header: "secret"})


def test_user_agent_must_identify_the_tool():
    with pytest.raises(ValueError, match="vulnpriority"):
        make_request(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    assert "vulnpriority" in make_request().user_agent.lower()


def test_user_agent_cannot_be_overridden_through_raw_headers():
    """Smuggling a User-Agent into ``headers`` must not disguise the scanner."""
    sent: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        sent.append(http_request.headers.get("user-agent", ""))
        return httpx.Response(200, html="<html>ok</html>")

    request = make_request(headers={"User-Agent": "Mozilla/5.0 (definitely a browser)"})
    _, client, _, _ = make_client(handler, request)
    client.fetch(TARGET)

    assert sent == [request.user_agent]
    assert "vulnpriority" in sent[0].lower()


def test_bounds_are_unrepresentable_rather_than_merely_discouraged():
    with pytest.raises(ValueError):
        make_request(requests_per_second=1000.0)
    with pytest.raises(ValueError):
        make_request(max_pages=0)
    with pytest.raises(ValueError):
        make_request(time_budget_s=10_000.0)
    with pytest.raises(ValueError):
        make_request(target_url="ftp://shop.example.com/")


def test_defaults_are_polite():
    """The documented default posture: passive, slow, shallow, bounded, robots-respecting.

    The page budget is what the scan is *allowed* to cover, not how hard it leans on the
    target; the request rate is the property that makes a scan a nuisance, and it is the
    one held at two per second here.
    """
    request = ScanRequest(target_url=TARGET)
    assert request.profile == ScanProfile.PASSIVE
    assert request.requests_per_second == 2.0
    assert request.max_pages == 200
    assert request.max_depth == 3
    assert request.time_budget_s == 150.0
    assert request.respect_robots is False
    assert request.allow_private_targets is False


def test_loopback_gets_a_faster_default_rate_but_only_by_default():
    """Courtesy is owed to someone else's server, not to the operator's own machine.

    Two requests a second against localhost buys nothing and costs coverage: the scan runs
    out of time budget before it reaches the end of its own queue. An explicit rate is
    still honoured exactly, including an explicitly slow one against loopback and the
    ordinary default against a public host.
    """
    assert ScanRequest(target_url="http://localhost:3000/").requests_per_second == 10.0
    assert ScanRequest(target_url="http://127.0.0.1:8080/").requests_per_second == 10.0
    assert ScanRequest(target_url=TARGET).requests_per_second == 2.0
    explicit = ScanRequest(target_url="http://localhost:3000/", requests_per_second=0.5)
    assert explicit.requests_per_second == 0.5


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_scope_is_exact_hostname_matching():
    request = make_request(extra_hosts=("api.shop.example.com",))
    assert host_in_scope("https://shop.example.com/a", request)
    assert host_in_scope("http://api.shop.example.com/b", request)
    # A subdomain that was not listed is out of scope.
    assert not host_in_scope("https://cdn.shop.example.com/c", request)
    # The classic suffix-matching bypasses.
    assert not host_in_scope("https://shop.example.com.attacker.net/d", request)
    assert not host_in_scope("https://evil-shop.example.com/e", request)
    # Non-http schemes are never in scope.
    assert not host_in_scope("javascript:alert(1)", request)
    assert not host_in_scope("file:///etc/passwd", request)


def test_out_of_scope_fetch_raises_and_issues_no_request():
    request, client, log, _ = make_client(lambda _: httpx.Response(200, html="<html></html>"))
    with pytest.raises(OutOfScopeError):
        client.fetch("https://evil.example.net/steal")
    assert log == []
    assert client.out_of_scope_blocked == 1


def test_redirect_off_scope_is_never_followed():
    """A 302 to another host stops the chain; the off-scope URL is never requested."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/go":
            return httpx.Response(302, headers={"Location": "https://evil.example.net/landing"})
        return httpx.Response(200, html="<html>ok</html>")

    request, client, log, _ = make_client(handler)
    result = client.fetch("https://shop.example.com/go")

    assert result.status == 302
    assert "out-of-scope" in (result.error or "")
    assert all("evil.example.net" not in url for url in log)
    assert client.out_of_scope_blocked == 1


def test_in_scope_redirect_is_followed_and_recorded():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"Location": "/new"})
        return httpx.Response(200, html="<html>new</html>")

    request, client, log, _ = make_client(handler)
    result = client.fetch("https://shop.example.com/old")
    assert result.status == 200
    assert result.redirects == ("https://shop.example.com/new",)
    assert log == ["https://shop.example.com/old", "https://shop.example.com/new"]


# ---------------------------------------------------------------------------
# Private and metadata targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "localhost",
        "10.0.0.5",
        "172.16.3.4",
        "192.168.1.1",
        "169.254.169.254",     # cloud instance metadata
        "metadata.google.internal",
        "100.100.100.200",     # Alibaba metadata
        "192.0.0.192",         # Oracle metadata
        "100.64.0.1",          # carrier-grade NAT
        "::1",
        "fd00::1",
        "printer.local",
        "db.internal",
        "0.0.0.0",
    ],
)
def test_private_and_metadata_hosts_are_recognised(host: str):
    assert is_private_host(host) is True


@pytest.mark.parametrize("host", ["shop.example.com", "93.184.216.34", "2606:2800:220:1::1"])
def test_public_hosts_are_not_private(host: str):
    assert is_private_host(host) is False


def test_metadata_target_is_refused_unless_explicitly_allowed():
    request = make_request(target_url="http://169.254.169.254/latest/meta-data/")
    with pytest.raises(OutOfScopeError, match="metadata"):
        require_target_allowed(request)
    with pytest.raises(OutOfScopeError):
        run_scan(request)

    permitted = make_request(
        target_url="http://169.254.169.254/latest/meta-data/", allow_private_targets=True
    )
    require_target_allowed(permitted)   # explicit opt-in, no exception


def test_private_host_is_out_of_scope_for_links_too():
    request = make_request()
    assert not host_in_scope("http://127.0.0.1:8080/admin", request)


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------


def test_robots_disallow_is_honoured():
    policy = RobotsPolicy.from_text("User-agent: *\nDisallow: /admin\nDisallow: /private\n")
    agent = make_request().user_agent
    assert policy.allows("https://shop.example.com/", agent)
    assert not policy.allows("https://shop.example.com/admin", agent)
    assert not policy.allows("https://shop.example.com/private/report.pdf", agent)


def test_robots_can_be_switched_off_deliberately():
    policy = RobotsPolicy.disabled()
    assert policy.allows("https://shop.example.com/admin", "vulnpriority-scan")


def test_robots_fetch_failure_fails_open_but_is_recorded():
    """An unreachable robots.txt allows the crawl - and says so, rather than silently."""
    request, client, log, _ = make_client(lambda _: httpx.Response(404, text=""))
    policy = fetch_robots_policy(client, request)
    assert policy.allows("https://shop.example.com/anything", request.user_agent)
    assert policy.fetch_error is not None


def _robots_handler(http_request: httpx.Request) -> httpx.Response:
    assert http_request.url.path == "/robots.txt"
    return httpx.Response(200, text="User-agent: *\nDisallow: /admin\n")


def test_robots_binds_when_the_operator_asks_for_it():
    request, client, _, _ = make_client(_robots_handler, make_request(respect_robots=True))
    policy = fetch_robots_policy(client, request)
    assert policy.fetch_error is None
    assert policy.binding is True
    assert not policy.allows("https://shop.example.com/admin", request.user_agent)


def test_robots_is_advisory_by_default_and_still_names_what_it_hides():
    """The file is read either way; only whether it *binds* changes.

    robots.txt is a crawler convention, not an access control. Honouring it in an
    authorised assessment reports a directory as clean when the scan never looked at it,
    which is the most misleading result this scanner can produce. The Disallow list is
    kept in both modes because it is the operator's own inventory of what they would
    rather nobody found - worth reading first rather than throwing away.
    """
    request, client, _, _ = make_client(_robots_handler)
    policy = fetch_robots_policy(client, request)
    assert policy.fetch_error is None
    assert policy.binding is False
    assert policy.allows("https://shop.example.com/admin", request.user_agent)
    assert "/admin" in policy.blocked_paths


def test_a_crawl_delay_is_honoured_whether_or_not_disallow_binds():
    """Unlike Disallow, a crawl delay is a statement about what the server can take."""

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="User-agent: *\nCrawl-delay: 3\nDisallow: /admin\n")

    for respect in (True, False):
        request, client, _, _ = make_client(handler, make_request(respect_robots=respect))
        policy = fetch_robots_policy(client, request)
        assert policy.crawl_delay(request.user_agent) == 3.0


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def test_request_budget_stops_the_scan():
    budget = Budget(max_requests=3, max_pages=100, max_bytes=10**9, time_budget_s=1000.0)
    budget.start()
    for _ in range(3):
        assert not budget.exhausted()
        budget.note_request()
    assert budget.exhausted()
    assert "request budget" in budget.reason()


def test_page_budget_stops_the_scan():
    budget = Budget(max_requests=1000, max_pages=2, max_bytes=10**9, time_budget_s=1000.0)
    budget.start()
    budget.note_page()
    assert not budget.exhausted()
    budget.note_page()
    assert "page budget" in budget.reason()


def test_byte_budget_stops_the_scan():
    budget = Budget(max_requests=1000, max_pages=1000, max_bytes=1000, time_budget_s=1000.0)
    budget.start()
    budget.note_bytes(999)
    assert not budget.exhausted()
    budget.note_bytes(1)
    assert "byte budget" in budget.reason()


def test_time_budget_stops_the_scan():
    clock = FakeClock()
    budget = Budget(max_requests=10**6, max_pages=10**6, max_bytes=10**9, time_budget_s=5.0, clock=clock)
    budget.start()
    clock.advance(4.9)
    assert not budget.exhausted()
    clock.advance(0.2)
    assert "time budget" in budget.reason()


def test_response_byte_cap_truncates_the_body():
    """A huge response is abandoned at the cap rather than read into memory."""
    payload = "<html>" + ("A" * 50_000) + "</html>"
    request = make_request(max_response_bytes=2048)
    clock = FakeClock()
    transport, _ = recording_transport(lambda _: httpx.Response(200, html=payload))
    client = HttpClient(request, transport=transport, clock=clock, sleep=clock.sleep)

    result = client.fetch(TARGET)
    assert result.truncated is True
    assert result.body_bytes_len <= 2048
    assert len(result.body_text) <= 2048


def test_client_refuses_to_send_once_a_budget_is_spent():
    request = make_request(max_requests=2)
    clock = FakeClock()
    transport, log = recording_transport(lambda _: httpx.Response(200, html="<html>ok</html>"))
    client = HttpClient(request, transport=transport, clock=clock, sleep=clock.sleep)

    client.fetch(TARGET)
    client.fetch(TARGET)
    spent = client.fetch(TARGET)

    assert len(log) == 2
    assert "budget exhausted" in (spent.error or "")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limiter_spaces_requests_with_an_injected_clock():
    """Two requests per second means the second request waits half a second."""
    clock = FakeClock()
    limiter = RateLimiter(2.0, clock=clock, sleep=clock.sleep)

    assert limiter.acquire() == 0.0            # the initial token is free
    assert limiter.acquire() == pytest.approx(0.5, abs=1e-6)
    assert limiter.acquire() == pytest.approx(0.5, abs=1e-6)
    assert clock.now == pytest.approx(1.0, abs=1e-6)


def test_rate_limiter_does_not_wait_when_time_already_passed():
    clock = FakeClock()
    limiter = RateLimiter(2.0, clock=clock, sleep=clock.sleep)
    limiter.acquire()
    clock.advance(10.0)
    assert limiter.acquire() == 0.0


def test_rate_limiter_terminates_under_floating_point_shortfall():
    """A bucket a few ulps short of a token must not spin forever."""
    clock = FakeClock()
    limiter = RateLimiter(20.0, clock=clock, sleep=clock.sleep)
    for _ in range(200):
        limiter.acquire()
    assert limiter.acquired == 200


def test_rate_limiter_rejects_a_nonsensical_rate():
    with pytest.raises(ValueError):
        RateLimiter(0.0)


def test_client_applies_the_rate_limit():
    request = make_request(requests_per_second=2.0)
    clock = FakeClock()
    transport, log = recording_transport(lambda _: httpx.Response(200, html="<html>ok</html>"))
    client = HttpClient(request, transport=transport, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        client.fetch(TARGET)
    assert clock.now == pytest.approx(1.0, abs=1e-6)
    assert len(log) == 3


# ---------------------------------------------------------------------------
# The active probe allowlist
# ---------------------------------------------------------------------------


def test_probe_allowlist_is_exactly_the_three_benign_probes():
    """The closed allowlist. Adding anything here is a deliberate, reviewable change."""
    assert ALLOWED_PROBE_KINDS == frozenset(
        {ProbeKind.REFLECTED_MARKER, ProbeKind.OPTIONS_METHOD, ProbeKind.WELL_KNOWN_PATH}
    )
    assert set(ProbeKind) == set(ALLOWED_PROBE_KINDS)
    for kind in ALLOWED_PROBE_KINDS:
        assert require_allowed_probe(kind) is kind


def test_probe_outside_the_allowlist_is_refused():
    with pytest.raises(ProbeNotAllowedError):
        require_allowed_probe("sql_injection")          # type: ignore[arg-type]
    with pytest.raises(ProbeNotAllowedError):
        require_allowed_probe("brute_force")            # type: ignore[arg-type]


@pytest.mark.parametrize(
    "marker",
    [
        "<script>alert(1)</script>",
        "' OR '1'='1",
        "../../etc/passwd",
        "${jndi:ldap://x}",
        "a;rm -rf /",
        "",
        "ab",
    ],
)
def test_markers_that_could_express_a_payload_are_refused(marker: str):
    with pytest.raises(ProbeNotAllowedError):
        assert_benign_marker(marker)


def test_generated_markers_are_always_benign():
    for _ in range(50):
        marker = make_marker()
        assert marker.isalnum()
        assert marker.startswith("vulnpriority")
        assert assert_benign_marker(marker) == marker


def test_probes_only_ever_send_get_or_options():
    """Recorded end to end: the two probe helpers issue no state-changing verb."""
    methods: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        methods.append(http_request.method)
        return httpx.Response(200, html="<html>ok</html>", headers={"Allow": "GET, POST"})

    request = make_request(profile=ScanProfile.ACTIVE)
    clock = FakeClock()
    transport, _ = recording_transport(handler)
    client = HttpClient(request, transport=transport, clock=clock, sleep=clock.sleep)

    page = client.fetch("https://shop.example.com/echo?q=1").to_page()
    probe_reflected_marker(request, client, page)
    probe_options(request, client, page)

    assert set(methods) <= {HttpMethod.GET.value, HttpMethod.OPTIONS.value}
