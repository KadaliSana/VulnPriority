"""Crawling, probing and Scan assembly, against a fake site served by a mock transport.

The site below exercises the shapes a real application has: a home page, a login form, an
admin area behind robots.txt, a static asset with a versioned library, a stack-trace page,
a permissive-CORS API, a link to ``/.git/config``, a redirect that leaves the authorised
host, and a link that looks destructive.

Every request the scanner makes is recorded, so the assertions about what it *did not* do
are assertions about a complete log rather than about intent.
"""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest

from vulnprio.core.enums import HttpMethod, PrivilegeLevel, ScannerSeverity
from vulnprio.core.models import Scan
from vulnprio.ingest.correlate import FindingCorrelator, dedup_key_for
from vulnprio.ingest.generic import GenericJsonParser, detect_parser
from vulnprio.ingest.normalize import make_app_id, make_endpoint_id, make_scan_id
from vulnprio.scan import (
    Crawler,
    HttpClient,
    ScanPhase,
    ScanProfile,
    ScanRequest,
    assess_target,
    extract_forms,
    extract_links,
    run_scan,
    visit_key,
)
from vulnprio.scan.models import SCANNER_NAME

HOST = "shop.example.com"
ORIGIN = f"https://{HOST}"
SCANNED_AT = datetime(2024, 6, 1, 9, 30, 0)

HOME = """<html><head><title>Example Shop</title></head><body>
  <h1>Shop</h1>
  <a href="/login">Sign in</a>
  <a href="/search?q=boots">Search</a>
  <a href="/api/orders">Orders API</a>
  <a href="/oops">Broken page</a>
  <a href="/admin/">Admin</a>
  <a href="/.git/config">config</a>
  <a href="/logout">Sign out</a>
  <a href="/go">Partner</a>
  <a href="https://cdn.other-example.net/asset.js">CDN</a>
  <script src="/static/jquery-1.12.4.min.js"></script>
</body></html>"""

LOGIN = """<html><body>
  <form method="post" action="/login">
    <input type="text" name="username">
    <input type="password" name="password">
    <input type="submit" value="Sign in">
  </form>
</body></html>"""

SEARCH = """<html><body><p>No results for boots.</p>
  <a href="/product/1">One</a><a href="/product/2">Two</a><a href="/product/3">Three</a>
</body></html>"""

STACKTRACE = """<html><body><pre>Traceback (most recent call last):
  File "/srv/app/views.py", line 88, in render
    raise ValueError("boom")
ValueError: boom</pre></body></html>"""

ROBOTS = "User-agent: *\nDisallow: /admin\n"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += float(seconds)


class FakeSite:
    """The mock target. Records every request it receives, verb included."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, str(request.url)))
        path = request.url.path
        if request.method == "OPTIONS":
            return httpx.Response(200, headers={"Allow": "GET, HEAD, OPTIONS, PUT, DELETE"}, text="")
        if path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS)
        if path == "/.well-known/security.txt":
            return httpx.Response(404, text="")
        if path == "/":
            return httpx.Response(
                200,
                html=HOME,
                headers={"Server": "nginx/1.18.0", "Set-Cookie": "sid=abc123; Path=/"},
            )
        if path == "/login":
            return httpx.Response(200, html=LOGIN)
        if path == "/search":
            marker = request.url.params.get("q", "")
            return httpx.Response(200, html=SEARCH.replace("boots", marker or "boots"))
        if path == "/api/orders":
            return httpx.Response(
                200,
                json={"orders": []},
                headers={
                    "access-control-allow-origin": "*",
                    "access-control-allow-credentials": "true",
                },
            )
        if path == "/oops":
            return httpx.Response(500, html=STACKTRACE)
        if path == "/.git/config":
            return httpx.Response(200, text="[core]\n\trepositoryformatversion = 0\n")
        if path.startswith("/static/"):
            return httpx.Response(
                200, text="/* jquery */", headers={"content-type": "application/javascript"}
            )
        if path.startswith("/product/"):
            return httpx.Response(200, html="<html><body>a product</body></html>")
        if path == "/go":
            return httpx.Response(302, headers={"Location": "https://cdn.other-example.net/landing"})
        if path.startswith("/admin"):
            return httpx.Response(200, html="<html><body>secret admin</body></html>")
        if path == "/logout":
            return httpx.Response(200, html="<html><body>bye</body></html>")
        return httpx.Response(404, html="<html><body>not found</body></html>")

    @property
    def urls(self) -> list[str]:
        return [url for _, url in self.requests]

    @property
    def methods(self) -> set[str]:
        return {method for method, _ in self.requests}


def make_request(**overrides) -> ScanRequest:
    base = {
        "target_url": f"{ORIGIN}/",
        "authorized": True,
        "authorization_note": "Engagement PT-2024-114, authorised by the application owner",
        "requests_per_second": 20.0,
        "scanned_at": SCANNED_AT,
    }
    base.update(overrides)
    return ScanRequest(**base)


def make_client(request: ScanRequest, site: FakeSite) -> tuple[HttpClient, FakeClock]:
    clock = FakeClock()
    client = HttpClient(
        request, transport=httpx.MockTransport(site), clock=clock, sleep=clock.sleep
    )
    return client, clock


@pytest.fixture
def site() -> FakeSite:
    return FakeSite()


# ---------------------------------------------------------------------------
# Link and form extraction
# ---------------------------------------------------------------------------


def test_links_are_absolutised_and_filtered():
    markup = """<a href="/a">a</a><a href="b">b</a><a href="#frag">f</a>
    <a href="mailto:x@example.com">m</a><a href="javascript:alert(1)">j</a>
    <script src="/static/app.js"></script>"""
    links = extract_links(markup, f"{ORIGIN}/dir/page")
    assert f"{ORIGIN}/a" in links
    assert f"{ORIGIN}/dir/b" in links
    assert f"{ORIGIN}/static/app.js" in links
    assert not any("mailto" in link or "javascript" in link for link in links)


def test_script_src_survives_script_stripping():
    """The versioned-library evidence lives on a script element; it must not be lost."""
    links = extract_links('<script src="/static/jquery-1.12.4.min.js"></script>', f"{ORIGIN}/")
    assert links == (f"{ORIGIN}/static/jquery-1.12.4.min.js",)


def test_forms_are_parsed_with_their_fields():
    forms = extract_forms(LOGIN, f"{ORIGIN}/login")
    assert len(forms) == 1
    form = forms[0]
    assert form.method == HttpMethod.POST
    assert form.action == f"{ORIGIN}/login"
    assert form.field_names == ("username", "password")
    assert form.has_password_field is True
    assert form.is_state_changing is True


def test_visit_key_templates_instance_urls():
    """The crawl collapses instance URLs the same way ingest collapses endpoints."""
    assert visit_key(f"{ORIGIN}/users/1") == visit_key(f"{ORIGIN}/users/2")
    assert visit_key(f"{ORIGIN}/users/1") != visit_key(f"{ORIGIN}/orders/1")
    assert visit_key(f"{ORIGIN}/s?q=a") == visit_key(f"{ORIGIN}/s?q=b")
    assert visit_key(f"{ORIGIN}/s?q=a") != visit_key(f"{ORIGIN}/s")


# ---------------------------------------------------------------------------
# Crawl behaviour
# ---------------------------------------------------------------------------


def test_crawl_stays_on_the_authorised_host(site: FakeSite):
    request = make_request()
    client, _ = make_client(request, site)
    Crawler().crawl(request, client)

    assert site.urls, "the crawl fetched nothing"
    for url in site.urls:
        assert httpx.URL(url).host == HOST, f"requested an out-of-scope host: {url}"
    assert not any("other-example.net" in url for url in site.urls)


def test_redirect_off_scope_is_not_followed(site: FakeSite):
    request = make_request()
    client, _ = make_client(request, site)
    Crawler().crawl(request, client)

    assert any(url.endswith("/go") for url in site.urls), "the in-scope redirect was not tried"
    assert not any("other-example.net" in url for url in site.urls)


def test_robots_disallow_binds_when_the_operator_asks_for_it(site: FakeSite):
    request = make_request(respect_robots=True)
    client, _ = make_client(request, site)
    crawler = Crawler()
    crawler.crawl(request, client)

    assert not any("/admin" in url for url in site.urls)
    assert crawler.stats.robots_blocked >= 1


def test_a_disallowed_path_is_crawled_by_default_and_reported(site: FakeSite):
    """An authorised assessment covers ``/admin``; a search engine does not.

    robots.txt is a crawler convention rather than an access control, so honouring it
    would report the directory as clean having never requested it. The path is still
    named in the coverage notes, because the operator listing it there is information in
    its own right.
    """
    request = make_request()
    client, _ = make_client(request, site)
    crawler = Crawler()
    crawler.crawl(request, client)

    assert any("/admin" in url for url in site.urls)
    note = " ".join(crawler.coverage_notes)
    assert "/admin" in note and "not an access control" in note


def test_destructive_looking_links_are_not_followed(site: FakeSite):
    request = make_request()
    client, _ = make_client(request, site)
    crawler = Crawler()
    crawler.crawl(request, client)

    assert not any(url.endswith("/logout") for url in site.urls)
    assert crawler.stats.skipped_dangerous >= 1


def test_no_state_changing_verb_is_ever_sent_in_passive_mode(site: FakeSite):
    """Forms are recorded, never submitted: the log must contain GET and nothing else."""
    request = make_request()
    client, _ = make_client(request, site)
    outcome = run_scan(request, client=client)

    assert site.methods == {"GET"}
    assert outcome.forms_recorded >= 1
    assert "POST" not in site.methods


def test_forms_are_recorded_as_endpoints_but_never_submitted(site: FakeSite):
    request = make_request()
    client, _ = make_client(request, site)
    outcome = run_scan(request, client=client)

    posts = [ep for ep in outcome.scan.endpoints if ep.method == HttpMethod.POST]
    assert posts, "the login form should appear as a POST endpoint"
    assert posts[0].path == "/login"
    assert set(posts[0].parameters) == {"username", "password"}
    assert not any(method == "POST" for method, _ in site.requests)


def test_page_budget_stops_the_crawl(site: FakeSite):
    request = make_request(max_pages=3)
    client, _ = make_client(request, site)
    crawler = Crawler()
    pages = crawler.crawl(request, client)

    assert len(pages) == 3
    assert "page budget" in (crawler.stats.stopped_reason or "")


def test_depth_limit_stops_the_crawl(site: FakeSite):
    request = make_request(max_depth=0)
    client, _ = make_client(request, site)
    pages = Crawler().crawl(request, client)
    assert [page.url for page in pages] == [f"{ORIGIN}/"]


def test_request_budget_stops_the_crawl(site: FakeSite):
    request = make_request(max_requests=4)
    client, _ = make_client(request, site)
    crawler = Crawler()
    crawler.crawl(request, client)
    assert len(site.requests) <= 4
    assert "request budget" in (crawler.stats.stopped_reason or "")


def test_time_budget_stops_the_crawl(site: FakeSite):
    """Each request costs fake time through the rate limiter, so the clock runs out."""
    request = make_request(time_budget_s=0.2, requests_per_second=10.0)
    client, _ = make_client(request, site)
    crawler = Crawler()
    crawler.crawl(request, client)
    assert "time budget" in (crawler.stats.stopped_reason or "")
    assert len(site.requests) < 10


def test_total_byte_budget_stops_the_crawl(site: FakeSite):
    request = make_request(max_total_bytes=1024)
    client, _ = make_client(request, site)
    crawler = Crawler()
    crawler.crawl(request, client)
    assert "byte budget" in (crawler.stats.stopped_reason or "")


def test_instance_urls_are_crawled_once(site: FakeSite):
    request = make_request()
    client, _ = make_client(request, site)
    Crawler().crawl(request, client)
    products = [url for url in site.urls if "/product/" in url]
    assert len(products) == 1, f"templated routes should be fetched once, got {products}"


def test_progress_events_are_emitted(site: FakeSite):
    request = make_request()
    client, _ = make_client(request, site)
    events: list[str] = []
    outcome = run_scan(request, client=client, on_progress=lambda event: events.append(event.phase.value))

    assert ScanPhase.AUTHORIZE.value in events
    assert ScanPhase.CRAWL.value in events
    assert ScanPhase.DONE.value in events
    assert outcome.progress, "the outcome carries its own progress log"
    assert outcome.progress[0].phase == ScanPhase.AUTHORIZE


# ---------------------------------------------------------------------------
# Active profile
# ---------------------------------------------------------------------------


def test_active_profile_sends_only_allowlisted_probe_kinds(site: FakeSite):
    """Recorded end to end: GET and OPTIONS only, and every value alphanumeric."""
    request = make_request(profile=ScanProfile.ACTIVE)
    client, _ = make_client(request, site)
    outcome = run_scan(request, client=client)

    assert site.methods <= {"GET", "OPTIONS"}
    assert outcome.probes_sent >= 1

    well_known = [url for url in site.urls if url.endswith("/.well-known/security.txt")]
    assert len(well_known) == 1

    markers = [url for url in site.urls if "vulnprio" in url]
    assert markers, "a reflection probe should have been sent"
    for url in markers:
        value = httpx.URL(url).params.get("q", "")
        assert value.isalnum(), f"a probe value must be alphanumeric, got {value!r}"


def test_active_profile_sends_no_payload_shaped_value(site: FakeSite):
    """No request URL may contain a character that could express an exploit."""
    request = make_request(profile=ScanProfile.ACTIVE)
    client, _ = make_client(request, site)
    run_scan(request, client=client)

    forbidden = ("<", ">", "'", '"', "..", "${", "`", "|", ";", "--", "SELECT", "alert(")
    for url in site.urls:
        decoded = str(httpx.URL(url))
        for token in forbidden:
            assert token not in decoded, f"suspicious token {token!r} in requested URL {decoded}"


def test_active_profile_detects_the_reflection_point(site: FakeSite):
    request = make_request(profile=ScanProfile.ACTIVE)
    client, _ = make_client(request, site)
    outcome = run_scan(request, client=client)

    reflected = [f for f in outcome.scan.findings if f.scanner_plugin_id == "reflected-input"]
    assert reflected, "the /search page echoes its q parameter and should be detected"
    assert reflected[0].cwe_id == 79


def test_passive_profile_sends_no_probes(site: FakeSite):
    request = make_request(profile=ScanProfile.PASSIVE)
    client, _ = make_client(request, site)
    outcome = run_scan(request, client=client)

    assert outcome.probes_sent == 0
    assert site.methods == {"GET"}
    assert not any("vulnprio" in url for url in site.urls)
    assert not any(url.endswith("/.well-known/security.txt") for url in site.urls)


# ---------------------------------------------------------------------------
# The produced Scan
# ---------------------------------------------------------------------------


def run_site(site: FakeSite, **overrides):
    request = make_request(**overrides)
    client, _ = make_client(request, site)
    return run_scan(request, client=client)


def test_scan_is_a_valid_canonical_scan(site: FakeSite):
    outcome = run_site(site)
    scan = outcome.scan

    assert isinstance(scan, Scan)
    Scan.model_validate(scan.model_dump(mode="json"))
    assert scan.scanner_name == SCANNER_NAME
    assert scan.hosts == (HOST,)
    assert scan.endpoints and scan.findings
    assert scan.scan_id == make_scan_id(make_app_id(HOST), SCANNED_AT, SCANNER_NAME)


def test_identifiers_follow_the_design_construction(site: FakeSite):
    outcome = run_site(site)
    scan = outcome.scan
    app_id = make_app_id(HOST)

    for endpoint in scan.endpoints:
        assert endpoint.endpoint_id == make_endpoint_id(app_id, HOST, endpoint.method, endpoint.path)
        assert endpoint.app_id == app_id
    for finding in scan.findings:
        assert finding.scan_id == scan.scan_id
        assert scan.endpoint_by_id(finding.endpoint_id) is not None


def test_identifiers_are_stable_across_runs():
    """Same site, same request: byte-identical identifiers. Reproducibility (Gap 4)."""
    first = run_site(FakeSite())
    second = run_site(FakeSite())

    assert first.scan.scan_id == second.scan.scan_id
    assert [e.endpoint_id for e in first.scan.endpoints] == [e.endpoint_id for e in second.scan.endpoints]
    assert sorted(f.finding_id for f in first.scan.findings) == sorted(
        f.finding_id for f in second.scan.findings
    )
    assert first.scan.model_dump(mode="json") == second.scan.model_dump(mode="json")


def test_findings_are_correlated_with_dedup_keys_and_cluster_sizes(site: FakeSite):
    outcome = run_site(site)
    scan = outcome.scan

    assert all(finding.dedup_key for finding in scan.findings)
    for finding in scan.findings:
        assert finding.dedup_key == dedup_key_for(
            scan.app_id, finding.cwe_id, finding.cve_ids, finding.scanner_plugin_id, finding.name
        )
    counts: dict[str, int] = {}
    for finding in scan.findings:
        counts[finding.dedup_key] = counts.get(finding.dedup_key, 0) + 1
    for finding in scan.findings:
        assert finding.cluster_size == counts[finding.dedup_key]

    # Correlating an already-correlated scan is a no-op.
    assert FindingCorrelator().correlate(scan).model_dump() == scan.model_dump()


def test_scan_round_trips_through_the_existing_ingest_pipeline(site: FakeSite, tmp_path):
    """The scanner's product is indistinguishable from a parsed report on disk."""
    outcome = run_site(site)
    path = tmp_path / "self-scan.json"
    path.write_text(json.dumps(outcome.scan.model_dump(mode="json"), default=str), encoding="utf-8")

    parser = detect_parser(path)
    assert isinstance(parser, GenericJsonParser)
    reparsed = parser.parse(path)
    assert reparsed.scan_id == outcome.scan.scan_id
    assert len(reparsed.findings) == len(outcome.scan.findings)
    assert len(reparsed.endpoints) == len(outcome.scan.endpoints)


def test_untrusted_text_provenance_is_set_correctly(site: FakeSite):
    """Scanner prose is SCANNER_OUTPUT; anything echoed from the target is TARGET_RESPONSE."""
    outcome = run_site(site)
    for finding in outcome.scan.findings:
        assert finding.description.provenance.value == "scanner_output"
        for evidence in finding.evidence:
            assert evidence.provenance.value == "target_response"
    for endpoint in outcome.scan.endpoints:
        if endpoint.response_sample is not None:
            assert endpoint.response_sample.provenance.value == "target_response"


def test_expected_findings_are_present(site: FakeSite):
    outcome = run_site(site)
    by_plugin = {finding.scanner_plugin_id for finding in outcome.scan.findings}

    assert "missing-security-headers" in by_plugin       # CWE-693
    assert "cookie-flags" in by_plugin                   # CWE-614/1004/1275
    assert "version-disclosure" in by_plugin             # CWE-200
    assert "verbose-error" in by_plugin                  # CWE-209
    assert "permissive-cors" in by_plugin                # CWE-942
    assert "missing-csrf-token" in by_plugin             # CWE-352
    assert "exposed-sensitive-file" in by_plugin         # CWE-538
    assert "outdated-js-library" in by_plugin            # CWE-1104
    assert "insecure-form" in by_plugin                  # CWE-319 / 525


def test_endpoints_carry_structural_evidence(site: FakeSite):
    outcome = run_site(site)
    home = next(ep for ep in outcome.scan.endpoints if ep.path == "/")

    assert home.response_status == 200
    assert home.sets_cookie is True
    assert home.response_content_type == "text/html"
    assert any(component.product == "nginx" for component in home.observed_tech)
    assert home.links_to, "observed links become graph edges for Component C"


def test_technology_stack_is_fingerprinted(site: FakeSite):
    outcome = run_site(site)
    products = {component.product for component in outcome.scan.tech_stack}
    assert "nginx" in products
    assert "jquery" in products


def test_auth_inference_runs_on_discovered_endpoints(site: FakeSite):
    outcome = run_site(site, respect_robots=False)
    admin = next((ep for ep in outcome.scan.endpoints if ep.path.startswith("/admin")), None)
    assert admin is not None
    assert admin.auth_required == PrivilegeLevel.ADMIN


def test_outcome_counts_what_was_refused(site: FakeSite):
    outcome = run_site(site)
    assert outcome.skipped_out_of_scope >= 1     # the CDN link and the off-scope redirect
    # Counted, not enforced: robots.txt is advisory by default, so /admin is recorded as
    # named-by-robots and crawled anyway. The two counters are separate on purpose -
    # "the operator flagged this" and "the scan did not look" are different facts.
    assert outcome.robots_named >= 1             # /admin
    assert outcome.robots_blocked == 0
    assert outcome.skipped_dangerous_links >= 1  # /logout
    assert outcome.tool == SCANNER_NAME
    assert outcome.profile == ScanProfile.PASSIVE


def test_assess_target_is_the_entry_point(site: FakeSite):
    """``tools=()`` pins this to the built-in path: assess_target now prefers a real
    scanner by default, and a test must never depend on what happens to be installed."""
    request = make_request()
    client, _ = make_client(request, site)
    outcome = assess_target(request, tools=(), client=client)
    assert outcome.scan.findings
    assert outcome.tool == SCANNER_NAME
    assert "No external scanner was found" in outcome.tool_selection


def test_assess_target_defaults_to_preferring_a_real_scanner():
    """The default is external-first; the built-in crawler is the documented fallback."""
    import inspect

    signature = inspect.signature(assess_target)
    assert signature.parameters["prefer_external"].default is True
    assert signature.parameters["force_builtin"].default is False


def test_severities_are_real_scanner_severities(site: FakeSite):
    outcome = run_site(site)
    for finding in outcome.scan.findings:
        assert isinstance(finding.scanner_severity, ScannerSeverity)
        assert 0.0 <= finding.scanner_confidence <= 1.0
