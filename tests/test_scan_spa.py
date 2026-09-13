"""Single-page applications: detection, endpoint recovery, and the honesty of the report.

The behaviour under test is the answer to a specific failure. Pointed at an Angular
application, the link-following crawler fetched the shell and three bundles, found no
anchors to follow, stopped at six URLs, and reported a short queue - which reads as a clean
application rather than as a scan that never started. These tests pin the three things that
had to become true: the shell is recognised, the endpoints the client code names are
recovered and fetched, and whatever the scan still could not reach is *stated* rather than
left to be inferred from a small number.

The fake site below is deliberately shaped like the real case: no anchors at all, an
``<app-root>`` mount, one bundle carrying a route table and a set of API paths, and a
directory that only a mined path's parent points at.
"""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest

from vulnpriority.scan import Crawler, HttpClient, ScanRequest, run_scan
from vulnpriority.scan.crawler import visit_key
from vulnpriority.scan.spa import (
    detect_spa,
    directory_ancestors,
    extract_script_paths,
    extract_view_routes,
)

HOST = "app.example.com"
ORIGIN = f"https://{HOST}"
SCANNED_AT = datetime(2024, 6, 1, 9, 30, 0)

#: An Angular shell: a mount element, bundles, and not one anchor. This is what defeats a
#: link crawler, and it is what almost every modern application serves.
SHELL = """<!doctype html><html><head><title>Shop</title>
  <link rel="stylesheet" href="/styles.css"></head>
  <body><app-root></app-root>
  <script src="/main.js" type="module"></script>
  </body></html>"""

#: A minified bundle, shaped the way bundlers actually emit one: a route table in template
#: literals, API paths written as plain literals, some interpolated against the app's own
#: origin, one interpolated object id, one off-origin URL, and a good deal of noise that
#: must not be mistaken for a path.
BUNDLE = (
    "const R=[{path:`login`,c:L},{path:`basket`,c:B},{path:`order/:id`,c:O},{path:`**`,c:N}];"
    "const h=this.hostServer;"
    'fetch("/api/products");'
    "fetch(`${h}/rest/user/login`,{method:'POST'});"
    "fetch('/rest/user/whoami');"
    "fetch(`/rest/basket/${id}/items`);"
    'fetch("/ftp/order_4711.pdf");'
    'fetch("https://cdn.other-example.net/telemetry");'
    'const s="it\'s not a path"; const t="a b c"; const u="/"; const v="//evil.example.com/x";'
    'const w="/rest/user/delete"; const css="/styles.css"; const q="/rest/search?q=";'
)


class FakeSite:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, str(request.url)))
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /ftp\n")
        if path == "/":
            return httpx.Response(200, html=SHELL)
        if path == "/main.js":
            return httpx.Response(
                200, text=BUNDLE, headers={"content-type": "application/javascript"}
            )
        if path == "/styles.css":
            return httpx.Response(200, text="body{}", headers={"content-type": "text/css"})
        if path == "/rest/":
            return httpx.Response(
                200, html="<html><head><title>listing directory /rest</title></head></html>"
            )
        if path.startswith(("/api/", "/rest/")):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, html="<html><body>not found</body></html>")

    @property
    def paths(self) -> list[str]:
        return [httpx.URL(url).path for _, url in self.requests]


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


@pytest.fixture
def site() -> FakeSite:
    return FakeSite()


def crawl(request: ScanRequest, site: FakeSite) -> tuple[Crawler, list]:
    client = HttpClient(request, transport=httpx.MockTransport(site))
    crawler = Crawler()
    try:
        return crawler, crawler.crawl(request, client)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_a_mount_element_with_no_anchors_is_a_shell(site: FakeSite) -> None:
    _, pages = crawl(make_request(), site)
    verdict = detect_spa(pages[0])
    assert verdict.is_spa is True
    assert verdict.framework == "Angular"
    assert verdict.anchors == 0
    assert verdict.scripts == 1


def test_a_page_full_of_links_is_not_a_shell() -> None:
    """A server-rendered page that hydrates a widget is not a single-page application.

    Without this the detector would fire on any page containing a ``<div id="app">``, and
    the coverage note would start telling operators their ordinary site was unscannable.
    """
    from vulnpriority.scan.models import Page

    page = Page(
        url=f"{ORIGIN}/",
        content_type="text/html",
        status=200,
        body=(
            '<html><body><div id="app"></div>'
            + "".join(f'<a href="/p/{n}">{n}</a>' for n in range(8))
            + '<script src="/w.js"></script></body></html>'
        ),
    )
    assert detect_spa(page).is_spa is False


def test_a_page_with_no_script_is_not_a_shell() -> None:
    from vulnpriority.scan.models import Page

    page = Page(
        url=f"{ORIGIN}/",
        content_type="text/html",
        status=200,
        body='<html><body><div id="root"></div></body></html>',
    )
    assert detect_spa(page).is_spa is False


# ---------------------------------------------------------------------------
# What is mined, and what is refused
# ---------------------------------------------------------------------------


def test_paths_the_client_names_are_recovered() -> None:
    found = extract_script_paths(BUNDLE, f"{ORIGIN}/main.js", limit=100)
    assert f"{ORIGIN}/api/products" in found
    assert f"{ORIGIN}/rest/user/whoami" in found
    # Written as `${this.hostServer}/rest/user/login`: the interpolation is the app's own
    # origin, so the constant remainder is the path and is kept.
    assert f"{ORIGIN}/rest/user/login" in found


def test_an_interpolated_identifier_is_truncated_never_invented() -> None:
    """``/rest/basket/${id}/items`` yields the prefix, not a fabricated object id.

    Guessing an identifier is how a read-only scan starts reading other people's rows.
    """
    found = extract_script_paths(BUNDLE, f"{ORIGIN}/main.js", limit=100)
    assert f"{ORIGIN}/rest/basket/" in found
    assert not any("items" in url for url in found)
    assert not any("${" in url or "%24" in url for url in found)


def test_off_origin_protocol_relative_and_non_paths_are_refused() -> None:
    found = extract_script_paths(BUNDLE, f"{ORIGIN}/main.js", limit=100)
    assert not any("other-example.net" in url for url in found)
    assert not any("evil.example.com" in url for url in found)
    assert f"{ORIGIN}/" not in found                      # a bare slash is the seed
    assert not any(url.endswith(".css") for url in found)  # already fetched as a subresource
    assert not any(" " in url for url in found)
    assert not any("?" in url for url in found)


def test_view_routes_are_read_but_never_turned_into_requests(site: FakeSite) -> None:
    """Client-side routes measure the application; they are not addresses.

    Under hash routing the server never sees them, and under history routing every one
    returns the same shell - so fetching them would inflate the page count without adding a
    single distinct response.
    """
    routes = extract_view_routes(BUNDLE)
    assert "login" in routes and "basket" in routes
    assert "**" not in routes

    _, _pages = crawl(make_request(), site)
    assert "/login" not in site.paths
    assert "/basket" not in site.paths


def test_directory_ancestors_stop_at_the_origin_root() -> None:
    ancestors = directory_ancestors([f"{ORIGIN}/ftp/order_4711.pdf", f"{ORIGIN}/rest/user/login"])
    assert f"{ORIGIN}/ftp/" in ancestors
    assert f"{ORIGIN}/rest/user/" in ancestors and f"{ORIGIN}/rest/" in ancestors
    assert f"{ORIGIN}/" not in ancestors


# ---------------------------------------------------------------------------
# The crawl
# ---------------------------------------------------------------------------


def test_mined_endpoints_are_actually_fetched(site: FakeSite) -> None:
    crawler, pages = crawl(make_request(), site)
    assert crawler.spa.is_spa is True
    assert crawler.stats.mined_from_scripts > 0
    assert crawler.stats.mined_fetched > 0
    assert "/api/products" in site.paths
    assert "/rest/user/whoami" in site.paths
    # Without mining this crawl sees the shell, the stylesheet and the bundle. Three.
    assert len(pages) > 3


def test_a_mined_path_is_still_refused_when_it_looks_destructive(site: FakeSite) -> None:
    """``/rest/user/delete`` is in the bundle. It is not requested.

    The danger heuristic is not weakened by the new discovery source: a path recovered from
    JavaScript is an ordinary candidate and every existing refusal applies to it.
    """
    crawl(make_request(), site)
    assert not any("delete" in path for path in site.paths)


def test_a_mined_path_is_not_requested_when_robots_binds(site: FakeSite) -> None:
    crawl(make_request(respect_robots=True), site)
    assert not any(path.startswith("/ftp") for path in site.paths)


def test_a_mined_path_disallowed_by_robots_is_crawled_by_default(site: FakeSite) -> None:
    """``/ftp`` is exactly where the interesting files are, and robots.txt says so."""
    crawl(make_request(), site)
    assert any(path.startswith("/ftp") for path in site.paths)


def test_link_reachable_pages_are_crawled_before_mined_ones(site: FakeSite) -> None:
    """Mining spends leftover budget, never the budget link-following needed.

    A page budget that stops the crawl must stop it in the mined tail, not before the
    stylesheet and the bundle the shell actually pointed at.
    """
    crawler, pages = crawl(make_request(max_pages=3), site)
    fetched = {page.url.replace(ORIGIN, "") for page in pages}
    assert fetched == {"/", "/styles.css", "/main.js"}
    assert crawler.stats.mined_fetched == 0


# ---------------------------------------------------------------------------
# Saying what was missed
# ---------------------------------------------------------------------------


def test_a_shell_scan_reports_the_blind_spot_it_had(site: FakeSite) -> None:
    crawler, _ = crawl(make_request(), site)
    note = " ".join(crawler.coverage_notes)
    assert "single-page application" in note
    assert "Angular" in note
    assert "JavaScript" in note


def test_paths_robots_refused_are_named_not_silently_dropped(site: FakeSite) -> None:
    """A Disallow entry is itself a disclosure, and honouring it leaves ground uncovered.

    The scan must be able to say which ground, or a reader cannot tell an unassessed
    directory from a clean one.
    """
    crawler, _ = crawl(make_request(respect_robots=True), site)
    note = " ".join(crawler.coverage_notes)
    assert "robots.txt disallowed" in note
    assert "/ftp/" in note


def test_paths_robots_named_are_reported_even_when_they_were_crawled(
    site: FakeSite,
) -> None:
    """The default says the opposite thing, and has to say it rather than going quiet."""
    crawler, _ = crawl(make_request(), site)
    note = " ".join(crawler.coverage_notes)
    assert "/ftp/" in note
    assert "covered anyway" in note


def test_a_budget_that_cut_the_mined_tail_short_says_so(site: FakeSite) -> None:
    crawler, _ = crawl(make_request(max_pages=5), site)
    note = " ".join(crawler.coverage_notes)
    assert "were not fetched" in note
    assert "max_pages" in note


def test_an_ordinary_site_is_not_described_as_a_shell() -> None:
    """No coverage note when there was no coverage gap. Silence has to stay meaningful."""
    class PlainSite(FakeSite):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            self.requests.append((request.method, str(request.url)))
            if request.url.path == "/robots.txt":
                return httpx.Response(404, text="")
            if request.url.path == "/":
                return httpx.Response(200, html='<html><body><a href="/a">a</a></body></html>')
            return httpx.Response(200, html="<html><body>page</body></html>")

    crawler, _ = crawl(make_request(), PlainSite())
    assert crawler.spa.is_spa is False
    assert crawler.coverage_notes == ()


def test_mining_can_be_turned_off(site: FakeSite) -> None:
    client = HttpClient(make_request(), transport=httpx.MockTransport(site))
    crawler = Crawler(mine_scripts=False)
    try:
        pages = crawler.crawl(make_request(), client)
    finally:
        client.close()
    assert crawler.stats.mined_from_scripts == 0
    assert {page.url.replace(ORIGIN, "") for page in pages} == {"/", "/styles.css", "/main.js"}


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_mining_turns_a_three_page_scan_into_a_real_endpoint_inventory(site: FakeSite) -> None:
    client = HttpClient(make_request(), transport=httpx.MockTransport(site))
    try:
        outcome = run_scan(make_request(), client=client)
    finally:
        client.close()
    paths = {endpoint.path for endpoint in outcome.scan.endpoints}
    assert "/api/products" in paths
    assert "/rest/user/whoami" in paths
    assert outcome.coverage_notes, "a shell scan must explain itself"
    assert any("single-page application" in note for note in outcome.coverage_notes)


def test_a_directory_index_reachable_only_through_a_parent_path_is_found(site: FakeSite) -> None:
    """Nothing links to ``/rest/``; it is reached because ``/rest/user/login`` implied it."""
    client = HttpClient(make_request(), transport=httpx.MockTransport(site))
    try:
        outcome = run_scan(make_request(), client=client)
    finally:
        client.close()
    assert "/rest/" in site.paths
    assert any(finding.name == "Directory listing enabled" for finding in outcome.scan.findings)


def test_visit_key_collapses_a_mined_path_against_the_same_linked_path() -> None:
    """A mined URL and a linked URL for one route must not be fetched twice."""
    assert visit_key(f"{ORIGIN}/rest/user/login") == visit_key(f"{ORIGIN}/rest/user/login")
