"""Every check fires on a positive fixture and stays quiet on a clean one.

A detection rule that only ever fires is as useless as one that never does: the false
positives are what makes an operator stop reading the report. So each check here is
exercised twice - against a page exhibiting the weakness, and against a page that is
correct in exactly that respect.

Pages are constructed directly rather than crawled, so these tests are pure functions over
data with no transport involved at all.
"""

from __future__ import annotations

from datetime import date

import pytest

from vulnprio.core.enums import HttpMethod, ScannerSeverity
from vulnprio.scan.checks import (
    CHECKS,
    CheckContext,
    checks_for_profile,
    run_checks,
)
from vulnprio.scan.models import (
    CheckFinding,
    FormField,
    FormInfo,
    Page,
    ProbeKind,
    ProbeResult,
    ScanProfile,
)
from vulnprio.scan.passive import PASSIVE_CHECKS
from vulnprio.scan.active import ACTIVE_CHECKS

URL = "https://shop.example.com/page"

#: A response that is correct on every axis the passive checks look at.
CLEAN_HEADERS = {
    "content-type": "text/html; charset=utf-8",
    "content-security-policy": "default-src 'self'; frame-ancestors 'none'",
    "strict-transport-security": "max-age=31536000; includeSubDomains",
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
}


def page(
    *,
    url: str = URL,
    status: int = 200,
    headers: dict[str, str] | None = None,
    body: str = "<html><body>hello</body></html>",
    set_cookies: tuple[str, ...] = (),
    links: tuple[str, ...] = (),
    forms: tuple[FormInfo, ...] = (),
    content_type: str | None = "text/html",
    method: HttpMethod = HttpMethod.GET,
) -> Page:
    merged = dict(CLEAN_HEADERS)
    merged.update(headers or {})
    return Page(
        url=url,
        final_url=url,
        method=method,
        status=status,
        headers=merged,
        set_cookies=set_cookies,
        body=body,
        body_bytes_len=len(body),
        content_type=content_type,
        links=links,
        forms=forms,
    )


def fire(check_id: str, target: Page, context: CheckContext | None = None) -> list[CheckFinding]:
    return CHECKS[check_id](target, context or CheckContext(pages=(target,)))


def clean_page(**overrides) -> Page:
    return page(**overrides)


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_every_check_declares_a_cwe_and_a_rationale():
    for check in CHECKS.values():
        assert check.cwe_id > 0, f"{check.id} has no CWE"
        assert isinstance(check.severity, ScannerSeverity)
        assert len(check.rationale) > 80, f"{check.id} does not explain why it matters"


def test_profiles_partition_the_registry():
    passive = set(checks_for_profile(ScanProfile.PASSIVE))
    active = set(checks_for_profile(ScanProfile.ACTIVE))
    assert passive <= active, "the active profile must include every passive check"
    assert passive == set(PASSIVE_CHECKS)
    assert active - passive == set(ACTIVE_CHECKS)
    assert all(check.profile == ScanProfile.PASSIVE for check in PASSIVE_CHECKS)


def test_the_active_check_set_is_exactly_the_three_probe_consumers():
    """Active mode adds these and nothing else."""
    assert {check.id for check in ACTIVE_CHECKS} == {
        "reflected-input",
        "options-methods",
        "state-changing-methods",
    }


def test_active_checks_are_silent_without_probe_results():
    """They read probe results; they never generate traffic or guess."""
    target = page()
    context = CheckContext(pages=(target,), probes=())
    for check in ACTIVE_CHECKS:
        assert check(target, context) == []


# ---------------------------------------------------------------------------
# Security headers (CWE-693)
# ---------------------------------------------------------------------------


def test_missing_security_headers_fires_per_missing_header():
    bare = page(headers={key: "" for key in CLEAN_HEADERS if key != "content-type"})
    bare = bare.model_copy(update={"headers": {"content-type": "text/html"}})
    params = {finding.param for finding in fire("missing-security-headers", bare)}
    assert {
        "content-security-policy",
        "x-content-type-options",
        "x-frame-options",
        "referrer-policy",
        "strict-transport-security",
    } <= params
    assert all(finding.cwe_id == 693 for finding in fire("missing-security-headers", bare))


def test_missing_security_headers_is_silent_on_a_correct_page():
    assert fire("missing-security-headers", clean_page()) == []


def test_csp_frame_ancestors_substitutes_for_x_frame_options():
    target = page(headers={"x-frame-options": ""})
    target = target.model_copy(
        update={"headers": {k: v for k, v in target.headers.items() if k != "x-frame-options"}}
    )
    assert not any(f.param == "x-frame-options" for f in fire("missing-security-headers", target))


def test_weak_csp_and_short_hsts_are_reported():
    target = page(
        headers={
            "content-security-policy": "default-src *; script-src 'unsafe-inline'",
            "strict-transport-security": "max-age=60",
        }
    )
    params = {finding.param for finding in fire("missing-security-headers", target)}
    assert "content-security-policy-weak" in params
    assert "strict-transport-security-weak" in params


def test_hsts_is_not_expected_on_plaintext_http():
    target = page(url="http://shop.example.com/page", headers={"strict-transport-security": ""})
    target = target.model_copy(
        update={"headers": {k: v for k, v in target.headers.items() if k != "strict-transport-security"}}
    )
    assert not any(
        f.param == "strict-transport-security" for f in fire("missing-security-headers", target)
    )


# ---------------------------------------------------------------------------
# Cookies (CWE-614 / 1004 / 1275)
# ---------------------------------------------------------------------------


def test_cookie_flags_reports_each_missing_flag_with_its_own_cwe():
    target = page(set_cookies=("sid=abc123; Path=/",))
    by_cwe = {finding.cwe_id: finding for finding in fire("cookie-flags", target)}
    assert set(by_cwe) == {614, 1004, 1275}
    assert all("sid" in (finding.param or "") for finding in by_cwe.values())


def test_cookie_flags_is_silent_on_a_hardened_cookie():
    target = page(set_cookies=("sid=abc123; Path=/; Secure; HttpOnly; SameSite=Lax",))
    assert fire("cookie-flags", target) == []


def test_samesite_none_without_secure_is_reported():
    target = page(set_cookies=("sid=abc; HttpOnly; SameSite=None",))
    findings = fire("cookie-flags", target)
    assert any(finding.cwe_id == 1275 for finding in findings)
    assert any(finding.cwe_id == 614 for finding in findings)


def test_cookie_secure_is_not_demanded_over_plaintext_http():
    target = page(
        url="http://shop.example.com/page", set_cookies=("sid=abc; HttpOnly; SameSite=Lax",)
    )
    assert [finding.cwe_id for finding in fire("cookie-flags", target)] == []


# ---------------------------------------------------------------------------
# Version disclosure (CWE-200)
# ---------------------------------------------------------------------------


def test_version_disclosure_fires_on_a_versioned_server_header():
    target = page(headers={"server": "Apache/2.4.41 (Ubuntu)", "x-powered-by": "PHP/7.4.3"})
    params = {finding.param for finding in fire("version-disclosure", target)}
    assert params == {"server", "x-powered-by"}
    assert all(finding.cwe_id == 200 for finding in fire("version-disclosure", target))


def test_version_disclosure_ignores_an_unversioned_server_header():
    assert fire("version-disclosure", page(headers={"server": "nginx"})) == []


def test_version_disclosure_reads_the_generator_meta_tag():
    target = page(body='<html><head><meta name="generator" content="WordPress 6.4.2"></head></html>')
    assert any(finding.param == "meta-generator" for finding in fire("version-disclosure", target))


# ---------------------------------------------------------------------------
# Verbose errors (CWE-209)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "<pre>Traceback (most recent call last):\n  File \"/app/views.py\", line 40</pre>",
        "<pre>  at com.example.Service.load(Service.java:88)</pre>",
        "Fatal error: Uncaught Error: Class not found in /var/www/app.php on line 12",
        "SQLSTATE[42000]: Syntax error or access violation",
        "System.NullReferenceException: Object reference not set to an instance",
        "<p>You have an error in your SQL syntax; check the manual</p>",
    ],
)
def test_verbose_error_fires_on_real_stack_traces(body: str):
    findings = fire("verbose-error", page(body=body, status=500))
    assert len(findings) == 1
    assert findings[0].cwe_id == 209
    assert findings[0].evidence


def test_verbose_error_is_silent_on_a_friendly_error_page():
    friendly = page(body="<html><body><h1>Sorry, something went wrong.</h1></body></html>", status=500)
    assert fire("verbose-error", friendly) == []


# ---------------------------------------------------------------------------
# Directory listing (CWE-548)
# ---------------------------------------------------------------------------


def test_directory_listing_fires_on_an_apache_index():
    body = "<html><head><title>Index of /uploads</title></head><body><h1>Index of /uploads</h1></body></html>"
    findings = fire("directory-listing", page(body=body))
    assert len(findings) == 1 and findings[0].cwe_id == 548


def test_directory_listing_is_silent_on_a_normal_page():
    assert fire("directory-listing", page(body="<html><body>Our products index</body></html>")) == []


# ---------------------------------------------------------------------------
# Transport (CWE-319)
# ---------------------------------------------------------------------------


def test_plaintext_http_is_reported():
    findings = fire("insecure-transport", page(url="http://shop.example.com/page"))
    assert [finding.param for finding in findings] == ["scheme"]
    assert findings[0].cwe_id == 319


def test_mixed_content_is_reported_on_https():
    body = '<html><body><script src="http://cdn.example.net/a.js"></script></body></html>'
    findings = fire("insecure-transport", page(body=body))
    assert [finding.param for finding in findings] == ["mixed-content"]


def test_clean_https_page_has_no_transport_finding():
    assert fire("insecure-transport", clean_page()) == []


# ---------------------------------------------------------------------------
# CORS (CWE-942)
# ---------------------------------------------------------------------------


def test_wildcard_cors_with_credentials_is_high():
    target = page(
        headers={
            "access-control-allow-origin": "*",
            "access-control-allow-credentials": "true",
        }
    )
    findings = fire("permissive-cors", target)
    assert len(findings) == 1
    assert findings[0].cwe_id == 942
    assert findings[0].severity == ScannerSeverity.HIGH


def test_wildcard_cors_alone_is_low():
    target = page(headers={"access-control-allow-origin": "*"})
    assert fire("permissive-cors", target)[0].severity == ScannerSeverity.LOW


def test_null_origin_is_reported():
    target = page(headers={"access-control-allow-origin": "null"})
    assert fire("permissive-cors", target)[0].cwe_id == 942


def test_absent_cors_headers_produce_nothing():
    assert fire("permissive-cors", clean_page()) == []


# ---------------------------------------------------------------------------
# CSRF (CWE-352)
# ---------------------------------------------------------------------------


def form(method: HttpMethod = HttpMethod.POST, fields: tuple[FormField, ...] = ()) -> FormInfo:
    return FormInfo(
        action="https://shop.example.com/transfer",
        method=method,
        fields=fields,
        source_url=URL,
    )


def test_missing_csrf_token_fires_on_a_post_form():
    target = page(
        forms=(form(fields=(FormField(name="amount"), FormField(name="to"))),)
    )
    findings = fire("missing-csrf-token", target)
    assert len(findings) == 1 and findings[0].cwe_id == 352


@pytest.mark.parametrize(
    "token_name",
    ["csrf_token", "authenticity_token", "__RequestVerificationToken", "csrfmiddlewaretoken", "_token"],
)
def test_csrf_token_recognised_under_its_common_names(token_name: str):
    target = page(
        forms=(form(fields=(FormField(name="amount"), FormField(name=token_name, type="hidden"))),)
    )
    assert fire("missing-csrf-token", target) == []


def test_get_forms_are_not_expected_to_carry_a_csrf_token():
    target = page(forms=(form(method=HttpMethod.GET, fields=(FormField(name="q"),)),))
    assert fire("missing-csrf-token", target) == []


# ---------------------------------------------------------------------------
# Sensitive data in URLs (CWE-598)
# ---------------------------------------------------------------------------


def test_sensitive_query_parameters_are_reported():
    target = page(url="https://shop.example.com/reset?token=abc123&user=7")
    findings = fire("sensitive-data-in-url", target)
    assert [finding.param for finding in findings] == ["token"]
    assert findings[0].cwe_id == 598


def test_sensitive_parameters_in_links_are_reported():
    target = page(links=("https://shop.example.com/account?session=deadbeef",))
    assert any(finding.param == "session" for finding in fire("sensitive-data-in-url", target))


def test_get_form_submitting_a_password_is_reported():
    target = page(
        forms=(
            FormInfo(
                action="https://shop.example.com/login",
                method=HttpMethod.GET,
                fields=(FormField(name="password", type="password"),),
                source_url=URL,
            ),
        )
    )
    assert any(finding.param == "password" for finding in fire("sensitive-data-in-url", target))


def test_ordinary_query_parameters_are_not_reported():
    target = page(url="https://shop.example.com/search?q=boots&page=2")
    assert fire("sensitive-data-in-url", target) == []


# ---------------------------------------------------------------------------
# Open redirect (CWE-601)
# ---------------------------------------------------------------------------


def test_open_redirect_parameter_is_detected_structurally():
    target = page(url="https://shop.example.com/go?next=https://elsewhere.example.org/")
    findings = fire("open-redirect-param", target)
    assert [finding.param for finding in findings] == ["next"]
    assert findings[0].cwe_id == 601
    assert findings[0].confidence < 0.5, "a structural signal must not masquerade as confirmed"


def test_open_redirect_ignores_a_non_url_value():
    target = page(url="https://shop.example.com/go?next=checkout")
    assert fire("open-redirect-param", target) == []


def test_open_redirect_ignores_unrelated_parameters():
    target = page(url="https://shop.example.com/go?sort=https://x.example/")
    assert fire("open-redirect-param", target) == []


# ---------------------------------------------------------------------------
# Forms: plaintext posting and password autocomplete (CWE-319 / CWE-525)
# ---------------------------------------------------------------------------


def test_form_posting_over_http_is_reported():
    target = page(
        forms=(
            FormInfo(
                action="http://shop.example.com/login",
                method=HttpMethod.POST,
                fields=(FormField(name="password", type="password"),),
                source_url=URL,
            ),
        )
    )
    findings = fire("insecure-form", target)
    assert any(finding.cwe_id == 319 for finding in findings)
    assert any(finding.severity == ScannerSeverity.HIGH for finding in findings)


def test_password_autocomplete_is_reported():
    target = page(
        forms=(
            FormInfo(
                action="https://shop.example.com/login",
                method=HttpMethod.POST,
                fields=(FormField(name="password", type="password"),),
                source_url=URL,
            ),
        )
    )
    findings = fire("insecure-form", target)
    assert [finding.cwe_id for finding in findings] == [525]


def test_password_autocomplete_off_is_accepted():
    target = page(
        forms=(
            FormInfo(
                action="https://shop.example.com/login",
                method=HttpMethod.POST,
                fields=(FormField(name="password", type="password", autocomplete="off"),),
                source_url=URL,
            ),
        )
    )
    assert fire("insecure-form", target) == []


# ---------------------------------------------------------------------------
# Exposed files (CWE-538)
# ---------------------------------------------------------------------------


def test_linked_git_directory_is_reported():
    target = page(links=("https://shop.example.com/.git/config",))
    findings = fire("exposed-sensitive-file", target)
    assert len(findings) == 1
    assert findings[0].cwe_id == 538
    assert findings[0].url.endswith("/.git/config")


def test_reachable_sensitive_file_is_escalated_to_critical():
    linked = "https://shop.example.com/.env"
    fetched = page(url=linked, status=200, content_type="text/plain", body="DB_PASSWORD=x")
    home = page(links=(linked,))
    context = CheckContext(pages=(home, fetched))
    findings = CHECKS["exposed-sensitive-file"](home, context)
    assert findings[0].severity == ScannerSeverity.CRITICAL


@pytest.mark.parametrize(
    "link",
    [
        "https://shop.example.com/backup.zip",
        "https://shop.example.com/wp-config.php.bak",
        "https://shop.example.com/app.js.old",
        "https://shop.example.com/.htpasswd",
    ],
)
def test_backup_and_credential_paths_are_recognised(link: str):
    assert fire("exposed-sensitive-file", page(links=(link,)))


def test_sensitive_paths_are_never_guessed():
    """Nothing is reported for a page with ordinary links: the check reads, it does not probe."""
    target = page(links=("https://shop.example.com/about", "https://shop.example.com/static/app.js"))
    assert fire("exposed-sensitive-file", target) == []


# ---------------------------------------------------------------------------
# Outdated libraries (CWE-1104)
# ---------------------------------------------------------------------------


def _feed(known: set[str]):
    def lookup(cve_id: str, when: date | None) -> bool:
        return cve_id in known

    return lookup


def test_outdated_library_is_detected_and_cross_referenced():
    target = page(links=("https://shop.example.com/static/jquery-1.12.4.min.js",))
    context = CheckContext(
        pages=(target,), as_of=date(2024, 6, 1), cve_lookup=_feed({"CVE-2020-11022"})
    )
    findings = CHECKS["outdated-js-library"](target, context)
    assert len(findings) == 1
    assert findings[0].cwe_id == 1104
    assert "CVE-2020-11022" in findings[0].detail
    assert "CVE-2020-11023" not in findings[0].detail, "unconfirmed CVEs must not be claimed"
    assert findings[0].confidence > 0.8


def test_confidence_drops_when_no_feed_is_available():
    target = page(links=("https://shop.example.com/static/jquery-1.12.4.min.js",))
    findings = CHECKS["outdated-js-library"](target, CheckContext(pages=(target,)))
    assert findings and findings[0].confidence <= 0.6
    assert "no intelligence feed consulted" in findings[0].evidence


def test_current_library_version_is_not_reported():
    target = page(links=("https://shop.example.com/static/jquery-3.7.1.min.js",))
    assert CHECKS["outdated-js-library"](target, CheckContext(pages=(target,))) == []


# ---------------------------------------------------------------------------
# Active checks, driven by probe results
# ---------------------------------------------------------------------------


def test_reflected_marker_finding_requires_a_reflected_probe():
    target = page()
    reflected = ProbeResult(
        kind=ProbeKind.REFLECTED_MARKER,
        url=URL,
        param="q",
        marker="vulnprioabc123",
        status=200,
        reflected=True,
        evidence="you said vulnprioabc123",
    )
    not_reflected = reflected.model_copy(update={"reflected": False, "evidence": ""})

    assert CHECKS["reflected-input"](target, CheckContext(pages=(target,), probes=(reflected,)))
    assert CHECKS["reflected-input"](target, CheckContext(pages=(target,), probes=(not_reflected,))) == []


def test_reflected_marker_finding_is_cwe_79_and_hedged():
    target = page()
    probe = ProbeResult(
        kind=ProbeKind.REFLECTED_MARKER, url=URL, param="q", reflected=True, evidence="x"
    )
    finding = CHECKS["reflected-input"](target, CheckContext(pages=(target,), probes=(probe,)))[0]
    assert finding.cwe_id == 79
    assert finding.confidence <= 0.6
    assert "requires manual review" in finding.detail


def test_options_enumeration_reports_the_allow_set():
    target = page()
    probe = ProbeResult(
        kind=ProbeKind.OPTIONS_METHOD,
        url=URL,
        method=HttpMethod.OPTIONS,
        allow_methods=("GET", "HEAD", "OPTIONS"),
    )
    context = CheckContext(pages=(target,), probes=(probe,))
    assert CHECKS["options-methods"](target, context)[0].cwe_id == 650
    assert CHECKS["state-changing-methods"](target, context) == []


def test_state_changing_methods_are_reported_when_advertised():
    target = page()
    probe = ProbeResult(
        kind=ProbeKind.OPTIONS_METHOD,
        url=URL,
        method=HttpMethod.OPTIONS,
        allow_methods=("DELETE", "GET", "PUT"),
    )
    finding = CHECKS["state-changing-methods"](target, CheckContext(pages=(target,), probes=(probe,)))[0]
    assert finding.severity == ScannerSeverity.HIGH
    assert "does not send them" in finding.detail


# ---------------------------------------------------------------------------
# The whole passive set over a clean page
# ---------------------------------------------------------------------------


def test_a_correctly_configured_page_produces_no_passive_findings():
    """The false-positive floor: a page that does everything right is reported clean."""
    target = page(set_cookies=("sid=abc; Path=/; Secure; HttpOnly; SameSite=Lax",))
    assert run_checks([target], CheckContext(pages=(target,)), ScanProfile.PASSIVE) == []


def test_passive_checks_never_look_at_probe_results():
    """Passive findings must be reproducible from the response alone."""
    target = page(url="http://shop.example.com/page")
    probe = ProbeResult(kind=ProbeKind.REFLECTED_MARKER, url=target.url, reflected=True)
    with_probe = run_checks([target], CheckContext(pages=(target,), probes=(probe,)), ScanProfile.PASSIVE)
    without = run_checks([target], CheckContext(pages=(target,)), ScanProfile.PASSIVE)
    assert [finding.check_id for finding in with_probe] == [finding.check_id for finding in without]


# ---------------------------------------------------------------------------
# Collapsing what the origin, not the endpoint, is guilty of
# ---------------------------------------------------------------------------


def test_an_origin_wide_misconfiguration_is_reported_once():
    """One server missing one header is one defect, not one per page crawled.

    Before this, a queue's length was a function of crawl budget: raising max_pages from
    six to a hundred and twenty turned fifteen findings into three hundred and sixty-two
    without discovering a single new fact. The endpoint-specific findings that mattered
    were then buried under the repetition.
    """
    pages = [
        page(url=f"https://shop.example.com/p/{n}", headers={"content-security-policy": ""})
        for n in range(12)
    ]
    findings = run_checks(pages, CheckContext(pages=tuple(pages)), ScanProfile.PASSIVE)
    headers = [f for f in findings if f.check_id == "missing-security-headers"]
    assert len(headers) == 1
    assert "12 endpoint(s)" in headers[0].detail


def test_a_collapsed_finding_names_the_paths_it_covers():
    """"Which endpoints" is the whole question; a bare count would not answer it."""
    pages = [
        page(url=f"https://shop.example.com/p/{n}", headers={"content-security-policy": ""})
        for n in range(3)
    ]
    finding = [
        f
        for f in run_checks(pages, CheckContext(pages=tuple(pages)), ScanProfile.PASSIVE)
        if f.check_id == "missing-security-headers"
    ][0]
    for n in range(3):
        assert f"/p/{n}" in finding.detail


def test_two_different_policies_on_one_host_stay_two_findings():
    """Collapse is by evidence, not by check: differing configuration is differing news."""
    permissive = page(
        url="https://shop.example.com/a",
        headers={"access-control-allow-origin": "*", "access-control-allow-credentials": "true"},
    )
    reflected = page(
        url="https://shop.example.com/b",
        headers={
            "access-control-allow-origin": "https://evil.example.net",
            "access-control-allow-credentials": "true",
        },
    )
    findings = run_checks(
        [permissive, reflected], CheckContext(pages=(permissive, reflected)), ScanProfile.PASSIVE
    )
    cors = [f for f in findings if f.check_id == "permissive-cors"]
    assert len(cors) == 2


def test_endpoint_scoped_findings_are_never_collapsed():
    """An exposed file on two paths is two exposed files."""
    pages = [
        page(url="https://shop.example.com/a", links=("https://shop.example.com/.git/config",)),
        page(url="https://shop.example.com/b", links=("https://shop.example.com/.env",)),
    ]
    context = CheckContext(
        pages=tuple(pages),
        crawled_urls=frozenset(
            {"https://shop.example.com/.git/config", "https://shop.example.com/.env"}
        ),
    )
    exposed = [
        f
        for f in run_checks(pages, context, ScanProfile.PASSIVE)
        if f.check_id == "exposed-sensitive-file"
    ]
    assert len(exposed) == 2


def test_collapse_can_be_turned_off_to_see_the_raw_emissions():
    pages = [
        page(url=f"https://shop.example.com/p/{n}", headers={"content-security-policy": ""})
        for n in range(5)
    ]
    raw = run_checks(pages, CheckContext(pages=tuple(pages)), ScanProfile.PASSIVE, collapse=False)
    assert len([f for f in raw if f.check_id == "missing-security-headers"]) == 5


# ---------------------------------------------------------------------------
# Error disclosure: which signature wins, and what collapses with what
# ---------------------------------------------------------------------------

_SQL_ERROR_PAGE = """<html><body>
  <h2><em>500</em> Error: WHERE parameter "email" has invalid "undefined" value</h2>
  <ul id="stacktrace"><li>at SQLiteQueryGenerator.whereItemQuery (/app/node_modules/x.js:12:3)</li></ul>
</body></html>"""

_PLAIN_ERROR_PAGE = """<html><body>
  <h2><em>401</em> UnauthorizedError: No Authorization header was found</h2>
  <ul id="stacktrace"></ul>
</body></html>"""


def test_the_most_revealing_error_signature_wins_not_the_first_listed():
    """A page carrying both a stack trace and the database error underneath it is the
    database error: the trace says which file threw, the SQL error says attacker-shaped
    input reached the engine. Taking whichever pattern was listed first downgraded exactly
    the findings worth reading."""
    target = page(status=500, body=_SQL_ERROR_PAGE)
    finding = fire("verbose-error", target)[0]
    assert finding.severity == ScannerSeverity.HIGH
    assert "SQL error" in finding.detail


def test_an_error_handler_leaking_on_many_routes_collapses_but_a_sql_error_does_not():
    routes = [
        page(url=f"https://shop.example.com/r/{n}", status=401, body=_PLAIN_ERROR_PAGE)
        for n in range(6)
    ]
    injectable = page(url="https://shop.example.com/search", status=500, body=_SQL_ERROR_PAGE)
    pages = [*routes, injectable]
    findings = [
        f
        for f in run_checks(pages, CheckContext(pages=tuple(pages)), ScanProfile.PASSIVE)
        if f.check_id == "verbose-error"
    ]
    by_severity = {f.severity for f in findings}
    assert len(findings) == 2, "one leaky handler plus one database error"
    assert ScannerSeverity.HIGH in by_severity


def test_a_node_serve_index_listing_is_recognised():
    """Directory listing is not only an Apache phrasing; Node's serve-index has its own."""
    target = page(body="<html><head><title>listing directory /ftp</title></head></html>")
    assert fire("directory-listing", target)


def test_an_ordinary_page_is_not_an_error_page():
    assert fire("verbose-error", page()) == []
