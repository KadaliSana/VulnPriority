"""The Parallel Search provider: request shape, mapping, caps, and every way it fails.

Offline and deterministic like the rest of the suite. Every test drives the real provider
through an injected ``httpx.MockTransport``, so the parsing, the retry loop, the caps and
the accounting are the shipped ones -- no test here needs a key, opens a socket, or reads
the environment.

The response bodies come from ``data/fixtures/intel/parallel_searches.json`` wherever a
realistic one is wanted, for the reason :mod:`vulnprio.intel.offline` gives: a parser tested
only against dicts written by the person who wrote the parser is tested against its own
assumptions. The recorded shapes carry the things the documentation does not mention and a
live key did -- the undocumented ``metadata`` key, ``publish_date: null`` on every result,
``usage`` as one ``sku_search`` entry per call, and excerpts full of scraped navigation
furniture.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from vulnprio.core.enums import IntelQueryKind, IntelSourceKind, Provenance, TrustTier
from vulnprio.intel.models import IntelConfig, IntelQuery
from vulnprio.intel.parallel_search import (
    PARALLEL_ENDPOINT,
    PARALLEL_MODES,
    PARALLEL_PRICE_PER_REQUEST_USD,
    ParallelSearchProvider,
    build_objective,
    excerpt_text,
    load_recorded_responses,
    parallel_documents,
    parse_publish_date,
    rate_limited,
    search_body,
    search_cost_usd,
    usage_of,
)
from vulnprio.intel.provider import BaseSearchProvider, build_search_provider

API_KEY = "test-key-not-a-real-one"

RECORDED = load_recorded_responses()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def config(**overrides: Any) -> IntelConfig:
    """A live Parallel configuration. Never reaches the network: the client is injected."""
    base: dict[str, Any] = {
        "enabled": True,
        "mode": "live",
        "search_provider": "parallel",
    }
    base.update(overrides)
    return IntelConfig(**base)


def queries(cve: str = "CVE-2017-5638") -> tuple[IntelQuery, ...]:
    """A search plan shaped like the one :mod:`vulnprio.intel.queries` builds."""
    return (
        IntelQuery(text=f"{cve} exploit", kind=IntelQueryKind.POC, cve_id=cve, finding_id="f1"),
        IntelQuery(
            text=f"{cve} proof of concept github",
            kind=IntelQueryKind.POC,
            cve_id=cve,
            finding_id="f1",
        ),
        IntelQuery(
            text="apache struts 2.5.10 remote code execution advisory",
            kind=IntelQueryKind.PRODUCT,
            finding_id="f1",
        ),
    )


def transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[httpx.Client, list[httpx.Request]]:
    """A client whose transport is ``handler``, plus the list of requests it saw."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(record)), seen


def responding(
    status: int = 200, body: Any = None, headers: dict[str, str] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_request: httpx.Request) -> httpx.Response:
        if isinstance(body, (bytes, str)):
            return httpx.Response(status, content=body, headers=headers)
        return httpx.Response(status, json=body if body is not None else {}, headers=headers)

    return handler


def provider(
    handler: Callable[[httpx.Request], httpx.Response],
    cfg: IntelConfig | None = None,
    *,
    api_key: str = API_KEY,
) -> tuple[ParallelSearchProvider, list[httpx.Request], list[float]]:
    """A provider wired to ``handler``, with the requests it made and the sleeps it took."""
    cfg = cfg or config()
    client, seen = transport(handler)
    slept: list[float] = []
    return (
        ParallelSearchProvider(cfg, client=client, api_key=api_key, sleep=slept.append),
        seen,
        slept,
    )


def body_of(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content.decode("utf-8"))


def result(url: str, title: str = "t", excerpts: Any = ("body",), publish_date: Any = None) -> dict[str, Any]:
    return {"url": url, "title": title, "publish_date": publish_date, "excerpts": list(excerpts)}


def response_body(results: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """A response in the observed live shape, ``metadata`` key and all."""
    payload: dict[str, Any] = {
        "search_id": "search_1",
        "session_id": "session_1",
        "metadata": None,
        "warnings": None,
        "usage": [{"name": "sku_search", "count": 1}],
        "results": results,
    }
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# The request that goes on the wire
# ---------------------------------------------------------------------------


def test_the_request_is_the_documented_one() -> None:
    """Endpoint, method, key header and the three body keys. Nothing invented."""
    search, seen, _ = provider(responding(200, response_body([result("https://github.com/a/b")])))
    search.gather(queries(), config())

    assert len(seen) == 1, "one request per finding, not one per query"
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == PARALLEL_ENDPOINT
    assert request.headers["x-api-key"] == API_KEY
    assert request.headers["content-type"] == "application/json"
    # The key travels in x-api-key, not as a bearer token.
    assert "authorization" not in {name.lower() for name in request.headers}

    body = body_of(request)
    assert set(body) == {"objective", "search_queries", "mode"}
    assert body["mode"] == "turbo"
    assert body["objective"].startswith("Find public exploit code")
    assert "CVE-2017-5638" in body["objective"]
    assert body["search_queries"] == [query.text for query in queries()]


def test_turbo_is_the_default_and_the_other_three_modes_are_reachable() -> None:
    assert IntelConfig().parallel_mode == "turbo"
    for mode in PARALLEL_MODES:
        search, seen, _ = provider(
            responding(200, response_body([])), config(parallel_mode=mode)
        )
        search.gather(queries(), search.config)
        assert body_of(seen[0])["mode"] == mode


def test_the_whole_query_set_goes_in_one_request_capped_by_the_plan_ceiling() -> None:
    """The cost model is per request, so batching the plan is the cheap and correct shape."""
    plan = tuple(
        IntelQuery(text=f"query {index}", kind=IntelQueryKind.POC, finding_id="f1")
        for index in range(8)
    )
    cfg = config(max_queries=3)
    search, seen, _ = provider(responding(200, response_body([])), cfg)
    search.gather(plan, cfg)

    assert len(seen) == 1
    assert body_of(seen[0])["search_queries"] == ["query 0", "query 1", "query 2"]


def test_duplicate_query_text_is_sent_once() -> None:
    plan = (
        IntelQuery(text="CVE-2017-5638 exploit", kind=IntelQueryKind.POC, finding_id="f1"),
        IntelQuery(text="CVE-2017-5638   exploit", kind=IntelQueryKind.CVE, finding_id="f1"),
    )
    assert search_body(plan, "objective", config())["search_queries"] == ["CVE-2017-5638 exploit"]


def test_the_objective_names_the_identifiers_and_the_observed_component() -> None:
    objective = build_objective(queries())
    assert "CVE-2017-5638" in objective
    assert "apache struts 2.5.10" in objective
    assert objective.endswith(".")


def test_the_objective_is_built_without_a_cve_too() -> None:
    """Most web application findings have no CVE; the reviewed literature usually skips them."""
    plan = (
        IntelQuery(
            text="Reflected Cross Site Scripting cross site scripting exploit technique",
            kind=IntelQueryKind.WEAKNESS,
            finding_id="f2",
        ),
    )
    objective = build_objective(plan)
    assert "cross site scripting" in objective
    assert "CVE-" not in objective


def test_the_phase1_instruction_is_not_smuggled_into_the_objective() -> None:
    """It is written for a model with tools. A search API can act on none of it."""
    instruction = "Research this finding.\n\nAlready read by the feeds: https://example.com/x"
    assert "example.com" not in build_objective(queries(), instruction)


def test_the_recorded_request_audit_never_carries_the_key() -> None:
    search, _, _ = provider(responding(200, response_body([])))
    search.gather(queries(), config())
    audit = json.dumps(search.requests)
    assert API_KEY not in audit
    assert '"x-api-key": "***"' in audit


# ---------------------------------------------------------------------------
# Mapping a response into documents
# ---------------------------------------------------------------------------


def test_a_recorded_response_maps_into_documents() -> None:
    search, _, _ = provider(responding(200, RECORDED["CVE-2017-5638"]))
    gathered = search.gather(queries(), config())

    urls = [document.url for document in gathered.documents]
    assert "https://nvd.nist.gov/vuln/detail/CVE-2017-5638" in urls
    assert "https://github.com/payatu/CVE-2017-5638" in urls
    assert all(document.title for document in gathered.documents)
    assert all(document.query_text == "CVE-2017-5638 exploit" for document in gathered.documents)
    assert all(document.retrieved_at.tzinfo is not None for document in gathered.documents)


def test_every_document_is_reference_page_tier() -> None:
    """The invariant the explainer's tiering of the intel features depends on."""
    search, _, _ = provider(responding(200, RECORDED["CVE-2017-5638"]))
    gathered = search.gather(queries(), config())

    assert gathered.documents
    for document in gathered.documents:
        assert document.snippet.provenance is Provenance.REFERENCE_PAGE
        assert document.tier is TrustTier.REFERENCE_PAGE


def test_multiple_excerpts_are_joined_into_one_snippet() -> None:
    payload = response_body(
        [result("https://github.com/a/b", excerpts=("first span", "second span"))]
    )
    documents, _, _ = parallel_documents(payload, config())
    assert len(documents) == 1
    assert "first span" in documents[0].snippet.text
    assert "second span" in documents[0].snippet.text
    # Blank line, not a space: these are disjoint spans, not consecutive prose.
    assert "first span\n\nsecond span" == documents[0].snippet.text


def test_a_null_publish_date_is_the_normal_case_and_keeps_the_document() -> None:
    """Every result a live key returned had publish_date null, NVD pages included."""
    payload = response_body([result("https://github.com/a/b", publish_date=None)])
    documents, dates, _ = parallel_documents(payload, config())

    assert len(documents) == 1
    assert dates["https://github.com/a/b"] is None


def test_a_publish_date_is_parsed_where_it_parses_and_dropped_where_it_does_not() -> None:
    payload = response_body(
        [
            result("https://github.com/a/plain", publish_date="2017-03-07"),
            result("https://github.com/a/stamped", publish_date="2017-03-07T10:11:12Z"),
            result("https://github.com/a/prose", publish_date="sometime last spring"),
        ]
    )
    _, dates, _ = parallel_documents(payload, config())

    assert dates["https://github.com/a/plain"].isoformat() == "2017-03-07"
    assert dates["https://github.com/a/stamped"].isoformat() == "2017-03-07"
    assert dates["https://github.com/a/prose"] is None


@pytest.mark.parametrize("value", [None, "", "   ", "not a date", "2017-13-45", 12345])
def test_an_unparseable_publish_date_never_raises(value: Any) -> None:
    assert parse_publish_date(value) is None


def test_source_kinds_are_classified_from_the_url() -> None:
    """A GitHub proof of concept, an exploit-db entry and an advisory must be distinguishable."""
    payload = response_body(
        [
            result("https://github.com/payatu/CVE-2017-5638"),
            result("https://www.exploit-db.com/exploits/41570"),
            result("https://nvd.nist.gov/vuln/detail/CVE-2017-5638"),
            result("https://www.cisa.gov/known-exploited-vulnerabilities-catalog"),
            result("https://portswigger.net/daily-swig/struts"),
        ]
    )
    documents, _, _ = parallel_documents(payload, config())
    kinds = {document.url: document.source_kind for document in documents}

    assert kinds["https://github.com/payatu/CVE-2017-5638"] is IntelSourceKind.POC
    assert kinds["https://www.exploit-db.com/exploits/41570"] is IntelSourceKind.POC
    assert kinds["https://nvd.nist.gov/vuln/detail/CVE-2017-5638"] is IntelSourceKind.ADVISORY
    assert kinds["https://www.cisa.gov/known-exploited-vulnerabilities-catalog"] is IntelSourceKind.ADVISORY
    assert kinds["https://portswigger.net/daily-swig/struts"] is IntelSourceKind.WRITEUP
    assert IntelSourceKind.UNKNOWN not in set(kinds.values())


def test_the_real_proof_of_concept_repositories_land_in_the_poc_bucket() -> None:
    """The three repositories a live search actually found for CVE-2017-5638."""
    search, _, _ = provider(responding(200, RECORDED["CVE-2017-5638"]))
    gathered = search.gather(queries(), config())
    poc_urls = {
        document.url
        for document in gathered.documents
        if document.source_kind is IntelSourceKind.POC
    }
    assert poc_urls == {
        "https://github.com/payatu/CVE-2017-5638",
        "https://github.com/mazen160/struts-pwn",
        "https://github.com/deepfence/apache-struts",
    }


def test_results_are_ordered_by_rank_with_decaying_relevance() -> None:
    payload = response_body([result(f"https://github.com/a/{index}") for index in range(4)])
    documents, _, _ = parallel_documents(payload, config())
    relevances = [document.relevance for document in documents]
    assert relevances == sorted(relevances, reverse=True)
    assert relevances[0] > relevances[-1]


def test_a_result_with_no_excerpt_falls_back_to_its_title() -> None:
    payload = response_body([result("https://github.com/a/b", title="A title", excerpts=())])
    documents, _, errors = parallel_documents(payload, config())
    assert documents[0].snippet.text == "A title"
    assert not [message for message in errors if "without_text" in message]


def test_a_result_with_neither_excerpt_nor_title_is_recorded_and_skipped() -> None:
    payload = response_body([result("https://github.com/a/b", title="", excerpts=())])
    documents, _, errors = parallel_documents(payload, config())
    assert documents == []
    assert any("parallel_result_without_text" in message for message in errors)


def test_duplicate_urls_in_one_response_are_collapsed() -> None:
    payload = response_body(
        [result("https://github.com/a/b"), result("https://github.com/a/b", title="again")]
    )
    documents, _, _ = parallel_documents(payload, config())
    assert len(documents) == 1


def test_there_is_no_narrative_and_no_citation_on_this_path() -> None:
    """A search API returns pages, not judgement. Inventing prose would be worse than none."""
    search, _, _ = provider(responding(200, RECORDED["CVE-2017-5638"]))
    gathered = search.gather(queries(), config())
    assert gathered.documents
    assert gathered.narrative == ""
    assert gathered.citations == ()
    assert gathered.model == ""
    assert gathered.provider == "parallel_search"


# ---------------------------------------------------------------------------
# Client-side caps: ours, because the API offers none
# ---------------------------------------------------------------------------


def test_documents_per_finding_are_capped_before_anything_reaches_the_sandbox() -> None:
    """Ten results per request with no parameter to ask for fewer, so the cap is here."""
    payload = response_body([result(f"https://github.com/a/{index}") for index in range(10)])
    documents, _, _ = parallel_documents(payload, config(max_documents=3))
    assert len(documents) == 3


def test_each_excerpt_is_truncated_to_the_configured_character_cap() -> None:
    long_excerpt = "x" * 3000  # one live excerpt ran to 2,777 characters
    payload = response_body([result("https://github.com/a/b", excerpts=(long_excerpt,))])
    documents, _, _ = parallel_documents(payload, config(parallel_max_excerpt_chars=100))
    assert len(documents[0].snippet.text) == 100


def test_the_cap_is_per_excerpt_so_one_long_span_cannot_crowd_out_the_rest() -> None:
    text = excerpt_text(["a" * 500, "b" * 500, "c" * 500], 100)
    assert text.count("\n\n") == 2
    assert len(text) == 300 + 4


def test_the_snippet_budget_still_bounds_the_whole_document() -> None:
    """Two caps, deliberately: per excerpt here, per document in make_document."""
    payload = response_body(
        [result("https://github.com/a/b", excerpts=["y" * 900 for _ in range(10)])]
    )
    documents, _, _ = parallel_documents(
        payload, config(parallel_max_excerpt_chars=900, snippet_char_budget=1000)
    )
    assert len(documents[0].snippet.text) <= 1000


def test_a_host_outside_the_allowlist_is_dropped_and_recorded() -> None:
    """The API has no domain parameter, so an unenforced allowlist would mean nothing."""
    payload = response_body(
        [result("https://github.com/a/b"), result("https://example-blog.invalid/post")]
    )
    documents, _, errors = parallel_documents(payload, config())

    assert [document.url for document in documents] == ["https://github.com/a/b"]
    assert any("domain_not_allowed: example-blog.invalid" in message for message in errors)


# ---------------------------------------------------------------------------
# Failure: never an exception into the scan
# ---------------------------------------------------------------------------


def test_no_key_is_a_recorded_error_not_an_exception() -> None:
    search, seen, _ = provider(responding(200, response_body([])), api_key="")

    assert search.available() is False
    gathered = search.gather(queries(), config())
    assert gathered.documents == ()
    assert any("PARALLEL_API_KEY" in message for message in gathered.errors)
    assert seen == [], "a keyless provider must not open a request"


def test_a_custom_key_env_is_named_in_the_error() -> None:
    search = ParallelSearchProvider(config(parallel_api_key_env="MY_KEY"), api_key="")
    gathered = search.gather(queries(), config(parallel_api_key_env="MY_KEY"))
    assert any("$MY_KEY" in message for message in gathered.errors)


def test_an_empty_search_plan_is_an_error_not_a_request() -> None:
    search, seen, _ = provider(responding(200, response_body([])))
    gathered = search.gather([], config())
    assert any("empty search plan" in message for message in gathered.errors)
    assert seen == []


def test_a_429_is_recognised_retried_and_then_degrades_to_zero_documents() -> None:
    cfg = config(max_retries=2, retry_backoff_s=1.0, max_backoff_s=30.0)
    search, seen, slept = provider(responding(429, {"error": "rate limited"}), cfg)
    gathered = search.gather(queries(), cfg)

    assert len(seen) == 3, "the full retry budget is spent on a rate limit"
    assert slept == [1.0, 2.0], "exponential backoff between attempts, none after the last"
    assert gathered.documents == ()
    assert any(message.startswith("rate_limited") for message in gathered.errors)
    assert search.search_requests == 0, "a 429 is not a search and is not billed as one"


def test_a_429_honours_a_retry_after_header_over_the_exponential_guess() -> None:
    cfg = config(max_retries=1, retry_backoff_s=1.0, max_backoff_s=60.0)
    search, _, slept = provider(
        responding(429, {"error": "slow down"}, headers={"retry-after": "7"}), cfg
    )
    search.gather(queries(), cfg)
    assert slept == [7.0]


def test_a_429_that_clears_on_retry_returns_documents() -> None:
    attempts: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=response_body([result("https://github.com/a/b")]))

    cfg = config(max_retries=1)
    search, _, _ = provider(handler, cfg)
    gathered = search.gather(queries(), cfg)

    assert len(gathered.documents) == 1
    assert any(message.startswith("rate_limited") for message in gathered.errors)
    assert search.search_requests == 1


def test_rate_limited_recognises_the_status_and_the_exception_form() -> None:
    assert rate_limited(429) is True
    assert rate_limited(200) is False
    assert rate_limited(500, RuntimeError("429 RESOURCE_EXHAUSTED")) is True
    assert rate_limited(500, RuntimeError("boom")) is False


def test_a_500_is_retried_and_then_degrades() -> None:
    cfg = config(max_retries=1)
    search, seen, _ = provider(responding(500, "upstream exploded"), cfg)
    gathered = search.gather(queries(), cfg)

    assert len(seen) == 2
    assert gathered.documents == ()
    assert any("api_status_500" in message for message in gathered.errors)


def test_a_400_is_not_retried_because_it_will_be_rejected_again() -> None:
    cfg = config(max_retries=2)
    search, seen, _ = provider(responding(400, {"error": "bad mode"}), cfg)
    gathered = search.gather(queries(), cfg)

    assert len(seen) == 1
    assert gathered.documents == ()
    assert any("api_status_400" in message for message in gathered.errors)


def test_a_timeout_degrades_to_zero_documents_with_a_recorded_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    cfg = config(max_retries=1)
    search, seen, _ = provider(handler, cfg)
    gathered = search.gather(queries(), cfg)

    assert len(seen) == 2, "a timeout reached no decision, so it is retried"
    assert gathered.documents == ()
    assert any("ReadTimeout" in message for message in gathered.errors)


def test_a_connection_error_degrades_the_same_way() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    cfg = config(max_retries=0)
    search, seen, _ = provider(handler, cfg)
    gathered = search.gather(queries(), cfg)

    assert len(seen) == 1
    assert gathered.documents == ()
    assert any("ConnectError" in message for message in gathered.errors)


def test_a_body_that_is_not_json_does_not_raise() -> None:
    search, _, _ = provider(responding(200, "<html>not json</html>"))
    gathered = search.gather(queries(), config())
    assert gathered.documents == ()
    assert any("malformed_response" in message for message in gathered.errors)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "a string",
        {"results": None},
        {"results": "not an array"},
        {"results": [None, 7, "x"]},
        {"results": [{"url": ""}]},
        {"results": [{"title": "no url at all"}]},
    ],
)
def test_a_malformed_body_is_zero_documents_and_a_reason_never_an_exception(payload: Any) -> None:
    documents, _, errors = parallel_documents(payload, config())
    assert documents == []
    assert errors


def test_an_empty_result_set_is_a_recorded_error() -> None:
    search, _, _ = provider(responding(200, RECORDED["EMPTY"]))
    gathered = search.gather(queries(), config())
    assert gathered.documents == ()
    assert any("no_usable_results" in message for message in gathered.errors)


def test_warnings_are_surfaced_rather_than_swallowed() -> None:
    payload = response_body(
        [result("https://github.com/a/b")], warnings={"truncated": "some queries were dropped"}
    )
    documents, _, errors = parallel_documents(payload, config())
    assert len(documents) == 1
    assert any("parallel_warning: truncated" in message for message in errors)


def test_an_unknown_top_level_key_does_not_break_parsing() -> None:
    """``metadata`` was exactly such a key until a live call returned one."""
    payload = response_body(
        [result("https://github.com/a/b")], metadata=None, a_field_nobody_has_seen_yet={"x": 1}
    )
    documents, _, _ = parallel_documents(payload, config())
    assert len(documents) == 1


def test_the_provider_never_raises_into_the_scan() -> None:
    """The contract the whole failure-handling design exists to satisfy."""
    handlers: list[Callable[[httpx.Request], httpx.Response]] = [
        responding(200, "not json"),
        responding(200, {"results": "nope"}),
        responding(429, {}),
        responding(500, ""),
        responding(403, {"error": "forbidden"}),
    ]
    for handler in handlers:
        search, _, _ = provider(handler, config(max_retries=0))
        gathered = search.gather(queries(), config(max_retries=0))
        assert gathered.documents == ()
        assert gathered.errors


# ---------------------------------------------------------------------------
# Usage and spend
# ---------------------------------------------------------------------------


def test_usage_is_read_from_the_responses_own_usage_array() -> None:
    search, _, _ = provider(responding(200, RECORDED["CVE-2017-5638"]))
    gathered = search.gather(queries(), config())

    assert gathered.usage.web_search_requests == 1
    assert gathered.usage.calls == 1
    assert gathered.usage.total_tokens == 0, "there is no model in this phase"
    assert search.usage_entries == [{"name": "sku_search", "count": 1}]


def test_one_request_is_billed_however_many_queries_it_carried() -> None:
    """The observed live behaviour, and the reason one request per finding is correct."""
    payload = response_body([result("https://github.com/a/b")])
    assert usage_of(payload).web_search_requests == 1
    assert len(search_body(queries(), "objective", config())["search_queries"]) == 3


def test_a_missing_or_unusable_usage_array_falls_back_to_one_request() -> None:
    for usage in (None, [], "nonsense", [{"name": "sku_search"}], [{"count": "many"}]):
        assert usage_of({"results": [], "usage": usage}).web_search_requests == 1


def test_a_token_counter_in_usage_cannot_inflate_the_search_count() -> None:
    payload = {
        "results": [],
        "usage": [{"name": "sku_search", "count": 1}, {"name": "tokens", "count": 5000}],
    }
    assert usage_of(payload).web_search_requests == 1


def test_spend_accumulates_across_findings_at_the_published_rate() -> None:
    search, _, _ = provider(responding(200, response_body([result("https://github.com/a/b")])))
    for _ in range(3):
        search.gather(queries(), config())

    assert search.search_requests == 3
    assert search.spend_usd == pytest.approx(0.003)


def test_search_cost_is_the_published_price_per_mode() -> None:
    assert search_cost_usd(1000, "turbo") == pytest.approx(1.0)
    assert search_cost_usd(1000, "fast") == pytest.approx(1.0)
    assert search_cost_usd(1000, "basic") == pytest.approx(5.0)
    assert search_cost_usd(1000, "advanced") == pytest.approx(5.0)
    assert search_cost_usd(0, "turbo") == 0.0
    # An unknown mode is costed at the higher rate: an estimate must never understate a bill.
    assert search_cost_usd(1000, "something-new") == pytest.approx(5.0)
    assert set(PARALLEL_PRICE_PER_REQUEST_USD) == set(PARALLEL_MODES)


def test_an_advanced_mode_run_reports_the_higher_spend() -> None:
    cfg = config(parallel_mode="advanced")
    search, _, _ = provider(responding(200, response_body([result("https://github.com/a/b")])), cfg)
    search.gather(queries(), cfg)
    assert search.spend_usd == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# Reachability through configuration
# ---------------------------------------------------------------------------


def test_the_provider_is_reachable_through_build_search_provider(monkeypatch) -> None:
    monkeypatch.setenv("PARALLEL_API_KEY", API_KEY)
    built = build_search_provider(config())

    assert isinstance(built, ParallelSearchProvider)
    assert isinstance(built, BaseSearchProvider)
    assert built.name == "parallel_search"
    assert built.available() is True


def test_without_a_key_the_provider_is_still_built_but_unavailable(monkeypatch) -> None:
    """Degraded, not absent: "we could not search" stays an ordinary empty result."""
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    built = build_search_provider(config())

    assert isinstance(built, ParallelSearchProvider)
    assert built.available() is False


def test_offline_mode_still_wins_over_a_configured_parallel_provider(monkeypatch) -> None:
    from vulnprio.intel.offline import FixtureSearchProvider

    monkeypatch.setenv("PARALLEL_API_KEY", API_KEY)
    built = build_search_provider(config(mode="offline"))
    assert isinstance(built, FixtureSearchProvider)


def test_the_search_method_of_the_protocol_returns_the_documents() -> None:
    """``ExploitIntelAgent`` codes against ``search``; it must work with no changes."""
    search, _, _ = provider(responding(200, RECORDED["CVE-2017-5638"]))
    documents = search.search(queries(), config())
    assert documents
    assert all(document.snippet.provenance is Provenance.REFERENCE_PAGE for document in documents)


def test_the_endpoint_and_timeout_are_configurable() -> None:
    cfg = config(parallel_endpoint="https://proxy.invalid/v1/search", parallel_timeout_s=5.0)
    search, seen, _ = provider(responding(200, response_body([])), cfg)
    search.gather(queries(), cfg)

    assert str(seen[0].url) == "https://proxy.invalid/v1/search"
    assert search.timeout_s == 5.0


def test_constructing_a_provider_opens_no_connection() -> None:
    search = ParallelSearchProvider(config(), api_key=API_KEY)
    assert search._client is None  # noqa: SLF001 - the point of the test
    search.close()


# ---------------------------------------------------------------------------
# The recorded corpus
# ---------------------------------------------------------------------------


def test_the_recorded_fixture_holds_the_shapes_a_live_key_returned() -> None:
    assert set(RECORDED) >= {"CVE-2017-5638", "EMPTY"}
    for key, body in RECORDED.items():
        assert set(body) >= {"search_id", "session_id", "metadata", "warnings", "usage", "results"}
        assert isinstance(body["results"], list), key
        assert body["usage"] == [{"name": "sku_search", "count": 1}]
        for entry in body["results"]:
            assert set(entry) == {"url", "title", "publish_date", "excerpts"}


def test_the_recorded_corpus_is_absent_rather_than_fatal_when_the_file_is_missing(tmp_path) -> None:
    assert load_recorded_responses(tmp_path / "nothing-here.json") == {}
