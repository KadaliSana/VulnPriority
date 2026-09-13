"""The retrieval half, done by something that only retrieves: the Parallel Search API.

The other two providers in this package are models that happen to be able to search.
Parallel is a search API with no model attached, and that difference is the reason this
module exists twice over -- once for cost, once for architecture.

**Cost.** :mod:`vulnpriority.intel.anthropic_search` works and bills per finding.
:mod:`vulnpriority.intel.gemini_search` is free in principle, but grounding carries its own
quota far tighter than the model's: against a live key, grounded requests return
``429 RESOURCE_EXHAUSTED`` every time while plain generation succeeds seconds earlier. So
the search half of the free path is, in practice, unavailable. Parallel's ``turbo`` mode is
$1 per 1,000 requests and this layer spends exactly one request per finding, which puts a
live intelligence run at a tenth of a cent per finding instead of a wall.

**Architecture.** The two-phase design elsewhere in this package is forced: citations and
structured output cannot coexist in one Anthropic request, and grounding and a response
schema cannot coexist in one Gemini request. Here there is no such conflict to work around,
because there is no model in phase 1 at all. Phase 1 is retrieval and nothing else; phase 2
is extraction and nothing else. The existing split maps onto that cleanly, which is the
shape DESIGN 6.5 argues for on its own merits -- "gathering and judging are different jobs"
-- rather than the shape an API constraint imposed.

The consequence worth stating plainly: **this path produces no researcher narrative, and
therefore no report summary.** :func:`~vulnpriority.intel.summarize.summarize_intel` returns
``None`` without prose, and that is correct. A search API returns pages, not judgement;
inventing prose to fill the field would be exactly the kind of unlabelled model-written
text DESIGN 6.2 forbids. The features still land -- phase 2 still reads the retrieved pages
and still emits bounded numbers -- so the ranking is unaffected. Only the human-readable
paragraph is absent, and its absence is honest.

**The wire contract**, which is small and is not guessed at::

    POST https://api.parallel.ai/v1/search
    Content-Type: application/json
    x-api-key: $PARALLEL_API_KEY

    {"objective": "...", "search_queries": ["...", ...], "mode": "turbo"}

    -> {"search_id": str,
        "results": [{"url": str, "title": str,
                     "publish_date": str | null, "excerpts": [str, ...]}],
        "warnings": object | null,
        "usage": [{"name": str, "count": number}],
        "session_id": str}

``mode`` is one of ``turbo`` ($1/1k, English and Japanese queries), ``fast`` ($1/1k),
``basic`` ($5/1k) and ``advanced`` ($5/1k, the API's own default).
:attr:`~vulnpriority.core.config.IntelConfig.parallel_mode` defaults to ``turbo`` and reaches
the other three.

**What a live key showed that the documentation does not**, all of which this module is
built against rather than around:

* The response carries a top-level ``metadata`` key the docs omit (observed ``null``
  throughout), so the full observed key set is ``metadata``, ``results``, ``search_id``,
  ``session_id``, ``usage``, ``warnings``. Parsing here is by name and never asserts a
  closed shape: a key nobody has seen yet must not be able to break a scan.
* ``publish_date`` was ``null`` on **every** result, NVD detail pages included. Null is the
  normal case, not the exception, and a null date neither suppresses a document nor
  produces a fabricated one.
* ``turbo`` returns in about a second -- 1578ms cold, then 1051ms and 1028ms -- not the
  ~200ms advertised. :attr:`~vulnpriority.core.config.IntelConfig.parallel_timeout_s` is 30s,
  set from the measured number with headroom, and nothing here assumes a fast return.
* Ten results per request, with no parameter to ask for fewer, and one observed excerpt ran
  to 2,777 characters. A finding's search therefore lands 10-30KB of untrusted text. **The
  client-side caps below are load-bearing, not decorative.**
* ``usage`` came back as ``[{"name": "sku_search", "count": 1}]`` per call regardless of how
  many ``search_queries`` it carried, which confirms the request -- not the query -- as the
  billing unit and makes batching the plan into one call the correct cost shape.
* Excerpts are raw scraped page text, newlines and navigation furniture included ("You must
  be signed in to change notification settings"). Nothing here cleans it. Normalisation is
  :mod:`vulnpriority.sandbox.pipeline`'s job, and a second, unaudited cleaner would be a second
  path into the prompt.

**One request per finding, not one per query.** Parallel takes an objective *plus* a list
of queries, so the finding's whole search plan goes in a single call -- the billing unit,
and also the right shape: the objective says what the request is for and the queries say
where to look, which is precisely how :mod:`vulnpriority.intel.queries` already decomposes a
finding.

**Every cap here is ours.** The API documents no max-results parameter, no max-chars
parameter, no domain filter and no published rate limit, so nothing of the sort is sent.
Documents per finding, characters per excerpt and the domain allowlist are all applied
client-side, after the response arrives and before anything reaches the sandbox, and the
arithmetic is deliberately conservative: ``max_documents`` (8) documents, each held to
``snippet_char_budget`` (4,000) characters, is 32,000 characters worst case against a
sandbox ceiling of ``max_segments`` x ``max_chars_per_segment`` = 12 x 6,000 = 72,000.

**Everything retrieved is ``Provenance.REFERENCE_PAGE``**, via
:func:`~vulnpriority.intel.provider.make_document`, which is the only place that provenance is
set and which :class:`~vulnpriority.core.models.IntelDocument` enforces. A search API surfacing
a page says nothing about who wrote it; ADR-002 applies to it identically.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

import httpx

from vulnpriority.core.config import PROJECT_ROOT
from vulnpriority.core.enums import IntelQueryKind
from vulnpriority.intel.models import (
    IntelConfig,
    IntelDocument,
    IntelGather,
    IntelQuery,
    IntelUsage,
    utc_now,
)
from vulnpriority.intel.provider import BaseSearchProvider, make_document

# The allowlist check is the Gemini module's, not a second copy of it. Its reasoning holds
# here for the same reason: the API offers no domain parameter, so an allowlist that is not
# enforced on the results is an allowlist that does nothing.
from vulnpriority.intel.gemini_search import host_allowed
from vulnpriority.llm.gemini_backend import is_rate_limited
from vulnpriority.llm.openai_compatible import RateLimiter, backoff_delay, retry_after_seconds

__all__ = [
    "PARALLEL_ENDPOINT",
    "PARALLEL_FIXTURE_PATH",
    "PARALLEL_KEY_ENV",
    "PARALLEL_MODES",
    "PARALLEL_PRICE_PER_REQUEST_USD",
    "ParallelSearchProvider",
    "build_objective",
    "excerpt_text",
    "load_recorded_responses",
    "parallel_documents",
    "parse_publish_date",
    "rate_limited",
    "search_body",
    "search_cost_usd",
    "usage_of",
]

#: Documented endpoint, key header name and modes. Mirrored in
#: :class:`~vulnpriority.core.config.IntelConfig` as defaults so both are configurable, and kept
#: here as the values this module was written and verified against.
PARALLEL_ENDPOINT = "https://api.parallel.ai/v1/search"
PARALLEL_KEY_ENV = "PARALLEL_API_KEY"
PARALLEL_MODES: tuple[str, ...] = ("turbo", "fast", "basic", "advanced")

#: Published price per search request, in dollars. ``turbo`` and ``fast`` are $1 per 1,000;
#: ``basic`` and ``advanced`` are $5 per 1,000. An unrecognised mode is costed at the higher
#: rate, so an estimate over-states rather than under-states the bill -- the same direction
#: :meth:`~vulnpriority.core.models.IntelUsage.cost_usd` errs in for cache reads.
PARALLEL_PRICE_PER_REQUEST_USD: dict[str, float] = {
    "turbo": 0.001,
    "fast": 0.001,
    "basic": 0.005,
    "advanced": 0.005,
}

#: Recorded ``/v1/search`` response bodies, keyed by CVE. Raw wire shapes rather than the
#: :mod:`vulnpriority.intel.offline` corpus shape, for the reason that module's docstring gives
#: about drift: a fixture hand-written to look like the internet stops resembling it, and a
#: parser tested only against hand-written dicts is tested against its own assumptions.
PARALLEL_FIXTURE_PATH = Path("data/fixtures/intel/parallel_searches.json")

#: Client-side ceiling on the objective string. The API documents no limit; this exists so
#: a pathological search plan cannot assemble an unbounded request body.
MAX_OBJECTIVE_CHARS = 2000

#: How many CVE identifiers the objective names before it stops listing them. Ours.
MAX_OBJECTIVE_CVES = 6

_OBJECTIVE_HEAD = (
    "Find public exploit code, proof-of-concept repositories, vendor and CERT advisories "
    "stating affected versions, and credible reports of active exploitation"
)


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


def build_objective(queries: Sequence[IntelQuery], instruction: str = "") -> str:
    """The natural-language statement of what this request is for.

    Assembled from the same operator-tier facts the queries were -- the CVE identifiers and
    the observed component -- and never from fetched text, for the reason
    :mod:`vulnpriority.intel.queries` states: an objective built from a page would let a page
    choose what the framework reads next.

    ``instruction`` is accepted for protocol compatibility and deliberately not used as the
    objective. The phase-1 instruction is written for a model holding search and fetch
    tools; it names pages to avoid re-reading and asks for prose with citations, none of
    which a search API can act on. Passing it through would make the objective long and
    mostly irrelevant, which is the one thing an objective must not be.
    """
    cves = tuple(
        dict.fromkeys(query.cve_id.strip().upper() for query in queries if query.cve_id)
    )
    subject = next(
        (
            query.text
            for query in queries
            if query.kind in (IntelQueryKind.PRODUCT, IntelQueryKind.WEAKNESS) and query.text
        ),
        "",
    )
    parts = [_OBJECTIVE_HEAD]
    if cves:
        parts.append("for " + ", ".join(cves[:MAX_OBJECTIVE_CVES]))
        if subject:
            parts.append("affecting " + subject)
    elif subject:
        parts.append("for " + subject)
    elif queries:
        parts.append("for " + queries[0].text)
    else:  # pragma: no cover - gather refuses an empty plan before reaching here
        parts.append("for this vulnerability")
    return (" ".join(parts).strip().rstrip(".") + ".")[:MAX_OBJECTIVE_CHARS]


def search_body(
    queries: Sequence[IntelQuery], objective: str, config: IntelConfig
) -> dict[str, Any]:
    """The exact wire body: an objective, a query list, and a mode.

    Three keys, and deliberately nothing else. The Search API documents no max-results,
    max-chars, domain or freshness parameter, and sending an invented one is at best
    ignored and at worst a 400. Every bound this framework wants is applied to the response
    instead -- see :func:`parallel_documents`.

    The query list is deduplicated and capped by
    :attr:`~vulnpriority.core.config.IntelConfig.max_queries`, the same ceiling the search plan
    was built under, so a request cannot exceed the plan that was costed.
    """
    texts = tuple(
        dict.fromkeys(
            " ".join(str(query.text).split()) for query in queries if str(query.text).strip()
        )
    )
    return {
        "objective": objective,
        "search_queries": list(texts[: max(1, int(config.max_queries))]),
        "mode": str(config.parallel_mode),
    }


# ---------------------------------------------------------------------------
# Response mapping
# ---------------------------------------------------------------------------


def parse_publish_date(value: Any) -> date | None:
    """``publish_date`` as a date where it parses, ``None`` where it does not. Never raises.

    Null is the ordinary case rather than a failure: against a live key every result came
    back with ``publish_date: null``, NVD detail pages included. So this returns ``None``
    freely and the caller keeps the document regardless -- a page whose date the framework
    cannot read is still a page worth reading, and a guessed date would be worse than none.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    for candidate in (raw[:10], raw.replace("Z", "+00:00")):
        try:
            return date.fromisoformat(candidate)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(candidate).date()
        except ValueError:
            continue
    return None


def excerpt_text(excerpts: Any, max_chars: int) -> str:
    """The excerpts joined into one snippet, each truncated to ``max_chars``.

    **The truncation is ours, not the API's** -- Parallel documents no length parameter, and
    one observed excerpt ran to 2,777 characters against ten results per request. It is
    applied per excerpt rather than to the join so a single long excerpt cannot crowd the
    others out of the snippet entirely.

    Blank lines separate them because they are disjoint spans of a page, not consecutive
    prose, and running them together would manufacture sentences nobody wrote. Nothing else
    is done to the text: it arrives as raw scraped page content, navigation furniture and
    all, and normalising it is :mod:`vulnpriority.sandbox.pipeline`'s job. Cleaning it here
    would hide from the injection detector what the page actually said.
    """
    parts: list[str] = []
    for excerpt in excerpts or ():
        text = str(excerpt).strip()
        if not text:
            continue
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip()
        parts.append(text)
    return "\n\n".join(parts)


def _warning_messages(warnings: Any) -> list[str]:
    """Readable strings out of the response's ``warnings``, whatever shape it arrives in.

    Observed ``null`` on every live call, so every branch here is defensive. ``warnings`` is
    documented as an object and could reasonably be a string or a list; none of those may
    raise, because a warning is the API being helpful and must never cost a scan a finding.
    """
    if not warnings:
        return []
    if isinstance(warnings, str):
        return [warnings]
    if isinstance(warnings, dict):
        return [f"{key}: {value}" for key, value in warnings.items()]
    if isinstance(warnings, (list, tuple)):
        return [str(item) for item in warnings if item]
    return [str(warnings)]  # pragma: no cover - defensive


def _rank_relevance(index: int) -> float:
    """Relevance from rank, because rank is the only ordering signal in the response.

    Parallel returns results ordered against the objective and nothing here knows better
    than it does why. A gentle decay keeps the provider's ordering while still letting
    :func:`~vulnpriority.intel.queries.dedupe_documents` and the influence budget see a
    difference between the first result and the tenth.
    """
    return max(0.3, 0.9 - 0.05 * max(0, int(index)))


def parallel_documents(
    payload: Any,
    config: IntelConfig,
    query_text: str = "",
    retrieved_at: datetime | None = None,
) -> tuple[list[IntelDocument], dict[str, date | None], list[str]]:
    """A response body -> (documents, publish dates by URL, errors). Never raises.

    Read by name, never by shape. A body that is not an object, a missing or non-list
    ``results``, a result that is not an object, a result with no URL and an empty result
    set are all ordinary outcomes: each produces zero documents and a recorded reason. A
    top-level key nobody has seen before -- ``metadata`` was exactly that until a live call
    returned one -- is simply not looked at. One finding's web research must not be able to
    abort a scan, and a malformed body is precisely when an exception would escape at the
    least convenient moment.

    Three client-side bounds are applied, in this order and all of them ours:

    1. the domain allowlist, because the API has no domain parameter and an unenforced
       allowlist would mean reading the whole internet while the config said otherwise;
    2. :attr:`~vulnpriority.core.config.IntelConfig.parallel_max_excerpt_chars` per excerpt;
    3. :attr:`~vulnpriority.core.config.IntelConfig.max_documents` per finding, enforced by
       stopping rather than by building everything and slicing.

    The publish dates come back beside the documents rather than on them:
    :class:`~vulnpriority.core.models.IntelDocument` has no publication field and it is not this
    module's to add. ``retrieved_at`` on the document means when *this run* fetched the
    page, which is the as-of property DESIGN 6.5 depends on, and a publication date must
    never be allowed to impersonate it.
    """
    stamp = retrieved_at or utc_now()
    documents: list[IntelDocument] = []
    publish_dates: dict[str, date | None] = {}
    errors: list[str] = []

    if not isinstance(payload, dict):
        return documents, publish_dates, ["malformed_response: body is not a JSON object"]

    errors.extend(
        f"parallel_warning: {message}" for message in _warning_messages(payload.get("warnings"))
    )

    results = payload.get("results")
    if not isinstance(results, (list, tuple)):
        errors.append(
            "malformed_response: 'results' is missing"
            if results is None
            else "malformed_response: 'results' is not an array"
        )
        return documents, publish_dates, errors

    limit = max(1, int(config.max_documents))
    seen: set[str] = set()
    for result in results:
        if len(documents) >= limit:
            break  # our cap, not theirs: the API documents no max-results parameter
        if not isinstance(result, dict):
            errors.append("parallel_result_is_not_an_object")
            continue
        url = str(result.get("url") or "").strip()
        if not url:
            errors.append("parallel_result_without_url")
            continue
        if url in seen:
            continue
        seen.add(url)
        if not host_allowed(url, None, config):
            errors.append(f"domain_not_allowed: {urlsplit(url).hostname or '?'}")
            continue
        raw_title = result.get("title")
        title = str(raw_title) if raw_title else None
        text = excerpt_text(result.get("excerpts"), int(config.parallel_max_excerpt_chars))
        if not text:
            # A result with no excerpt is still a page worth handing to phase 2, and the
            # title is what was actually retrieved about it. Padding it out would not be.
            text = title or ""
        if not text:
            errors.append(f"parallel_result_without_text: {url}")
            continue
        # Null here is the common case and changes nothing about whether the page is kept.
        publish_dates[url] = parse_publish_date(result.get("publish_date"))
        documents.append(
            make_document(
                url,
                text,
                title=title,
                retrieved_at=stamp,
                # Source kind is left to make_document's classify_source. Unlike Gemini's
                # grounding chunks there is no redirect in the way, so the URL's host is the
                # publisher's host: github.com lands in POC, nvd.nist.gov in ADVISORY.
                relevance=_rank_relevance(len(documents)),
                query_text=query_text,
                char_budget=config.snippet_char_budget,
            )
        )

    if not documents:
        errors.append("parallel_search_returned_no_usable_results")
    return documents, publish_dates, errors


def usage_of(payload: Any) -> IntelUsage:
    """Accounting for one successful call, read from the response's own ``usage`` array.

    Observed live as ``[{"name": "sku_search", "count": 1}]`` per call, whatever the length
    of ``search_queries`` -- which is the fact that makes one-request-per-finding the cost
    model rather than a convenience. Entries whose name mentions a search are preferred over
    a bare total, so that a token counter appearing later cannot silently inflate the
    reported search count; a body with no usable ``usage`` falls back to one request, which
    is what a successful call cost.

    Token fields stay zero because there is no model in this phase. That is not a gap in the
    accounting; it is the accounting.
    """
    entries = payload.get("usage") if isinstance(payload, dict) else None
    named = 0
    total = 0
    for entry in entries or ():
        if not isinstance(entry, dict):
            continue
        try:
            count = int(entry.get("count") or 0)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        total += count
        if "search" in str(entry.get("name") or "").lower():
            named += count
    return IntelUsage(web_search_requests=max(1, named or total or 1), calls=1)


def search_cost_usd(requests: int, mode: str = "turbo") -> float:
    """Dollars for ``requests`` search requests in ``mode``, at the published rates."""
    rate = PARALLEL_PRICE_PER_REQUEST_USD.get(
        str(mode), max(PARALLEL_PRICE_PER_REQUEST_USD.values())
    )
    return max(0, int(requests)) * rate


def rate_limited(status_code: int, exc: BaseException | None = None) -> bool:
    """Whether this is the provider saying "too many requests".

    A 429 status is the documented form. The exception form is delegated to
    :func:`~vulnpriority.llm.gemini_backend.is_rate_limited` rather than re-derived, so the
    framework keeps one definition of what a rate limit looks like -- which matters here
    because Parallel publishes no rate limits at all, and the shape of the rejection is
    therefore the only thing there is to recognise one by.
    """
    try:
        if int(status_code or 0) == 429:
            return True
    except (TypeError, ValueError):  # pragma: no cover - defensive
        pass
    return bool(exc is not None and is_rate_limited(exc))


# ---------------------------------------------------------------------------
# Recorded fixtures
# ---------------------------------------------------------------------------


def load_recorded_responses(path: str | Path | None = None) -> dict[str, Any]:
    """Recorded ``/v1/search`` bodies keyed by CVE, or ``{}`` when the file is absent.

    Raw responses rather than the :mod:`vulnpriority.intel.offline` corpus shape: these exist so
    the parser is exercised against what the API actually returns, ``metadata`` key, null
    publish dates, scraped navigation furniture and all. Replaying a live session into the
    offline corpus is :class:`~vulnpriority.intel.offline.RecordingProvider`'s job, which works
    on this provider unchanged because it wraps any :class:`BaseSearchProvider`.
    """
    resolved = Path(path) if path is not None else PARALLEL_FIXTURE_PATH
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.is_file():
        return {}
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"parallel fixture {resolved} must hold an object")
    responses = data.get("responses")
    if responses is None:
        return {}
    if not isinstance(responses, dict):
        raise ValueError(f"parallel fixture {resolved} has a non-object 'responses'")
    return responses


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------


class ParallelSearchProvider(BaseSearchProvider):
    """One Parallel Search request per finding, carrying the finding's whole query set."""

    name = "parallel_search"

    def __init__(
        self,
        config: IntelConfig | None = None,
        client: httpx.Client | None = None,
        *,
        api_key: str | None = None,
        sleep: Any | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        """``client`` is injectable so the suite can assert request shape with no key.

        A missing key is *not* raised here, matching both sibling providers: the layer
        degrades to an empty result with a recorded reason rather than failing a scan, so
        "no key" is an unavailable provider and never an exception.

        The key is read here and sent as a header on every request, which is why
        :meth:`available` tests the key and not the client: an injected client proves a
        transport exists, not that this run may authenticate.
        """
        self.config = config or IntelConfig()
        self.api_key_env = str(self.config.parallel_api_key_env or PARALLEL_KEY_ENV)
        self._api_key = api_key if api_key is not None else os.environ.get(self.api_key_env, "")
        self.endpoint = str(self.config.parallel_endpoint or PARALLEL_ENDPOINT)
        self.mode = str(self.config.parallel_mode)
        self._sleep = sleep or (lambda _seconds: None)
        self._client = client
        self._owns_client = client is None
        self.limiter = (
            limiter
            if limiter is not None
            else RateLimiter(self.config.requests_per_minute, sleep=self._sleep)
        )
        #: Every request this provider issued, for tests and for a live-run audit. The key
        #: is redacted before it is stored: an audit record carrying the credential would
        #: defeat the point of keeping keys out of the config hash and the run manifest.
        self.requests: list[dict[str, Any]] = []
        #: Billable search requests, taken from each successful response's ``usage``. A 429
        #: or a 500 is not a search and is not counted as one.
        self.search_requests: int = 0
        #: ``publish_date`` per result URL, where it parsed -- usually ``None``, which is
        #: what the live API returns. Beside the documents rather than on them; see
        #: :func:`parallel_documents`.
        self.publish_dates: dict[str, date | None] = {}
        #: Each response's ``usage`` array, verbatim, for an audit.
        self.usage_entries: list[Any] = []

    # -- plumbing ----------------------------------------------------------

    @property
    def timeout_s(self) -> float:
        """Request timeout. Measured turbo latency is ~1s, so this is headroom, not a guess."""
        return float(self.config.parallel_timeout_s)

    @property
    def client(self) -> httpx.Client:
        """The HTTP client, created on first use so constructing a provider opens nothing."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
            self._owns_client = True
        return self._client

    def close(self) -> None:
        """Close the client only when this provider created it."""
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def spend_usd(self) -> float:
        """What this provider's searches have cost so far, at the published rate."""
        return search_cost_usd(self.search_requests, self.mode)

    def headers(self) -> dict[str, str]:
        """The two documented headers. ``x-api-key``, not ``Authorization: Bearer``."""
        return {"Content-Type": "application/json", "x-api-key": self._api_key}

    # -- the call ----------------------------------------------------------

    def gather(
        self,
        queries: Sequence[IntelQuery],
        config: IntelConfig | None = None,
        *,
        instruction: str = "",
    ) -> IntelGather:
        """Run one search for the whole plan and map what came back.

        Every failure is returned rather than raised -- no key, a non-200, a timeout, a
        malformed body, an empty result set -- because a scan must not abort because one
        finding's web research went wrong. A 429 and a 5xx are retried within the configured
        budget; a 4xx is not, because a request the API rejected will be rejected again.
        """
        config = config or self.config
        if not self._api_key:
            return IntelGather(
                errors=(
                    f"no Parallel API key in ${self.api_key_env}; live intel search cannot "
                    "run. Keys are issued at https://platform.parallel.ai/",
                ),
                provider=self.name,
            )
        if not any(str(query.text).strip() for query in queries):
            # Unlike the model-backed providers, an instruction alone is not a search plan:
            # the API's input is a query list and there is nothing here to invent one from.
            return IntelGather(
                errors=("empty search plan: the Parallel Search API needs at least one query",),
                provider=self.name,
            )

        body = search_body(queries, build_objective(queries, instruction), config)
        endpoint = str(config.parallel_endpoint or self.endpoint)
        timeout = float(config.parallel_timeout_s)
        attempts = max(1, int(config.max_retries) + 1)
        errors: list[str] = []
        payload: Any | None = None

        for attempt in range(attempts):
            self.requests.append(
                {
                    "url": endpoint,
                    "headers": {"Content-Type": "application/json", "x-api-key": "***"},
                    "json": body,
                }
            )
            self.limiter.wait()
            try:
                response = self.client.post(
                    endpoint, json=body, headers=self.headers(), timeout=timeout
                )
            except httpx.HTTPError as exc:
                # Timeouts and connection failures land here, and are retryable: the request
                # never reached a decision, so nothing is known about whether it would work.
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt + 1 >= attempts:
                    return IntelGather(errors=tuple(errors), provider=self.name)
                self._sleep(backoff_delay(attempt, config.retry_backoff_s, config.max_backoff_s))
                continue

            status = int(response.status_code)
            limited = rate_limited(status)
            if limited or status >= 500:
                errors.append(
                    f"rate_limited: HTTP {status}"
                    if limited
                    else f"api_status_{status}: {_body_hint(response)}"
                )
                if attempt + 1 >= attempts:
                    return IntelGather(errors=tuple(errors), provider=self.name)
                self._sleep(
                    backoff_delay(
                        attempt,
                        config.retry_backoff_s,
                        config.max_backoff_s,
                        retry_after_seconds(response.headers),
                    )
                )
                continue

            if status != 200:
                errors.append(f"api_status_{status}: {_body_hint(response)}")
                return IntelGather(errors=tuple(errors), provider=self.name)

            try:
                payload = response.json()
            except ValueError as exc:  # json.JSONDecodeError is a ValueError
                errors.append(f"malformed_response: {type(exc).__name__}: {exc}")
                return IntelGather(errors=tuple(errors), provider=self.name)
            break

        if payload is None:  # pragma: no cover - the loop always returns or breaks
            return IntelGather(errors=tuple(errors or ("no response",)), provider=self.name)

        gathered = self.parse(payload, config, queries)
        if errors:
            gathered = gathered.model_copy(update={"errors": (*errors, *gathered.errors)})
        return gathered

    # -- parsing -----------------------------------------------------------

    def parse(
        self,
        payload: Any,
        config: IntelConfig,
        queries: Sequence[IntelQuery] = (),
    ) -> IntelGather:
        """Turn one response body into documents, and bank what the call cost.

        ``narrative`` and ``citations`` stay empty, and that is the honest answer rather
        than a missing feature: there is no model in this phase to write prose or to quote a
        page, and :class:`~vulnpriority.core.models.IntelCitation` means "a span a model
        quoted". Excerpts are spans a search engine extracted, which is a different claim.
        """
        query_text = queries[0].text if queries else ""
        documents, publish_dates, errors = parallel_documents(
            payload, config, query_text=query_text, retrieved_at=utc_now()
        )
        self.publish_dates.update(publish_dates)
        if isinstance(payload, dict) and isinstance(payload.get("usage"), (list, tuple)):
            self.usage_entries.extend(payload["usage"])
        usage = usage_of(payload)
        self.search_requests += usage.web_search_requests
        return IntelGather(
            documents=tuple(documents[: config.max_documents]),
            narrative="",
            citations=(),
            usage=usage,
            errors=tuple(errors),
            model="",
            provider=self.name,
        )


def _body_hint(response: httpx.Response) -> str:
    """A short, whitespace-collapsed excerpt of an error body, for the recorded message."""
    try:
        return " ".join(response.text.split())[:200]
    except Exception:  # noqa: BLE001 - a diagnostic hint must never raise
        return ""  # pragma: no cover - defensive
