"""The free half of the intelligence layer: Google Search grounding.

:mod:`vulnprio.intel` exists to search the internet for exploit material the curated feeds
do not link. Doing that with the Anthropic web-search server tool works and costs money per
finding. Gemini has Google Search grounding built in, ``gemini-3.5-flash-lite`` carries a free
grounded-request quota, and that combination is the closest no-cost equivalent there is --
which makes this module, not the backend, the thing that decides whether the framework's
central claim is reachable by someone who cannot spend anything.

**Grounding and structured output still do not compose, so the two phases stay.** Checked
rather than assumed, and the answer is nuanced enough to write down:

* On ``generateContent`` with the 2.5 model family -- what this provider uses -- combining
  ``tools=[google_search]`` with ``response_mime_type="application/json"`` and a
  ``response_schema`` is not documented as supported anywhere in Google's current
  documentation, and the historical behaviour is a 400 rejecting the combination. Treat it
  as incompatible.
* Google *has* since documented combining structured output with built-in tools including
  Search grounding -- but only for the Gemini 3 series, only through the newer
  ``interactions`` API, and only as a preview feature.

So the split is still forced for the path taken here, exactly as it is on the Anthropic
path. It would be kept anyway: phase one gathers and phase two judges, and a call that
reads the internet should not be the call that emits the numbers. See
:mod:`vulnprio.intel.anthropic_search` and DESIGN 6.5.

**Everything retrieved is ``Provenance.REFERENCE_PAGE``.** Documents are built through
:func:`~vulnprio.intel.provider.make_document`, which is the only place that provenance is
set, and :class:`~vulnprio.core.models.IntelDocument` rejects anything else outright. There
is no fast path for grounded results: Google surfacing a page does not make the page's
author trustworthy, and the whole sandbox applies to it identically.

**Two shape details that are easy to get wrong and are asserted in the suite:**

* ``GoogleSearch`` has no ``allowed_domains``. It accepts ``exclude_domains`` only, so
  :attr:`~vulnprio.core.config.IntelConfig.blocked_domains` maps to the API while
  ``allowed_domains`` has to be enforced here, after the fact, by discarding documents
  from hosts that are not on the list. A provider that quietly ignored the allowlist would
  be reading the whole internet while the config said otherwise.
* ``grounding_chunks[i].web.uri`` is a Vertex AI Search redirect, not the publisher's URL.
  ``web.domain`` carries the real host, so that is what classification and the allowlist
  check are done against.
"""

from __future__ import annotations

import os
from typing import Any, Sequence
from urllib.parse import urlsplit

from vulnprio.core.errors import ConfigError
from vulnprio.intel.models import (
    IntelCitation,
    IntelConfig,
    IntelGather,
    IntelQuery,
    IntelUsage,
    utc_now,
)
from vulnprio.intel.provider import BaseSearchProvider, make_document
from vulnprio.intel.queries import PHASE1_SYSTEM, classify_source
from vulnprio.llm.gemini_backend import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_KEY_ENV,
    is_rate_limited,
    resolve_key_env,
    resolve_model,
    retry_delay_of,
)
from vulnprio.llm.openai_compatible import RateLimiter, backoff_delay

__all__ = [
    "GeminiSearchProvider",
    "build_gemini_client",
    "grounding_documents",
    "host_allowed",
    "search_tool",
    "usage_of",
]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute or key access. Responses arrive as SDK objects or as plain dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def build_gemini_client(api_key: str) -> Any:
    """Construct the SDK client. Imported lazily so the package imports without the SDK."""
    try:
        from google import genai  # noqa: PLC0415 - lazy: the package is an optional extra
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigError(
            "Gemini grounded search needs the 'google-genai' package; "
            "install vulnprio[gemini] or use mode='offline'"
        ) from exc
    return genai.Client(api_key=api_key)


def search_tool(config: IntelConfig) -> Any:
    """The Google Search grounding tool, domain-excluded where the config asks for it.

    Built as ``types.Tool(google_search=types.GoogleSearch())`` when the SDK is importable
    and as the equivalent dict otherwise, so a request shape can be asserted without it.

    ``exclude_domains`` is the only domain control the API offers. ``allowed_domains`` is
    the framework's preferred direction and has no counterpart, so it is enforced in
    :func:`host_allowed` on the results instead.
    """
    excluded = list(config.blocked_domains or ())
    try:  # pragma: no cover - exercised whenever the SDK is installed
        from google.genai import types  # noqa: PLC0415

        search = types.GoogleSearch(exclude_domains=excluded) if excluded else types.GoogleSearch()
        return types.Tool(google_search=search)
    except Exception:  # noqa: BLE001 - absence is never fatal
        return {"google_search": ({"exclude_domains": excluded} if excluded else {})}


def host_allowed(url: str, domain: str | None, config: IntelConfig) -> bool:
    """Whether a grounded result may be kept, given the config's domain lists.

    The allowlist is applied here because the API cannot apply it. An empty allowlist means
    "no restriction", matching :class:`IntelConfig`, which refuses to hold both lists at
    once. Matching is on suffix so ``github.com`` admits ``gist.github.com``.
    """
    host = (domain or urlsplit(url).hostname or "").lower().strip()
    if not host:
        return False
    if config.blocked_domains and any(
        host == blocked.lower() or host.endswith("." + blocked.lower())
        for blocked in config.blocked_domains
    ):
        return False
    if not config.allowed_domains:
        return True
    return any(
        host == allowed.lower() or host.endswith("." + allowed.lower())
        for allowed in config.allowed_domains
    )


def usage_of(response: Any) -> IntelUsage:
    """Token and grounded-search accounting for one response, defaulting to zero.

    ``web_search_queries`` is the list of queries grounding actually issued, so its length
    is the grounded-request count -- which is the number that is metered against the free
    daily quota, and therefore the one worth recording.
    """
    usage = _get(response, "usage_metadata")
    queries: list[Any] = []
    for candidate in _get(response, "candidates") or ():
        metadata = _get(candidate, "grounding_metadata")
        queries.extend(_get(metadata, "web_search_queries") or ())
    return IntelUsage(
        input_tokens=max(0, int(_get(usage, "prompt_token_count", 0) or 0)),
        output_tokens=max(0, int(_get(usage, "candidates_token_count", 0) or 0)),
        cache_read_tokens=max(0, int(_get(usage, "cached_content_token_count", 0) or 0)),
        web_search_requests=len(queries),
        calls=1,
    )


def grounding_documents(
    response: Any,
    config: IntelConfig,
    query_text: str = "",
    retrieved_at: Any | None = None,
) -> tuple[list[Any], list[IntelCitation], list[str]]:
    """Grounding metadata -> (documents, citations, errors). Never raises.

    A response with no grounding metadata, empty chunks, or chunks carrying no usable URL
    is an ordinary empty result. That case is common and not exceptional: the model may
    answer without searching, the quota may be spent, or the query may genuinely find
    nothing, and all three must produce "no documents" rather than an exception that aborts
    a scan.

    Grounding returns titles and citation spans, not page bodies. The text of a document is
    therefore the segments the model actually quoted from it, falling back to the title --
    which is honest about what was retrieved rather than padding it out.
    """
    stamp = retrieved_at or utc_now()
    chunks: list[dict[str, Any]] = []
    citations: list[IntelCitation] = []
    errors: list[str] = []
    quoted: dict[int, list[str]] = {}

    for candidate in _get(response, "candidates") or ():
        metadata = _get(candidate, "grounding_metadata")
        if metadata is None:
            continue
        base = len(chunks)
        for chunk in _get(metadata, "grounding_chunks") or ():
            web = _get(chunk, "web") or _get(chunk, "retrieved_context")
            url = str(_get(web, "uri", "") or "")
            domain = _get(web, "domain")
            title = _get(web, "title")
            chunks.append(
                {
                    "url": url,
                    "domain": str(domain) if domain else None,
                    "title": str(title) if title else None,
                    "text": str(_get(web, "text", "") or ""),
                }
            )
        for support in _get(metadata, "grounding_supports") or ():
            segment = _get(support, "segment")
            text = str(_get(segment, "text", "") or "").strip()
            if not text:
                continue
            for index in _get(support, "grounding_chunk_indices") or ():
                try:
                    quoted.setdefault(base + int(index), []).append(text)
                except (TypeError, ValueError):
                    continue

    documents: list[Any] = []
    seen: set[str] = set()
    for index, chunk in enumerate(chunks):
        url = chunk["url"]
        if not url:
            errors.append("grounding_chunk_without_uri")
            continue
        if not host_allowed(url, chunk["domain"], config):
            errors.append(f"domain_not_allowed: {chunk['domain'] or urlsplit(url).hostname or '?'}")
            continue
        if url in seen:
            continue
        seen.add(url)
        text = chunk["text"] or "\n".join(quoted.get(index, ())) or (chunk["title"] or "")
        if not text:
            continue
        documents.append(
            make_document(
                url,
                text,
                title=chunk["title"],
                retrieved_at=stamp,
                # Classified from the publisher's domain, because the chunk URL is a
                # Vertex AI Search redirect whose host says nothing about the source.
                source_kind=classify_source(
                    f"https://{chunk['domain']}" if chunk["domain"] else url, chunk["title"]
                ),
                relevance=0.6 if quoted.get(index) else 0.5,
                query_text=query_text,
                char_budget=config.snippet_char_budget,
            )
        )
        for span in quoted.get(index, ()):
            citations.append(
                IntelCitation(url=url[:500], cited_text=span[:400], title=chunk["title"])
            )

    return documents, citations, errors


class GeminiSearchProvider(BaseSearchProvider):
    """One grounded research call per finding, over Google Search."""

    name = "gemini_search"

    def __init__(
        self,
        config: IntelConfig | None = None,
        client: Any | None = None,
        *,
        api_key: str | None = None,
        sleep: Any | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        """``client`` is injectable so the test suite can assert request shape with no key.

        A missing key is *not* raised here, matching
        :class:`~vulnprio.intel.anthropic_search.AnthropicSearchProvider`: the intelligence
        layer degrades to an empty result with a recorded reason rather than failing a
        scan, so "no key" is an unavailable provider, not an exception.
        """
        self.config = config or IntelConfig()
        self.model_id = resolve_model(self.config.model, DEFAULT_GEMINI_MODEL)
        self.api_key_env = resolve_key_env(self.config.api_key_env, DEFAULT_KEY_ENV)
        self._api_key = api_key if api_key is not None else os.environ.get(self.api_key_env, "")
        self._sleep = sleep or (lambda _seconds: None)
        self.limiter = limiter if limiter is not None else RateLimiter(
            self.config.requests_per_minute, sleep=self._sleep
        )
        #: Every request this provider issued, for tests and for a live-run audit.
        self.requests: list[dict[str, Any]] = []
        if client is not None:
            self.client = client
            return
        self.client = build_gemini_client(self._api_key) if self._api_key else None

    def available(self) -> bool:
        return self.client is not None

    # -- the call ----------------------------------------------------------

    def request_kwargs(self, config: IntelConfig, instruction: str) -> dict[str, Any]:
        """The phase-1 request.

        Note what is absent: no ``response_mime_type`` and no ``response_schema``. Search
        grounding and structured output cannot be combined on this path (see the module
        docstring), so phase one asks for prose with citations and phase two, which never
        touches the network, is where a claim becomes a number.

        ``max_output_tokens`` is a spend ceiling as much as a safety one; on the free tier
        the binding limit is usually tokens per minute rather than requests per day.
        """
        return {
            "model": self.model_id,
            "contents": instruction,
            "config": {
                "system_instruction": PHASE1_SYSTEM,
                "max_output_tokens": int(config.max_tokens),
                "tools": [search_tool(config)],
            },
        }

    def gather(
        self,
        queries: Sequence[IntelQuery],
        config: IntelConfig | None = None,
        *,
        instruction: str = "",
    ) -> IntelGather:
        """Run the grounded research call and parse what came back.

        Every failure is returned rather than raised, including a rate limit: a scan must
        not abort because one finding's web research hit a free-tier quota.
        """
        config = config or self.config
        if self.client is None:
            return IntelGather(
                errors=(
                    f"no Gemini API key in ${self.api_key_env}; live intel search cannot "
                    "run. A free key is available at https://aistudio.google.com/apikey",
                ),
                provider=self.name,
            )
        if not queries and not instruction:
            return IntelGather(errors=("empty search plan",), provider=self.name)

        body = instruction or "\n".join(f"- {query.text}" for query in queries)
        kwargs = self.request_kwargs(config, body)
        attempts = max(1, config.max_retries + 1)
        errors: list[str] = []
        response: Any | None = None

        for attempt in range(attempts):
            self.requests.append(kwargs)
            self.limiter.wait()
            try:
                response = self.client.models.generate_content(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - classified immediately below
                limited = is_rate_limited(exc)
                errors.append(
                    f"rate_limited: {exc}" if limited else f"{type(exc).__name__}: {exc}"
                )
                if not limited or attempt + 1 >= attempts:
                    return IntelGather(errors=tuple(errors), provider=self.name)
                self._sleep(
                    backoff_delay(
                        attempt, config.retry_backoff_s, config.max_backoff_s, retry_delay_of(exc)
                    )
                )

        if response is None:  # pragma: no cover - the loop always returns or breaks
            return IntelGather(errors=tuple(errors or ("no response",)), provider=self.name)

        gathered = self.parse(response, config, queries)
        if errors:
            gathered = gathered.model_copy(update={"errors": (*errors, *gathered.errors)})
        return gathered

    # -- parsing -----------------------------------------------------------

    def parse(
        self,
        response: Any,
        config: IntelConfig,
        queries: Sequence[IntelQuery] = (),
    ) -> IntelGather:
        """Turn a grounded response into documents, prose and citations."""
        query_text = queries[0].text if queries else ""
        narrative_parts: list[str] = []
        for candidate in _get(response, "candidates") or ():
            content = _get(candidate, "content")
            for part in _get(content, "parts") or ():
                text = _get(part, "text")
                if isinstance(text, str) and text.strip():
                    narrative_parts.append(text)
        if not narrative_parts:
            text = _get(response, "text")
            if isinstance(text, str) and text.strip():
                narrative_parts.append(text)

        documents, citations, errors = grounding_documents(
            response, config, query_text=query_text, retrieved_at=utc_now()
        )
        return IntelGather(
            documents=tuple(documents[: config.max_documents]),
            narrative="\n".join(narrative_parts).strip(),
            citations=tuple(citations),
            usage=usage_of(response),
            errors=tuple(errors),
            model=str(_get(response, "model_version", "") or self.model_id),
            provider=self.name,
        )
