"""The search-provider contract.

A provider answers one question: given a search plan, what did the internet say? It
returns :class:`~vulnprio.intel.models.IntelDocument` objects whose text is untrusted by
construction, and -- when the provider is an agentic one -- the researcher prose and
citations that came out of the same call.

Two methods matter. ``search`` is the narrow contract every provider satisfies and the one
callers should code against. ``gather`` is the wider one, returning the narrative and
citations alongside the documents, because for an agentic provider those are by-products
of the same single call and throwing them away would mean paying for them twice.

No provider sanitises anything. Sanitisation belongs to
:mod:`vulnprio.sandbox.pipeline` and doing it here would hide from the injection detector
what the page actually said -- the same reasoning that keeps
:mod:`vulnprio.feeds.references` from doing it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol, Sequence, runtime_checkable

from vulnprio.core.enums import Provenance
from vulnprio.core.models import UntrustedText
from vulnprio.intel.models import (
    IntelConfig,
    IntelDocument,
    IntelGather,
    IntelQuery,
    IntelSourceKind,
    utc_now,
)
from vulnprio.intel.queries import classify_source

__all__ = [
    "SearchProvider",
    "BaseSearchProvider",
    "NullSearchProvider",
    "build_search_provider",
    "make_document",
]


@runtime_checkable
class SearchProvider(Protocol):
    """Structural type of a search provider."""

    name: str

    def available(self) -> bool:
        """True when this provider can actually run: fixtures present, or a key set."""

    def search(
        self, queries: Sequence[IntelQuery], config: IntelConfig
    ) -> list[IntelDocument]:
        """Run the plan and return what was found, most relevant first."""


def make_document(
    url: str,
    text: str,
    *,
    title: str | None = None,
    retrieved_at: datetime | None = None,
    source_kind: IntelSourceKind | None = None,
    relevance: float = 0.5,
    query_text: str = "",
    char_budget: int = 4000,
) -> IntelDocument:
    """Build an :class:`IntelDocument`, wrapping the text as untrusted reference content.

    The single place a retrieved page becomes an object, so the provenance cannot be got
    wrong in one provider and right in another. The character budget is applied here as
    well as at the API boundary: a provider that ignored ``max_content_tokens`` would
    otherwise be able to put an unbounded page into a prompt.
    """
    body = str(text or "")
    if char_budget > 0 and len(body) > char_budget:
        body = body[:char_budget].rstrip()
    stamp = retrieved_at or utc_now()
    return IntelDocument(
        url=str(url)[:500],
        title=(str(title)[:300] if title else None),
        snippet=UntrustedText(
            text=body,
            provenance=Provenance.REFERENCE_PAGE,
            source_url=str(url)[:500],
            fetched_at=stamp,
        ),
        retrieved_at=stamp,
        source_kind=source_kind or classify_source(url, title),
        relevance=min(1.0, max(0.0, float(relevance))),
        query_text=str(query_text)[:300],
    )


class BaseSearchProvider(ABC):
    """Shared implementation: ``search`` is ``gather`` with the extras discarded."""

    name: str = "base"

    def available(self) -> bool:
        return True

    @abstractmethod
    def gather(
        self,
        queries: Sequence[IntelQuery],
        config: IntelConfig,
        *,
        instruction: str = "",
    ) -> IntelGather:
        """Run the plan, returning documents plus any narrative and citations."""

    def search(
        self, queries: Sequence[IntelQuery], config: IntelConfig
    ) -> list[IntelDocument]:
        return list(self.gather(queries, config).documents)


class NullSearchProvider(BaseSearchProvider):
    """Finds nothing, successfully.

    Used when intel is enabled but no usable provider could be constructed. It exists so
    that "we could not search" is an ordinary empty result with a recorded reason rather
    than a branch every caller has to remember to write.
    """

    name = "null"

    def __init__(self, reason: str = "no search provider available") -> None:
        self.reason = str(reason)

    def available(self) -> bool:
        return False

    def gather(
        self,
        queries: Sequence[IntelQuery],
        config: IntelConfig,
        *,
        instruction: str = "",
    ) -> IntelGather:
        return IntelGather(errors=(self.reason,), provider=self.name)


def build_search_provider(config: IntelConfig) -> BaseSearchProvider:
    """The provider named by ``config``. The single place that choice is made.

    Offline always wins and is checked first, so a run configured offline cannot reach a
    live provider whatever else is set -- the offline-by-default claim is structural here
    rather than a matter of remembering to check.

    Imports are local because two of the three live providers depend on an optional SDK,
    and a missing one is a degraded run, never a fatal import. (Parallel needs only
    ``httpx``, which is already a hard dependency; its import is kept lazy for symmetry, so
    adding a fourth provider does not require noticing that this one branch differs.)
    Anything that goes wrong becomes a :class:`NullSearchProvider` carrying the reason, so
    "we could not search" stays an ordinary empty result with an explanation attached.

    No live provider is a second path into the score. All of them return
    :class:`~vulnprio.core.models.IntelDocument` objects built by :func:`make_document`, so
    everything retrieved is ``REFERENCE_PAGE`` tier and goes through the same sandbox, the
    same influence budget and the same phase-2 ``GuardedBackend`` as any other page.

    Anthropic is the fallback rather than a named branch because it is the field default;
    ``gemini`` and ``parallel`` are chosen, so they are asked for by name.
    """
    if config.mode == "offline":
        from vulnprio.intel.offline import FixtureSearchProvider  # noqa: PLC0415

        return FixtureSearchProvider(config.fixture_path)

    if config.search_provider == "parallel":
        try:
            from vulnprio.intel.parallel_search import (  # noqa: PLC0415 - kept lazy for symmetry
                ParallelSearchProvider,
            )
        except Exception as exc:  # noqa: BLE001 - absence is a degraded run, never fatal
            return NullSearchProvider(f"Parallel search unavailable: {exc}")
        return ParallelSearchProvider(config)

    if config.search_provider == "gemini":
        try:
            from vulnprio.intel.gemini_search import (  # noqa: PLC0415 - optional SDK
                GeminiSearchProvider,
            )
        except Exception as exc:  # noqa: BLE001 - absence is a degraded run, never fatal
            return NullSearchProvider(f"Gemini grounded search unavailable: {exc}")
        return GeminiSearchProvider(config)

    try:
        from vulnprio.intel.anthropic_search import (  # noqa: PLC0415 - optional SDK
            AnthropicSearchProvider,
        )
    except Exception as exc:  # noqa: BLE001 - absence is a degraded run, never fatal
        return NullSearchProvider(f"live search unavailable: {exc}")
    return AnthropicSearchProvider(config)
