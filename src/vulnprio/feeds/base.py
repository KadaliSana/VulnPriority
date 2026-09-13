"""Shared feed machinery: as-of guards, fixture loading, HTTP plumbing, caching wrappers.

Three invariants are enforced here so that every concrete feed inherits them:

* **As-of.** Nothing dated after ``as_of`` may leave a feed. ``as_of_guard`` is the single
  predicate every feed uses, so the rule is implemented once.
* **Offline by default.** A :class:`FixtureFeed` has no HTTP client at all and raises
  :class:`~vulnprio.core.errors.OfflineViolationError` from any method that would fetch.
* **Reproducibility.** :class:`CachingFeed` makes a live run replayable by storing the
  serialised model next to the ``(feed, key, as_of)`` it answered.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Iterable, NoReturn, Sequence

import httpx

from vulnprio.core.config import PROJECT_ROOT, FeedsConfig
from vulnprio.core.enums import FeedMode, TrustTier
from vulnprio.core.errors import FeedUnavailableError, OfflineViolationError, TemporalLeakageError
from vulnprio.core.interfaces import (
    EpssFeed,
    ExploitFeed,
    FeedClient,
    KevFeed,
    NvdFeed,
    ReferenceFetcher,
)
from vulnprio.core.models import EpssRecord, ExploitEvidence, KevRecord, ReferenceDoc, VulnIntel
from vulnprio.feeds.cache import FileCacheStore

__all__ = [
    "as_of_guard",
    "require_not_future",
    "parse_feed_date",
    "parse_feed_datetime",
    "resolve_fixture_dir",
    "load_fixture_records",
    "clear_fixture_cache",
    "FixtureFeed",
    "LiveFeedBase",
    "CachingFeed",
    "CachingNvdFeed",
    "CachingEpssFeed",
    "CachingKevFeed",
    "CachingExploitFeed",
    "CachingReferenceFetcher",
    "wrap_with_cache",
]


# ---------------------------------------------------------------------------
# As-of guards
# ---------------------------------------------------------------------------


def as_of_guard(value_date: date | datetime | str | None, as_of: date) -> bool:
    """True when ``value_date`` is admissible at ``as_of``.

    Undated evidence is admissible (there is nothing to leak); anything dated strictly
    after the cut-off is not. Every feed filters through this one predicate so the
    time-ordered evaluation cannot be undermined by a per-feed variation.
    """
    parsed = parse_feed_date(value_date)
    if parsed is None:
        return True
    return parsed <= as_of


def require_not_future(value_date: date | datetime | str | None, as_of: date, what: str) -> None:
    """Raise :class:`TemporalLeakageError` when ``value_date`` post-dates ``as_of``."""
    if not as_of_guard(value_date, as_of):
        raise TemporalLeakageError(f"{what} is dated after the as-of cut-off {as_of.isoformat()}")


def parse_feed_date(value: date | datetime | str | None) -> date | None:
    """Best-effort date parsing for the assorted shapes upstream feeds emit."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    parsed = parse_feed_datetime(text)
    return parsed.date() if parsed is not None else None


def parse_feed_datetime(value: datetime | str | None) -> datetime | None:
    """Parse ISO-8601 with or without a zone, ``Z`` suffixes and trailing sub-second noise."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # CISA writes fractional seconds with four digits, which fromisoformat rejects pre-3.11
    # semantics for; normalise to six digits.
    if "." in text:
        head, _, tail = text.partition(".")
        digits = ""
        rest = ""
        for index, char in enumerate(tail):
            if char.isdigit():
                digits += char
            else:
                rest = tail[index:]
                break
        if digits:
            text = f"{head}.{digits[:6].ljust(6, '0')}{rest}"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromisoformat(text[:10])
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

#: Parsed fixture files, keyed by (path, mtime, size) so an edited file is re-read but a
#: repeated construction of the same feed is free.
_FIXTURE_CACHE: dict[tuple[str, int, int], list[dict[str, Any]]] = {}


def clear_fixture_cache() -> None:
    """Drop the process-wide fixture cache (used by tests that rewrite fixture files)."""
    _FIXTURE_CACHE.clear()


def resolve_fixture_dir(fixture_dir: str | Path | None) -> Path:
    """Absolute fixture directory; relative paths resolve against the project root."""
    base = Path(fixture_dir) if fixture_dir is not None else Path(FeedsConfig().fixture_dir)
    return base if base.is_absolute() else (PROJECT_ROOT / base)


def load_fixture_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a ``.json`` or ``.jsonl`` fixture exactly once per (path, mtime, size)."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FeedUnavailableError(f"fixture file not found: {resolved}")
    stat = resolved.stat()
    cache_key = (str(resolved.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _FIXTURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    text = resolved.read_text(encoding="utf-8")
    if resolved.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        records = list(data) if isinstance(data, list) else [data]
    _FIXTURE_CACHE[cache_key] = records
    return records


class FixtureFeed(FeedClient[Any]):
    """Offline feed served from a checked-in fixture file.

    Subclasses declare ``fixture_relpath``, say which key(s) a record answers to
    (``_key_of``) and turn the matching records into the feed's model (``_build``).
    Nothing in this class opens a socket, and the network-shaped entry points raise
    :class:`OfflineViolationError` so an accidental fetch is a loud failure.
    """

    fixture_relpath: ClassVar[str] = ""
    mode: FeedMode = FeedMode.OFFLINE

    def __init__(self, fixture_dir: str | Path | None = None, *, path: str | Path | None = None) -> None:
        self.fixture_dir = resolve_fixture_dir(fixture_dir)
        self.path = Path(path) if path is not None else self.fixture_dir / self.fixture_relpath
        self._index_cache: dict[str, list[dict[str, Any]]] | None = None

    # -- fixture access -----------------------------------------------------

    def records(self) -> list[dict[str, Any]]:
        """Every record in the fixture, after container unwrapping."""
        return self._extract(load_fixture_records(self.path))

    def _extract(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Hook for fixtures whose top level is an API envelope rather than a list."""
        return raw

    @staticmethod
    def _normalise_key(key: str) -> str:
        """CVE ids are case-insensitive; URL-keyed feeds override this."""
        return key.strip().upper()

    @abstractmethod
    def _key_of(self, record: dict[str, Any]) -> str | Sequence[str]:
        """Key(s) the record answers to."""

    @abstractmethod
    def _build(self, entries: tuple[dict[str, Any], ...], key: str, as_of: date) -> Any | None:
        """Turn the matching raw records into the feed's model, applying the as-of rule."""

    def index(self) -> dict[str, list[dict[str, Any]]]:
        """Key to records, built once per instance."""
        if self._index_cache is None:
            built: dict[str, list[dict[str, Any]]] = {}
            for record in self.records():
                keys = self._key_of(record)
                if isinstance(keys, str):
                    keys = (keys,)
                for key in keys:
                    if not key:
                        continue
                    built.setdefault(self._normalise_key(str(key)), []).append(record)
            self._index_cache = built
        return self._index_cache

    def keys(self) -> tuple[str, ...]:
        """Every key the fixture can answer, sorted for deterministic iteration."""
        return tuple(sorted(self.index()))

    def get(self, key: str, as_of: date) -> Any | None:
        normalised = self._normalise_key(str(key))
        entries = tuple(self.index().get(normalised, ()))
        return self._build(entries, normalised, as_of)

    # -- offline guards -----------------------------------------------------

    def _offline_violation(self, detail: str) -> NoReturn:
        raise OfflineViolationError(
            f"{self.name}: offline fixture feed attempted a network access ({detail})"
        )

    def fetch(self, url: str, **_: Any) -> NoReturn:
        """Never fetches. Present so that a mis-wired caller fails loudly, not silently."""
        self._offline_violation(f"fetch({url!r})")

    @property
    def client(self) -> NoReturn:
        """Offline feeds have no HTTP client."""
        self._offline_violation("client property accessed")

    def close(self) -> None:
        """No resources to release; defined so bundles can close feeds uniformly."""
        return None


# ---------------------------------------------------------------------------
# Live HTTP plumbing
# ---------------------------------------------------------------------------


class LiveFeedBase(FeedClient[Any]):
    """HTTP client management shared by every live feed.

    The client is injectable so tests drive real parsing code through
    ``httpx.MockTransport`` instead of the network, and the configured timeout is applied
    to every request rather than left at httpx's default.
    """

    mode: FeedMode = FeedMode.LIVE
    user_agent: ClassVar[str] = "vulnprio/0.1 (offline-by-default research framework)"

    def __init__(
        self,
        *,
        timeout_s: float = 20.0,
        client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self._client = client
        self._owns_client = client is None
        self._headers: dict[str, str] = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if headers:
            self._headers.update(headers)

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s, follow_redirects=True)
            self._owns_client = True
        return self._client

    def _request_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        missing_status: Iterable[int] = (404,),
    ) -> Any | None:
        """GET and decode JSON. ``None`` for the statuses that mean 'no such record'."""
        try:
            response = self.client.get(url, params=params, headers=self._headers, timeout=self.timeout_s)
            if response.status_code in tuple(missing_status):
                return None
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise FeedUnavailableError(f"{self.name}: request to {url} failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise FeedUnavailableError(f"{self.name}: {url} did not return JSON: {exc}") from exc

    def _request_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        missing_status: Iterable[int] = (404,),
    ) -> str | None:
        """GET and decode text (used for the Exploit-DB CSV index)."""
        try:
            response = self.client.get(url, params=params, headers=self._headers, timeout=self.timeout_s)
            if response.status_code in tuple(missing_status):
                return None
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as exc:
            raise FeedUnavailableError(f"{self.name}: request to {url} failed: {exc}") from exc

    def close(self) -> None:
        """Close the client only when this feed created it."""
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "LiveFeedBase":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Caching wrapper
# ---------------------------------------------------------------------------


class CachingFeed(FeedClient[Any]):
    """Composes a live feed with a :class:`FileCacheStore`.

    A cached ``None`` is a real answer ("upstream has nothing for this key at this date"),
    so payloads are stored inside an envelope; without it a negative result would be
    re-fetched forever.
    """

    def __init__(
        self,
        inner: FeedClient[Any],
        store: FileCacheStore,
        *,
        feed_name: str | None = None,
    ) -> None:
        self.inner = inner
        self.store = store
        self.name = feed_name or getattr(inner, "name", "feed")
        self.tier: TrustTier = getattr(inner, "tier", TrustTier.CURATED_FEED)
        self.mode = FeedMode.LIVE_WITH_CACHE

    @abstractmethod
    def _encode(self, value: Any) -> Any:
        """JSON-ready form of the wrapped feed's return value."""

    @abstractmethod
    def _decode(self, payload: Any) -> Any:
        """Inverse of :meth:`_encode`."""

    def get(self, key: str, as_of: date) -> Any | None:
        envelope = self.store.get(self.name, key, as_of)
        if isinstance(envelope, dict) and "value" in envelope:
            payload = envelope["value"]
            return None if payload is None else self._decode(payload)
        value = self.inner.get(key, as_of)
        self.store.put(self.name, key, as_of, {"value": None if value is None else self._encode(value)})
        return value

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


class CachingNvdFeed(CachingFeed, NvdFeed):
    """Cached NVD feed."""

    def _encode(self, value: VulnIntel) -> Any:
        return value.model_dump(mode="json")

    def _decode(self, payload: Any) -> VulnIntel:
        return VulnIntel.model_validate(payload)


class CachingEpssFeed(CachingFeed, EpssFeed):
    """Cached EPSS feed."""

    def _encode(self, value: EpssRecord) -> Any:
        return value.model_dump(mode="json")

    def _decode(self, payload: Any) -> EpssRecord:
        return EpssRecord.model_validate(payload)


class CachingKevFeed(CachingFeed, KevFeed):
    """Cached CISA KEV feed."""

    def _encode(self, value: KevRecord) -> Any:
        return value.model_dump(mode="json")

    def _decode(self, payload: Any) -> KevRecord:
        return KevRecord.model_validate(payload)


class CachingExploitFeed(CachingFeed, ExploitFeed):
    """Cached exploit evidence feed (the value is a tuple, not a single model)."""

    def _encode(self, value: tuple[ExploitEvidence, ...]) -> Any:
        return [item.model_dump(mode="json") for item in value]

    def _decode(self, payload: Any) -> tuple[ExploitEvidence, ...]:
        return tuple(ExploitEvidence.model_validate(item) for item in payload)


class CachingReferenceFetcher(CachingFeed, ReferenceFetcher):
    """Cached reference-page fetcher. Content stays untrusted after a round trip."""

    tier = TrustTier.REFERENCE_PAGE

    def _encode(self, value: ReferenceDoc) -> Any:
        return value.model_dump(mode="json")

    def _decode(self, payload: Any) -> ReferenceDoc:
        return ReferenceDoc.model_validate(payload)


#: Which caching wrapper belongs to which interface.
_CACHING_BY_INTERFACE: tuple[tuple[type, type[CachingFeed]], ...] = (
    (NvdFeed, CachingNvdFeed),
    (EpssFeed, CachingEpssFeed),
    (KevFeed, CachingKevFeed),
    (ExploitFeed, CachingExploitFeed),
    (ReferenceFetcher, CachingReferenceFetcher),
)


def wrap_with_cache(inner: FeedClient[Any], store: FileCacheStore) -> CachingFeed:
    """Wrap ``inner`` in the caching class that matches its feed interface."""
    for interface, wrapper in _CACHING_BY_INTERFACE:
        if isinstance(inner, interface):
            return wrapper(inner, store)
    raise FeedUnavailableError(f"no caching wrapper for feed {type(inner).__name__}")
