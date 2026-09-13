"""On-disk cache for live feed responses.

Why this exists: live feeds are rate limited and non-deterministic, but the evaluation
protocol needs the *same* answer every time a run is replayed. Caching by
``(feed, key, as_of)`` gives that, and storing a content hash alongside each entry means
a truncated or hand-edited cache file is treated as a miss rather than silently trusted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from vulnprio.core.hashing import canonical_json, sha256_text, stable_id

__all__ = ["CacheEntry", "FileCacheStore", "utc_now"]


def utc_now() -> datetime:
    """Timezone-aware current time. Injectable everywhere so tests stay deterministic."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CacheEntry:
    """One cached feed response plus the metadata needed to decide whether to trust it."""

    feed: str
    key: str
    as_of: date
    fetched_at: datetime
    content_hash: str
    payload: Any

    def is_fresh(self, now: datetime, ttl_days: int) -> bool:
        """Fresh when the entry is younger than the TTL. ``ttl_days <= 0`` disables reuse."""
        if ttl_days <= 0:
            return False
        age_seconds = (_aware(now) - _aware(self.fetched_at)).total_seconds()
        return 0 <= age_seconds < ttl_days * 86400.0

    def to_json(self) -> dict[str, Any]:
        return {
            "feed": self.feed,
            "key": self.key,
            "as_of": self.as_of.isoformat(),
            "fetched_at": _aware(self.fetched_at).isoformat(),
            "content_hash": self.content_hash,
            "payload": self.payload,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CacheEntry":
        return cls(
            feed=str(data["feed"]),
            key=str(data["key"]),
            as_of=date.fromisoformat(str(data["as_of"])),
            fetched_at=datetime.fromisoformat(str(data["fetched_at"])),
            content_hash=str(data["content_hash"]),
            payload=data.get("payload"),
        )


def _aware(value: datetime) -> datetime:
    """Naive datetimes are interpreted as UTC so age arithmetic never raises."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def content_hash_of(payload: Any) -> str:
    """Hash of the cached payload; mismatch on read means the file was tampered with."""
    return sha256_text(canonical_json(payload))


class FileCacheStore:
    """JSON-on-disk cache keyed by ``(feed, key, as_of)``.

    Layout is ``<cache_dir>/<feed>/<hashed key>.json``. The key is hashed because feed keys
    are CVE ids *and* URLs, and URLs are not valid Windows filenames.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        ttl_days: int = 7,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl_days = int(ttl_days)
        self._now = now or utc_now

    # -- addressing ---------------------------------------------------------

    @staticmethod
    def entry_key(feed: str, key: str, as_of: date) -> str:
        """Deterministic, filesystem-safe identifier for one cache slot."""
        return stable_id("fc", feed, key, as_of.isoformat())

    def path_for(self, feed: str, key: str, as_of: date) -> Path:
        return self.cache_dir / feed / f"{self.entry_key(feed, key, as_of)}.json"

    # -- reads --------------------------------------------------------------

    def read_entry(self, feed: str, key: str, as_of: date) -> CacheEntry | None:
        """Raw entry, ignoring the TTL. Returns ``None`` for missing or corrupt files."""
        path = self.path_for(feed, key, as_of)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry = CacheEntry.from_json(data)
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if entry.content_hash != content_hash_of(entry.payload):
            return None
        if entry.feed != feed or entry.key != key or entry.as_of != as_of:
            return None
        return entry

    def get(self, feed: str, key: str, as_of: date) -> Any | None:
        """Cached payload when a fresh entry exists, otherwise ``None`` (a miss)."""
        entry = self.read_entry(feed, key, as_of)
        if entry is None:
            return None
        if not entry.is_fresh(self._now(), self.ttl_days):
            return None
        return entry.payload

    # -- writes -------------------------------------------------------------

    def put(self, feed: str, key: str, as_of: date, payload: Any) -> Path:
        """Store ``payload`` and return the file it was written to."""
        path = self.path_for(feed, key, as_of)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = CacheEntry(
            feed=feed,
            key=key,
            as_of=as_of,
            fetched_at=_aware(self._now()),
            content_hash=content_hash_of(payload),
            payload=payload,
        )
        path.write_text(json.dumps(entry.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def invalidate(self, feed: str, key: str, as_of: date) -> bool:
        """Delete one entry. True when something was removed."""
        path = self.path_for(feed, key, as_of)
        if path.is_file():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Delete every entry; returns how many files were removed."""
        removed = 0
        for path in self.iter_files():
            path.unlink()
            removed += 1
        return removed

    def iter_files(self) -> Iterator[Path]:
        if not self.cache_dir.is_dir():
            return iter(())
        return iter(sorted(self.cache_dir.rglob("*.json")))

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_files())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FileCacheStore(cache_dir={self.cache_dir!s}, ttl_days={self.ttl_days})"
