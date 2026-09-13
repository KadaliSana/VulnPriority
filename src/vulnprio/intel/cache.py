"""On-disk cache of gathered intelligence.

A warm cache is what makes this layer affordable and what makes a run containing it
reproducible. The key covers everything that determines what would be asked and read --
the CVE, the finding, the exact query set, the as-of date, the model and the phase-2
prompt hash -- and nothing about what came back. A cache hit therefore means "this exact
question was asked before", and re-running an evaluation costs nothing and returns the
same numbers.

The stored payload is :meth:`IntelResult.canonical_json`, which excludes ``cache_hit``.
That one exclusion is deliberate: whether a particular copy of a result arrived from disk
is a fact about the retrieval, not about the intelligence, and including it would make a
warm result differ from the cold one it is supposed to reproduce exactly.

Entries fail closed. A file written by an older schema no longer validates and is treated
as a miss rather than being coerced into the current shape.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from vulnprio.core.config import PROJECT_ROOT
from vulnprio.core.hashing import config_hash, sha256_text
from vulnprio.intel.models import IntelQuery, IntelResult

__all__ = ["IntelCache", "intel_cache_key"]


def intel_cache_key(
    *,
    finding_id: str,
    cve_id: str | None,
    queries: Sequence[IntelQuery],
    as_of: date,
    model: str,
    prompt_hash: str = "",
    provider: str = "",
    schema_name: str = "",
) -> str:
    """Content hash of the question being asked.

    Queries are sorted by their normalised key so that a reordered but identical plan is
    the same question. Nothing derived from a retrieved page participates, because a key
    that depended on the answer could never be looked up before getting one.
    """
    return config_hash(
        {
            "finding_id": finding_id,
            "cve_id": cve_id or "",
            "queries": sorted(query.key for query in queries),
            "as_of": as_of.isoformat(),
            "model": model,
            "prompt_hash": prompt_hash,
            "provider": provider,
            "schema": schema_name,
        }
    )


class IntelCache:
    """One JSON file per cached result under ``cache_dir``."""

    def __init__(self, cache_dir: str | Path, enabled: bool = True) -> None:
        directory = Path(cache_dir)
        self.cache_dir = directory if directory.is_absolute() else (PROJECT_ROOT / directory)
        self.enabled = bool(enabled)

    def path_for(self, key: str) -> Path:
        """Hashed filename, so an odd key can never escape the cache directory."""
        return self.cache_dir / f"intel_{sha256_text(key)[:32]}.json"

    def load(self, key: str) -> IntelResult | None:
        """Validated cached result with ``cache_hit`` set, or ``None`` on a miss."""
        if not self.enabled or not key:
            return None
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        payload = entry.get("result") if isinstance(entry, dict) else None
        if payload is None:
            return None
        try:
            result = IntelResult.model_validate(payload)
        except ValidationError:
            return None
        return result.model_copy(update={"cache_hit": True})

    def store(self, key: str, result: IntelResult) -> Path | None:
        """Write one entry. Returns the path written, or ``None`` when caching is off."""
        if not self.enabled or not key:
            return None
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"key": key, "result": json.loads(result.canonical_json())},
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    def clear(self) -> int:
        """Delete every entry; returns how many files were removed."""
        if not self.cache_dir.is_dir():
            return 0
        removed = 0
        for path in self.cache_dir.glob("intel_*.json"):
            path.unlink()
            removed += 1
        return removed
