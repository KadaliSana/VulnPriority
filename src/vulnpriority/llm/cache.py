"""On-disk cache of raw model responses.

A cached run is a reproducible run: with the cache warm the pipeline produces byte
identical assessments with no network and no key, which is what makes an ablation over
eight cells and three seeds affordable and what lets a reviewer re-run a published
result. The key is the prompt hash plus the schema name, because those two together
determine what was asked; nothing about the response participates in the key.

Only raw JSON is stored. Validation happens on read, so a cache written by an older
schema version fails closed (treated as a miss) rather than injecting a stale shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from vulnpriority.core.hashing import sha256_text
from vulnpriority.core.models import LLMAudit

__all__ = ["LLMResponseCache", "mark_cached"]


def mark_cached(audit: LLMAudit) -> LLMAudit:
    """Copy of ``audit`` with ``cached`` set, for results served from disk."""
    return audit.model_copy(update={"cached": True})


class LLMResponseCache:
    """JSON-per-entry cache under ``cache_dir``.

    One file per (prompt_hash, schema) pair keeps entries independently inspectable and
    independently deletable, which matters when debugging a single bad assessment.
    """

    def __init__(self, cache_dir: str | Path, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

    # -- keys and paths ----------------------------------------------------

    @staticmethod
    def key(prompt_hash: str, schema_name: str) -> str:
        """Stable cache key. Hashed so an odd prompt hash can never escape the directory."""
        return sha256_text(f"{schema_name}|{prompt_hash}")[:32]

    def path_for(self, prompt_hash: str, schema_name: str) -> Path:
        return self.cache_dir / f"{schema_name}_{self.key(prompt_hash, schema_name)}.json"

    # -- read / write ------------------------------------------------------

    def load_raw(self, prompt_hash: str, schema_name: str) -> str | None:
        """Raw response text for a prompt, or ``None`` on a miss or unreadable entry."""
        if not self.enabled or not prompt_hash:
            return None
        path = self.path_for(prompt_hash, schema_name)
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        raw = entry.get("raw_text")
        return raw if isinstance(raw, str) else None

    def load(self, prompt_hash: str, schema: type[BaseModel]) -> BaseModel | None:
        """Validated cached answer, or ``None`` if absent or no longer schema-valid."""
        raw = self.load_raw(prompt_hash, schema.__name__)
        if raw is None:
            return None
        try:
            return schema.model_validate_json(raw)
        except ValidationError:
            return None

    def store(self, prompt_hash: str, schema_name: str, raw_text: str, task: str = "") -> Path | None:
        """Write one entry. Returns the path written, or ``None`` when caching is off."""
        if not self.enabled or not prompt_hash:
            return None
        path = self.path_for(prompt_hash, schema_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prompt_hash": prompt_hash,
            "schema": schema_name,
            "task": task,
            "raw_text": raw_text,
        }
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return path

    def store_model(self, prompt_hash: str, parsed: BaseModel, task: str = "") -> Path | None:
        """Convenience wrapper storing a validated model as its canonical JSON."""
        return self.store(prompt_hash, type(parsed).__name__, parsed.model_dump_json(), task=task)

    def clear(self) -> int:
        """Delete every entry; returns how many files were removed."""
        if not self.cache_dir.is_dir():
            return 0
        removed = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
            removed += 1
        return removed
