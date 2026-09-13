"""Deterministic hashing and identifier helpers.

Everything that must be reproducible across runs (identifiers, config hashes, prompt
hashes, dataset hashes) goes through this module so reproducibility is testable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "sha256_text",
    "sha256_bytes",
    "stable_id",
    "canonical_json",
    "config_hash",
    "dataset_hash",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_id(prefix: str, *parts: str) -> str:
    """Deterministic identifier: ``<prefix>_<first 16 hex of sha256 over parts>``."""
    return f"{prefix}_{sha256_text('|'.join(parts))[:16]}"


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys and no insignificant whitespace, for stable hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True)


def config_hash(obj: Any) -> str:
    """16-hex-character hash of any JSON-serialisable object (used for configs and prompts)."""
    return sha256_text(canonical_json(obj))[:16]


def dataset_hash(items: list[Any]) -> str:
    """Order-independent hash of a collection of JSON-serialisable items."""
    digests = sorted(sha256_text(canonical_json(item)) for item in items)
    return sha256_text("|".join(digests))[:16]
