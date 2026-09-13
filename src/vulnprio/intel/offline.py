"""The offline provider, and the recorder that lets its corpus grow.

``FixtureSearchProvider`` serves search results from ``data/fixtures/intel/searches.json``
keyed by CVE. It is what makes "offline by default" true for this package: the default
configuration runs the whole two-phase flow, including the sandbox and the extraction,
with no socket and no key. The test suite never does anything else.

``RecordingProvider`` wraps a live provider and writes what it saw into that same file. It
exists because the alternative -- hand-writing JSON that is supposed to look like the
internet -- produces a corpus that quietly drifts away from the shapes the real provider
returns, and a fixture that no longer resembles reality tests nothing. Recording is
explicit and opt-in; nothing records by accident.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from vulnprio.core.config import PROJECT_ROOT
from vulnprio.intel.models import (
    IntelCitation,
    IntelConfig,
    IntelDocument,
    IntelGather,
    IntelQuery,
    IntelSourceKind,
    IntelUsage,
)
from vulnprio.intel.provider import BaseSearchProvider, make_document

__all__ = [
    "FIXTURE_VERSION",
    "FixtureSearchProvider",
    "RecordingProvider",
    "fixture_key_for",
    "load_fixture",
]

#: Bumped when the on-disk shape changes. A file with a different version is refused
#: rather than best-effort parsed, so a stale corpus fails loudly.
FIXTURE_VERSION = "1"


def fixture_key_for(query: IntelQuery) -> str:
    """Corpus key a query reads from.

    CVE identifiers key the corpus because they are the stable, shareable name for a
    vulnerability. A finding with no CVE -- most web application findings -- falls back to
    its own identifier, which keeps the mechanism working for the case the reviewed
    literature usually skips.
    """
    if query.cve_id:
        return query.cve_id.strip().upper()
    if query.finding_id:
        return f"finding:{query.finding_id}"
    return ""


def _parse_stamp(value: Any, default: datetime) -> datetime:
    if not value:
        return default
    try:
        stamp = datetime.fromisoformat(str(value))
    except ValueError:
        return default
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)


def load_fixture(path: str | Path) -> dict[str, Any]:
    """Read and validate the corpus file. Returns ``{}`` when it does not exist."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.is_file():
        return {}
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"intel fixture {resolved} must hold an object")
    version = str(data.get("version", ""))
    if version and version != FIXTURE_VERSION:
        raise ValueError(
            f"intel fixture {resolved} is version {version}; this build reads {FIXTURE_VERSION}"
        )
    entries = data.get("entries")
    if entries is not None and not isinstance(entries, dict):
        raise ValueError(f"intel fixture {resolved} has a non-object 'entries'")
    return data


class FixtureSearchProvider(BaseSearchProvider):
    """Search results replayed from the recorded corpus."""

    name = "fixture"

    def __init__(self, path: str | Path | None = None, *, strict: bool = False) -> None:
        """``strict`` makes a missing corpus an error instead of an empty result."""
        self.path = Path(path) if path is not None else IntelConfig().fixture_path
        self.strict = bool(strict)
        self._data: dict[str, Any] | None = None

    # -- corpus ------------------------------------------------------------

    @property
    def data(self) -> dict[str, Any]:
        if self._data is None:
            self._data = load_fixture(self.path)
        return self._data

    def reload(self) -> None:
        """Drop the cached corpus so a recording written this run is picked up."""
        self._data = None

    def entries(self) -> dict[str, Any]:
        entries = self.data.get("entries") or {}
        return entries if isinstance(entries, dict) else {}

    def available(self) -> bool:
        return bool(self.entries())

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.entries()))

    # -- provider ----------------------------------------------------------

    def _documents_of(
        self, entry: dict[str, Any], key: str, config: IntelConfig
    ) -> list[IntelDocument]:
        default_stamp = _parse_stamp(
            entry.get("recorded_at"), datetime(1970, 1, 1, tzinfo=timezone.utc)
        )
        documents: list[IntelDocument] = []
        for record in entry.get("documents") or ():
            if not isinstance(record, dict):
                continue
            url = str(record.get("url") or "")
            if not url:
                continue
            kind_raw = str(record.get("source_kind") or "").strip().lower()
            try:
                kind = IntelSourceKind(kind_raw) if kind_raw else None
            except ValueError:
                kind = IntelSourceKind.UNKNOWN
            documents.append(
                make_document(
                    url,
                    str(record.get("text") or ""),
                    title=record.get("title"),
                    retrieved_at=_parse_stamp(record.get("retrieved_at"), default_stamp),
                    source_kind=kind,
                    relevance=float(record.get("relevance", 0.5)),
                    query_text=str(record.get("query") or key),
                    char_budget=config.snippet_char_budget,
                )
            )
        return documents

    @staticmethod
    def _citations_of(entry: dict[str, Any]) -> list[IntelCitation]:
        citations: list[IntelCitation] = []
        for record in entry.get("citations") or ():
            if not isinstance(record, dict) or not record.get("url"):
                continue
            citations.append(
                IntelCitation(
                    url=str(record["url"])[:500],
                    cited_text=str(record.get("cited_text") or "")[:400],
                    title=(str(record["title"])[:300] if record.get("title") else None),
                )
            )
        return citations

    @staticmethod
    def _usage_of(entry: dict[str, Any]) -> IntelUsage:
        record = entry.get("usage")
        if not isinstance(record, dict):
            return IntelUsage()
        return IntelUsage(
            input_tokens=int(record.get("input_tokens", 0) or 0),
            output_tokens=int(record.get("output_tokens", 0) or 0),
            cache_read_tokens=int(record.get("cache_read_tokens", 0) or 0),
            web_search_requests=int(record.get("web_search_requests", 0) or 0),
            calls=int(record.get("calls", 0) or 0),
        )

    def gather(
        self,
        queries: Sequence[IntelQuery],
        config: IntelConfig,
        *,
        instruction: str = "",
    ) -> IntelGather:
        """Replay every corpus entry the plan's keys name, merged and deduplicated."""
        entries = self.entries()
        if not entries:
            message = f"intel fixture corpus is empty or missing: {self.path}"
            if self.strict:
                raise FileNotFoundError(message)
            return IntelGather(errors=(message,), provider=self.name)

        keys = tuple(dict.fromkeys(fixture_key_for(query) for query in queries))
        documents: list[IntelDocument] = []
        citations: list[IntelCitation] = []
        narrative = ""
        model = ""
        usage = IntelUsage()
        errors: list[str] = []
        matched = False

        for key in keys:
            entry = entries.get(key)
            if not isinstance(entry, dict):
                continue
            matched = True
            documents.extend(self._documents_of(entry, key, config))
            if not narrative and entry.get("narrative"):
                narrative = str(entry["narrative"])
                citations = self._citations_of(entry)
                model = str(entry.get("model") or "")
            usage = usage + self._usage_of(entry)
            for message in entry.get("errors") or ():
                errors.append(str(message))

        if not matched and keys:
            errors.append(
                f"no recorded intel for {', '.join(key for key in keys if key) or 'this finding'}"
            )

        seen: set[str] = set()
        unique: list[IntelDocument] = []
        for document in documents:
            if document.url in seen:
                continue
            seen.add(document.url)
            unique.append(document)

        return IntelGather(
            documents=tuple(unique[: config.max_documents]),
            narrative=narrative,
            citations=tuple(citations),
            usage=usage,
            errors=tuple(errors),
            model=model,
            provider=self.name,
        )


class RecordingProvider(BaseSearchProvider):
    """Runs a real provider and writes the session into the fixture corpus.

    Merging is by URL within a key, so recording the same CVE twice grows the entry rather
    than replacing it, and a corpus assembled over several sessions stays coherent. The
    narrative is only overwritten when the new session produced one.
    """

    name = "recording"

    def __init__(
        self,
        inner: BaseSearchProvider,
        path: str | Path | None = None,
        *,
        note: str = "",
    ) -> None:
        self.inner = inner
        self.path = Path(path) if path is not None else IntelConfig().fixture_path
        self.note = str(note)

    def available(self) -> bool:
        return self.inner.available()

    @property
    def resolved_path(self) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    def gather(
        self,
        queries: Sequence[IntelQuery],
        config: IntelConfig,
        *,
        instruction: str = "",
    ) -> IntelGather:
        result = self.inner.gather(queries, config, instruction=instruction)
        if result.documents or result.narrative:
            self.record(queries, result)
        return result

    # -- writing -----------------------------------------------------------

    def record(self, queries: Sequence[IntelQuery], gathered: IntelGather) -> Path:
        """Merge one gathering into the corpus file and write it back."""
        path = self.resolved_path
        try:
            data = load_fixture(path)
        except (ValueError, json.JSONDecodeError):
            data = {}
        if not data:
            data = {"version": FIXTURE_VERSION, "entries": {}}
        data.setdefault("version", FIXTURE_VERSION)
        entries = data.setdefault("entries", {})
        if self.note:
            data["note"] = self.note

        key = next((fixture_key_for(query) for query in queries if fixture_key_for(query)), "")
        if not key:
            return path
        entry = entries.get(key) if isinstance(entries.get(key), dict) else {}
        entry = dict(entry)
        entry["recorded_at"] = datetime.now(timezone.utc).isoformat()
        entry["model"] = gathered.model or entry.get("model", "")
        entry["queries"] = sorted(
            {*(entry.get("queries") or ()), *(query.text for query in queries)}
        )
        if gathered.narrative:
            entry["narrative"] = gathered.narrative
            entry["citations"] = [
                {
                    "url": citation.url,
                    "cited_text": citation.cited_text,
                    "title": citation.title,
                }
                for citation in gathered.citations
            ]
        entry["usage"] = gathered.usage.model_dump()
        entry["documents"] = _merge_documents(entry.get("documents") or (), gathered.documents)
        entries[key] = entry

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path


def _merge_documents(
    existing: Iterable[Any], fresh: Iterable[IntelDocument]
) -> list[dict[str, Any]]:
    """Union of recorded documents by URL, fresh text winning on a collision."""
    merged: dict[str, dict[str, Any]] = {}
    for record in existing:
        if isinstance(record, dict) and record.get("url"):
            merged[str(record["url"])] = dict(record)
    for document in fresh:
        merged[document.url] = {
            "url": document.url,
            "title": document.title,
            "text": document.snippet.text,
            "retrieved_at": document.retrieved_at.isoformat(),
            "source_kind": document.source_kind.value,
            "relevance": document.relevance,
            "query": document.query_text,
        }
    return [merged[url] for url in sorted(merged)]
