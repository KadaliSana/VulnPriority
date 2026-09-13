"""Nuclei JSONL result parsing.

Nuclei emits one JSON object per match, one per line. Each object names the template that
fired (``template-id``, which becomes the plugin id and therefore part of the dedup key),
the exact URL it matched at, and - crucially for Component B - the template's own
``classification`` block carrying CVE and CWE ids.

A malformed line is skipped rather than fatal: Nuclei output is frequently truncated by a
cancelled run, and one bad line must not cost an entire scan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from vulnprio.core.errors import ParseError
from vulnprio.core.interfaces import ScannerParser
from vulnprio.core.models import Finding, Scan, UntrustedText
from vulnprio.core.registry import register_parser

from vulnprio.ingest.normalize import (
    DEFAULT_SCANNED_AT,
    EndpointAccumulator,
    canonical_url,
    dedupe_untrusted,
    extract_cves,
    extract_cwe,
    make_app_id,
    make_finding_id,
    make_scan_id,
    method_from_request,
    parse_http_message,
    parse_timestamp,
    scanner_text,
    severity_from_string,
    target_text,
    url_host,
)
from vulnprio.ingest.tech_fingerprint import fingerprint_library, fingerprint_response

__all__ = ["NucleiParser"]

#: Nuclei reports a template match, not a probabilistic guess: a fired matcher is strong
#: evidence, a recorded-but-unmatched result much weaker.
_MATCHED_CONFIDENCE = 0.9
_UNMATCHED_CONFIDENCE = 0.5


@dataclass
class _Result:
    """One Nuclei match, flattened out of its nested JSON."""

    template_id: str = ""
    name: str = ""
    severity: str = ""
    description: str = ""
    cve_ids: tuple[str, ...] = ()
    cwe_id: int | None = None
    matched_at: str = ""
    host: str = ""
    request: str = ""
    response: str = ""
    extracted: tuple[str, ...] = ()
    matched: bool = True
    timestamp: datetime | None = None
    tags: tuple[str, ...] = ()


@register_parser("nuclei")
class NucleiParser(ScannerParser):
    """``ScannerParser`` for Nuclei JSONL output (``-jsonl`` / ``-json-export``)."""

    name = "nuclei"

    def sniff(self, path: str | Path) -> bool:
        """True when the first JSON line looks like a Nuclei result."""
        for record in _iter_lines(path, limit=5):
            if "template-id" in record or "template_id" in record:
                return True
            if "matched-at" in record and isinstance(record.get("info"), dict):
                return True
            return False
        return False

    def parse(self, path: str | Path, app_id: str | None = None) -> Scan:
        """Read a Nuclei JSONL file into a :class:`Scan`."""
        source = Path(path)
        results = [_read_result(record) for record in _iter_lines(source)]
        if not results:
            raise ParseError(f"no Nuclei results found in {source}")
        return _build_scan(results, app_id)


def _iter_lines(path: str | Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    """Yield the JSON objects of a JSONL file, skipping blank and malformed lines."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        yield record
        count += 1
        if limit is not None and count >= limit:
            return


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _as_tuple(value: Any) -> tuple[str, ...]:
    """Nuclei writes classification ids as either a string or a list of strings."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _read_result(record: dict[str, Any]) -> _Result:
    info = record.get("info") if isinstance(record.get("info"), dict) else {}
    classification = (
        info.get("classification") if isinstance(info.get("classification"), dict) else {}
    )
    cwe_values = _as_tuple(classification.get("cwe-id") or classification.get("cwe_id"))
    cve_values = _as_tuple(classification.get("cve-id") or classification.get("cve_id"))
    template_id = _as_text(record.get("template-id") or record.get("template_id"))
    return _Result(
        template_id=template_id,
        name=_as_text(info.get("name")) or template_id,
        severity=_as_text(info.get("severity")),
        description=_as_text(info.get("description")),
        cve_ids=extract_cves(cve_values, template_id, _as_text(info.get("name"))),
        cwe_id=extract_cwe(*cwe_values),
        matched_at=_as_text(record.get("matched-at") or record.get("matched_at")),
        host=_as_text(record.get("host")),
        request=_as_text(record.get("request")),
        response=_as_text(record.get("response")),
        extracted=_as_tuple(record.get("extracted-results") or record.get("extracted_results")),
        matched=bool(record.get("matcher-status", True)),
        timestamp=parse_timestamp(record.get("timestamp")),
        tags=_as_tuple(info.get("tags")),
    )


def _build_scan(results: list[_Result], app_id: str | None) -> Scan:
    timestamps = [result.timestamp for result in results if result.timestamp is not None]
    scanned_at = min(timestamps) if timestamps else DEFAULT_SCANNED_AT

    primary_host = ""
    for result in results:
        primary_host = url_host(result.host or result.matched_at)
        if primary_host:
            break
    resolved_app_id = app_id or make_app_id(primary_host or "nuclei")
    scan_id = make_scan_id(resolved_app_id, scanned_at, "nuclei")

    accumulator = EndpointAccumulator(resolved_app_id)
    findings: dict[str, Finding] = {}

    for result in results:
        base = result.host or (f"https://{primary_host}" if primary_host else None)
        url = canonical_url(result.matched_at or result.host or "/", base=base)
        response = parse_http_message(result.response)
        method = method_from_request(result.request)

        endpoint_id = accumulator.add(
            url,
            method,
            base=base,
            response_status=response.status,
            response_content_type=response.content_type,
            response_size_bytes=len(response.body.encode("utf-8")) if response.body else None,
            sets_cookie=response.sets_cookie,
            response_sample=target_text(response.body, source_url=url),
            observed_tech=fingerprint_response(
                headers=response.headers,
                cookies=response.cookie_names,
                body=response.body,
                url=url,
            ),
        )

        param = _param_of(url)
        finding_id = make_finding_id(scan_id, endpoint_id, result.template_id or result.name, param)
        evidence: list[UntrustedText] = []
        echoed = target_text(result.response, source_url=url)
        if echoed is not None:
            evidence.append(echoed)
        for value in result.extracted:
            wrapped = target_text(value, source_url=url)
            if wrapped is not None:
                evidence.append(wrapped)
        probe = scanner_text(result.request, source_url=url)
        if probe is not None:
            evidence.append(probe)

        existing = findings.get(finding_id)
        if existing is not None:
            findings[finding_id] = existing.model_copy(
                update={"evidence": dedupe_untrusted(list(existing.evidence) + evidence)}
            )
            continue

        description = (
            scanner_text(" ".join(part for part in (result.name, result.description) if part))
            or scanner_text(result.template_id or "Nuclei match")
        )
        if description is None:  # pragma: no cover - unreachable, literal is non-empty
            raise ParseError("Nuclei result carries no usable description")

        findings[finding_id] = Finding(
            finding_id=finding_id,
            scan_id=scan_id,
            app_id=resolved_app_id,
            endpoint_id=endpoint_id,
            name=result.name or result.template_id or "Nuclei match",
            cwe_id=result.cwe_id,
            cve_ids=result.cve_ids,
            scanner="nuclei",
            scanner_plugin_id=result.template_id or None,
            scanner_severity=severity_from_string(result.severity),
            scanner_confidence=_MATCHED_CONFIDENCE if result.matched else _UNMATCHED_CONFIDENCE,
            description=description,
            evidence=dedupe_untrusted(evidence),
            affected_component=fingerprint_library(
                " ".join(result.extracted), result.matched_at, result.name
            ),
            observed_at=result.timestamp or scanned_at,
        )

    hosts = accumulator.hosts() or ((primary_host,) if primary_host else ())
    return Scan(
        scan_id=scan_id,
        app_id=resolved_app_id,
        app_name=hosts[0] if hosts else "unknown",
        scanned_at=scanned_at,
        scanner_name="nuclei",
        scanner_version=None,
        hosts=hosts,
        tech_stack=accumulator.tech_stack(),
        endpoints=accumulator.endpoints(),
        findings=tuple(findings.values()),
    )


def _param_of(url: str) -> str:
    """Nuclei has no explicit parameter field; the matched query key is the best proxy."""
    query = urlsplit(url).query
    if not query:
        return ""
    return sorted(pair.split("=", 1)[0] for pair in query.split("&") if pair)[0]
