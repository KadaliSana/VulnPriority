"""Wapiti JSON report parsing.

Wapiti groups its results by *category* ("SQL Injection", "Cross Site Scripting", ...) and
lists every instance under it, which is the same shape ZAP uses and is handled the same
way here: an instance becomes a finding only after its URL has been canonicalised and
templated, so two injections differing by a row id collapse into one finding.

Three details are specific to Wapiti and worth stating:

* **Three result sections.** ``vulnerabilities`` holds confirmed findings, ``anomalies``
  holds server errors and timeouts observed while testing, and ``additionals`` holds
  informational observations such as technology fingerprints. All three are ingested:
  an anomaly is weak evidence, not no evidence, and the framework's job is to weigh
  evidence rather than to discard it at the door.
* **CWE comes from the report when present.** Recent Wapiti versions carry ``cwe`` in the
  ``classifications`` block. Older ones do not, so a category lexicon fills the gap and
  free-text extraction is the last resort. A category with no defensible CWE gets
  ``None`` rather than a guess.
* **Severity is Wapiti's own ``level``** (0-4), read through the shared riskcode mapping.

Trust boundaries are set here and nowhere else: Wapiti's own prose is ``SCANNER_OUTPUT``
and anything echoed back from the application - the HTTP request it built, the response
body it recorded - is ``TARGET_RESPONSE``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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
    http_method,
    make_app_id,
    make_finding_id,
    make_scan_id,
    parse_timestamp,
    scanner_text,
    severity_from_riskcode,
    target_text,
    url_host,
)
from vulnprio.ingest.tech_fingerprint import fingerprint_library, fingerprint_response

__all__ = ["WapitiParser"]

#: Wapiti category to CWE, for reports that do not carry ``cwe`` in their classifications.
#: Only mappings that are unambiguous are listed; anything else is left to free-text
#: extraction and then to ``None``, because a wrong CWE is worse than a missing one - it
#: joins the finding to the wrong intelligence downstream.
_CATEGORY_CWE: dict[str, int] = {
    "sql injection": 89,
    "blind sql injection": 89,
    "ldap injection": 90,
    "cross site scripting": 79,
    "stored cross site scripting": 79,
    "permanent xss": 79,
    "command execution": 78,
    "path traversal": 22,
    "file handling": 22,
    "crlf injection": 93,
    "cross site request forgery": 352,
    "csrf": 352,
    "server side request forgery": 918,
    "xml external entity": 611,
    "open redirect": 601,
    "redirect": 601,
    "secure flag cookie": 614,
    "httponly flag cookie": 1004,
    "content security policy configuration": 693,
    "http secure headers": 693,
    "clickjacking protection": 1021,
    "htaccess bypass": 538,
    "backup file": 530,
    "potentially dangerous file": 538,
    "weak credentials": 1392,
    "fingerprint web technology": 200,
    "fingerprint web server": 200,
    "vulnerable software": 1104,
    "log4shell": 917,
    "spring4shell": 94,
    "resource consumption": 400,
    "unencrypted channels": 319,
}

#: Sections of a Wapiti report that carry results, and the confidence each deserves.
#: A confirmed vulnerability is a test that fired; an anomaly is a server that misbehaved
#: while being tested, which is suggestive rather than conclusive.
_SECTIONS: tuple[tuple[str, float], ...] = (
    ("vulnerabilities", 0.8),
    ("anomalies", 0.4),
    ("additionals", 0.6),
)

_VERSION = re.compile(r"(\d+(?:\.\d+)+)")


@dataclass
class _Entry:
    """One Wapiti result instance, flattened out of its section and category."""

    category: str = ""
    module: str = ""
    section: str = "vulnerabilities"
    confidence: float = 0.5
    level: Any = None
    method: str = "GET"
    path: str = ""
    parameter: str = ""
    info: str = ""
    http_request: str = ""
    curl_command: str = ""
    references: tuple[str, ...] = ()
    description: str = ""
    solution: str = ""
    cwe_id: int | None = None
    response_status: int | None = None
    response_body: str = ""
    response_headers: dict[str, str] = field(default_factory=dict)


@register_parser("wapiti")
class WapitiParser(ScannerParser):
    """``ScannerParser`` for the Wapiti 3.x JSON report (``wapiti -f json``)."""

    name = "wapiti"

    def sniff(self, path: str | Path) -> bool:
        """True for a JSON object with Wapiti's ``vulnerabilities`` *mapping* and ``infos``.

        The mapping test is what separates Wapiti from Nikto: both name a
        ``vulnerabilities`` key, but Wapiti's is a dict keyed by category and Nikto's is a
        list of individual findings.
        """
        data = _load(path)
        if not isinstance(data, dict):
            return False
        vulnerabilities = data.get("vulnerabilities")
        if not isinstance(vulnerabilities, dict):
            return False
        return isinstance(data.get("infos"), dict) or isinstance(data.get("classifications"), dict)

    def parse(self, path: str | Path, app_id: str | None = None) -> Scan:
        """Read a Wapiti JSON report into a :class:`Scan`."""
        source = Path(path)
        data = _load(source)
        if not isinstance(data, dict):
            raise ParseError(f"invalid Wapiti JSON report {source}: not an object")
        if not isinstance(data.get("vulnerabilities"), dict):
            raise ParseError(f"{source} has no Wapiti vulnerabilities mapping")

        infos = data.get("infos") if isinstance(data.get("infos"), dict) else {}
        classifications = (
            data.get("classifications") if isinstance(data.get("classifications"), dict) else {}
        )
        entries = _read_entries(data, classifications)
        return _build_scan(entries, infos, app_id)


def _load(path: str | Path) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _references_of(classification: dict[str, Any]) -> tuple[str, ...]:
    """Wapiti writes references as ``{title: url}``; both halves may name a CVE."""
    references = classification.get("ref")
    if isinstance(references, dict):
        return tuple(f"{key} {value}" for key, value in references.items())
    if isinstance(references, (list, tuple)):
        return tuple(str(item) for item in references)
    return ()


def _category_cwe(category: str, classification: dict[str, Any]) -> int | None:
    """CWE from the report first, then the lexicon, then free text, then nothing."""
    explicit = extract_cwe(classification.get("cwe"), classification.get("cwe_id"))
    if explicit is not None:
        return explicit
    mapped = _CATEGORY_CWE.get(category.strip().lower())
    if mapped is not None:
        return mapped
    return extract_cwe(_as_text(classification.get("desc")))


def _read_entries(
    data: dict[str, Any], classifications: dict[str, Any]
) -> list[_Entry]:
    entries: list[_Entry] = []
    for section, confidence in _SECTIONS:
        block = data.get(section)
        if not isinstance(block, dict):
            continue
        for category, instances in block.items():
            if not isinstance(instances, list):
                continue
            classification = (
                classifications.get(category) if isinstance(classifications.get(category), dict) else {}
            )
            cwe_id = _category_cwe(str(category), classification)
            references = _references_of(classification)
            for raw in instances:
                if not isinstance(raw, dict):
                    continue
                entries.append(
                    _read_entry(
                        raw,
                        category=str(category),
                        section=section,
                        confidence=confidence,
                        cwe_id=cwe_id,
                        classification=classification,
                        references=references,
                    )
                )
    return entries


def _read_entry(
    raw: dict[str, Any],
    *,
    category: str,
    section: str,
    confidence: float,
    cwe_id: int | None,
    classification: dict[str, Any],
    references: tuple[str, ...],
) -> _Entry:
    detail = raw.get("detail") if isinstance(raw.get("detail"), dict) else {}
    response = detail.get("response") if isinstance(detail.get("response"), dict) else {}
    headers = response.get("headers")
    if isinstance(headers, list):   # some versions write [[name, value], ...]
        headers = {str(item[0]): str(item[1]) for item in headers if len(item) >= 2}
    return _Entry(
        category=category,
        module=_as_text(raw.get("module")),
        section=section,
        confidence=confidence,
        level=raw.get("level"),
        method=_as_text(raw.get("method")) or "GET",
        path=_as_text(raw.get("path")) or _as_text(raw.get("url")),
        parameter=_as_text(raw.get("parameter")),
        info=_as_text(raw.get("info")),
        http_request=_as_text(raw.get("http_request")),
        curl_command=_as_text(raw.get("curl_command")),
        references=references,
        description=_as_text(classification.get("desc")),
        solution=_as_text(classification.get("sol")),
        cwe_id=cwe_id,
        response_status=response.get("status_code") if isinstance(response.get("status_code"), int) else None,
        response_body=_as_text(response.get("body")),
        response_headers={str(key).lower(): str(value) for key, value in (headers or {}).items()}
        if isinstance(headers, dict)
        else {},
    )


def _parameter_name(raw: str) -> str:
    """``"q=__XSS__"`` and ``"q"`` both name the parameter ``q``."""
    cleaned = (raw or "").strip()
    if not cleaned:
        return ""
    return cleaned.split("=", 1)[0].strip()


def _scanner_version(infos: dict[str, Any]) -> str | None:
    raw = _as_text(infos.get("version")).strip()
    if not raw:
        return None
    match = _VERSION.search(raw)
    return match.group(1) if match else raw


def _build_scan(entries: list[_Entry], infos: dict[str, Any], app_id: str | None) -> Scan:
    target = _as_text(infos.get("target"))
    scanned_at: datetime = parse_timestamp(infos.get("date")) or DEFAULT_SCANNED_AT
    primary_host = url_host(target) if target else ""
    if not primary_host:
        for entry in entries:
            primary_host = url_host(entry.path)
            if primary_host:
                break

    resolved_app_id = app_id or make_app_id(primary_host or "wapiti")
    scan_id = make_scan_id(resolved_app_id, scanned_at, "wapiti")

    accumulator = EndpointAccumulator(resolved_app_id)
    findings: dict[str, Finding] = {}
    base = target or (f"https://{primary_host}" if primary_host else None)

    for entry in entries:
        url = canonical_url(entry.path or target or "/", base=base)
        method = http_method(entry.method)
        parameter = _parameter_name(entry.parameter)

        endpoint_id = accumulator.add(
            url,
            method,
            base=base,
            parameters=(parameter,) if parameter else (),
            response_status=entry.response_status,
            response_content_type=entry.response_headers.get("content-type", "").split(";")[0] or None,
            response_size_bytes=len(entry.response_body.encode("utf-8")) if entry.response_body else None,
            sets_cookie="set-cookie" in entry.response_headers,
            response_sample=target_text(entry.response_body, source_url=url),
            observed_tech=fingerprint_response(
                headers=entry.response_headers, body=entry.response_body, url=url
            ),
        )

        plugin_id = entry.module or entry.category
        finding_id = make_finding_id(scan_id, endpoint_id, plugin_id, parameter)

        evidence: list[UntrustedText] = []
        request_text = scanner_text(entry.http_request or entry.curl_command, source_url=url)
        if request_text is not None:
            evidence.append(request_text)
        echoed = target_text(entry.response_body, source_url=url)
        if echoed is not None:
            evidence.append(echoed)
        observed = target_text(entry.info, source_url=url)
        if observed is not None:
            evidence.append(observed)

        existing = findings.get(finding_id)
        if existing is not None:
            findings[finding_id] = existing.model_copy(
                update={"evidence": dedupe_untrusted(list(existing.evidence) + evidence)}
            )
            continue

        description = (
            scanner_text(
                " ".join(part for part in (entry.category, entry.description, entry.solution) if part)
            )
            or scanner_text(entry.category or "Wapiti finding")
        )
        if description is None:  # pragma: no cover - the literal is never empty
            raise ParseError("Wapiti entry carries no usable description")

        findings[finding_id] = Finding(
            finding_id=finding_id,
            scan_id=scan_id,
            app_id=resolved_app_id,
            endpoint_id=endpoint_id,
            name=entry.category or plugin_id or "Wapiti finding",
            cwe_id=entry.cwe_id,
            cve_ids=extract_cves(entry.info, entry.description, entry.references),
            scanner="wapiti",
            scanner_plugin_id=plugin_id or None,
            scanner_severity=severity_from_riskcode(entry.level),
            scanner_confidence=entry.confidence,
            description=description,
            evidence=dedupe_untrusted(evidence),
            affected_component=fingerprint_library(entry.info, entry.response_body, url),
            observed_at=scanned_at,
        )

    hosts = accumulator.hosts() or ((primary_host,) if primary_host else ())
    return Scan(
        scan_id=scan_id,
        app_id=resolved_app_id,
        app_name=hosts[0] if hosts else "unknown",
        scanned_at=scanned_at,
        scanner_name="wapiti",
        scanner_version=_scanner_version(infos),
        hosts=hosts,
        tech_stack=accumulator.tech_stack(),
        endpoints=accumulator.endpoints(),
        findings=tuple(findings.values()),
    )
