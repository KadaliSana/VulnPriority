"""Burp Suite XML issue export parsing.

Burp reports one ``<issue>`` per (host, path, issue type) and attaches the full
request/response pair as base64. Those bodies are the most attacker-influenced text the
framework ever ingests, so they are decoded defensively - never interpreted, never
executed, never allowed to fail a whole report - and wrapped as ``TARGET_RESPONSE``.

Burp does not emit CWE ids, so a best-effort map from its numeric issue ``type`` supplies
one. Any CWE actually written in the issue text always wins over the table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

from vulnprio.core.errors import ParseError
from vulnprio.core.interfaces import ScannerParser
from vulnprio.core.models import Finding, Scan, UntrustedText
from vulnprio.core.registry import register_parser

from vulnprio.ingest.normalize import (
    DEFAULT_SCANNED_AT,
    EndpointAccumulator,
    canonical_url,
    decode_maybe_base64,
    dedupe_untrusted,
    extract_cves,
    extract_cwe,
    http_method,
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

__all__ = ["BurpParser", "BURP_TYPE_CWE", "BURP_CONFIDENCE"]

#: Burp's textual confidence to a probability in [0, 1].
BURP_CONFIDENCE: dict[str, float] = {
    "certain": 0.95,
    "firm": 0.7,
    "tentative": 0.4,
}

#: Best-effort CWE for Burp's numeric issue types. Text-extracted CWEs take priority,
#: and an unmapped type simply leaves ``cwe_id`` unset rather than guessing.
BURP_TYPE_CWE: dict[str, int] = {
    "1048832": 78,    # OS command injection
    "1049088": 89,    # SQL injection
    "1049344": 91,    # XML / SOAP injection
    "1049600": 90,    # LDAP injection
    "2097408": 22,    # File path traversal
    "2097664": 611,   # XML external entity injection
    "2097920": 79,    # Cross-site scripting (reflected)
    "2097936": 79,    # Cross-site scripting (stored)
    "2098944": 502,   # Serialized object in HTTP message
    "4194560": 601,   # Open redirection
    "4195328": 352,   # Cross-site request forgery
    "5243136": 693,   # Path-relative style sheet import
    "5243392": 319,   # Cleartext submission of password
    "5244416": 522,   # Password field with autocomplete enabled
    "5244544": 200,   # Information disclosure
    "5245344": 1104,  # Vulnerable JavaScript dependency
    "6291968": 548,   # Directory listing
    "8389632": 16,    # Security header not set
}

_PARAM_IN_LOCATION = re.compile(r"\[([^\]]+?)\s+(?:parameter|cookie|header)\]", re.IGNORECASE)


@dataclass
class _Issue:
    """One decoded Burp issue, ready to become an endpoint plus a finding."""

    name: str = ""
    issue_type: str = ""
    host: str = ""
    path: str = ""
    location: str = ""
    severity: str = ""
    confidence: str = ""
    background: str = ""
    detail: str = ""
    remediation: str = ""
    request: str = ""
    response: str = ""
    request_method: str = ""


@register_parser("burp")
class BurpParser(ScannerParser):
    """``ScannerParser`` for Burp Suite ``<issues>`` XML exports."""

    name = "burp"

    def sniff(self, path: str | Path) -> bool:
        """True for an XML document whose root element is ``<issues>``."""
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        stripped = text.lstrip()
        if not stripped.startswith("<"):
            return False
        try:
            root = ElementTree.fromstring(stripped)
        except ElementTree.ParseError:
            return False
        return root.tag == "issues"

    def parse(self, path: str | Path, app_id: str | None = None) -> Scan:
        """Read a Burp XML export into a :class:`Scan`."""
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError as error:  # pragma: no cover - filesystem failure
            raise ParseError(f"cannot read Burp export {source}: {error}") from error
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError as error:
            raise ParseError(f"invalid Burp XML export {source}: {error}") from error
        if root.tag != "issues":
            raise ParseError(f"{source} is not a Burp issue export (root <{root.tag}>)")

        issues = [_read_issue(element) for element in root.findall("issue")]
        exported_at = parse_timestamp(root.get("exportTime")) or DEFAULT_SCANNED_AT
        return _build_scan(issues, root.get("burpVersion"), exported_at, app_id)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _read_issue(element: ElementTree.Element) -> _Issue:
    pair = element.find("requestresponse")
    request_element = pair.find("request") if pair is not None else None
    response_element = pair.find("response") if pair is not None else None

    request = ""
    request_method = ""
    if request_element is not None:
        request = decode_maybe_base64(
            _text(request_element), (request_element.get("base64") or "").lower() == "true"
        )
        request_method = request_element.get("method") or ""
    response = ""
    if response_element is not None:
        response = decode_maybe_base64(
            _text(response_element), (response_element.get("base64") or "").lower() == "true"
        )

    return _Issue(
        name=_text(element.find("name")),
        issue_type=_text(element.find("type")),
        host=_text(element.find("host")),
        path=_text(element.find("path")),
        location=_text(element.find("location")),
        severity=_text(element.find("severity")),
        confidence=_text(element.find("confidence")),
        background=_text(element.find("issueBackground")),
        detail=_text(element.find("issueDetail")),
        remediation=_text(element.find("remediationBackground")),
        request=request,
        response=response,
        request_method=request_method,
    )


# ---------------------------------------------------------------------------
# Scan assembly
# ---------------------------------------------------------------------------


def _build_scan(
    issues: list[_Issue],
    scanner_version: str | None,
    exported_at: datetime,
    app_id: str | None,
) -> Scan:
    primary_host = ""
    for issue in issues:
        primary_host = url_host(issue.host)
        if primary_host:
            break
    resolved_app_id = app_id or make_app_id(primary_host or "burp")
    scan_id = make_scan_id(resolved_app_id, exported_at, "burp")

    accumulator = EndpointAccumulator(resolved_app_id)
    findings: dict[str, Finding] = {}

    for issue in issues:
        url = canonical_url(issue.path or "/", base=issue.host)
        response = parse_http_message(issue.response)
        request = parse_http_message(issue.request)
        method = (
            http_method(issue.request_method)
            if issue.request_method
            else method_from_request(issue.request)
        )
        param = _param_of(issue)

        endpoint_id = accumulator.add(
            url,
            method,
            base=issue.host,
            parameters=(param,) if param else (),
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

        plugin_id = issue.issue_type or issue.name
        finding_id = make_finding_id(scan_id, endpoint_id, plugin_id, param)
        evidence: list[UntrustedText] = []
        echoed = target_text(issue.response, source_url=url)
        if echoed is not None:
            evidence.append(echoed)
        for value in (issue.detail, issue.request):
            wrapped = scanner_text(value, source_url=url)
            if wrapped is not None:
                evidence.append(wrapped)

        existing = findings.get(finding_id)
        if existing is not None:
            findings[finding_id] = existing.model_copy(
                update={"evidence": dedupe_untrusted(list(existing.evidence) + evidence)}
            )
            continue

        description = (
            scanner_text(" ".join(part for part in (issue.name, issue.background) if part))
            or scanner_text(issue.name or "Burp issue")
        )
        if description is None:  # pragma: no cover - unreachable, literal is non-empty
            raise ParseError("Burp issue carries no usable description")

        findings[finding_id] = Finding(
            finding_id=finding_id,
            scan_id=scan_id,
            app_id=resolved_app_id,
            endpoint_id=endpoint_id,
            name=issue.name or "Burp issue",
            cwe_id=extract_cwe(
                *_cwe_candidates(issue), BURP_TYPE_CWE.get(issue.issue_type.strip())
            ),
            cve_ids=extract_cves(issue.name, issue.detail, issue.background, issue.remediation),
            scanner="burp",
            scanner_plugin_id=issue.issue_type or None,
            scanner_severity=severity_from_string(issue.severity),
            scanner_confidence=BURP_CONFIDENCE.get(issue.confidence.strip().lower(), 0.5),
            description=description,
            evidence=dedupe_untrusted(evidence),
            affected_component=fingerprint_library(issue.detail, issue.path, request.start_line),
            observed_at=exported_at,
        )

    hosts = accumulator.hosts() or ((primary_host,) if primary_host else ())
    return Scan(
        scan_id=scan_id,
        app_id=resolved_app_id,
        app_name=hosts[0] if hosts else "unknown",
        scanned_at=exported_at,
        scanner_name="burp",
        scanner_version=scanner_version,
        hosts=hosts,
        tech_stack=accumulator.tech_stack(),
        endpoints=accumulator.endpoints(),
        findings=tuple(findings.values()),
    )


def _param_of(issue: _Issue) -> str:
    """Burp names the injection point in ``location``: ``/x [username parameter]``."""
    match = _PARAM_IN_LOCATION.search(issue.location)
    return match.group(1).strip() if match else ""


def _cwe_candidates(issue: _Issue) -> tuple[str, ...]:
    return (issue.detail, issue.background, issue.remediation, issue.name)
