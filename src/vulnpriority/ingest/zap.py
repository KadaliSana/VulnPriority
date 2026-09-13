"""OWASP ZAP report parsing (JSON report and XML traditional report).

ZAP groups by *alert* and lists concrete *instances* underneath it. The framework ranks
root causes rather than alerts, so an instance becomes a finding only after its URI has
been canonicalised and templated: two instances that differ solely by a row id collapse
into one finding, while instances on genuinely different routes stay separate and later
cluster through :mod:`vulnpriority.ingest.correlate`.

Trust boundaries are set here and nowhere else: ZAP's own prose is ``SCANNER_OUTPUT``
(tier ``SCANNER``) and anything echoed back from the application - evidence strings,
response headers and bodies - is ``TARGET_RESPONSE`` (tier ``TARGET_CONTENT``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from vulnpriority.core.errors import ParseError
from vulnpriority.core.interfaces import ScannerParser
from vulnpriority.core.models import Finding, Scan, UntrustedText
from vulnpriority.core.registry import register_parser

from vulnpriority.ingest.normalize import (
    DEFAULT_SCANNED_AT,
    EndpointAccumulator,
    canonical_url,
    confidence_from_code,
    dedupe_untrusted,
    extract_cves,
    extract_cwe,
    http_method,
    make_app_id,
    make_finding_id,
    make_scan_id,
    parse_headers,
    parse_http_message,
    parse_timestamp,
    scanner_text,
    severity_from_riskcode,
    severity_from_string,
    target_text,
    url_host,
)
from vulnpriority.ingest.tech_fingerprint import fingerprint_library, fingerprint_response

__all__ = ["ZapParser"]

_INSTANCE_FIELDS = (
    "uri",
    "method",
    "param",
    "attack",
    "evidence",
    "otherinfo",
    "requestheader",
    "requestbody",
    "responseheader",
    "responsebody",
)


@dataclass
class _Alert:
    """One ZAP alert, independent of whether it came from JSON or XML."""

    plugin_id: str | None = None
    name: str = ""
    riskcode: str | None = None
    riskdesc: str | None = None
    confidence: str | None = None
    description: str = ""
    solution: str = ""
    reference: str = ""
    other_info: str = ""
    cwe_id: str | None = None
    instances: list[dict[str, str]] = field(default_factory=list)


@dataclass
class _Site:
    name: str = ""
    host: str = ""
    alerts: list[_Alert] = field(default_factory=list)


@register_parser("zap")
class ZapParser(ScannerParser):
    """``ScannerParser`` for both ZAP report dialects."""

    name = "zap"

    def sniff(self, path: str | Path) -> bool:
        """True for a ZAP JSON report (``site`` array) or an ``OWASPZAPReport`` XML root."""
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        stripped = text.lstrip()
        if not stripped:
            return False
        if stripped.startswith("<"):
            return "OWASPZAPReport" in stripped[:2000]
        try:
            data = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return False
        if not isinstance(data, dict):
            return False
        if "scan_id" in data and "findings" in data:
            return False  # the framework's own canonical format
        if "@programName" in data and "zap" in str(data["@programName"]).lower():
            return True
        return isinstance(data.get("site"), (list, dict))

    def parse(self, path: str | Path, app_id: str | None = None) -> Scan:
        """Read a ZAP report into a :class:`Scan`."""
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError as error:  # pragma: no cover - filesystem failure
            raise ParseError(f"cannot read ZAP report {source}: {error}") from error
        if text.lstrip().startswith("<"):
            sites, scanner_version, generated = _read_xml(text, source)
        else:
            sites, scanner_version, generated = _read_json(text, source)
        return _build_scan(sites, scanner_version, generated, app_id)


# ---------------------------------------------------------------------------
# Dialect readers
# ---------------------------------------------------------------------------


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _read_json(text: str, source: Path) -> tuple[list[_Site], str | None, datetime | None]:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as error:
        raise ParseError(f"invalid ZAP JSON report {source}: {error}") from error
    if not isinstance(data, dict):
        raise ParseError(f"ZAP JSON report {source} is not an object")

    raw_sites = data.get("site")
    if isinstance(raw_sites, dict):
        raw_sites = [raw_sites]
    if not isinstance(raw_sites, list):
        raise ParseError(f"ZAP JSON report {source} has no site array")

    sites: list[_Site] = []
    for raw_site in raw_sites:
        if not isinstance(raw_site, dict):
            continue
        site = _Site(name=_as_text(raw_site.get("@name")), host=_as_text(raw_site.get("@host")))
        raw_alerts = raw_site.get("alerts")
        if isinstance(raw_alerts, dict):
            raw_alerts = [raw_alerts]
        for raw_alert in raw_alerts if isinstance(raw_alerts, list) else []:
            if not isinstance(raw_alert, dict):
                continue
            site.alerts.append(_alert_from_json(raw_alert))
        sites.append(site)
    # ZAP writes two timestamps and they are not equally parseable: ``@generated`` is a
    # human-readable RFC-1123-ish string, ``created`` is ISO-8601. Both are read, in that
    # order, because the first is the one ZAP documents and the second is the one that
    # survives a locale quirk. Taking only ``@generated`` was silently stamping every JSON
    # report with the epoch, which then became the run's intelligence as-of date and made
    # every feed lookup return nothing.
    generated = parse_timestamp(data.get("@generated")) or parse_timestamp(data.get("created"))
    return sites, _as_text(data.get("@version")) or None, generated


def _alert_from_json(raw: dict[str, Any]) -> _Alert:
    raw_instances = raw.get("instances")
    if isinstance(raw_instances, dict):
        raw_instances = [raw_instances]
    instances: list[dict[str, str]] = []
    for raw_instance in raw_instances if isinstance(raw_instances, list) else []:
        if not isinstance(raw_instance, dict):
            continue
        instances.append(
            {key: _as_text(raw_instance.get(key)) for key in _INSTANCE_FIELDS}
        )
    return _Alert(
        plugin_id=_as_text(raw.get("pluginid")) or _as_text(raw.get("alertRef")) or None,
        name=_as_text(raw.get("name")) or _as_text(raw.get("alert")),
        riskcode=_as_text(raw.get("riskcode")) or None,
        riskdesc=_as_text(raw.get("riskdesc")) or None,
        confidence=_as_text(raw.get("confidence")) or None,
        description=_as_text(raw.get("desc")),
        solution=_as_text(raw.get("solution")),
        reference=_as_text(raw.get("reference")),
        other_info=_as_text(raw.get("otherinfo")),
        cwe_id=_as_text(raw.get("cweid")) or None,
        instances=instances,
    )


def _read_xml(text: str, source: Path) -> tuple[list[_Site], str | None, datetime | None]:
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise ParseError(f"invalid ZAP XML report {source}: {error}") from error
    if root.tag != "OWASPZAPReport":
        raise ParseError(f"{source} is not an OWASP ZAP XML report (root <{root.tag}>)")

    sites: list[_Site] = []
    for site_element in root.findall("site"):
        site = _Site(
            name=site_element.get("name", ""),
            host=site_element.get("host", ""),
        )
        for alert_element in site_element.iter("alertitem"):
            site.alerts.append(_alert_from_xml(alert_element))
        sites.append(site)
    return sites, root.get("version"), parse_timestamp(root.get("generated"))


def _child_text(element: ElementTree.Element, tag: str) -> str:
    child = element.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _alert_from_xml(element: ElementTree.Element) -> _Alert:
    instances: list[dict[str, str]] = []
    for instance_element in element.iter("instance"):
        instances.append(
            {key: _child_text(instance_element, key) for key in _INSTANCE_FIELDS}
        )
    if not instances:
        uri = _child_text(element, "uri")
        if uri:
            instances.append({key: "" for key in _INSTANCE_FIELDS} | {"uri": uri})
    return _Alert(
        plugin_id=_child_text(element, "pluginid") or None,
        name=_child_text(element, "name") or _child_text(element, "alert"),
        riskcode=_child_text(element, "riskcode") or None,
        riskdesc=_child_text(element, "riskdesc") or None,
        confidence=_child_text(element, "confidence") or None,
        description=_child_text(element, "desc"),
        solution=_child_text(element, "solution"),
        reference=_child_text(element, "reference"),
        other_info=_child_text(element, "otherinfo"),
        cwe_id=_child_text(element, "cweid") or None,
        instances=instances,
    )


# ---------------------------------------------------------------------------
# Scan assembly
# ---------------------------------------------------------------------------


def _severity(alert: _Alert):
    """``riskcode`` is authoritative; ``riskdesc`` ("High (Medium)") is the fallback."""
    if alert.riskcode is not None and str(alert.riskcode).strip().lstrip("-").isdigit():
        return severity_from_riskcode(alert.riskcode)
    return severity_from_string(alert.riskdesc)


def _build_scan(
    sites: list[_Site],
    scanner_version: str | None,
    generated: datetime | None,
    app_id: str | None,
) -> Scan:
    primary_host = ""
    for site in sites:
        primary_host = (site.host or url_host(site.name or "")).lower()
        if primary_host:
            break
    resolved_app_id = app_id or make_app_id(primary_host or "zap")
    scanned_at = generated or DEFAULT_SCANNED_AT
    scan_id = make_scan_id(resolved_app_id, scanned_at, "zap")

    accumulator = EndpointAccumulator(resolved_app_id)
    findings: dict[str, Finding] = {}

    for site in sites:
        base = site.name or (f"https://{site.host}" if site.host else None)
        for alert in site.alerts:
            severity = _severity(alert)
            confidence = confidence_from_code(alert.confidence)
            description = scanner_text(
                " ".join(part for part in (alert.name, alert.description, alert.solution) if part)
            ) or scanner_text(alert.name or "ZAP alert")

            for instance in alert.instances:
                _absorb_instance(
                    accumulator=accumulator,
                    findings=findings,
                    scan_id=scan_id,
                    app_id=resolved_app_id,
                    base=base,
                    alert=alert,
                    instance=instance,
                    severity=severity,
                    confidence=confidence,
                    description=description,
                    scanned_at=scanned_at,
                )

    hosts = accumulator.hosts() or ((primary_host,) if primary_host else ())
    return Scan(
        scan_id=scan_id,
        app_id=resolved_app_id,
        app_name=hosts[0] if hosts else "unknown",
        scanned_at=scanned_at,
        scanner_name="zap",
        scanner_version=scanner_version,
        hosts=hosts,
        tech_stack=accumulator.tech_stack(),
        endpoints=accumulator.endpoints(),
        findings=tuple(findings.values()),
    )


def _absorb_instance(
    *,
    accumulator: EndpointAccumulator,
    findings: dict[str, Finding],
    scan_id: str,
    app_id: str,
    base: str | None,
    alert: _Alert,
    instance: dict[str, str],
    severity,
    confidence: float,
    description: UntrustedText | None,
    scanned_at: datetime,
) -> None:
    """Fold one alert instance into the endpoint set and the finding table.

    CWE and CVE ids are read from the instance as well as the alert: ZAP puts the CVE
    list of a vulnerable-library alert in the *instance* ``otherinfo``, not the alert.
    """
    uri = instance.get("uri") or base or ""
    if not uri:
        return
    url = canonical_url(uri, base=base)
    method = http_method(instance.get("method") or "GET")
    param = (instance.get("param") or "").strip()

    response = parse_http_message(
        _join_message(instance.get("responseheader"), instance.get("responsebody"))
    )
    headers = response.headers or parse_headers(instance.get("responseheader"))
    body = response.body or (instance.get("responsebody") or "")
    tech = fingerprint_response(headers=headers, body=body, url=url)

    endpoint_id = accumulator.add(
        url,
        method,
        base=base,
        parameters=(param,) if param else (),
        response_status=response.status,
        response_content_type=response.content_type,
        response_size_bytes=len(body.encode("utf-8")) if body else None,
        sets_cookie=response.sets_cookie,
        response_sample=target_text(body, source_url=url),
        observed_tech=tech,
    )

    finding_id = make_finding_id(scan_id, endpoint_id, alert.plugin_id or alert.name, param)
    evidence: list[UntrustedText] = []
    for value in (instance.get("evidence"), body):
        wrapped = target_text(value, source_url=url)
        if wrapped is not None:
            evidence.append(wrapped)
    for value in (instance.get("attack"), instance.get("otherinfo"), alert.other_info):
        wrapped = scanner_text(value, source_url=url)
        if wrapped is not None:
            evidence.append(wrapped)

    existing = findings.get(finding_id)
    if existing is not None:
        findings[finding_id] = existing.model_copy(
            update={"evidence": dedupe_untrusted(list(existing.evidence) + evidence)}
        )
        return

    findings[finding_id] = Finding(
        finding_id=finding_id,
        scan_id=scan_id,
        app_id=app_id,
        endpoint_id=endpoint_id,
        name=alert.name or "ZAP alert",
        cwe_id=extract_cwe(alert.cwe_id, alert.description, alert.name),
        cve_ids=extract_cves(
            alert.name,
            alert.description,
            alert.reference,
            alert.other_info,
            instance.get("otherinfo"),
            instance.get("evidence"),
        ),
        scanner="zap",
        scanner_plugin_id=alert.plugin_id,
        scanner_severity=severity,
        scanner_confidence=confidence,
        description=description
        or scanner_text(alert.name or "ZAP alert")
        or _fallback_description(),
        evidence=dedupe_untrusted(evidence),
        affected_component=fingerprint_library(
            instance.get("evidence"), instance.get("otherinfo"), uri, alert.other_info
        ),
        observed_at=scanned_at,
    )


def _join_message(header: str | None, body: str | None) -> str:
    if not header and not body:
        return ""
    return f"{(header or '').rstrip()}\r\n\r\n{body or ''}"


def _fallback_description() -> UntrustedText:
    """Findings must carry a description; an empty alert still gets a truthful one."""
    text = scanner_text("ZAP reported an alert with no description text.")
    assert text is not None  # the literal is non-empty
    return text
