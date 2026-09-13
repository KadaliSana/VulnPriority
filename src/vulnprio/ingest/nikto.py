"""Nikto JSON report parsing.

Nikto is a web server misconfiguration scanner: it reports missing headers, interesting
files, dangerous methods and outdated server software against one host. Two properties
shape this parser:

* **Nikto assigns no severity and no CWE.** Its output is a test id and a sentence of
  English. Inventing a severity from that sentence would be dressing a guess up as data,
  so findings default to ``LOW`` and only rise when the message matches a pattern whose
  meaning is unambiguous. The CWE mapping is built the same way, and leaves ``None``
  rather than guessing. This is not a gap the parser should paper over: deciding what
  matters from weak scanner output is exactly what the rest of this framework is for.
* **The test id is a real plugin id.** ``id`` (for example ``999957``) is stable across
  runs and across hosts, so it becomes ``scanner_plugin_id`` and therefore part of the
  dedup key - one missing header on forty hosts is one thing to fix.

Both report dialects are handled: a single host object, and the list of host objects older
versions emit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vulnprio.core.enums import ScannerSeverity
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
    target_text,
)
from vulnprio.ingest.tech_fingerprint import fingerprint_library, fingerprint_response, parse_product_tokens

__all__ = ["NiktoParser"]

#: Message pattern to CWE. Every entry is a phrase Nikto emits verbatim and whose meaning
#: is not in doubt. Anything unmatched falls through to free-text extraction and then to
#: ``None``: a wrong CWE joins the finding to the wrong intelligence downstream, which is
#: worse than having no CWE at all.
_MESSAGE_CWE: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"(?i)x-frame-options|clickjacking"), 1021),
    (re.compile(r"(?i)x-content-type-options|content-type-options header"), 693),
    (re.compile(r"(?i)strict-transport-security|content-security-policy|permissions-policy"), 693),
    (re.compile(r"(?i)cookie.*without.*httponly|httponly flag"), 1004),
    (re.compile(r"(?i)cookie.*without.*secure|secure flag"), 614),
    (re.compile(r"(?i)directory indexing|index of /|directory listing"), 548),
    (re.compile(r"(?i)leaks inodes|etag|server banner|x-powered-by|reveals?.*version|phpinfo"), 200),
    (re.compile(r"(?i)\bTRACE\b|allowed http methods|method.*is allowed|webdav|\bPUT\b.*allowed"), 650),
    (re.compile(r"(?i)backup|\.bak\b|\.old\b|\.git|\.svn|\.env\b|config file"), 530),
    (re.compile(r"(?i)default (?:account|credential|password|login)"), 1392),
    (re.compile(r"(?i)sql injection"), 89),
    (re.compile(r"(?i)cross[- ]site scripting|\bxss\b"), 79),
    (re.compile(r"(?i)traversal|\.\./"), 22),
    (re.compile(r"(?i)remote file (?:inclusion|retrieval)"), 98),
    (re.compile(r"(?i)command execution|remote code"), 78),
    (re.compile(r"(?i)outdated|is out of date|appears to be outdated"), 1104),
    (re.compile(r"(?i)admin login|administration (?:page|interface)|interesting file"), 538),
)

#: Message pattern to severity, applied in order. Nikto reports none, so these are the
#: only claims this parser makes about how much a finding matters, and they are limited to
#: phrases whose consequence is well understood.
_MESSAGE_SEVERITY: tuple[tuple[re.Pattern[str], ScannerSeverity], ...] = (
    (re.compile(r"(?i)remote code|command execution|sql injection|arbitrary file|web ?shell"),
     ScannerSeverity.HIGH),
    (re.compile(r"(?i)default (?:account|credential|password)|traversal|remote file inclusion"),
     ScannerSeverity.HIGH),
    (re.compile(r"(?i)directory indexing|backup|\.git|\.env\b|phpinfo|is vulnerable|may allow"),
     ScannerSeverity.MEDIUM),
    (re.compile(r"(?i)\bTRACE\b|webdav|\bPUT\b.*allowed|outdated|out of date"),
     ScannerSeverity.MEDIUM),
    (re.compile(r"(?i)header is not present|uncommon header|retrieved|no cgi directories"),
     ScannerSeverity.LOW),
)

#: Nikto is a probe-and-observe scanner: a hit is a real observation, but its messages are
#: frequently generic, so confidence sits below a template match and above a guess.
_CONFIDENCE = 0.6


@dataclass
class _Host:
    """One scanned host, with the findings recorded against it."""

    host: str = ""
    ip: str = ""
    port: str = "80"
    banner: str = ""
    scanned_at: Any = None
    items: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.items is None:
            self.items = []

    @property
    def scheme(self) -> str:
        """Nikto records the port, not the scheme; 443 and 8443 mean TLS."""
        return "https" if str(self.port).strip() in {"443", "8443"} else "http"

    @property
    def base(self) -> str:
        name = (self.host or self.ip).strip()
        if not name:
            return ""
        port = str(self.port).strip()
        default = (self.scheme == "https" and port == "443") or (self.scheme == "http" and port == "80")
        netloc = name if default or not port else f"{name}:{port}"
        return f"{self.scheme}://{netloc}"


@register_parser("nikto")
class NiktoParser(ScannerParser):
    """``ScannerParser`` for Nikto JSON output (``nikto -Format json -output file``)."""

    name = "nikto"

    def sniff(self, path: str | Path) -> bool:
        """True for Nikto's host object (or list of them) with a ``vulnerabilities`` *list*.

        The list test separates Nikto from Wapiti, whose ``vulnerabilities`` is a mapping
        keyed by category.
        """
        data = _load(path)
        for record in _as_hosts(data):
            items = record.get("vulnerabilities")
            if not isinstance(items, list):
                return False
            return bool({"host", "ip", "banner", "port"} & set(record))
        return False

    def parse(self, path: str | Path, app_id: str | None = None) -> Scan:
        """Read a Nikto JSON report into a :class:`Scan`."""
        source = Path(path)
        data = _load(source)
        hosts = [_read_host(record) for record in _as_hosts(data)]
        if not hosts:
            raise ParseError(f"no Nikto host records found in {source}")
        return _build_scan(hosts, app_id)


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


def _as_hosts(data: Any) -> list[dict[str, Any]]:
    """Both dialects: a single host object, or the list older versions write."""
    if isinstance(data, dict):
        if isinstance(data.get("niktoscan"), list):     # some builds wrap the list
            return [item for item in data["niktoscan"] if isinstance(item, dict)]
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _read_host(record: dict[str, Any]) -> _Host:
    items = record.get("vulnerabilities")
    return _Host(
        host=_as_text(record.get("host")),
        ip=_as_text(record.get("ip")),
        port=_as_text(record.get("port")) or "80",
        banner=_as_text(record.get("banner")),
        scanned_at=parse_timestamp(record.get("starttime") or record.get("start_time")),
        items=[item for item in items if isinstance(item, dict)] if isinstance(items, list) else [],
    )


def _severity_of(message: str) -> ScannerSeverity:
    for pattern, severity in _MESSAGE_SEVERITY:
        if pattern.search(message):
            return severity
    return ScannerSeverity.LOW


def _cwe_of(message: str, references: str) -> int | None:
    for pattern, cwe_id in _MESSAGE_CWE:
        if pattern.search(message):
            return cwe_id
    return extract_cwe(message, references)


def _build_scan(hosts: list[_Host], app_id: str | None) -> Scan:
    primary = next((host for host in hosts if host.base), hosts[0])
    scanned_at = next(
        (host.scanned_at for host in hosts if host.scanned_at is not None), DEFAULT_SCANNED_AT
    )
    primary_host = (primary.host or primary.ip).strip().lower()
    resolved_app_id = app_id or make_app_id(primary_host or "nikto")
    scan_id = make_scan_id(resolved_app_id, scanned_at, "nikto")

    accumulator = EndpointAccumulator(resolved_app_id)
    findings: dict[str, Finding] = {}

    for host in hosts:
        base = host.base or None
        banner_tech = parse_product_tokens(host.banner)
        if banner_tech:
            accumulator.add_tech(banner_tech)

        for item in host.items:
            message = _as_text(item.get("msg"))
            references = _as_text(item.get("references"))
            url = canonical_url(_as_text(item.get("url")) or "/", base=base)
            method = http_method(item.get("method"), )

            endpoint_id = accumulator.add(
                url,
                method,
                base=base,
                observed_tech=fingerprint_response(
                    headers={"server": host.banner} if host.banner else None, url=url
                ),
            )
            plugin_id = _as_text(item.get("id")) or None
            finding_id = make_finding_id(scan_id, endpoint_id, plugin_id or message[:60], "")

            evidence: list[UntrustedText] = []
            observed = target_text(message, source_url=url)
            if observed is not None:
                evidence.append(observed)

            existing = findings.get(finding_id)
            if existing is not None:
                findings[finding_id] = existing.model_copy(
                    update={"evidence": dedupe_untrusted(list(existing.evidence) + evidence)}
                )
                continue

            description = scanner_text(
                " ".join(part for part in (message, references) if part)
            ) or scanner_text("Nikto finding")
            if description is None:  # pragma: no cover - the literal is never empty
                raise ParseError("Nikto item carries no usable description")

            findings[finding_id] = Finding(
                finding_id=finding_id,
                scan_id=scan_id,
                app_id=resolved_app_id,
                endpoint_id=endpoint_id,
                name=_name_of(message, plugin_id),
                cwe_id=_cwe_of(message, references),
                cve_ids=extract_cves(message, references),
                scanner="nikto",
                scanner_plugin_id=plugin_id,
                scanner_severity=_severity_of(message),
                scanner_confidence=_CONFIDENCE,
                description=description,
                evidence=dedupe_untrusted(evidence),
                affected_component=fingerprint_library(host.banner, message),
                observed_at=scanned_at,
            )

    hosts_seen = accumulator.hosts() or ((primary_host,) if primary_host else ())
    return Scan(
        scan_id=scan_id,
        app_id=resolved_app_id,
        app_name=hosts_seen[0] if hosts_seen else "unknown",
        scanned_at=scanned_at,
        scanner_name="nikto",
        scanner_version=None,
        hosts=hosts_seen,
        tech_stack=accumulator.tech_stack(),
        endpoints=accumulator.endpoints(),
        findings=tuple(findings.values()),
    )


def _name_of(message: str, plugin_id: str | None) -> str:
    """A short title from Nikto's sentence: the first clause, capped."""
    cleaned = " ".join(message.split())
    if not cleaned:
        return f"Nikto test {plugin_id}" if plugin_id else "Nikto finding"
    head = re.split(r"(?<=[a-z])\. |: ", cleaned)[0]
    return head[:120].rstrip(" .,:;")
