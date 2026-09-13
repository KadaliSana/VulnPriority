"""NVD 2.0 feed: fixture and live implementations over one shared parser.

The fixture file stores payloads in the real NVD 2.0 shape, so ``parse_nvd_cve`` is the
only parser in the package and the offline tests exercise exactly the code the live feed
runs. The feed returns a *skeleton* :class:`VulnIntel`: description, CVSS records, affected
products and reference URLs. EPSS, KEV, exploit evidence and reference bodies are merged in
by :class:`~vulnpriority.feeds.bundle.DefaultIntelAssembler`.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar, Sequence

import httpx

from vulnpriority.core.enums import CvssVersion, Provenance, ScoreSource
from vulnpriority.core.interfaces import NvdFeed
from vulnpriority.core.models import AffectedProduct, CvssRecord, ReferenceDoc, UntrustedText, VulnIntel
from vulnpriority.feeds.base import FixtureFeed, LiveFeedBase, as_of_guard, parse_feed_date
from vulnpriority.feeds.cvss_policy import parse_cvss_vector

__all__ = [
    "NVD_API_URL",
    "NvdFixtureFeed",
    "NvdLiveFeed",
    "parse_nvd_cve",
    "parse_nvd_cvss",
    "parse_nvd_configurations",
    "reference_urls",
]

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

#: NVD 2.0 metric container -> CVSS version. Ordered newest first, purely cosmetically.
_METRIC_CONTAINERS: tuple[tuple[str, CvssVersion], ...] = (
    ("cvssMetricV40", CvssVersion.V40),
    ("cvssMetricV31", CvssVersion.V31),
    ("cvssMetricV30", CvssVersion.V30),
    ("cvssMetricV2", CvssVersion.V2),
)

#: cvssData field -> vector abbreviation, for payloads that omit ``vectorString``.
_V3_FIELDS: tuple[tuple[str, str], ...] = (
    ("attackVector", "AV"),
    ("attackComplexity", "AC"),
    ("attackRequirements", "AT"),
    ("privilegesRequired", "PR"),
    ("userInteraction", "UI"),
    ("scope", "S"),
    ("confidentialityImpact", "C"),
    ("integrityImpact", "I"),
    ("availabilityImpact", "A"),
    ("vulnConfidentialityImpact", "VC"),
    ("vulnIntegrityImpact", "VI"),
    ("vulnAvailabilityImpact", "VA"),
    ("subConfidentialityImpact", "SC"),
    ("subIntegrityImpact", "SI"),
    ("subAvailabilityImpact", "SA"),
)
_V2_FIELDS: tuple[tuple[str, str], ...] = (
    ("accessVector", "AV"),
    ("accessComplexity", "AC"),
    ("authentication", "AU"),
    ("confidentialityImpact", "C"),
    ("integrityImpact", "I"),
    ("availabilityImpact", "A"),
)

#: NVD spells metric values out; CVSS vectors abbreviate them.
_VALUE_ABBREV: dict[str, str] = {
    "NETWORK": "N",
    "ADJACENT_NETWORK": "A",
    "ADJACENT": "A",
    "LOCAL": "L",
    "PHYSICAL": "P",
    "LOW": "L",
    "MEDIUM": "M",
    "HIGH": "H",
    "NONE": "N",
    "SINGLE": "S",
    "MULTIPLE": "M",
    "REQUIRED": "R",
    "PASSIVE": "P",
    "ACTIVE": "A",
    "PARTIAL": "P",
    "COMPLETE": "C",
    "UNCHANGED": "U",
    "CHANGED": "C",
    "PRESENT": "P",
    "SAFETY": "S",
}


def _score_source(source: str | None, metric_type: str | None) -> ScoreSource:
    """Map NVD's ``source``/``type`` pair onto :class:`ScoreSource`.

    NVD's own analysis is published under a nist.gov identifier and marked ``Primary``;
    everything else is the CNA's own score.
    """
    identifier = (source or "").strip().lower()
    if "nist.gov" in identifier:
        return ScoreSource.NVD
    if (metric_type or "").strip().lower() == "primary":
        return ScoreSource.NVD
    if identifier:
        return ScoreSource.CNA
    return ScoreSource.OTHER


def _submetrics_from_fields(cvss_data: dict[str, Any], version: CvssVersion) -> dict[str, str]:
    fields = _V2_FIELDS if version == CvssVersion.V2 else _V3_FIELDS
    out: dict[str, str] = {}
    for field, abbrev in fields:
        value = cvss_data.get(field)
        if value is None:
            continue
        text = str(value).strip().upper()
        out[abbrev] = _VALUE_ABBREV.get(text, text[:1] if text else "")
    return out


def parse_nvd_cvss(metrics: dict[str, Any] | None) -> tuple[CvssRecord, ...]:
    """Every CVSS record in an NVD 2.0 ``metrics`` block, across all versions and sources."""
    records: list[CvssRecord] = []
    for container, version in _METRIC_CONTAINERS:
        for entry in (metrics or {}).get(container, ()) or ():
            if not isinstance(entry, dict):
                continue
            cvss_data = entry.get("cvssData") or {}
            score = cvss_data.get("baseScore")
            if score is None:
                continue
            vector = cvss_data.get("vectorString")
            submetrics = parse_cvss_vector(vector) or _submetrics_from_fields(cvss_data, version)
            severity = cvss_data.get("baseSeverity") or entry.get("baseSeverity")
            records.append(
                CvssRecord(
                    version=version,
                    source=_score_source(entry.get("source"), entry.get("type")),
                    base_score=max(0.0, min(10.0, float(score))),
                    vector=str(vector) if vector else None,
                    severity=str(severity) if severity else None,
                    submetrics=submetrics,
                )
            )
    return tuple(records)


def parse_nvd_configurations(configurations: Sequence[dict[str, Any]] | None) -> tuple[AffectedProduct, ...]:
    """Flatten ``configurations[].nodes[].cpeMatch[]`` into vulnerable product ranges."""
    products: list[AffectedProduct] = []
    seen: set[tuple[str, str | None, str | None, str | None, str | None]] = set()
    for configuration in configurations or ():
        if not isinstance(configuration, dict):
            continue
        for node in configuration.get("nodes", ()) or ():
            if not isinstance(node, dict):
                continue
            for match in node.get("cpeMatch", ()) or ():
                if not isinstance(match, dict) or not match.get("vulnerable", True):
                    continue
                criteria = match.get("criteria") or match.get("cpe23Uri")
                if not criteria:
                    continue
                identity = (
                    str(criteria),
                    match.get("versionStartIncluding"),
                    match.get("versionStartExcluding"),
                    match.get("versionEndIncluding"),
                    match.get("versionEndExcluding"),
                )
                if identity in seen:
                    continue
                seen.add(identity)
                products.append(
                    AffectedProduct(
                        cpe=str(criteria),
                        version_start_including=identity[1],
                        version_start_excluding=identity[2],
                        version_end_including=identity[3],
                        version_end_excluding=identity[4],
                    )
                )
    return tuple(products)


def _english_description(descriptions: Sequence[dict[str, Any]] | None) -> tuple[str, str]:
    """Preferred description text and its language tag; English first, else the first entry."""
    entries = [entry for entry in (descriptions or ()) if isinstance(entry, dict) and entry.get("value")]
    if not entries:
        return "", "und"
    for entry in entries:
        if str(entry.get("lang", "")).lower().startswith("en"):
            return str(entry["value"]), "en"
    first = entries[0]
    return str(first["value"]), str(first.get("lang") or "und")


def _reference_placeholders(references: Sequence[dict[str, Any]] | None) -> tuple[ReferenceDoc, ...]:
    """URL-only reference stubs.

    ``ReferenceDoc`` requires a body, but NVD supplies only links; the stub carries an empty
    :class:`UntrustedText` and the assembler replaces it with the fetched page when the
    reference fetcher can retrieve one.
    """
    docs: list[ReferenceDoc] = []
    seen: set[str] = set()
    for entry in references or ():
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not url or str(url) in seen:
            continue
        seen.add(str(url))
        tags = tuple(str(tag) for tag in (entry.get("tags") or ()))
        docs.append(
            ReferenceDoc(
                url=str(url),
                title=None,
                tags=tags,
                content=UntrustedText(text="", provenance=Provenance.REFERENCE_PAGE, source_url=str(url)),
            )
        )
    return tuple(docs)


def reference_urls(intel: VulnIntel | None) -> tuple[str, ...]:
    """Reference URLs carried by an NVD skeleton, in NVD's own order."""
    if intel is None:
        return ()
    return tuple(doc.url for doc in intel.references)


def parse_nvd_cve(cve: dict[str, Any], as_of: date) -> VulnIntel | None:
    """Turn one NVD 2.0 ``vulnerabilities[].cve`` object into a :class:`VulnIntel` skeleton.

    Returns ``None`` when the CVE was published after ``as_of``: at that date the record did
    not exist, and pretending otherwise is exactly the leakage the protocol forbids.
    ``lastModified`` is dropped rather than clamped when it post-dates the cut-off, because a
    fabricated revision date is worse than a missing one.
    """
    cve_id = str(cve.get("id") or "").strip()
    if not cve_id:
        return None
    published = parse_feed_date(cve.get("published"))
    if published is not None and not as_of_guard(published, as_of):
        return None
    last_modified = parse_feed_date(cve.get("lastModified"))
    if last_modified is not None and not as_of_guard(last_modified, as_of):
        last_modified = None

    text, language = _english_description(cve.get("descriptions"))
    description = (
        UntrustedText(text=text, provenance=Provenance.NVD, language=language) if text else None
    )
    return VulnIntel(
        cve_id=cve_id.upper(),
        as_of=as_of,
        description=description,
        published=published,
        last_modified=last_modified,
        cvss=parse_nvd_cvss(cve.get("metrics")),
        affected=parse_nvd_configurations(cve.get("configurations")),
        references=_reference_placeholders(cve.get("references")),
    )


def _unwrap_vulnerabilities(payload: Any) -> list[dict[str, Any]]:
    """``{"vulnerabilities": [{"cve": {...}}]}`` -> a list of ``cve`` objects."""
    if isinstance(payload, dict):
        if "vulnerabilities" in payload:
            items = payload.get("vulnerabilities") or []
        elif "cve" in payload:
            items = [payload]
        else:
            items = [{"cve": payload}]
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item.get("cve") if isinstance(item.get("cve"), dict) else item)
    return [item for item in out if isinstance(item, dict)]


class NvdFixtureFeed(FixtureFeed, NvdFeed):
    """Offline NVD feed over ``data/fixtures/feeds/nvd/cves.json`` (real NVD 2.0 shape)."""

    fixture_relpath: ClassVar[str] = "nvd/cves.json"
    name = "nvd"

    def _extract(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for envelope in raw:
            records.extend(_unwrap_vulnerabilities(envelope))
        return records

    def _key_of(self, record: dict[str, Any]) -> str | Sequence[str]:
        return str(record.get("id") or "")

    def _build(self, entries: tuple[dict[str, Any], ...], key: str, as_of: date) -> VulnIntel | None:
        if not entries:
            return None
        return parse_nvd_cve(entries[0], as_of)


class NvdLiveFeed(LiveFeedBase, NvdFeed):
    """NVD 2.0 REST API. Sends ``apiKey`` when one is configured (higher rate limit)."""

    name = "nvd"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = NVD_API_URL,
        timeout_s: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        headers = {"apiKey": api_key} if api_key else None
        super().__init__(timeout_s=timeout_s, client=client, headers=headers)
        self.base_url = base_url
        self.api_key = api_key

    def get(self, key: str, as_of: date) -> VulnIntel | None:
        payload = self._request_json(self.base_url, params={"cveId": str(key).strip().upper()})
        if payload is None:
            return None
        for cve in _unwrap_vulnerabilities(payload):
            if str(cve.get("id") or "").strip().upper() != str(key).strip().upper():
                continue
            return parse_nvd_cve(cve, as_of)
        return None
