"""CISA Known Exploited Vulnerabilities feed.

KEV is the single most leakage-prone feed in the framework: the catalogue is a *current*
snapshot with no history, so naively joining it against a scan from last year tells the
model which vulnerabilities would later be exploited. ``date_added`` is therefore honoured
strictly - a CVE added after ``as_of`` comes back with ``in_kev=False`` and no dates at all,
so neither the flag nor the age feature can smuggle the future in.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar, Sequence

import httpx

from vulnprio.core.interfaces import KevFeed
from vulnprio.core.models import KevRecord
from vulnprio.feeds.base import FixtureFeed, LiveFeedBase, as_of_guard, parse_feed_date

__all__ = ["KEV_CATALOG_URL", "KevFixtureFeed", "KevLiveFeed", "kev_record_from_entry"]

KEV_CATALOG_URL = (
    "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
)

#: CISA writes this field as a human-readable word, not a boolean.
_RANSOMWARE_KNOWN = "known"


def kev_record_from_entry(
    entry: dict[str, Any] | None, cve_id: str, as_of: date
) -> KevRecord:
    """Build the as-of view of one KEV catalogue entry.

    A record is always returned: "not in the catalogue as of this date" is a real answer and
    a useful feature, and returning ``None`` would force every caller to re-implement that.
    """
    cve = cve_id.strip().upper()
    if not entry:
        return KevRecord(cve_id=cve, in_kev=False, as_of=as_of)
    date_added = parse_feed_date(entry.get("dateAdded") or entry.get("date_added"))
    if date_added is None or not as_of_guard(date_added, as_of):
        # Added later than the cut-off (or undated): as of ``as_of`` this CVE was not in KEV.
        return KevRecord(cve_id=cve, in_kev=False, as_of=as_of)
    ransomware = str(entry.get("knownRansomwareCampaignUse") or entry.get("known_ransomware_use") or "")
    return KevRecord(
        cve_id=cve,
        in_kev=True,
        date_added=date_added,
        due_date=parse_feed_date(entry.get("dueDate") or entry.get("due_date")),
        known_ransomware_use=ransomware.strip().lower() == _RANSOMWARE_KNOWN
        or ransomware.strip().lower() == "true",
        as_of=as_of,
    )


def _unwrap_catalog(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get("vulnerabilities") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


class KevFixtureFeed(FixtureFeed, KevFeed):
    """Offline KEV feed over ``data/fixtures/feeds/kev/kev.json`` (real CISA shape)."""

    fixture_relpath: ClassVar[str] = "kev/kev.json"
    name = "kev"

    def _extract(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for envelope in raw:
            records.extend(_unwrap_catalog(envelope))
        return records

    def _key_of(self, record: dict[str, Any]) -> str | Sequence[str]:
        return str(record.get("cveID") or record.get("cve_id") or "")

    def _build(self, entries: tuple[dict[str, Any], ...], key: str, as_of: date) -> KevRecord:
        return kev_record_from_entry(entries[0] if entries else None, key, as_of)


class KevLiveFeed(LiveFeedBase, KevFeed):
    """CISA KEV catalogue.

    The catalogue is one JSON document for every CVE, so it is fetched at most once per
    instance and indexed in memory; per-CVE requests would be pointless traffic.
    """

    name = "kev"

    def __init__(
        self,
        *,
        catalog_url: str = KEV_CATALOG_URL,
        timeout_s: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(timeout_s=timeout_s, client=client)
        self.catalog_url = catalog_url
        self._catalog: dict[str, dict[str, Any]] | None = None

    def catalog(self) -> dict[str, dict[str, Any]]:
        """CVE id to catalogue entry, fetched once and memoised on the instance."""
        if self._catalog is None:
            payload = self._request_json(self.catalog_url)
            index: dict[str, dict[str, Any]] = {}
            for entry in _unwrap_catalog(payload):
                cve = str(entry.get("cveID") or "").strip().upper()
                if cve and cve not in index:
                    index[cve] = entry
            self._catalog = index
        return self._catalog

    def get(self, key: str, as_of: date) -> KevRecord:
        cve_id = str(key).strip().upper()
        return kev_record_from_entry(self.catalog().get(cve_id), cve_id, as_of)
