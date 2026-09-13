"""FIRST EPSS feed: per-date exploit-prediction snapshots.

EPSS is a time series, and that is the whole point: a score published in September says
nothing about what a triage team knew in June. Both implementations therefore return the
**latest snapshot at or before** ``as_of`` and drop anything later, which is what makes the
time-ordered evaluation in DESIGN.md 4 honest.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar, Sequence

import httpx

from vulnpriority.core.interfaces import EpssFeed
from vulnpriority.core.models import EpssRecord
from vulnpriority.feeds.base import FixtureFeed, LiveFeedBase, as_of_guard, parse_feed_date

__all__ = ["EPSS_API_URL", "EpssFixtureFeed", "EpssLiveFeed", "parse_epss_rows"]

EPSS_API_URL = "https://api.first.org/data/v1/epss"


def _clip_unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def parse_epss_rows(
    rows: Sequence[dict[str, Any]] | None, cve_id: str, as_of: date
) -> EpssRecord | None:
    """Latest admissible snapshot for ``cve_id``.

    Rows dated after ``as_of`` are discarded rather than clamped, and ties on the same date
    resolve to the last row so the answer does not depend on iteration order of equal keys.
    """
    best: EpssRecord | None = None
    wanted = cve_id.strip().upper()
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        row_cve = str(row.get("cve") or row.get("cve_id") or "").strip().upper()
        if row_cve and row_cve != wanted:
            continue
        snapshot = parse_feed_date(row.get("date"))
        if snapshot is None or not as_of_guard(snapshot, as_of):
            continue
        if best is not None and snapshot < best.as_of:
            continue
        best = EpssRecord(
            cve_id=wanted,
            score=_clip_unit(row.get("epss", row.get("score"))),
            percentile=_clip_unit(row.get("percentile")),
            as_of=snapshot,
        )
    return best


class EpssFixtureFeed(FixtureFeed, EpssFeed):
    """Offline EPSS feed over ``data/fixtures/feeds/epss/epss_snapshots.jsonl``."""

    fixture_relpath: ClassVar[str] = "epss/epss_snapshots.jsonl"
    name = "epss"

    def _key_of(self, record: dict[str, Any]) -> str | Sequence[str]:
        return str(record.get("cve") or record.get("cve_id") or "")

    def _build(self, entries: tuple[dict[str, Any], ...], key: str, as_of: date) -> EpssRecord | None:
        if not entries:
            return None
        ordered = sorted(entries, key=lambda row: str(row.get("date") or ""))
        return parse_epss_rows(ordered, key, as_of)


class EpssLiveFeed(LiveFeedBase, EpssFeed):
    """FIRST EPSS API (``/data/v1/epss?cve=...&date=...``).

    The ``date`` parameter asks the API for the historical snapshot directly; the response
    is still filtered locally, because a feed must be correct even when upstream is not.
    """

    name = "epss"

    def __init__(
        self,
        *,
        base_url: str = EPSS_API_URL,
        timeout_s: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(timeout_s=timeout_s, client=client)
        self.base_url = base_url

    def get(self, key: str, as_of: date) -> EpssRecord | None:
        cve_id = str(key).strip().upper()
        payload = self._request_json(
            self.base_url, params={"cve": cve_id, "date": as_of.isoformat()}
        )
        if not isinstance(payload, dict):
            return None
        rows = payload.get("data")
        if not isinstance(rows, list):
            return None
        return parse_epss_rows(rows, cve_id, as_of)
