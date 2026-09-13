"""Synthetic feed fixtures in the exact shapes ``vulnprio.feeds`` reads (DESIGN.md 3.11).

The generated world has to be consumed by the *real* feed readers, not by a synthetic
shortcut, or the offline pipeline would exercise code that never runs in a live one. So this
module emits:

* ``nvd/cves.json`` in the NVD 2.0 envelope :class:`~vulnprio.feeds.nvd.NvdFixtureFeed`
  unwraps, with ``metrics`` containing **two disagreeing CVSS v3.1 records** (NVD primary and
  a CNA secondary) and sometimes a v2 record, which is what gives
  :func:`~vulnprio.feeds.cvss_policy.select_cvss` a real selection to make and produces a
  ``cvss_source_agreement`` below 1 (Gap 3);
* ``epss/epss_snapshots.jsonl`` as one row per (CVE, date), so the as-of rule in
  :class:`~vulnprio.feeds.epss.EpssFixtureFeed` returns different evidence in March than in
  September;
* ``kev/kev.json`` in the CISA catalogue shape, with ``dateAdded`` honoured strictly;
* ``exploitdb/exploits.jsonl`` with the Exploit-DB column names and a declared maturity;
* ``references/index.json`` for :class:`~vulnprio.feeds.references.ReferenceFixtureFetcher`.

Every file carries the whole time series. Cutting it to an as-of date is the feeds' job, and
duplicating that logic here would let a bug in one hide a bug in the other.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from vulnprio.core.enums import ExploitMaturity
from vulnprio.synth.world import LatentVuln, LatentWorld

__all__ = [
    "FEED_FILES",
    "severity_word",
    "nvd_document",
    "nvd_catalog",
    "epss_rows",
    "kev_catalog",
    "exploit_rows",
    "write_feed_fixtures",
    "write_reference_index",
]

#: Relative paths inside a fixture directory, matching every ``fixture_relpath`` in
#: :mod:`vulnprio.feeds`.
FEED_FILES: dict[str, str] = {
    "nvd": "nvd/cves.json",
    "epss": "epss/epss_snapshots.jsonl",
    "kev": "kev/kev.json",
    "exploits": "exploitdb/exploits.jsonl",
    "references": "references/index.json",
}

_MATURITY_WORD: dict[ExploitMaturity, str] = {
    ExploitMaturity.UNKNOWN: "unknown",
    ExploitMaturity.UNPROVEN: "unproven",
    ExploitMaturity.POC: "poc",
    ExploitMaturity.FUNCTIONAL: "functional",
    ExploitMaturity.WEAPONIZED: "weaponized",
}

#: CVSS v3.1 metric abbreviation -> the spelled-out value NVD publishes.
_AV = {"N": "NETWORK", "A": "ADJACENT_NETWORK", "L": "LOCAL", "P": "PHYSICAL"}
_LMH = {"L": "LOW", "H": "HIGH", "N": "NONE", "R": "REQUIRED", "U": "UNCHANGED", "C": "CHANGED"}


def severity_word(score: float) -> str:
    """CVSS v3.1 qualitative severity for a base score."""
    if score >= 9.0:
        return "CRITICAL"
    if score >= 7.0:
        return "HIGH"
    if score >= 4.0:
        return "MEDIUM"
    if score > 0.0:
        return "LOW"
    return "NONE"


def _iso_datetime(value: date, hour: int = 12) -> str:
    return datetime(value.year, value.month, value.day, hour, 0, 0).isoformat(timespec="milliseconds")


def _submetrics(vector: str) -> dict[str, str]:
    parts = [chunk for chunk in vector.split("/") if ":" in chunk and not chunk.startswith("CVSS")]
    return {key: value for key, _, value in (chunk.partition(":") for chunk in parts)}


def _cvss_v31_entry(vuln: LatentVuln, score: float, *, source: str, kind: str) -> dict[str, Any]:
    submetrics = _submetrics(vuln.cvss_vector)
    return {
        "source": source,
        "type": kind,
        "cvssData": {
            "version": "3.1",
            "vectorString": vuln.cvss_vector,
            "attackVector": _AV.get(submetrics.get("AV", "N"), "NETWORK"),
            "attackComplexity": _LMH.get(submetrics.get("AC", "L"), "LOW"),
            "privilegesRequired": _LMH.get(submetrics.get("PR", "N"), "NONE"),
            "userInteraction": _LMH.get(submetrics.get("UI", "N"), "NONE"),
            "scope": _LMH.get(submetrics.get("S", "U"), "UNCHANGED"),
            "confidentialityImpact": _LMH.get(submetrics.get("C", "N"), "NONE"),
            "integrityImpact": _LMH.get(submetrics.get("I", "N"), "NONE"),
            "availabilityImpact": _LMH.get(submetrics.get("A", "N"), "NONE"),
            "baseScore": score,
            "baseSeverity": severity_word(score),
        },
        "exploitabilityScore": round(min(3.9, 0.4 * score), 1),
        "impactScore": round(min(6.0, 0.6 * score), 1),
    }


def _cvss_v2_entry(vuln: LatentVuln, score: float) -> dict[str, Any]:
    submetrics = _submetrics(vuln.cvss_vector)
    complexity = "LOW" if submetrics.get("AC", "L") == "L" else "HIGH"
    impact = "PARTIAL" if score < 8.0 else "COMPLETE"
    return {
        "source": "nvd@nist.gov",
        "type": "Primary",
        "cvssData": {
            "version": "2.0",
            "vectorString": f"AV:N/AC:{'L' if complexity == 'LOW' else 'H'}/Au:N/C:P/I:P/A:P",
            "accessVector": "NETWORK",
            "accessComplexity": complexity,
            "authentication": "NONE",
            "confidentialityImpact": impact,
            "integrityImpact": impact,
            "availabilityImpact": "PARTIAL",
            "baseScore": score,
        },
        "baseSeverity": severity_word(score),
        "exploitabilityScore": 10.0,
        "impactScore": 6.4,
    }


def nvd_document(vuln: LatentVuln) -> dict[str, Any]:
    """One ``vulnerabilities[].cve`` object in the NVD 2.0 shape."""
    metrics: dict[str, Any] = {
        "cvssMetricV31": [
            _cvss_v31_entry(vuln, vuln.cvss_nvd, source="nvd@nist.gov", kind="Primary"),
            _cvss_v31_entry(
                vuln, vuln.cvss_cna, source=f"security@{vuln.vendor}.example", kind="Secondary"
            ),
        ]
    }
    if vuln.cvss_v2 is not None:
        metrics["cvssMetricV2"] = [_cvss_v2_entry(vuln, vuln.cvss_v2)]

    last_modified = vuln.kev_date_added or vuln.exploit_published or vuln.published
    return {
        "id": vuln.cve_id,
        "sourceIdentifier": "cve@mitre.org",
        "published": _iso_datetime(vuln.published, hour=15),
        "lastModified": _iso_datetime(max(last_modified, vuln.published), hour=11),
        "vulnStatus": "Analyzed",
        "descriptions": [{"lang": "en", "value": vuln.description}],
        "metrics": metrics,
        "weaknesses": [
            {
                "source": "nvd@nist.gov",
                "type": "Primary",
                "description": [{"lang": "en", "value": f"CWE-{vuln.cwe_id}"}],
            }
        ],
        "configurations": [
            {
                "nodes": [
                    {
                        "operator": "OR",
                        "negate": False,
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": vuln.cpe,
                                "versionEndExcluding": vuln.version_end_excluding,
                                "matchCriteriaId": f"MC-{vuln.cve_id}",
                            }
                        ],
                    }
                ]
            }
        ],
        "references": [
            {"url": url, "source": "cve@mitre.org", "tags": _reference_tags(url)}
            for url in vuln.reference_urls
        ],
    }


def _reference_tags(url: str) -> list[str]:
    lowered = url.lower()
    if "exploit-db" in lowered:
        return ["Exploit", "Third Party Advisory"]
    if "cisa.gov" in lowered:
        return ["US Government Resource"]
    if "github.com" in lowered:
        return ["Third Party Advisory"]
    if "nvd.nist.gov" in lowered:
        return ["Vendor Advisory"]
    return ["Technical Description"]


def nvd_catalog(world: LatentWorld) -> dict[str, Any]:
    """The whole generated universe in one NVD 2.0 response envelope."""
    documents = [nvd_document(vuln) for vuln in world.vulns]
    return {
        "resultsPerPage": len(documents),
        "startIndex": 0,
        "totalResults": len(documents),
        "format": "NVD_CVE",
        "version": "2.0",
        "timestamp": _iso_datetime(max((v.published for v in world.vulns), default=date(2024, 1, 1)), 0),
        "vulnerabilities": [{"cve": document} for document in documents],
    }


def epss_rows(world: LatentWorld) -> list[dict[str, str]]:
    """One row per (CVE, snapshot date), with FIRST's six-decimal string formatting."""
    return [
        {
            "cve": point.cve_id,
            "date": point.as_of.isoformat(),
            "epss": f"{point.score:.6f}",
            "percentile": f"{point.percentile:.6f}",
        }
        for point in world.epss_series()
    ]


def kev_catalog(world: LatentWorld, *, released: date | None = None) -> dict[str, Any]:
    """CISA KEV catalogue holding every CVE that ever enters it, with its ``dateAdded``."""
    members = [vuln for vuln in world.vulns if vuln.in_kev and vuln.kev_date_added is not None]
    members.sort(key=lambda vuln: (vuln.kev_date_added or date.min, vuln.cve_id))
    release_date = released or max(
        (vuln.kev_date_added for vuln in members if vuln.kev_date_added), default=date(2024, 1, 1)
    )
    return {
        "title": "CISA Catalog of Known Exploited Vulnerabilities",
        "catalogVersion": release_date.strftime("%Y.%m.%d"),
        "dateReleased": f"{release_date.isoformat()}T14:00:00.0000Z",
        "count": len(members),
        "vulnerabilities": [
            {
                "cveID": vuln.cve_id,
                "vendorProject": vuln.vendor.title(),
                "product": vuln.product,
                "vulnerabilityName": f"{vuln.vendor.title()} {vuln.product} {vuln.name}",
                "dateAdded": (vuln.kev_date_added or vuln.published).isoformat(),
                "shortDescription": vuln.description,
                "requiredAction": "Apply updates per vendor instructions.",
                "dueDate": (vuln.kev_due_date or vuln.published).isoformat(),
                "knownRansomwareCampaignUse": "Known" if vuln.kev_ransomware else "Unknown",
                "notes": f"https://nvd.nist.gov/vuln/detail/{vuln.cve_id}",
            }
            for vuln in members
        ],
    }


def exploit_rows(world: LatentWorld) -> list[dict[str, Any]]:
    """Exploit-DB index rows for every CVE that acquired a public exploit."""
    rows: list[dict[str, Any]] = []
    for vuln in world.vulns:
        if vuln.exploit_published is None or not vuln.exploit_id:
            continue
        weaponized = vuln.exploit_maturity >= ExploitMaturity.WEAPONIZED
        rows.append(
            {
                "codes": vuln.cve_id,
                "cve": vuln.cve_id,
                "date_published": vuln.exploit_published.isoformat(),
                "id": vuln.exploit_id,
                "language": "en",
                "maturity": _MATURITY_WORD[ExploitMaturity(vuln.exploit_maturity)],
                "platform": "multiple",
                "source": "exploit_db",
                "tags": "Metasploit Framework (MSF)" if weaponized else "",
                "title": vuln.exploit_title,
                "type": "webapps",
                "url": f"https://www.exploit-db.com/exploits/{vuln.exploit_id}",
                "verified": 1 if vuln.exploit_verified else 0,
            }
        )
    rows.sort(key=lambda row: (row["date_published"], row["cve"]))
    return rows


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    return path


def write_reference_index(pages: Sequence[dict[str, Any]], fixture_dir: str | Path) -> Path:
    """Write ``references/index.json`` from the pages built by :mod:`vulnprio.synth.pages`."""
    target = Path(fixture_dir) / FEED_FILES["references"]
    return _write_json(target, list(pages))


def write_feed_fixtures(
    world: LatentWorld,
    fixture_dir: str | Path,
    *,
    pages: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write every feed fixture for ``world`` under ``fixture_dir``.

    The returned directory is directly usable as ``FeedsConfig.fixture_dir``: no adapter, no
    translation layer, the same readers a live-mode cache replay would use.
    """
    root = Path(fixture_dir)
    written = {
        "nvd": _write_json(root / FEED_FILES["nvd"], nvd_catalog(world)),
        "epss": _write_jsonl(root / FEED_FILES["epss"], epss_rows(world)),
        "kev": _write_json(root / FEED_FILES["kev"], kev_catalog(world)),
        "exploits": _write_jsonl(root / FEED_FILES["exploits"], exploit_rows(world)),
    }
    written["references"] = write_reference_index(pages or (), root)
    return written
