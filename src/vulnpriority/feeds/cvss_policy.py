"""CVSS selection policy and submetric flags (DESIGN.md 3.2, Gap 3).

A CVE routinely carries several CVSS records: different versions, and different scoring
organisations that disagree. Most prioritisation work silently picks one. Here the choice
is an explicit, testable policy, and the *disagreement itself* becomes a feature so the
instrument's own inconsistency is measured rather than hidden.

Policy:

* the newest CVSS version present wins (4.0 > 3.1 > 3.0 > 2.0);
* among records of that version, ``NVD`` beats ``CNA`` beats ``SCANNER`` beats ``OTHER``;
* remaining ties break on the higher base score, then on the vector string, so the result
  never depends on dictionary or file ordering;
* ``source_agreement = 1 - (max - min) / 10`` over one base score per scoring source (each
  source contributes its own newest-version score, so a v2/v3.1 pair from the *same* source
  is a version difference, not a disagreement).
"""

from __future__ import annotations

from vulnpriority.core.enums import CvssVersion, ScoreSource
from vulnpriority.core.models import CvssRecord

__all__ = [
    "CVSS_VERSION_ORDER",
    "SOURCE_PRIORITY",
    "SUBMETRIC_FLAG_NAMES",
    "cvss_version_ordinal",
    "parse_cvss_vector",
    "select_cvss",
    "source_agreement",
    "submetric_flags",
    "cvss_features",
]

#: Newest version last. Used both for selection and for the ``cvss_version_ord`` feature.
CVSS_VERSION_ORDER: dict[CvssVersion, int] = {
    CvssVersion.V2: 0,
    CvssVersion.V30: 1,
    CvssVersion.V31: 2,
    CvssVersion.V40: 3,
}

#: Lower wins. NVD is the analysed, comparable score; a CNA scores its own product.
SOURCE_PRIORITY: dict[ScoreSource, int] = {
    ScoreSource.NVD: 0,
    ScoreSource.CNA: 1,
    ScoreSource.SCANNER: 2,
    ScoreSource.OTHER: 3,
}

SUBMETRIC_FLAG_NAMES: tuple[str, ...] = (
    "cvss_ac_low",
    "cvss_pr_none",
    "cvss_ui_none",
    "cvss_c_high",
    "cvss_i_high",
    "cvss_a_high",
)


def cvss_version_ordinal(version: CvssVersion) -> int:
    """Integer rank of a CVSS version; unknown versions sort oldest."""
    return CVSS_VERSION_ORDER.get(version, 0)


def parse_cvss_vector(vector: str | None) -> dict[str, str]:
    """Split a CVSS vector string into its metric abbreviations.

    Handles the ``CVSS:3.1/`` prefix used by v3 and v4 and the bare v2 form
    ``AV:N/AC:L/Au:N/C:P/I:P/A:P``. Keys are upper-cased so ``Au`` and ``AU`` agree.
    """
    if not vector:
        return {}
    parsed: dict[str, str] = {}
    for part in str(vector).split("/"):
        if ":" not in part:
            continue
        key, _, value = part.partition(":")
        key = key.strip().upper()
        if key == "CVSS":
            continue
        parsed[key] = value.strip().upper()
    return parsed


def _metrics_of(record: CvssRecord) -> dict[str, str]:
    """Submetrics for a record, falling back to the vector string when the dict is empty."""
    metrics = {str(k).strip().upper(): str(v).strip().upper() for k, v in (record.submetrics or {}).items()}
    if metrics:
        return metrics
    return parse_cvss_vector(record.vector)


def _selection_key(record: CvssRecord) -> tuple[int, float, str, str]:
    return (
        SOURCE_PRIORITY.get(record.source, len(SOURCE_PRIORITY)),
        -record.base_score,
        record.vector or "",
        record.source.value,
    )


def select_cvss(records: tuple[CvssRecord, ...] | list[CvssRecord] | None) -> tuple[CvssRecord | None, float]:
    """Apply the selection policy.

    Returns ``(chosen record or None, source_agreement in [0, 1])``. With no records the
    agreement is 1.0: there is no disagreement to report, and downstream features must not
    read "absent" as "contested".
    """
    usable = tuple(record for record in (records or ()) if record is not None)
    if not usable:
        return None, 1.0
    newest = max(cvss_version_ordinal(record.version) for record in usable)
    candidates = [record for record in usable if cvss_version_ordinal(record.version) == newest]
    chosen = min(candidates, key=_selection_key)
    return chosen, source_agreement(usable)


def source_agreement(records: tuple[CvssRecord, ...] | list[CvssRecord] | None) -> float:
    """1 minus the normalised spread of base scores across scoring sources.

    Each source contributes a single score - its own newest-version record - so that the
    metric measures organisations disagreeing, not CVSS versions differing.
    """
    usable = [record for record in (records or ()) if record is not None]
    if len(usable) <= 1:
        return 1.0
    per_source: dict[ScoreSource, CvssRecord] = {}
    for record in usable:
        current = per_source.get(record.source)
        if current is None:
            per_source[record.source] = record
            continue
        new_rank = cvss_version_ordinal(record.version)
        old_rank = cvss_version_ordinal(current.version)
        if new_rank > old_rank or (new_rank == old_rank and record.base_score > current.base_score):
            per_source[record.source] = record
    scores = [record.base_score for record in per_source.values()]
    if len(scores) <= 1:
        return 1.0
    spread = max(scores) - min(scores)
    return max(0.0, min(1.0, 1.0 - spread / 10.0))


def submetric_flags(record: CvssRecord | None) -> dict[str, float]:
    """Binary exploitability/impact flags from a CVSS record, as 0.0 / 1.0.

    CVSS v2 has no user-interaction metric - its base score already assumes none - so a v2
    record with usable metrics reports ``cvss_ui_none = 1.0``. v2 privileges map from the
    authentication metric (``Au:N`` means no credentials), and v2 impacts count only
    ``COMPLETE`` as high. v4 vectors expose the impacts as ``VC``/``VI``/``VA``.
    """
    flags = {name: 0.0 for name in SUBMETRIC_FLAG_NAMES}
    if record is None:
        return flags
    metrics = _metrics_of(record)
    if not metrics:
        return flags

    if record.version == CvssVersion.V2:
        flags["cvss_ac_low"] = 1.0 if metrics.get("AC") == "L" else 0.0
        flags["cvss_pr_none"] = 1.0 if metrics.get("AU") == "N" else 0.0
        flags["cvss_ui_none"] = 1.0
        for abbrev, name in (("C", "cvss_c_high"), ("I", "cvss_i_high"), ("A", "cvss_a_high")):
            flags[name] = 1.0 if metrics.get(abbrev) == "C" else 0.0
        return flags

    flags["cvss_ac_low"] = 1.0 if metrics.get("AC") == "L" else 0.0
    flags["cvss_pr_none"] = 1.0 if metrics.get("PR") == "N" else 0.0
    flags["cvss_ui_none"] = 1.0 if metrics.get("UI") == "N" else 0.0
    impacts = (
        ("cvss_c_high", metrics.get("VC", metrics.get("C"))),
        ("cvss_i_high", metrics.get("VI", metrics.get("I"))),
        ("cvss_a_high", metrics.get("VA", metrics.get("A"))),
    )
    for name, value in impacts:
        flags[name] = 1.0 if value == "H" else 0.0
    return flags


def cvss_features(records: tuple[CvssRecord, ...] | list[CvssRecord] | None) -> dict[str, float]:
    """The BASE-group CVSS features of ``FEATURE_SPECS``, computed from one CVE's records."""
    usable = tuple(record for record in (records or ()) if record is not None)
    chosen, agreement = select_cvss(usable)
    features: dict[str, float] = {
        "cvss_base_max": max((record.base_score for record in usable), default=0.0),
        "cvss_version_ord": float(cvss_version_ordinal(chosen.version)) if chosen else 0.0,
        "cvss_source_agreement": agreement,
    }
    features.update(submetric_flags(chosen))
    return features
