"""Finding correlation: the same root cause seen across many endpoints.

The framework ranks root causes, not alerts. One missing output encoding produces a
reflected-XSS finding on every page that echoes a parameter; an engineer fixes it once.
Correlation gives every finding a ``dedup_key`` and the size of its cluster, so the ranker
can use ``cluster_size`` as a feature (a defect on forty routes is worse than on one) and
the selection layer charges remediation cost once per cluster rather than once per alert.

``dedup_key = stable_id("dk", app_id, str(cwe_id), sorted_cve_ids, plugin_id or name)``
exactly as DESIGN 3.1 specifies, with ``sorted_cve_ids`` rendered as a comma-joined,
sorted list so the key cannot depend on scanner output ordering.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence

from vulnpriority.core.hashing import stable_id
from vulnpriority.core.models import Finding, Scan

__all__ = ["FindingCorrelator", "dedup_key_for", "clusters", "cluster_sizes"]


def dedup_key_for(
    app_id: str,
    cwe_id: int | None,
    cve_ids: Iterable[str],
    plugin_id: str | None,
    name: str,
) -> str:
    """The DESIGN 3.1 dedup key for one finding's identity.

    The key deliberately ignores the endpoint: that is the whole point, since the same
    defect on ten routes is one thing to fix.
    """
    sorted_cve_ids = ",".join(sorted({str(cve).strip().upper() for cve in cve_ids if cve}))
    return stable_id("dk", app_id, str(cwe_id), sorted_cve_ids, plugin_id or name)


def dedup_key_of(finding: Finding) -> str:
    """Dedup key of an existing finding."""
    return dedup_key_for(
        finding.app_id,
        finding.cwe_id,
        finding.cve_ids,
        finding.scanner_plugin_id,
        finding.name,
    )


def cluster_sizes(findings: Sequence[Finding]) -> dict[str, int]:
    """Number of findings sharing each dedup key within one scan."""
    counts: dict[str, int] = {}
    for finding in findings:
        key = dedup_key_of(finding)
        counts[key] = counts.get(key, 0) + 1
    return counts


def clusters(scan: Scan) -> dict[str, list[Finding]]:
    """Findings of a scan grouped by dedup key, in first-appearance order.

    Works whether or not :meth:`FindingCorrelator.correlate` has already run: the key is
    recomputed from the finding's own identity fields either way.
    """
    grouped: "OrderedDict[str, list[Finding]]" = OrderedDict()
    for finding in scan.findings:
        grouped.setdefault(finding.dedup_key or dedup_key_of(finding), []).append(finding)
    return dict(grouped)


class FindingCorrelator:
    """Assigns ``dedup_key`` and ``cluster_size`` to every finding of a scan.

    Stateless and pure: ``correlate`` returns a new :class:`Scan` and never mutates its
    argument, so the same scan can be correlated repeatedly with identical results.
    """

    def dedup_key(self, finding: Finding) -> str:
        """Dedup key for one finding."""
        return dedup_key_of(finding)

    def correlate(self, scan: Scan) -> Scan:
        """Return a copy of ``scan`` with correlation fields filled in on every finding."""
        counts = cluster_sizes(scan.findings)
        correlated = tuple(
            finding.model_copy(
                update={
                    "dedup_key": dedup_key_of(finding),
                    "cluster_size": counts[dedup_key_of(finding)],
                }
            )
            for finding in scan.findings
        )
        return scan.model_copy(update={"findings": correlated})

    def clusters(self, scan: Scan) -> dict[str, list[Finding]]:
        """Findings grouped by dedup key (see the module-level :func:`clusters`)."""
        return clusters(scan)
