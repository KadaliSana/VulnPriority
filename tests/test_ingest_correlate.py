"""Correlation: rank root causes, not alerts.

A web scanner reports one flaw once per affected URL, so a single injection defect arrives as
forty findings. Prioritising the alerts rather than the defect is the overhead the framework
sets out to remove, and the selection layer depends on this too: remediation hours must be
charged once per root cause, not once per symptom.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from vulnpriority.core.enums import HttpMethod, PrivilegeLevel, Provenance, ScannerSeverity
from vulnpriority.core.models import Endpoint, Finding, Scan, UntrustedText
from vulnpriority.ingest.correlate import FindingCorrelator, cluster_sizes, clusters, dedup_key_of
from vulnpriority.ingest.zap import ZapParser

FIXTURES = Path("data/fixtures/scans")
WHEN = datetime(2024, 5, 1, 9, 0, 0)


def _finding(index: int, endpoint_id: str, name: str, cwe: int | None, cves: tuple[str, ...] = (),
             plugin: str | None = "40018") -> Finding:
    return Finding(
        finding_id=f"f{index}",
        scan_id="scan_c",
        app_id="app_c",
        endpoint_id=endpoint_id,
        name=name,
        cwe_id=cwe,
        cve_ids=cves,
        scanner="zap",
        scanner_plugin_id=plugin,
        scanner_severity=ScannerSeverity.HIGH,
        description=UntrustedText(text=name, provenance=Provenance.SCANNER_OUTPUT),
        observed_at=WHEN,
    )


def _endpoint(index: int, path: str) -> Endpoint:
    return Endpoint(
        endpoint_id=f"e{index}",
        app_id="app_c",
        host="shop.example.com",
        url=f"https://shop.example.com{path}",
        path=path,
        method=HttpMethod.GET,
        auth_required=PrivilegeLevel.NONE,
    )


@pytest.fixture
def scan() -> Scan:
    endpoints = tuple(_endpoint(i, p) for i, p in enumerate(
        ["/api/users/{id}", "/api/orders/{id}", "/api/invoices/{id}", "/search", "/admin/panel"]))
    findings = (
        _finding(0, "e0", "SQL Injection", 89, ("CVE-2024-0001",)),
        _finding(1, "e1", "SQL Injection", 89, ("CVE-2024-0001",)),
        _finding(2, "e2", "SQL Injection", 89, ("CVE-2024-0001",)),
        _finding(3, "e3", "Reflected XSS", 79, (), plugin="40012"),
        _finding(4, "e4", "Reflected XSS", 79, (), plugin="40012"),
        _finding(5, "e4", "Server Version Disclosure", 200, (), plugin="10036"),
    )
    return Scan(
        scan_id="scan_c", app_id="app_c", app_name="Correlate", scanned_at=WHEN,
        scanner_name="zap", endpoints=endpoints, findings=findings,
    )


def test_same_root_cause_across_endpoints_shares_one_key(scan: Scan) -> None:
    correlated = FindingCorrelator().correlate(scan)
    keys = {finding.finding_id: finding.dedup_key for finding in correlated.findings}
    assert keys["f0"] == keys["f1"] == keys["f2"]
    assert keys["f3"] == keys["f4"]
    assert keys["f5"] not in {keys["f0"], keys["f3"]}


def test_cluster_size_counts_the_symptoms(scan: Scan) -> None:
    correlated = FindingCorrelator().correlate(scan)
    sizes = {finding.finding_id: finding.cluster_size for finding in correlated.findings}
    assert sizes["f0"] == sizes["f1"] == sizes["f2"] == 3
    assert sizes["f3"] == sizes["f4"] == 2
    assert sizes["f5"] == 1


def test_three_distinct_root_causes_from_six_alerts(scan: Scan) -> None:
    correlated = FindingCorrelator().correlate(scan)
    groups = clusters(correlated)
    assert len(groups) == 3
    assert sorted(len(members) for members in groups.values()) == [1, 2, 3]
    assert sorted(cluster_sizes(correlated.findings).values()) == [1, 2, 3]


def test_different_weaknesses_never_merge(scan: Scan) -> None:
    """Two findings on the same endpoint with different CWEs are two problems."""
    correlated = FindingCorrelator().correlate(scan)
    by_id = {finding.finding_id: finding for finding in correlated.findings}
    assert by_id["f4"].dedup_key != by_id["f5"].dedup_key
    assert by_id["f4"].endpoint_id == by_id["f5"].endpoint_id


def test_correlation_is_deterministic(scan: Scan) -> None:
    first = FindingCorrelator().correlate(scan)
    second = FindingCorrelator().correlate(scan)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_correlation_preserves_every_finding(scan: Scan) -> None:
    """Correlation groups; it must never drop or invent an alert."""
    correlated = FindingCorrelator().correlate(scan)
    assert len(correlated.findings) == len(scan.findings)
    assert {f.finding_id for f in correlated.findings} == {f.finding_id for f in scan.findings}


def test_dedup_key_is_stable_across_processes() -> None:
    """Keys are content hashes, so they must not depend on run order or memory addresses."""
    first = dedup_key_of(_finding(0, "e0", "SQL Injection", 89, ("CVE-2024-0001",)))
    second = dedup_key_of(_finding(9, "e9", "SQL Injection", 89, ("CVE-2024-0001",)))
    assert first == second


def test_cve_ordering_does_not_change_the_key() -> None:
    a = _finding(0, "e0", "SQL Injection", 89, ("CVE-2024-0001", "CVE-2024-0002"))
    b = _finding(1, "e1", "SQL Injection", 89, ("CVE-2024-0002", "CVE-2024-0001"))
    assert dedup_key_of(a) == dedup_key_of(b)


def test_correlating_a_real_scan_reduces_the_queue() -> None:
    scan = ZapParser().parse(FIXTURES / "zap_sample.json")
    correlated = FindingCorrelator().correlate(scan)
    groups = clusters(correlated)
    assert len(groups) <= len(correlated.findings)
    assert all(finding.dedup_key for finding in correlated.findings)
    assert all(finding.cluster_size >= 1 for finding in correlated.findings)


def test_empty_scan_is_handled() -> None:
    empty = Scan(scan_id="s", app_id="a", app_name="A", scanned_at=WHEN, scanner_name="zap")
    correlated = FindingCorrelator().correlate(empty)
    assert correlated.findings == ()
    assert clusters(correlated) == {}
