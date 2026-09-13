"""Scanner parsers: three real formats into one canonical model.

This is the population gap the literature review names. Only three studies in the reviewed
corpus work on web application vulnerabilities at all, and each on a single scanner against a
single target. Parsing OWASP ZAP, Burp Suite and Nuclei into the same ``Scan`` is what makes
the rest of the framework web-native rather than CVE-native, so the parsers are tested against
committed fixtures in each tool's own export format.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vulnpriority.core.enums import HttpMethod, PrivilegeLevel, Provenance, ScannerSeverity, TrustTier
from vulnpriority.core.models import Scan
from vulnpriority.core.registry import SCANNER_PARSERS
from vulnpriority.ingest.burp import BurpParser
from vulnpriority.ingest.generic import GenericJsonParser, detect_parser
from vulnpriority.ingest.nuclei import NucleiParser
from vulnpriority.ingest.tech_fingerprint import fingerprint_response
from vulnpriority.ingest.zap import ZapParser

FIXTURES = Path("data/fixtures/scans")
CASES = [
    ("zap_sample.json", ZapParser, "zap"),
    ("burp_sample.xml", BurpParser, "burp"),
    ("nuclei_sample.jsonl", NucleiParser, "nuclei"),
    ("generic_sample.json", GenericJsonParser, "generic"),
]


@pytest.mark.parametrize("filename,parser_cls,name", CASES, ids=[c[2] for c in CASES])
def test_every_fixture_parses_into_a_valid_scan(filename: str, parser_cls: type, name: str) -> None:
    scan = parser_cls().parse(FIXTURES / filename)
    assert isinstance(scan, Scan)
    assert scan.findings, f"{name} produced no findings"
    assert scan.endpoints, f"{name} produced no endpoints"
    assert scan.scan_id and scan.app_id


@pytest.mark.parametrize("filename,parser_cls,name", CASES, ids=[c[2] for c in CASES])
def test_detect_parser_recognises_each_format(filename: str, parser_cls: type, name: str) -> None:
    detected = detect_parser(FIXTURES / filename)
    assert detected.name == name
    assert isinstance(detected, parser_cls)


@pytest.mark.parametrize("filename,parser_cls,name", CASES, ids=[c[2] for c in CASES])
def test_every_finding_points_at_a_real_endpoint(filename: str, parser_cls: type, name: str) -> None:
    scan = parser_cls().parse(FIXTURES / filename)
    endpoint_ids = {endpoint.endpoint_id for endpoint in scan.endpoints}
    orphans = [f.finding_id for f in scan.findings if f.endpoint_id not in endpoint_ids]
    assert not orphans, f"{name} produced findings with no endpoint: {orphans}"


@pytest.mark.parametrize("filename,parser_cls,name", CASES, ids=[c[2] for c in CASES])
def test_scanner_text_is_untrusted(filename: str, parser_cls: type, name: str) -> None:
    """Descriptions and echoed responses are attacker-influenced and must be typed as such."""
    scan = parser_cls().parse(FIXTURES / filename)
    for finding in scan.findings:
        assert finding.description.provenance != Provenance.OPERATOR
        assert finding.description.tier >= TrustTier.SCANNER
        for evidence in finding.evidence:
            assert evidence.tier >= TrustTier.SCANNER


@pytest.mark.parametrize("filename,parser_cls,name", CASES, ids=[c[2] for c in CASES])
def test_parsing_is_deterministic(filename: str, parser_cls: type, name: str) -> None:
    """Identifiers are content hashes, so two parses of one file must be identical."""
    first = parser_cls().parse(FIXTURES / filename)
    second = parser_cls().parse(FIXTURES / filename)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


@pytest.mark.parametrize("filename,parser_cls,name", CASES, ids=[c[2] for c in CASES])
def test_paths_are_templated_not_literal(filename: str, parser_cls: type, name: str) -> None:
    """A path carrying an identifier must be collapsed, or one flaw becomes forty findings."""
    scan = parser_cls().parse(FIXTURES / filename)
    for endpoint in scan.endpoints:
        segments = [s for s in endpoint.path.split("/") if s]
        bare_numbers = [s for s in segments if s.isdigit()]
        assert not bare_numbers, f"{endpoint.path} still contains a literal identifier"


def test_zap_alert_instances_become_separate_endpoints() -> None:
    """One ZAP alert can carry many instances; each URI is its own endpoint."""
    scan = ZapParser().parse(FIXTURES / "zap_sample.json")
    paths = {endpoint.path for endpoint in scan.endpoints}
    assert len(paths) > 1
    # The fixture deliberately repeats one alert across URIs so correlation has work to do.
    names = [finding.name for finding in scan.findings]
    assert len(names) > len(set(names)) or any(f.cluster_size > 1 for f in scan.findings)


def test_zap_severity_and_confidence_are_mapped() -> None:
    scan = ZapParser().parse(FIXTURES / "zap_sample.json")
    severities = {finding.scanner_severity for finding in scan.findings}
    assert severities <= set(ScannerSeverity)
    assert severities & {ScannerSeverity.HIGH, ScannerSeverity.CRITICAL, ScannerSeverity.MEDIUM}
    for finding in scan.findings:
        assert 0.0 <= finding.scanner_confidence <= 1.0


def test_burp_decodes_request_and_response_without_executing_anything() -> None:
    scan = BurpParser().parse(FIXTURES / "burp_sample.xml")
    assert scan.scanner_name.lower().startswith("burp")
    with_evidence = [finding for finding in scan.findings if finding.evidence]
    assert with_evidence, "the Burp fixture carries request/response bodies"
    for finding in with_evidence:
        for evidence in finding.evidence:
            assert isinstance(evidence.text, str)


def test_nuclei_carries_cve_and_cwe_classification() -> None:
    scan = NucleiParser().parse(FIXTURES / "nuclei_sample.jsonl")
    assert any(finding.cve_ids for finding in scan.findings)
    assert any(finding.cwe_id for finding in scan.findings)


def test_generic_parser_round_trips_the_canonical_model() -> None:
    """The framework's own format must survive a write/read cycle unchanged."""
    original = ZapParser().parse(FIXTURES / "zap_sample.json")
    payload = original.model_dump(mode="json")
    restored = Scan.model_validate(payload)
    assert restored == original


def test_auth_levels_are_inferred_from_structure() -> None:
    scan = ZapParser().parse(FIXTURES / "zap_sample.json")
    by_path = {endpoint.path: endpoint for endpoint in scan.endpoints}
    admin = [e for path, e in by_path.items() if path.startswith("/admin")]
    if admin:
        assert all(e.auth_required >= PrivilegeLevel.USER for e in admin)
    static = [e for path, e in by_path.items() if path.startswith("/static")]
    if static:
        assert all(e.auth_required == PrivilegeLevel.NONE for e in static)


def test_methods_are_parsed_as_enum_members() -> None:
    for filename, parser_cls, _ in CASES:
        scan = parser_cls().parse(FIXTURES / filename)
        for endpoint in scan.endpoints:
            assert isinstance(endpoint.method, HttpMethod)


def test_all_four_parsers_are_registered() -> None:
    assert {"zap", "burp", "nuclei", "generic"} <= set(SCANNER_PARSERS)


def test_fingerprinting_recovers_products_and_versions() -> None:
    tech = fingerprint_response(
        headers={"Server": "Apache/2.4.49", "X-Powered-By": "PHP/8.1.2"},
        cookies={"PHPSESSID": "abc"},
        body="<meta name='generator' content='WordPress 6.4.1'>",
        url="https://shop.example.com/wp-admin/",
    )
    # Products are normalised towards CPE naming, so Apache httpd is vendor "apache",
    # product "http_server" rather than the raw header token.
    names = {f"{component.vendor or ''}:{component.product}".lower() for component in tech}
    assert any("apache" in name for name in names), names
    assert any("php" in name for name in names), names
    assert any("wordpress" in name for name in names), names
    assert any(component.version for component in tech)


def test_unknown_file_is_rejected_rather_than_guessed(tmp_path: Path) -> None:
    path = tmp_path / "notascan.txt"
    path.write_text("this is not scanner output", encoding="utf-8")
    with pytest.raises(Exception):
        detect_parser(path)


def test_a_zap_json_report_falls_back_to_created_when_generated_will_not_parse(tmp_path):
    """ZAP spells September "Sept", which ``%b`` rejects, and also writes an ISO ``created``.

    Reading only ``@generated`` stamped every JSON report with the epoch. That is not a
    cosmetic default: the scan timestamp becomes the run's intelligence as-of date, so every
    CVE, KEV and EPSS lookup was being asked what was known in 1970 - and nothing is. A
    whole class of run was silently doing its enrichment against an empty feed.
    """
    import json
    from datetime import date

    from vulnpriority.ingest.normalize import DEFAULT_SCANNED_AT

    payload = {
        "@programName": "ZAP",
        "@version": "2.17.0",
        "@generated": "Sun, 13 Sept 2026 09:16:41",
        "created": "2026-09-13T09:16:41.716202535Z",
        "site": [
            {
                "@name": "https://shop.example.com",
                "@host": "shop.example.com",
                "@port": "443",
                "@ssl": "true",
                "alerts": [],
            }
        ],
    }
    path = tmp_path / "zap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    scan = ZapParser().parse(path)
    assert scan.scanned_at != DEFAULT_SCANNED_AT
    assert scan.scanned_at.date() == date(2026, 9, 13)


def test_the_sept_spelling_is_handled_by_the_timestamp_parser_itself(tmp_path):
    """It belongs to the parser, not to one report format, and it is word-bounded."""
    from datetime import datetime

    from vulnpriority.ingest.normalize import parse_timestamp

    assert parse_timestamp("Sun, 13 Sept 2026 09:16:41") == datetime(2026, 9, 13, 9, 16, 41)
    assert parse_timestamp("Sun, 13 Sep 2026 09:16:41") == datetime(2026, 9, 13, 9, 16, 41)
    assert parse_timestamp("Septimus") is None
