"""Shared pytest fixtures.

Every fixture here is offline and deterministic: no network, no API key, fixed seeds.
Group-specific fixtures belong in the test module that needs them, not here.

Since the shipped defaults became ``auto`` (DESIGN: prefer live, degrade to offline), that
guarantee needs enforcing rather than assuming. :func:`_offline_resolution` below is
autouse: it strips every provider key from the environment and pins the network probe to
"unreachable" for the whole suite. Without it the suite's behaviour would depend on whether
the machine running it happened to have a key exported and a working uplink, which is
exactly the non-determinism the framework spends its time eliminating everywhere else.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from vulnpriority.core import resolve
from vulnpriority.core.config import PROJECT_ROOT, PipelineConfig, load_config
from vulnpriority.core.enums import (
    EndpointFunction,
    ExploitMaturity,
    ExploitSource,
    HttpMethod,
    PrivilegeLevel,
    Provenance,
    ScannerSeverity,
    ScoreSource,
    CvssVersion,
)
from vulnpriority.core.models import (
    CvssRecord,
    Endpoint,
    EpssRecord,
    ExploitEvidence,
    Finding,
    KevRecord,
    Scan,
    TechComponent,
    UntrustedText,
    VulnIntel,
)

SEED = 7
AS_OF = date(2024, 6, 1)

#: Every variable ``auto`` backend resolution consults. Cleared for the whole suite: a test
#: that passed only on a laptop with a key exported would be worse than no test.
PROVIDER_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "CEREBRAS_API_KEY",
    # Search credentials belong here too. A key the suite can see is a key the suite can
    # spend: Parallel bills per search, and a test that reached a real one would charge the
    # developer for running the tests and make the result depend on the network.
    "PARALLEL_API_KEY",
    "NVD_API_KEY",
)


@pytest.fixture(autouse=True)
def _offline_resolution(monkeypatch):
    """Pin ``auto`` resolution to offline for every test in the suite.

    Two halves, and both are needed. Clearing the keys stops backend resolution finding
    credentials; pinning the probe stops feed resolution opening a socket -- and stops it
    spending 1.5 seconds discovering it cannot. A test that wants to exercise resolution
    sets its own environment on top of this.
    """
    for name in PROVIDER_KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(resolve, "network_is_reachable", lambda *_args, **_kwargs: False)
    resolve.reset_network_probe()
    yield
    resolve.reset_network_probe()


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def as_of() -> date:
    return AS_OF


@pytest.fixture
def offline_config(tmp_path: Path) -> PipelineConfig:
    """Pinned-offline configuration, writing into a temporary directory.

    ``configs/offline.yaml`` rather than ``configs/default.yaml``: the default is now
    ``auto``, and a fixture whose offline-ness depended on resolution would be relying on
    the very thing it is supposed to hold constant.
    """
    config = load_config(PROJECT_ROOT / "configs" / "offline.yaml")
    return config.model_copy(update={"output_dir": tmp_path / "runs", "seed": SEED})


@pytest.fixture
def auto_config(tmp_path: Path) -> PipelineConfig:
    """The shipped ``auto`` defaults, for tests about resolution itself."""
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    return config.model_copy(update={"output_dir": tmp_path / "runs", "seed": SEED})


@pytest.fixture
def untrusted() -> "callable":
    def _make(text: str, provenance: Provenance = Provenance.REFERENCE_PAGE) -> UntrustedText:
        return UntrustedText(text=text, provenance=provenance)

    return _make


@pytest.fixture
def sample_endpoints() -> tuple[Endpoint, ...]:
    return (
        Endpoint(
            endpoint_id="ep_login",
            app_id="app1",
            host="shop.example.com",
            url="https://shop.example.com/api/login",
            path="/api/login",
            method=HttpMethod.POST,
            auth_required=PrivilegeLevel.NONE,
            internet_facing=True,
            response_status=200,
            response_content_type="application/json",
            response_size_bytes=512,
            sets_cookie=True,
            parameters=("username", "password"),
            links_to=("ep_admin",),
            observed_tech=(TechComponent(vendor="apache", product="struts", version="2.5.12"),),
        ),
        Endpoint(
            endpoint_id="ep_admin",
            app_id="app1",
            host="shop.example.com",
            url="https://shop.example.com/admin/users/{id}",
            path="/admin/users/{id}",
            method=HttpMethod.GET,
            auth_required=PrivilegeLevel.ADMIN,
            internet_facing=True,
            response_status=403,
            response_content_type="text/html",
            response_size_bytes=2048,
            parameters=("id",),
        ),
        Endpoint(
            endpoint_id="ep_static",
            app_id="app1",
            host="shop.example.com",
            url="https://shop.example.com/static/app.css",
            path="/static/app.css",
            method=HttpMethod.GET,
            auth_required=PrivilegeLevel.NONE,
            internet_facing=True,
            response_status=200,
            response_content_type="text/css",
            response_size_bytes=10240,
        ),
    )


@pytest.fixture
def sample_scan(sample_endpoints: tuple[Endpoint, ...]) -> Scan:
    scanned_at = datetime(2024, 5, 1, 9, 0, 0)
    findings = (
        Finding(
            finding_id="f_sqli",
            scan_id="scan_1",
            app_id="app1",
            endpoint_id="ep_login",
            name="SQL Injection",
            cwe_id=89,
            cve_ids=("CVE-2024-0001",),
            scanner="zap",
            scanner_plugin_id="40018",
            scanner_severity=ScannerSeverity.HIGH,
            scanner_confidence=0.9,
            description=UntrustedText(
                text="SQL injection in the username parameter of /api/login",
                provenance=Provenance.SCANNER_OUTPUT,
            ),
            evidence=(
                UntrustedText(text="' OR '1'='1 returned 200 with 42 rows", provenance=Provenance.TARGET_RESPONSE),
            ),
            affected_component=TechComponent(vendor="apache", product="struts", version="2.5.12"),
            observed_at=scanned_at,
            dedup_key="dk_sqli",
            cluster_size=2,
        ),
        Finding(
            finding_id="f_xss",
            scan_id="scan_1",
            app_id="app1",
            endpoint_id="ep_admin",
            name="Reflected Cross Site Scripting",
            cwe_id=79,
            cve_ids=(),
            scanner="zap",
            scanner_plugin_id="40012",
            scanner_severity=ScannerSeverity.MEDIUM,
            scanner_confidence=0.6,
            description=UntrustedText(
                text="Reflected XSS in the id parameter", provenance=Provenance.SCANNER_OUTPUT
            ),
            observed_at=scanned_at,
            dedup_key="dk_xss",
        ),
        Finding(
            finding_id="f_info",
            scan_id="scan_1",
            app_id="app1",
            endpoint_id="ep_static",
            name="Server Version Disclosure",
            cwe_id=200,
            scanner="zap",
            scanner_plugin_id="10036",
            scanner_severity=ScannerSeverity.LOW,
            scanner_confidence=0.4,
            description=UntrustedText(text="Server header discloses version", provenance=Provenance.SCANNER_OUTPUT),
            observed_at=scanned_at,
            dedup_key="dk_info",
        ),
    )
    return Scan(
        scan_id="scan_1",
        app_id="app1",
        app_name="Example Shop",
        sector="ecommerce",
        scanned_at=scanned_at,
        scanner_name="zap",
        scanner_version="2.14.0",
        hosts=("shop.example.com",),
        tech_stack=(TechComponent(vendor="apache", product="struts", version="2.5.12"),),
        endpoints=sample_endpoints,
        findings=findings,
    )


@pytest.fixture
def sample_intel() -> VulnIntel:
    return VulnIntel(
        cve_id="CVE-2024-0001",
        as_of=AS_OF,
        description=UntrustedText(
            text="A SQL injection in Example Struts allows remote attackers to read the database.",
            provenance=Provenance.NVD,
        ),
        published=date(2024, 1, 10),
        cvss=(
            CvssRecord(
                version=CvssVersion.V31,
                source=ScoreSource.NVD,
                base_score=9.8,
                vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                submetrics={"AV": "N", "AC": "L", "PR": "N", "UI": "N", "C": "H", "I": "H", "A": "H"},
            ),
        ),
        epss=EpssRecord(cve_id="CVE-2024-0001", score=0.42, percentile=0.97, as_of=AS_OF),
        kev=KevRecord(cve_id="CVE-2024-0001", in_kev=True, date_added=date(2024, 2, 1), as_of=AS_OF),
        exploits=(
            ExploitEvidence(
                source=ExploitSource.EXPLOIT_DB,
                url="https://www.exploit-db.com/exploits/00000",
                published=date(2024, 1, 20),
                maturity=ExploitMaturity.FUNCTIONAL,
                verified=True,
            ),
        ),
    )


@pytest.fixture
def endpoint_functions() -> tuple[EndpointFunction, ...]:
    return tuple(EndpointFunction)
