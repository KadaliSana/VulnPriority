"""The interactive web application.

Driven through ``fastapi.testclient.TestClient``, so no port is bound and nothing touches
the network. The analysis these tests run is the real one - ingest, assessment, enrichment,
chain scoring, ranking and selection over a fixture report - because the thing worth
asserting is that the endpoint produces a payload the dashboard can actually render, not
that a mock was called.

``vulnprio.scan`` is stubbed throughout. It is developed in parallel, and more importantly
the test that matters most here is that an unauthorised request never reaches it at all,
which can only be asserted against something that records whether it was called.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vulnprio.core.config import PROJECT_ROOT
from vulnprio.web.app import (
    ACCEPTED_REPORT_SUFFIXES,
    TOKEN_HEADER,
    AppSettings,
    create_app,
)
from vulnprio.web.schema import DashboardData

ZAP_FIXTURE = PROJECT_ROOT / "data" / "fixtures" / "scans" / "zap_sample.json"
TERMINAL = {"done", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_and_token(tmp_path: Path):
    api = create_app(AppSettings())
    yield api, api.state.vulnprio.token
    api.state.vulnprio.close()


@pytest.fixture
def client(app_and_token):
    api, _token = app_and_token
    with TestClient(api) as test_client:
        yield test_client


@pytest.fixture
def token(app_and_token) -> str:
    return app_and_token[1]


@pytest.fixture
def auth(token: str) -> dict[str, str]:
    return {TOKEN_HEADER: token}


def wait_for_job(client: TestClient, job_id: str, timeout: float = 180.0) -> dict[str, Any]:
    """Poll until the job settles. The real pipeline takes seconds, not milliseconds."""
    deadline = time.monotonic() + timeout
    snapshot: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        snapshot = response.json()
        if snapshot["status"] in TERMINAL:
            return snapshot
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish: {snapshot}")


class FakeScan:
    """A stand-in for ``vulnprio.scan`` that records whether it was asked to do anything."""

    def __init__(self) -> None:
        self.calls: list[Any] = []


@pytest.fixture
def fake_scan(monkeypatch: pytest.MonkeyPatch) -> FakeScan:
    """Install a stub ``vulnprio.scan`` and reset the capability cache around it."""
    from vulnprio.web import app as app_module
    from vulnprio.core.models import Scan

    recorder = FakeScan()
    module = types.ModuleType("vulnprio.scan")

    class ScanProfile(str):
        pass

    class ScanRequest:
        model_fields = {
            "target_url": None, "authorized": None, "authorization_note": None, "profile": None,
        }

        def __init__(self, **kwargs: Any) -> None:
            if not kwargs.get("authorized"):
                raise ValueError("ScanRequest requires authorized=True")
            self.__dict__.update(kwargs)

    def assess_target(request: Any, *, on_progress: Any = None, **_: Any) -> Any:
        recorder.calls.append(request)
        raise AssertionError("the fake scanner should not be reached in these tests")

    module.ScanProfile = ScanProfile
    module.ScanRequest = ScanRequest
    module.assess_target = assess_target
    module.Scan = Scan
    monkeypatch.setitem(sys.modules, "vulnprio.scan", module)
    app_module.probe_capabilities(refresh=True)
    yield recorder
    app_module.probe_capabilities(refresh=True)


# ---------------------------------------------------------------------------
# health and capabilities
# ---------------------------------------------------------------------------


def test_health_reports_version_and_capabilities(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["version"]
    assert body["schema_version"]
    assert set(body["capabilities"]) >= {"scan", "report", "novelty", "detail"}
    for name in ("scan", "report", "novelty"):
        assert isinstance(body["capabilities"][name], bool)


def test_health_reflects_a_missing_sibling_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent package is reported false, with the reason, rather than silently assumed."""
    import importlib

    from vulnprio.web import app as app_module

    real_import = importlib.import_module

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "vulnprio.novelty":
            raise ImportError("no module named vulnprio.novelty")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(app_module.importlib, "import_module", refuse)
    app_module.probe_capabilities(refresh=True)
    try:
        api = create_app(AppSettings())
        with TestClient(api) as test_client:
            capabilities = test_client.get("/api/health").json()["capabilities"]
            assert capabilities["novelty"] is False
            assert "vulnprio.novelty" in capabilities["detail"]["novelty"]

            novelty = test_client.get("/api/novelty").json()
            assert novelty["available"] is False
            assert novelty["detail"]
    finally:
        monkeypatch.undo()
        app_module.probe_capabilities(refresh=True)


def test_novelty_endpoint_answers_either_way(client: TestClient) -> None:
    body = client.get("/api/novelty").json()
    assert "available" in body
    if body["available"] is False:
        assert body["detail"], "an unavailable analysis must say why"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_config_offers_presets_profiles_and_defaults(client: TestClient) -> None:
    body = client.get("/api/config").json()

    names = {preset["name"] for preset in body["attackers"]}
    assert {"opportunistic", "targeted_criminal"} <= names, names
    assert {preset["name"] for preset in body["impact_models"]} >= {"default_ecommerce"}

    profiles = {profile["name"]: profile for profile in body["profiles"]}
    assert set(profiles) == {"passive", "active"}
    for profile in profiles.values():
        assert profile["description"], "a profile the operator authorises must be described"
        assert profile["does_not_send"], "say what it will not do, not only what it will"

    defaults = body["defaults"]
    assert defaults["attacker"] and defaults["impact_model"]
    assert defaults["budget_hours"] > 0
    assert set(defaults["components"]) == {"a", "b", "c"}
    assert body["accepted_report_suffixes"] == list(ACCEPTED_REPORT_SUFFIXES)
    assert body["max_upload_bytes"] > 0


# ---------------------------------------------------------------------------
# the token on mutating endpoints
# ---------------------------------------------------------------------------


def test_analyze_requires_the_session_token(client: TestClient) -> None:
    response = client.post("/api/analyze", json={"mode": "demo"})
    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "bad_token"
    assert TOKEN_HEADER in body["detail"]


def test_a_wrong_token_is_refused(client: TestClient) -> None:
    response = client.post(
        "/api/analyze", json={"mode": "demo"}, headers={TOKEN_HEADER: "not-the-token"}
    )
    assert response.status_code == 403
    assert response.json()["error"] == "bad_token"


def test_cancel_requires_the_token(client: TestClient, auth: dict[str, str]) -> None:
    created = client.post("/api/analyze", json={"mode": "demo"}, headers=auth).json()
    assert client.post(f"/api/jobs/{created['job_id']}/cancel").status_code == 403
    assert client.post(
        f"/api/jobs/{created['job_id']}/cancel", headers=auth
    ).status_code == 200


def test_reading_endpoints_do_not_need_the_token(client: TestClient) -> None:
    for path in ("/api/health", "/api/config", "/api/novelty"):
        assert client.get(path).status_code == 200, path


def test_two_applications_do_not_share_a_token() -> None:
    first, second = create_app(AppSettings()), create_app(AppSettings())
    try:
        assert first.state.vulnprio.token != second.state.vulnprio.token
        with TestClient(second) as client:
            response = client.post(
                "/api/analyze", json={"mode": "demo"},
                headers={TOKEN_HEADER: first.state.vulnprio.token},
            )
            assert response.status_code == 403
    finally:
        first.state.vulnprio.close()
        second.state.vulnprio.close()


# ---------------------------------------------------------------------------
# demo mode, end to end
# ---------------------------------------------------------------------------


def test_demo_analysis_produces_a_valid_dashboard_payload(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post("/api/analyze", json={"mode": "demo"}, headers=auth)
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    snapshot = wait_for_job(client, job_id)
    assert snapshot["status"] == "done", snapshot
    assert snapshot["progress"] == 1.0
    assert snapshot["log"], "the run should have said what it was doing"

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    data = DashboardData.model_validate(result.json())
    assert data.findings, "a finished analysis must carry findings"
    assert data.summary.n_findings == len(data.findings)


# ---------------------------------------------------------------------------
# upload mode against a real scanner fixture
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_upload_mode_runs_the_pipeline_on_a_real_zap_report(
    client: TestClient, auth: dict[str, str]
) -> None:
    content = base64.b64encode(ZAP_FIXTURE.read_bytes()).decode("ascii")
    response = client.post(
        "/api/analyze",
        json={
            "mode": "upload",
            "report": {"filename": "zap_sample.json", "content_base64": content},
            "attacker": "opportunistic",
            "impact_model": "default_ecommerce",
            "budget_hours": 12.0,
            "components": {"a": True, "b": True, "c": True},
        },
        headers=auth,
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    snapshot = wait_for_job(client, job_id)
    assert snapshot["status"] == "done", snapshot

    data = DashboardData.model_validate(client.get(f"/api/jobs/{job_id}/result").json())
    assert data.findings, "the ZAP fixture contains findings, so the payload must too"
    assert data.scans and data.scans[0].scanner
    assert data.summary.n_findings == len(data.findings)
    assert any(finding.expected_loss > 0 for finding in data.findings)
    assert all(finding.finding_id for finding in data.findings)

    # A single-scan triage has no labelled history, so these sections are absent rather
    # than fabricated; the page already hides what is missing.
    assert data.metrics == []
    assert data.ablation.cells == []

    assert data.notes["source"]["mode"] == "upload"
    assert data.notes["source"]["attacker"] == "opportunistic"


@pytest.mark.slow
def test_multipart_upload_starts_the_same_analysis(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post(
        "/api/analyze/upload",
        headers=auth,
        files={"file": ("zap_sample.json", ZAP_FIXTURE.read_bytes(), "application/json")},
        data={"attacker": "opportunistic", "budget_hours": "8"},
    )
    assert response.status_code == 202, response.text
    snapshot = wait_for_job(client, response.json()["job_id"])
    assert snapshot["status"] == "done", snapshot
    data = DashboardData.model_validate(
        client.get(f"/api/jobs/{snapshot['job_id']}/result").json()
    )
    assert data.findings


def test_a_report_with_an_unreadable_suffix_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post(
        "/api/analyze",
        json={
            "mode": "upload",
            "report": {"filename": "report.pdf", "content_base64": base64.b64encode(b"%PDF-1.4").decode()},
        },
        headers=auth,
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "unsupported_report"
    assert ".json" in body["detail"]


def test_upload_mode_without_a_file_is_refused(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post("/api/analyze", json={"mode": "upload"}, headers=auth)
    assert response.status_code == 400
    assert response.json()["error"] == "missing_report"


def test_an_unparseable_report_fails_the_job_with_a_readable_message(
    client: TestClient, auth: dict[str, str]
) -> None:
    content = base64.b64encode(b"this is not a scanner report at all").decode("ascii")
    response = client.post(
        "/api/analyze",
        json={"mode": "upload", "report": {"filename": "junk.json", "content_base64": content}},
        headers=auth,
    )
    assert response.status_code == 202
    snapshot = wait_for_job(client, response.json()["job_id"])
    assert snapshot["status"] == "failed"
    assert snapshot["error"]
    assert "Traceback" not in snapshot["error"]
    assert snapshot["has_result"] is False


# ---------------------------------------------------------------------------
# scanner selection
#
# Which tool ran is not an implementation detail: ZAP and the built-in crawler find
# different things. Profile and availability interact, because Nuclei and Nikto request
# paths nobody linked to and so cannot serve a passive request, and the interface has to
# be able to say that while the operator is still choosing.
# ---------------------------------------------------------------------------


def test_scanners_endpoint_describes_the_environment_per_profile(client: TestClient) -> None:
    for profile in ("passive", "active"):
        response = client.get(f"/api/scanners?profile={profile}")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["profile"] == profile
        if not body["available"]:
            assert body["notice"], "an unavailable scanner package must say so"
            continue
        assert body["tools"], "the scan package registers tools, so some should be listed"
        for tool in body["tools"]:
            assert tool["name"]
            assert tool["summary"], f"{tool['name']} should describe itself"
            assert isinstance(tool["supports_requested_profile"], bool)
            if not tool["installed"]:
                assert tool["install_hint"], f"{tool['name']} should say how to install it"


def test_an_unknown_profile_is_refused(client: TestClient) -> None:
    response = client.get("/api/scanners?profile=aggressive")
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_profile"


def test_active_only_tools_are_marked_unusable_for_a_passive_request(
    client: TestClient,
) -> None:
    """Nuclei and Nikto are registered active-only on purpose; the UI depends on that flag."""
    passive = client.get("/api/scanners?profile=passive").json()
    active = client.get("/api/scanners?profile=active").json()
    if not passive["available"]:
        pytest.skip("no scan package in this build")

    by_name = {tool["name"]: tool for tool in passive["tools"]}
    active_by_name = {tool["name"]: tool for tool in active["tools"]}
    for name in ("nuclei", "nikto"):
        if name not in by_name:
            continue
        assert by_name[name]["supports_requested_profile"] is False, (
            f"{name} requests unlinked paths, which passive promises not to do"
        )
        assert active_by_name[name]["supports_requested_profile"] is True
        assert "passive" not in by_name[name]["profiles"]


def test_the_builtin_fallback_is_reported_when_nothing_is_installed(
    client: TestClient,
) -> None:
    body = client.get("/api/scanners?profile=passive").json()
    if not body["available"]:
        pytest.skip("no scan package in this build")
    if any(tool["installed"] and tool["supports_requested_profile"] for tool in body["tools"]):
        pytest.skip("this machine has a usable external scanner installed")
    assert body["builtin_fallback"] is True
    assert body["preferred"] == ""
    assert body["notice"], "say that the weaker built-in scanner will run, and how to fix it"


def test_the_scanner_choice_is_part_of_the_request_contract() -> None:
    """The page can pin a scanner, and an unknown field would be a 400 rather than ignored."""
    from vulnprio.web.schema import AnalyzeRequest

    assert AnalyzeRequest().scanner == "", "the default lets the scan package choose"
    assert AnalyzeRequest(scanner="builtin").scanner == "builtin"
    assert AnalyzeRequest(scanner="zap-docker").scanner == "zap-docker"


# ---------------------------------------------------------------------------
# report inspection and the as-of decision
#
# A report from months ago and an internet read today describe different moments. The
# product answer is to offer the operator a re-scan, so the decision has to reach the
# browser as data with an action attached, not as a sentence in a log.
# ---------------------------------------------------------------------------


def test_inspect_describes_a_report_without_running_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post(
        "/api/inspect",
        headers=auth,
        files={"file": ("zap_sample.json", ZAP_FIXTURE.read_bytes(), "application/json")},
        data={"use_intel": "true"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["host"], "the report names a host, so the inspection should carry it"
    assert body["scanned_at"].startswith("2024-"), body["scanned_at"]
    assert body["n_findings"] > 0
    # The target the operator could assess right now, derived from the report itself.
    assert body["suggested_target_url"].startswith("http")
    assert body["host"] in body["suggested_target_url"]


def test_an_old_report_is_reported_stale_with_a_rescan_remedy(
    client: TestClient, auth: dict[str, str]
) -> None:
    """The fixture is years old, so live intelligence cannot honestly describe its moment."""
    body = client.post(
        "/api/inspect",
        headers=auth,
        files={"file": ("zap_sample.json", ZAP_FIXTURE.read_bytes(), "application/json")},
        data={"use_intel": "true"},
    ).json()
    as_of = body["as_of"]
    if as_of["status"] == "not_applicable":
        pytest.skip("this build has no live-intelligence package to be stale against")
    assert as_of["status"] in {"stale", "refused"}
    assert as_of["remedy"] == "rescan_target", "the page needs an action, not just a verdict"
    assert as_of["search_allowed"] is False
    assert as_of["evidence_is_current"] is False
    assert as_of["age_days"] > as_of["max_age_days"] > 0
    assert as_of["message"]


def test_staleness_does_not_arise_when_intelligence_is_off(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Nothing is being read from today's internet, so the scan's age changes nothing."""
    body = client.post(
        "/api/inspect",
        headers=auth,
        files={"file": ("zap_sample.json", ZAP_FIXTURE.read_bytes(), "application/json")},
        data={"use_intel": "false"},
    ).json()
    assert body["as_of"]["status"] == "not_applicable"
    assert body["as_of"]["search_allowed"] is True
    assert body["as_of"]["as_of"].startswith("2024-"), "the scan date is still recorded"


def test_inspect_requires_the_token_and_a_readable_report(
    client: TestClient, auth: dict[str, str]
) -> None:
    unauthenticated = client.post(
        "/api/inspect",
        files={"file": ("zap_sample.json", b"{}", "application/json")},
    )
    assert unauthenticated.status_code == 403

    junk = client.post(
        "/api/inspect", headers=auth,
        files={"file": ("junk.json", b"not a scanner report", "application/json")},
    )
    assert junk.status_code == 400
    assert junk.json()["error"] == "unreadable_report"
    assert "Traceback" not in junk.json()["detail"]


@pytest.mark.slow
def test_the_payload_records_whether_intelligence_ran(
    client: TestClient, auth: dict[str, str]
) -> None:
    """A reader months later must be able to tell what the assessment knew and when."""
    content = base64.b64encode(ZAP_FIXTURE.read_bytes()).decode("ascii")
    created = client.post(
        "/api/analyze",
        json={
            "mode": "upload",
            "report": {"filename": "zap_sample.json", "content_base64": content},
            "use_intel": False,
        },
        headers=auth,
    ).json()
    assert wait_for_job(client, created["job_id"])["status"] == "done"

    data = DashboardData.model_validate(
        client.get(f"/api/jobs/{created['job_id']}/result").json()
    )
    intel = data.notes["intel"]
    assert intel["requested"] is False
    assert intel["ran"] is False
    assert intel["disabled_reason"], "say why nobody looked, not just that nothing was found"
    assert intel["scan_date"].startswith("2024-")


# ---------------------------------------------------------------------------
# scanning: authorisation is checked before anything is sent
# ---------------------------------------------------------------------------


def test_a_scan_without_authorisation_is_refused_and_nothing_is_attempted(
    client: TestClient, auth: dict[str, str], fake_scan: FakeScan
) -> None:
    response = client.post(
        "/api/analyze",
        json={
            "mode": "scan",
            "target_url": "https://app.example.com/",
            "authorized": False,
            "authorization_note": "",
        },
        headers=auth,
    )
    assert 400 <= response.status_code < 500
    body = response.json()
    assert body["error"] == "not_authorized"
    assert "authorised" in body["detail"]
    assert "Nothing was sent" in body["detail"]
    assert fake_scan.calls == [], "the scanner must not be reached by an unauthorised request"


def test_a_scan_without_an_authorisation_note_is_refused(
    client: TestClient, auth: dict[str, str], fake_scan: FakeScan
) -> None:
    response = client.post(
        "/api/analyze",
        json={
            "mode": "scan",
            "target_url": "https://app.example.com/",
            "authorized": True,
            "authorization_note": "ok",
        },
        headers=auth,
    )
    assert response.status_code == 400
    assert response.json()["error"] == "missing_authorization_note"
    assert fake_scan.calls == []


def test_a_scan_with_no_target_or_a_relative_target_is_refused(
    client: TestClient, auth: dict[str, str], fake_scan: FakeScan
) -> None:
    base = {"mode": "scan", "authorized": True, "authorization_note": "Authorised by the owner."}
    missing = client.post("/api/analyze", json={**base, "target_url": ""}, headers=auth)
    assert missing.status_code == 400 and missing.json()["error"] == "missing_target"

    # A bare host is now accepted and normalised rather than refused: "app.example.com" is
    # how people name a target, and "localhost:3000" is how they name a container. What must
    # still be refused is anything that is not plainly a host, because a looser rule would
    # turn "javascript:alert(1)" into a string that passes the http(s) scheme check.
    for hostile in ("javascript:alert(1)", "data:text/html,x", "file:///etc/passwd",
                    "ftp://example.com", "../etc/passwd", "not a host"):
        refused = client.post(
            "/api/analyze", json={**base, "target_url": hostile}, headers=auth
        )
        assert refused.status_code == 400, f"{hostile!r} was not refused"
        assert refused.json()["error"] == "invalid_target"
    assert fake_scan.calls == []


def test_a_bare_host_is_normalised_rather_than_refused(
    client: TestClient, auth: dict[str, str], fake_scan: FakeScan
) -> None:
    """The scheme follows the host: loopback is plain HTTP, everything else is TLS."""
    base = {
        "mode": "scan", "authorized": True,
        "authorization_note": "Authorised by the owner.",
        "allow_private_target": True,
    }
    accepted = client.post(
        "/api/analyze", json={**base, "target_url": "app.example.com"}, headers=auth
    )
    assert accepted.status_code in (200, 202), accepted.text

    local = client.post(
        "/api/analyze", json={**base, "target_url": "localhost:3000"}, headers=auth
    )
    assert local.status_code in (200, 202), local.text


def test_scanning_can_be_disabled_at_startup(fake_scan: FakeScan) -> None:
    api = create_app(AppSettings(allow_scan=False))
    try:
        with TestClient(api) as client:
            response = client.post(
                "/api/analyze",
                json={
                    "mode": "scan",
                    "target_url": "https://app.example.com/",
                    "authorized": True,
                    "authorization_note": "Authorised by the system owner, staging only.",
                },
                headers={TOKEN_HEADER: api.state.vulnprio.token},
            )
            assert response.status_code == 403
            assert response.json()["error"] == "scanning_disabled"
            assert fake_scan.calls == []
    finally:
        api.state.vulnprio.close()


def test_an_unknown_preset_is_refused_before_a_job_is_created(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post(
        "/api/analyze", json={"mode": "demo", "attacker": "nation_state_of_atlantis"}, headers=auth
    )
    assert response.status_code == 400
    assert response.json()["error"] == "unknown_attacker"


# ---------------------------------------------------------------------------
# request validation and limits
# ---------------------------------------------------------------------------


def test_body_size_cap_refuses_an_oversized_request(auth: dict[str, str]) -> None:
    api = create_app(AppSettings(max_body_bytes=2048))
    try:
        with TestClient(api) as client:
            headers = {TOKEN_HEADER: api.state.vulnprio.token}
            payload = {
                "mode": "upload",
                "report": {"filename": "big.json", "content_base64": "A" * 8000},
            }
            response = client.post("/api/analyze", json=payload, headers=headers)
            assert response.status_code == 413
            body = response.json()
            assert body["error"] == "body_too_large"
            assert "2048" in body["detail"]

            # A small body on the same server still works.
            assert client.post("/api/analyze", json={"mode": "demo"}, headers=headers).status_code == 202
    finally:
        api.state.vulnprio.close()


def test_multipart_upload_is_capped_while_streaming() -> None:
    api = create_app(AppSettings(max_body_bytes=4096))
    try:
        with TestClient(api) as client:
            response = client.post(
                "/api/analyze/upload",
                headers={TOKEN_HEADER: api.state.vulnprio.token},
                files={"file": ("big.json", b"x" * 200_000, "application/json")},
            )
            assert response.status_code == 413
            assert response.json()["error"] == "body_too_large"
    finally:
        api.state.vulnprio.close()


def test_a_malformed_body_is_a_400_with_the_standard_error_shape(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post(
        "/api/analyze", content=b"{not json", headers={**auth, "Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert set(response.json()) == {"error", "detail"}


def test_an_unknown_mode_is_rejected_by_the_contract(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post("/api/analyze", json={"mode": "mine"}, headers=auth)
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "invalid_request"
    assert "mode" in body["detail"]


def test_an_unknown_field_is_rejected(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post(
        "/api/analyze", json={"mode": "demo", "run_as_root": True}, headers=auth
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def test_an_unknown_job_id_is_a_404_everywhere(client: TestClient, auth: dict[str, str]) -> None:
    for path in (
        "/api/jobs/deadbeef",
        "/api/jobs/deadbeef/result",
        "/api/jobs/deadbeef/report.md",
        "/api/jobs/deadbeef/events",
    ):
        response = client.get(path)
        assert response.status_code == 404, path
        assert response.json()["error"] == "unknown_job", path

    cancelled = client.post("/api/jobs/deadbeef/cancel", headers=auth)
    assert cancelled.status_code == 404
    assert cancelled.json()["error"] == "unknown_job"


def test_a_result_is_not_readable_before_the_job_is_done(
    client: TestClient, auth: dict[str, str]
) -> None:
    created = client.post("/api/analyze", json={"mode": "demo"}, headers=auth).json()
    response = client.get(f"/api/jobs/{created['job_id']}/result")
    if response.status_code == 409:
        assert response.json()["error"] == "not_ready"
    else:
        assert response.status_code == 200, "the only other legal answer is a finished result"
    wait_for_job(client, created["job_id"])


def test_cancelling_a_queued_job_reports_it_cancelled(
    client: TestClient, auth: dict[str, str]
) -> None:
    created = client.post(
        "/api/analyze",
        json={
            "mode": "upload",
            "report": {
                "filename": "zap_sample.json",
                "content_base64": base64.b64encode(ZAP_FIXTURE.read_bytes()).decode("ascii"),
            },
        },
        headers=auth,
    ).json()
    client.post(f"/api/jobs/{created['job_id']}/cancel", headers=auth)
    snapshot = wait_for_job(client, created["job_id"])
    assert snapshot["status"] in {"cancelled", "done"}
    if snapshot["status"] == "cancelled":
        assert snapshot["has_result"] is False
        assert client.get(f"/api/jobs/{created['job_id']}/result").status_code == 409


# ---------------------------------------------------------------------------
# progress streaming
# ---------------------------------------------------------------------------


def test_the_event_stream_reports_progress_and_ends_with_done(
    client: TestClient, auth: dict[str, str]
) -> None:
    created = client.post("/api/analyze", json={"mode": "demo"}, headers=auth).json()
    job_id = created["job_id"]

    events: list[tuple[str, dict[str, Any]]] = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        name = ""
        for line in response.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.split(":", 1)[1].strip())))
                if name == "done":
                    break

    assert events, "the stream produced nothing"
    assert events[-1][0] == "done"
    final = events[-1][1]
    assert final["status"] in TERMINAL
    assert final["job_id"] == job_id
    assert "log_offset" in final, "the client needs an offset to reconnect without gaps"
    assert any(event[1].get("log") for event in events), "log lines should have been streamed"


def test_the_event_stream_resumes_from_an_offset(
    client: TestClient, auth: dict[str, str]
) -> None:
    created = client.post("/api/analyze", json={"mode": "demo"}, headers=auth).json()
    job_id = created["job_id"]
    wait_for_job(client, job_id)

    with client.stream("GET", f"/api/jobs/{job_id}/events?after=9999") as response:
        payloads = [
            json.loads(line.split(":", 1)[1].strip())
            for line in response.iter_lines()
            if line.startswith("data:")
        ]
    assert payloads, "a finished job should still emit its terminal event"
    assert payloads[-1]["status"] in TERMINAL
    assert payloads[-1]["log"] == [], "nothing beyond the requested offset should be resent"


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------


def test_reports_are_served_in_three_formats(client: TestClient, auth: dict[str, str]) -> None:
    created = client.post("/api/analyze", json={"mode": "demo"}, headers=auth).json()
    job_id = created["job_id"]
    assert wait_for_job(client, job_id)["status"] == "done"

    markdown = client.get(f"/api/jobs/{job_id}/report.md")
    assert markdown.status_code == 200
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert markdown.text.strip().startswith("#")

    html = client.get(f"/api/jobs/{job_id}/report.html")
    assert html.status_code == 200
    assert html.headers["content-type"].startswith("text/html")
    assert "<html" in html.text.lower()

    payload = client.get(f"/api/jobs/{job_id}/report.json")
    assert payload.status_code == 200
    assert payload.headers["content-type"].startswith("application/json")
    json.loads(payload.text)

    download = client.get(f"/api/jobs/{job_id}/report.md?download=1")
    assert "attachment" in download.headers["content-disposition"]
    assert "vulnprio-report.md" in download.headers["content-disposition"]


def test_the_embedded_report_is_dressed_for_the_page_and_the_download_is_not(
    client: TestClient, auth: dict[str, str]
) -> None:
    """The results page embeds the report; a download has to stand on its own afterwards.

    The page passes its own palette and asks for the page furniture to be dropped, because
    a report carrying its own background and reading width inside the page reads as a
    second document framed in the first. A downloaded file has no host page, so it must
    keep both whatever the page happened to ask for.
    """
    created = client.post("/api/analyze", json={"mode": "demo"}, headers=auth).json()
    job_id = created["job_id"]
    assert wait_for_job(client, job_id)["status"] == "done"

    embedded = client.get(f"/api/jobs/{job_id}/report.html?embed=1&theme=dark")
    assert embedded.status_code == 200
    root = re.search(r"<html[^>]*>", embedded.text).group(0)
    assert 'data-theme="dark"' in root
    assert "data-embedded" in root

    plain = client.get(f"/api/jobs/{job_id}/report.html")
    assert re.search(r"<html[^>]*>", plain.text).group(0) == '<html lang="en">'

    saved = client.get(f"/api/jobs/{job_id}/report.html?embed=1&theme=dark&download=1")
    assert "attachment" in saved.headers["content-disposition"]
    assert re.search(r"<html[^>]*>", saved.text).group(0) == '<html lang="en">'


def test_an_unknown_report_format_is_a_404(client: TestClient, auth: dict[str, str]) -> None:
    created = client.post("/api/analyze", json={"mode": "demo"}, headers=auth).json()
    wait_for_job(client, created["job_id"])
    response = client.get(f"/api/jobs/{created['job_id']}/report.docx")
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_format"


def test_the_builtin_summary_is_written_when_the_report_package_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A build without vulnprio.report still hands the operator a usable document."""
    import importlib

    from vulnprio.web import app as app_module
    from vulnprio.web.demo import demo_dashboard

    real_import = importlib.import_module

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "vulnprio.report":
            raise ImportError("no module named vulnprio.report")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(app_module.importlib, "import_module", refuse)
    data = demo_dashboard()

    markdown = app_module.build_report_document(data, "md")
    assert markdown.startswith("# vulnprio")
    assert "vulnprio.report is not available" in markdown
    assert "Remediation queue" in markdown
    assert data.findings[0].name in markdown

    html = app_module.build_report_document(data, "html")
    assert html.lstrip().lower().startswith("<!doctype html")

    payload = json.loads(app_module.build_report_document(data, "json"))
    assert payload["summary"]["n_findings"] == len(data.findings)
    assert "report_generator" in payload["notes"]


# ---------------------------------------------------------------------------
# static assets and the served page
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/", "text/html"),
        ("/index.html", "text/html"),
        ("/app.js", "text/javascript"),
        ("/analyze.js", "text/javascript"),
        ("/styles.css", "text/css"),
        ("/data.js", "text/javascript"),
        ("/data.json", "application/json"),
    ],
)
def test_static_assets_are_served_with_the_right_content_type(
    client: TestClient, path: str, content_type: str
) -> None:
    response = client.get(path)
    assert response.status_code == 200, path
    assert response.headers["content-type"].startswith(content_type), (
        path, response.headers["content-type"]
    )


def test_the_index_carries_this_process_session_token(client: TestClient, token: str) -> None:
    body = client.get("/").text
    assert "window.VULNPRIO_SESSION" in body
    assert token in body
    assert "window.VULNPRIO_SESSION = null;" not in body, "the marker should be replaced"


def test_the_bundled_payload_is_empty_without_a_run(client: TestClient) -> None:
    assert client.get("/data.js").text.strip() == "window.VULNPRIO_DATA = {};"
    assert client.get("/data.json").json() == {}


def test_a_bundled_run_is_served_as_data_js() -> None:
    from vulnprio.web.demo import demo_dashboard

    data = demo_dashboard()
    api = create_app(AppSettings(data=data))
    try:
        with TestClient(api) as client:
            assert client.get("/api/health").json()["has_bundled_run"] is True
            payload = client.get("/data.json").json()
            assert payload["summary"]["n_findings"] == len(data.findings)
            assert client.get("/data.js").text.startswith("window.VULNPRIO_DATA = {")
    finally:
        api.state.vulnprio.close()


def test_unknown_api_endpoints_return_the_standard_error_shape(client: TestClient) -> None:
    response = client.get("/api/nonexistent")
    assert response.status_code == 404
    assert set(response.json()) == {"error", "detail"}
    assert response.json()["error"] == "unknown_endpoint"


def test_security_headers_are_present(client: TestClient) -> None:
    headers = client.get("/").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"


def test_openapi_documents_the_contract(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    paths = set(schema["paths"])
    assert {
        "/api/health", "/api/config", "/api/analyze", "/api/analyze/upload",
        "/api/jobs/{job_id}", "/api/jobs/{job_id}/result", "/api/jobs/{job_id}/events",
        "/api/jobs/{job_id}/cancel", "/api/novelty",
    } <= paths, sorted(paths)
