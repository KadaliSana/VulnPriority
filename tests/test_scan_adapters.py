"""External scanner adapters: ranking, safe argv, profile mapping, ingestion, fallback.

No binary is ever executed here. ``shutil.which`` is replaced with a lookup over a
synthetic PATH and ``subprocess.run`` with a recorder that writes a committed fixture to
the report path the tool was told to use. That makes the whole external path testable end
to end - argv, timeout, report parsing, identifier construction, fallback - on a machine
with no scanner installed, which is what this machine is.

What these tests cannot cover is stated plainly in the module docstring of
:mod:`vulnpriority.scan.adapters`: the exact flag spellings are taken from each tool's
documentation and have not been run against a real binary here.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from vulnpriority.core.models import Scan
from vulnpriority.ingest.nikto import NiktoParser
from vulnpriority.ingest.wapiti import WapitiParser
from vulnpriority.scan import (
    NotAuthorizedError,
    OutOfScopeError,
    ScanProfile,
    ScanRequest,
    assess_target,
)
from vulnpriority.scan.adapters import (
    TOOL_PREFERENCE,
    TOOL_SPECS,
    ZAP_DOCKER_IMAGE,
    ExternalTool,
    ExternalToolError,
    build_argv,
    describe_external_tools,
    detect_external_tools,
    ingest_report,
    run_external,
    safe_argument,
    scanner_environment,
    select_tools,
    tool_version,
)

TARGET = "https://shop.example.com/"
FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "scans"


def make_request(**overrides) -> ScanRequest:
    base = {
        "target_url": TARGET,
        "authorized": True,
        "authorization_note": "Engagement PT-2024-114, authorised by the application owner",
        "requests_per_second": 20.0,
    }
    base.update(overrides)
    return ScanRequest(**base)


def tool(name: str, *, version: str | None = None) -> ExternalTool:
    spec = TOOL_SPECS[name]
    return ExternalTool(
        name=name,
        executable=Path(f"/usr/local/bin/{spec.candidates[0]}"),
        report_suffix=spec.report_suffix,
        version=version,
        image=spec.image,
    )


def fake_which(installed: set[str]):
    """A ``shutil.which`` replacement over a synthetic set of installed executables."""

    def _which(command: str, path: str | None = None, **kwargs) -> str | None:
        return f"/usr/local/bin/{command}" if command in installed else None

    return _which


class RecordingRunner:
    """Stands in for ``subprocess.run``; records argv and writes a report if asked to."""

    def __init__(
        self,
        *,
        report_body: str | None = None,
        returncode: int = 0,
        raises: BaseException | None = None,
        stdout: str = "",
    ) -> None:
        self.calls: list[dict] = []
        self.report_body = report_body
        self.returncode = returncode
        self.raises = raises
        self.stdout = stdout

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": argv, **kwargs})
        if self.raises is not None:
            raise self.raises
        if self.report_body is not None:
            destination = _output_path(argv, kwargs.get("cwd"))
            if destination is not None:
                Path(destination).parent.mkdir(parents=True, exist_ok=True)
                Path(destination).write_text(self.report_body, encoding="utf-8")
        return subprocess.CompletedProcess(argv, self.returncode, stdout=self.stdout, stderr="")

    @property
    def argvs(self) -> list[list[str]]:
        return [call["argv"] for call in self.calls]


def _output_path(argv: list[str], cwd: str | None) -> str | None:
    """Where this command was told to write its report."""
    for flag in ("-output", "--output", "-quickout", "-o"):
        if flag in argv:
            return argv[argv.index(flag) + 1]
    if "-J" in argv:                      # ZAP writes -J relative to its working directory
        name = argv[argv.index("-J") + 1]
        return str(Path(cwd or ".") / name)
    return None


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


CANONICAL_SCAN = {
    "scan_id": "scan_test",
    "app_id": "app_test",
    "app_name": "Example Shop",
    "sector": "ecommerce",
    "scanned_at": "2024-06-01T09:30:00",
    "scanner_name": "nuclei",
    "hosts": ["shop.example.com"],
    "endpoints": [
        {
            "endpoint_id": "ep_1",
            "app_id": "app_test",
            "host": "shop.example.com",
            "url": "https://shop.example.com/login",
            "path": "/login",
            "method": "POST",
        }
    ],
    "findings": [
        {
            "finding_id": "f_1",
            "scan_id": "scan_test",
            "app_id": "app_test",
            "endpoint_id": "ep_1",
            "name": "Example issue",
            "cwe_id": 79,
            "scanner": "nuclei",
            "scanner_severity": "high",
            "description": {"text": "found something", "provenance": "scanner_output"},
            "observed_at": "2024-06-01T09:30:00",
        }
    ],
}


# ---------------------------------------------------------------------------
# Detection and ranking
# ---------------------------------------------------------------------------


def test_no_tool_installed_is_reported_clearly_with_an_install_hint():
    found = detect_external_tools(which=fake_which(set()))
    assert found == ()
    message = describe_external_tools(found)
    assert "No external scanner" in message
    assert "built-in" in message
    assert "install" in message.lower(), "a clean machine must be told how to get a real scanner"


def test_each_supported_tool_is_detected():
    for name in TOOL_PREFERENCE:
        for candidate in TOOL_SPECS[name].candidates:
            found = detect_external_tools(which=fake_which({candidate}))
            assert [item.name for item in found] == [name], f"{candidate} was not detected"
            assert found[0].executable == Path(f"/usr/local/bin/{candidate}")


def test_zap_is_detected_through_docker():
    """Docker on PATH means ZAP is available, because that is how most people run it."""
    found = detect_external_tools(which=fake_which({"docker"}))
    assert [item.name for item in found] == ["zap-docker"]
    assert found[0].image == ZAP_DOCKER_IMAGE


def test_detection_honours_the_preference_order():
    found = detect_external_tools(which=fake_which({"nikto", "nuclei", "docker", "wapiti"}))
    assert [item.name for item in found] == ["zap-docker", "nuclei", "wapiti", "nikto"]


def test_zap_outranks_nuclei_because_it_crawls():
    usable, _ = select_tools(
        make_request(profile=ScanProfile.ACTIVE), tools=(tool("nuclei"), tool("zap-docker"))
    )
    assert [item.name for item in usable] == ["zap-docker", "nuclei"]


def test_caller_can_override_the_preference_order():
    usable, _ = select_tools(
        make_request(profile=ScanProfile.ACTIVE),
        tools=(tool("nuclei"), tool("zap-docker")),
        order=("nuclei", "zap-docker"),
    )
    assert [item.name for item in usable] == ["nuclei", "zap-docker"]


def test_detection_can_search_an_explicit_path(tmp_path):
    """With a real ``shutil.which`` and a synthetic PATH, only the planted tool is found."""
    executable = tmp_path / "nuclei"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    def _which(command, path=None, **kwargs):
        candidate = Path(path or "") / command
        return str(candidate) if candidate.exists() else None

    found = detect_external_tools(path=str(tmp_path), which=_which)
    assert [item.name for item in found] == ["nuclei"]


# ---------------------------------------------------------------------------
# Profile mapping: the user's choice is never quietly upgraded
# ---------------------------------------------------------------------------


def test_passive_request_skips_tools_that_cannot_be_passive():
    """Nuclei and Nikto request paths nobody linked to, so they are not passive scanners."""
    usable, skipped = select_tools(
        make_request(profile=ScanProfile.PASSIVE),
        tools=(tool("nuclei"), tool("nikto"), tool("wapiti")),
    )
    assert [item.name for item in usable] == ["wapiti"]
    assert any("nuclei" in reason and "passive" in reason for reason in skipped)
    assert any("nikto" in reason for reason in skipped)


def test_active_request_can_use_every_tool():
    usable, skipped = select_tools(
        make_request(profile=ScanProfile.ACTIVE),
        tools=tuple(tool(name) for name in TOOL_PREFERENCE),
    )
    assert [item.name for item in usable] == list(TOOL_PREFERENCE)
    assert skipped == ()


def test_build_argv_refuses_a_profile_the_tool_cannot_honour(tmp_path):
    with pytest.raises(ExternalToolError, match="did not choose"):
        build_argv(tool("nuclei"), make_request(profile=ScanProfile.PASSIVE), tmp_path / "r.jsonl")


def test_zap_docker_maps_passive_to_baseline_and_active_to_full_scan(tmp_path):
    report = tmp_path / "zap-docker-report.json"

    passive = build_argv(tool("zap-docker"), make_request(profile=ScanProfile.PASSIVE), report)[0]
    active = build_argv(tool("zap-docker"), make_request(profile=ScanProfile.ACTIVE), report)[0]

    assert "zap-baseline.py" in passive and "zap-full-scan.py" not in passive
    assert "zap-full-scan.py" in active and "zap-baseline.py" not in active


def test_zap_docker_mounts_only_the_report_directory(tmp_path):
    report = tmp_path / "zap-docker-report.json"
    argv = build_argv(tool("zap-docker"), make_request(), report)[0]

    mount = argv[argv.index("-v") + 1]
    assert mount == f"{tmp_path.resolve()}:/zap/wrk/:rw"
    assert argv[:3] == [str(Path("/usr/local/bin/docker")), "run", "--rm"]
    assert ZAP_DOCKER_IMAGE in argv
    assert argv[argv.index("-J") + 1] == report.name
    assert TARGET in argv


def test_zap_docker_rewrites_a_loopback_target_it_could_not_otherwise_reach(tmp_path):
    """localhost inside a container is the container. Pointing ZAP at it scans nothing.

    The scan then completes in seconds having found no findings, which reads as a clean
    target rather than as a scan that never happened - the most common way a containerised
    assessment produces a silent false negative.
    """
    argv = build_argv(
        tool("zap-docker"),
        make_request(target_url="http://localhost:3000/", allow_private_targets=True),
        tmp_path / "r.json",
    )[0]
    assert "http://host.docker.internal:3000/" in argv
    assert not any("localhost" in item for item in argv)
    assert argv[argv.index("--add-host") + 1] == "host.docker.internal:host-gateway"


def test_a_routable_target_is_passed_through_untouched(tmp_path):
    """Only loopback is rewritten. A private LAN address is reachable from the container's
    bridge network as it stands, and rewriting it would scan the wrong machine."""
    for target in (TARGET, "http://192.168.1.10/"):
        argv = build_argv(
            tool("zap-docker"),
            make_request(target_url=target, allow_private_targets=True),
            tmp_path / "r.json",
        )[0]
        assert argv[argv.index("-t") + 1] == target
        assert not any(item.startswith("http") and "host.docker.internal" in item for item in argv)


def test_zap_docker_is_capped_by_the_external_budget_not_the_crawler_budget(tmp_path):
    """The two budgets measure different things and conflating them truncated every scan.

    ``time_budget_s`` bounds the built-in crawler, which is polite and quick.
    ``external_time_budget_s`` bounds a real scanner, which is neither. Passing the first
    to ZAP gave a full active scan two minutes, so it was killed part-way through at a
    different point on every run and the same target produced different findings each time.
    """
    argv = build_argv(
        tool("zap-docker"),
        make_request(time_budget_s=60.0, external_time_budget_s=600.0),
        tmp_path / "r.json",
    )[0]
    assert argv[argv.index("-T") + 1] == "10"       # minutes, from the external budget

    # The crawler's budget must not reach it at all.
    argv = build_argv(
        tool("zap-docker"),
        make_request(time_budget_s=3000.0, external_time_budget_s=120.0),
        tmp_path / "r.json",
    )[0]
    assert argv[argv.index("-T") + 1] == "2"


# ---------------------------------------------------------------------------
# argv construction
# ---------------------------------------------------------------------------


def test_argv_is_a_list_of_strings_never_a_shell_string(tmp_path):
    request = make_request(profile=ScanProfile.ACTIVE)
    for name in TOOL_PREFERENCE:
        steps = build_argv(tool(name), request, tmp_path / "report.json")
        assert isinstance(steps, list) and steps
        for argv in steps:
            assert isinstance(argv, list)
            assert all(isinstance(item, str) for item in argv)
        # The URL is one whole argument of the scanning step, never concatenated into
        # another one and never part of a string a shell would parse.
        assert any(TARGET in argv for argv in steps), f"{name} never passes the target"


def test_argv_never_interpolates_the_target_into_another_argument(tmp_path):
    """A hostile URL cannot escape into a neighbouring argument or a shell."""
    request = make_request(target_url="https://shop.example.com/?a=1", profile=ScanProfile.ACTIVE)
    for name in TOOL_PREFERENCE:
        for argv in build_argv(tool(name), request, tmp_path / "r.json"):
            assert argv.count(request.target_url) <= 1
            for item in argv:
                if item != request.target_url:
                    assert request.target_url not in item


def test_option_looking_arguments_are_refused():
    assert safe_argument("https://example.com/") == "https://example.com/"
    for bad in ("-oN /tmp/x", "--config=evil", "", "  padded  ", "line\nbreak", "nul\x00byte"):
        with pytest.raises(ExternalToolError):
            safe_argument(bad)


def test_build_argv_demands_authorisation(tmp_path):
    with pytest.raises(NotAuthorizedError):
        build_argv(tool("zap-docker"), ScanRequest(target_url=TARGET), tmp_path / "r.json")


def test_build_argv_refuses_a_metadata_target(tmp_path):
    request = make_request(target_url="http://169.254.169.254/latest/")
    with pytest.raises(OutOfScopeError):
        build_argv(tool("zap-docker"), request, tmp_path / "r.json")


def test_the_authorised_host_is_what_reaches_the_tool(tmp_path):
    request = make_request(target_url="https://shop.example.com/basket")
    argv = build_argv(tool("zap-docker"), request, tmp_path / "r.json")[0]
    assert argv[argv.index("-t") + 1] == "https://shop.example.com/basket"


def test_passive_wapiti_runs_no_attack_module(tmp_path):
    argv = build_argv(tool("wapiti"), make_request(profile=ScanProfile.PASSIVE), tmp_path / "r.json")[0]
    assert argv[argv.index("-m") + 1] == ""

    active = build_argv(tool("wapiti"), make_request(profile=ScanProfile.ACTIVE), tmp_path / "r.json")[0]
    assert "-m" not in active


def test_nuclei_excludes_destructive_templates(tmp_path):
    argv = build_argv(tool("nuclei"), make_request(profile=ScanProfile.ACTIVE), tmp_path / "r.jsonl")[0]
    excluded = argv[argv.index("-exclude-tags") + 1]
    for tag in ("dos", "fuzzing", "intrusive", "brute-force"):
        assert tag in excluded


def test_rate_limit_and_budget_reach_the_external_tool(tmp_path):
    request = make_request(requests_per_second=2.0, time_budget_s=30.0, profile=ScanProfile.ACTIVE)
    argv = build_argv(tool("nuclei"), request, tmp_path / "r.jsonl")[0]
    assert argv[argv.index("-rate-limit") + 1] == "120"

    # Nikto is an external scanner too, so it is bounded by the external budget.
    timed = make_request(external_time_budget_s=30.0, profile=ScanProfile.ACTIVE)
    nikto_argv = build_argv(tool("nikto"), timed, tmp_path / "r.json")[0]
    assert nikto_argv[nikto_argv.index("-maxtime") + 1] == "30"


def test_the_zap_container_cannot_eat_the_machine_that_hosts_it(tmp_path):
    """A container with no memory limit killed the VM running the container runtime.

    An active scan reached 4.94GiB of a Docker Desktop WSL VM's 7.68GiB in three minutes.
    The VM died, the client reported ``exit 125: error waiting for container: unexpected
    EOF``, and - the expensive part - the runtime was left half-alive, accepting connections
    on its pipes and port forwards with nothing behind them, so every subsequent scan failed
    too. Three consecutive runs returned zero findings against a daemon the first one killed.
    """
    request = make_request(external_memory_gb=6.0, profile=ScanProfile.ACTIVE)
    argv = build_argv(tool("zap-docker"), request, tmp_path / "r.json")[0]

    assert argv[argv.index("--memory") + 1] == "6g"
    # Equal to --memory, so the container cannot swap its way to the same exhaustion.
    assert argv[argv.index("--memory-swap") + 1] == "6g"

    # Measured: with no limit the JVM took a 1.92GB heap and the container still reached
    # 4.94GB, so ZAP's non-heap footprint is around 3GB. The heap must leave room for it.
    heap = next(v for v in argv if v.startswith("ZAP_JVM_OPTS="))
    megabytes = int(heap.split("-Xmx")[1].rstrip("m"))
    assert megabytes <= 6 * 1024 * 0.5, (
        "the JVM heap must leave room for the part of ZAP that -Xmx cannot see: thread "
        "stacks, direct buffers and the in-memory session. A container OOM-killed while "
        "the JVM still believes it has room writes no report at all"
    )


def test_the_memory_ceiling_is_configurable(tmp_path):
    argv = build_argv(tool("zap-docker"),
                      make_request(external_memory_gb=2.0, profile=ScanProfile.ACTIVE),
                      tmp_path / "r.json")[0]
    assert argv[argv.index("--memory") + 1] == "2g"
    assert "ZAP_JVM_OPTS=-Xmx1024m" in argv


def test_zap_cli_is_two_argv_steps(tmp_path):
    steps = build_argv(tool("zap-cli"), make_request(profile=ScanProfile.ACTIVE), tmp_path / "r.json")
    assert len(steps) == 2
    assert steps[0][1] == "quick-scan"
    assert steps[1][1] == "report"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def test_run_external_never_uses_a_shell(tmp_path):
    runner = RecordingRunner(report_body='{"x": 1}')
    report = run_external(tool("nuclei"), make_request(profile=ScanProfile.ACTIVE),
                          out_dir=tmp_path, runner=runner)

    assert report.exists()
    call = runner.calls[0]
    assert call["shell"] is False
    assert isinstance(call["argv"], list)
    assert call["timeout"] is None      # no outer cap unless a caller asks for one
    assert call["capture_output"] is True


def test_run_external_applies_a_timeout(tmp_path):
    runner = RecordingRunner(raises=subprocess.TimeoutExpired(cmd="nuclei", timeout=5))
    with pytest.raises(ExternalToolError, match="timeout"):
        run_external(tool("nuclei"), make_request(profile=ScanProfile.ACTIVE),
                     out_dir=tmp_path, runner=runner, timeout_s=5)


def test_run_external_reports_a_missing_executable(tmp_path):
    runner = RecordingRunner(raises=FileNotFoundError("no such file"))
    with pytest.raises(ExternalToolError, match="not executable"):
        run_external(tool("nuclei"), make_request(profile=ScanProfile.ACTIVE),
                     out_dir=tmp_path, runner=runner)


def test_run_external_fails_when_no_report_is_produced(tmp_path):
    runner = RecordingRunner(report_body=None)
    with pytest.raises(ExternalToolError, match="no report"):
        run_external(tool("nuclei"), make_request(profile=ScanProfile.ACTIVE),
                     out_dir=tmp_path, runner=runner)


def test_run_external_surfaces_a_hard_failure(tmp_path):
    runner = RecordingRunner(returncode=127)
    with pytest.raises(ExternalToolError, match="status 127"):
        run_external(tool("nuclei"), make_request(profile=ScanProfile.ACTIVE),
                     out_dir=tmp_path, runner=runner)


def test_tool_version_is_read_from_the_banner():
    runner = RecordingRunner(stdout="Nuclei Engine Version: v3.1.4")
    assert tool_version(tool("nuclei"), runner=runner) == "3.1.4"
    assert runner.calls[0]["shell"] is False


def test_tool_version_failure_is_not_fatal():
    runner = RecordingRunner(raises=FileNotFoundError("gone"))
    assert tool_version(tool("nuclei"), runner=runner) is None


# ---------------------------------------------------------------------------
# Ingestion: every tool's report reaches a vulnpriority parser
# ---------------------------------------------------------------------------


def test_ingest_report_reads_the_canonical_format(tmp_path):
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(CANONICAL_SCAN), encoding="utf-8")
    scan = ingest_report(path)
    assert isinstance(scan, Scan)
    assert scan.scan_id == "scan_test"


def test_ingest_report_reads_a_zap_report():
    scan = ingest_report(FIXTURES / "zap_sample.json")
    assert scan.scanner_name == "zap"
    assert scan.findings


def test_ingest_report_reads_a_wapiti_report():
    """Without this the adapter would run Wapiti and then throw the output away."""
    scan = ingest_report(FIXTURES / "wapiti_sample.json")
    assert scan.scanner_name == "wapiti"
    assert scan.scanner_version == "3.1.7"
    assert scan.hosts == ("shop.example.com",)
    assert {finding.cwe_id for finding in scan.findings} >= {89, 79, 614}
    assert all(finding.dedup_key is None or finding.dedup_key for finding in scan.findings)


def test_wapiti_parser_reads_all_three_result_sections():
    scan = WapitiParser().parse(FIXTURES / "wapiti_sample.json")
    names = {finding.name for finding in scan.findings}
    assert "SQL Injection" in names           # vulnerabilities
    assert "Internal Server Error" in names   # anomalies
    assert "Fingerprint web technology" in names   # additionals


def test_wapiti_findings_carry_provenance_and_parameters():
    scan = WapitiParser().parse(FIXTURES / "wapiti_sample.json")
    product = next(ep for ep in scan.endpoints if ep.path == "/product.php")
    assert set(product.parameters) == {"id", "ref"}

    sqli = next(finding for finding in scan.findings if finding.cwe_id == 89)
    assert sqli.description.provenance.value == "scanner_output"
    assert any(item.provenance.value == "target_response" for item in sqli.evidence)
    assert sqli.scanner_severity.value == "high"


def test_ingest_report_reads_a_nikto_report():
    scan = ingest_report(FIXTURES / "nikto_sample.json")
    assert scan.scanner_name == "nikto"
    assert scan.hosts == ("shop.example.com",)
    assert scan.findings


def test_nikto_findings_get_a_cwe_where_the_message_is_unambiguous():
    scan = NiktoParser().parse(FIXTURES / "nikto_sample.json")
    by_plugin = {finding.scanner_plugin_id: finding for finding in scan.findings}
    assert by_plugin["999957"].cwe_id == 1021        # X-Frame-Options / clickjacking
    assert by_plugin["000330"].cwe_id == 548         # directory indexing
    assert by_plugin["000451"].cwe_id == 650         # TRACE
    assert by_plugin["000004"].cve_ids == ("CVE-2003-1418",)


def test_nikto_banner_becomes_technology_evidence():
    scan = NiktoParser().parse(FIXTURES / "nikto_sample.json")
    products = {component.product for component in scan.tech_stack}
    assert "http_server" in products and "openssl" in products


def test_nikto_https_port_yields_https_endpoints():
    scan = NiktoParser().parse(FIXTURES / "nikto_sample.json")
    assert all(endpoint.url.startswith("https://") for endpoint in scan.endpoints)


def test_ingest_report_rejects_an_unrecognised_file(tmp_path):
    path = tmp_path / "junk.txt"
    path.write_text("not a scanner report", encoding="utf-8")
    with pytest.raises(ExternalToolError, match="no vulnpriority parser"):
        ingest_report(path)


# ---------------------------------------------------------------------------
# The environment report the web application shows the operator
# ---------------------------------------------------------------------------


def test_environment_lists_every_tool_installed_or_not():
    env = scanner_environment(ScanProfile.PASSIVE, which=fake_which(set()), include_versions=False)
    assert {status.name for status in env.tools} == set(TOOL_PREFERENCE)
    assert env.preferred is None
    assert env.builtin_fallback is True
    assert all(status.install_hint for status in env.missing)
    assert "install" in env.notice.lower()


def test_environment_does_not_report_the_container_runtime_as_the_scanner_version():
    """``docker --version`` answers for Docker, not for ZAP, so it must not be shown.

    Caught on a real machine: the site advertised "zap-docker 29.7.2", which is the Docker
    Desktop version. Carried into a report that becomes "scanned by zap 29.7.2" -- a false
    statement about which scanner produced the findings, in the one document a reader trusts
    to tell them that. ZAP's own version is only knowable once the image has run, so before
    then the honest answer is no version at all.
    """
    runner = RecordingRunner(stdout="Docker version 24.0.7, build afdd53b")
    env = scanner_environment(ScanProfile.PASSIVE, which=fake_which({"docker"}), runner=runner)

    docker = next(status for status in env.tools if status.name == "zap-docker")
    assert docker.installed is True
    assert not docker.version, "the container runtime's version was reported as the scanner's"
    assert docker.image == ZAP_DOCKER_IMAGE
    assert env.preferred == "zap-docker"
    assert env.builtin_fallback is False


def test_environment_still_reports_a_native_scanner_version():
    """The narrowing must not blind the case where the binary IS the scanner."""
    runner = RecordingRunner(stdout="nuclei 3.1.4")
    env = scanner_environment(ScanProfile.ACTIVE, which=fake_which({"nuclei"}), runner=runner)

    nuclei = next(status for status in env.tools if status.name == "nuclei")
    assert nuclei.installed is True
    assert nuclei.version


def test_environment_explains_a_profile_mismatch():
    """Nuclei installed but a passive scan asked for: say so, do not silently downgrade."""
    env = scanner_environment(
        ScanProfile.PASSIVE, which=fake_which({"nuclei"}), include_versions=False
    )
    assert env.preferred is None
    assert env.builtin_fallback is True
    assert any("nuclei" in reason for reason in env.skipped)
    assert "passive" in env.notice

    active = scanner_environment(
        ScanProfile.ACTIVE, which=fake_which({"nuclei"}), include_versions=False
    )
    assert active.preferred == "nuclei"


def test_environment_marks_profile_support_per_tool():
    env = scanner_environment(ScanProfile.PASSIVE, which=fake_which(set()), include_versions=False)
    support = {status.name: status.supports_requested_profile for status in env.tools}
    assert support["zap-docker"] is True
    assert support["wapiti"] is True
    assert support["nuclei"] is False
    assert support["nikto"] is False


# ---------------------------------------------------------------------------
# assess_target: a real scanner first, the built-in one as fallback
# ---------------------------------------------------------------------------


def _offline_client(request: ScanRequest):
    """An HttpClient wired to a trivial fake site, so the built-in path needs no network."""
    import httpx

    from vulnpriority.scan import HttpClient

    class Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    clock = Clock()
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, html="<html><body>hello</body></html>")
    )
    return HttpClient(request, transport=transport, clock=clock, sleep=clock.sleep)


def test_external_tools_are_preferred_by_default(tmp_path):
    """The headline change: an installed scanner runs, the built-in one does not."""
    request = make_request(profile=ScanProfile.ACTIVE)
    runner = RecordingRunner(report_body=fixture("zap_sample.json"))

    outcome = assess_target(
        request, tools=(tool("zap-docker"),), runner=runner, out_dir=tmp_path
    )

    assert outcome.tool == "zap-docker"
    assert outcome.scan.scanner_name == "zap"
    assert outcome.scan.scanner_version == "2.14.0"
    assert outcome.scan.findings
    assert "zap-docker ran this active scan" in outcome.tool_selection
    assert runner.calls, "the tool was not actually invoked"


def test_the_builtin_scanner_can_still_be_forced(tmp_path):
    request = make_request()
    runner = RecordingRunner(report_body=fixture("zap_sample.json"))

    outcome = assess_target(
        request,
        force_builtin=True,
        tools=(tool("zap-docker"),),
        runner=runner,
        out_dir=tmp_path,
        client=_offline_client(request),
    )

    assert outcome.tool == "vulnpriority-scan"
    assert outcome.scan.scanner_name == "vulnpriority-scan"
    assert runner.calls == [], "force_builtin must not launch an external tool"
    assert "requested explicitly" in outcome.tool_selection


def test_prefer_external_false_also_forces_the_builtin(tmp_path):
    request = make_request()
    runner = RecordingRunner(report_body=fixture("zap_sample.json"))
    outcome = assess_target(
        request,
        prefer_external=False,
        tools=(tool("zap-docker"),),
        runner=runner,
        out_dir=tmp_path,
        client=_offline_client(request),
    )
    assert outcome.tool == "vulnpriority-scan"
    assert runner.calls == []


def test_wapiti_report_reaches_the_scan_end_to_end(tmp_path):
    request = make_request(profile=ScanProfile.PASSIVE)
    runner = RecordingRunner(report_body=fixture("wapiti_sample.json"))

    outcome = assess_target(request, tools=(tool("wapiti"),), runner=runner, out_dir=tmp_path)

    assert outcome.tool == "wapiti"
    assert outcome.scan.scanner_name == "wapiti"
    assert outcome.scan.scanner_version == "3.1.7"
    assert outcome.scan.findings
    assert outcome.report_path is not None and outcome.report_path.exists()


def test_nikto_report_reaches_the_scan_end_to_end(tmp_path):
    request = make_request(profile=ScanProfile.ACTIVE)
    runner = RecordingRunner(report_body=fixture("nikto_sample.json"), stdout="Nikto 2.5.0")

    outcome = assess_target(request, tools=(tool("nikto"),), runner=runner, out_dir=tmp_path)

    assert outcome.tool == "nikto"
    assert outcome.scan.scanner_name == "nikto"
    assert outcome.scan.scanner_version == "2.5.0", "the version must reach the report"
    assert outcome.scan.findings


def test_a_failing_tool_falls_through_to_the_next_one(tmp_path):
    """Ranking is not a single shot: if ZAP fails, nuclei gets a turn."""
    request = make_request(profile=ScanProfile.ACTIVE)

    class Sequenced(RecordingRunner):
        def __call__(self, argv, **kwargs):
            if "docker" in str(argv[0]):
                self.calls.append({"argv": argv, **kwargs})
                raise FileNotFoundError("docker daemon not running")
            return super().__call__(argv, **kwargs)

    runner = Sequenced(report_body=json.dumps(CANONICAL_SCAN))
    outcome = assess_target(
        request, tools=(tool("zap-docker"), tool("nuclei")), runner=runner, out_dir=tmp_path
    )

    assert outcome.tool == "nuclei"
    assert any("zap-docker failed" in note for note in outcome.external_attempts)
    assert outcome.available_tools == ("zap-docker", "nuclei")


def test_falls_back_to_the_builtin_when_no_tool_is_installed():
    request = make_request()
    outcome = assess_target(request, tools=(), client=_offline_client(request))

    assert outcome.tool == "vulnpriority-scan"
    assert outcome.scan.findings
    assert "No external scanner was found" in outcome.tool_selection
    assert "install" in outcome.tool_selection.lower()
    assert any("No external scanner" in error for error in outcome.errors)


def test_falls_back_to_the_builtin_when_every_tool_fails(tmp_path):
    request = make_request(profile=ScanProfile.ACTIVE)
    runner = RecordingRunner(raises=FileNotFoundError("gone"))

    outcome = assess_target(
        request,
        tools=(tool("zap-docker"), tool("nuclei")),
        runner=runner,
        out_dir=tmp_path,
        client=_offline_client(request),
    )

    assert outcome.tool == "vulnpriority-scan"
    assert len(outcome.external_attempts) == 2
    assert all("failed" in note for note in outcome.external_attempts)


def test_falls_back_when_the_only_tool_cannot_serve_the_profile(tmp_path):
    """A passive request with only nuclei installed: built-in, and an explanation."""
    request = make_request(profile=ScanProfile.PASSIVE)
    runner = RecordingRunner(report_body=json.dumps(CANONICAL_SCAN))

    outcome = assess_target(
        request,
        tools=(tool("nuclei"),),
        runner=runner,
        out_dir=tmp_path,
        client=_offline_client(request),
    )

    assert outcome.tool == "vulnpriority-scan"
    assert runner.calls == [], "a tool that cannot be passive must not be run for a passive scan"
    assert any("nuclei" in note and "passive" in note for note in outcome.external_attempts)


def test_tool_attempts_are_bounded(tmp_path):
    request = make_request(profile=ScanProfile.ACTIVE)
    runner = RecordingRunner(raises=FileNotFoundError("gone"))

    outcome = assess_target(
        request,
        tools=tuple(tool(name) for name in TOOL_PREFERENCE),
        runner=runner,
        out_dir=tmp_path,
        max_tool_attempts=1,
        client=_offline_client(request),
    )

    assert outcome.tool == "vulnpriority-scan"
    assert len(runner.calls) == 1, "only one tool should have been tried"


def test_assess_target_still_demands_authorisation_for_external_tools(tmp_path):
    runner = RecordingRunner(report_body=json.dumps(CANONICAL_SCAN))
    with pytest.raises(NotAuthorizedError):
        assess_target(
            ScanRequest(target_url=TARGET),
            tools=(tool("zap-docker"),),
            runner=runner,
            out_dir=tmp_path,
        )
    assert runner.calls == [], "nothing may be executed for an unauthorised request"


def test_assess_target_refuses_a_metadata_target_before_launching_a_tool(tmp_path):
    runner = RecordingRunner(report_body=json.dumps(CANONICAL_SCAN))
    request = make_request(target_url="http://169.254.169.254/latest/")
    with pytest.raises(OutOfScopeError):
        assess_target(request, tools=(tool("zap-docker"),), runner=runner, out_dir=tmp_path)
    assert runner.calls == []


def test_no_outer_kill_switch_by_default(tmp_path):
    """Killing the scanner from out here loses the report it was about to write.

    There used to be a derived subprocess timeout, and every outcome it produced was bad:
    a scanner killed mid-run has written nothing, so the scan is paid for in full and the
    findings are discarded. A real 20 minute ZAP budget became a 32 minute wait ending in
    "exceeded its 1920s timeout" and a silent fallback to the built-in crawler, which
    reported zero findings for an application that has plenty.

    The tools carry their own caps on their own argv where they have them, and where they
    do not - zap-full-scan's active phase - an unbounded scan that returns findings is the
    trade this package chooses over a capped one that returns none.
    """
    request = make_request(external_time_budget_s=600.0, profile=ScanProfile.ACTIVE)
    runner = RecordingRunner(report_body=fixture("zap_sample.json"))
    run_external(tool("zap-docker"), request, out_dir=tmp_path, runner=runner)

    assert runner.calls[0]["timeout"] is None


def test_a_caller_can_still_ask_for_a_ceiling(tmp_path):
    """Removing the default is not removing the capability."""
    runner = RecordingRunner(report_body=fixture("zap_sample.json"))
    run_external(tool("zap-docker"), make_request(profile=ScanProfile.ACTIVE),
                 out_dir=tmp_path, runner=runner, timeout_s=45.0)

    assert runner.calls[0]["timeout"] == 45.0
