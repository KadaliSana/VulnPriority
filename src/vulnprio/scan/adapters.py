"""External scanner adapters: drive a real scanner whenever one is installed.

ZAP and Nuclei are better web vulnerability scanners than anything worth writing inside
this project, and they are maintained by people who do nothing else. This framework's
contribution is what happens to findings *after* they exist - intelligence enrichment,
attacker modelling, monetary impact, attack-graph position, learned ranking - so the
scanner in :mod:`vulnprio.scan.crawler` is the fallback for a machine with nothing
installed, not the first choice.

This module finds the installed tools, ranks them, maps the requested scan profile onto
each tool's own flags, runs the chosen one, and hands its report to :mod:`vulnprio.ingest`.

Three rules govern everything here:

* **argv is a list, never a shell string.** Every command is assembled as a list of
  separate arguments and executed with ``shell=False``. No caller-supplied value is ever
  interpolated into a string a shell will parse, so there is no command-injection surface
  even if the target URL contains ``;``, backticks or ``$(...)``.
* **Authorisation gates the invocation, not just our own crawler.** An external scanner is
  far more capable than the built-in one, so :func:`build_argv` re-checks the attestation,
  the private-target policy and the scope before a single argument is assembled. A tool
  cannot be launched without ``authorized=True`` and a written note.
* **The profile is never quietly upgraded.** A tool that cannot honour a passive scan is
  not run for a passive request; it is skipped, with the reason recorded, and the next
  tool (or the built-in scanner) takes over. ZAP's baseline scan is the passive mapping and
  its full scan is the active one - they are different scripts, not a flag we might forget.

The subprocess runner is injectable, so the tests exercise ranking, argv construction,
timeouts, report ingestion and fallback without executing any binary.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from vulnprio.core.errors import ParseError, VulnprioError
from vulnprio.core.interfaces import ScannerParser
from vulnprio.core.models import Scan
from vulnprio.core.registry import SCANNER_PARSERS
from vulnprio.ingest.generic import detect_parser

# Importing these registers them in SCANNER_PARSERS, which is what lets ingest_report read
# a Wapiti or Nikto report. Without the import the adapters would run those tools and then
# throw their output away.
from vulnprio.ingest.nikto import NiktoParser          # noqa: F401  (registration side effect)
from vulnprio.ingest.wapiti import WapitiParser        # noqa: F401  (registration side effect)
from vulnprio.scan.models import (
    ScanProfile,
    ScanRequest,
    ScannerEnvironment,
    ToolStatus,
)
from vulnprio.scan.safety import (
    is_http_url,
    require_authorization,
    require_target_allowed,
)

__all__ = [
    "ExternalToolError",
    "ExternalTool",
    "ToolSpec",
    "TOOL_SPECS",
    "TOOL_PREFERENCE",
    "PREFERENCE",
    "ZAP_DOCKER_IMAGE",
    "DOCKER_HOST_ALIAS",
    "container_target",
    "detect_external_tools",
    "describe_external_tools",
    "select_tools",
    "scanner_environment",
    "tool_version",
    "build_argv",
    "run_external",
    "ingest_report",
    "install_hint_for",
    "parser_for",
    "safe_argument",
]


class ExternalToolError(VulnprioError):
    """An external scanner could not be located, run, or understood."""


#: The official ZAP image. ``owasp/zap2docker-stable`` is the legacy name for the same
#: project and still works on older installations; this one is the currently published
#: repository, so it is what a clean machine will be able to pull.
ZAP_DOCKER_IMAGE = "ghcr.io/zaproxy/zaproxy:stable"

_VERSION = re.compile(r"(\d+\.\d+(?:\.\d+)*)")


def safe_argument(value: str, *, what: str = "argument") -> str:
    """Refuse a value that could be mistaken for an option.

    Command injection is already impossible here because argv is a list and no shell is
    involved. This guards the remaining hazard: *argument* injection, where a value
    beginning with ``-`` is read by the tool as a flag rather than as data.
    """
    text = str(value)
    if not text or text.strip() != text:
        raise ExternalToolError(f"refusing an empty or padded {what}: {value!r}")
    if text.startswith("-"):
        raise ExternalToolError(f"refusing a {what} that looks like an option: {value!r}")
    if "\x00" in text or "\n" in text or "\r" in text:
        raise ExternalToolError(f"refusing a {what} containing control characters: {value!r}")
    return text


@dataclass(frozen=True)
class ExternalTool:
    """A scanner found on ``PATH``."""

    name: str
    executable: Path
    report_suffix: str = ".json"
    version: str | None = None
    image: str | None = None                 # container image, for containerised tools

    @property
    def spec(self) -> "ToolSpec":
        return TOOL_SPECS[self.name]

    def supports(self, profile: ScanProfile) -> bool:
        return profile in TOOL_SPECS[self.name].profiles


@dataclass(frozen=True)
class ToolSpec:
    """How to invoke one external scanner safely, and what it can honestly do.

    ``build`` returns a *sequence of argv lists*: most tools are one command, but ZAP's
    Python CLI needs a scan step followed by a report step, and modelling that honestly is
    better than pretending a single command produces a file it does not.
    """

    name: str
    candidates: tuple[str, ...]
    report_suffix: str
    profiles: frozenset[ScanProfile]
    build: Callable[["ExternalTool", ScanRequest, Path], list[list[str]]]
    version_argv: Callable[["ExternalTool"], list[str]]
    parser: str
    summary: str
    install_hint: str
    #: True when ``version_argv`` reports the *scanner's* version. False for containerised
    #: tools, where it reports the container runtime and must not be written into
    #: ``Scan.scanner_version``.
    version_is_scanner: bool = True
    image: str | None = None


# ---------------------------------------------------------------------------
# argv builders, one per tool, profile mapped onto the tool's own flags
# ---------------------------------------------------------------------------


#: The name a container uses to reach a service listening on its own host. Docker Desktop
#: provides it on every platform; on plain Linux the ``host-gateway`` alias below supplies
#: it. Needed because ``localhost`` inside a container is the *container*, which is the
#: one thing a loopback target is guaranteed not to be.
DOCKER_HOST_ALIAS = "host.docker.internal"


def container_target(target_url: str) -> tuple[str, str]:
    """Rewrite a loopback target into something a container can actually reach.

    Returns the URL to pass the containerised scanner and a note explaining the change, or
    the URL unaltered and an empty note. ``http://localhost:3000`` handed to a scanner
    running inside Docker resolves to port 3000 *of the scanner's own container*, where
    nothing is listening; the scan then completes in seconds having found nothing, which
    reads as a clean target rather than as a scan that never happened. This is the single
    most common way a containerised scan silently produces a false negative, so it is
    corrected here and the correction is reported rather than done quietly.

    Only the loopback names are rewritten. A private LAN address such as
    ``192.168.1.10`` is reachable from the container's bridge network as it stands, and
    rewriting it would point the scan at the wrong machine.
    """
    parts = urlsplit(target_url)
    host = (parts.hostname or "").lower()
    if host not in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}:
        return target_url, ""
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{DOCKER_HOST_ALIAS}{port}"
    rewritten = urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))
    note = (
        f"the scanner ran in a container, where the loopback name {host!r} would have meant "
        f"the container itself, so it was pointed at {DOCKER_HOST_ALIAS}{port} instead - the "
        f"same service. Every URL in this report therefore names {DOCKER_HOST_ALIAS} where "
        f"you would say {host}"
    )
    return rewritten, note


def _zap_docker_argv(tool: ExternalTool, request: ScanRequest, report: Path) -> list[list[str]]:
    """ZAP through the official image: baseline for passive, full scan for active.

    ``zap-baseline.py`` spiders the target and runs the passive rules only - it sends no
    attacks. ``zap-full-scan.py`` is the same crawl followed by the active scanner. They
    are separate scripts, so a passive request cannot be turned into an active scan by a
    forgotten flag. The report directory is bind-mounted at ZAP's working directory and
    ``-J`` writes the JSON report into it, which :class:`~vulnprio.ingest.zap.ZapParser`
    then reads.

    A loopback target is rewritten by :func:`container_target` and the container is given
    the ``host-gateway`` alias, without which the scan would run against the empty inside
    of its own container.
    """
    rewritten, _ = container_target(request.target_url)
    target = safe_argument(rewritten, what="target URL")
    workdir = Path(report).parent.resolve()
    script = "zap-baseline.py" if request.profile == ScanProfile.PASSIVE else "zap-full-scan.py"
    # The external scanner's own budget, not the built-in crawler's: see
    # ``ScanRequest.external_time_budget_s`` for why conflating them made every run
    # of the same target return different findings.
    minutes = max(1, int(round(request.external_time_budget_s / 60.0)))
    image = safe_argument(tool.image or ZAP_DOCKER_IMAGE, what="container image")
    return [
        [
            str(tool.executable),
            "run",
            "--rm",
            # Harmless when Docker already provides the name (Docker Desktop does); the
            # difference between a working scan and a silent no-op on plain Linux.
            "--add-host", f"{DOCKER_HOST_ALIAS}:host-gateway",
            "-v", f"{workdir}:/zap/wrk/:rw",
            image,
            script,
            "-t", target,
            "-J", report.name,
            "-I",                       # report warnings without failing the run
            "-T", str(minutes),         # hard cap on the whole scan, in minutes
            "-z", f"-config spider.maxDepth={request.max_depth}",
        ]
    ]


def _zap_sh_argv(tool: ExternalTool, request: ScanRequest, report: Path) -> list[list[str]]:
    """Native ZAP's headless quick-scan: spider plus active scan, so ACTIVE only."""
    target = safe_argument(request.target_url, what="target URL")
    return [
        [
            str(tool.executable),
            "-cmd",
            "-quickurl", target,
            "-quickout", str(report),
            "-quickprogress",
        ]
    ]


def _zap_cli_argv(tool: ExternalTool, request: ScanRequest, report: Path) -> list[list[str]]:
    """``zap-cli``: scan, then export the report. Two commands, both argv lists."""
    target = safe_argument(request.target_url, what="target URL")
    scan = [str(tool.executable), "quick-scan", "--self-contained", "--spider", "--recursive", target]
    export = [str(tool.executable), "report", "-o", str(report), "-f", "json"]
    return [scan, export]


def _nuclei_argv(tool: ExternalTool, request: ScanRequest, report: Path) -> list[list[str]]:
    """Nuclei with destructive template classes excluded and the rate limit honoured.

    Registered as ACTIVE only, deliberately. Nuclei requests paths the application never
    linked to, which is precisely what the passive profile promises not to do; calling it
    passive because its templates are read-only would be a lie about the traffic the
    target will see.
    """
    target = safe_argument(request.target_url, what="target URL")
    return [
        [
            str(tool.executable),
            "-target", target,
            "-jsonl",
            "-output", str(report),
            "-silent",
            "-no-interactsh",
            "-disable-update-check",
            "-rate-limit", str(max(1, int(request.requests_per_second * 60))),
            "-timeout", str(max(1, int(request.timeout_s))),
            "-exclude-tags", "dos,fuzz,fuzzing,intrusive,brute-force",
            "-header", f"User-Agent: {request.user_agent}",
        ]
    ]


def _wapiti_argv(tool: ExternalTool, request: ScanRequest, report: Path) -> list[list[str]]:
    """Wapiti restricted to the authorised folder; ``-m ""`` disables every attack module.

    With no modules loaded Wapiti crawls and analyses without attacking, which is a
    faithful passive scan, so this tool honestly supports both profiles.
    """
    target = safe_argument(request.target_url, what="target URL")
    argv = [
        str(tool.executable),
        "-u", target,
        "-f", "json",
        "-o", str(report),
        "--scope", "folder",
        "--max-links-per-page", str(request.max_pages),
        "--depth", str(request.max_depth),
        "--timeout", str(int(request.timeout_s)),
        "--flush-session",
    ]
    if request.profile == ScanProfile.PASSIVE:
        argv += ["-m", ""]          # no attack modules: crawl and passive analysis only
    return [argv]


def _nikto_argv(tool: ExternalTool, request: ScanRequest, report: Path) -> list[list[str]]:
    """Nikto against the authorised host, bounded by its own max-time.

    ACTIVE only: Nikto's entire method is requesting thousands of paths nobody linked to.
    """
    target = safe_argument(request.target_url, what="target URL")
    return [
        [
            str(tool.executable),
            "-h", target,
            "-Format", "json",
            "-output", str(report),
            "-maxtime", str(max(1, int(request.external_time_budget_s))),
            "-nointeractive",
            "-useragent", request.user_agent,
        ]
    ]


# ---------------------------------------------------------------------------
# The tool table
# ---------------------------------------------------------------------------


TOOL_SPECS: dict[str, ToolSpec] = {
    "zap-docker": ToolSpec(
        name="zap-docker",
        candidates=("docker",),
        report_suffix=".json",
        profiles=frozenset({ScanProfile.PASSIVE, ScanProfile.ACTIVE}),
        build=_zap_docker_argv,
        version_argv=lambda tool: [str(tool.executable), "--version"],
        parser="zap",
        summary=(
            "OWASP ZAP via the official container image: a real spider plus either the "
            "passive rules (baseline) or the active scanner (full scan)."
        ),
        install_hint=(
            "Install Docker, then pull the image: docker pull ghcr.io/zaproxy/zaproxy:stable "
            "(the legacy name owasp/zap2docker-stable also works on older installs)."
        ),
        version_is_scanner=False,
        image=ZAP_DOCKER_IMAGE,
    ),
    "zap.sh": ToolSpec(
        name="zap.sh",
        candidates=("zap.sh", "zap.bat"),
        report_suffix=".json",
        profiles=frozenset({ScanProfile.ACTIVE}),
        build=_zap_sh_argv,
        version_argv=lambda tool: [str(tool.executable), "-version"],
        parser="zap",
        summary=(
            "Natively installed OWASP ZAP. Its headless quick-scan is a spider followed by "
            "an active scan, so it cannot serve a passive request."
        ),
        install_hint="Download OWASP ZAP from https://www.zaproxy.org/download/ and put zap.sh on PATH.",
    ),
    "nuclei": ToolSpec(
        name="nuclei",
        candidates=("nuclei",),
        report_suffix=".jsonl",
        profiles=frozenset({ScanProfile.ACTIVE}),
        build=_nuclei_argv,
        version_argv=lambda tool: [str(tool.executable), "-version"],
        parser="nuclei",
        summary=(
            "Template-driven and very fast, with excellent coverage of known CVEs and "
            "misconfigurations. It does not crawl, so it yields fewer endpoints than ZAP."
        ),
        install_hint=(
            "go install github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest, or take a "
            "release binary from https://github.com/projectdiscovery/nuclei/releases"
        ),
    ),
    "zap-cli": ToolSpec(
        name="zap-cli",
        candidates=("zap-cli",),
        report_suffix=".json",
        profiles=frozenset({ScanProfile.ACTIVE}),
        build=_zap_cli_argv,
        version_argv=lambda tool: [str(tool.executable), "--version"],
        parser="zap",
        summary="A thin client for a running ZAP daemon; its quick-scan is an active scan.",
        install_hint="pip install zapcli, and have a ZAP daemon for it to drive.",
    ),
    "wapiti": ToolSpec(
        name="wapiti",
        candidates=("wapiti",),
        report_suffix=".json",
        profiles=frozenset({ScanProfile.PASSIVE, ScanProfile.ACTIVE}),
        build=_wapiti_argv,
        version_argv=lambda tool: [str(tool.executable), "--version"],
        parser="wapiti",
        summary=(
            "A real crawler with a JSON report. With its attack modules disabled it is a "
            "faithful passive scan; with them enabled it tests for injection classes."
        ),
        install_hint="pip install wapiti3",
    ),
    "nikto": ToolSpec(
        name="nikto",
        candidates=("nikto", "nikto.pl"),
        report_suffix=".json",
        profiles=frozenset({ScanProfile.ACTIVE}),
        build=_nikto_argv,
        version_argv=lambda tool: [str(tool.executable), "-Version"],
        parser="nikto",
        summary=(
            "Server misconfiguration and interesting-file checks against one host. It "
            "assigns no severity of its own, which is what the enrichment layer is for."
        ),
        install_hint="apt install nikto, or clone https://github.com/sullo/nikto",
    ),
}


#: Preference order when several tools are installed, best first. The reasoning:
#:
#: 1. ``zap-docker`` - ZAP does a genuine crawl and its baseline/full-scan split maps
#:    cleanly onto both profiles, so it is the only tool that can serve a passive request
#:    *and* an active one with the same engine. Running it from the pinned official image
#:    is also how most people actually have ZAP, and vulnprio has a first-class ZAP parser.
#: 2. ``zap.sh`` - the same engine natively, with no image pull, but only an active mode.
#: 3. ``nuclei`` - fastest and best at known CVEs, but it does not crawl, so the endpoint
#:    structure Components A and C depend on is much thinner.
#: 4. ``zap-cli`` - the same engine again, through more moving parts (a separate daemon).
#: 5. ``wapiti`` - a real crawler and a good passive mode, but a smaller rule set.
#: 6. ``nikto`` - server misconfiguration only, no severity, weakest endpoint evidence.
#:
#: A caller can override this wholesale by passing ``order=`` to :func:`select_tools` or
#: :func:`vulnprio.scan.runner.assess_target`.
TOOL_PREFERENCE: tuple[str, ...] = (
    "zap-docker",
    "zap.sh",
    "nuclei",
    "zap-cli",
    "wapiti",
    "nikto",
)

#: Backwards-compatible alias.
PREFERENCE = TOOL_PREFERENCE


# ---------------------------------------------------------------------------
# Detection, versions and environment reporting
# ---------------------------------------------------------------------------


def detect_external_tools(
    *,
    path: str | None = None,
    which: Callable[..., str | None] | None = None,
    order: Sequence[str] | None = None,
) -> tuple[ExternalTool, ...]:
    """Find the supported scanners on ``PATH``, in preference order.

    Detection only looks at ``PATH``; it runs nothing. Asking every scanner for its
    version costs a subprocess each, so that is :func:`tool_version`'s job and it is opt-in.

    ``path`` and ``which`` are injectable so tests can present a synthetic ``PATH``
    without installing anything.
    """
    lookup = which or shutil.which
    found: list[ExternalTool] = []
    for name in (order or TOOL_PREFERENCE):
        spec = TOOL_SPECS.get(name)
        if spec is None:
            continue
        for candidate in spec.candidates:
            located = lookup(candidate, path=path) if path is not None else lookup(candidate)
            if located:
                found.append(
                    ExternalTool(
                        name=name,
                        executable=Path(located),
                        report_suffix=spec.report_suffix,
                        image=spec.image,
                    )
                )
                break
    return tuple(found)


def tool_version(
    tool: ExternalTool,
    *,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    timeout_s: float = 15.0,
) -> str | None:
    """Ask a tool for its version. Returns ``None`` when it cannot be determined.

    Never raises: an unreadable version is a cosmetic problem, and a scan must not fail
    because a banner changed format.
    """
    spec = TOOL_SPECS.get(tool.name)
    if spec is None:
        return None
    execute = runner or subprocess.run
    try:
        completed = execute(
            spec.version_argv(tool),
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception:       # a missing binary, a timeout, a permission error - all cosmetic here
        return None
    blob = f"{getattr(completed, 'stdout', '') or ''}\n{getattr(completed, 'stderr', '') or ''}"
    match = _VERSION.search(blob)
    return match.group(1) if match else None


def select_tools(
    wanted: ScanRequest | ScanProfile,
    tools: Sequence[ExternalTool] | None = None,
    *,
    order: Sequence[str] | None = None,
    which: Callable[..., str | None] | None = None,
) -> tuple[tuple[ExternalTool, ...], tuple[str, ...]]:
    """Rank the installed tools that can honour a request's (or a profile's) mode.

    Returns ``(usable, skipped_reasons)``. A tool that cannot serve the requested profile
    is skipped with an explanation rather than run in a mode the operator did not ask for.
    """
    profile = wanted.profile if isinstance(wanted, ScanRequest) else wanted
    available = tuple(tools) if tools is not None else detect_external_tools(which=which, order=order)
    ranking = list(order or TOOL_PREFERENCE)

    def rank(tool: ExternalTool) -> int:
        return ranking.index(tool.name) if tool.name in ranking else len(ranking)

    usable: list[ExternalTool] = []
    skipped: list[str] = []
    for tool in sorted(available, key=rank):
        spec = TOOL_SPECS.get(tool.name)
        if spec is None:
            skipped.append(f"{tool.name}: not a supported tool")
            continue
        if profile not in spec.profiles:
            supported = ", ".join(sorted(item.value for item in spec.profiles))
            skipped.append(
                f"{tool.name} is installed but only supports the {supported} profile, and this "
                f"scan asked for {profile.value}; it was not run"
            )
            continue
        usable.append(tool)
    return tuple(usable), tuple(skipped)


def scanner_environment(
    profile: ScanProfile = ScanProfile.PASSIVE,
    *,
    which: Callable[..., str | None] | None = None,
    path: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    include_versions: bool = True,
    order: Sequence[str] | None = None,
) -> ScannerEnvironment:
    """What is installed, what it can do, and how to install what is missing.

    This is what the web application should call to show the operator their options. Every
    known tool appears, installed or not, with an accurate installation hint for the ones
    that are not - a user on a clean machine should be told how to get ZAP, not silently
    handed a weaker scan.
    """
    installed = {
        tool.name: tool
        for tool in detect_external_tools(which=which, path=path, order=order)
    }
    if include_versions:
        # Only report a version that is actually the SCANNER's. For a containerised tool
        # `version_argv` asks the container runtime, so `docker --version` answers 29.7.2 --
        # Docker's version, not ZAP's. Surfacing it puts "scanned by zap 29.7.2" in a report,
        # which is simply false, and the real ZAP version is only knowable once the image has
        # run. `version_is_scanner` already records the distinction; honour it here.
        installed = {
            name: ExternalTool(
                name=tool.name,
                executable=tool.executable,
                report_suffix=tool.report_suffix,
                version=(
                    tool_version(tool, runner=runner)
                    if TOOL_SPECS[tool.name].version_is_scanner
                    else None
                ),
                image=tool.image,
            )
            for name, tool in installed.items()
        }

    ranking = list(order or TOOL_PREFERENCE)
    statuses: list[ToolStatus] = []
    for index, name in enumerate(ranking):
        spec = TOOL_SPECS[name]
        tool = installed.get(name)
        statuses.append(
            ToolStatus(
                name=name,
                installed=tool is not None,
                executable=str(tool.executable) if tool is not None else None,
                version=tool.version if tool is not None else None,
                image=spec.image,
                profiles=tuple(sorted(spec.profiles, key=lambda item: item.value)),
                supports_requested_profile=profile in spec.profiles,
                summary=spec.summary,
                install_hint=spec.install_hint,
                report_parser=spec.parser,
                preference_rank=index,
            )
        )

    usable, skipped = select_tools(profile, tools=tuple(installed.values()), order=order)
    preferred = usable[0].name if usable else None

    if preferred is not None:
        notice = (
            f"{preferred} will run this {profile.value} scan. "
            f"{TOOL_SPECS[preferred].summary}"
        )
    elif installed:
        notice = (
            f"No installed scanner supports the {profile.value} profile "
            f"({'; '.join(skipped)}). The built-in vulnprio scanner will run instead. "
            + install_hint_for(profile)
        )
    else:
        notice = (
            "No external scanner was found on PATH, so the built-in vulnprio scanner will "
            "run. It is deliberately conservative and finds less than a dedicated scanner. "
            + install_hint_for(profile)
        )

    return ScannerEnvironment(
        profile=profile,
        tools=tuple(statuses),
        preferred=preferred,
        builtin_fallback=preferred is None,
        skipped=tuple(skipped),
        notice=notice,
    )


def install_hint_for(profile: ScanProfile) -> str:
    """How to install the best tool that could have served this profile.

    A user on a clean machine should be told how to get ZAP, not silently handed a weaker
    scan, so this string is carried in the environment notice and in the outcome of every
    scan that fell back to the built-in scanner.
    """
    for name in TOOL_PREFERENCE:
        spec = TOOL_SPECS[name]
        if profile in spec.profiles:
            return f"To get a full scan, install {name}: {spec.install_hint}"
    return ""   # pragma: no cover - every profile has at least one capable tool


def describe_external_tools(tools: Sequence[ExternalTool] | None = None) -> str:
    """A sentence the CLI or web UI can show the operator verbatim."""
    found = tuple(tools) if tools is not None else detect_external_tools()
    if not found:
        return (
            "No external scanner (ZAP, nuclei, wapiti, nikto, or ZAP via Docker) was found "
            "on PATH. Using the built-in vulnprio scanner instead. "
            + install_hint_for(ScanProfile.PASSIVE)
        )
    listed = ", ".join(
        f"{tool.name}{f' {tool.version}' if tool.version else ''} ({tool.executable})"
        for tool in found
    )
    return f"External scanners available: {listed}."


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


def build_argv(tool: ExternalTool, request: ScanRequest, report_path: Path) -> list[list[str]]:
    """Assemble the command(s) for a tool as argv lists.

    Authorisation, target policy and profile support are all re-checked here, because
    handing an unauthorised target to a third-party scanner is more serious than scanning
    it ourselves, not less: an external tool will not respect this package's limits.
    """
    require_authorization(request)
    require_target_allowed(request)
    if not is_http_url(request.target_url):
        raise ExternalToolError(f"not an http(s) target: {request.target_url!r}")
    spec = TOOL_SPECS.get(tool.name)
    if spec is None:
        raise ExternalToolError(f"unsupported external tool: {tool.name!r}")
    if request.profile not in spec.profiles:
        supported = ", ".join(sorted(profile.value for profile in spec.profiles))
        raise ExternalToolError(
            f"{tool.name} supports only the {supported} profile and this scan asked for "
            f"{request.profile.value}; refusing to run it in a mode the operator did not choose"
        )
    steps = spec.build(tool, request, Path(report_path))
    for argv in steps:
        if not argv or not isinstance(argv, list):
            raise ExternalToolError(f"{tool.name}: argv must be a non-empty list")
        for item in argv:
            if not isinstance(item, str):
                raise ExternalToolError(f"{tool.name}: argv entries must be strings, got {item!r}")
    return steps


def run_external(
    tool: ExternalTool,
    request: ScanRequest,
    *,
    timeout_s: float | None = None,
    out_dir: str | Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Run an external scanner into a temporary report file and return its path.

    The command is executed with ``shell=False`` and a hard timeout derived from the
    request's time budget. Standard output is captured rather than inherited, so a tool
    that decides to be interactive cannot block the process.
    """
    directory = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="vulnprio-scan-"))
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / f"{tool.name.replace('.', '_')}-report{tool.report_suffix}"
    steps = build_argv(tool, request, report)

    execute = runner or subprocess.run
    # Our own kill switch has to sit *outside* the tool's cap, not inside it. It used to
    # be twice the built-in crawler's budget - five minutes - so a ZAP scan told to take
    # twenty was killed at five, part-way through, before it had written its report. The
    # headroom covers container startup and the image pull on a cold machine.
    budget = float(
        timeout_s
        if timeout_s is not None
        else max(request.external_time_budget_s * 1.5 + 120.0, 120.0)
    )
    for argv in steps:
        try:
            completed = execute(
                argv,
                shell=False,           # never a shell: argv stays a list of separate arguments
                capture_output=True,
                text=True,
                timeout=budget,
                cwd=str(directory),
                env=dict(env) if env is not None else None,
                check=False,
            )
        except FileNotFoundError as error:
            raise ExternalToolError(f"{tool.name} is not executable at {tool.executable}: {error}") from error
        except subprocess.TimeoutExpired as error:
            raise ExternalToolError(f"{tool.name} exceeded its {budget:.0f}s timeout") from error
        returncode = getattr(completed, "returncode", 0)
        if returncode not in (0, 1, 2):   # scanners use small non-zero codes for "findings present"
            stderr = (getattr(completed, "stderr", "") or "")[:500]
            raise ExternalToolError(f"{tool.name} exited with status {returncode}: {stderr}")

    if not report.exists() or report.stat().st_size == 0:
        raise ExternalToolError(
            f"{tool.name} produced no report at {report}. "
            "Falling back to the built-in vulnprio scanner is the safe response."
        )
    return report


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest_report(path: str | Path, app_id: str | None = None) -> Scan:
    """Parse any supported scanner report into a :class:`~vulnprio.core.models.Scan`.

    Format detection goes through :func:`vulnprio.ingest.generic.detect_parser` first, so
    the canonical, ZAP, Burp and Nuclei formats behave exactly as they do everywhere else.
    Wapiti and Nikto are then sniffed from the parser registry, because
    ``detect_parser``'s built-in order lives in ``ingest/generic.py`` and this package
    does not own that file.
    """
    source = Path(path)
    try:
        return detect_parser(source).parse(source, app_id=app_id)
    except ParseError as first_error:
        parser = _sniff_registered(source)
        if parser is None:
            raise ExternalToolError(
                f"no vulnprio parser recognises {source}: {first_error}"
            ) from first_error
        try:
            return parser.parse(source, app_id=app_id)
        except ParseError as error:
            raise ExternalToolError(f"{parser.name} could not read {source}: {error}") from error


def _sniff_registered(path: Path) -> ScannerParser | None:
    """Ask every registered parser, newest registrations included, in a stable order."""
    for name in sorted(SCANNER_PARSERS):
        parser_class = SCANNER_PARSERS[name]
        try:
            parser = parser_class()
            if parser.sniff(path):
                return parser
        except Exception:       # a parser that crashes while sniffing must not block the rest
            continue
    return None


def parser_for(tool_name: str) -> str | None:
    """Name of the vulnprio parser that reads a tool's report."""
    spec = TOOL_SPECS.get(tool_name)
    return spec.parser if spec is not None else None
