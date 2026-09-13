"""The interactive application: routes, and the analysis job behind them.

The static export in :mod:`vulnpriority.web.exporter` answers "what did that run find?". This
module answers "run one now": the page it serves collects a scanner report or an authorised
target, starts the real pipeline on a background thread, streams its progress, and hands
back the same :class:`~vulnpriority.web.schema.DashboardData` payload the static site consumes.
The front end is therefore one page with two supplies of data, and every existing view
works identically whichever supply filled it.

The pipeline runs on a worker thread owned by :mod:`vulnpriority.web.jobs`, never on the event
loop, because an analysis takes seconds to minutes and an async framework that blocks for
that long is a single-user framework. The browser is told about progress two ways: a Server
Sent Events stream at ``/api/jobs/{id}/events``, and plain polling of ``/api/jobs/{id}``
for anything that cannot hold a stream open. The front end prefers the stream and falls
back on its own.

.. warning::

   **This is a local tool, not a service.** It binds ``127.0.0.1`` by default and should
   stay there. It has no user accounts, no TLS, no rate limiting and no audit log, and it
   will start a scan of whatever target a request names. Exposing it on ``0.0.0.0``, or
   behind a proxy, hands anyone who can reach the port the ability to run scans from this
   machine, read any run it has produced, and consume its CPU. If you need a shared
   instance, put a real service in front of it and do not reuse this code for it.

What protection there is, is proportionate to that threat model:

* the default bind address is the loopback interface;
* the ``Host`` header must name this machine. Binding loopback is not on its own enough:
  a page on any site the operator visits can point its own hostname at ``127.0.0.1`` and
  the browser will then treat this server's responses as same-origin, which is how a
  foreign origin reads a token out of the page it was never supposed to see. Refusing a
  ``Host`` we do not answer to is what makes the token below a control rather than a
  formality. A wildcard bind switches the check off, because an operator who asked for
  every interface has chosen exposure and the set of names that reach them is not
  knowable from in here;
* a per-process random token, minted at startup and written into the page it serves, is
  required on every mutating endpoint, so a hostile page in another tab cannot silently
  start a scan on the operator's behalf. It is accepted in a header and nowhere else: a
  credential in a query string ends up in request logs, browser history and every proxy
  in between;
* request bodies are capped before they are read, and uploads are capped while streaming
  rather than after landing in memory;
* an authorisation flag *and* a written note are required before a scan request is even
  turned into a job, and they are checked at the door - an unauthorised request never
  reaches :mod:`vulnpriority.scan`;
* no stack trace is ever written to a response; failures are logged server-side and the
  client is told what failed, not where.

Run it with ``python -m vulnpriority.web.app`` (add ``--demo`` to preload the worked example),
or point any ASGI server at ``vulnpriority.web.app:app``. Interactive API documentation is at
``/docs``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import html as html_module
import importlib
import inspect
import json
import logging
import secrets
import tempfile
import threading
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping, Sequence
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from vulnpriority.core.money import format_money, format_money_compact
from vulnpriority.web.jobs import Job, JobContext, JobStatus, JobStore
from vulnpriority.web.schema import (
    DASHBOARD_SCHEMA_VERSION,
    INTERACTIVE_API_VERSION,
    AnalyzeComponents,
    AnalyzeRequest,
    AnalyzeUpload,
    DashboardData,
    WebAsOf,
    WebCapabilities,
    WebConfigOptions,
    WebDefaults,
    WebError,
    WebHealth,
    WebJobCreated,
    WebJobState,
    WebPreset,
    WebReportInspection,
    WebScanProfile,
    WebScannerEnvironment,
    WebScannerTool,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "MAX_BODY_BYTES",
    "ACCEPTED_REPORT_SUFFIXES",
    "TOKEN_HEADER",
    "ApiError",
    "AnalysisUnavailable",
    "AppSettings",
    "WebApplication",
    "create_app",
    "app",
    "serve",
    "run_analysis",
    "build_report_document",
    "probe_capabilities",
    "config_options",
    "main",
]

LOGGER = logging.getLogger("vulnpriority.web.app")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: Request bodies above this are refused. A base64-encoded scanner report is about a third
#: larger than the file, so this admits roughly a 6 MB report, which is a very large one.
MAX_BODY_BYTES = 8 * 1024 * 1024

#: Report formats the upload path will try to parse. The parsers sniff content rather than
#: trusting the extension; this list exists so the page can refuse the obvious mistakes
#: (a PDF, an executable) before spending a round trip on them.
ACCEPTED_REPORT_SUFFIXES: tuple[str, ...] = (".json", ".xml", ".jsonl")

ASSET_DIR = Path(__file__).resolve().parent / "assets"

#: The marker in ``index.html`` that the session token replaces. It is a complete, valid
#: statement on its own, so the file stays usable as a static asset: opened from ``file://``
#: the page simply reads "there is no session", which is true.
SESSION_MARKER = "window.VULNPRIORITY_SESSION = null;"

TOKEN_HEADER = "X-VulnPriority-Token"

#: How often the event stream re-reads the job. Fast enough to feel live, slow enough that
#: a hundred polls do not contend on the store's lock.
STREAM_INTERVAL_S = 0.25

#: What the two scanning postures do, in the words an operator has to be able to agree to.
#: :mod:`vulnpriority.scan` is the authority; these are used when it does not describe itself,
#: and are marked non-authoritative in the payload so the page can say so.
FALLBACK_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "name": "passive",
        "label": "Passive: observe only",
        "description": (
            "Retrieves pages the way a browser would and reads what comes back. It reaches "
            "only URLs the target's own responses link to, and it never sends anything "
            "designed to change behaviour."
        ),
        "sends": [
            "GET and HEAD requests to URLs discovered from the target's own pages",
            "GET requests to endpoints named in the application's own JavaScript, so a "
            "single-page application is not scanned as its own loading screen",
            "Requests to paths robots.txt asks crawlers away from, which it reports rather "
            "than skipping: robots.txt is a crawler convention, not an access control, and "
            "skipping them would report them as clean",
            "An identifying User-Agent naming vulnpriority",
            "Nothing in a request body",
        ],
        "does_not_send": [
            "No injected payloads of any kind",
            "No POST, PUT, PATCH or DELETE",
            "No credential guessing, session fixation or brute force",
            "No load generation, and no traffic to hosts outside the target's scope",
        ],
    },
    {
        "name": "active",
        "label": "Active: bounded probes",
        "description": (
            "Everything the passive profile does, plus small bounded probe values sent to "
            "parameters the passive pass discovered, to see whether input reaches an "
            "interpreter. This is not read-only. Run it only against a system you are "
            "authorised to test, and expect it to appear in the target's logs."
        ),
        "sends": [
            "Everything the passive profile sends",
            "Benign, bounded probe values in parameters already observed on the target",
            "Form submissions to endpoints discovered during the passive pass",
        ],
        "does_not_send": [
            "No destructive payloads, and no attempt to write or delete application data",
            "No credential guessing or brute force",
            "No denial-of-service, flooding or sustained load",
            "Nothing to a host outside the scope derived from the target URL",
        ],
    },
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ApiError(HTTPException):
    """An HTTP failure carrying the machine-readable slug the page switches on."""

    def __init__(self, status_code: int, error: str, detail: str = "") -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.error = error


class AnalysisUnavailable(RuntimeError):
    """A requested capability is not present in this build. Reported, never hidden."""


def _error_body(error: str, detail: str) -> dict[str, str]:
    return WebError(error=error, detail=detail).model_dump()


def _json_error(status: int, error: str, detail: str = "") -> JSONResponse:
    return JSONResponse(status_code=status, content=_error_body(error, detail))


# ---------------------------------------------------------------------------
# Small reflection helpers, used only where a sibling package's shape is unsettled
# ---------------------------------------------------------------------------


def _accepts_keyword(func: Callable[..., Any], name: str) -> bool:
    """Whether ``func`` will accept ``name=`` - assume yes if the signature is unreadable."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - C callables and the like
        return True
    return any(
        parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _accepted_fields(target: Any) -> set[str]:
    """Field names a model, dataclass or callable will accept as keywords."""
    fields = getattr(target, "model_fields", None)
    if isinstance(fields, dict):
        return set(fields)
    dataclass_fields = getattr(target, "__dataclass_fields__", None)
    if isinstance(dataclass_fields, dict):
        return set(dataclass_fields)
    try:
        return {
            name
            for name, parameter in inspect.signature(target).parameters.items()
            if parameter.kind is not inspect.Parameter.VAR_KEYWORD
        }
    except (TypeError, ValueError):  # pragma: no cover
        return set()


# ---------------------------------------------------------------------------
# Capability probing
# ---------------------------------------------------------------------------


_CAPABILITY_CACHE: dict[str, WebCapabilities] = {}
_CAPABILITY_LOCK = threading.Lock()


def probe_capabilities(refresh: bool = False) -> WebCapabilities:
    """Which sibling packages import in this process.

    A real import, not a filesystem guess: a package that is present but broken is not a
    capability, and the page should say so with the error rather than offering a button
    that fails later. Cached, because it cannot change without a restart.
    """
    with _CAPABILITY_LOCK:
        if refresh:
            _CAPABILITY_CACHE.clear()
        cached = _CAPABILITY_CACHE.get("value")
        if cached is not None:
            return cached

        detail: dict[str, str] = {}
        present: dict[str, bool] = {}
        for key, module, attribute in (
            ("scan", "vulnpriority.scan", "assess_target"),
            ("report", "vulnpriority.report", "build_report"),
            ("novelty", "vulnpriority.novelty", "novelty_payload"),
        ):
            try:
                loaded = importlib.import_module(module)
            except Exception as error:  # noqa: BLE001 - a broken package is a missing capability
                present[key] = False
                detail[key] = f"{module} did not import: {type(error).__name__}: {error}"
                continue
            if not hasattr(loaded, attribute):
                present[key] = False
                detail[key] = f"{module} imported but does not define {attribute}()"
                continue
            present[key] = True
        capabilities = WebCapabilities(
            scan=present.get("scan", False),
            report=present.get("report", False),
            novelty=present.get("novelty", False),
            detail=detail,
        )
        _CAPABILITY_CACHE["value"] = capabilities
        return capabilities


# ---------------------------------------------------------------------------
# Configuration surfaced to the page
# ---------------------------------------------------------------------------


def _prettify(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def _load_presets(directory: Path, keep: Sequence[str]) -> list[WebPreset]:
    """Summarise every YAML preset in ``directory`` for a dropdown."""
    import yaml

    out: list[WebPreset] = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            LOGGER.warning("skipping unreadable preset %s: %s", path, error)
            continue
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or path.stem)
        out.append(
            WebPreset(
                name=name,
                label=_prettify(name),
                description=" ".join(str(payload.get("description", "")).split()),
                detail={key: payload[key] for key in keep if key in payload},
            )
        )
    return out


def _scan_profiles(capabilities: WebCapabilities) -> list[WebScanProfile]:
    """What each profile actually does, read out of :mod:`vulnpriority.scan` where possible.

    The operator is being asked to tick a box saying they are authorised to send this at a
    live system, so the description had better be the truth rather than prose someone wrote
    once. When the scanner is installed the numbers below - how many checks run, which
    probe kinds are permitted at all, which User-Agent identifies the traffic - are read
    from it, and the payload says so. The hand-written fallback is only for a build without
    the scanner, where there is nothing to describe authoritatively.
    """
    if not capabilities.scan:
        return [WebScanProfile(**entry, authoritative=False) for entry in FALLBACK_PROFILES]
    try:
        scan = importlib.import_module("vulnpriority.scan")

        # A package that publishes its own descriptions is believed over anything derived.
        published = getattr(scan, "PROFILE_DESCRIPTIONS", None)
        if isinstance(published, Mapping) and published:
            return [
                WebScanProfile(
                    name=str(name),
                    label=str((entry or {}).get("label") or _prettify(str(name))),
                    description=str((entry or {}).get("description", "")),
                    sends=[str(item) for item in (entry or {}).get("sends", []) or []],
                    does_not_send=[str(item) for item in (entry or {}).get("does_not_send", []) or []],
                    authoritative=True,
                )
                for name, entry in published.items()
            ]

        agent = str(getattr(scan, "DEFAULT_USER_AGENT", "")).split(" ")[0] or "vulnpriority-scan"
        probes = sorted(
            str(getattr(kind, "value", kind)).replace("_", " ")
            for kind in getattr(scan, "ALLOWED_PROBE_KINDS", ()) or ()
        )
        counts: dict[str, int] = {}
        for profile in getattr(scan, "ScanProfile", ()):
            counts[str(profile.value)] = len(scan.checks_for_profile(profile))

        passive_n = counts.get("passive", 0)
        active_n = counts.get("active", 0)
        return [
            WebScanProfile(
                name="passive",
                label="Passive: observe only",
                description=(
                    f"Crawls from the target URL the way a browser would and reads what comes "
                    f"back, then runs {passive_n} checks over what it saw. It sends nothing "
                    f"designed to change the application's behaviour."
                ),
                sends=[
                    "GET and HEAD requests to URLs the target's own pages link to",
                    "GET requests to endpoints named in the application's own JavaScript, "
                    "so a single-page application is not scanned as its own loading screen",
                    "Requests to paths robots.txt asks crawlers away from, which it reports "
                    "rather than skipping: robots.txt is a crawler convention, not an access "
                    "control, and skipping them would report them as clean",
                    f"A User-Agent identifying itself as {agent}",
                    "Nothing in a request body",
                ],
                does_not_send=[
                    "No probe payloads of any kind",
                    "No POST, PUT, PATCH or DELETE",
                    "No credential guessing or brute force",
                    "Nothing to a host outside the scope derived from the target URL",
                ],
                authoritative=True,
            ),
            WebScanProfile(
                name="active",
                label="Active: bounded probes",
                description=(
                    f"Everything the passive profile does, plus {max(0, active_n - passive_n)} "
                    f"further checks that send bounded probes to what the crawl found. This is "
                    f"not read-only: run it only against a system you are authorised to test, "
                    f"and expect it in the target's logs."
                ),
                sends=[
                    "Everything the passive profile sends",
                    "Only these probe kinds, which the scanner enforces: "
                    + (", ".join(probes) if probes else "none declared"),
                ],
                does_not_send=[
                    "No destructive payloads, and no attempt to write or delete application data",
                    "No credential guessing or brute force",
                    "No denial-of-service, flooding or sustained load",
                    "Nothing to a host outside the scope derived from the target URL",
                ],
                authoritative=True,
            ),
        ]
    except Exception as error:  # noqa: BLE001 - fall back to our own wording, marked as such
        LOGGER.debug("could not derive scan profile descriptions: %s", error)
        return [WebScanProfile(**entry, authoritative=False) for entry in FALLBACK_PROFILES]


def _default_pipeline_config() -> Any:
    """``configs/default.yaml`` when it is readable, else the built-in defaults."""
    from vulnpriority.core.config import PROJECT_ROOT, PipelineConfig, load_config

    path = PROJECT_ROOT / "configs" / "default.yaml"
    if not path.exists():
        return PipelineConfig()
    try:
        return load_config(path)
    except Exception as error:  # noqa: BLE001 - a broken config file must not break the page
        LOGGER.warning("configs/default.yaml is unreadable (%s); using built-in defaults", error)
        return PipelineConfig()


def config_options(capabilities: WebCapabilities | None = None) -> WebConfigOptions:
    """Everything the Analyze form needs to populate itself, read from ``configs/``."""
    from vulnpriority.core.config import PROJECT_ROOT

    capabilities = capabilities or probe_capabilities()
    attackers = _load_presets(
        PROJECT_ROOT / "configs" / "attacker_models",
        ("skill", "resources", "entry_privilege", "horizon_days", "max_chain_length"),
    )
    impacts = _load_presets(
        PROJECT_ROOT / "configs" / "impact_models",
        ("currency", "cost_per_record", "downtime_cost_per_hour",
         "regulatory_multiplier", "max_impact"),
    )
    config = _default_pipeline_config()
    defaults = WebDefaults(
        attacker=config.component_b.attacker_preset,
        impact_model=config.component_b.impact_preset,
        profile="passive",
        budget_hours=float(config.selection.budget_hours),
        components={
            "a": bool(config.component_a.enabled),
            "b": bool(config.component_b.enabled),
            "c": bool(config.component_c.enabled),
        },
        llm_backend=str(getattr(config.llm.backend, "value", config.llm.backend)),
        feed_mode=str(getattr(config.feeds.mode, "value", config.feeds.mode)),
    )
    return WebConfigOptions(
        attackers=attackers,
        impact_models=impacts,
        profiles=_scan_profiles(capabilities),
        defaults=defaults,
        accepted_report_suffixes=list(ACCEPTED_REPORT_SUFFIXES),
        max_upload_bytes=MAX_BODY_BYTES,
        capabilities=capabilities,
    )


#: The built-in crawler, named so the interface can offer it as a deliberate choice rather
#: than only as what happens when nothing better is installed.
BUILTIN_SCANNER = "builtin"


def scanner_environment(profile: str = "passive") -> Any:
    """What would actually scan, for one profile, on this machine right now.

    Which scanner ran is not an implementation detail: a ZAP scan and the built-in
    crawler find different things, and a reader of the report needs to be able to tell
    them apart. :mod:`vulnpriority.scan` already works all this out, including the awkward
    part - Nuclei and Nikto cannot serve a passive request, so asking for passive on a
    Nuclei-only machine silently falls back to the built-in scanner. This projects that
    decision, and its reasons, into something the page can render.
    """
    from vulnpriority.web.schema import WebScannerEnvironment, WebScannerTool

    try:
        import vulnpriority.scan as scan
    except Exception as error:  # noqa: BLE001
        return WebScannerEnvironment(
            profile=profile,
            available=False,
            notice=f"This build cannot assess a target: vulnpriority.scan did not import ({error}).",
        )
    try:
        wanted = scan.ScanProfile(profile)
        environment = scan.scanner_environment(wanted)
    except Exception as error:  # noqa: BLE001 - probing PATH can fail in odd ways
        LOGGER.warning("could not probe the scanner environment: %s", error)
        return WebScannerEnvironment(
            profile=profile,
            available=False,
            notice=f"The installed scanners could not be probed: {type(error).__name__}: {error}",
        )

    preferred = environment.preferred or ""
    tools = [
        WebScannerTool(
            name=tool.name,
            installed=bool(tool.installed),
            version=tool.version or "",
            executable=tool.executable or "",
            image=tool.image or "",
            profiles=[str(getattr(p, "value", p)) for p in tool.profiles],
            supports_requested_profile=bool(tool.supports_requested_profile),
            summary=tool.summary,
            install_hint=tool.install_hint,
            preference_rank=int(tool.preference_rank),
            would_run=bool(tool.name == preferred),
        )
        for tool in environment.tools
    ]
    return WebScannerEnvironment(
        profile=str(getattr(environment.profile, "value", environment.profile)),
        tools=tools,
        preferred=preferred,
        builtin_fallback=bool(environment.builtin_fallback),
        skipped=list(environment.skipped),
        notice=environment.notice,
        available=True,
    )


def _analysis_config(request: AnalyzeRequest) -> Any:
    """The run configuration this request describes."""
    from vulnpriority.core.models import ComponentFlags

    config = _default_pipeline_config()
    config = config.with_flags(
        ComponentFlags(a=request.components.a, b=request.components.b, c=request.components.c)
    )
    component_b = config.component_b.model_copy(
        update={
            "attacker_preset": request.attacker or config.component_b.attacker_preset,
            "impact_preset": request.impact_model or config.component_b.impact_preset,
        }
    )
    selection = config.selection.model_copy(update={"budget_hours": float(request.budget_hours)})
    return config.model_copy(update={"component_b": component_b, "selection": selection})


# ---------------------------------------------------------------------------
# The analysis itself
# ---------------------------------------------------------------------------


def _decode_upload(request: AnalyzeRequest, workdir: Path) -> Path:
    """Write a base64 report body to a temporary file, refusing anything obviously wrong."""
    upload = request.report
    if upload is None or not upload.content_base64:
        raise AnalysisUnavailable("Upload mode needs a report file.")
    name = _safe_report_name(upload.filename)
    try:
        content = base64.b64decode(upload.content_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise AnalysisUnavailable(f"The report was not valid base64: {error}") from error
    if not content.strip():
        raise AnalysisUnavailable("The report file was empty.")
    if len(content) > MAX_BODY_BYTES:
        raise AnalysisUnavailable(f"The decoded report exceeds {MAX_BODY_BYTES} bytes.")
    path = workdir / name
    path.write_bytes(content)
    return path


def _safe_report_name(filename: str | None) -> str:
    """Basename only, with a suffix this build can read. Raises :class:`ApiError` otherwise."""
    name = Path(str(filename or "report.json")).name or "report.json"
    suffix = Path(name).suffix.lower()
    if suffix not in ACCEPTED_REPORT_SUFFIXES:
        raise ApiError(
            400,
            "unsupported_report",
            f"{name!r} has suffix {suffix or '(none)'}; this build reads "
            f"{', '.join(ACCEPTED_REPORT_SUFFIXES)} scanner reports.",
        )
    return name


#: The scanner reports a phase, not a percentage. This is the order those phases occur in,
#: which is what turns them into a bar. An unrecognised phase simply does not move it.
SCAN_PHASE_ORDER: tuple[str, ...] = (
    "authorize", "robots", "crawl", "probe", "check", "fingerprint",
    "assemble", "external", "done",
)


def judge_report_freshness(scanned_at: datetime | None, *, live: bool) -> Any:
    """Whether live exploit intelligence can honestly describe this scan's moment.

    Delegated to :func:`vulnpriority.intel.agent.judge_as_of`, which owns the rule and the
    tolerance. When the intel package is absent there is no live search to be wrong about,
    so the answer is "not applicable" rather than a guess made here with a hardcoded age.
    """
    from vulnpriority.web.schema import WebAsOf

    scan_date = scanned_at.date().isoformat() if scanned_at is not None else ""
    if scanned_at is None or not live:
        return WebAsOf(
            status="not_applicable",
            as_of=scan_date,
            message="No live exploit intelligence was requested, so the scan's age does not "
                    "affect what this assessment knows.",
        )
    try:
        from vulnpriority.intel.agent import judge_as_of
        from vulnpriority.intel.models import IntelConfig
    except Exception as error:  # noqa: BLE001 - no live search in this build
        LOGGER.debug("vulnpriority.intel unavailable for an as-of judgement: %s", error)
        return WebAsOf(
            status="not_applicable",
            as_of=scan_date,
            message="This build cannot search the web for exploit intelligence, so "
                    "everything it knows is dated with the scan itself.",
        )

    config = IntelConfig()
    verdict = judge_as_of(scanned_at.date(), datetime.now().date(), config, live=True)
    return WebAsOf(
        status=str(getattr(verdict.status, "value", verdict.status)),
        remedy=str(getattr(verdict.remedy, "value", verdict.remedy)),
        search_allowed=bool(verdict.search_allowed),
        evidence_is_current=bool(verdict.evidence_is_current),
        as_of=verdict.as_of.isoformat() if verdict.as_of else scanned_at.date().isoformat(),
        evaluated_at=verdict.evaluated_at.isoformat() if verdict.evaluated_at else "",
        age_days=int(verdict.age_days),
        max_age_days=int(verdict.max_age_days),
        message=str(verdict.message),
    )


def inspect_report(path: Path, *, live_intel: bool) -> Any:
    """Parse a report enough to describe it, without running anything on it.

    The Analyze view needs the scan's date and host before it can offer the right choice
    between "run this anyway without live intelligence" and "assess the target now". Doing
    that here rather than in the browser keeps date arithmetic and scanner sniffing in one
    place, and means the page never has to parse a report itself.
    """
    from vulnpriority.ingest.generic import detect_parser
    from vulnpriority.web.schema import WebReportInspection

    try:
        scan = detect_parser(path).parse(path)
    except Exception as error:  # noqa: BLE001
        raise ApiError(
            400, "unreadable_report",
            f"No scanner parser recognised {path.name}: {type(error).__name__}: {error}",
        ) from error

    host = scan.hosts[0] if scan.hosts else ""
    if not host:
        for endpoint in scan.endpoints:
            if endpoint.host:
                host = endpoint.host
                break
    suggested = ""
    if host:
        sample = next((e.url for e in scan.endpoints if e.url.startswith("http")), "")
        scheme = "http" if sample.startswith("http://") else "https"
        suggested = f"{scheme}://{host}/"

    return WebReportInspection(
        app_name=scan.app_name or scan.app_id,
        scanner=scan.scanner_name,
        host=host,
        scanned_at=scan.scanned_at.isoformat(),
        n_findings=len(scan.findings),
        n_endpoints=len(scan.endpoints),
        suggested_target_url=suggested,
        as_of=judge_report_freshness(scan.scanned_at, live=live_intel),
    )


def _scan_progress(context: JobContext, low: float, high: float) -> Callable[..., None]:
    """An ``on_progress`` for :func:`vulnpriority.scan.assess_target`.

    The scanner's contract is ``Callable[[ScanProgress], None]`` with a ``phase``,
    ``pages_fetched``, ``findings``, ``elapsed_s`` and ``message``; this reads that, and
    stays tolerant of positional strings, a bare fraction or a mapping so that a change to
    the callback's shape degrades to a less precise bar rather than to a crash mid-scan.

    It is also the point at which a cancelled scan stops: it raises out of the callback,
    which unwinds the scan rather than letting it run on unattended.
    """

    def on_progress(*args: Any, **kwargs: Any) -> None:
        context.raise_if_cancelled()
        phase: Any = kwargs.get("phase") or kwargs.get("stage")
        message: Any = kwargs.get("message") or kwargs.get("detail")
        fraction: Any = kwargs.get("progress", kwargs.get("fraction"))
        counters: list[str] = []

        for item in args:
            if isinstance(item, str):
                if phase is None and len([a for a in args if isinstance(a, str)]) > 1:
                    phase = item
                else:
                    message = message or item
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                fraction = fraction if fraction is not None else item
            elif isinstance(item, Mapping):
                phase = phase or item.get("phase") or item.get("stage")
                message = message or item.get("message") or item.get("detail")
                fraction = fraction if fraction is not None else item.get(
                    "progress", item.get("fraction")
                )
            else:
                # A ScanProgress (or anything else that reports itself by attribute).
                phase = phase or getattr(item, "phase", None)
                message = message or getattr(item, "message", None)
                pages = getattr(item, "pages_fetched", None)
                findings = getattr(item, "findings", None)
                if isinstance(pages, int) and pages:
                    counters.append(f"{pages} page(s)")
                if isinstance(findings, int) and findings:
                    counters.append(f"{findings} finding(s)")

        name = str(getattr(phase, "value", phase) or "").lower()
        if fraction is None and name in SCAN_PHASE_ORDER:
            fraction = SCAN_PHASE_ORDER.index(name) / (len(SCAN_PHASE_ORDER) - 1)

        scaled: float | None = None
        if isinstance(fraction, (int, float)) and not isinstance(fraction, bool):
            scaled = low + (high - low) * min(1.0, max(0.0, float(fraction)))

        text = str(message or "").strip()
        if counters:
            text = f"{text} ({', '.join(counters)})".strip()
        context.progress(
            phase=f"scan:{name}" if name else None,
            fraction=scaled,
            message=text or (f"Scanner phase: {name}." if name else None),
        )

    return on_progress


def _scan_target(request: AnalyzeRequest, context: JobContext) -> tuple[Any, dict[str, Any]]:
    """Run :mod:`vulnpriority.scan` against the authorised target.

    Returns the ``Scan`` and a record of which tool actually produced it, because "ZAP
    found this" and "the built-in crawler found this" are different claims and the page
    and the report both have to be able to say which one they are making.
    """
    try:
        module = importlib.import_module("vulnpriority.scan")
        assess_target = module.assess_target
        scan_request_class = module.ScanRequest
        profile_class = getattr(module, "ScanProfile", None)
    except Exception as error:  # noqa: BLE001
        raise AnalysisUnavailable(
            "Assessing a target needs the vulnpriority.scan package, which this build cannot "
            f"import ({type(error).__name__}: {error}). Upload a scanner report instead."
        ) from error

    profile: Any = request.profile
    if profile_class is not None:
        try:
            profile = profile_class(request.profile)
        except Exception:  # noqa: BLE001 - not an enum, or a different spelling
            candidate = getattr(profile_class, request.profile.upper(), None)
            if candidate is not None:
                profile = candidate

    # Normalise before building the request, and through the scan package's own helper so
    # the site and the CLI cannot disagree about what "localhost:3000" means. Without it the
    # bare host parses as a scheme, and a loopback target gets https:// and fails on TLS.
    try:
        from vulnpriority.scan.safety import normalise_target

        target_url = normalise_target(request.target_url)
    except ImportError:  # pragma: no cover - scan package already imported above
        target_url = request.target_url

    candidates: dict[str, Any] = {
        "target_url": target_url,
        "url": target_url,
        "target": target_url,
        "authorized": True,
        "authorization_note": request.authorization_note,
        "profile": profile,
        "allow_private_targets": request.allow_private_target,
    }
    accepted = _accepted_fields(scan_request_class)
    kwargs = {key: value for key, value in candidates.items() if not accepted or key in accepted}
    for alias in ("url", "target"):        # never send two spellings of the same thing
        if "target_url" in kwargs and alias in kwargs:
            kwargs.pop(alias)
    try:
        scan_request = scan_request_class(**kwargs)
    except Exception as error:  # noqa: BLE001
        raise AnalysisUnavailable(
            f"vulnpriority.scan rejected the request this page built ({type(error).__name__}: "
            f"{error}). The scanner's request contract has probably moved."
        ) from error

    context.progress(
        "scan", 0.06, f"Assessing {target_url} on the {request.profile} profile."
    )
    call_kwargs: dict[str, Any] = {}
    if _accepts_keyword(assess_target, "on_progress"):
        call_kwargs["on_progress"] = _scan_progress(context, 0.06, 0.22)

    # An explicit choice is honoured; an empty one leaves the scan package to prefer a real
    # scanner over the built-in crawler, which is its own default and the better one.
    chosen = (request.scanner or "").strip()
    if chosen == BUILTIN_SCANNER and _accepts_keyword(assess_target, "force_builtin"):
        call_kwargs["force_builtin"] = True
    elif chosen and chosen != BUILTIN_SCANNER and _accepts_keyword(assess_target, "order"):
        call_kwargs["order"] = [chosen]
        try:
            detected = module.detect_external_tools()
            picked = tuple(tool for tool in detected if getattr(tool, "name", "") == chosen)
            if picked and _accepts_keyword(assess_target, "tools"):
                call_kwargs["tools"] = picked
        except Exception as error:  # noqa: BLE001 - fall back to ordering alone
            LOGGER.debug("could not pin the scanner to %s: %s", chosen, error)

    outcome = assess_target(scan_request, **call_kwargs)
    context.raise_if_cancelled()

    selection = str(getattr(outcome, "tool_selection", "") or "")
    if selection:
        context.progress("scan", None, selection)
    used = {
        "tool": str(getattr(outcome, "tool", "") or ""),
        "tool_version": str(getattr(outcome, "tool_version", "") or ""),
        "selection": selection,
        "available_tools": [str(t) for t in (getattr(outcome, "available_tools", ()) or ())],
        "external_attempts": [str(t) for t in (getattr(outcome, "external_attempts", ()) or ())],
        # What the scan could not see. Carried through to the page because a short queue
        # from a single-page application whose routes live in JavaScript looks exactly like
        # a short queue from a clean application, and the reader cannot tell which they have
        # unless the scan says so.
        "coverage_notes": [str(n) for n in (getattr(outcome, "coverage_notes", ()) or ())],
        "requested": chosen,
    }

    scan = getattr(outcome, "scan", None)
    if scan is None:
        from vulnpriority.core.models import Scan

        scan = outcome if isinstance(outcome, Scan) else None
    if scan is None:
        raise AnalysisUnavailable(
            "vulnpriority.scan returned an outcome with no .scan on it, so there is nothing to "
            "prioritise. This is a scanner-side contract problem, not a configuration one."
        )
    return scan, used


def run_analysis(
    request: AnalyzeRequest,
    context: JobContext,
    *,
    source_path: Path | None = None,
    cleanup: Callable[[], None] | None = None,
    run_id: str | None = None,
) -> DashboardData:
    """Produce a dashboard payload for one request.

    The path through the framework is the real one: ingest, Component A, Component B,
    Component C, features, ranking, explanation and knapsack selection, then
    :func:`vulnpriority.web.exporter.build_dashboard`. Evaluation and ablation need labelled
    history across many scans and are not part of a single-scan triage, so they are absent
    from the payload and the page hides those sections, exactly as it already does for a
    partial run loaded from disk.

    ``source_path`` is a report already streamed to disk by the multipart endpoint;
    ``cleanup`` is how that endpoint's temporary directory is released, and it runs whether
    this succeeds, fails or is cancelled.
    """
    try:
        return _run_analysis(request, context, source_path=source_path, run_id=run_id)
    finally:
        if cleanup is not None:
            try:
                cleanup()
            except Exception:  # noqa: BLE001 - a temp-dir failure must not mask the result
                LOGGER.warning("could not clean up the upload directory", exc_info=True)


def _run_analysis(
    request: AnalyzeRequest,
    context: JobContext,
    *,
    source_path: Path | None,
    run_id: str | None,
) -> DashboardData:
    from vulnpriority.pipeline import stages
    from vulnpriority.web.exporter import build_dashboard

    started = datetime.now(timezone.utc)

    if request.mode == "demo":
        context.progress("demo", 0.3, "Loading the worked example.")
        from vulnpriority.web.demo import demo_dashboard

        data = demo_dashboard()
        data.notes["source"] = {"mode": "demo"}
        context.progress("complete", 1.0, "Worked example ready.")
        return data

    config = _analysis_config(request)
    context.progress("input", 0.03, "Resolving the run configuration.")

    workdir: tempfile.TemporaryDirectory[str] | None = None
    scanner_used: dict[str, Any] = {}
    try:
        if request.mode == "upload":
            path = source_path
            if path is None:
                workdir = tempfile.TemporaryDirectory(prefix="vulnpriority-upload-")
                path = _decode_upload(request, Path(workdir.name))
            context.progress("ingest", 0.08, f"Parsing {path.name}.")
            scans = stages.ingest_stage(config, scan_paths=[path])
        else:
            scan, scanner_used = _scan_target(request, context)
            context.progress("ingest", 0.24, "Correlating findings into root causes.")
            scans = stages.ingest_stage(config, scans=[scan])
    finally:
        if workdir is not None:
            workdir.cleanup()

    if not scans:
        raise AnalysisUnavailable("Nothing was ingested: the report produced no scan document.")

    # Whether live exploit intelligence can honestly describe this scan's moment. A target
    # assessed just now always agrees with itself; an uploaded report may not, and the page
    # has already offered the operator the choice, so this only records what was decided.
    as_of = judge_report_freshness(
        max(scan.scanned_at for scan in scans), live=request.use_intel
    )
    findings = sum(len(scan.findings) for scan in scans)
    endpoints = sum(len(scan.endpoints) for scan in scans)
    context.progress(
        "ingest", 0.28,
        f"{findings} findings across {endpoints} endpoints in {len(scans)} scan(s).",
    )

    context.progress("assess", 0.32, "Component A: asset criticality, exploitability, applicability.")
    assessments = stages.assess_stage(config, scans)

    context.progress(
        "enrich", 0.52,
        f"Component B: attacker '{config.component_b.attacker_preset}', "
        f"impact model '{config.component_b.impact_preset}'.",
    )
    enriched = stages.enrich_stage(config, scans, assessments)

    chain_by_scan: dict[str, dict[str, Any]] = {}
    graphs: list[Any] = []
    if config.component_c.enabled:
        context.progress("chain", 0.66, "Component C: attack graph and reachability contribution.")
        graphs, chain_by_scan = stages.chain_stage(config, scans, enriched)
    else:
        context.progress("chain", 0.66, "Component C disabled: no attack graph built.")
    chain = {
        finding_id: score
        for scores in chain_by_scan.values()
        for finding_id, score in scores.items()
    }

    context.progress("rank", 0.76, "Building features and ranking.")
    _frame, ranking, _explanations = stages.rank_stage(config, enriched, chain_by_scan)

    context.progress(
        "select", 0.9, f"Knapsack selection under {config.selection.budget_hours:g} hours."
    )
    selections = stages.select_stage(config, enriched, ranking, chain_scores=chain_by_scan)

    context.progress("payload", 0.96, "Assembling the dashboard payload.")
    manifest = _manifest_for(config, scans, run_id or context.job_id)
    data = build_dashboard(
        scans=scans,
        enriched=enriched,
        chain=chain,
        graphs=graphs,
        ranking=ranking,
        selections=selections,
        manifest=manifest,
        config=config,
        generated_at=started,
    )
    data.notes["source"] = {
        "mode": request.mode,
        "target_url": request.target_url if request.mode == "scan" else "",
        "report_filename": (source_path.name if source_path is not None else
                            (request.report.filename if request.report else "")) if request.mode == "upload" else "",
        "profile": request.profile if request.mode == "scan" else "",
        "attacker": config.component_b.attacker_preset,
        "impact_model": config.component_b.impact_preset,
        "budget_hours": float(config.selection.budget_hours),
        "components": {
            "a": config.component_a.enabled,
            "b": config.component_b.enabled,
            "c": config.component_c.enabled,
        },
        "ranker": str(getattr(ranking.ranker, "value", ranking.ranker)),
    }
    # Which tool actually produced the findings, and why that one. A reader must be able to
    # tell a ZAP scan from the built-in crawler, because they find different things.
    if scanner_used:
        data.notes["scanner"] = scanner_used
    # How the queue was ordered. ``ranking.ranker`` is the policy that was asked for and says
    # nothing about whether a model produced the order; for a single scan the answer used to
    # be no while the payload still said "lambdamart".
    data.notes["ranking"] = {
        "ranker": str(getattr(ranking.ranker, "value", ranking.ranker)),
        "model_fitted": bool(getattr(ranking, "model_fitted", False)),
        "model_pretrained": bool(getattr(ranking, "model_pretrained", False)),
        "fallback_reason": str(getattr(ranking, "fallback_reason", "")),
    }
    data.notes["single_scan"] = (
        "This is a single-scan triage. Evaluation, ablation and the longitudinal simulation "
        "need labelled history across many scans, so they are absent rather than fabricated."
    )
    # What this assessment knew, and when. A reader months from now can tell from the payload
    # alone whether live intelligence ran, and if it did not, why not.
    data.notes["intel"] = {
        "requested": bool(request.use_intel),
        "ran": bool(request.use_intel and as_of.search_allowed),
        "gathered_on": datetime.now().date().isoformat()
        if (request.use_intel and as_of.search_allowed) else "",
        "scan_date": as_of.as_of,
        "status": as_of.status,
        "evidence_is_current": bool(as_of.evidence_is_current),
        "message": as_of.message,
        "disabled_reason": (
            "The operator chose to assess this report without live exploit intelligence, so "
            "that the evidence and the scan describe the same moment."
            if not request.use_intel and request.mode == "upload" else ""
        ),
    }
    context.progress("complete", 1.0, f"Ranked {len(data.findings)} findings.")
    return data


def _manifest_for(config: Any, scans: Sequence[Any], run_id: str) -> Any:
    """A run manifest for this analysis, so the page's provenance line is not blank."""
    try:
        from vulnpriority.pipeline.runner import build_manifest

        return build_manifest(
            config, f"web-{run_id[:12]}", scans=list(scans),
            command="vulnpriority.web.app /api/analyze",
        )
    except Exception as error:  # noqa: BLE001 - the payload is still valid without it
        LOGGER.warning("could not build a run manifest: %s", error)
        return None


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def build_report_document(
    data: DashboardData, fmt: str, *, theme: str = "", embedded: bool = False
) -> str:
    """The generated report in ``md``, ``html`` or ``json``.

    :mod:`vulnpriority.report` owns this when it is present. When it is not - or when it
    raises - the framework writes its own minimal summary from the payload instead of
    handing the operator an error page, and says in the document itself that that is what
    happened.
    """
    fmt = str(fmt).lower()
    if fmt not in {"md", "html", "json"}:
        raise ApiError(404, "unknown_format", f"No report format {fmt!r}; use md, html or json.")

    try:
        build_report = importlib.import_module("vulnpriority.report").build_report
    except Exception as error:  # noqa: BLE001
        LOGGER.info("vulnpriority.report unavailable (%s); writing the built-in summary", error)
        return _fallback_report(
            data, fmt, reason=f"vulnpriority.report is not available in this build ({error})."
        )

    # ``theme`` and ``embedded`` are presentation-only and only the HTML renderer knows what
    # to do with them. An older vulnpriority.report will not accept them, so they are offered
    # and then dropped rather than required: a report that renders with the wrong palette is
    # better than no report.
    extra = {k: v for k, v in (("theme", theme), ("embedded", embedded)) if v}
    try:
        try:
            produced = build_report(data, fmt, **extra)
        except TypeError:
            try:
                produced = build_report(data, fmt)
            except TypeError:
                produced = build_report(data.model_dump(mode="json"), fmt)
    except Exception as error:  # noqa: BLE001 - a broken generator must not cost the report
        LOGGER.exception("vulnpriority.report failed for format %s", fmt)
        return _fallback_report(
            data, fmt, reason=f"vulnpriority.report raised {type(error).__name__}: {error}."
        )
    if not isinstance(produced, str):
        produced = json.dumps(produced, ensure_ascii=False, default=str)
    return produced


def _fallback_report(data: DashboardData, fmt: str, reason: str) -> str:
    if fmt == "json":
        payload = data.model_dump(mode="json")
        payload.setdefault("notes", {})["report_generator"] = reason
        return json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    if fmt == "md":
        return _fallback_report_markdown(data, reason)
    return _fallback_report_html(data, reason)


def _money(value: float | None, currency: str) -> str:
    """A grouped amount for a table cell: every row written at the same scale."""
    return format_money(value, currency, missing="–")


def _money_phrase(value: float | None, currency: str) -> str:
    """An amount for prose: ``₹1.23 crore``, ``$1.23M``, per the run's currency."""
    return format_money_compact(value, currency, missing="–")


REPORT_COLUMNS = (
    "#", "Finding", "Endpoint", "P(exploit)", "Impact", "Expected loss",
    "Chain", "Hours", "Signals", "Why it sits here",
)


def _report_rows(data: DashboardData, limit: int = 25) -> list[tuple[str, ...]]:
    currency = data.meta.currency
    ranked = sorted(data.findings, key=lambda f: (f.rank or 10**6, -f.expected_loss))[:limit]
    rows: list[tuple[str, ...]] = []
    for finding in ranked:
        signals = []
        if finding.kev:
            signals.append("KEV")
        if finding.epss is not None and finding.epss >= 0.05:
            signals.append(f"EPSS {finding.epss:.2f}")
        if finding.is_chokepoint:
            signals.append("chokepoint")
        if finding.applicability == "not_applicable":
            signals.append("not applicable")
        if finding.alerts:
            signals.append("flagged")
        rows.append(
            (
                str(finding.rank or "–"),
                finding.name,
                f"{finding.endpoint_method} {finding.endpoint_path}",
                f"{finding.p_exploit * 100:.1f}%",
                _money(finding.impact, currency),
                _money(finding.expected_loss, currency),
                _money(finding.chain_delta, currency) if finding.chain_delta else "–",
                f"{finding.remediation_hours:.1f}" if finding.remediation_hours else "–",
                ", ".join(signals) or " - ",
                (finding.reason_codes[0] if finding.reason_codes else ""),
            )
        )
    return rows


def _fallback_report_markdown(data: DashboardData, reason: str) -> str:
    meta, summary = data.meta, data.summary
    currency = meta.currency
    source = data.notes.get("source") or {}
    lines: list[str] = []
    add = lines.append

    add("# vulnpriority - prioritised findings")
    add("")
    add(
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by vulnpriority "
        f"{meta.package_version or ''}".strip() + "."
    )
    add("")
    add(f"> {reason} This is the framework's own minimal summary of the run payload.")
    add("")

    add("## The run")
    add("")
    add("| field | value |")
    add("|---|---|")
    for key, value in (
        ("Run id", meta.run_id),
        ("Source", str(source.get("mode", ""))),
        ("Target / report", str(source.get("target_url") or source.get("report_filename") or "")),
        ("Attacker model", meta.attacker or str(source.get("attacker", ""))),
        ("Impact model", meta.impact_model or str(source.get("impact_model", ""))),
        ("Components", "".join(k.upper() for k, on in (meta.components or {}).items() if on)),
        ("Ranker", str(source.get("ranker", ""))),
        ("LLM backend", f"{meta.llm_backend} {meta.llm_model}".strip()),
        ("Feed mode", meta.feed_mode),
        ("Config hash", meta.config_hash),
    ):
        if value:
            add(f"| {key} | {value} |")
    add("")

    add("## What is at stake")
    add("")
    add(f"- {summary.n_findings} findings across {summary.n_scans} scan(s) and "
        f"{summary.n_endpoints} endpoints, covering {summary.n_clusters} distinct root causes.")
    add(f"- Total expected loss: **{_money_phrase(summary.total_expected_loss, currency)}** "
        f"against {_money_phrase(summary.total_impact, currency)} of modelled business impact.")
    add(f"- The top decile of findings carries {summary.top_decile_loss_share * 100:.0f}% of that "
        f"expected loss, which is the whole argument for ordering rather than patching down a list.")
    add(f"- {summary.n_kev} finding(s) are listed in the CISA known-exploited catalogue.")
    add(f"- {summary.total_remediation_hours:.1f} engineer-hours would clear the whole queue.")
    add("")

    add("## Remediation queue")
    add("")
    add("Ordered by expected loss over the attacker's horizon, plus what each finding opens up "
        "elsewhere on the way to something worth stealing.")
    add("")
    add("| " + " | ".join(REPORT_COLUMNS) + " |")
    add("|" + "---|" * len(REPORT_COLUMNS))
    for row in _report_rows(data):
        add("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
    if len(data.findings) > 25:
        add("")
        add(f"_{len(data.findings) - 25} further findings are in the JSON payload._")
    add("")

    if data.selections:
        add("## Under the remediation budget")
        add("")
        add("| policy | budget (h) | fixed | hours used | risk captured | share |")
        add("|---|---|---|---|---|---|")
        for selection in data.selections:
            add(
                f"| {selection.ranker} | {selection.budget_hours:g} | {selection.n_selected} | "
                f"{selection.total_hours:.1f} | {_money(selection.risk_captured, currency)} | "
                f"{selection.risk_capture_fraction * 100:.0f}% |"
            )
        add("")

    if data.graphs:
        add("## Attack chains")
        add("")
        for graph in data.graphs:
            add(f"- Scan `{graph.scan_id}`: {len(graph.nodes)} privilege states, "
                f"{len(graph.edges)} transitions, {_money_phrase(graph.total_risk, currency)} of reachable risk. "
                f"Monotonicity under patching "
                f"{'verified' if graph.monotone_verified else 'not checked'}; "
                f"{graph.rejected_untrusted_edges} edge(s) asserted only by untrusted text "
                f"were rejected.")
            for index, path in enumerate(graph.top_paths[:3], start=1):
                route = " → ".join(node.replace("state:", "") for node in path.nodes)
                add(f"  {index}. {route} - {path.probability * 100:.1f}% × "
                    f"{_money_phrase(path.target_value, currency)} = "
                    f"{_money_phrase(path.expected_value, currency)}")
        add("")

    flagged = [f for f in data.findings if f.alerts]
    add("## Provenance")
    add("")
    scanner = data.notes.get("scanner") or {}
    if scanner.get("tool"):
        version = f" {scanner['tool_version']}" if scanner.get("tool_version") else ""
        add(f"The findings were produced by **{scanner['tool']}{version}**. "
            f"{scanner.get('selection', '')}".strip())
        add("")
    intel = data.notes.get("intel") or {}
    if intel.get("ran") and intel.get("gathered_on"):
        add(f"Exploit intelligence was gathered from the internet on "
            f"**{intel['gathered_on']}**, against a scan dated {intel.get('scan_date', 'unknown')}.")
    elif intel.get("disabled_reason"):
        add(f"**No live exploit intelligence was gathered.** {intel['disabled_reason']} "
            f"Everything below is dated with the scan itself "
            f"({intel.get('scan_date', 'unknown')}).")
    if intel:
        add("")
    add("Every number above is traceable to the tier of evidence that produced it. Text authored "
        "by the target application or fetched from the internet is capped in how far it may move "
        "any feature, and cannot argue a known-exploited listing away.")
    add("")
    add(f"- {sum(f.injection_signals for f in data.findings)} injection signal(s) were recorded "
        f"while reading untrusted text.")
    add(f"- {len(flagged)} finding(s) are flagged for possible ranking manipulation.")
    for finding in flagged[:10]:
        add(f"  - {finding.name}: {' '.join(finding.alerts)}")
    add("")
    return "\n".join(lines) + "\n"


def _fallback_report_html(data: DashboardData, reason: str) -> str:
    esc = html_module.escape
    meta, summary = data.meta, data.summary
    currency = meta.currency
    rows = "".join(
        "<tr>" + "".join(f"<td>{esc(str(cell))}</td>" for cell in row) + "</tr>"
        for row in _report_rows(data)
    )
    head = "".join(f"<th>{esc(column)}</th>" for column in REPORT_COLUMNS)
    selection_rows = "".join(
        f"<tr><td>{esc(s.ranker)}</td><td>{s.budget_hours:g}</td><td>{s.n_selected}</td>"
        f"<td>{s.total_hours:.1f}</td><td>{esc(_money(s.risk_captured, currency))}</td>"
        f"<td>{s.risk_capture_fraction * 100:.0f}%</td></tr>"
        for s in data.selections
    )
    selections = (
        "<h2>Under the remediation budget</h2><table><thead><tr><th>Policy</th>"
        "<th>Budget (h)</th><th>Fixed</th><th>Hours used</th><th>Risk captured</th>"
        f"<th>Share</th></tr></thead><tbody>{selection_rows}</tbody></table>"
        if selection_rows else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>VulnPriority report</title>
<style>
 body {{ font: 14px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #14171c; background: #fff; margin: 0; padding: 24px; }}
 h1 {{ font-size: 20px; margin: 0 0 4px; }}
 h2 {{ font-size: 15px; margin: 26px 0 8px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
 th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #e2e5ea; vertical-align: top; }}
 th {{ color: #79828f; font-size: 11px; text-transform: uppercase; letter-spacing: .4px; }}
 .note {{ background: #fbf6e8; border: 1px solid #e7d6a8; border-radius: 8px;
          padding: 10px 13px; font-size: 12.5px; margin: 12px 0 20px; }}
 ul {{ padding-left: 18px; }}
</style></head><body>
<h1>vulnpriority - prioritised findings</h1>
<div>Generated {esc(datetime.now(timezone.utc).isoformat(timespec="seconds"))}
 by vulnpriority {esc(meta.package_version or "")}.</div>
<div class="note">{esc(reason)} This is the framework's own minimal summary of the run payload.</div>
<h2>What is at stake</h2>
<ul>
 <li>{summary.n_findings} findings across {summary.n_scans} scan(s) and {summary.n_endpoints}
     endpoints, covering {summary.n_clusters} distinct root causes.</li>
 <li>Total expected loss <b>{esc(_money_phrase(summary.total_expected_loss, currency))}</b>
     against {esc(_money_phrase(summary.total_impact, currency))} of modelled business
     impact.</li>
 <li>The top decile of findings carries {summary.top_decile_loss_share * 100:.0f}% of that
     expected loss.</li>
 <li>{summary.n_kev} finding(s) are listed in the known-exploited catalogue;
     {summary.total_remediation_hours:.1f} engineer-hours would clear the queue.</li>
</ul>
<h2>Remediation queue</h2>
<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>
{selections}
</body></html>
"""


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


#: Names that always mean "this machine, reached the way this server expects".
LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

#: Bind addresses that mean "every interface". A ``Host`` allowlist cannot be derived from
#: one - the set of names that legitimately reach the process is whatever DNS says - so the
#: check switches off and says so rather than guessing and breaking a deliberate setup.
WILDCARD_BINDS: frozenset[str] = frozenset({"0.0.0.0", "::", "*", ""})


def _host_allowlist(
    bind_host: str, extra: Sequence[str] = ()
) -> tuple[frozenset[str], bool]:
    """The hostnames this server answers to, and whether to enforce the list at all.

    Returns ``(allowed, enforced)``. Enforcement is off for a wildcard bind, because an
    operator who asked for every interface has chosen exposure and the set of names that
    reach them is not knowable from here.
    """
    if str(bind_host).strip().lower() in WILDCARD_BINDS:
        return frozenset(), False
    names = {str(bind_host).strip().lower()} | set(LOOPBACK_HOSTS)
    names |= {str(name).strip().lower() for name in extra if str(name).strip()}
    return frozenset(names - {""}), True


def _request_hostname(raw: str | None) -> str:
    """The hostname part of a ``Host`` header, lower-cased, port and brackets removed."""
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    if text.startswith("["):                      # [::1]:8765
        closing = text.find("]")
        return text[1:closing] if closing > 0 else text.strip("[]")
    return text.rsplit(":", 1)[0] if text.count(":") == 1 else text


@dataclass
class AppSettings:
    """Everything ``create_app`` needs. Passed as one object so tests can vary one field."""

    data: DashboardData | None = None
    token: str | None = None
    asset_dir: Path = ASSET_DIR
    max_body_bytes: int = MAX_BODY_BYTES
    allow_scan: bool = True
    jobs: JobStore | None = None
    max_history: int = 32
    #: The address the server is bound to. Part of the settings because the ``Host`` check
    #: has to know which names legitimately reach this process.
    host: str = DEFAULT_HOST
    #: Extra hostnames to accept in the ``Host`` header, for a reverse proxy or a name the
    #: operator has pointed at this machine on purpose.
    allowed_hosts: tuple[str, ...] = ()


class WebApplication:
    """The service layer: token, jobs, bundled payload and the checks before a job starts.

    Deliberately free of HTTP. The routes below are thin, which is what makes it possible
    to reason about the authorisation rule in one place instead of across five handlers.
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        settings = settings or AppSettings()
        self.settings = settings
        self.token = settings.token or secrets.token_urlsafe(24)
        self.jobs = settings.jobs or JobStore(max_history=settings.max_history)
        self.data = settings.data
        self.asset_dir = Path(settings.asset_dir)
        self.max_body_bytes = int(settings.max_body_bytes)
        self.allow_scan = bool(settings.allow_scan)
        self.allowed_hosts, self.host_check_enabled = _host_allowlist(
            settings.host, settings.allowed_hosts
        )
        self._options: WebConfigOptions | None = None
        self._lock = threading.Lock()

    # -- capabilities and configuration -------------------------------------

    @property
    def capabilities(self) -> WebCapabilities:
        return probe_capabilities()

    def options(self) -> WebConfigOptions:
        with self._lock:
            if self._options is None:
                self._options = config_options(self.capabilities)
            return self._options

    # -- assets and payload --------------------------------------------------

    def asset(self, name: str) -> Path:
        path = self.asset_dir / name
        if not path.is_file():
            raise ApiError(404, "missing_asset", f"{name} is not present in this install.")
        return path

    def index_html(self) -> str:
        """``index.html`` with this process's session token written into it."""
        text = self.asset("index.html").read_text(encoding="utf-8")
        session = json.dumps(
            {
                "token": self.token,
                "server": True,
                "api_version": INTERACTIVE_API_VERSION,
                "token_header": TOKEN_HEADER,
            }
        )
        injected = f"window.VULNPRIORITY_SESSION = {session};"
        if SESSION_MARKER in text:
            return text.replace(SESSION_MARKER, injected, 1)
        return text.replace("</head>", f"<script>{injected}</script>\n</head>", 1)

    def payload(self) -> dict[str, Any]:
        return self.data.model_dump(mode="json") if self.data is not None else {}

    # -- starting work -------------------------------------------------------

    def check_preconditions(
        self, request: AnalyzeRequest, *, has_source_file: bool = False
    ) -> None:
        """Everything that must be true *before* a job exists.

        The authorisation check lives here rather than in the worker on purpose: a request
        that has not claimed authorisation must never become a job at all, so there is no
        state in which a scan is queued against a target nobody took responsibility for.

        ``has_source_file`` says the report is already on disk, which is the multipart
        path: there is no base64 body to require in that case, only a usable filename.
        """
        options = self.options()
        if request.attacker and request.attacker not in {p.name for p in options.attackers}:
            raise ApiError(
                400, "unknown_attacker",
                f"No attacker preset named {request.attacker!r}. Available: "
                f"{', '.join(sorted(p.name for p in options.attackers)) or 'none'}.",
            )
        if request.impact_model and request.impact_model not in {p.name for p in options.impact_models}:
            raise ApiError(
                400, "unknown_impact_model",
                f"No impact model named {request.impact_model!r}. Available: "
                f"{', '.join(sorted(p.name for p in options.impact_models)) or 'none'}.",
            )
        if request.mode == "upload":
            if request.report is None:
                raise ApiError(400, "missing_report", "Upload mode needs a report file.")
            if not has_source_file and not request.report.content_base64:
                raise ApiError(400, "missing_report", "Upload mode needs a report file.")
            _safe_report_name(request.report.filename)
            return
        if request.mode != "scan":
            return

        if not self.allow_scan:
            raise ApiError(
                403, "scanning_disabled",
                "This server was started with target assessment disabled. "
                "Upload a scanner report instead.",
            )
        target = request.target_url.strip()
        if not target:
            raise ApiError(400, "missing_target", "Scan mode needs a target URL.")
        # Normalise before validating, through the scan package's own helper, so this gate
        # and the scanner agree about what a bare host means. Validating the raw string
        # rejected "localhost:3000" -- which urlparse reads as the scheme "localhost" -- even
        # though it is the obvious way to name a container and the scanner accepts it once
        # normalised. The request the job runs is the normalised one.
        try:
            from vulnpriority.scan.safety import normalise_target

            target = normalise_target(target)
        except ImportError:  # pragma: no cover - scanning is gated on the package above
            pass
        request = request.model_copy(update={"target_url": target})
        parsed = urlparse(target)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ApiError(
                400, "invalid_target",
                f"{target!r} is not an absolute http(s) URL, so there is no target to "
                f"scope a scan to.",
            )
        if not request.authorized:
            raise ApiError(
                400, "not_authorized",
                "A scan will not be started without an explicit statement that you are "
                "authorised to test this target. Nothing was sent to it.",
            )
        if len(request.authorization_note.strip()) < 8:
            raise ApiError(
                400, "missing_authorization_note",
                "Record who authorised this test and under what scope (at least a short "
                "sentence). Nothing was sent to the target.",
            )

    def start(
        self,
        request: AnalyzeRequest,
        *,
        source_path: Path | None = None,
        cleanup: Callable[[], None] | None = None,
    ) -> Job:
        """Validate, register and launch. Returns as soon as the thread is running."""
        self.check_preconditions(request, has_source_file=source_path is not None)
        job = self.jobs.create(
            kind="analyze",
            meta={
                "mode": request.mode,
                "target_url": request.target_url,
                "profile": request.profile,
            },
        )
        self.jobs.update(
            job.job_id, phase="queued",
            message=f"Queued ({request.mode}).", log=f"Queued ({request.mode}).",
        )
        self.jobs.submit(
            job,
            lambda context: run_analysis(
                request, context, source_path=source_path, cleanup=cleanup
            ),
        )
        return job

    # -- reading work back ---------------------------------------------------

    def state(self, job_id: str) -> WebJobState:
        snapshot = self.jobs.snapshot(job_id)
        if snapshot is None:
            raise ApiError(
                404, "unknown_job", f"No job {job_id}. It may have aged out of the history."
            )
        return _job_state(snapshot)

    def result(self, job_id: str) -> DashboardData:
        snapshot = self.jobs.snapshot(job_id)
        if snapshot is None:
            raise ApiError(404, "unknown_job", f"No job {job_id}.")
        status = str(snapshot.get("status"))
        if status != JobStatus.DONE.value:
            raise ApiError(
                409, "not_ready",
                f"Job {job_id} is {status}"
                + (f": {snapshot.get('error')}" if snapshot.get("error") else "")
                + ". There is no result to read.",
            )
        produced = self.jobs.result(job_id)
        if not isinstance(produced, DashboardData):
            raise ApiError(
                500, "bad_result", f"Job {job_id} finished without a dashboard payload."
            )
        return produced

    def novelty(self) -> dict[str, Any]:
        capabilities = self.capabilities
        if not capabilities.novelty:
            return {
                "available": False,
                "detail": capabilities.detail.get(
                    "novelty", "vulnpriority.novelty is not available in this build."
                ),
            }
        try:
            produced = importlib.import_module("vulnpriority.novelty").novelty_payload()
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("novelty_payload() failed")
            return {
                "available": False,
                "detail": f"vulnpriority.novelty.novelty_payload() failed: "
                          f"{type(error).__name__}: {error}",
            }
        if not isinstance(produced, Mapping):
            return {"available": False, "detail": "novelty_payload() did not return a mapping."}
        out = dict(produced)
        out.setdefault("available", True)
        return out

    def close(self) -> None:
        self.jobs.shutdown()


def _job_state(snapshot: Mapping[str, Any]) -> WebJobState:
    """The wire form of a job snapshot, dropping fields the contract does not promise."""
    return WebJobState(
        job_id=str(snapshot["job_id"]),
        mode=str(snapshot.get("mode", "")),
        status=str(snapshot.get("status", "queued")),
        phase=str(snapshot.get("phase", "")),
        progress=float(snapshot.get("progress", 0.0)),
        message=str(snapshot.get("message", "")),
        log=list(snapshot.get("log", [])),
        error=snapshot.get("error"),
        created_at=str(snapshot.get("created_at", "")),
        updated_at=str(snapshot.get("updated_at", "")),
        finished_at=snapshot.get("finished_at"),
        has_result=bool(snapshot.get("has_result", False)),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _state(request: Request) -> WebApplication:
    return request.app.state.vulnpriority  # type: ignore[no-any-return]


def require_token(
    request: Request,
    supplied_header: str | None = Header(None, alias=TOKEN_HEADER),
) -> None:
    """Mutating endpoints need the token this process minted at startup.

    It is not authentication - anything that can read the served page has it. Together with
    the ``Host`` check above it stops a page on another site from driving this server: that
    one keeps a foreign origin from ever seeing the token, and this one keeps a request
    without it from doing anything.

    Header only, deliberately. It used to be accepted as ``?token=`` as well, which put a
    credential into request logs, browser history and any proxy in between; nothing in the
    page ever used that path, and the endpoints a browser cannot set headers for are not
    the mutating ones.
    """
    application = _state(request)
    candidate = supplied_header
    if not candidate or not secrets.compare_digest(str(candidate), application.token):
        raise ApiError(
            403, "bad_token",
            f"This endpoint needs the session token issued into the page, sent as the "
            f"{TOKEN_HEADER} header. Reload the page served by this process.",
        )


async def _stream_upload(upload: UploadFile, destination: Path, cap: int) -> int:
    """Copy an upload to disk, refusing it the moment it passes the cap.

    Streamed in chunks rather than read whole: a size limit that only applies after the
    bytes are already in memory is not a limit.
    """
    written = 0
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(64 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > cap:
                handle.close()
                destination.unlink(missing_ok=True)
                raise ApiError(
                    413, "body_too_large",
                    f"The uploaded report passed {cap} bytes, which is this server's limit.",
                )
            handle.write(chunk)
    if written == 0:
        destination.unlink(missing_ok=True)
        raise ApiError(400, "invalid_report", "The uploaded report was empty.")
    return written


def _register_routes(api: FastAPI) -> None:
    # -- page and generated assets ------------------------------------------

    @api.get("/", include_in_schema=False)
    @api.get("/index.html", include_in_schema=False)
    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(
            _state(request).index_html(), headers={"Cache-Control": "no-store"}
        )

    @api.get("/data.js", include_in_schema=False)
    async def data_js(request: Request) -> Response:
        text = json.dumps(_state(request).payload(), ensure_ascii=False, default=str)
        return Response(
            content=f"window.VULNPRIORITY_DATA = {text};\n",
            media_type="text/javascript; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @api.get("/data.json", include_in_schema=False)
    async def data_json(request: Request) -> JSONResponse:
        return JSONResponse(_state(request).payload())

    @api.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    # -- introspection -------------------------------------------------------

    @api.get("/api/health", response_model=WebHealth, tags=["meta"])
    async def health(request: Request) -> WebHealth:
        """Whether the server is up, and which sibling packages it can import."""
        from vulnpriority import __version__

        application = _state(request)
        return WebHealth(
            ok=True,
            version=__version__,
            schema_version=DASHBOARD_SCHEMA_VERSION,
            api_version=INTERACTIVE_API_VERSION,
            capabilities=application.capabilities,
            has_bundled_run=bool(application.data is not None and application.data.findings),
        )

    @api.get("/api/config", response_model=WebConfigOptions, tags=["meta"])
    async def configuration(request: Request) -> WebConfigOptions:
        """Attacker presets, impact models, scan profiles and the defaults, from ``configs/``."""
        return _state(request).options()

    @api.get("/api/scanners", response_model=WebScannerEnvironment, tags=["meta"])
    async def scanners(
        profile: str = Query("passive", description="The profile the scan would request."),
    ) -> WebScannerEnvironment:
        """Which scanner would actually run for a profile, and what else is installed.

        Profile and availability interact - Nuclei and Nikto cannot serve a passive request
 - so this is queried per profile rather than answered once.
        """
        if profile not in ("passive", "active"):
            raise ApiError(400, "invalid_profile", f"No scan profile {profile!r}.")
        return scanner_environment(profile)

    @api.get("/api/novelty", tags=["meta"])
    async def novelty(request: Request) -> dict[str, Any]:
        """The novelty analysis, or an honest ``{"available": false}`` when it is absent."""
        return _state(request).novelty()

    # -- starting an analysis ------------------------------------------------

    @api.post(
        "/api/analyze",
        response_model=WebJobCreated,
        status_code=202,
        tags=["analyze"],
        dependencies=[Depends(require_token)],
    )
    async def analyze(request: Request, body: AnalyzeRequest) -> WebJobCreated:
        """Start an analysis. The body carries an uploaded report as base64, or a target."""
        job = _state(request).start(body)
        return WebJobCreated(job_id=job.job_id, status=JobStatus.QUEUED.value)

    @api.post(
        "/api/analyze/upload",
        response_model=WebJobCreated,
        status_code=202,
        tags=["analyze"],
        dependencies=[Depends(require_token)],
    )
    async def analyze_upload(
        request: Request,
        file: UploadFile = File(..., description="A ZAP, Burp, Nuclei or canonical scan report."),
        attacker: str = Form(""),
        impact_model: str = Form(""),
        budget_hours: float = Form(40.0),
        component_a: bool = Form(True),
        component_b: bool = Form(True),
        component_c: bool = Form(True),
    ) -> WebJobCreated:
        """Start an analysis from a real multipart upload, streamed to disk under the cap."""
        application = _state(request)
        name = _safe_report_name(file.filename)
        workdir = tempfile.TemporaryDirectory(prefix="vulnpriority-upload-")
        try:
            path = Path(workdir.name) / name
            await _stream_upload(file, path, application.max_body_bytes)
            analyze_request = AnalyzeRequest(
                mode="upload",
                report=AnalyzeUpload(filename=name),
                attacker=attacker,
                impact_model=impact_model,
                budget_hours=budget_hours,
                components=AnalyzeComponents(a=component_a, b=component_b, c=component_c),
            )
            job = application.start(
                analyze_request, source_path=path, cleanup=workdir.cleanup
            )
        except BaseException:
            workdir.cleanup()
            raise
        return WebJobCreated(job_id=job.job_id, status=JobStatus.QUEUED.value)

    @api.post(
        "/api/inspect",
        response_model=WebReportInspection,
        tags=["analyze"],
        dependencies=[Depends(require_token)],
    )
    async def inspect_upload(
        request: Request,
        file: UploadFile = File(..., description="A scanner report to describe, not to run."),
        use_intel: bool = Form(True),
    ) -> WebReportInspection:
        """Describe a report before running it: its date, its host, and whether live
        exploit intelligence could honestly describe the same moment.

        This is what lets the page offer a re-scan instead of quietly producing an
        assessment whose evidence and whose scan describe different months.
        """
        application = _state(request)
        name = _safe_report_name(file.filename)
        workdir = tempfile.TemporaryDirectory(prefix="vulnpriority-inspect-")
        try:
            path = Path(workdir.name) / name
            await _stream_upload(file, path, application.max_body_bytes)
            return inspect_report(path, live_intel=use_intel)
        finally:
            workdir.cleanup()

    # -- watching it ---------------------------------------------------------

    @api.get("/api/jobs/{job_id}", response_model=WebJobState, tags=["jobs"])
    async def job_state(request: Request, job_id: str) -> WebJobState:
        """Poll one job. The baseline; ``/events`` is the same information, pushed."""
        return _state(request).state(job_id)

    @api.get("/api/jobs/{job_id}/events", tags=["jobs"])
    async def job_events(
        request: Request,
        job_id: str,
        after: int = Query(0, ge=0, description="Log lines already seen, for reconnects."),
    ) -> EventSourceResponse:
        """Server-Sent Events: phase, progress and new log lines until the job ends.

        Each ``state`` event carries the job's fields plus only the log lines the client has
        not seen, so a long run does not re-send its whole log every quarter second. A
        client that drops the connection reconnects with ``?after=`` and misses nothing.
        """
        application = _state(request)
        application.state(job_id)          # 404 before the stream opens, not inside it

        async def events() -> AsyncIterator[dict[str, str]]:
            seen = int(after)
            previous: tuple[Any, ...] | None = None
            while True:
                if await request.is_disconnected():
                    return
                snapshot = application.jobs.snapshot(job_id)
                if snapshot is None:  # aged out of the history mid-stream
                    yield {
                        "event": "error",
                        "data": json.dumps(_error_body("unknown_job", f"Job {job_id} is gone.")),
                    }
                    return
                log = list(snapshot.get("log", []))
                fresh = log[seen:]
                seen = len(log)
                signature = (
                    snapshot.get("status"), snapshot.get("phase"),
                    snapshot.get("progress"), snapshot.get("message"), bool(fresh),
                )
                terminal = str(snapshot.get("status")) not in (
                    JobStatus.QUEUED.value, JobStatus.RUNNING.value
                )
                if signature != previous or terminal:
                    previous = signature
                    payload = _job_state(snapshot).model_dump(mode="json")
                    payload["log"] = fresh
                    payload["log_offset"] = seen
                    yield {
                        "event": "done" if terminal else "state",
                        "data": json.dumps(payload, default=str),
                    }
                if terminal:
                    return
                await asyncio.sleep(STREAM_INTERVAL_S)

        return EventSourceResponse(events())

    @api.get("/api/jobs/{job_id}/result", response_model=DashboardData, tags=["jobs"])
    async def job_result(request: Request, job_id: str) -> DashboardData:
        """The finished dashboard payload - the same document the static site consumes."""
        return _state(request).result(job_id)

    @api.get("/api/jobs/{job_id}/report.{fmt}", tags=["jobs"])
    async def job_report(
        request: Request,
        job_id: str,
        fmt: str,
        download: bool = Query(False, description="Send as an attachment rather than inline."),
        theme: str = Query("", description="light or dark, to match the page embedding this."),
        embed: bool = Query(False, description="Drop the page furniture a host page provides."),
    ) -> Response:
        """The generated report in ``md``, ``html`` or ``json``.

        ``theme`` and ``embed`` shape the HTML for display inside the results page; a
        download is always the standalone document, whatever the page asked for, because
        the file has to make sense on its own afterwards.
        """
        data = _state(request).result(job_id)
        text = build_report_document(
            data,
            fmt,
            theme="" if download else theme,
            embedded=bool(embed) and not download,
        )
        media = {
            "md": "text/markdown; charset=utf-8",
            "html": "text/html; charset=utf-8",
            "json": "application/json; charset=utf-8",
        }[fmt.lower()]
        headers = (
            {"Content-Disposition": f'attachment; filename="vulnpriority-report.{fmt.lower()}"'}
            if download else {}
        )
        return Response(content=text, media_type=media, headers=headers)

    @api.post(
        "/api/jobs/{job_id}/cancel",
        response_model=WebJobState,
        tags=["jobs"],
        dependencies=[Depends(require_token)],
    )
    async def cancel_job(request: Request, job_id: str) -> WebJobState:
        """Ask a job to stop. Queued jobs stop at once, running ones at their next phase."""
        application = _state(request)
        job = application.jobs.cancel(job_id)
        if job is None:
            raise ApiError(404, "unknown_job", f"No job {job_id}.")
        return application.state(job_id)

    # -- anything else under /api -------------------------------------------

    @api.api_route(
        "/api/{rest:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    async def unknown_api(request: Request, rest: str) -> JSONResponse:
        return _json_error(
            404, "unknown_endpoint", f"No {request.method} endpoint /api/{rest}."
        )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(config: AppSettings | None = None, **kwargs: Any) -> FastAPI:
    """Build the ASGI application.

    ``config`` is an :class:`AppSettings`; loose keyword arguments are accepted as a
    convenience and folded into one. Every piece of mutable state lives on the returned
    application, so two of these in one process - which is what the tests are - never see
    each other's jobs or tokens.
    """
    if kwargs and config is not None:  # pragma: no cover - caller error, made loud
        raise TypeError("pass either an AppSettings or keyword arguments, not both")
    settings = config or AppSettings(**kwargs)
    application = WebApplication(settings)

    @asynccontextmanager
    async def lifespan(_api: FastAPI) -> AsyncIterator[None]:
        yield
        # Ask any live analysis to stop rather than leaving a thread writing into a store
        # nobody will read. Cancellation is cooperative, so this is quick.
        application.close()

    api = FastAPI(
        title="vulnpriority",
        version=INTERACTIVE_API_VERSION,
        summary="Local, offline web application vulnerability prioritization.",
        description=(
            "A local tool, not a service: it binds loopback, has no authentication beyond a "
            "per-process session token on mutating endpoints, and will scan whatever target "
            "a request names. Do not expose it."
        ),
        lifespan=lifespan,
    )
    api.state.vulnpriority = application

    # -- error shape ---------------------------------------------------------

    @api.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        slug = getattr(exc, "error", None) or _SLUGS.get(exc.status_code, "error")
        return _json_error(exc.status_code, slug, str(exc.detail or ""))

    @api.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = (exc.errors() or [{}])[0]
        where = ".".join(str(part) for part in first.get("loc", ()) if part != "body") or "body"
        return _json_error(
            400, "invalid_request", f"{where}: {first.get('msg', 'is not valid')}"
        )

    @api.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The traceback goes to the log on this machine; the client is told what failed,
        # never where, because a stack trace is a map of the filesystem.
        LOGGER.exception("unhandled error handling %s %s", request.method, request.url.path)
        return _json_error(
            500, "internal_error",
            "The server failed while handling this request. The details are in the server "
            "log on this machine.",
        )

    # -- body cap ------------------------------------------------------------

    @api.middleware("http")
    async def _limit_body(request: Request, call_next: Callable[..., Any]) -> Response:
        cap = application.max_body_bytes
        raw = request.headers.get("content-length")
        if raw is not None:
            try:
                length = int(raw)
            except ValueError:
                return _json_error(400, "bad_content_length", "Content-Length is not a number.")
            if length < 0:
                return _json_error(400, "bad_content_length", "Content-Length is negative.")
            if length > cap:
                return _json_error(
                    413, "body_too_large",
                    f"The request body is {length} bytes; this server reads at most {cap}.",
                )
        elif request.method in ("POST", "PUT", "PATCH") and not request.url.path.endswith("/upload"):
            # Without a length there is nothing to check up front. The multipart endpoint
            # caps while streaming, so it is allowed through; a JSON body is not.
            return _json_error(
                411, "length_required",
                "Send a Content-Length; an unbounded body cannot be capped before it is read.",
            )
        return await call_next(request)

    # -- who we answer to ----------------------------------------------------
    #
    # Runs before the router and before the body cap, and reads only a header, so a request
    # from a hostname we do not answer to is refused without its body ever being read.
    #
    # Without it the session token is not the control it claims to be. A page on any site
    # the operator visits can point its own domain at 127.0.0.1 (DNS rebinding, a one-line
    # DNS record), fetch "/" from it, and read the token straight out of the served HTML -
    # the browser considers that response same-origin, because as far as it knows the page
    # and the server share a hostname. With the token it can drive /api/analyze, and that
    # endpoint takes ``target_url``, ``authorized`` and ``allow_private_target`` as request
    # fields, so the victim's machine becomes an internal port scanner. The scope checks in
    # vulnpriority.scan.safety do not help: the attacker supplies the attestation.
    #
    # Checking the Host header is the standard answer, and it is sufficient here because
    # the attack's whole mechanism is that the browser sends the *attacker's* hostname.
    @api.middleware("http")
    async def _check_host(request: Request, call_next: Callable[..., Any]) -> Response:
        if not application.host_check_enabled:
            return await call_next(request)
        raw = request.headers.get("host")
        # A client that sends no Host at all is speaking HTTP/1.0 by hand, not a browser
        # under a rebinding attack; there is nothing for this check to be wrong about.
        if raw is None:
            return await call_next(request)
        if _request_hostname(raw) not in application.allowed_hosts:
            return _json_error(
                421, "wrong_host",
                f"This server answers to {', '.join(sorted(application.allowed_hosts))} and "
                f"was asked for {raw!r}. That usually means a page on another site pointed "
                "its own hostname at this machine. Open the page from the address the "
                "server printed at startup.",
            )
        return await call_next(request)

    @api.middleware("http")
    async def _security_headers(request: Request, call_next: Callable[..., Any]) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        # Nothing here is worth caching and a stale asset is worth less than nothing: the
        # files can change under a running server, and a job result is never the same twice.
        response.headers["Cache-Control"] = "no-store"
        return response

    _register_routes(api)

    # Mounted last so every explicit route above wins. It serves app.js, analyze.js and
    # styles.css straight off disk with the right content types, and nothing else is in
    # the directory, so there is no path arithmetic to get wrong.
    if application.asset_dir.is_dir():
        api.mount("/", StaticFiles(directory=str(application.asset_dir)), name="assets")

    return api


#: The status-code fallbacks used when an ``HTTPException`` carries no slug of its own.
_SLUGS: dict[int, str] = {
    400: "bad_request",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    411: "length_required",
    413: "body_too_large",
    422: "invalid_request",
    500: "internal_error",
}


#: The default application, for ``uvicorn vulnpriority.web.app:app``.
app = create_app()


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    data: DashboardData | None = None,
    allow_scan: bool = True,
    open_browser: bool = False,
    log_level: str = "warning",
    allowed_hosts: Sequence[str] = (),
    api: FastAPI | None = None,
) -> None:
    """Run the interactive site under uvicorn until interrupted.

    Binding anywhere other than loopback is possible and is logged as the warning it is;
    see this module's docstring for why you should not.

    ``host`` reaches the application as well as uvicorn, because the ``Host`` check needs to
    know which names legitimately arrive here; ``allowed_hosts`` adds any others, for a
    reverse proxy or a name the operator has pointed at this machine deliberately.
    """
    import uvicorn

    api = api or create_app(
        AppSettings(
            data=data,
            allow_scan=allow_scan,
            host=host,
            allowed_hosts=tuple(allowed_hosts),
        )
    )
    if host not in ("127.0.0.1", "localhost", "::1"):
        LOGGER.warning(
            "vulnpriority's interactive server is bound to %s, not loopback. It has no "
            "authentication and will start scans on request. Do not leave it exposed.",
            host,
        )
    url = f"http://{host if ':' not in host else f'[{host}]'}:{port}/"
    if open_browser:  # pragma: no cover - depends on the desktop
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    uvicorn.run(api, host=host, port=int(port), log_level=log_level)


def main(argv: list[str] | None = None) -> int:
    """``python -m vulnpriority.web.app`` - start the interactive site on localhost."""
    parser = argparse.ArgumentParser(
        prog="python -m vulnpriority.web.app",
        description="Serve the interactive vulnpriority site (local tool; do not expose it).",
    )
    parser.add_argument("run_dir", nargs="?", help="Run directory to preload as the bundled run.")
    parser.add_argument("--demo", action="store_true", help="Preload the worked example.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port (default 8765).")
    parser.add_argument("--open", action="store_true", help="Open a browser at the URL.")
    parser.add_argument("--no-scan", action="store_true", help="Refuse target assessment; uploads only.")
    parser.add_argument("--log-level", default="warning", help="uvicorn log level.")
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Additional hostname to accept in the Host header. Loopback names and the bind "
            "address are always accepted; anything else is refused, which is what stops a "
            "page on another site from reaching this server by pointing its own name here."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s"
    )

    data: DashboardData | None = None
    if args.demo:
        from vulnpriority.web.demo import demo_dashboard

        data = demo_dashboard()
    elif args.run_dir:
        from vulnpriority.web.loader import load_run

        data = load_run(Path(args.run_dir))

    api = create_app(
        AppSettings(
            data=data,
            allow_scan=not args.no_scan,
            host=args.host,
            allowed_hosts=tuple(args.allowed_host),
        )
    )
    capabilities = probe_capabilities()
    print(f"vulnpriority interactive site at http://{args.host}:{args.port}/   (docs at /docs)")
    print(f"  scan={capabilities.scan}  report={capabilities.report}  novelty={capabilities.novelty}")
    print("  local tool: bound to this machine, no authentication. Ctrl-C to stop.")
    try:
        serve(
            args.host, args.port, open_browser=args.open,
            log_level=args.log_level, api=api,
        )
    except KeyboardInterrupt:  # pragma: no cover
        print("\nstopping")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
