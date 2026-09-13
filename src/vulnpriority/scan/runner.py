"""The scan orchestrator: authorise, crawl, check, and produce a ``Scan``.

The product of an assessment is a :class:`vulnpriority.core.models.Scan` built with the same
helpers a parsed ZAP report goes through - :class:`~vulnpriority.ingest.normalize.EndpointAccumulator`
for endpoints and auth inference, :func:`~vulnpriority.ingest.normalize.make_finding_id` and
friends for identifiers, :func:`~vulnpriority.ingest.tech_fingerprint.fingerprint_response`
for the technology stack, and :class:`~vulnpriority.ingest.correlate.FindingCorrelator` for
dedup keys and cluster sizes. Nothing downstream can tell the difference, and nothing
downstream needed changing.

:func:`assess_target` is the single entry point the web server and the CLI call. It
prefers an installed external scanner when asked to, and falls back to the built-in one
with a clear message when none is present.

Determinism: ``scan_id`` is derived from ``scanned_at`` (DESIGN 3.1), so a request that
sets :attr:`~vulnpriority.scan.models.ScanRequest.scanned_at` produces identical identifiers
on every run over the same site. Leaving it unset stamps the current time, which is what
a real assessment wants.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from vulnpriority import __version__
from vulnpriority.core.models import Finding, Scan, TechComponent
from vulnpriority.ingest.correlate import FindingCorrelator
from vulnpriority.ingest.normalize import (
    EndpointAccumulator,
    make_app_id,
    make_finding_id,
    make_scan_id,
    scanner_text,
    target_text,
)
from vulnpriority.ingest.tech_fingerprint import fingerprint_library, fingerprint_response, merge_tech
from vulnpriority.scan.active import run_active_probes
from vulnpriority.scan.adapters import (
    TOOL_SPECS,
    ExternalTool,
    ExternalToolError,
    detect_external_tools,
    ingest_report,
    run_external,
    container_target,
    select_tools,
    tool_version,
)
from vulnpriority.scan.checks import CheckContext, run_checks
from vulnpriority.scan.crawler import Crawler
from vulnpriority.scan.http import HttpClient, fetch_robots_policy
from vulnpriority.scan.models import (
    SCANNER_NAME,
    CheckFinding,
    Page,
    ProbeResult,
    ScanOutcome,
    ScanPhase,
    ScanProfile,
    ScanProgress,
    ScanRequest,
)
from vulnpriority.scan.safety import Budget, RateLimiter, RobotsPolicy, require_authorization, require_target_allowed

__all__ = [
    "run_scan",
    "assess_target",
    "build_scan",
    "ProgressLog",
    "default_cve_lookup",
]

ProgressCallback = Callable[[ScanProgress], None]


class ProgressLog:
    """Collects progress events and forwards them to an optional caller callback."""

    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self.events: list[ScanProgress] = []
        self._callback = callback

    def __call__(self, event: ScanProgress) -> None:
        self.events.append(event)
        if self._callback is not None:
            self._callback(event)

    def emit(
        self,
        phase: ScanPhase,
        *,
        message: str = "",
        pages: int = 0,
        findings: int = 0,
        elapsed_s: float = 0.0,
    ) -> None:
        self(
            ScanProgress(
                phase=phase,
                pages_fetched=pages,
                findings=findings,
                elapsed_s=elapsed_s,
                message=message,
            )
        )

    def as_tuple(self) -> tuple[ScanProgress, ...]:
        return tuple(self.events)


def default_cve_lookup(as_of: date | None = None) -> Callable[[str, date | None], bool] | None:
    """Confirm candidate CVE ids against the offline NVD fixture feed.

    Used by the outdated-library check so that a claimed CVE is one the intelligence layer
    actually knows about as of the scan date. Returns ``None`` when no feed is available,
    in which case the check lowers its own confidence rather than asserting anything.
    """
    try:
        from vulnpriority.feeds.nvd import NvdFixtureFeed
    except Exception:  # pragma: no cover - feeds are an optional dependency of a scan
        return None
    try:
        feed = NvdFixtureFeed()
    except Exception:  # pragma: no cover - missing fixture directory
        return None

    def _lookup(cve_id: str, when: date | None) -> bool:
        try:
            return feed.get(cve_id, when or as_of or date.today()) is not None
        except Exception:  # pragma: no cover - a feed failure must never fail a scan
            return False

    return _lookup


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Scan assembly
# ---------------------------------------------------------------------------


def build_scan(
    request: ScanRequest,
    pages: Sequence[Page],
    findings: Sequence[CheckFinding],
    *,
    scanned_at: datetime | None = None,
) -> Scan:
    """Turn crawled pages and check findings into a canonical :class:`Scan`.

    Every identifier comes from :mod:`vulnpriority.ingest.normalize`, so an endpoint this
    scanner discovered and the same endpoint discovered by ZAP carry the same
    ``endpoint_id`` and correlate against each other without special handling.
    """
    host = request.target_host
    app_id = make_app_id(host or request.target_url)
    stamp = scanned_at or request.scanned_at or _now()
    scan_id = make_scan_id(app_id, stamp, SCANNER_NAME)
    base = request.target_origin

    accumulator = EndpointAccumulator(app_id)
    url_to_endpoint: dict[str, str] = {}

    # Pass 1: every fetched page becomes an endpoint observation.
    for page in pages:
        url = page.effective_url
        tech = fingerprint_response(
            headers=page.headers, cookies=page.set_cookies, body=page.body, url=url
        )
        endpoint_id = accumulator.add(
            url,
            page.method,
            base=base,
            response_status=page.status,
            response_content_type=page.content_type,
            response_size_bytes=page.body_bytes_len,
            sets_cookie=bool(page.set_cookies),
            response_sample=target_text(page.body, source_url=url),
            observed_tech=tech,
        )
        url_to_endpoint[url] = endpoint_id
        url_to_endpoint.setdefault(page.url, endpoint_id)

    # Pass 2: forms are recorded as endpoints (never submitted), and observed links become
    # ``links_to`` edges, which is what Component C reads to build lateral movement.
    for page in pages:
        source_id = url_to_endpoint.get(page.effective_url)
        targets = [
            url_to_endpoint[link] for link in page.links if link in url_to_endpoint
        ]
        if source_id and targets:
            accumulator.add(page.effective_url, page.method, base=base, links_to=targets)
        for form in page.forms:
            # Recorded as an endpoint so the ranking layer sees a state-changing route.
            # Deliberately not mapped into ``url_to_endpoint``: the form's action may share
            # a URL with a GET page, and a finding must never be attributed to the wrong verb.
            accumulator.add(
                form.action,
                form.method,
                base=base,
                parameters=form.field_names,
            )

    endpoints = accumulator.endpoints()
    for endpoint in endpoints:
        url_to_endpoint.setdefault(endpoint.url, endpoint.endpoint_id)

    # Findings, keyed the way DESIGN 3.1 specifies.
    built: dict[str, Finding] = {}
    for item in findings:
        endpoint_id = url_to_endpoint.get(item.url)
        if not endpoint_id:
            endpoint_id = accumulator.add(item.url, item.method, base=base)
            url_to_endpoint[item.url] = endpoint_id
        finding_id = make_finding_id(scan_id, endpoint_id, item.check_id, item.param)
        if finding_id in built:
            continue
        description = scanner_text(f"{item.name}: {item.detail}") or scanner_text(item.name)
        evidence = target_text(item.evidence, source_url=item.url)
        component = fingerprint_library(item.evidence) if item.check_id == "outdated-js-library" else None
        built[finding_id] = Finding(
            finding_id=finding_id,
            scan_id=scan_id,
            app_id=app_id,
            endpoint_id=endpoint_id,
            name=item.name,
            cwe_id=item.cwe_id,
            cve_ids=(),
            scanner=SCANNER_NAME,
            scanner_plugin_id=item.check_id,
            scanner_severity=item.severity,
            scanner_confidence=item.confidence,
            description=description,  # type: ignore[arg-type]
            evidence=(evidence,) if evidence is not None else (),
            affected_component=component,
            observed_at=stamp,
        )

    endpoints = accumulator.endpoints()
    scan = Scan(
        scan_id=scan_id,
        app_id=app_id,
        app_name=request.app_name or host or "target",
        sector=request.sector,
        scanned_at=stamp,
        scanner_name=SCANNER_NAME,
        scanner_version=__version__,
        hosts=accumulator.hosts() or ((host,) if host else ()),
        tech_stack=_scan_tech(accumulator.tech_stack(), pages),
        endpoints=endpoints,
        findings=tuple(built.values()),
    )
    return FindingCorrelator().correlate(scan)


def _scan_tech(
    from_endpoints: tuple[TechComponent, ...], pages: Iterable[Page]
) -> tuple[TechComponent, ...]:
    """Scan-level technology: endpoint observations plus anything seen in page links."""
    extra: list[TechComponent] = list(from_endpoints)
    for page in pages:
        extra.extend(
            fingerprint_response(body=" ".join(page.links), url=page.effective_url)
        )
    return merge_tech(extra)


# ---------------------------------------------------------------------------
# The built-in scanner
# ---------------------------------------------------------------------------


def run_scan(
    request: ScanRequest,
    client: HttpClient | None = None,
    on_progress: ProgressCallback | None = None,
    *,
    robots: RobotsPolicy | None = None,
    cve_lookup: Callable[[str, date | None], bool] | None = None,
    clock: Callable[[], float] | None = None,
) -> ScanOutcome:
    """Assess one authorised target with the built-in scanner.

    The first thing this does, before anything else and unconditionally, is demand the
    authorisation attestation. Everything after that is bounded by the budgets on the
    request.
    """
    progress = ProgressLog(on_progress)
    progress.emit(ScanPhase.AUTHORIZE, message=f"checking authorisation for {request.target_url}")
    require_authorization(request)
    require_target_allowed(request)

    tick = clock or time.monotonic
    started = tick()
    owns_client = client is None
    if client is None:
        budget = Budget.from_request(request, clock=tick)
        limiter = RateLimiter.from_request(request, clock=tick)
        client = HttpClient(request, budget=budget, limiter=limiter, clock=tick)

    errors: list[str] = []
    try:
        if robots is None:
            progress.emit(ScanPhase.ROBOTS, message="reading robots.txt")
            robots = fetch_robots_policy(client, request)
        if robots.fetch_error:
            errors.append(f"robots.txt: {robots.fetch_error}")

        crawler = Crawler(robots=robots, on_progress=progress)
        pages = crawler.crawl(request, client)
        progress.emit(
            ScanPhase.CRAWL,
            message=f"crawl complete: {len(pages)} page(s)",
            pages=len(pages),
            elapsed_s=client.budget.elapsed_s(),
        )

        probes: tuple[ProbeResult, ...] = ()
        if request.profile == ScanProfile.ACTIVE:
            progress.emit(ScanPhase.PROBE, message="running benign active probes", pages=len(pages))
            probes = run_active_probes(request, client, pages)
            progress.emit(
                ScanPhase.PROBE,
                message=f"{len(probes)} allowlisted probe(s) sent",
                pages=len(pages),
                elapsed_s=client.budget.elapsed_s(),
            )

        as_of = (request.scanned_at or _now()).date()
        context = CheckContext(
            target_url=request.target_url,
            pages=tuple(pages),
            probes=probes,
            as_of=as_of,
            cve_lookup=cve_lookup if cve_lookup is not None else default_cve_lookup(as_of),
            crawled_urls=frozenset(page.effective_url for page in pages),
        )
        progress.emit(ScanPhase.CHECK, message="running checks", pages=len(pages))
        found = run_checks(pages, context, request.profile)

        progress.emit(
            ScanPhase.ASSEMBLE,
            message="building scan",
            pages=len(pages),
            findings=len(found),
            elapsed_s=client.budget.elapsed_s(),
        )
        scan = build_scan(request, pages, found)
        errors.extend(crawler.stats.errors)
        errors.extend(client.errors)

        duration = max(0.0, tick() - started)
        progress.emit(
            ScanPhase.DONE,
            message=f"{len(scan.findings)} finding(s) across {len(scan.endpoints)} endpoint(s)",
            pages=len(pages),
            findings=len(scan.findings),
            elapsed_s=duration,
        )
        return ScanOutcome(
            scan=scan,
            profile=request.profile,
            tool=SCANNER_NAME,
            tool_version=__version__,
            tool_selection=(
                "the built-in vulnpriority scanner produced these findings; it is deliberately "
                "conservative and a dedicated scanner such as ZAP will find more"
            ),
            progress=progress.as_tuple(),
            pages_fetched=len(pages),
            requests_made=len(client.requests),
            bytes_downloaded=client.budget.bytes_downloaded,
            skipped_out_of_scope=crawler.stats.skipped_out_of_scope + client.out_of_scope_blocked,
            robots_blocked=crawler.stats.robots_blocked,
            robots_named=int(getattr(robots, "blocked", 0) or 0),
            skipped_dangerous_links=crawler.stats.skipped_dangerous,
            forms_recorded=crawler.stats.forms_recorded,
            probes_sent=len(probes),
            errors=tuple(dict.fromkeys(errors)),
            coverage_notes=crawler.coverage_notes,
            stopped_reason=crawler.stats.stopped_reason,
            duration_s=duration,
        )
    finally:
        if owns_client:
            client.close()


# ---------------------------------------------------------------------------
# The single high-level entry point
# ---------------------------------------------------------------------------


def assess_target(
    request: ScanRequest,
    *,
    prefer_external: bool = True,
    force_builtin: bool = False,
    client: HttpClient | None = None,
    on_progress: ProgressCallback | None = None,
    tools: Sequence[ExternalTool] | None = None,
    order: Sequence[str] | None = None,
    runner: Callable[..., object] | None = None,
    out_dir: str | Path | None = None,
    max_tool_attempts: int = 2,
    **kwargs: object,
) -> ScanOutcome:
    """Assess a target and return a :class:`ScanOutcome` carrying a canonical ``Scan``.

    This is the function the web server and the CLI should call.

    **A real scanner runs when one is installed.** ZAP and Nuclei are better web scanners
    than the one in this package, and this framework's contribution is what happens to
    findings afterwards, so the built-in crawler is the fallback rather than the default.
    The tools are ranked (see :data:`~vulnpriority.scan.adapters.TOOL_PREFERENCE`, overridable
    through ``order``), filtered to those that can honour the requested profile, and tried
    in turn; the built-in scanner runs only when none of them can or all of them fail.

    ``force_builtin=True`` (or ``prefer_external=False``) pins the built-in scanner
    explicitly - useful when the operator wants this package's conservative guarantees
    rather than a third-party tool's behaviour.

    Whichever path runs, :attr:`ScanOutcome.tool`, :attr:`ScanOutcome.tool_selection` and
    ``scan.scanner_name`` say what actually produced the findings.
    """
    require_authorization(request)
    require_target_allowed(request)

    progress = ProgressLog(on_progress)
    if force_builtin or not prefer_external:
        reason = (
            "the built-in vulnpriority scanner was requested explicitly"
            if (force_builtin or tools is not None)
            else "external scanners were disabled for this run"
        )
        progress.emit(ScanPhase.EXTERNAL, message=reason)
        outcome = run_scan(request, client=client, on_progress=progress, **kwargs)  # type: ignore[arg-type]
        return outcome.model_copy(
            update={"progress": progress.as_tuple(), "tool_selection": reason}
        )

    available = tuple(tools) if tools is not None else detect_external_tools(order=order)
    usable, skipped = select_tools(request, tools=available, order=order)
    attempts: list[str] = list(skipped)

    for note in skipped:
        progress.emit(ScanPhase.EXTERNAL, message=note)

    for tool in usable[: max(0, int(max_tool_attempts))]:
        progress.emit(
            ScanPhase.EXTERNAL,
            message=f"running {tool.name} for the {request.profile.value} profile",
        )
        try:
            report = run_external(tool, request, out_dir=out_dir, runner=runner)  # type: ignore[arg-type]
            scan = ingest_report(report)
        except ExternalToolError as error:
            note = f"{tool.name} failed: {error}"
            attempts.append(note)
            progress.emit(ScanPhase.EXTERNAL, message=note)
            continue

        version = _stamp_version(tool, scan, runner=runner)
        scan = version[0]
        selection = (
            f"{tool.name} ran this {request.profile.value} scan"
            f"{f' (version {version[1]})' if version[1] else ''}; "
            f"{TOOL_SPECS[tool.name].summary}"
        )
        # A containerised scanner may have been pointed at a rewritten host. The URLs in
        # its report then name that host, so saying so is the difference between a reader
        # trusting the report and doubting it.
        notes = tuple(
            note for note in (container_target(request.target_url)[1],)
            if note and TOOL_SPECS[tool.name].image
        )
        progress.emit(
            ScanPhase.DONE,
            message=f"{tool.name} reported {len(scan.findings)} finding(s)",
            findings=len(scan.findings),
        )
        return ScanOutcome(
            scan=scan,
            profile=request.profile,
            tool=tool.name,
            tool_version=version[1],
            tool_selection=selection,
            available_tools=tuple(item.name for item in available),
            external_attempts=tuple(attempts),
            coverage_notes=notes,
            progress=progress.as_tuple(),
            pages_fetched=len(scan.endpoints),
            report_path=Path(report),
        )

    reason = _fallback_reason(request, available, attempts)
    progress.emit(ScanPhase.EXTERNAL, message=reason)
    outcome = run_scan(request, client=client, on_progress=progress, **kwargs)  # type: ignore[arg-type]
    # ``progress`` already holds the inner scan's events: run_scan forwards to it.
    return outcome.model_copy(
        update={
            "progress": progress.as_tuple(),
            "tool_selection": reason,
            "available_tools": tuple(item.name for item in available),
            "external_attempts": tuple(attempts),
            "errors": (reason,) + outcome.errors,
        }
    )


def _stamp_version(
    tool: ExternalTool, scan: Scan, *, runner: Callable[..., object] | None = None
) -> tuple[Scan, str | None]:
    """Make sure the ``Scan`` names the version of the tool that actually produced it.

    The report is authoritative when it carries a version (ZAP and Wapiti both do). When
    it does not, the tool is asked - but only for tools whose version command reports the
    *scanner*: for a containerised tool it reports the container runtime, which must never
    be written into ``Scan.scanner_version``.
    """
    spec = TOOL_SPECS.get(tool.name)
    if scan.scanner_version:
        return scan, scan.scanner_version
    detected = tool.version or tool_version(tool, runner=runner)  # type: ignore[arg-type]
    if detected and spec is not None and spec.version_is_scanner:
        return scan.model_copy(update={"scanner_version": detected}), detected
    return scan, detected


def _fallback_reason(
    request: ScanRequest, available: Sequence[ExternalTool], attempts: Sequence[str]
) -> str:
    """Why the built-in scanner is about to run, in words an operator can act on."""
    from vulnpriority.scan.adapters import install_hint_for

    if not available:
        return (
            "No external scanner was found on PATH, so the built-in vulnpriority scanner ran. "
            "It is deliberately conservative and finds less than a dedicated scanner. "
            + install_hint_for(request.profile)
        )
    if attempts:
        return (
            "The built-in vulnpriority scanner ran because no installed tool could serve this "
            f"{request.profile.value} scan: " + "; ".join(attempts)
        )
    return (  # pragma: no cover - unreachable: tools exist and none was skipped or tried
        "The built-in vulnpriority scanner ran."
    )
