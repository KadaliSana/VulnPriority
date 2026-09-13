"""Target assessment: produce a ``Scan`` when no scanner report exists.

Everything else in vulnpriority starts from a scanner report. This package removes that
precondition: given a URL the operator is **authorised to test**, it crawls the
application, analyses what comes back, and emits a
:class:`vulnpriority.core.models.Scan` identical in shape to a parsed OWASP ZAP report - same
endpoint identifiers, same finding identifiers, same dedup keys - so enrichment, the
attack graph, the ranker and the evaluation protocol all work unchanged.

**A real scanner runs when one is installed.** :func:`~vulnpriority.scan.runner.assess_target`
prefers ZAP (natively or through the official container image), then Nuclei, then Wapiti or
Nikto, ranked by how much useful evidence each produces; the built-in crawler is the
fallback for a machine with none of them. ZAP and Nuclei are better web scanners than
anything worth writing here - this framework's contribution is what happens to the findings
afterwards. :func:`~vulnpriority.scan.adapters.scanner_environment` reports what is installed,
what each tool supports and how to install what is missing.

The safety model, in full:

* **Authorisation is mandatory.** :class:`~vulnpriority.scan.models.ScanRequest` carries
  ``authorized`` (default ``False``) and ``authorization_note``;
  :func:`~vulnpriority.scan.safety.require_authorization` raises
  :class:`~vulnpriority.scan.safety.NotAuthorizedError` otherwise. There is no bypass.
* **Passive by default.** The default profile sends no attack payloads at all: it fetches
  linked pages and reads headers, cookies, markup and transport.
* **Active means three benign probes.** A random alphanumeric reflection marker, an
  ``OPTIONS`` request, and one well-known path. The allowlist is closed
  (:data:`~vulnpriority.scan.safety.ALLOWED_PROBE_KINDS`) and enforced in code.
* **Scope is locked** to the authorised host plus explicitly listed extras, re-checked on
  every redirect hop.
* **Hard caps**: 2 requests/second, 60 pages, depth 3, 60 seconds and 1 MB per response by
  default, all configurable within bounded ranges.
* **robots.txt is respected**, the User-Agent is honest, forms are recorded but never
  submitted, logout- and delete-shaped links are never followed, credentials are never
  sent, and private, loopback and cloud-metadata addresses are refused unless
  ``allow_private_targets`` is set.

Typical use::

    from vulnpriority.scan import ScanRequest, assess_target

    request = ScanRequest(
        target_url="https://shop.example.com/",
        authorized=True,
        authorization_note="Engagement PT-2024-114, signed by the application owner",
    )
    outcome = assess_target(request)          # ZAP or nuclei if installed, else built-in
    scan = outcome.scan                       # an ordinary vulnpriority Scan
    print(outcome.tool, outcome.tool_selection)

    # What the operator could be running, and how to install it:
    from vulnpriority.scan import scanner_environment
    print(scanner_environment(request.profile).notice)
"""

from __future__ import annotations

from vulnpriority.scan.active import (
    ACTIVE_CHECKS,
    probe_options,
    probe_reflected_marker,
    probe_well_known_path,
    run_active_checks,
    run_active_probes,
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
    scanner_environment,
    select_tools,
    tool_version,
)
from vulnpriority.scan.checks import CHECKS, Check, CheckContext, checks_for_profile, run_checks
from vulnpriority.scan.crawler import Crawler, extract_forms, extract_links, visit_key
from vulnpriority.scan.http import FetchResult, HttpClient, fetch_robots_policy
from vulnpriority.scan.models import (
    DEFAULT_USER_AGENT,
    SCANNER_NAME,
    CheckFinding,
    FormField,
    FormInfo,
    Page,
    ProbeKind,
    ProbeResult,
    ScanOutcome,
    ScanPhase,
    ScanProfile,
    ScanProgress,
    ScanRequest,
    ScannerEnvironment,
    ToolStatus,
)
from vulnpriority.scan.passive import PASSIVE_CHECKS, run_passive_checks
from vulnpriority.scan.runner import assess_target, build_scan, run_scan
from vulnpriority.scan.safety import (
    ALLOWED_PROBE_KINDS,
    Budget,
    NotAuthorizedError,
    OutOfScopeError,
    ProbeNotAllowedError,
    RateLimiter,
    RobotsPolicy,
    host_in_scope,
    is_private_host,
    require_allowed_probe,
    require_authorization,
    require_target_allowed,
)

__all__ = [
    # request / result
    "ScanRequest",
    "ScanOutcome",
    "ScanProfile",
    "ScanPhase",
    "ScanProgress",
    "Page",
    "FormField",
    "FormInfo",
    "ProbeKind",
    "ProbeResult",
    "CheckFinding",
    "SCANNER_NAME",
    "DEFAULT_USER_AGENT",
    # entry points
    "assess_target",
    "run_scan",
    "build_scan",
    # safety
    "NotAuthorizedError",
    "OutOfScopeError",
    "ProbeNotAllowedError",
    "ALLOWED_PROBE_KINDS",
    "require_authorization",
    "require_target_allowed",
    "require_allowed_probe",
    "host_in_scope",
    "is_private_host",
    "RateLimiter",
    "RobotsPolicy",
    "Budget",
    # machinery
    "HttpClient",
    "FetchResult",
    "fetch_robots_policy",
    "Crawler",
    "extract_links",
    "extract_forms",
    "visit_key",
    "Check",
    "CheckContext",
    "CHECKS",
    "checks_for_profile",
    "run_checks",
    "PASSIVE_CHECKS",
    "run_passive_checks",
    "ACTIVE_CHECKS",
    "run_active_checks",
    "run_active_probes",
    "probe_reflected_marker",
    "probe_options",
    "probe_well_known_path",
    # external tools
    "ExternalTool",
    "ExternalToolError",
    "ToolStatus",
    "ScannerEnvironment",
    "TOOL_PREFERENCE",
    "TOOL_SPECS",
    "ZAP_DOCKER_IMAGE",
    "detect_external_tools",
    "describe_external_tools",
    "scanner_environment",
    "select_tools",
    "tool_version",
    "build_argv",
    "run_external",
    "ingest_report",
]
