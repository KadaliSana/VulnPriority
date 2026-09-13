"""Remediation cost, deliberately unequal (DESIGN.md 3.6, Gap 10).

Treating every finding as "one ticket" is the modelling error that makes budget-aware
evaluation meaningless: a missing security header and a broken authorisation model do
not cost the same to fix. Cost here is driven by the *class of change* the CWE implies -
a configuration flip, a dependency bump, a code change, or an architectural change - plus
a per-extra-endpoint term, because a root cause that shows up on twenty endpoints costs
more to roll out even though the framework ranks it once.
"""

from __future__ import annotations

from vulnprio.core.config import ComponentBConfig
from vulnprio.core.models import Finding, RemediationCost

__all__ = [
    "CONFIGURATION_FIX_HOURS",
    "DEPENDENCY_UPGRADE_HOURS",
    "CODE_CHANGE_HOURS",
    "ARCHITECTURAL_CHANGE_HOURS",
    "CWE_CLASS_HOURS",
    "CWE_CLASSES",
    "DEFAULT_CWE_HOURS",
    "MIN_HOURS",
    "cwe_class",
    "base_hours",
    "estimate_cost",
]

#: Change a setting, a header, a permission bit. Cheap, testable, shippable same day.
CONFIGURATION_FIX_HOURS: float = 1.5

#: Bump a library and re-run the suite. Cheap unless it is not, which is why it is above
#: a configuration fix but well below writing code.
DEPENDENCY_UPGRADE_HOURS: float = 3.0

#: Write and review application code: validation, encoding, parameterised queries.
CODE_CHANGE_HOURS: float = 8.0

#: Redesign a trust boundary: authentication, authorisation, session or access control.
ARCHITECTURAL_CHANGE_HOURS: float = 24.0

CWE_CLASS_HOURS: dict[str, float] = {
    "configuration_fix": CONFIGURATION_FIX_HOURS,
    "dependency_upgrade": DEPENDENCY_UPGRADE_HOURS,
    "code_change": CODE_CHANGE_HOURS,
    "architectural_change": ARCHITECTURAL_CHANGE_HOURS,
}

#: CWE ids grouped by the class of change they demand. Not exhaustive - unlisted CWEs
#: fall back to ``ComponentBConfig.remediation_default_hours``.
CWE_CLASSES: dict[str, tuple[int, ...]] = {
    "configuration_fix": (
        16,    # configuration
        200,   # exposure of sensitive information
        209,   # error message information leak
        319,   # cleartext transmission
        523,   # unprotected credentials transport
        525,   # web browser cache containing sensitive information
        548,   # directory listing
        614,   # sensitive cookie without Secure
        693,   # protection mechanism failure
        1004,  # sensitive cookie without HttpOnly
        1021,  # improper restriction of rendered UI layers (clickjacking)
        1275,  # sensitive cookie with improper SameSite
    ),
    "dependency_upgrade": (
        829,   # inclusion of functionality from untrusted control sphere
        937,   # using components with known vulnerabilities (OWASP A9)
        1035,  # using components with known vulnerabilities (2017 A9)
        1104,  # use of unmaintained third party components
        1395,  # dependency on vulnerable third-party component
    ),
    "code_change": (
        20,    # improper input validation
        22,    # path traversal
        77,    # command injection
        78,    # OS command injection
        79,    # cross-site scripting
        89,    # SQL injection
        90,    # LDAP injection
        91,    # XML injection
        94,    # code injection
        113,   # HTTP response splitting
        352,   # cross-site request forgery
        434,   # unrestricted file upload
        502,   # deserialization of untrusted data
        601,   # open redirect
        611,   # XML external entity
        643,   # XPath injection
        917,   # expression language injection
        918,   # server-side request forgery
    ),
    "architectural_change": (
        250,   # execution with unnecessary privileges
        269,   # improper privilege management
        284,   # improper access control
        285,   # improper authorization
        287,   # improper authentication
        288,   # authentication bypass using alternate path
        306,   # missing authentication for critical function
        384,   # session fixation
        566,   # authorization bypass through user-controlled key
        613,   # insufficient session expiration
        639,   # authorization bypass through user-controlled key
        807,   # reliance on untrusted inputs in a security decision
        862,   # missing authorization
        863,   # incorrect authorization
    ),
}

#: Flattened ``cwe_id -> base hours`` table. Public because the selection layer and the
#: report both want to show what a fix was assumed to cost.
DEFAULT_CWE_HOURS: dict[int, float] = {
    cwe: CWE_CLASS_HOURS[class_name]
    for class_name, cwes in CWE_CLASSES.items()
    for cwe in cwes
}

#: ``RemediationCost.hours`` is ``gt=0``; nothing is ever free to ship.
MIN_HOURS: float = 0.25

_CWE_TO_CLASS: dict[int, str] = {
    cwe: class_name for class_name, cwes in CWE_CLASSES.items() for cwe in cwes
}


def cwe_class(cwe_id: int | None) -> str:
    """Class of change a CWE implies, or ``"unclassified"`` when it is not in the table."""
    if cwe_id is None:
        return "unclassified"
    return _CWE_TO_CLASS.get(int(cwe_id), "unclassified")


def base_hours(cwe_id: int | None, config: ComponentBConfig) -> float:
    """Base hours for one instance of this CWE, operator overrides winning."""
    if cwe_id is not None:
        override = config.remediation_hours_by_cwe.get(int(cwe_id))
        if override is not None:
            return float(override)
        default = DEFAULT_CWE_HOURS.get(int(cwe_id))
        if default is not None:
            return float(default)
    return float(config.remediation_default_hours)


def estimate_cost(finding: Finding, config: ComponentBConfig) -> RemediationCost:
    """Hours and money to remediate the root cause behind this finding.

    ``cluster_size`` is the number of findings sharing the finding's ``dedup_key``, so the
    rollout term is charged once per root cause rather than once per alert - which is what
    makes the knapsack in :mod:`vulnprio.select` a fair budget model.
    """
    base = base_hours(finding.cwe_id, config)
    extra_endpoints = max(0, int(finding.cluster_size) - 1)
    rollout = float(config.remediation_hours_per_extra_endpoint) * extra_endpoints
    hours = max(base + rollout, MIN_HOURS)
    cost = hours * float(config.remediation_hourly_rate)

    class_name = cwe_class(finding.cwe_id)
    basis = (
        f"{class_name} base {base:g}h"
        f" + {rollout:g}h rollout across {extra_endpoints} extra endpoint(s)"
        f" @ {config.remediation_hourly_rate:g}/h"
    )

    return RemediationCost(
        finding_id=finding.finding_id,
        hours=hours,
        cost=cost,
        basis=basis,
    )
