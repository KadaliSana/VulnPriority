"""The passive profile: observation only.

This is the default profile and the one that should be used unless there is a reason not
to. A passive assessment issues ordinary ``GET`` requests for pages the application links
to, reads the responses, and draws conclusions from headers, cookies, markup and
transport. It is indistinguishable from a polite crawler in the target's access log, and
it cannot change anything on the target because it never sends anything a browser
following a link would not send.

Specifically, the passive profile:

* sends **no attack payloads of any kind** - no SQL, no shell metacharacters, no markup,
  no traversal sequences, no template expressions;
* sends **no credentials** and attempts **no authentication**;
* **never submits a form** and never issues a state-changing verb;
* **brute-forces nothing and uses no wordlist** - every URL requested was named by the
  application itself, whether in a link, a form action, or a string literal in the
  JavaScript it served (plus ``/robots.txt``, which exists to be read by crawlers). The one
  inference is the containing directory of a path the application named, which is how an
  auto-generated index is found at all, since nothing ever links to one;
* stays on the authorised host, under the configured rate, page, depth, byte and time
  budgets.

Everything it concludes therefore comes from evidence the application volunteered.
``robots.txt`` is read for what it discloses rather than obeyed as a boundary - it is a
crawler convention, not an access control - and the paths it names are reported.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from vulnprio.scan.checks import CHECKS, Check, CheckContext, run_checks
from vulnprio.scan.models import CheckFinding, Page, ScanProfile

__all__ = ["PASSIVE_CHECKS", "passive_check_ids", "run_passive_checks"]


def _passive() -> tuple[Check, ...]:
    return tuple(check for check in CHECKS.values() if check.profile == ScanProfile.PASSIVE)


#: Every check that runs in the passive profile, in registration order.
PASSIVE_CHECKS: tuple[Check, ...] = _passive()


def passive_check_ids() -> tuple[str, ...]:
    """Ids of the passive checks, recomputed from the registry."""
    return tuple(check.id for check in _passive())


def run_passive_checks(
    pages: Iterable[Page],
    context: CheckContext,
    *,
    checks: Sequence[Check] | None = None,
) -> list[CheckFinding]:
    """Run the passive check set over already-fetched pages. Touches no network."""
    return run_checks(pages, context, ScanProfile.PASSIVE, checks=checks or _passive())
