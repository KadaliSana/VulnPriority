"""Same-origin, depth-limited, breadth-first crawl.

The crawler's job is to produce the set of pages the checks reason about, and to do it
without ever doing anything the operator did not consent to. Concretely:

* **Scope.** A link off the authorised host is dropped and counted before it is queued, so
  the out-of-scope URL is never requested at all (:mod:`vulnpriority.scan.safety` re-checks it
  at the client, and again on every redirect hop).
* **Forms are recorded, never submitted.** A form tells the checks a great deal - its
  method, its fields, whether it carries a CSRF token, whether it posts a password over
  plaintext HTTP - and none of that requires sending it. Submitting forms is how a
  "read-only" scan creates orders and deletes rows.
* **Dangerous-looking links are not followed.** Anything that reads like a logout, a
  delete, a reset or a revoke is dropped even when it is in scope. The scanner is not
  clever enough to know that ``/orders/17/delete`` is only a confirmation page, so it does
  not guess.
* **Deduplication by templated path.** Visits are keyed on method, host, the
  :func:`~vulnpriority.ingest.normalize.template_path` form of the path and the sorted query
  parameter *names*, reusing the ingest layer's templating so ``/users/1`` and
  ``/users/2`` are one route here exactly as they are one endpoint downstream.

Link extraction is done on text with regular expressions. No DOM is built, no script is
evaluated and no subresource is fetched as a side effect of parsing.

**Single-page applications.** An application whose HTML is a mount element and three
bundles offers a link crawler nothing to follow, and the crawl ends after the shell and
its assets - a handful of URLs on an application with fifty endpoints. That is not a
clean result, it is an unreported blind spot. So when a fetched resource is JavaScript,
:mod:`vulnpriority.scan.spa` mines the request paths the client code names in string
literals and this crawler queues them *behind* everything link-reachable, where they
consume only budget that would otherwise have gone unused. A mined path is an ordinary
candidate from that point on: scope, robots, the dangerous-link refusal and the budgets
all apply to it unchanged. What the crawl still could not reach is recorded as a
coverage note rather than left to be misread as absence of surface.
"""

from __future__ import annotations

import hashlib
import re
from collections import deque
from typing import Callable, Iterable
from urllib.parse import urljoin, urlsplit

from vulnpriority.core.enums import HttpMethod
from vulnpriority.ingest.normalize import (
    canonical_url,
    http_method,
    query_parameters,
    template_path,
    url_host,
    url_path,
)
from vulnpriority.scan.http import HttpClient
from vulnpriority.scan.models import (
    FormField,
    FormInfo,
    Page,
    ScanPhase,
    ScanProgress,
    ScanRequest,
)
from vulnpriority.scan.safety import OutOfScopeError, RobotsPolicy, host_in_scope, is_http_url
from vulnpriority.scan.spa import (
    SpaVerdict,
    coverage_note,
    detect_spa,
    directory_ancestors,
    extract_script_paths,
    extract_view_routes,
)

__all__ = [
    "Crawler",
    "CrawlStats",
    "DANGEROUS_LINK",
    "extract_links",
    "extract_forms",
    "visit_key",
    "strip_scripts",
]

_SCRIPTISH = re.compile(
    r"(?is)<(script|style|template|noscript|svg)\b[^>]*>.*?</\1\s*>"
)
_HREF = re.compile(r"""(?is)<a\b[^>]*?\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")
_SRC = re.compile(
    r"""(?is)<(?:script|img|iframe|link|source|embed)\b[^>]*?\b(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))"""
)
_FORM = re.compile(r"(?is)<form\b(?P<attrs>[^>]*)>(?P<body>.*?)</form\s*>")
_FORM_OPEN = re.compile(r"(?is)<form\b(?P<attrs>[^>]*)>")
_INPUT = re.compile(r"(?is)<(?:input|select|textarea|button)\b(?P<attrs>[^>]*)>")
_ATTR = re.compile(r"""(?is)([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")

#: Link shapes this scanner refuses to follow even inside the authorised scope. A GET
#: request is only safe in theory; in practice applications hang destructive actions off
#: links, and a crawler that follows them is a vandal with good intentions.
DANGEROUS_LINK = re.compile(
    r"(?i)(?:^|[/?&#=_.-])(?:log[-_]?out|sign[-_]?out|log[-_]?off|disconnect|"
    r"delete|destroy|remove|drop|purge|truncate|wipe|erase|"
    r"deactivate|disable|revoke|reset|restore|rollback|shutdown|reboot|"
    r"unsubscribe|cancel|refund|empty[-_]?cart|clear[-_]?cache)(?:[/?&#=_.-]|$)"
)

#: Extensions that are fetched (they carry version evidence and mixed-content signals) but
#: never treated as a source of further links.
_LEAF_EXTENSIONS = (
    ".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".pdf", ".zip", ".gz", ".tar", ".mp4", ".mp3",
)


def strip_scripts(markup: str) -> str:
    """Remove script/style/template blocks *as text* before link extraction."""
    return _SCRIPTISH.sub(" ", markup or "")


def _fingerprint(body: str) -> str:
    """Identity of a response body, for recognising a single-page application's shell.

    Whitespace is collapsed before hashing so that a template rendered with different
    indentation still compares equal, and nothing else is normalised: two responses that
    differ in a single character are two responses.
    """
    if not body:
        return ""
    return hashlib.sha1(" ".join(body.split()).encode("utf-8", "replace")).hexdigest()


def _attributes(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in _ATTR.finditer(raw or ""):
        name = match.group(1).lower()
        value = match.group(2) or match.group(3) or match.group(4) or ""
        out.setdefault(name, value.strip())
    return out


def _first_group(match: re.Match[str]) -> str:
    return (match.group(1) or match.group(2) or match.group(3) or "").strip()


def extract_links(markup: str, base_url: str, *, include_assets: bool = True) -> tuple[str, ...]:
    """Absolute URLs referenced by anchors, and optionally by subresource attributes.

    Fragments, ``mailto:``, ``javascript:``, ``tel:`` and ``data:`` are dropped: none of
    them is a request this scanner can or should make.
    """
    if not markup:
        return ()
    # Anchors and form actions are read from the script-stripped text, so a URL inside a
    # JavaScript string is not mistaken for a link. Subresource references are read from
    # the original markup, because ``<script src=...>`` is itself a script element and
    # stripping script blocks would discard exactly the versioned-library evidence that
    # the outdated-component check depends on.
    text = strip_scripts(markup)
    raw: list[str] = [_first_group(match) for match in _HREF.finditer(text)]
    if include_assets:
        raw.extend(_first_group(match) for match in _SRC.finditer(markup))
    for match in _FORM_OPEN.finditer(text):
        action = _attributes(match.group("attrs")).get("action")
        if action:
            raw.append(action)

    seen: list[str] = []
    for candidate in raw:
        value = (candidate or "").strip()
        if not value or value.startswith("#"):
            continue
        lowered = value.lower()
        if lowered.startswith(("mailto:", "javascript:", "tel:", "data:", "sms:", "ftp:", "file:")):
            continue
        absolute = canonical_url(urljoin(base_url, value))
        if not is_http_url(absolute):
            continue
        if absolute not in seen:
            seen.append(absolute)
    return tuple(seen)


def extract_forms(markup: str, base_url: str) -> tuple[FormInfo, ...]:
    """Every form on the page, with its fields. Recorded only - nothing is submitted."""
    if not markup:
        return ()
    forms: list[FormInfo] = []
    for match in _FORM.finditer(markup):
        attrs = _attributes(match.group("attrs"))
        action = canonical_url(urljoin(base_url, attrs.get("action") or base_url))
        fields: list[FormField] = []
        for field_match in _INPUT.finditer(match.group("body")):
            field_attrs = _attributes(field_match.group("attrs"))
            fields.append(
                FormField(
                    name=field_attrs.get("name", ""),
                    type=(field_attrs.get("type") or "text").lower(),
                    value=field_attrs.get("value", ""),
                    autocomplete=field_attrs.get("autocomplete"),
                )
            )
        forms.append(
            FormInfo(
                action=action,
                method=http_method(attrs.get("method", "GET")),
                fields=tuple(fields),
                enctype=attrs.get("enctype"),
                autocomplete=attrs.get("autocomplete"),
                source_url=base_url,
            )
        )
    return tuple(forms)


def visit_key(url: str, method: HttpMethod = HttpMethod.GET) -> str:
    """Identity of a route for crawl deduplication.

    Method, host, templated path and sorted query parameter names - the same templating
    the ingest layer applies, so the crawl collapses instance URLs exactly the way the
    endpoint identifiers downstream will.
    """
    canonical = canonical_url(url)
    params = ",".join(query_parameters(canonical))
    return f"{method.value} {url_host(canonical)}{template_path(url_path(canonical))}?{params}"


class CrawlStats:
    """The crawler's ledger, including everything it declined to do."""

    def __init__(self) -> None:
        self.pages = 0
        self.requests = 0
        self.skipped_out_of_scope = 0
        self.robots_blocked = 0
        self.skipped_dangerous = 0
        self.skipped_duplicate = 0
        self.forms_recorded = 0
        #: Paths recovered from JavaScript bundles and queued.
        self.mined_from_scripts = 0
        #: Requests answered with the application's own shell rather than with a distinct
        #: resource. A single-page application does this for every unknown path, so a high
        #: count means the mined list contained client-side view routes, not endpoints.
        self.shell_routes = 0
        #: How many of those were reached before a budget ran out. The gap between the
        #: two numbers is the honest size of what this scan did not look at.
        self.mined_fetched = 0
        self.errors: list[str] = []
        self.stopped_reason: str | None = None


class Crawler:
    """Breadth-first crawl of one authorised origin."""

    def __init__(
        self,
        *,
        robots: RobotsPolicy | None = None,
        on_progress: Callable[[ScanProgress], None] | None = None,
        include_assets: bool = True,
        mine_scripts: bool = True,
        script_path_limit: int = 150,
    ) -> None:
        self.robots = robots
        self.on_progress = on_progress
        self.include_assets = bool(include_assets)
        #: Recover request paths from served JavaScript. On by default: without it a
        #: single-page application is scanned as though it were its own loading screen.
        self.mine_scripts = bool(mine_scripts)
        self.script_path_limit = max(0, int(script_path_limit))
        self.stats = CrawlStats()
        #: What the shell page turned out to be, filled in once the seed is fetched.
        self.spa = SpaVerdict()

    # -- helpers --------------------------------------------------------------

    def _emit(self, phase: ScanPhase, client: HttpClient, message: str) -> None:
        if self.on_progress is None:
            return
        self.on_progress(
            ScanProgress(
                phase=phase,
                pages_fetched=self.stats.pages,
                elapsed_s=client.budget.elapsed_s(),
                message=message,
            )
        )

    @staticmethod
    def _is_leaf(url: str) -> bool:
        path = url_path(url).lower()
        return path.endswith(_LEAF_EXTENSIONS)

    @staticmethod
    def _is_script(page: Page) -> bool:
        content_type = (page.content_type or "").lower()
        if "javascript" in content_type or "ecmascript" in content_type:
            return True
        # A bundle served as ``text/plain`` or with no type at all is still a bundle;
        # the extension decides when the header declines to.
        if content_type.startswith(("text/html", "image/", "font/")):
            return False
        return url_path(page.effective_url).lower().endswith((".js", ".mjs", ".cjs"))

    def _admissible(
        self, url: str, request: ScanRequest, robots: RobotsPolicy, *, is_seed: bool = False
    ) -> bool:
        """Scope, robots and danger checks - all three before anything is queued.

        The seed URL is exempt from the dangerous-link heuristic only: the operator named
        it explicitly, so refusing it would be second-guessing an authorised instruction.
        Scope and robots still apply to it.
        """
        if not host_in_scope(url, request):
            self.stats.skipped_out_of_scope += 1
            return False
        target = urlsplit(url)
        probe = target.path + (("?" + target.query) if target.query else "")
        if not is_seed and DANGEROUS_LINK.search(probe):
            self.stats.skipped_dangerous += 1
            return False
        if not robots.allows(url, request.user_agent):
            self.stats.robots_blocked += 1
            return False
        return True

    # -- the crawl ------------------------------------------------------------

    def crawl(self, request: ScanRequest, client: HttpClient) -> list[Page]:
        """Fetch up to the configured budget of pages, breadth-first from the target."""
        robots = self.robots
        if robots is None:
            from vulnpriority.scan.http import fetch_robots_policy

            robots = fetch_robots_policy(client, request)
            self.robots = robots

        start = canonical_url(request.target_url)
        pages: list[Page] = []
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        # Paths recovered from JavaScript wait here until every link-reachable URL has
        # been visited, so mining can only ever spend budget that link-following left
        # over. On an ordinary site with plenty of anchors this queue is never drained
        # and the crawl behaves exactly as it did before.
        deferred: deque[tuple[str, int]] = deque()
        seen: set[str] = {visit_key(start)}
        mined_keys: set[str] = set()
        mined_urls: dict[str, None] = {}
        view_routes: dict[str, None] = {}
        shell_fingerprint = ""
        shell_routes = 0

        while queue or deferred:
            reason = client.budget.reason()
            if reason is not None:
                self.stats.stopped_reason = reason
                self._emit(ScanPhase.CRAWL, client, f"stopping: {reason}")
                break

            if not queue:
                self._emit(
                    ScanPhase.CRAWL,
                    client,
                    f"link-reachable pages exhausted; trying {len(deferred)} path(s) "
                    "recovered from JavaScript",
                )
                queue, deferred = deferred, deque()

            url, depth = queue.popleft()
            if not self._admissible(url, request, robots, is_seed=(url == start)):
                continue

            try:
                result = client.fetch(url, HttpMethod.GET)
            except OutOfScopeError as error:  # pragma: no cover - _admissible already filtered
                self.stats.skipped_out_of_scope += 1
                self.stats.errors.append(str(error))
                continue
            self.stats.requests += 1
            if result.error is not None and result.status is None:
                self.stats.errors.append(f"{url}: {result.error}")
                self._emit(ScanPhase.CRAWL, client, f"error on {url}: {result.error}")
                continue
            if result.error is not None:
                # A redirect stopped at the scope boundary still yields a usable response;
                # the client has already counted the hop it refused to follow.
                self.stats.errors.append(f"{url}: {result.error}")

            page = result.to_page(depth=depth)
            body_base = page.effective_url
            links = (
                extract_links(page.body, body_base, include_assets=self.include_assets)
                if page.is_html or not self._is_leaf(body_base)
                else ()
            )
            forms = extract_forms(page.body, body_base) if page.is_html else ()
            # A single-page application answers every unknown path with its shell. Those
            # responses are not new pages, and following their links is actively harmful:
            # the shell's relative <script src="main.js"> resolves against whatever path
            # asked for it, so one mined route breeds three fake endpoints
            # (/address/edit/main.js and friends), and a few dozen of them exhaust the page
            # budget before the crawl reaches the API paths that were the point of mining.
            body_fingerprint = _fingerprint(page.body) if page.is_html else ""
            if bool(shell_fingerprint) and body_fingerprint == shell_fingerprint and url != start:
                # Byte-for-byte the page already analysed. Recording it again would add an
                # endpoint that does not exist, repeat every header finding the seed
                # already produced, and spend a page of budget that a real endpoint needs.
                # It is counted, and the count is reported, because "forty routes are
                # client-side views" is a fact about the application worth stating.
                shell_routes += 1
                if visit_key(url) in mined_keys:
                    self.stats.mined_fetched += 1
                self._emit(ScanPhase.CRAWL, client, f"{url} returned the application shell")
                continue

            page = page.model_copy(update={"links": links, "forms": forms})
            pages.append(page)
            self.stats.pages += 1
            self.stats.forms_recorded += len(forms)
            if visit_key(url) in mined_keys:
                self.stats.mined_fetched += 1
            client.budget.note_page()
            self._emit(ScanPhase.CRAWL, client, f"fetched {url} ({page.status})")

            if url == start:
                self.spa = detect_spa(page)
                if self.spa.is_spa:
                    shell_fingerprint = body_fingerprint
                    self._emit(
                        ScanPhase.CRAWL,
                        client,
                        f"{self.spa.framework} single-page application: routes are in "
                        "JavaScript, so bundles will be mined for request paths",
                    )

            if self.mine_scripts and self.script_path_limit and self._is_script(page):
                for route in extract_view_routes(page.body):
                    view_routes.setdefault(route, None)
                remaining = self.script_path_limit - self.stats.mined_from_scripts
                candidates = (
                    extract_script_paths(page.body, body_base, limit=remaining)
                    if remaining > 0
                    else ()
                )
                # Directories the application demonstrably serves, computed from the wider
                # list that includes static files: ``/ftp/order_4711.pdf`` is not worth
                # fetching and is the only evidence that ``/ftp/`` exists. Queued behind
                # the literal paths, because an auto-index is a bonus and an endpoint is
                # not.
                referenced = (
                    extract_script_paths(page.body, body_base, limit=remaining * 2, assets=True)
                    if remaining > 0
                    else ()
                )
                for candidate in (*candidates, *directory_ancestors(referenced)):
                    key = visit_key(candidate)
                    if key in seen:
                        self.stats.skipped_duplicate += 1
                        continue
                    if not host_in_scope(candidate, request):
                        self.stats.skipped_out_of_scope += 1
                        continue
                    if DANGEROUS_LINK.search(urlsplit(candidate).path):
                        self.stats.skipped_dangerous += 1
                        continue
                    seen.add(key)
                    mined_keys.add(key)
                    mined_urls.setdefault(candidate, None)
                    self.stats.mined_from_scripts += 1
                    deferred.append((candidate, depth + 1))
                if candidates:
                    self._emit(
                        ScanPhase.CRAWL,
                        client,
                        f"{self.stats.mined_from_scripts} request path(s) recovered from "
                        f"{url_path(body_base)}",
                    )

            if depth >= request.max_depth:
                continue
            for link in links:
                if self._is_leaf(link) and not self.include_assets:
                    continue
                key = visit_key(link)
                if key in seen:
                    self.stats.skipped_duplicate += 1
                    continue
                if not host_in_scope(link, request):
                    # Counted here so an off-scope link is visible even when it is never
                    # reached because a budget ran out first.
                    self.stats.skipped_out_of_scope += 1
                    continue
                seen.add(key)
                queue.append((link, depth + 1))

        if self.stats.stopped_reason is None:
            self.stats.stopped_reason = client.budget.reason()
        self.stats.shell_routes = shell_routes
        self.spa = self.spa.model_copy(
            update={
                "view_routes": tuple(view_routes),
                "script_paths": tuple(mined_urls),
            }
        )
        return pages

    @property
    def coverage_notes(self) -> tuple[str, ...]:
        """What this crawl could not reach, stated rather than left to be inferred."""
        notes: list[str] = []
        note = coverage_note(
            self.spa,
            pages_fetched=self.stats.pages,
            mined_fetched=self.stats.mined_fetched,
        )
        if note:
            notes.append(note)
        unreached = self.stats.mined_from_scripts - self.stats.mined_fetched
        if unreached > 0:
            why = (
                f"a scan budget ran out first ({self.stats.stopped_reason}); raise "
                "max_pages or time_budget_s to cover them"
                if self.stats.stopped_reason
                else "robots.txt disallowed them, or the server did not answer"
            )
            notes.append(
                f"{unreached} request path(s) recovered from JavaScript were not fetched: {why}."
            )

        if self.stats.shell_routes:
            notes.append(
                f"{self.stats.shell_routes} mined path(s) were answered with the "
                "application's own shell rather than with a distinct response, so they are "
                "client-side views rather than server endpoints and carry no separate "
                "server-side surface."
            )

        blocked = getattr(self.robots, "blocked_paths", []) if self.robots else []
        if blocked:
            shown = ", ".join(blocked[:8])
            if len(blocked) > 8:
                shown += f" and {len(blocked) - 8} more"
            if getattr(self.robots, "binding", False):
                notes.append(
                    f"robots.txt disallowed {len(blocked)} path(s) and this scan honoured that, "
                    f"so they are unassessed rather than clean: {shown}. A Disallow entry is "
                    "itself a disclosure - it names what the operator would rather nobody looked "
                    "at - and an attacker will not honour it. Re-run with respect_robots off, "
                    "under the same authorisation, to cover them."
                )
            else:
                notes.append(
                    f"robots.txt asks crawlers away from {len(blocked)} path(s), which this "
                    f"authorised scan covered anyway: {shown}. Read these first: the file is not "
                    "an access control, and it is the operator's own list of what they would "
                    "rather nobody found."
                )
        return tuple(notes)

    def all_forms(self, pages: Iterable[Page]) -> tuple[FormInfo, ...]:
        """Every form seen across the crawl, de-duplicated by action and method."""
        seen: dict[tuple[str, str], FormInfo] = {}
        for page in pages:
            for form in page.forms:
                seen.setdefault((form.action, form.method.value), form)
        return tuple(seen.values())
