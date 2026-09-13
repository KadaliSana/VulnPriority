"""Single-page applications: detecting them, and recovering the surface they hide.

A link-following crawler assumes the application tells it where it can go. A
single-page application does not. Its HTML is a shell - one mount element, a handful
of bundle references, and no anchors at all - and every route, every form and every
API call is assembled by JavaScript at run time. Point a link crawler at one and it
fetches the shell, fetches the bundles, finds nothing to follow and stops, having
seen perhaps six URLs on an application with fifty endpoints. The scan is not wrong;
it is *thin*, and worse, it is thin without saying so. A short queue then looks like
a clean application.

This module fixes both halves of that.

**Detection.** :func:`detect_spa` reads the shell for what it is: a framework mount
element, few or no anchors, and script bundles doing the work. That verdict becomes a
coverage note on the outcome, so a six-page scan explains itself instead of implying
that six pages was all there was.

**Recovery.** :func:`extract_script_paths` mines same-origin request paths out of the
JavaScript the application itself served. Client code has to name the endpoints it
calls, and it names them in string literals: ``/rest/user/login``,
``/api/Products``. Those literals are evidence of server-side surface in exactly the
way an anchor is, and they are read from bytes the target volunteered - nothing is
guessed, brute-forced or taken from a wordlist. A mined path is then subject to every
control an anchor is subject to: scope, robots, the dangerous-link refusal, the
request and time budgets.

What is deliberately *not* mined:

* **Client-side view routes.** A framework route table (Angular's ``path:`` entries,
  say) describes navigation inside the browser. Under hash routing the server never
  sees the fragment at all, and under history routing every one of those paths returns
  the same shell. Fetching them would inflate the page count without adding a single
  distinct response, which is the metric-flattering move this module exists to avoid.
  :func:`extract_view_routes` extracts them anyway - for the coverage note, so the
  operator can see the size of the application the scanner is looking at.
* **Interpolated paths.** ``/rest/basket/${id}`` is truncated at the placeholder to
  ``/rest/basket/``. The prefix is kept because it is a real route; the fabricated
  identifier is not, because inventing object ids is how a read-only scan starts
  touching other people's rows.
* **Anything that fails the danger heuristic**, checked here as well as in the crawler,
  so a mined ``/rest/user/delete`` is dropped at the source.
"""

from __future__ import annotations

import re
from typing import Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit

from vulnprio.core.models import Frozen
from vulnprio.scan.models import Page

__all__ = [
    "SpaVerdict",
    "MOUNT_ELEMENTS",
    "detect_spa",
    "extract_script_paths",
    "extract_view_routes",
    "directory_ancestors",
    "coverage_note",
]

#: Mount points that mean "a framework renders everything below here". The element name
#: or id is the only stable signature a built bundle leaves in the served HTML.
MOUNT_ELEMENTS: tuple[tuple[str, str], ...] = (
    (r"(?i)<app-root\b", "Angular"),
    (r"(?i)<div\b[^>]*\bid\s*=\s*[\"']?root[\"']?", "React"),
    (r"(?i)<div\b[^>]*\bid\s*=\s*[\"']?app[\"']?", "Vue"),
    (r"(?i)<div\b[^>]*\bid\s*=\s*[\"']?__next[\"']?", "Next.js"),
    (r"(?i)<div\b[^>]*\bid\s*=\s*[\"']?svelte[\"']?", "Svelte"),
    (r"(?i)<\w+\b[^>]*\bng-app\b", "AngularJS"),
    (r"(?i)\bdata-reactroot\b", "React"),
    (r"(?i)<div\b[^>]*\bid\s*=\s*[\"']?ember-?app[\"']?", "Ember"),
)

_ANCHOR = re.compile(r"(?is)<a\b[^>]*?\bhref\s*=\s*[\"']?(?!#)[^\s>\"']")
_SCRIPT_SRC = re.compile(r"(?is)<script\b[^>]*?\bsrc\s*=\s*[\"']?([^\s>\"']+)")

#: An absolute path written as its own string literal.
#:
#: The delimiters are matched as zero-width lookaround rather than consumed, which matters
#: more than it sounds. Pairing quotes off by scanning left to right is defeated by a
#: single unbalanced quote character - an apostrophe in a message, a backtick in a comment,
#: a quote inside a regular expression literal - after which every subsequent "literal" is
#: the *gap between* two real ones and the paths inside them are never seen. Minified
#: bundles contain thousands of each quote character and are nearly guaranteed to contain
#: such a flip. Anchoring on the boundaries instead makes each match independent of every
#: other, so one malformed region costs one path rather than the rest of the file.
#:
#: The closing boundary admits ``$``, ``?`` and ``#`` as well as a quote, because an
#: interpolation, a query string or a fragment ends the path without ending the literal.
_PATH_LITERAL = re.compile(
    r"""(?<=['"`])(?P<value>/[A-Za-z0-9/_.~:@%-]{1,120})(?=['"`$?#])"""
)

#: A path written after the application's own origin: ``` `${this.hostServer}/rest/x` ```.
#: The interpolation is a host, so the constant remainder is the path and is kept.
#:
#: This is a separate pattern rather than a ``}`` added to the opening boundary above, and
#: the distinction is the whole point: ``}`` only introduces a path when the ``${`` that
#: opened it began the literal. In ``` `/rest/basket/${id}/items` ``` the same ``}`` is
#: mid-path, and treating ``/items`` as a route would invent an endpoint that does not
#: exist - the fabrication this module exists to avoid. Requiring the quote immediately
#: before ``${`` separates the two cases exactly.
_ORIGIN_PREFIXED = re.compile(
    r"""['"`]\$\{[^{}'"`]{0,80}\}(?P<value>/[A-Za-z0-9/_.~:@%-]{1,120})(?=['"`$?#])"""
)

#: A framework route-table entry, in each of the three quotings bundlers emit.
_ROUTE_ENTRY = re.compile(r"""(?is)\bpath\s*:\s*(['"`])(?P<value>[^'"`]{0,80})\1""")

#: Extensions that are already reached as subresources and carry no further surface.
_ASSET_SUFFIXES = (
    ".js", ".mjs", ".cjs", ".css", ".map", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".webp", ".avif", ".woff", ".woff2", ".ttf", ".otf", ".eot", ".mp4",
    ".mp3", ".wav", ".webm", ".pdf", ".zip", ".gz",
)

#: Characters that mean the literal was never a URL: markup, code, formatting.
_NOT_A_PATH = re.compile(r"""[\s<>\\{}()\[\]'"`;,|^*!=+]""")

#: Where a template literal stops being a constant. Everything from here on was
#: computed at run time and is not ours to invent.
_INTERPOLATION = re.compile(r"\$\{|%s\b|:[A-Za-z_]")

_MIN_PATH_LENGTH = 2
_MAX_PATH_LENGTH = 120


class SpaVerdict(Frozen):
    """What the shell page turned out to be, and why."""

    is_spa: bool = False
    framework: str = ""
    anchors: int = 0
    scripts: int = 0
    #: Client-side view routes read out of the bundles, if any were mined.
    view_routes: tuple[str, ...] = ()
    #: Same-origin request paths mined out of the bundles.
    script_paths: tuple[str, ...] = ()

    @property
    def evidence(self) -> str:
        if not self.is_spa:
            return ""
        return (
            f"{self.framework} shell: {self.anchors} anchor(s), {self.scripts} script bundle(s)"
        )


def detect_spa(page: Page, *, anchor_threshold: int = 3) -> SpaVerdict:
    """Decide whether ``page`` is an application shell rather than a document.

    Three signals have to agree: a framework mount element, at most
    ``anchor_threshold`` real anchors, and at least one script bundle. A server-rendered
    page that happens to hydrate a React island has plenty of anchors and is not a
    shell; a static page with no scripts cannot be one either.
    """
    if not page.is_html or not page.body:
        return SpaVerdict()

    body = page.body
    framework = ""
    for pattern, name in MOUNT_ELEMENTS:
        if re.search(pattern, body):
            framework = name
            break

    anchors = len(_ANCHOR.findall(body))
    scripts = len(_SCRIPT_SRC.findall(body))
    is_spa = bool(framework) and anchors <= anchor_threshold and scripts >= 1
    return SpaVerdict(is_spa=is_spa, framework=framework or "", anchors=anchors, scripts=scripts)


def _plausible_path(value: str, *, assets: bool = False) -> str | None:
    """Reduce one string literal to a same-origin request path, or reject it.

    Returns the path, or ``None`` if the literal was never one. ``assets`` keeps paths
    ending in a static-file extension, which are worthless as endpoints but do prove the
    directory around them is served - see :func:`directory_ancestors`.
    """
    text = (value or "").strip()
    if not text.startswith("/") or text.startswith("//"):
        # A protocol-relative or absolute URL points off-origin, and a relative
        # fragment cannot be resolved without knowing which module emitted it.
        return None

    # Stop at the first interpolation: the constant prefix is a route, the rest was a
    # run-time value and inventing one would mean addressing objects we were not shown.
    cut = _INTERPOLATION.search(text)
    if cut is not None:
        text = text[: cut.start()]
    text = text.split("#", 1)[0]

    if not (_MIN_PATH_LENGTH <= len(text) <= _MAX_PATH_LENGTH):
        return None
    if _NOT_A_PATH.search(text):
        return None
    if not re.search(r"[A-Za-z]", text):
        return None
    if not assets and text.lower().endswith(_ASSET_SUFFIXES):
        return None
    # ``/**``, ``/:id``, ``/*`` and similar are route *patterns*, not addresses.
    if any(ch in text for ch in "*?&"):
        return None
    return text


def extract_script_paths(
    source: str,
    base_url: str,
    *,
    limit: int = 80,
    assets: bool = False,
) -> tuple[str, ...]:
    """Same-origin URLs named as string literals in one JavaScript source.

    The application's own client code is the authority on which endpoints exist. This
    reads that list rather than guessing at one, keeps at most ``limit`` of them, and
    returns them resolved against ``base_url`` in source order so the first-mentioned
    endpoints survive a small budget.

    ``assets`` includes references to static files. They are never worth fetching - the
    crawl already has them as subresources, or they are inert - but they are the best
    evidence of which directories the server actually serves, so
    :func:`directory_ancestors` is given the wider list.
    """
    if not source:
        return ()

    origin = urlsplit(base_url)
    root = f"{origin.scheme}://{origin.netloc}"

    seen: dict[str, None] = {}
    for pattern in (_PATH_LITERAL, _ORIGIN_PREFIXED):
        for match in pattern.finditer(source):
            path = _plausible_path(match.group("value"), assets=assets)
            if path is None:
                continue
            seen.setdefault(urljoin(root, path), None)
            if len(seen) >= limit:
                return tuple(seen)
    return tuple(seen)


def directory_ancestors(urls: Sequence[str], *, limit: int = 40) -> tuple[str, ...]:
    """Containing directories of paths the application named, one level up at a time.

    This is the one inference in this module, and it is worth being explicit about. A
    served reference to ``/ftp/order_4711.pdf`` does not prove that ``/ftp/`` is listable,
    but it does prove the directory exists and is served, and asking a server about a
    directory it demonstrably uses is a question the operator authorised. It is also how
    an auto-index gets found at all: nothing links to the index, so a link-following crawl
    - and the literal mining above - can never reach it.

    The inference stays shallow by construction. Only ancestors of paths the application
    itself referenced are produced, never of arbitrary URLs; the origin root is excluded
    because it is the seed; and ``limit`` caps the whole set, so a deep bundle cannot turn
    into a directory sweep.
    """
    out: dict[str, None] = {}
    for url in urls:
        parts = urlsplit(url)
        segments = [segment for segment in parts.path.split("/") if segment]
        # Drop the last segment: it is the resource, not a directory. Then walk up.
        for depth in range(len(segments) - 1, 0, -1):
            candidate = urlunsplit(
                (parts.scheme, parts.netloc, "/" + "/".join(segments[:depth]) + "/", "", "")
            )
            out.setdefault(candidate, None)
            if len(out) >= limit:
                return tuple(out)
    return tuple(out)


def extract_view_routes(source: str, *, limit: int = 200) -> tuple[str, ...]:
    """Client-side route names from a framework route table.

    These are reported, never fetched - see this module's docstring for why. They are
    the honest measure of how large the application is compared with how much of it a
    request-level crawl can reach.
    """
    if not source:
        return ()
    routes: dict[str, None] = {}
    for match in _ROUTE_ENTRY.finditer(source):
        value = (match.group("value") or "").strip().strip("/")
        if not value or value in {"**", "*"}:
            continue
        if len(value) > 80 or _NOT_A_PATH.search(value):
            continue
        if not re.search(r"[A-Za-z]", value):
            continue
        routes.setdefault(value, None)
        if len(routes) >= limit:
            break
    return tuple(routes)


def coverage_note(verdict: SpaVerdict, *, pages_fetched: int, mined_fetched: int) -> str:
    """One sentence an operator can act on, or ``""`` when there is nothing to say."""
    if not verdict.is_spa:
        return ""
    article = "an" if verdict.framework[:1].upper() in "AEIOU" else "a"
    parts = [
        f"the target is {article} {verdict.framework} single-page application "
        f"({verdict.anchors} anchor(s) in the served HTML), so its routes exist only in "
        f"JavaScript and a link-following crawl cannot see them"
    ]
    if verdict.view_routes:
        parts.append(f"{len(verdict.view_routes)} client-side view route(s) were read from the bundles")
    if verdict.script_paths:
        parts.append(
            f"{len(verdict.script_paths)} request path(s) were mined from the bundles and "
            f"{mined_fetched} of them were fetched within the budget"
        )
    else:
        parts.append(
            "no request paths could be mined from the bundles, so this scan saw "
            f"{pages_fetched} page(s) and a browser-driving scanner such as OWASP ZAP's "
            "AJAX spider will find substantially more"
        )
    return "; ".join(parts) + "."
