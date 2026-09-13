"""Reference-page fetcher: the framework's only door to arbitrary internet text.

Everything this module returns is tier ``REFERENCE_PAGE`` - untrusted by construction. Three
controls apply before the text exists as an object at all:

* **Host allowlist.** A URL whose host is not configured is never requested. Advisory links
  in an NVD record are attacker-influenceable in the general case, so the fetcher does not
  follow wherever they point.
* **Byte cap.** The response body is read incrementally and abandoned at
  ``FeedsConfig.max_reference_bytes``, then trimmed again to a character budget.
* **HTML to text without execution.** Scripts, styles, templates and frames are removed as
  *text*; nothing is ever evaluated, no DOM is built, no subresource is fetched.

Neutralising instruction patterns is deliberately *not* done here: that is the sandbox's job
(DESIGN.md 3.3), and doing it twice would hide from the injection detector what the page
actually said.
"""

from __future__ import annotations

import html as html_module
import re
from datetime import date, datetime
from typing import Any, Callable, ClassVar, Iterable, Sequence
from urllib.parse import urlsplit

import httpx

from vulnpriority.core.config import FeedsConfig
from vulnpriority.core.enums import Provenance, TrustTier
from vulnpriority.core.interfaces import ReferenceFetcher
from vulnpriority.core.models import ReferenceDoc, UntrustedText
from vulnpriority.feeds.base import FixtureFeed, LiveFeedBase, as_of_guard, parse_feed_datetime
from vulnpriority.feeds.cache import utc_now

__all__ = [
    "DEFAULT_ALLOWLIST",
    "ReferenceFixtureFetcher",
    "ReferenceLiveFetcher",
    "LiveReferenceFetcher",
    "detect_language",
    "extract_title",
    "html_to_text",
    "is_allowed_host",
]

DEFAULT_ALLOWLIST: tuple[str, ...] = FeedsConfig().reference_host_allowlist

#: Content types worth converting. Anything else (PDF, archives, images) is refused.
_TEXTUAL_CONTENT_TYPES: tuple[str, ...] = (
    "text/html",
    "text/plain",
    "text/markdown",
    "application/xhtml+xml",
    "application/xml",
    "text/xml",
)

_SCRIPTISH = re.compile(
    r"(?is)<(script|style|template|noscript|svg|iframe|object|embed|canvas)\b[^>]*>.*?</\1\s*>"
)
_UNCLOSED_SCRIPTISH = re.compile(r"(?is)<(script|style)\b[^>]*>.*\Z")
_COMMENT = re.compile(r"(?s)<!--.*?-->")
_DOCTYPE = re.compile(r"(?is)<!doctype[^>]*>")
_LINE_BREAKING = re.compile(
    r"(?i)<\s*/?\s*(br|p|div|li|tr|h[1-6]|section|article|table|ul|ol|blockquote|pre)\b[^>]*>"
)
_TAG = re.compile(r"(?s)<[^>]*>")
_TITLE = re.compile(r"(?is)<title[^>]*>(.*?)</title\s*>")
_WS_RUN = re.compile(r"[ \t  - ]+")
_BLANK_RUN = re.compile(r"\n{3,}")


def is_allowed_host(url: str, allowlist: Iterable[str] | None = None) -> bool:
    """True when ``url`` is an http(s) URL on an explicitly allowed host.

    Matching is exact on the hostname (case-insensitive). Suffix matching is deliberately not
    used: ``evil-github.com`` and ``github.com.attacker.net`` would both pass a naive
    ``endswith`` check.
    """
    hosts = {str(host).strip().lower() for host in (allowlist if allowlist is not None else DEFAULT_ALLOWLIST)}
    try:
        parts = urlsplit(str(url))
    except ValueError:
        return False
    if parts.scheme.lower() not in {"http", "https"}:
        return False
    hostname = (parts.hostname or "").lower()
    return bool(hostname) and hostname in hosts


def extract_title(markup: str | None) -> str | None:
    """Text of the first ``<title>`` element, unescaped and whitespace-collapsed."""
    if not markup:
        return None
    match = _TITLE.search(markup)
    if not match:
        return None
    title = _WS_RUN.sub(" ", html_module.unescape(_TAG.sub(" ", match.group(1)))).strip()
    return title or None


def html_to_text(markup: str | None, char_budget: int | None = None) -> str:
    """Convert markup to plain text without executing or resolving anything.

    Elements that carry code or non-prose (``script``, ``style``, ``iframe``, ...) are removed
    with their content; block-level tags become newlines so sentence boundaries survive;
    remaining tags are dropped and entities are unescaped exactly once.
    """
    if not markup:
        return ""
    text = str(markup)
    text = _COMMENT.sub(" ", text)
    text = _DOCTYPE.sub(" ", text)
    text = _SCRIPTISH.sub(" ", text)
    text = _UNCLOSED_SCRIPTISH.sub(" ", text)
    text = _LINE_BREAKING.sub("\n", text)
    text = _TAG.sub(" ", text)
    text = html_module.unescape(text)
    lines = [_WS_RUN.sub(" ", line).strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    text = _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()
    if char_budget is not None and char_budget > 0 and len(text) > char_budget:
        text = text[:char_budget].rstrip()
    return text


#: Unicode blocks that identify a script unambiguously enough for a language tag.
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("ja", 0x3040, 0x30FF),   # hiragana + katakana
    ("zh", 0x4E00, 0x9FFF),   # CJK unified ideographs (ja also uses these; kana wins above)
    ("ko", 0xAC00, 0xD7AF),
    ("ru", 0x0400, 0x04FF),
    ("ar", 0x0600, 0x06FF),
    ("hi", 0x0900, 0x097F),
)

#: Small, deterministic stop-word probes for Latin-script languages.
_STOPWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("es", ("que", "para", "con", "una", "los", "las", "por", "vulnerabilidad", "actualizar")),
    ("fr", ("les", "des", "une", "pour", "avec", "vulnérabilité", "mise")),
    ("de", ("und", "der", "die", "das", "nicht", "sicherheitslücke", "eine")),
    ("pt", ("para", "uma", "não", "com", "vulnerabilidade", "atualizar")),
    ("en", ("the", "and", "that", "with", "vulnerability", "attacker", "remote")),
)


def detect_language(text: str | None) -> str:
    """Cheap, deterministic language tag for a reference page.

    Not a linguistics tool: it exists so multilingual evidence (Gap 8) is labelled and so the
    sandbox knows to apply its non-English instruction patterns. Returns ``"und"`` when the
    text is too short to judge.
    """
    if not text:
        return "und"
    counts: dict[str, int] = {}
    for char in text:
        code = ord(char)
        for tag, low, high in _SCRIPT_RANGES:
            if low <= code <= high:
                counts[tag] = counts.get(tag, 0) + 1
                break
    if counts:
        if counts.get("ja", 0) > 0:
            return "ja"
        return max(sorted(counts), key=lambda tag: counts[tag])
    tokens = re.findall(r"[a-zà-öø-ÿ]+", text.lower())
    if len(tokens) < 5:
        return "und"
    bag = set(tokens)
    scores = {tag: sum(1 for word in words if word in bag) for tag, words in _STOPWORDS}
    best = max(sorted(scores), key=lambda tag: scores[tag])
    return best if scores[best] > 0 else "und"


def _make_doc(
    url: str,
    body: str,
    *,
    title: str | None,
    tags: Sequence[str],
    language: str,
    fetched_at: datetime | None,
) -> ReferenceDoc:
    """Assemble the untrusted document. Content provenance is always ``REFERENCE_PAGE``."""
    return ReferenceDoc(
        url=url,
        title=title,
        tags=tuple(str(tag) for tag in tags),
        language=language,
        content=UntrustedText(
            text=body,
            provenance=Provenance.REFERENCE_PAGE,
            source_url=url,
            fetched_at=fetched_at,
            language=language,
        ),
    )


class ReferenceFixtureFetcher(FixtureFeed, ReferenceFetcher):
    """Offline reference fetcher over ``data/fixtures/feeds/references/index.json``.

    The allowlist is enforced here too, so offline and live runs skip exactly the same
    documents and a fixture cannot quietly widen the framework's attack surface.
    """

    fixture_relpath: ClassVar[str] = "references/index.json"
    name = "references"
    tier = TrustTier.REFERENCE_PAGE

    def __init__(
        self,
        fixture_dir: str | Any = None,
        *,
        path: str | Any = None,
        allowlist: Iterable[str] | None = None,
        char_budget: int = 6000,
    ) -> None:
        super().__init__(fixture_dir, path=path)
        self.allowlist = tuple(allowlist) if allowlist is not None else DEFAULT_ALLOWLIST
        self.char_budget = int(char_budget)

    @staticmethod
    def _normalise_key(key: str) -> str:
        return str(key).strip()

    def _key_of(self, record: dict[str, Any]) -> str | Sequence[str]:
        return str(record.get("url") or "")

    def _build(self, entries: tuple[dict[str, Any], ...], key: str, as_of: date) -> ReferenceDoc | None:
        if not entries or not is_allowed_host(key, self.allowlist):
            return None
        record = entries[0]
        fetched_at = parse_feed_datetime(record.get("fetched_at"))
        if fetched_at is not None and not as_of_guard(fetched_at.date(), as_of):
            return None
        markup = record.get("html")
        if markup:
            body = html_to_text(markup, self.char_budget)
            title = record.get("title") or extract_title(markup)
        else:
            body = str(record.get("text") or "")[: self.char_budget]
            title = record.get("title")
        language = str(record.get("language") or "") or detect_language(body)
        return _make_doc(
            key,
            body,
            title=str(title) if title else None,
            tags=record.get("tags") or (),
            language=language,
            fetched_at=fetched_at,
        )


class ReferenceLiveFetcher(LiveFeedBase, ReferenceFetcher):
    """Fetches advisory pages over HTTP under an allowlist and a hard byte cap."""

    name = "references"
    tier = TrustTier.REFERENCE_PAGE

    def __init__(
        self,
        *,
        allowlist: Iterable[str] | None = None,
        max_bytes: int = 400_000,
        char_budget: int = 6000,
        timeout_s: float = 20.0,
        client: httpx.Client | None = None,
        now: Callable[[], datetime] | None = None,
        strict_as_of: bool = False,
    ) -> None:
        super().__init__(
            timeout_s=timeout_s,
            client=client,
            headers={"Accept": "text/html, text/plain;q=0.9, */*;q=0.1"},
        )
        self.allowlist = tuple(allowlist) if allowlist is not None else DEFAULT_ALLOWLIST
        self.max_bytes = int(max_bytes)
        self.char_budget = int(char_budget)
        self.strict_as_of = bool(strict_as_of)
        self._now = now or utc_now

    @staticmethod
    def _charset_of(content_type: str) -> str:
        for part in content_type.split(";")[1:]:
            name, _, value = part.strip().partition("=")
            if name.strip().lower() == "charset" and value:
                return value.strip().strip('"') or "utf-8"
        return "utf-8"

    def _read_capped(self, response: httpx.Response) -> bytes:
        """Read at most ``max_bytes``; a hostile server cannot make this allocate forever."""
        buffer = bytearray()
        for chunk in response.iter_bytes():
            buffer.extend(chunk)
            if len(buffer) >= self.max_bytes:
                break
        return bytes(buffer[: self.max_bytes])

    def get(self, key: str, as_of: date) -> ReferenceDoc | None:
        url = str(key).strip()
        if not is_allowed_host(url, self.allowlist):
            return None
        fetched_at = self._now()
        if self.strict_as_of and not as_of_guard(fetched_at.date(), as_of):
            # A page fetched today is not evidence that existed at a historical cut-off.
            return None
        try:
            with self.client.stream("GET", url, headers=self._headers, timeout=self.timeout_s) as response:
                if response.status_code >= 400:
                    return None
                content_type = response.headers.get("content-type", "text/html")
                if not any(kind in content_type.lower() for kind in _TEXTUAL_CONTENT_TYPES):
                    return None
                raw = self._read_capped(response)
        except httpx.HTTPError:
            return None
        markup = raw.decode(self._charset_of(content_type), errors="replace")
        body = html_to_text(markup, self.char_budget)
        language = detect_language(body)
        return _make_doc(
            url,
            body,
            title=extract_title(markup),
            tags=(),
            language=language,
            fetched_at=fetched_at,
        )


#: Name used in the implementation brief; kept as an alias of the DESIGN.md name.
LiveReferenceFetcher = ReferenceLiveFetcher
