"""Stage 1 of the sandbox: make untrusted text mean exactly what it looks like.

Every trick an injection uses to hide an imperative from a pattern matcher while still
being read by a language model is an *encoding* trick: a zero-width character inside a
keyword, a Cyrillic ``о`` that renders as a Latin ``o``, a ``display:none`` paragraph that
only the model sees, a base64 blob carrying the real payload. The instruction filter that
runs after this module can only be as good as the text it is handed, so normalisation
happens first and is deliberately lossy in the attacker's favour: when in doubt, remove.

The stages run in a fixed order and each one reports how much it removed, because those
counts are evidence in the adversarial evaluation, not debug output.
"""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser

__all__ = [
    "HOMOGLYPHS",
    "ZERO_WIDTH_CHARS",
    "BIDI_CONTROL_CHARS",
    "ELIDED_BLOB_TEMPLATE",
    "normalize_untrusted",
    "fold_homoglyphs",
    "strip_invisible",
    "html_to_text",
    "collapse_whitespace",
    "elide_blobs",
]

#: Rendered as nothing, but a model still tokenises around them: ``ig<ZWSP>nore``.
ZERO_WIDTH_CHARS: frozenset[str] = frozenset(
    "​‌‍‎‏"      # ZWSP, ZWNJ, ZWJ, LRM, RLM (U+200B..U+200F)
    "﻿"                               # BOM / zero-width no-break space
    "⁠⁡⁢⁣⁤"       # word joiner and invisible operators
    "­"                               # soft hyphen
    "᠎"                               # Mongolian vowel separator
)

#: Bidirectional overrides let a payload be stored reversed and render forwards.
BIDI_CONTROL_CHARS: frozenset[str] = frozenset(
    "‪‫‬‭‮"       # LRE, RLE, PDF, LRO, RLO (U+202A..U+202E)
    "⁦⁧⁨⁩"             # LRI, RLI, FSI, PDI (U+2066..U+2069)
)

#: Explicit confusable table (a curated subset of UTS#39). Only characters whose Latin
#: lookalike is unambiguous are listed: folding more would corrupt genuine Russian or
#: Greek advisory prose, which the multilingual patterns still need to match.
HOMOGLYPHS: dict[str, str] = {
    # --- Cyrillic, upper case ---
    "А": "A",  # А
    "В": "B",  # В
    "Е": "E",  # Е
    "Ё": "E",  # Ё
    "К": "K",  # К
    "М": "M",  # М
    "Н": "H",  # Н
    "О": "O",  # О
    "Р": "P",  # Р
    "С": "C",  # С
    "Т": "T",  # Т
    "У": "Y",  # У
    "Х": "X",  # Х
    "Ѕ": "S",  # Ѕ
    "І": "I",  # І
    "Ї": "I",  # Ї
    "Ј": "J",  # Ј
    "Ӏ": "I",  # Ӏ
    "Ԛ": "Q",  # Ԛ
    "Ԝ": "W",  # Ԝ
    "Ғ": "F",  # Ғ
    "һ": "h",  # һ
    # --- Cyrillic, lower case ---
    "а": "a",  # а
    "е": "e",  # е
    "к": "k",  # к
    "м": "m",  # м
    "о": "o",  # о
    "р": "p",  # р
    "с": "c",  # с
    "т": "t",  # т
    "у": "y",  # у
    "х": "x",  # х
    "ѕ": "s",  # ѕ
    "і": "i",  # і
    "ї": "i",  # ї
    "ј": "j",  # ј
    "ԛ": "q",  # ԛ
    "ԝ": "w",  # ԝ
    "ԁ": "d",  # ԁ
    "ғ": "f",  # ғ
    # --- Greek ---
    "Α": "A",  # Α
    "Β": "B",  # Β
    "Ε": "E",  # Ε
    "Ζ": "Z",  # Ζ
    "Η": "H",  # Η
    "Ι": "I",  # Ι
    "Κ": "K",  # Κ
    "Μ": "M",  # Μ
    "Ν": "N",  # Ν
    "Ο": "O",  # Ο
    "Ρ": "P",  # Ρ
    "Τ": "T",  # Τ
    "Υ": "Y",  # Υ
    "Χ": "X",  # Χ
    "Ϲ": "C",  # Ϲ
    "α": "a",  # α
    "ε": "e",  # ε
    "ι": "i",  # ι
    "κ": "k",  # κ
    "ν": "v",  # ν
    "ο": "o",  # ο
    "ρ": "p",  # ρ
    "τ": "t",  # τ
    "υ": "u",  # υ
    "χ": "x",  # χ
    "ϲ": "c",  # ϲ
}

ELIDED_BLOB_TEMPLATE = "[ELIDED-BLOB {n} bytes]"

_CYRILLIC_GREEK = re.compile(r"[Ͱ-ϿЀ-ӿԀ-ԯ]")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_LOOKS_LIKE_HTML = re.compile(r"<\s*/?\s*[a-zA-Z][^>]{0,400}>|<!--")
_WHITESPACE_RUN = re.compile(r"\s+")
_B64_RUN = re.compile(r"[A-Za-z0-9+/]{48,}={0,2}")
_HEX_RUN = re.compile(r"\b[0-9a-fA-F]{64,}\b")
_ELIDED_RUN = re.compile(r"\[ELIDED-BLOB (\d+) bytes\](?:\s*\[ELIDED-BLOB (\d+) bytes\])+")
_ELIDED_ONE = re.compile(r"\[ELIDED-BLOB (\d+) bytes\]")

_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)
_DROPPED_TAGS = frozenset({"script", "style", "template", "noscript", "svg", "canvas", "iframe", "object"})
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "div", "dd", "dl", "dt", "fieldset", "figcaption",
        "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav",
        "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
    }
)
_HIDDEN_STYLE = re.compile(
    r"(?:display\s*:\s*none)"
    r"|(?:visibility\s*:\s*hidden)"
    r"|(?:font-size\s*:\s*0(?:\.0+)?\s*(?:px|pt|em|rem|%)?\s*(?:;|$))"
    r"|(?:opacity\s*:\s*0(?:\.0+)?\s*(?:;|$))"
    r"|(?:text-indent\s*:\s*-\s*\d{3,})"
    r"|(?:left\s*:\s*-\s*\d{4,})",
    re.IGNORECASE,
)


def strip_invisible(text: str) -> tuple[str, int, int]:
    """Drop zero-width and bidirectional control characters.

    Returns ``(text, zero_width_removed, bidi_removed)``. These characters carry no
    meaning a security analyst needs, and every one of them is a way to split a keyword.
    """
    out: list[str] = []
    zero_width = 0
    bidi = 0
    for char in text:
        if char in ZERO_WIDTH_CHARS:
            zero_width += 1
            continue
        if char in BIDI_CONTROL_CHARS:
            bidi += 1
            continue
        if char in ("\t", "\n", "\r"):
            out.append(char)
            continue
        if unicodedata.category(char) in ("Cc", "Cf"):
            zero_width += 1
            continue
        out.append(char)
    return "".join(out), zero_width, bidi


def fold_homoglyphs(text: str) -> tuple[str, int]:
    """Fold Cyrillic/Greek lookalikes to Latin, token by token.

    Folding the whole document would turn genuine Russian prose into Latin mush and defeat
    the multilingual patterns, so a character is folded only when its *token* is already
    part Latin (``іgnore``), or when the document contains no real non-Latin letters at all
    and the token is entirely confusable (``раѕѕword`` in an otherwise English page).
    """
    latin = sum(1 for char in text if "a" <= char.lower() <= "z")
    confusable = sum(1 for char in text if char in HOMOGLYPHS)
    has_real_foreign = any(
        _CYRILLIC_GREEK.match(char) and char not in HOMOGLYPHS for char in text
    )
    allow_full_token_fold = latin > confusable and not has_real_foreign
    folded = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal folded
        token = match.group(0)
        hits = sum(1 for char in token if char in HOMOGLYPHS)
        if hits == 0:
            return token
        has_latin = any("a" <= char.lower() <= "z" for char in token)
        fully_confusable = len(token) >= 2 and all(
            char in HOMOGLYPHS or char.isdigit() or char == "_" for char in token
        )
        if has_latin or (fully_confusable and allow_full_token_fold):
            folded += hits
            return "".join(HOMOGLYPHS.get(char, char) for char in token)
        return token

    return _TOKEN_RE.sub(_replace, text), folded


class _TextExtractor(HTMLParser):
    """HTML-to-text that drops anything a human reader would not see.

    Hidden elements are where injections live: the page shows an advisory, the model reads
    an extra paragraph telling it to set the score to 1.0.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_elements = 0
        self.hidden_chars = 0
        self.comments = 0
        self.comment_chars = 0
        self._stack: list[tuple[str, bool]] = []
        self._skip_depth = 0

    @staticmethod
    def _is_hidden(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in _DROPPED_TAGS:
            return True
        for name, value in attrs:
            lowered = name.lower()
            if lowered == "hidden":
                return True
            if lowered == "aria-hidden" and (value or "").strip().lower() == "true":
                return True
            if lowered == "style" and value and _HIDDEN_STYLE.search(value):
                return True
            if lowered == "type" and (value or "").strip().lower() == "hidden":
                return True
        return False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        hidden = self._is_hidden(tag, attrs)
        if hidden:
            self.hidden_elements += 1
        if tag in _VOID_TAGS:
            if tag in _BLOCK_TAGS and self._skip_depth == 0:
                self.parts.append("\n")
            return
        self._stack.append((tag, hidden))
        if hidden:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS and self._skip_depth == 0:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._is_hidden(tag.lower(), attrs):
            self.hidden_elements += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if not any(name == tag for name, _ in self._stack):
            return
        while self._stack:
            name, hidden = self._stack.pop()
            if hidden:
                self._skip_depth -= 1
            if name == tag:
                break
        if tag in _BLOCK_TAGS and self._skip_depth == 0:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            self.hidden_chars += len(data)
            return
        self.parts.append(data)

    def handle_comment(self, data: str) -> None:
        self.comments += 1
        self.comment_chars += len(data)

    def text(self) -> str:
        return "".join(self.parts)


def html_to_text(text: str) -> tuple[str, dict[str, int]]:
    """Convert HTML to visible text, dropping hidden elements and comments.

    Plain text that contains no markup is returned untouched so that ordinary prose
    containing ``a < b`` is never mangled.
    """
    counts = {"hidden_elements_removed": 0, "hidden_chars_removed": 0, "html_comments_removed": 0}
    if not _LOOKS_LIKE_HTML.search(text):
        return text, counts
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    counts["hidden_elements_removed"] = parser.hidden_elements
    counts["hidden_chars_removed"] = parser.hidden_chars + parser.comment_chars
    counts["html_comments_removed"] = parser.comments
    return parser.text(), counts


def collapse_whitespace(text: str) -> tuple[str, int]:
    """Collapse every run of whitespace to one space.

    Injections pad keywords with newlines and tabs to break naive patterns; after this the
    pattern library only has to tolerate single spaces.
    """
    collapsed = _WHITESPACE_RUN.sub(" ", text).strip()
    return collapsed, max(0, len(text) - len(collapsed))


def _looks_like_blob(candidate: str) -> bool:
    """A long opaque run: mixed case with digits, i.e. not a word and not a path."""
    if candidate.count("/") > 2:
        return False
    has_upper = any(char.isupper() for char in candidate)
    has_lower = any(char.islower() for char in candidate)
    has_digit = any(char.isdigit() for char in candidate)
    return has_upper and has_lower and has_digit


def elide_blobs(text: str) -> tuple[str, int]:
    """Replace long base64/hex runs with ``[ELIDED-BLOB n bytes]``.

    A model can decode base64; a pattern matcher cannot. Rather than try to decode every
    blob, the sandbox removes the channel: an opaque run is summarised by its size.
    Adjacent elisions (a wrapped blob whose line breaks became spaces) are merged so one
    payload reports as one blob.
    """

    def _b64(match: re.Match[str]) -> str:
        candidate = match.group(0)
        if not _looks_like_blob(candidate):
            return candidate
        return ELIDED_BLOB_TEMPLATE.format(n=len(candidate))

    staged = _B64_RUN.sub(_b64, text)
    staged = _HEX_RUN.sub(lambda m: ELIDED_BLOB_TEMPLATE.format(n=len(m.group(0))), staged)

    def _merge(match: re.Match[str]) -> str:
        total = sum(int(value) for value in _ELIDED_ONE.findall(match.group(0)))
        return ELIDED_BLOB_TEMPLATE.format(n=total)

    merged = _ELIDED_RUN.sub(_merge, staged)
    return merged, len(_ELIDED_ONE.findall(merged))


def normalize_untrusted(text: str, max_chars: int = 6000) -> tuple[str, dict[str, int]]:
    """Run every normalisation stage in order and report what each one removed.

    Order matters: NFKC first (so mathematical-alphanumeric and fullwidth spoofs become
    ordinary letters), invisible characters next (so homoglyph tokens are contiguous),
    homoglyph folding next (so ``<ѕcript>`` becomes ``<script>`` before the HTML parser
    sees it), then HTML, whitespace, blob elision and finally the hard length cap.

    Returns the sanitized text and a count per stage.
    """
    original_length = len(text)
    counts: dict[str, int] = {
        "original_length": original_length,
        "nfkc_normalized": 0,
        "zero_width_removed": 0,
        "bidi_removed": 0,
        "homoglyphs_folded": 0,
        "hidden_elements_removed": 0,
        "html_comments_removed": 0,
        "hidden_text_removed": 0,
        "whitespace_collapsed": 0,
        "blobs_elided": 0,
        "chars_truncated": 0,
        "final_length": 0,
    }

    counts["nfkc_normalized"] = sum(
        1 for char in text if unicodedata.normalize("NFKC", char) != char
    )
    staged = unicodedata.normalize("NFKC", text)

    staged, zero_width, bidi = strip_invisible(staged)
    counts["zero_width_removed"] = zero_width
    counts["bidi_removed"] = bidi

    staged, folded = fold_homoglyphs(staged)
    counts["homoglyphs_folded"] = folded

    staged, html_counts = html_to_text(staged)
    counts["hidden_elements_removed"] = html_counts["hidden_elements_removed"]
    counts["html_comments_removed"] = html_counts["html_comments_removed"]
    counts["hidden_text_removed"] = html_counts["hidden_chars_removed"]

    staged, whitespace = collapse_whitespace(staged)
    counts["whitespace_collapsed"] = whitespace

    staged, blobs = elide_blobs(staged)
    counts["blobs_elided"] = blobs

    cap = max(0, int(max_chars))
    if len(staged) > cap:
        counts["chars_truncated"] = len(staged) - cap
        staged = staged[:cap]

    counts["final_length"] = len(staged)
    return staged, counts
