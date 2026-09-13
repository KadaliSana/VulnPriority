"""How a money figure is written down.

One module, because the alternative is what this framework had: a ``$`` hard-coded into
the CLI, another into the research report, a third into the web fallback renderer, and a
fourth into the assessment narrative -- four places that all had to be found and changed
the day the framework stopped being denominated in dollars, and any one of which could
be missed and go on printing a symbol that no longer matched the numbers.

Two things vary by currency and both live here:

**The symbol.** ``ImpactModel.currency`` is the authority on what the numbers mean; every
formatter takes the currency and asks this module what to draw. Nothing downstream hard-
codes a glyph, so a run configured in USD prints dollars and a run configured in INR
prints rupees without either caller knowing the difference.

**The grouping.** Western grouping puts a separator every three digits. The Indian system
groups the last three and then in twos -- ``12345678`` is ``1,23,45,678``, not
``12,345,678`` -- and names the groups: a *lakh* is ``100,000`` and a *crore* is
``10,000,000``. Nobody reads ``195636364`` digit by digit; they read it as
``19.56 crore``. :func:`format_money_compact` is that reading, and
:func:`format_money` is the grouped form for tables, where a scale word in one row and
not the next would destroy the column.

This module is deliberately dependency-free and importable from anywhere in the package,
so the report layer and the CLI can share it without either one depending on the other.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "DEFAULT_CURRENCY",
    "CURRENCY_SYMBOLS",
    "ASCII_CURRENCY_SYMBOLS",
    "ASCII_FALLBACK_CHARS",
    "INDIAN_GROUPED_CURRENCIES",
    "LAKH",
    "CRORE",
    "currency_symbol",
    "uses_indian_grouping",
    "group_digits",
    "format_money",
    "format_money_compact",
    "scale_words",
]

#: The currency the shipped impact models are denominated in. A caller that has an
#: :class:`~vulnprio.core.models.ImpactModel` should pass ``impact_model.currency`` rather
#: than rely on this; it exists so that a formatter reached without one still prints
#: something consistent with the shipped configuration instead of guessing.
DEFAULT_CURRENCY = "INR"

#: Symbols for the currencies an operator is plausibly going to configure. An unlisted
#: code is printed as the code itself followed by a space (``"CHF 12,000"``), which is
#: unambiguous and never silently draws the wrong glyph.
CURRENCY_SYMBOLS: dict[str, str] = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "AUD": "A$",
    "CAD": "C$",
    "SGD": "S$",
    "AED": "د.إ",
}

#: What to draw instead when the output channel cannot carry the real symbol. A Windows
#: console still defaults to cp1252, which has no U+20B9, so a rupee sign written to one
#: raises ``UnicodeEncodeError`` and takes the whole command down with it -- after the
#: pipeline has finished its work, which turns a successful run into a crash report.
#:
#: This is a property of the *channel*, not of the currency: the HTML page and the written
#: report are UTF-8 by construction and always get the real symbol. Only a terminal that
#: has told us it cannot encode the glyph gets the fallback, and the fallback is a real
#: attribution ("Rs") rather than a stripped symbol, because an amount with no currency on
#: it at all is worse than an unfashionable one.
ASCII_CURRENCY_SYMBOLS: dict[str, str] = {
    "INR": "Rs ",
    "USD": "$",
    "EUR": "EUR ",
    "GBP": "GBP ",
    "JPY": "JPY ",
    "CNY": "CNY ",
    "AUD": "A$",
    "CAD": "C$",
    "SGD": "S$",
    "AED": "AED ",
}

#: Single-character currency symbols and the ASCII spelling to put in their place when a
#: stream cannot encode them at all.
#:
#: :data:`ASCII_CURRENCY_SYMBOLS` covers the amounts a caller formats deliberately. This
#: covers the ones that arrive already formatted, embedded in text written elsewhere -- a
#: reason code, an impact rationale, a rendered report piped to a terminal. The CLI
#: installs it as a codec error handler so those degrade to ``Rs 50,000`` rather than to
#: ``?50,000``.
#:
#: U+00A5 is shared between the yen and the yuan; it is spelled as the yen here because
#: that is the commoner reading, and an ambiguous-but-present attribution still beats a
#: question mark.
ASCII_FALLBACK_CHARS: dict[str, str] = {
    "₹": "Rs ",
    "€": "EUR ",
    "£": "GBP ",
    "¥": "JPY ",
}

#: Currencies written with the Indian grouping and the lakh/crore scale words. This is a
#: property of the *numbering convention*, not of the country, which is why it is a set of
#: currencies rather than a locale lookup: the South Asian currencies below all share it.
INDIAN_GROUPED_CURRENCIES: frozenset[str] = frozenset(
    {"INR", "PKR", "BDT", "NPR", "LKR", "MVR"}
)

#: 100,000. One lakh.
LAKH = 100_000

#: 10,000,000. One crore. A hundred lakh, not a thousand.
CRORE = 10_000_000

#: ``(threshold, word, decimal places, separator)``, largest first. The decimal places
#: differ per scale on purpose: at crore scale two places carry real money (0.01 crore is
#: a lakh), while at lakh scale the second place is noise.
_INDIAN_SCALES: tuple[tuple[int, str, int, str], ...] = (
    (CRORE, "crore", 2, " "),
    (LAKH, "lakh", 1, " "),
)

_WESTERN_SCALES: tuple[tuple[int, str, int, str], ...] = (
    (1_000_000_000, "B", 2, ""),
    (1_000_000, "M", 2, ""),
    (1_000, "k", 1, ""),
)


def _finite(value: Any) -> float | None:
    """``float(value)`` when that is a real number, otherwise ``None``.

    NaN and the infinities are not money and must not be printed as though they were.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _code(currency: str | None) -> str:
    return (currency or DEFAULT_CURRENCY).strip().upper() or DEFAULT_CURRENCY


def currency_symbol(currency: str | None = None, *, ascii_only: bool = False) -> str:
    """The symbol to draw for ``currency``.

    ``ascii_only`` is for an output channel that cannot encode the real glyph -- a Windows
    console on cp1252, in practice. It is the caller's statement about where the text is
    going, never a property of the currency, so the same amount is written ``₹12,000``
    in the report and ``Rs 12,000`` in a terminal that cannot draw a rupee sign.

    An unknown code becomes ``"<CODE> "`` -- the code plus a space -- so that the amount is
    still attributed to a currency and the caller's concatenation still reads correctly.
    """
    code = _code(currency)
    table = ASCII_CURRENCY_SYMBOLS if ascii_only else CURRENCY_SYMBOLS
    return table.get(code, f"{code} ")


def uses_indian_grouping(currency: str | None = None) -> bool:
    """Whether ``currency`` is written with ``##,##,###`` grouping and lakh/crore."""
    return _code(currency) in INDIAN_GROUPED_CURRENCIES


def scale_words(currency: str | None = None) -> tuple[tuple[int, str, int, str], ...]:
    """The ``(threshold, word, digits, separator)`` table :func:`format_money_compact` uses."""
    return _INDIAN_SCALES if uses_indian_grouping(currency) else _WESTERN_SCALES


def _group_western(integer_digits: str) -> str:
    out = []
    while len(integer_digits) > 3:
        out.insert(0, integer_digits[-3:])
        integer_digits = integer_digits[:-3]
    if integer_digits:
        out.insert(0, integer_digits)
    return ",".join(out)


def _group_indian(integer_digits: str) -> str:
    """``"12345678"`` -> ``"1,23,45,678"``: the last three digits, then twos."""
    if len(integer_digits) <= 3:
        return integer_digits
    head, tail = integer_digits[:-3], integer_digits[-3:]
    out = [tail]
    while len(head) > 2:
        out.insert(0, head[-2:])
        head = head[:-2]
    if head:
        out.insert(0, head)
    return ",".join(out)


def group_digits(value: Any, currency: str | None = None, digits: int = 0) -> str:
    """``value`` with the digit grouping ``currency`` uses, and no symbol.

    Returns ``""`` when ``value`` is not a finite number; callers that need a placeholder
    say so themselves, because what to print instead of a number is a question about the
    document, not about arithmetic.
    """
    number = _finite(value)
    if number is None:
        return ""
    sign = "-" if number < 0 else ""
    text = f"{abs(number):.{max(0, int(digits))}f}"
    whole, _, fraction = text.partition(".")
    grouped = _group_indian(whole) if uses_indian_grouping(currency) else _group_western(whole)
    return f"{sign}{grouped}.{fraction}" if fraction else f"{sign}{grouped}"


def format_money(
    value: Any,
    currency: str | None = None,
    *,
    digits: int = 0,
    missing: str = "-",
    ascii_only: bool = False,
) -> str:
    """``value`` as a grouped amount with its symbol: ``"₹1,23,45,678"``.

    The form for tables and anywhere a column of figures has to line up, because every row
    is written at the same scale. Prose wants :func:`format_money_compact`.

    ``ascii_only`` swaps the symbol for one that encodes in any single-byte codepage; see
    :data:`ASCII_CURRENCY_SYMBOLS`. It never changes the digits.
    """
    number = _finite(value)
    if number is None:
        return missing
    body = group_digits(abs(number), currency, digits)
    sign = "-" if number < 0 else ""
    return f"{sign}{currency_symbol(currency, ascii_only=ascii_only)}{body}"


def format_money_compact(
    value: Any,
    currency: str | None = None,
    *,
    missing: str = "-",
    ascii_only: bool = False,
) -> str:
    """``value`` the way someone would say it: ``"₹1.23 crore"``, ``"$4.5k"``.

    For prose and headline figures. Below the smallest scale word the amount is written
    out in full by :func:`format_money`, so small numbers never turn into ``"0.02 lakh"``.
    """
    number = _finite(value)
    if number is None:
        return missing
    magnitude = abs(number)
    symbol = currency_symbol(currency, ascii_only=ascii_only)
    for threshold, word, places, separator in scale_words(currency):
        if magnitude >= threshold:
            scaled = magnitude / threshold
            sign = "-" if number < 0 else ""
            return f"{sign}{symbol}{scaled:,.{places}f}{separator}{word}"
    return format_money(number, currency, missing=missing, ascii_only=ascii_only)
