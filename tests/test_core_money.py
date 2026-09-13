"""How a money figure is written down (``vulnpriority.core.money``).

The framework's claim is that its money figures order remediation work, so a figure a
reader misreads is a figure that cannot do its job. Two things have to be right: the
symbol must come from the run's configured currency rather than from a glyph hard-coded
somewhere, and the digits must be grouped the way the currency's own convention groups
them. ``12345678`` rupees is ``1,23,45,678``, not ``12,345,678``, and in a sentence it is
``1.23 crore``.
"""

from __future__ import annotations

import math

import pytest

from vulnpriority.core.config import PipelineConfig
from vulnpriority.core.models import ImpactModel
from vulnpriority.core.money import (
    ASCII_FALLBACK_CHARS,
    CURRENCY_SYMBOLS,
    CRORE,
    DEFAULT_CURRENCY,
    LAKH,
    currency_symbol,
    format_money,
    format_money_compact,
    group_digits,
    uses_indian_grouping,
)

RUPEE = "₹"


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "0"),
        (7, "7"),
        (999, "999"),
        (1_000, "1,000"),
        (99_999, "99,999"),
        (1_00_000, "1,00,000"),          # one lakh: the first two-digit group appears here
        (12_34_567, "12,34,567"),
        (1_23_45_678, "1,23,45,678"),    # one crore, twenty-three lakh...
        (1_23_45_67_890, "1,23,45,67,890"),
    ],
)
def test_indian_grouping_is_three_digits_then_twos(value: int, expected: str) -> None:
    assert group_digits(value, "INR") == expected


@pytest.mark.parametrize(
    "value,expected",
    [(999, "999"), (1_000, "1,000"), (12_345_678, "12,345,678"), (1_234_567_890, "1,234,567,890")],
)
def test_western_grouping_is_every_three_digits(value: int, expected: str) -> None:
    assert group_digits(value, "USD") == expected


def test_the_two_conventions_genuinely_disagree() -> None:
    """The regression this module exists to prevent: one number, two correct renderings."""
    assert group_digits(12_345_678, "INR") == "1,23,45,678"
    assert group_digits(12_345_678, "USD") == "12,345,678"


def test_grouping_follows_the_currency_and_not_the_caller() -> None:
    assert uses_indian_grouping("INR") is True
    assert uses_indian_grouping("usd") is False
    # The convention is South Asian, not Indian alone.
    assert uses_indian_grouping("BDT") is True


def test_decimals_are_kept_outside_the_grouping() -> None:
    assert group_digits(1_23_456.789, "INR", digits=2) == "1,23,456.79"
    assert group_digits(1234.5, "USD", digits=1) == "1,234.5"


def test_negatives_keep_their_sign() -> None:
    assert group_digits(-12_34_567, "INR") == "-12,34,567"
    assert format_money(-12_34_567, "INR") == f"-{RUPEE}12,34,567"


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------


def test_the_symbol_comes_from_the_currency() -> None:
    assert format_money(1000, "INR") == f"{RUPEE}1,000"
    assert format_money(1000, "USD") == "$1,000"
    assert format_money(1000, "EUR") == "€1,000"


def test_an_unknown_currency_prints_its_code_rather_than_a_wrong_glyph() -> None:
    """Never silently draw a symbol the caller did not configure."""
    assert currency_symbol("CHF") == "CHF "
    assert format_money(1234, "CHF") == "CHF 1,234"


def test_the_default_currency_is_what_the_shipped_presets_use() -> None:
    assert DEFAULT_CURRENCY == "INR"
    assert format_money(1000) == format_money(1000, DEFAULT_CURRENCY)


# ---------------------------------------------------------------------------
# Scale words
# ---------------------------------------------------------------------------


def test_large_rupee_amounts_are_spoken_in_lakh_and_crore() -> None:
    assert format_money_compact(4_50_000, "INR") == f"{RUPEE}4.5 lakh"
    assert format_money_compact(1_23_45_678, "INR") == f"{RUPEE}1.23 crore"
    assert format_money_compact(25_50_00_000, "INR") == f"{RUPEE}25.50 crore"


def test_a_crore_is_a_hundred_lakh() -> None:
    assert CRORE == 100 * LAKH
    assert format_money_compact(CRORE, "INR") == f"{RUPEE}1.00 crore"
    assert format_money_compact(LAKH, "INR") == f"{RUPEE}1.0 lakh"


def test_large_dollar_amounts_keep_the_western_scale_words() -> None:
    assert format_money_compact(1_234_567, "USD") == "$1.23M"
    assert format_money_compact(450_000, "USD") == "$450.0k"


def test_small_amounts_are_written_out_rather_than_scaled() -> None:
    """``0.02 lakh`` is worse than ``2,000``, so below the smallest scale word we spell it."""
    assert format_money_compact(2_000, "INR") == f"{RUPEE}2,000"
    assert format_money_compact(99_999, "INR") == f"{RUPEE}99,999"


def test_the_scale_word_is_the_only_thing_a_sentence_needs() -> None:
    """Prose gets the spoken form; a table gets the full figure. Both describe one number."""
    value = 19_56_36_364
    assert format_money_compact(value, "INR") == f"{RUPEE}19.56 crore"
    assert format_money(value, "INR") == f"{RUPEE}19,56,36,364"


# ---------------------------------------------------------------------------
# Absent and unusable values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, float("nan"), math.inf, -math.inf, "not a number"])
def test_what_is_not_a_number_is_never_printed_as_one(value: object) -> None:
    assert format_money(value, "INR", missing="not measured") == "not measured"
    assert format_money_compact(value, "INR", missing="not measured") == "not measured"
    assert group_digits(value, "INR") == ""


# ---------------------------------------------------------------------------
# The authority
# ---------------------------------------------------------------------------


def test_the_impact_model_is_the_single_authority_on_denomination() -> None:
    """No field name carries a currency, so the model's own field has to be asked."""
    assert ImpactModel().currency == "INR"
    assert not any(name.endswith("_usd") for name in ImpactModel.model_fields)


def test_a_run_configured_in_another_currency_formats_in_that_currency() -> None:
    """The point of threading it: a USD run must print dollars with western grouping."""
    config = PipelineConfig()
    config = config.model_copy(
        update={
            "component_b": config.component_b.model_copy(
                update={"impact": ImpactModel(name="us", currency="USD")}
            )
        }
    )
    assert config.currency() == "USD"
    assert format_money(12_345_678, config.currency()) == "$12,345,678"


def test_the_shipped_default_run_is_denominated_in_rupees() -> None:
    config = PipelineConfig()
    assert config.currency() == "INR"
    assert config.impact_model().name == "default_ecommerce"


# ---------------------------------------------------------------------------
# What a terminal can actually print
#
# U+20B9 is not in cp1252, which is what a Windows console hands Python by default.
# Writing one to such a stream raises UnicodeEncodeError from inside typer.echo, and that
# killed ``run-all`` at the budget table *after* the pipeline had done all of its work.
# A test that only compares strings cannot catch this; these encode the result.
# ---------------------------------------------------------------------------


TERMINAL_CODEPAGES = ("cp1252", "cp437", "ascii", "latin-1")


@pytest.mark.parametrize("codepage", TERMINAL_CODEPAGES)
@pytest.mark.parametrize("value", [0, 4_50_000, 1_23_45_678, 19_56_36_364, -12_34_567])
def test_the_ascii_form_encodes_on_a_legacy_console(codepage: str, value: int) -> None:
    """The regression: this must not raise, whatever the console's code page is."""
    format_money(value, "INR", ascii_only=True).encode(codepage)
    format_money_compact(value, "INR", ascii_only=True).encode(codepage)


@pytest.mark.parametrize("currency", ["INR", "USD", "EUR", "GBP", "JPY", "AED", "CHF"])
def test_every_currency_has_an_encodable_spelling(currency: str) -> None:
    format_money(1_234_567, currency, ascii_only=True).encode("cp1252")
    currency_symbol(currency, ascii_only=True).encode("cp1252")


def test_the_rupee_sign_really_is_the_thing_that_breaks() -> None:
    """Pins the premise: without ``ascii_only`` this is exactly what blows up."""
    with pytest.raises(UnicodeEncodeError):
        format_money(1_23_45_678, "INR").encode("cp1252")


def test_the_ascii_form_still_names_the_currency() -> None:
    """Degrading to a bare number would be worse than degrading to an unfashionable symbol."""
    assert format_money(1_23_456, "INR", ascii_only=True) == "Rs 1,23,456"
    assert format_money_compact(1_23_45_678, "INR", ascii_only=True) == "Rs 1.23 crore"


def test_ascii_only_changes_the_symbol_and_nothing_else() -> None:
    rich = format_money(1_23_45_678, "INR")
    plain = format_money(1_23_45_678, "INR", ascii_only=True)
    assert rich.lstrip(RUPEE) == plain.removeprefix("Rs ") == "1,23,45,678"


def test_a_currency_already_in_the_codepage_is_left_alone() -> None:
    """Only the symbol that cannot be encoded is replaced; a USD run still prints ``$``."""
    assert format_money(1_234_567, "USD", ascii_only=True) == "$1,234,567"


def test_the_symbol_is_the_default_because_files_are_utf8() -> None:
    """The report and the page are written as UTF-8 and must keep the real glyph."""
    assert format_money(1_000, "INR") == f"{RUPEE}1,000"
    assert format_money_compact(CRORE, "INR").startswith(RUPEE)


def test_every_fallback_character_is_a_symbol_that_needs_one() -> None:
    for glyph, spelling in ASCII_FALLBACK_CHARS.items():
        spelling.encode("ascii")               # the replacement must always encode
        assert glyph in CURRENCY_SYMBOLS.values()
