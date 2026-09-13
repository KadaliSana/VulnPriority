"""Version matching: the deterministic half of applicability.

Deciding whether a CVE applies to the deployment in front of you is mostly arithmetic on
version ranges, and it must be arithmetic rather than judgement, because this is the step a
hostile page would most like to influence. A verdict reached here from curated feed data is
authoritative and the model cannot overturn it.
"""

from __future__ import annotations

import pytest

from vulnpriority.core.enums import VersionMatch
from vulnpriority.core.models import AffectedProduct, TechComponent
from vulnpriority.semantic.cpe_match import (
    cpe_product_matches,
    describe_range,
    match_affected,
    parse_cpe,
    parse_version,
    version_in_range,
)


@pytest.mark.parametrize(
    "raw,expected_prefix",
    [
        ("2.5.12", (2, 5, 12)),
        ("1.0.0-rc1", (1, 0, 0)),
        ("10.0", (10, 0)),
        ("8.5u31", (8, 5)),
        ("v3.2.1", (3, 2, 1)),
        ("2024.06", (2024, 6)),
    ],
)
def test_version_parsing_tolerates_real_world_strings(raw: str, expected_prefix: tuple[int, ...]) -> None:
    parsed = parse_version(raw)
    assert parsed[: len(expected_prefix)] == expected_prefix


def test_unparseable_version_is_zero_not_an_exception() -> None:
    """Missing and unparseable versions collapse to the same zero tuple, never an error."""
    zero = parse_version(None)
    assert set(zero) == {0}
    assert parse_version("") == zero
    assert parse_version("latest") == zero
    assert parse_version("2.5.12") > zero


def test_version_ordering_is_numeric_not_lexicographic() -> None:
    """String comparison would put 2.5.9 above 2.5.12, which is the classic off-by-one bug."""
    assert parse_version("2.5.9") < parse_version("2.5.12")
    assert parse_version("1.10.0") > parse_version("1.9.9")


RANGE = AffectedProduct(
    cpe="cpe:2.3:a:apache:struts:*:*:*:*:*:*:*:*",
    version_start_including="2.5.0",
    version_end_excluding="2.5.14",
)


@pytest.mark.parametrize(
    "version,inside",
    [
        ("2.5.0", True),    # start is inclusive
        ("2.5.12", True),
        ("2.5.13", True),
        ("2.5.14", False),  # end is exclusive
        ("2.5.15", False),
        ("2.4.99", False),
        ("3.0.0", False),
    ],
)
def test_inclusive_and_exclusive_bounds(version: str, inside: bool) -> None:
    assert version_in_range(version, RANGE) is inside


def test_exclusive_start_and_inclusive_end() -> None:
    product = AffectedProduct(
        cpe="cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
        version_start_excluding="1.0.0",
        version_end_including="2.0.0",
    )
    assert version_in_range("1.0.0", product) is False
    assert version_in_range("1.0.1", product) is True
    assert version_in_range("2.0.0", product) is True
    assert version_in_range("2.0.1", product) is False


def test_open_ended_range_matches_everything_above_the_floor() -> None:
    product = AffectedProduct(cpe="cpe:2.3:a:v:p:*:*:*:*:*:*:*:*", version_start_including="3.0")
    assert version_in_range("3.0", product) is True
    assert version_in_range("99.0", product) is True
    assert version_in_range("2.9", product) is False


def test_unknown_version_is_not_silently_treated_as_affected() -> None:
    assert version_in_range(None, RANGE) is False


def test_cpe_parsing_and_product_matching() -> None:
    parsed = parse_cpe("cpe:2.3:a:apache:struts:2.5.12:*:*:*:*:*:*:*")
    assert parsed is not None
    assert parsed["vendor"] == "apache"
    assert parsed["product"] == "struts"
    assert parsed["version"] == "2.5.12"

    assert cpe_product_matches(
        "cpe:2.3:a:apache:struts:*:*:*:*:*:*:*:*",
        TechComponent(vendor="apache", product="struts", version="2.5.12"),
    )
    assert not cpe_product_matches(
        "cpe:2.3:a:apache:struts:*:*:*:*:*:*:*:*",
        TechComponent(vendor="oracle", product="weblogic", version="12.2"),
    )


def test_matching_observed_stack_against_affected_products() -> None:
    tech = (TechComponent(vendor="apache", product="struts", version="2.5.12"),)
    verdict, reason = match_affected(tech, (RANGE,))
    assert verdict == VersionMatch.MATCH
    assert reason


def test_version_outside_every_range_is_a_mismatch() -> None:
    tech = (TechComponent(vendor="apache", product="struts", version="2.5.31"),)
    verdict, reason = match_affected(tech, (RANGE,))
    assert verdict == VersionMatch.MISMATCH
    assert "2.5.31" in reason or "range" in reason.lower()


def test_unrelated_product_is_unknown_not_a_mismatch() -> None:
    """Absence of evidence about the right product is not evidence that it is unaffected."""
    tech = (TechComponent(vendor="nginx", product="nginx", version="1.24.0"),)
    verdict, _ = match_affected(tech, (RANGE,))
    assert verdict == VersionMatch.UNKNOWN


def test_no_version_observed_is_unknown() -> None:
    tech = (TechComponent(vendor="apache", product="struts"),)
    verdict, _ = match_affected(tech, (RANGE,))
    assert verdict == VersionMatch.UNKNOWN


def test_no_affected_data_is_unknown() -> None:
    tech = (TechComponent(vendor="apache", product="struts", version="2.5.12"),)
    verdict, _ = match_affected(tech, ())
    assert verdict == VersionMatch.UNKNOWN


def test_a_single_matching_component_decides_the_verdict() -> None:
    tech = (
        TechComponent(vendor="nginx", product="nginx", version="1.24.0"),
        TechComponent(vendor="apache", product="struts", version="2.5.12"),
    )
    verdict, _ = match_affected(tech, (RANGE,))
    assert verdict == VersionMatch.MATCH


def test_range_description_is_human_readable() -> None:
    text = describe_range(RANGE)
    assert "2.5.0" in text and "2.5.14" in text
