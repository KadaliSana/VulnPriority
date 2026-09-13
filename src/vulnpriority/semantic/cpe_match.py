"""Deterministic version-range matching between observed components and CVE data.

Goal 3 asks whether a reported vulnerability actually applies to *this* application.
Most of that question is settled without any model at all: if the scanner fingerprinted
Struts 2.5.33 and the advisory affects ``< 2.5.22``, the finding does not apply, and no
amount of confident prose can change that. This module is the part of Component A that
produces that answer, which is why it is pure, total and has no dependency on the model
layer.

Version strings in the wild are not PEP 440. ``parse_version`` therefore accepts the
shapes that actually appear in advisories and fingerprints -- ``2.5.12``, ``1.0.0-rc1``,
``8.5u31``, ``10.0``, ``v3``, ``2.0.0.RELEASE`` -- and maps them all onto one comparable
tuple.
"""

from __future__ import annotations

import re
from typing import Sequence

from vulnpriority.core.enums import VersionMatch
from vulnpriority.core.models import AffectedProduct, TechComponent

__all__ = [
    "RELEASE_PARTS",
    "parse_version",
    "parse_cpe",
    "cpe_product_matches",
    "version_in_range",
    "describe_range",
    "match_affected",
]

#: Number of numeric release components kept in the comparable tuple.
RELEASE_PARTS = 4

#: Suffix token -> ordering rank. Negative ranks sort before the plain release, positive
#: after it, so ``1.0.0-rc1 < 1.0.0 < 1.0.0-p1``.
_SUFFIX_RANKS: dict[str, int] = {
    "dev": -5,
    "snapshot": -4,
    "alpha": -3,
    "a": -3,
    "beta": -2,
    "b": -2,
    "milestone": -1,
    "m": -1,
    "rc": -1,
    "cr": -1,
    "pre": -1,
    "final": 0,
    "ga": 0,
    "release": 0,
    "post": 1,
    "rev": 1,
    "r": 1,
    "patch": 1,
    "hotfix": 1,
    "sp": 1,
    "p": 1,
}

_SUFFIX_RE = re.compile(
    r"[-_+.]?(" + "|".join(sorted(_SUFFIX_RANKS, key=len, reverse=True)) + r")[-_.]?(\d+)?$"
)
_UPDATE_SEPARATOR_RE = re.compile(r"(?<=\d)[u](?=\d)")
_NUMBER_RE = re.compile(r"\d+")
_WILDCARDS = frozenset({"", "*", "-", "any", "na", "n/a", "none"})


def parse_version(version: str | None) -> tuple[int, ...]:
    """Comparable tuple for a loosely specified version string.

    The result is always ``RELEASE_PARTS`` numeric components followed by a suffix rank
    and a suffix number, so tuples from different inputs are always mutually comparable:
    ``parse_version("1.0") < parse_version("1.0.1")`` and
    ``parse_version("1.0.0-rc1") < parse_version("1.0.0")``. Missing components are
    zero, which is the conventional reading of ``10.0`` as ``10.0.0.0``.
    """
    if version is None:
        return (0,) * RELEASE_PARTS + (0, 0)
    text = str(version).strip().casefold()
    if text in _WILDCARDS:
        return (0,) * RELEASE_PARTS + (0, 0)
    text = re.sub(r"^(?:v|version[\s:_-]*)", "", text)

    rank = 0
    suffix_number = 0
    match = _SUFFIX_RE.search(text)
    if match:
        rank = _SUFFIX_RANKS[match.group(1)]
        suffix_number = int(match.group(2)) if match.group(2) else 0
        text = text[: match.start()]

    text = _UPDATE_SEPARATOR_RE.sub(".", text)
    numbers = [int(part) for part in _NUMBER_RE.findall(text)]
    numbers = (numbers + [0] * RELEASE_PARTS)[:RELEASE_PARTS]
    return tuple(numbers) + (rank, suffix_number)


def _split_cpe(cpe: str) -> list[str]:
    """Split a CPE on unescaped colons (``\\:`` is a literal colon inside a field)."""
    fields: list[str] = []
    current: list[str] = []
    escaped = False
    for char in cpe:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ":":
            fields.append("".join(current))
            current = []
        else:
            current.append(char)
    fields.append("".join(current))
    return fields


def parse_cpe(cpe: str | None) -> dict[str, str]:
    """Extract ``part``/``vendor``/``product``/``version`` from a CPE 2.2 URI or 2.3 name.

    Unknown or malformed input yields wildcards rather than an exception: the caller's
    job is to degrade to ``UNKNOWN``, not to crash a scan on one odd advisory record.
    """
    empty = {"part": "*", "vendor": "*", "product": "*", "version": "*"}
    if not cpe:
        return empty
    text = str(cpe).strip()
    if text.startswith("cpe:2.3:"):
        fields = _split_cpe(text)[2:]
    elif text.startswith("cpe:/"):
        fields = _split_cpe(text[len("cpe:/") :])
    else:
        # Not a CPE at all: the safest reading of a bare string is a product name.
        return {"part": "a", "vendor": "*", "product": text.casefold(), "version": "*"}
    fields = fields + ["*"] * 4
    return {
        "part": (fields[0] or "*").casefold(),
        "vendor": (fields[1] or "*").casefold(),
        "product": (fields[2] or "*").casefold(),
        "version": (fields[3] or "*").casefold(),
    }


def _name_tokens(name: str | None) -> frozenset[str]:
    if not name:
        return frozenset()
    return frozenset(token for token in re.split(r"[^0-9a-z]+", str(name).casefold()) if token)


def cpe_product_matches(cpe: str | None, tech: TechComponent) -> bool:
    """Whether a CPE names the product an observed component represents.

    Matching is token-based so that ``cpe:2.3:a:apache:struts:*`` matches a component
    fingerprinted as ``Apache Struts``. Vendor is only checked when both sides state one:
    scanners frequently fingerprint a product without its vendor, and treating that
    silence as a vendor disagreement would produce false ``MISMATCH`` verdicts, which is
    the one error class this module must never make.
    """
    parsed = parse_cpe(cpe)
    product = parsed["product"]
    if product in _WILDCARDS:
        product_ok = True
    else:
        product_tokens = _name_tokens(product)
        observed = _name_tokens(tech.product) | _name_tokens(parse_cpe(tech.cpe)["product"])
        product_ok = bool(product_tokens) and product_tokens <= observed

    if not product_ok:
        return False

    vendor = parsed["vendor"]
    observed_vendor = _name_tokens(tech.vendor) | _name_tokens(parse_cpe(tech.cpe)["vendor"])
    if vendor in _WILDCARDS or not observed_vendor:
        return True
    return bool(_name_tokens(vendor) & observed_vendor)


def version_in_range(version: str | None, affected: AffectedProduct) -> bool:
    """Whether ``version`` falls inside one :class:`AffectedProduct` range.

    When the record carries no range bounds, the CPE's own version field decides; a
    wildcard there means every version of the product is affected.
    """
    if version is None or str(version).strip() in _WILDCARDS:
        return False
    observed = parse_version(version)
    bounded = False

    if affected.version_start_including:
        bounded = True
        if observed < parse_version(affected.version_start_including):
            return False
    if affected.version_start_excluding:
        bounded = True
        if observed <= parse_version(affected.version_start_excluding):
            return False
    if affected.version_end_including:
        bounded = True
        if observed > parse_version(affected.version_end_including):
            return False
    if affected.version_end_excluding:
        bounded = True
        if observed >= parse_version(affected.version_end_excluding):
            return False
    if bounded:
        return True

    pinned = parse_cpe(affected.cpe)["version"]
    if pinned in _WILDCARDS:
        return True
    return parse_version(pinned) == observed


def describe_range(affected: AffectedProduct) -> str:
    """Human-readable form of one affected range, for the assessment rationale."""
    parts: list[str] = []
    if affected.version_start_including:
        parts.append(f">= {affected.version_start_including}")
    if affected.version_start_excluding:
        parts.append(f"> {affected.version_start_excluding}")
    if affected.version_end_including:
        parts.append(f"<= {affected.version_end_including}")
    if affected.version_end_excluding:
        parts.append(f"< {affected.version_end_excluding}")
    if not parts:
        pinned = parse_cpe(affected.cpe)["version"]
        parts.append("any version" if pinned in _WILDCARDS else f"== {pinned}")
    product = parse_cpe(affected.cpe)["product"]
    return f"{product} {' and '.join(parts)}"


def match_affected(
    tech: Sequence[TechComponent],
    affected: Sequence[AffectedProduct],
) -> tuple[VersionMatch, str]:
    """Verdict and reason for the observed stack against a CVE's affected products.

    ``MISMATCH`` is only returned when a product was actually observed *with* a version
    and that version lies outside every affected range. Every other shortfall -- no
    advisory data, product not observed, version not fingerprinted -- is ``UNKNOWN``,
    because a mismatch verdict is authoritative downstream and must never be guessed.
    """
    affected = tuple(affected)
    tech = tuple(tech)
    if not affected:
        return VersionMatch.UNKNOWN, "no affected-product data available for this CVE"
    if not tech:
        return VersionMatch.UNKNOWN, "no technology components were fingerprinted on the target"

    pairs = [
        (component, product)
        for product in affected
        for component in tech
        if cpe_product_matches(product.cpe, component)
    ]
    if not pairs:
        names = sorted({parse_cpe(product.cpe)["product"] for product in affected})
        return (
            VersionMatch.UNKNOWN,
            f"none of the {len(tech)} observed components matches the affected product(s): {', '.join(names)}",
        )

    versioned = [(component, product) for component, product in pairs if component.version]
    if not versioned:
        product_name = pairs[0][0].product
        return VersionMatch.UNKNOWN, f"observed {product_name} but no version was fingerprinted"

    for component, product in versioned:
        if version_in_range(component.version, product):
            return (
                VersionMatch.MATCH,
                f"observed {component.product} {component.version} falls within {describe_range(product)}",
            )

    component, _ = versioned[0]
    ranges = "; ".join(describe_range(product) for _, product in versioned[:4])
    return (
        VersionMatch.MISMATCH,
        f"observed {component.product} {component.version} is outside every affected range ({ranges})",
    )
