"""Deterministic prose.

There is no language model anywhere in this package. Every sentence in the assessment
report is assembled here from numbers and enumerated values by ordinary code, because the
report is a document someone may sign: the same payload must produce the same words, every
time, on every machine.

Three rules govern what may be written:

1. **Say only what the data says.** A finding is called critical only when a severity field
   holds the word "critical", and it is attributed to whoever said it. Nothing is exploited
   unless a label says it was exploited.
2. **Mark estimates as estimates.** Probabilities and money figures here are model output,
   not measurements, and the sentences that carry them say so.
3. **Say when something is missing.** A value that is absent prints as "not measured" or
   "not recorded", never as zero and never as an adjective.

Everything below is a pure function of its arguments. No clock, no randomness, no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vulnpriority.core.money import format_money, format_money_compact

__all__ = [
    "NOT_MEASURED",
    "NOT_RECORDED",
    "money",
    "money_phrase",
    "pct",
    "num",
    "hours",
    "plural",
    "count_phrase",
    "join_phrase",
    "titlecase_first",
    "yes_no",
    "estimate_note",
    "severity_phrase",
    "kev_phrase",
    "epss_phrase",
    "cvss_phrase",
    "maturity_phrase",
    "applicability_phrase",
    "exposure_phrase",
    "impact_phrase",
    "chain_phrase",
    "trust_phrase",
    "why_bullets",
    "state_phrase",
    "path_sentence",
    "concentration_sentence",
    "one_line_headline",
]

NOT_MEASURED = "not measured"
NOT_RECORDED = "not recorded"

#: Privilege ordinal to a phrase that reads as English inside a sentence.
_PRIVILEGE_PHRASES: dict[int, str] = {
    0: "unauthenticated access to {asset}",
    1: "an ordinary user account on {asset}",
    2: "administrator control of {asset}",
    3: "operating-system control of {asset}",
}

_PRIVILEGE_NAMES: dict[int, str] = {0: "NONE", 1: "USER", 2: "ADMIN", 3: "SYSTEM"}
_PRIVILEGE_BY_NAME: dict[str, int] = {name: value for value, name in _PRIVILEGE_NAMES.items()}

#: Words a severity field is allowed to contain. Anything else is printed verbatim and
#: attributed, never translated into a stronger word.
_SEVERITY_WORDS = ("info", "low", "medium", "high", "critical")


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def money(value: Any, currency: str | None = None, *, missing: str = NOT_MEASURED) -> str:
    """An amount written out in full, grouped the way ``currency`` groups digits.

    ``₹1,23,45,678``. The form for table cells, where every row has to be written at
    the same scale or the column stops being readable. Prose wants :func:`money_phrase`.

    ``currency`` comes from ``ImpactModel.currency`` by way of the payload; it is the only
    thing that decides which symbol is drawn and whether the grouping is ``##,##,###`` or
    ``###,###,###``.
    """
    return format_money(value, currency, missing=missing)


def money_phrase(value: Any, currency: str | None = None, *, missing: str = NOT_MEASURED) -> str:
    """An amount the way a person would say it out loud: ``₹1.23 crore``, ``$4.5k``.

    The form for prose and headline figures. Nobody reading a sentence wants to count the
    digits in ``₹12,34,56,789`` to find out whether it is twelve crore or one; the
    scale word is the number's meaning, so in a sentence it is what gets written.
    Amounts below the smallest scale word fall back to :func:`money` automatically.
    """
    return format_money_compact(value, currency, missing=missing)


def pct(value: Any, digits: int = 0, *, missing: str = NOT_MEASURED) -> str:
    """A fraction in ``[0, 1]`` rendered as a percentage."""
    number = _finite(value)
    if number is None:
        return missing
    return f"{number * 100:.{digits}f}%"


def num(value: Any, *, missing: str = NOT_RECORDED) -> str:
    """An integer with thousands separators."""
    number = _finite(value)
    if number is None:
        return missing
    return f"{number:,.0f}"


def hours(value: Any, *, missing: str = NOT_MEASURED) -> str:
    """A duration in engineering hours, written so 1 and 0.5 both read correctly."""
    number = _finite(value)
    if number is None:
        return missing
    if abs(number - round(number)) < 1e-9:
        whole = int(round(number))
        return f"{whole:,} hour" if whole == 1 else f"{whole:,} hours"
    return f"{number:,.1f} hours"


def plural(count: Any, singular: str, plural_form: str | None = None) -> str:
    """``singular`` when the count is exactly one, otherwise the plural form."""
    number = _finite(count)
    word = plural_form if plural_form is not None else f"{singular}s"
    return singular if number is not None and abs(number - 1.0) < 1e-9 else word


def count_phrase(count: Any, singular: str, plural_form: str | None = None) -> str:
    """``"3 findings"``, ``"1 finding"``, ``"no findings"``."""
    number = _finite(count)
    if number is None:
        return f"an unrecorded number of {plural_form or singular + 's'}"
    if abs(number) < 1e-9:
        return f"no {plural_form or singular + 's'}"
    return f"{number:,.0f} {plural(number, singular, plural_form)}"


def join_phrase(items: Sequence[str], *, conjunction: str = "and", empty: str = "") -> str:
    """``"a, b and c"``. Deterministic: the caller's order is preserved."""
    parts = [item for item in items if item]
    if not parts:
        return empty
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} {conjunction} {parts[1]}"
    return ", ".join(parts[:-1]) + f" {conjunction} {parts[-1]}"


def titlecase_first(text: str) -> str:
    """Capitalise the first character and leave the rest alone (names keep their case)."""
    return text[:1].upper() + text[1:] if text else text


def yes_no(value: Any, *, yes: str = "yes", no: str = "no") -> str:
    return yes if bool(value) else no


def estimate_note() -> str:
    return (
        "Every probability and every money figure in this report is a modelled estimate. "
        "None of them is a measured loss, and none of them is evidence that an attack "
        "occurred."
    )


# ---------------------------------------------------------------------------
# Finding-level sentences. `finding` is a WebFinding; attributes are read defensively so a
# payload written by hand does not have to fill every field.
# ---------------------------------------------------------------------------


def _get(finding: Any, name: str, default: Any = None) -> Any:
    value = getattr(finding, name, default)
    return default if value is None else value


def severity_phrase(finding: Any) -> str:
    """The scanner's own severity word, attributed to the scanner rather than asserted."""
    word = str(_get(finding, "scanner_severity", "")).strip().lower()
    if not word:
        return "The scanner recorded no severity for this finding."
    if word in _SEVERITY_WORDS:
        return f"The scanner rated this {word}."
    return f"The scanner recorded a severity of '{word}'."


def cvss_phrase(finding: Any) -> str:
    """CVSS base score with its version and, where present, the cross-source agreement."""
    base = _finite(_get(finding, "cvss_base"))
    if base is None:
        return "No CVSS base score was carried for this finding."
    version = str(_get(finding, "cvss_version", "")).strip()
    head = f"CVSS base score {base:.1f}"
    if version:
        head += f" (version {version})"
    agreement = _finite(_get(finding, "cvss_source_agreement"))
    if agreement is None:
        return (
            head
            + ". Agreement between scoring sources was not measured for this finding, so the "
            "score is carried as a single opinion rather than a consensus."
        )
    if agreement >= 0.99:
        return head + f". The scoring sources agreed exactly (agreement {agreement:.2f})."
    if agreement >= 0.8:
        return head + f". The scoring sources broadly agreed (agreement {agreement:.2f})."
    return (
        head
        + f". The scoring sources disagreed materially (agreement {agreement:.2f} on a scale "
        "where 1.00 is exact agreement), so the base score is a weaker piece of evidence here "
        "than it looks."
    )


def kev_phrase(finding: Any) -> str:
    """KEV membership. Never inflated: not in the catalogue is stated as exactly that."""
    if not bool(_get(finding, "kev", False)):
        return "It is not listed in the CISA Known Exploited Vulnerabilities catalogue."
    if bool(_get(finding, "kev_ransomware", False)):
        return (
            "It is listed in the CISA Known Exploited Vulnerabilities catalogue, with known "
            "ransomware use recorded against it."
        )
    return "It is listed in the CISA Known Exploited Vulnerabilities catalogue."


def epss_phrase(finding: Any) -> str:
    """EPSS score and percentile, described as a forecast rather than an observation."""
    score = _finite(_get(finding, "epss"))
    if score is None:
        return "No EPSS forecast was available for this finding."
    percentile = _finite(_get(finding, "epss_percentile"))
    text = f"EPSS puts the chance of exploitation activity in the next 30 days at {pct(score, 1)}"
    if percentile is not None:
        text += f", which is the {pct(percentile, 0)} percentile of all scored vulnerabilities"
    return text + "."


def maturity_phrase(finding: Any) -> str:
    """Exploit maturity and the count of exploit records behind it."""
    maturity = str(_get(finding, "exploit_maturity", "")).strip().lower()
    count = _finite(_get(finding, "exploit_count", 0)) or 0.0
    if not maturity or maturity == "unknown":
        head = "Exploit maturity is unknown"
    else:
        head = f"Exploit maturity is recorded as {maturity.replace('_', ' ')}"
    if count <= 0:
        return head + ", and no public exploit record was found."
    return head + f", from {count_phrase(count, 'public exploit record')}."


def applicability_phrase(finding: Any) -> str:
    """Applicability verdict together with the version reasoning that produced it."""
    verdict = str(_get(finding, "applicability", "")).strip().lower().replace("_", " ")
    probability = _finite(_get(finding, "p_applicable"))
    match = str(_get(finding, "version_match", "")).strip().lower()

    if not verdict:
        head = "Applicability was not assessed"
    else:
        head = f"Applicability verdict: {verdict}"
    if probability is not None:
        head += f" (estimated probability {pct(probability, 0)})"
    head += "."

    if match == "match":
        reason = (
            " The observed component version falls inside the affected range recorded for the "
            "vulnerability, which is version evidence from a curated feed and cannot be "
            "overridden by page text."
        )
    elif match == "mismatch":
        reason = (
            " The observed component version falls outside the affected range, so this finding "
            "is retained for review rather than treated as exploitable here."
        )
    elif match == "unknown":
        reason = (
            " No version could be matched against the affected range, so the verdict rests on "
            "preconditions rather than on version evidence."
        )
    else:
        reason = " No version comparison was recorded."
    return head + reason


def exposure_phrase(finding: Any) -> str:
    """Where the affected endpoint sits and what reaching it requires."""
    method = str(_get(finding, "endpoint_method", "")).strip()
    path = str(_get(finding, "endpoint_path", "")).strip()
    where = f"{method} {path}".strip() or "an endpoint the payload does not name"
    auth = _finite(_get(finding, "auth_required", 0)) or 0.0
    function = str(_get(finding, "endpoint_function", "")).strip().replace("_", " ")
    gate = {
        0: "reachable without authenticating",
        1: "reachable by any authenticated user",
        2: "reachable only by an administrator",
        3: "reachable only with operating-system level access",
    }.get(int(auth), "of unrecorded authentication level")
    text = f"The finding is on {where}, which is {gate}"
    if function and function != "unknown":
        text += f" and is classified as {function}"
    exposure = _finite(_get(finding, "exposure"))
    if exposure is not None:
        text += f". Structural exposure is scored {exposure:.2f} on a 0 to 1 scale"
    return text + "."


def impact_phrase(finding: Any, currency: str | None = None) -> str:
    """The expected-loss identity, spelled out with this finding's own numbers."""
    probability = _finite(_get(finding, "p_exploit"))
    impact = _finite(_get(finding, "impact"))
    loss = _finite(_get(finding, "expected_loss"))
    if probability is None or impact is None:
        return (
            "Expected loss could not be decomposed for this finding: the payload carries "
            f"{'no probability' if probability is None else 'no impact figure'}."
        )
    return (
        f"Expected loss is the product of the two: an estimated {pct(probability, 0)} chance of "
        f"exploitation over the attacker's horizon, against {money_phrase(impact, currency)} of "
        f"business impact if it happens, giving {money_phrase(loss, currency)}."
    )


def chain_phrase(finding: Any, currency: str | None = None) -> str:
    """What this finding contributes to reachable compromise beyond its own loss."""
    delta = _finite(_get(finding, "chain_delta"))
    adjusted = _finite(_get(finding, "chain_adjusted"))
    if delta is None or delta <= 0:
        if bool(_get(finding, "is_chokepoint", False)):
            return (
                "It is marked a chain chokepoint, but no reachable-risk contribution was "
                "carried in the payload."
            )
        return "It adds no measured reachable-risk contribution beyond its own expected loss."
    hops = _finite(_get(finding, "hops_from_entry"))
    text = (
        f"Patching it removes {money_phrase(delta, currency)} of reachable risk that no other "
        f"finding accounts for, which is why its chain-adjusted figure is "
        f"{money_phrase(adjusted, currency)}"
    )
    if hops is not None and hops > 0:
        text += f". It sits {count_phrase(hops, 'hop')} from the attacker's entry point"
    if bool(_get(finding, "is_chokepoint", False)):
        text += ". It is a chokepoint: several paths pass through it"
    return text + "."


def trust_phrase(finding: Any) -> str:
    """How much untrusted evidence was allowed to move this finding, and what it triggered."""
    share = _finite(_get(finding, "untrusted_influence_share"))
    tier = _finite(_get(finding, "max_tier_used"))
    signals = _finite(_get(finding, "injection_signals", 0)) or 0.0
    tier_names = {
        0: "operator configuration",
        1: "curated feeds",
        2: "scanner observation",
        3: "reference pages fetched from the internet",
        4: "content authored by the target application",
    }
    parts: list[str] = []
    if tier is not None:
        parts.append(
            f"The highest-risk evidence used was tier {int(tier)}, "
            f"{tier_names.get(int(tier), 'of an unrecorded kind')}."
        )
    if share is not None:
        parts.append(
            f"Untrusted evidence accounts for {pct(share, 0)} of the attribution behind this "
            "position, against a configured cap."
        )
    if signals > 0:
        parts.append(
            f"The sandbox recorded {count_phrase(signals, 'instruction-injection signal')} in "
            "the text attached to this finding; the text was redacted before any model saw it."
        )
    alerts = list(_get(finding, "alerts", ()) or ())
    if alerts:
        parts.append(
            f"{count_phrase(len(alerts), 'manipulation alert')} fired on this finding: "
            + join_phrase([str(alert) for alert in alerts])
            + "."
        )
    if not parts:
        return "No trust or manipulation signals were recorded for this finding."
    return " ".join(parts)


def why_bullets(finding: Any, currency: str | None = None) -> list[str]:
    """Why this finding is where it is.

    Reason codes only. These are templated strings produced by the explanation layer from
    feature attributions; no free text written by a model ever reaches this function. When a
    finding carries no reason codes, factual statements are derived from its own structured
    fields instead, and nothing is inferred beyond them.
    """
    codes = [str(code).strip() for code in (_get(finding, "reason_codes", ()) or ()) if str(code).strip()]
    if codes:
        return codes

    derived: list[str] = []
    if bool(_get(finding, "kev", False)):
        derived.append("Listed in the CISA Known Exploited Vulnerabilities catalogue.")
    epss = _finite(_get(finding, "epss"))
    if epss is not None:
        derived.append(f"EPSS {epss:.2f} for the associated vulnerability.")
    loss = _finite(_get(finding, "expected_loss"))
    probability = _finite(_get(finding, "p_exploit"))
    if loss is not None and probability is not None:
        derived.append(
            f"Expected loss {money_phrase(loss, currency)} at an estimated {pct(probability, 0)} "
            "probability of "
            "exploitation."
        )
    delta = _finite(_get(finding, "chain_delta"))
    if delta is not None and delta > 0:
        derived.append(f"Removing it removes {money_phrase(delta, currency)} of reachable risk.")
    if not derived:
        derived.append(
            "No reason codes were recorded for this finding, and its structured fields carry "
            "no figure to quote. Its position rests on the ordering alone."
        )
    return derived


def one_line_headline(finding: Any) -> str:
    """One line naming the finding and where it is, for a heading."""
    name = str(_get(finding, "name", "")).strip() or str(_get(finding, "finding_id", "")).strip()
    method = str(_get(finding, "endpoint_method", "")).strip()
    path = str(_get(finding, "endpoint_path", "")).strip()
    where = f"{method} {path}".strip()
    return f"{name} at {where}" if where else name


# ---------------------------------------------------------------------------
# Attack-chain sentences
# ---------------------------------------------------------------------------


def state_phrase(node_id: str, *, entry: bool = False) -> str:
    """Turn ``state:shop.example.com:ADMIN`` into something a reader can follow."""
    asset, privilege = _split_state(node_id)
    if asset == "internet" and privilege == 0:
        return "an unauthenticated attacker on the internet" if entry else "the public internet"
    template = _PRIVILEGE_PHRASES.get(privilege)
    if template is None:
        return node_id
    return template.format(asset=asset)


def _split_state(node_id: str) -> tuple[str, int]:
    """``("shop.example.com", 2)`` for ``"state:shop.example.com:ADMIN"``."""
    text = str(node_id or "")
    parts = text.split(":")
    if len(parts) >= 3 and parts[0] == "state":
        asset = ":".join(parts[1:-1])
        name = parts[-1].strip().upper()
        if name in _PRIVILEGE_BY_NAME:
            return asset, _PRIVILEGE_BY_NAME[name]
        try:
            return asset, int(name)
        except ValueError:
            return asset, -1
    return text, -1


def path_sentence(
    path: Any,
    *,
    finding_names: dict[str, str] | None = None,
    entry_node: str = "",
    currency: str | None = None,
) -> str:
    """One readable sentence for one attack path.

    Reads: who starts where, what they reach, how likely the whole path is, what sits at the
    end of it, and which findings the path depends on.
    """
    names = finding_names or {}
    nodes = [str(node) for node in (getattr(path, "nodes", ()) or ())]
    probability = _finite(getattr(path, "probability", None))
    target_value = _finite(getattr(path, "target_value", None))
    expected = _finite(getattr(path, "expected_value", None))
    finding_ids = [str(item) for item in (getattr(path, "finding_ids", ()) or ())]

    if not nodes:
        return "A path was recorded with no states, so it cannot be described."

    start = nodes[0]
    is_entry = bool(entry_node) and start == entry_node
    opening = titlecase_first(state_phrase(start, entry=True)) if (is_entry or _split_state(start)[0] == "internet") else (
        "An attacker holding " + state_phrase(start)
    )

    if len(nodes) == 1:
        body = f"{opening} already holds the state that carries the value"
    else:
        hops = [state_phrase(node) for node in nodes[1:]]
        body = f"{opening} can reach {hops[0]}"
        for hop in hops[1:]:
            body += f", and from there {hop}"

    tail_parts: list[str] = []
    if probability is not None:
        tail_parts.append(
            f"the whole path is estimated at {pct(probability, 0)} likely under the attacker "
            "model in force"
        )
    if target_value is not None:
        tail_parts.append(
            f"the state at the end of it is valued at {money_phrase(target_value, currency)}"
        )

    sentence = body
    if tail_parts:
        sentence += "; " + join_phrase(tail_parts)
    sentence += "."
    if expected is not None:
        sentence += f" That puts {money_phrase(expected, currency)} at risk on this path alone."

    if finding_ids:
        labelled = [names.get(item, item) for item in finding_ids]
        if len(labelled) == 1:
            sentence += (
                f" The path exists because of one finding, {labelled[0]}; closing it breaks the "
                "path."
            )
        else:
            quantifier = "both" if len(labelled) == 2 else f"all {len(labelled)} of"
            sentence += (
                f" The path depends on {quantifier} "
                + join_phrase(labelled)
                + ", so closing any one of them breaks it."
            )
    return sentence


# ---------------------------------------------------------------------------
# Portfolio-level sentences
# ---------------------------------------------------------------------------


def concentration_sentence(
    *,
    n_findings: int,
    total_loss: float,
    top_decile_share: float | None,
    half_count: int | None,
    currency: str | None = None,
) -> str:
    """How concentrated the expected loss is, stated with the numbers that make it true."""
    if n_findings <= 0 or total_loss <= 0:
        return (
            "Expected loss could not be concentrated across findings, because the payload "
            "carries no positive expected loss."
        )
    parts: list[str] = []
    if top_decile_share is not None:
        parts.append(
            f"The worst tenth of the findings carries {pct(top_decile_share, 0)} of the total "
            "estimated expected loss"
        )
    if half_count is not None and half_count > 0:
        clause = (
            f"half of the total sits in {count_phrase(half_count, 'finding')} out of "
            f"{num(n_findings)}"
        )
        parts.append(clause if parts else titlecase_first(clause))
    if not parts:
        return (
            f"The {count_phrase(n_findings, 'finding')} carry "
            f"{money_phrase(total_loss, currency)} of estimated expected loss in total; its "
            "distribution across them was not measured."
        )
    return join_phrase(parts, conjunction="and") + "."
