"""The explicit attacker model (DESIGN.md 3.6, Gap 2).

``p_exploit`` is the logistic model written out in DESIGN.md 3.6, implemented term for
term. Every weighted contribution is written into ``ExploitLikelihood.log_odds_terms``
so the resulting probability is auditable: ``sum(log_odds_terms.values()) == z`` by
construction, and :func:`explain_terms` turns those terms into reason codes.

The horizon factor deserves a note. DESIGN.md defines
``horizon_factor = 1 - exp(-horizon_days / 365)`` "normalised so that the default 90-day
horizon leaves the weights interpretable", which is implemented here as a ratio against
the 90-day reference: a 90-day attacker gets a factor of exactly 1.0 and the tuned
weights mean what they say, a 30-day mass scanner gets less, a 365-day APT gets more.
Because that factor can exceed 1, the result is clipped to ``[min_p, max_p]`` afterwards
as well - which is also what keeps ``p_exploit`` inside the [0, 1] contract of
:class:`ExploitLikelihood`.
"""

from __future__ import annotations

import math

from vulnprio.core.enums import EndpointFunction, ExploitMaturity
from vulnprio.core.models import AttackerModel, ExploitLikelihood
from vulnprio.attacker.likelihood import function_from_ordinal, function_ordinal

#: Ordinal used when an evidence vector omits ``target_function_ord`` entirely.
_UNKNOWN_FUNCTION_ORD: float = function_ordinal(EndpointFunction.UNKNOWN)

__all__ = [
    "HORIZON_REFERENCE_DAYS",
    "LOG_ODDS_TERMS",
    "TERM_LABELS",
    "sigmoid",
    "horizon_factor",
    "log_odds_terms",
    "p_exploit",
    "explain_terms",
]

#: The horizon at which ``horizon_factor`` is exactly 1.0 and the weights are as tuned.
HORIZON_REFERENCE_DAYS: int = 90

#: Keys written into ``ExploitLikelihood.log_odds_terms``, in formula order.
LOG_ODDS_TERMS: tuple[str, ...] = (
    "intercept",
    "epss_logit",
    "kev",
    "kev_ransomware",
    "exploit_maturity",
    "feasibility",
    "applicability",
    "exposure",
    "asset_criticality",
    "complexity_high",
    "user_interaction",
    "privileges_required",
    "skill",
    "resources",
)

#: Human-readable label per term, used to template reason codes. Model free text never
#: reaches a user (Goal 5), so these operator-authored strings are the only wording.
TERM_LABELS: dict[str, str] = {
    "intercept": "attacker base rate",
    "epss_logit": "EPSS exploitation probability",
    "kev": "listed in CISA KEV",
    "kev_ransomware": "known ransomware use",
    "exploit_maturity": "public exploit maturity",
    "feasibility": "assessed exploit feasibility",
    "applicability": "applicability to the observed stack",
    "exposure": "internet exposure of the endpoint",
    "asset_criticality": "asset criticality weighted by attacker target preference",
    "complexity_high": "high attack complexity",
    "user_interaction": "user interaction required",
    "privileges_required": "privileges required above the attacker's entry privilege",
    "skill": "attacker skill",
    "resources": "attacker resources",
}


def sigmoid(z: float) -> float:
    """Numerically stable logistic function (no overflow for large negative ``z``)."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def horizon_factor(horizon_days: int, reference_days: int = HORIZON_REFERENCE_DAYS) -> float:
    """``(1 - exp(-h/365)) / (1 - exp(-reference/365))``; 1.0 at the reference horizon."""
    numerator = 1.0 - math.exp(-float(horizon_days) / 365.0)
    denominator = 1.0 - math.exp(-float(reference_days) / 365.0)
    return numerator / denominator


def _maturity_weight(attacker: AttackerModel, ordinal: float) -> float:
    """Weight for an exploit-maturity level, tolerating int-keyed YAML overrides."""
    index = int(round(ordinal))
    index = min(max(index, int(ExploitMaturity.UNKNOWN)), int(ExploitMaturity.WEAPONIZED))
    maturity = ExploitMaturity(index)
    weights = attacker.w_exploit_maturity
    if maturity in weights:
        return float(weights[maturity])
    return float(weights.get(index, 0.0))  # type: ignore[arg-type]


def _preference(attacker: AttackerModel, ordinal: float) -> float:
    """Target preference multiplier for an endpoint function; 1.0 when unstated."""
    function = function_from_ordinal(ordinal)
    preference = attacker.target_preference
    if function in preference:
        return float(preference[function])
    return float(preference.get(function.value, 1.0))  # type: ignore[arg-type]


def log_odds_terms(attacker: AttackerModel, evidence: dict[str, float]) -> dict[str, float]:
    """Weighted contribution of every named term; they sum to ``z``.

    Split out from :func:`p_exploit` because the audit trail is the product here: the
    ranker, the explainer and the adversarial evaluator all read these terms.
    """
    get = evidence.get
    privileges_above_entry = max(
        0.0, float(get("privileges_required", 0.0)) - float(int(attacker.entry_privilege))
    )
    return {
        "intercept": float(attacker.w_intercept),
        "epss_logit": attacker.w_epss_logit * float(get("epss_logit", 0.0)),
        "kev": attacker.w_kev * float(get("kev", 0.0)),
        "kev_ransomware": attacker.w_kev_ransomware * float(get("kev_ransomware", 0.0)),
        "exploit_maturity": _maturity_weight(attacker, float(get("exploit_maturity", 0.0))),
        "feasibility": attacker.w_feasibility * float(get("feasibility", 0.0)),
        "applicability": attacker.w_applicability * float(get("applicability", 0.0)),
        "exposure": attacker.w_exposure * float(get("exposure", 0.0)),
        "asset_criticality": (
            attacker.w_asset_criticality
            * float(get("asset_criticality", 0.0))
            * _preference(attacker, float(get("target_function_ord", _UNKNOWN_FUNCTION_ORD)))
        ),
        "complexity_high": attacker.w_complexity_high * float(get("complexity_high", 0.0)),
        "user_interaction": attacker.w_user_interaction * float(get("user_interaction", 0.0)),
        "privileges_required": attacker.w_privileges_required * privileges_above_entry,
        "skill": attacker.w_skill * float(attacker.skill),
        "resources": attacker.w_resources * float(attacker.resources),
    }


def p_exploit(
    attacker: AttackerModel,
    evidence: dict[str, float],
    horizon_days: int | None = None,
    *,
    finding_id: str = "",
) -> ExploitLikelihood:
    """``P(exploit | evidence, attacker)`` over the attacker's horizon.

    Implements DESIGN.md 3.6 exactly:
    ``p = clip(sigmoid(z), min_p, max_p) * horizon_factor``, re-clipped to
    ``[min_p, max_p]`` so a long horizon cannot push the probability out of range.
    ``p_exploit_uncapped`` is the same quantity without the attacker's caps, kept so an
    evaluation can tell "the model said 0.999 and we capped it" from "the model said the
    cap value".

    ``finding_id`` only labels the returned record; it cannot be carried in ``evidence``
    because that vector is numeric by contract.
    """
    horizon = int(horizon_days if horizon_days is not None else attacker.horizon_days)
    horizon = max(horizon, 1)

    terms = log_odds_terms(attacker, evidence)
    z = math.fsum(terms.values())

    raw = sigmoid(z)
    factor = horizon_factor(horizon)
    capped = min(max(raw, attacker.min_p), attacker.max_p) * factor
    final = min(max(capped, attacker.min_p), attacker.max_p)
    uncapped = min(max(raw * factor, 0.0), 1.0)

    return ExploitLikelihood(
        finding_id=finding_id,
        attacker=attacker.name,
        p_exploit=final,
        p_exploit_uncapped=uncapped,
        log_odds_terms=terms,
        horizon_days=horizon,
    )


def explain_terms(likelihood: ExploitLikelihood, max_terms: int = 6) -> list[str]:
    """Reason codes for a likelihood, strongest contribution first.

    Templated operator text only: the numbers come from ``log_odds_terms``, the wording
    from :data:`TERM_LABELS`, so nothing an untrusted document wrote can reach the queue.
    """
    contributions = [
        (name, value)
        for name, value in likelihood.log_odds_terms.items()
        if name != "intercept" and abs(value) > 1e-9
    ]
    contributions.sort(key=lambda item: (-abs(item[1]), item[0]))

    codes = [
        f"{TERM_LABELS.get(name, name)} ({value:+.2f} log-odds)"
        for name, value in contributions[:max_terms]
    ]
    codes.append(
        f"attacker {likelihood.attacker} over {likelihood.horizon_days} days: "
        f"p_exploit = {likelihood.p_exploit:.3f}"
    )
    return codes
