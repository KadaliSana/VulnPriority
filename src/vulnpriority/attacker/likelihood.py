"""Evidence assembly for the attacker model (DESIGN.md 3.6, Gap 2).

The likelihood model is deliberately a *named* logistic model rather than an opaque
score: the whole point of Gap 2 is that the adversary is code, and that every input to
``P(exploit | evidence, attacker)`` can be pointed at and argued with. This module owns
the "evidence" half of that split - turning the scanner finding, the curated feed
intelligence and the Component A assessments into the exact vector of named terms that
``vulnpriority.attacker.model.p_exploit`` weights.

Two terms in the DESIGN formula are attacker-relative and therefore cannot be finalised
here (this module has no ``AttackerModel``):

* ``asset_criticality`` is multiplied by ``target_preference[function]``, and
* ``privileges_required`` is measured relative to ``entry_privilege``.

So the evidence vector carries the *raw* criticality plus ``target_function_ord`` (the
ordinal of the endpoint function in :data:`EVIDENCE_FUNCTIONS`) and the *raw* privilege
ordinal; :func:`vulnpriority.attacker.model.p_exploit` applies the attacker-relative part.
Passing an ``attacker`` here is optional and only fills the informational ``skill``,
``resources`` and ``entry_privilege`` entries - the model always trusts its own attacker.
"""

from __future__ import annotations

import math
from datetime import date

from vulnpriority.core.enums import (
    AttackComplexity,
    EndpointFunction,
    ExploitMaturity,
    UserInteraction,
)
from vulnpriority.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    AttackerModel,
    Endpoint,
    ExploitabilityAssessment,
    Finding,
    VulnIntel,
)

__all__ = [
    "EVIDENCE_TERMS",
    "EVIDENCE_FUNCTIONS",
    "EPSS_CLIP",
    "EPSS_LOGIT_SCALE",
    "NEUTRAL_EPSS",
    "logit",
    "function_ordinal",
    "function_from_ordinal",
    "epss_logit_term",
    "build_evidence",
]

#: EPSS is clipped before the logit so a 0.0 or 1.0 snapshot cannot produce +/-inf.
EPSS_CLIP: tuple[float, float] = (1e-6, 1.0 - 1e-6)

#: DESIGN.md divides the EPSS logit by 10 so its weight stays on the same scale as the
#: other (0/1 or [0,1]) terms.
EPSS_LOGIT_SCALE: float = 10.0

#: Used when no EPSS snapshot exists for the finding's CVEs. ``logit(0.5) == 0``, so an
#: unknown EPSS contributes nothing instead of pretending the CVE is un-exploitable.
NEUTRAL_EPSS: float = 0.5

#: Canonical ordering used to encode :class:`EndpointFunction` as a float in the evidence
#: vector. Frozen here so an evidence dict is portable between processes.
EVIDENCE_FUNCTIONS: tuple[EndpointFunction, ...] = tuple(EndpointFunction)

#: Every key :func:`build_evidence` produces, in formula order.
EVIDENCE_TERMS: tuple[str, ...] = (
    "epss_logit",
    "kev",
    "kev_ransomware",
    "exploit_maturity",
    "feasibility",
    "applicability",
    "exposure",
    "asset_criticality",
    "target_function_ord",
    "complexity_high",
    "user_interaction",
    "privileges_required",
    "entry_privilege",
    "skill",
    "resources",
)


def logit(p: float) -> float:
    """Log-odds of ``p``, clipped to :data:`EPSS_CLIP` so the result is always finite."""
    low, high = EPSS_CLIP
    clipped = min(max(p, low), high)
    return math.log(clipped / (1.0 - clipped))


def function_ordinal(function: EndpointFunction) -> float:
    """Encode an endpoint function as a float so it fits in the evidence vector."""
    return float(EVIDENCE_FUNCTIONS.index(function))


def function_from_ordinal(ordinal: float) -> EndpointFunction:
    """Inverse of :func:`function_ordinal`; out-of-range values decode to ``UNKNOWN``."""
    index = int(round(ordinal))
    if 0 <= index < len(EVIDENCE_FUNCTIONS):
        return EVIDENCE_FUNCTIONS[index]
    return EndpointFunction.UNKNOWN


def epss_logit_term(epss: float | None) -> float:
    """``logit(clip(epss)) / 10`` exactly as DESIGN.md 3.6 writes it."""
    score = NEUTRAL_EPSS if epss is None else epss
    return logit(score) / EPSS_LOGIT_SCALE


def _usable_intel(intel: tuple[VulnIntel, ...], finding: Finding, as_of: date) -> list[VulnIntel]:
    """Intel records that are for this finding's CVEs and are not dated after ``as_of``.

    Filtering here rather than trusting the caller keeps the as-of discipline (rule 2 of
    the architecture) local to the module that consumes the numbers.
    """
    wanted = {cve.upper() for cve in finding.cve_ids}
    usable: list[VulnIntel] = []
    for record in intel:
        if record.as_of > as_of:
            continue
        if wanted and record.cve_id.upper() not in wanted:
            continue
        usable.append(record)
    return usable


def _best_epss(records: list[VulnIntel], as_of: date) -> float | None:
    """Highest EPSS score across the finding's CVEs; the attacker picks the easiest one."""
    scores = [
        record.epss.score
        for record in records
        if record.epss is not None and record.epss.as_of <= as_of
    ]
    return max(scores) if scores else None


def _kev_flags(records: list[VulnIntel], as_of: date) -> tuple[bool, bool]:
    """``(in_kev, known_ransomware_use)`` honouring ``date_added <= as_of``."""
    in_kev = False
    ransomware = False
    for record in records:
        kev = record.kev
        if kev is None or not kev.in_kev:
            continue
        if kev.date_added is not None and kev.date_added > as_of:
            continue
        in_kev = True
        ransomware = ransomware or kev.known_ransomware_use
    return in_kev, ransomware


def _feed_maturity(records: list[VulnIntel], as_of: date) -> ExploitMaturity:
    """Highest exploit maturity attested by the curated feeds as of the cut-off."""
    best = ExploitMaturity.UNKNOWN
    for record in records:
        for exploit in record.exploits:
            if exploit.published is not None and exploit.published > as_of:
                continue
            if exploit.maturity > best:
                best = exploit.maturity
    return best


def build_evidence(
    finding: Finding,
    intel: tuple[VulnIntel, ...],
    asset: AssetCriticality,
    exploitability: ExploitabilityAssessment,
    applicability: ApplicabilityAssessment,
    endpoint: Endpoint,
    as_of: date,
    attacker: AttackerModel | None = None,
) -> dict[str, float]:
    """Assemble the named evidence vector of DESIGN.md 3.6 for one finding.

    Every key in :data:`EVIDENCE_TERMS` is present, so the model never has to guess at a
    missing term and an audit trail can be produced for a finding with no CVE at all.

    ``attacker`` is optional and only populates the attacker-owned informational entries
    (``skill``, ``resources``, ``entry_privilege``); the weighting in
    :func:`vulnpriority.attacker.model.p_exploit` always uses its own attacker.
    """
    records = _usable_intel(intel, finding, as_of)
    in_kev, ransomware = _kev_flags(records, as_of)
    maturity = max(exploitability.exploit_maturity, _feed_maturity(records, as_of))

    return {
        "epss_logit": epss_logit_term(_best_epss(records, as_of)),
        "kev": 1.0 if in_kev else 0.0,
        "kev_ransomware": 1.0 if ransomware else 0.0,
        "exploit_maturity": float(int(maturity)),
        "feasibility": float(exploitability.exploit_feasibility),
        # centred and rescaled to [-1, 1] so "uncertain" (0.5) is neutral, not positive
        "applicability": (float(applicability.p_applicable) - 0.5) * 2.0,
        "exposure": float(asset.exposure),
        "asset_criticality": float(asset.criticality),
        "target_function_ord": function_ordinal(asset.function),
        "complexity_high": 1.0 if exploitability.attack_complexity == AttackComplexity.HIGH else 0.0,
        "user_interaction": 1.0 if exploitability.user_interaction == UserInteraction.REQUIRED else 0.0,
        "privileges_required": float(int(exploitability.privileges_required)),
        "entry_privilege": float(int(attacker.entry_privilege)) if attacker is not None else 0.0,
        "skill": float(attacker.skill) if attacker is not None else 0.5,
        "resources": float(attacker.resources) if attacker is not None else 0.5,
    }
