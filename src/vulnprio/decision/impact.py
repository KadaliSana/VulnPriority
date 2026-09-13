"""Monetary business impact (DESIGN.md 3.6, Gap 9).

Impact is money, not a severity word. Which money is ``ImpactModel.currency``'s business
and no field name's: nothing here is called ``_usd``, so nothing here can claim a
denomination the numbers do not have. The formula is the one written in DESIGN.md 3.6:
records exposed x cost per record x the confidentiality impact x data sensitivity, plus
an integrity loss, plus downtime x hourly cost, all lifted by a regulatory multiplier
and then a reputational fraction, and finally capped.

One implementation decision worth stating: DESIGN.md applies ``regulatory_multiplier``
to the *subtotal*, so the three named components are pre-multiplier while the
reputational term is post-multiplier. Storing them that way would make
:class:`BusinessImpact` non-additive (its parts would not sum to its total), which is
exactly the kind of number nobody can audit. The multiplier is therefore distributed
across the three components, which is algebraically identical for the total and leaves
``confidentiality + integrity + availability + reputational == total`` whenever the cap
does not bind.
"""

from __future__ import annotations

from typing import Any, Mapping

from vulnprio.core.enums import EndpointFunction, PrivilegeLevel
from vulnprio.core.money import format_money
from vulnprio.core.models import (
    AssetCriticality,
    BusinessImpact,
    Endpoint,
    ExploitabilityAssessment,
    Finding,
    ImpactModel,
)

__all__ = ["estimate_impact", "impact_components", "lookup"]


def lookup(mapping: Mapping[Any, float] | Mapping[Any, int], key: Any, default: float) -> float:
    """Read an enum-keyed impact table tolerantly.

    Impact tables come from YAML, so depending on how pydantic coerced them a key may be
    the enum, its ``.value`` string, or a bare int. Being forgiving here is cheaper than
    a silent zero that quietly deletes a crore of impact.
    """
    for candidate in (key, getattr(key, "value", None)):
        if candidate is not None and candidate in mapping:
            return float(mapping[candidate])  # type: ignore[index]
    if isinstance(key, int):
        if key in mapping:
            return float(mapping[key])  # type: ignore[index]
    return float(default)


def impact_components(
    asset: AssetCriticality,
    exploitability: ExploitabilityAssessment,
    impact_model: ImpactModel,
) -> tuple[float, float, float]:
    """The three pre-multiplier components ``(confidentiality, integrity, availability)``."""
    function: EndpointFunction = asset.function
    privilege: PrivilegeLevel = exploitability.privilege_gained

    records = lookup(impact_model.records_by_function, function, 0.0)
    confidentiality = (
        records
        * impact_model.cost_per_record
        * exploitability.impact_c
        * asset.data_sensitivity
    )

    integrity = lookup(impact_model.integrity_loss_by_function, function, 0.0) * exploitability.impact_i

    downtime_hours = lookup(impact_model.downtime_hours_by_privilege, privilege, 0.0)
    availability = downtime_hours * impact_model.downtime_cost_per_hour * exploitability.impact_a

    return confidentiality, integrity, availability


def estimate_impact(
    finding: Finding,
    endpoint: Endpoint,
    asset: AssetCriticality,
    exploitability: ExploitabilityAssessment,
    impact_model: ImpactModel,
) -> BusinessImpact:
    """Monetary impact of this finding being exploited on this endpoint.

    ``impact_model.asset_overrides[endpoint.endpoint_id]`` is operator-tier truth and
    replaces the computed total. It is still held to ``max_impact``, because both
    numbers are operator-set and the cap is the model's stated ceiling: the invariant
    ``total <= max_impact`` holds for every finding, which is what downstream
    normalisation and the knapsack rely on.
    """
    pre_c, pre_i, pre_a = impact_components(asset, exploitability, impact_model)

    multiplier = impact_model.regulatory_multiplier
    confidentiality = pre_c * multiplier
    integrity = pre_i * multiplier
    availability = pre_a * multiplier

    subtotal = confidentiality + integrity + availability
    reputational = subtotal * impact_model.reputational_fraction
    computed_total = subtotal + reputational

    currency = impact_model.currency
    override = impact_model.asset_overrides.get(endpoint.endpoint_id)
    if override is not None:
        total = min(max(float(override), 0.0), impact_model.max_impact)
        rationale = (
            f"operator override for {endpoint.endpoint_id}: "
            f"{format_money(float(override), currency)} "
            f"(computed {format_money(computed_total, currency)}, cap "
            f"{format_money(impact_model.max_impact, currency)})"
        )
    else:
        total = min(computed_total, impact_model.max_impact)
        rationale = (
            f"{asset.function.value} endpoint, regulatory x{multiplier:g}, "
            f"reputational +{impact_model.reputational_fraction:.0%}"
            + (" (capped)" if computed_total > impact_model.max_impact else "")
        )

    return BusinessImpact(
        finding_id=finding.finding_id,
        confidentiality=confidentiality,
        integrity=integrity,
        availability=availability,
        reputational=reputational,
        total=total,
        rationale=rationale[:400],
    )
