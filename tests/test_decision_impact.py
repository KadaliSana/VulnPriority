"""Component B: monetary impact and unequal remediation cost (DESIGN.md 3.6, Gaps 9, 10)."""

from __future__ import annotations

from datetime import datetime

import pytest

from vulnpriority.core.config import ComponentBConfig
from vulnpriority.core.enums import (
    AttackComplexity,
    EndpointFunction,
    ExploitMaturity,
    HttpMethod,
    PrivilegeLevel,
    Provenance,
    ScannerSeverity,
    UserInteraction,
)
from vulnpriority.core.models import (
    AssetCriticality,
    Endpoint,
    ExploitabilityAssessment,
    Finding,
    ImpactModel,
    UntrustedText,
)
from vulnpriority.decision.impact import estimate_impact, impact_components, lookup
from vulnpriority.decision.remediation_cost import (
    ARCHITECTURAL_CHANGE_HOURS,
    CODE_CHANGE_HOURS,
    CONFIGURATION_FIX_HOURS,
    DEFAULT_CWE_HOURS,
    DEPENDENCY_UPGRADE_HOURS,
    cwe_class,
    estimate_cost,
)

# --------------------------------------------------------------------------
# Fixtures with round numbers, so the arithmetic can be checked by hand.
# --------------------------------------------------------------------------


@pytest.fixture
def endpoint() -> Endpoint:
    return Endpoint(
        endpoint_id="ep_pay",
        app_id="app1",
        host="shop.example.com",
        url="https://shop.example.com/api/pay",
        path="/api/pay",
        method=HttpMethod.POST,
        auth_required=PrivilegeLevel.USER,
    )


def make_finding(cwe_id: int | None = 89, cluster_size: int = 1) -> Finding:
    return Finding(
        finding_id=f"f_{cwe_id}_{cluster_size}",
        scan_id="scan_1",
        app_id="app1",
        endpoint_id="ep_pay",
        name="test finding",
        cwe_id=cwe_id,
        scanner="zap",
        scanner_severity=ScannerSeverity.HIGH,
        description=UntrustedText(text="finding", provenance=Provenance.SCANNER_OUTPUT),
        observed_at=datetime(2024, 5, 1, 9, 0, 0),
        cluster_size=cluster_size,
    )


@pytest.fixture
def asset() -> AssetCriticality:
    return AssetCriticality(
        endpoint_id="ep_pay",
        function=EndpointFunction.PAYMENT,
        criticality=0.9,
        data_sensitivity=0.5,
        exposure=1.0,
    )


@pytest.fixture
def exploitability() -> ExploitabilityAssessment:
    return ExploitabilityAssessment(
        finding_id="f_89_1",
        exploit_feasibility=0.8,
        exploit_maturity=ExploitMaturity.FUNCTIONAL,
        attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegeLevel.NONE,
        user_interaction=UserInteraction.NONE,
        impact_c=0.5,
        impact_i=0.4,
        impact_a=0.25,
        privilege_gained=PrivilegeLevel.ADMIN,
    )


@pytest.fixture
def impact_model() -> ImpactModel:
    """Round numbers everywhere; no cap and no multiplier unless a test asks for one."""
    return ImpactModel(
        name="round",
        cost_per_record=100.0,
        records_by_function={EndpointFunction.PAYMENT: 1000},
        downtime_cost_per_hour=1000.0,
        downtime_hours_by_privilege={PrivilegeLevel.ADMIN: 10.0},
        integrity_loss_by_function={EndpointFunction.PAYMENT: 20000.0},
        regulatory_multiplier=1.0,
        reputational_fraction=0.0,
        max_impact=5_000_000.0,
    )


# --------------------------------------------------------------------------
# Impact arithmetic
# --------------------------------------------------------------------------


def test_impact_components_are_hand_computable(asset, exploitability, impact_model) -> None:
    """confidentiality = 1000 * 100 * 0.5 * 0.5, integrity = 20000 * 0.4, availability = 10 * 1000 * 0.25."""
    confidentiality, integrity, availability = impact_components(asset, exploitability, impact_model)
    assert confidentiality == pytest.approx(1000 * 100.0 * 0.5 * 0.5)   # 25_000
    assert integrity == pytest.approx(20000.0 * 0.4)                    # 8_000
    assert availability == pytest.approx(10.0 * 1000.0 * 0.25)          # 2_500


def test_total_without_multiplier_or_reputation(endpoint, asset, exploitability, impact_model) -> None:
    impact = estimate_impact(make_finding(), endpoint, asset, exploitability, impact_model)
    assert impact.total == pytest.approx(25_000.0 + 8_000.0 + 2_500.0)
    assert impact.reputational == pytest.approx(0.0)
    assert impact.finding_id == "f_89_1"


def test_regulatory_multiplier_and_reputational_fraction(
    endpoint, asset, exploitability, impact_model
) -> None:
    """subtotal = 35_500 * 2.0 = 71_000; reputational = 71_000 * 0.3 = 21_300."""
    model = impact_model.model_copy(
        update={"regulatory_multiplier": 2.0, "reputational_fraction": 0.3}
    )
    impact = estimate_impact(make_finding(), endpoint, asset, exploitability, model)

    assert impact.confidentiality == pytest.approx(50_000.0)
    assert impact.integrity == pytest.approx(16_000.0)
    assert impact.availability == pytest.approx(5_000.0)
    assert impact.reputational == pytest.approx(21_300.0)
    assert impact.total == pytest.approx(92_300.0)


def test_business_impact_record_is_additive(endpoint, asset, exploitability, impact_model) -> None:
    """The parts sum to the total whenever the cap does not bind: an auditable record."""
    model = impact_model.model_copy(
        update={"regulatory_multiplier": 1.7, "reputational_fraction": 0.25}
    )
    impact = estimate_impact(make_finding(), endpoint, asset, exploitability, model)
    parts = (
        impact.confidentiality
        + impact.integrity
        + impact.availability
        + impact.reputational
    )
    assert parts == pytest.approx(impact.total)


def test_max_impact_cap_binds(endpoint, asset, exploitability, impact_model) -> None:
    model = impact_model.model_copy(
        update={"regulatory_multiplier": 2.0, "reputational_fraction": 0.3, "max_impact": 50_000.0}
    )
    impact = estimate_impact(make_finding(), endpoint, asset, exploitability, model)
    assert impact.total == pytest.approx(50_000.0)
    assert "capped" in impact.rationale


def test_asset_override_replaces_the_computed_total(endpoint, asset, exploitability, impact_model) -> None:
    model = impact_model.model_copy(update={"asset_overrides": {"ep_pay": 1_250_000.0}})
    impact = estimate_impact(make_finding(), endpoint, asset, exploitability, model)
    assert impact.total == pytest.approx(1_250_000.0)
    assert "operator override" in impact.rationale


def test_asset_override_for_another_endpoint_is_ignored(
    endpoint, asset, exploitability, impact_model
) -> None:
    model = impact_model.model_copy(update={"asset_overrides": {"ep_other": 1_250_000.0}})
    impact = estimate_impact(make_finding(), endpoint, asset, exploitability, model)
    assert impact.total == pytest.approx(35_500.0)


def test_asset_override_still_respects_the_cap(endpoint, asset, exploitability, impact_model) -> None:
    """``total <= max_impact`` is an invariant downstream normalisation relies on."""
    model = impact_model.model_copy(
        update={"asset_overrides": {"ep_pay": 9_000_000.0}, "max_impact": 1_000_000.0}
    )
    impact = estimate_impact(make_finding(), endpoint, asset, exploitability, model)
    assert impact.total == pytest.approx(1_000_000.0)


def test_unmapped_function_and_privilege_yield_zero(endpoint, exploitability, impact_model) -> None:
    """A static-content endpoint the model says nothing about costs nothing, not a default."""
    static_asset = AssetCriticality(
        endpoint_id="ep_pay",
        function=EndpointFunction.STATIC_CONTENT,
        criticality=0.1,
        data_sensitivity=0.0,
        exposure=1.0,
    )
    no_privilege = exploitability.model_copy(update={"privilege_gained": PrivilegeLevel.NONE})
    impact = estimate_impact(make_finding(), endpoint, static_asset, no_privilege, impact_model)
    assert impact.total == pytest.approx(0.0)


def test_lookup_tolerates_string_and_enum_keys() -> None:
    """Impact tables arrive from YAML; a key shape mismatch must not silently zero money."""
    assert lookup({EndpointFunction.PAYMENT: 5.0}, EndpointFunction.PAYMENT, 0.0) == 5.0
    assert lookup({"payment": 5.0}, EndpointFunction.PAYMENT, 0.0) == 5.0
    assert lookup({2: 8.0}, PrivilegeLevel.ADMIN, 0.0) == 8.0
    assert lookup({}, EndpointFunction.SEARCH, 3.0) == 3.0


def test_shipped_impact_presets_load_and_price_a_finding(endpoint, asset, exploitability) -> None:
    """The healthcare preset must cost more than the ecommerce one for the same finding."""
    from vulnpriority.core.config import load_impact_preset

    pii_asset = asset.model_copy(update={"function": EndpointFunction.PII_DATA, "data_sensitivity": 1.0})
    ecommerce = estimate_impact(
        make_finding(), endpoint, pii_asset, exploitability, load_impact_preset("default_ecommerce")
    )
    healthcare = estimate_impact(
        make_finding(), endpoint, pii_asset, exploitability, load_impact_preset("healthcare")
    )
    assert healthcare.total > ecommerce.total > 0.0


# --------------------------------------------------------------------------
# Remediation cost is genuinely unequal
# --------------------------------------------------------------------------


def test_cwe_classes_are_ordered_and_distinct() -> None:
    """Four classes of change, four different prices. This is the point of Gap 10."""
    assert (
        CONFIGURATION_FIX_HOURS
        < DEPENDENCY_UPGRADE_HOURS
        < CODE_CHANGE_HOURS
        < ARCHITECTURAL_CHANGE_HOURS
    )
    assert cwe_class(1021) == "configuration_fix"
    assert cwe_class(1104) == "dependency_upgrade"
    assert cwe_class(89) == "code_change"
    assert cwe_class(862) == "architectural_change"
    assert cwe_class(999999) == "unclassified"
    assert cwe_class(None) == "unclassified"


def test_cost_differs_across_cwe_classes() -> None:
    config = ComponentBConfig()
    hours = {
        cwe: estimate_cost(make_finding(cwe_id=cwe), config).hours
        for cwe in (1021, 1104, 89, 862)
    }
    assert len(set(hours.values())) == 4
    assert hours[1021] < hours[1104] < hours[89] < hours[862]


def test_cost_grows_with_cluster_size() -> None:
    """A root cause on twenty endpoints costs more to roll out than one on one."""
    config = ComponentBConfig()
    one = estimate_cost(make_finding(cwe_id=79, cluster_size=1), config)
    ten = estimate_cost(make_finding(cwe_id=79, cluster_size=10), config)

    assert ten.hours == pytest.approx(
        one.hours + config.remediation_hours_per_extra_endpoint * 9
    )
    assert ten.cost > one.cost
    assert ten.cost == pytest.approx(ten.hours * config.remediation_hourly_rate)


def test_unknown_cwe_falls_back_to_the_configured_default() -> None:
    config = ComponentBConfig()
    cost = estimate_cost(make_finding(cwe_id=999999), config)
    assert cost.hours == pytest.approx(config.remediation_default_hours)
    assert "unclassified" in cost.basis


def test_operator_override_beats_the_default_table() -> None:
    config = ComponentBConfig(remediation_hours_by_cwe={89: 40.0})
    assert estimate_cost(make_finding(cwe_id=89), config).hours == pytest.approx(40.0)
    assert DEFAULT_CWE_HOURS[89] != 40.0


def test_hours_are_always_positive() -> None:
    """``RemediationCost.hours`` is ``gt=0``; a zero-hour fix would break the knapsack."""
    config = ComponentBConfig(
        remediation_default_hours=0.01,
        remediation_hours_per_extra_endpoint=0.0,
    )
    cost = estimate_cost(make_finding(cwe_id=None), config)
    assert cost.hours > 0.0
