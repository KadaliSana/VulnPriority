"""Component B: the attacker model and its evidence vector (DESIGN.md 3.6, Gap 2).

The formula is checked against values computed by hand rather than against the
implementation's own output, because "the code agrees with itself" is exactly the
circularity the framework exists to avoid.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from vulnprio.attacker.likelihood import (
    EVIDENCE_TERMS,
    build_evidence,
    epss_logit_term,
    function_ordinal,
    logit,
)
from vulnprio.attacker.model import (
    explain_terms,
    horizon_factor,
    log_odds_terms,
    p_exploit,
    sigmoid,
)
from vulnprio.attacker.presets import list_presets, load_all_presets, load_preset
from vulnprio.core.enums import (
    ApplicabilityVerdict,
    AttackComplexity,
    EndpointFunction,
    ExploitMaturity,
    PrivilegeLevel,
    UserInteraction,
)
from vulnprio.core.errors import ConfigError
from vulnprio.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    AttackerModel,
    ExploitabilityAssessment,
)

# --------------------------------------------------------------------------
# A test attacker with round weights, so hand arithmetic is actually readable.
# --------------------------------------------------------------------------

MATURITY_WEIGHTS = {
    ExploitMaturity.UNKNOWN: 0.0,
    ExploitMaturity.UNPROVEN: 0.0,
    ExploitMaturity.POC: 0.8,
    ExploitMaturity.FUNCTIONAL: 1.6,
    ExploitMaturity.WEAPONIZED: 2.4,
}


@pytest.fixture
def plain_attacker() -> AttackerModel:
    """Round weights, 90-day horizon (factor exactly 1.0), no caps that bind."""
    return AttackerModel(
        name="plain",
        skill=0.5,
        resources=0.5,
        entry_privilege=PrivilegeLevel.NONE,
        horizon_days=90,
        target_preference={},
        w_intercept=-1.0,
        w_epss_logit=1.0,
        w_kev=2.0,
        w_kev_ransomware=0.5,
        w_exploit_maturity=MATURITY_WEIGHTS,
        w_feasibility=1.0,
        w_applicability=2.0,
        w_exposure=1.0,
        w_asset_criticality=1.0,
        w_complexity_high=-1.0,
        w_user_interaction=-0.5,
        w_privileges_required=-1.0,
        w_skill=1.0,
        w_resources=1.0,
        min_p=0.0,
        max_p=1.0,
    )


def evidence(**overrides: float) -> dict[str, float]:
    """A fully neutral evidence vector with named overrides."""
    base = {
        "epss_logit": 0.0,
        "kev": 0.0,
        "kev_ransomware": 0.0,
        "exploit_maturity": float(int(ExploitMaturity.UNKNOWN)),
        "feasibility": 0.0,
        "applicability": 0.0,
        "exposure": 0.0,
        "asset_criticality": 0.0,
        "target_function_ord": function_ordinal(EndpointFunction.UNKNOWN),
        "complexity_high": 0.0,
        "user_interaction": 0.0,
        "privileges_required": 0.0,
        "entry_privilege": 0.0,
        "skill": 0.5,
        "resources": 0.5,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Hand-computed likelihood
# --------------------------------------------------------------------------


def test_vector_one_neutral_evidence_is_hand_computable(plain_attacker: AttackerModel) -> None:
    """z = -1.0 (intercept) + 1.0*0.5 (skill) + 1.0*0.5 (resources) = 0.0 -> p = 0.5."""
    result = p_exploit(plain_attacker, evidence(), finding_id="f1")

    expected_z = -1.0 + 0.5 + 0.5
    assert math.fsum(result.log_odds_terms.values()) == pytest.approx(expected_z, abs=1e-12)
    assert result.p_exploit == pytest.approx(0.5, abs=1e-12)
    assert result.horizon_days == 90
    assert result.attacker == "plain"
    assert result.finding_id == "f1"


def test_vector_two_full_public_evidence_is_hand_computable(plain_attacker: AttackerModel) -> None:
    """KEV + full feasibility + full exposure + certain applicability."""
    result = p_exploit(
        plain_attacker,
        evidence(kev=1.0, feasibility=1.0, exposure=1.0, applicability=1.0),
    )

    #  intercept  kev  feasibility  applicability  exposure  skill  resources
    expected_z = -1.0 + 2.0 + 1.0 + 2.0 * 1.0 + 1.0 + 0.5 + 0.5
    assert expected_z == 6.0
    assert math.fsum(result.log_odds_terms.values()) == pytest.approx(expected_z, abs=1e-12)
    assert result.p_exploit == pytest.approx(1.0 / (1.0 + math.exp(-6.0)), abs=1e-12)


def test_vector_three_mixed_evidence_is_hand_computable(plain_attacker: AttackerModel) -> None:
    """A realistic mixture, including the negative terms and the EPSS logit."""
    result = p_exploit(
        plain_attacker,
        evidence(
            epss_logit=epss_logit_term(0.42),
            exploit_maturity=float(int(ExploitMaturity.FUNCTIONAL)),
            feasibility=0.6,
            applicability=0.4,           # p_applicable = 0.7
            exposure=0.9,
            asset_criticality=0.8,
            target_function_ord=function_ordinal(EndpointFunction.PAYMENT),
            complexity_high=1.0,
            user_interaction=1.0,
            privileges_required=float(int(PrivilegeLevel.ADMIN)),
        ),
    )

    epss_term = math.log(0.42 / 0.58) / 10.0
    expected_z = (
        -1.0            # intercept
        + epss_term     # 1.0 * logit(0.42)/10
        + 1.6           # w_exploit_maturity[FUNCTIONAL]
        + 0.6           # 1.0 * feasibility
        + 0.8           # 2.0 * applicability
        + 0.9           # 1.0 * exposure
        + 0.8           # 1.0 * criticality * preference 1.0
        - 1.0           # complexity high
        - 0.5           # user interaction required
        - 2.0           # -1.0 * max(0, ADMIN(2) - NONE(0))
        + 0.5           # skill
        + 0.5           # resources
    )
    assert expected_z == pytest.approx(1.2 + epss_term, abs=1e-12)

    terms = result.log_odds_terms
    assert terms["epss_logit"] == pytest.approx(epss_term, abs=1e-12)
    assert terms["exploit_maturity"] == pytest.approx(1.6, abs=1e-12)
    assert terms["privileges_required"] == pytest.approx(-2.0, abs=1e-12)
    assert math.fsum(terms.values()) == pytest.approx(expected_z, abs=1e-12)
    assert result.p_exploit == pytest.approx(1.0 / (1.0 + math.exp(-expected_z)), abs=1e-12)


def test_log_odds_terms_are_complete_and_sum_to_z(plain_attacker: AttackerModel) -> None:
    """Every named term is present: the audit trail is the point of the model."""
    terms = log_odds_terms(plain_attacker, evidence(kev=1.0))
    expected_names = {
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
    }
    assert set(terms) == expected_names

    result = p_exploit(plain_attacker, evidence(kev=1.0))
    z = math.fsum(terms.values())
    assert result.p_exploit == pytest.approx(sigmoid(z), abs=1e-12)


# --------------------------------------------------------------------------
# Caps, horizon
# --------------------------------------------------------------------------


def test_probability_is_clipped_to_the_attacker_caps() -> None:
    """min_p / max_p bind, and the uncapped value records what was clipped away."""
    attacker = AttackerModel(
        name="capped",
        horizon_days=90,
        w_exploit_maturity=MATURITY_WEIGHTS,
        w_intercept=8.0,
        w_skill=0.0,
        w_resources=0.0,
        min_p=0.1,
        max_p=0.9,
    )
    high = p_exploit(attacker, evidence())
    assert high.p_exploit == pytest.approx(0.9, abs=1e-12)
    assert high.p_exploit_uncapped == pytest.approx(sigmoid(8.0), abs=1e-9)

    floor_attacker = attacker.model_copy(update={"w_intercept": -12.0})
    low = p_exploit(floor_attacker, evidence())
    assert low.p_exploit == pytest.approx(0.1, abs=1e-12)
    assert low.p_exploit_uncapped < 0.1


def test_horizon_factor_is_one_at_ninety_days_and_monotone() -> None:
    """The reference horizon leaves the tuned weights meaning what they say."""
    assert horizon_factor(90) == pytest.approx(1.0, abs=1e-12)
    assert horizon_factor(30) < horizon_factor(90) < horizon_factor(365)


def test_shorter_horizon_never_raises_probability(plain_attacker: AttackerModel) -> None:
    """A mass scanner with 30 days cannot be more likely than the same model with 90."""
    vector = evidence(kev=1.0, feasibility=0.5)
    short = p_exploit(plain_attacker, vector, horizon_days=30)
    long = p_exploit(plain_attacker, vector, horizon_days=90)
    assert short.p_exploit <= long.p_exploit
    assert short.horizon_days == 30


# --------------------------------------------------------------------------
# Monotonicity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("epss", [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
def test_raising_epss_never_lowers_probability(plain_attacker: AttackerModel, epss: float) -> None:
    """EPSS is monotone in P(exploit) by construction, at every level."""
    lower = p_exploit(plain_attacker, evidence(epss_logit=epss_logit_term(max(epss - 0.05, 0.0))))
    higher = p_exploit(plain_attacker, evidence(epss_logit=epss_logit_term(epss)))
    assert higher.p_exploit >= lower.p_exploit - 1e-12


def test_adding_kev_never_lowers_probability() -> None:
    """True for every shipped preset, not just the test attacker."""
    for attacker in load_all_presets().values():
        without = p_exploit(attacker, evidence())
        with_kev = p_exploit(attacker, evidence(kev=1.0))
        assert with_kev.p_exploit >= without.p_exploit - 1e-12


def test_raising_feasibility_never_lowers_probability() -> None:
    for attacker in load_all_presets().values():
        previous = -1.0
        for feasibility in (0.0, 0.25, 0.5, 0.75, 1.0):
            current = p_exploit(attacker, evidence(feasibility=feasibility)).p_exploit
            assert current >= previous - 1e-12
            previous = current


def test_raising_applicability_and_maturity_never_lowers_probability(
    plain_attacker: AttackerModel,
) -> None:
    previous = -1.0
    for maturity in ExploitMaturity:
        current = p_exploit(plain_attacker, evidence(exploit_maturity=float(int(maturity)))).p_exploit
        assert current >= previous - 1e-12
        previous = current

    low = p_exploit(plain_attacker, evidence(applicability=-1.0)).p_exploit
    mid = p_exploit(plain_attacker, evidence(applicability=0.0)).p_exploit
    high = p_exploit(plain_attacker, evidence(applicability=1.0)).p_exploit
    assert low <= mid <= high


# --------------------------------------------------------------------------
# Presets differ in the documented directions
# --------------------------------------------------------------------------


def test_presets_are_all_loadable_and_listed() -> None:
    names = list_presets()
    assert set(names) >= {
        "apt",
        "content_controlling",
        "insider",
        "opportunistic",
        "targeted_criminal",
    }
    for name in names:
        assert load_preset(name).name == name


def test_unknown_preset_names_the_available_ones() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_preset("does_not_exist")
    assert "opportunistic" in str(excinfo.value)


def test_insider_is_least_moved_by_internet_exposure() -> None:
    """The insider is already inside; exposure is nearly irrelevant to them."""
    presets = load_all_presets()

    def exposure_swing(name: str) -> float:
        attacker = presets[name]
        high = log_odds_terms(attacker, evidence(exposure=1.0))["exposure"]
        low = log_odds_terms(attacker, evidence(exposure=0.0))["exposure"]
        return high - low

    insider_swing = exposure_swing("insider")
    assert insider_swing == pytest.approx(presets["insider"].w_exposure, abs=1e-12)
    for name in presets:
        if name != "insider":
            assert insider_swing < exposure_swing(name)


def test_apt_is_least_sensitive_to_public_exploit_signals() -> None:
    """A well-resourced adversary does not need somebody else's exploit code."""
    presets = load_all_presets()
    public = evidence(
        kev=1.0,
        kev_ransomware=1.0,
        epss_logit=epss_logit_term(0.95),
        exploit_maturity=float(int(ExploitMaturity.WEAPONIZED)),
    )

    def public_signal_swing(attacker: AttackerModel) -> float:
        with_signals = log_odds_terms(attacker, public)
        without = log_odds_terms(attacker, evidence())
        keys = ("kev", "kev_ransomware", "epss_logit", "exploit_maturity")
        return sum(with_signals[key] - without[key] for key in keys)

    swings = {name: public_signal_swing(attacker) for name, attacker in presets.items()}
    assert min(swings, key=lambda name: swings[name]) == "apt"
    assert swings["apt"] < swings["opportunistic"]


def test_opportunistic_penalises_required_privileges_hardest() -> None:
    """Mass scanners exploit what needs no credentials; everything else is someone else's job."""
    presets = load_all_presets()
    vector = evidence(privileges_required=float(int(PrivilegeLevel.ADMIN)))
    penalties = {
        name: log_odds_terms(attacker, vector)["privileges_required"]
        for name, attacker in presets.items()
    }
    assert min(penalties, key=lambda name: penalties[name]) == "opportunistic"
    assert penalties["opportunistic"] < penalties["insider"]


def test_insider_entry_privilege_cancels_the_user_privilege_penalty() -> None:
    """Needing USER is free for someone who already has USER."""
    insider = load_preset("insider")
    vector = evidence(privileges_required=float(int(PrivilegeLevel.USER)))
    assert log_odds_terms(insider, vector)["privileges_required"] == pytest.approx(0.0, abs=1e-12)


def test_target_preference_scales_asset_criticality() -> None:
    """The targeted criminal weights payment endpoints above admin ones.

    ``target_preference`` is a multiplier with a default of 1.0 (DESIGN.md 3.6), so a
    function the preset does not mention keeps full weight and the listed ones express
    relative preference among themselves.
    """
    criminal = load_preset("targeted_criminal")

    def term(function: EndpointFunction) -> float:
        return log_odds_terms(
            criminal,
            evidence(asset_criticality=1.0, target_function_ord=function_ordinal(function)),
        )["asset_criticality"]

    assert term(EndpointFunction.PAYMENT) > term(EndpointFunction.ADMIN)
    assert term(EndpointFunction.PAYMENT) == pytest.approx(criminal.w_asset_criticality * 1.0, abs=1e-12)
    assert term(EndpointFunction.ADMIN) == pytest.approx(criminal.w_asset_criticality * 0.7, abs=1e-12)


# --------------------------------------------------------------------------
# Evidence assembly
# --------------------------------------------------------------------------


def _assessments(finding_id: str) -> tuple[
    AssetCriticality, ExploitabilityAssessment, ApplicabilityAssessment
]:
    asset = AssetCriticality(
        endpoint_id="ep_login",
        function=EndpointFunction.AUTH,
        criticality=0.85,
        data_sensitivity=0.7,
        exposure=1.0,
        is_auth_boundary=True,
    )
    exploitability = ExploitabilityAssessment(
        finding_id=finding_id,
        exploit_feasibility=0.75,
        exploit_maturity=ExploitMaturity.POC,
        attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegeLevel.NONE,
        user_interaction=UserInteraction.NONE,
        impact_c=0.9,
        impact_i=0.6,
        impact_a=0.3,
        privilege_gained=PrivilegeLevel.USER,
    )
    applicability = ApplicabilityAssessment(
        finding_id=finding_id,
        verdict=ApplicabilityVerdict.APPLICABLE,
        p_applicable=0.9,
    )
    return asset, exploitability, applicability


def test_build_evidence_reads_feeds_and_assessments(sample_scan, sample_intel, sample_endpoints, as_of) -> None:
    """EPSS, KEV and the feed's exploit maturity all land in the named terms."""
    finding = sample_scan.findings[0]
    asset, exploitability, applicability = _assessments(finding.finding_id)

    vector = build_evidence(
        finding, (sample_intel,), asset, exploitability, applicability, sample_endpoints[0], as_of
    )

    assert set(vector) == set(EVIDENCE_TERMS)
    assert vector["kev"] == 1.0
    assert vector["kev_ransomware"] == 0.0
    assert vector["epss_logit"] == pytest.approx(logit(0.42) / 10.0, abs=1e-12)
    # the feed says FUNCTIONAL, Component A only said POC: the stronger evidence wins
    assert vector["exploit_maturity"] == float(int(ExploitMaturity.FUNCTIONAL))
    assert vector["feasibility"] == pytest.approx(0.75)
    assert vector["applicability"] == pytest.approx(0.8)
    assert vector["exposure"] == pytest.approx(1.0)
    assert vector["asset_criticality"] == pytest.approx(0.85)
    assert vector["complexity_high"] == 0.0
    assert vector["user_interaction"] == 0.0
    assert vector["privileges_required"] == 0.0


def test_build_evidence_without_cve_uses_the_neutral_epss(sample_scan, sample_endpoints, as_of) -> None:
    """An unknown EPSS contributes nothing rather than pretending nobody exploits this."""
    finding = sample_scan.findings[1]           # the XSS finding, no CVE
    asset, exploitability, applicability = _assessments(finding.finding_id)

    vector = build_evidence(finding, (), asset, exploitability, applicability, sample_endpoints[1], as_of)
    assert vector["epss_logit"] == pytest.approx(0.0, abs=1e-12)
    assert vector["kev"] == 0.0


def test_build_evidence_refuses_intel_dated_after_as_of(sample_scan, sample_intel, sample_endpoints) -> None:
    """As-of discipline: a later KEV listing must not colour an earlier scan."""
    finding = sample_scan.findings[0]
    asset, exploitability, applicability = _assessments(finding.finding_id)

    vector = build_evidence(
        finding,
        (sample_intel,),
        asset,
        exploitability,
        applicability,
        sample_endpoints[0],
        date(2024, 1, 15),      # before the KEV date_added of 2024-02-01
    )
    assert vector["kev"] == 0.0
    assert vector["epss_logit"] == pytest.approx(0.0, abs=1e-12)


def test_build_evidence_with_attacker_records_its_constants(sample_scan, sample_intel, sample_endpoints, as_of) -> None:
    finding = sample_scan.findings[0]
    asset, exploitability, applicability = _assessments(finding.finding_id)
    insider = load_preset("insider")

    vector = build_evidence(
        finding, (sample_intel,), asset, exploitability, applicability, sample_endpoints[0], as_of, insider
    )
    assert vector["skill"] == pytest.approx(insider.skill)
    assert vector["resources"] == pytest.approx(insider.resources)
    assert vector["entry_privilege"] == float(int(insider.entry_privilege))


# --------------------------------------------------------------------------
# Reason codes
# --------------------------------------------------------------------------


def test_explain_terms_orders_by_absolute_contribution(plain_attacker: AttackerModel) -> None:
    """Reason codes are templated operator text, strongest contribution first."""
    result = p_exploit(
        plain_attacker,
        evidence(kev=1.0, feasibility=0.2, privileges_required=float(int(PrivilegeLevel.SYSTEM))),
        finding_id="f_sqli",
    )
    codes = explain_terms(result)

    assert "privileges required" in codes[0]      # -3.0 is the largest magnitude
    assert "CISA KEV" in codes[1]                 # +2.0
    assert all(isinstance(code, str) for code in codes)
    assert codes[-1].startswith("attacker plain over 90 days")
    # the intercept is a baseline, not evidence, so it is never a reason code
    assert not any("base rate" in code for code in codes)


def test_explain_terms_is_deterministic(plain_attacker: AttackerModel) -> None:
    result = p_exploit(plain_attacker, evidence(kev=1.0, exposure=1.0))
    assert explain_terms(result) == explain_terms(result)
