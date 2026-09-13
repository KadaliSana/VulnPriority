"""The seven baselines (DESIGN.md 3.8, section 4).

Each baseline is a closed-form function of frame columns, so each one is tested by
constructing findings with exactly known evidence and asserting the resulting order. The
VMC chain baseline gets a hand-worked example with every band and tie-break written out,
because it is the published ordering the framework reports efficiency and coverage
against and an arithmetic slip there would quietly flatter every other number.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

from vulnprio.core.enums import (
    ApplicabilityVerdict,
    CvssVersion,
    EndpointFunction,
    HttpMethod,
    PrivilegeLevel,
    Provenance,
    RankerName,
    ScannerSeverity,
    ScoreSource,
)
from vulnprio.core.interfaces import Ranker
from vulnprio.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    BusinessImpact,
    ComponentFlags,
    CvssRecord,
    Endpoint,
    EnrichedFinding,
    EpssRecord,
    ExploitLikelihood,
    ExploitabilityAssessment,
    FeatureFrame,
    Finding,
    KevRecord,
    RemediationCost,
    UntrustedText,
    VulnIntel,
)
from vulnprio.core.registry import RANKERS, get_ranker
from vulnprio.rank.baselines import (
    VMC_CVSS_THRESHOLD,
    VMC_EPSS_THRESHOLD,
    CvssOnlyRanker,
    EpssOnlyRanker,
    ExpectedLossRanker,
    KevFirstRanker,
    RandomRanker,
    ScannerSeverityRanker,
    VmcChainRanker,
)
from vulnprio.rank.features import NEUTRAL, FeatureBuilder

AS_OF = date(2024, 6, 1)
OBSERVED_AT = datetime(2024, 5, 1, 9, 0, 0)

BASELINE_CLASSES = (
    CvssOnlyRanker,
    EpssOnlyRanker,
    KevFirstRanker,
    ScannerSeverityRanker,
    ExpectedLossRanker,
    VmcChainRanker,
    RandomRanker,
)


# ---------------------------------------------------------------------------
# A frame with exactly the evidence each test needs
# ---------------------------------------------------------------------------


def make_item(
    finding_id: str,
    *,
    kev: bool = False,
    epss: float = 0.0,
    cvss: float = 0.0,
    severity: ScannerSeverity = ScannerSeverity.MEDIUM,
    confidence: float = 0.5,
    p_exploit: float = 0.1,
    impact: float = 10_000.0,
    scan_id: str = "scan_1",
) -> EnrichedFinding:
    """One enriched finding whose frame row carries exactly the evidence named here."""
    cve_id = f"CVE-2024-{1000 + abs(hash(finding_id)) % 8000}"
    endpoint = Endpoint(
        endpoint_id=f"ep_{finding_id}",
        app_id="app1",
        host="shop.example.com",
        url=f"https://shop.example.com/{finding_id}",
        path=f"/{finding_id}",
        method=HttpMethod.GET,
        auth_required=PrivilegeLevel.NONE,
    )
    finding = Finding(
        finding_id=finding_id,
        scan_id=scan_id,
        app_id="app1",
        endpoint_id=endpoint.endpoint_id,
        name="Finding",
        cwe_id=79,
        cve_ids=(cve_id,),
        scanner="zap",
        scanner_severity=severity,
        scanner_confidence=confidence,
        description=UntrustedText(text="finding", provenance=Provenance.SCANNER_OUTPUT),
        observed_at=OBSERVED_AT,
    )
    intel = VulnIntel(
        cve_id=cve_id,
        as_of=AS_OF,
        published=date(2024, 1, 10),
        cvss=(
            CvssRecord(version=CvssVersion.V31, source=ScoreSource.NVD, base_score=cvss),
        ),
        epss=EpssRecord(cve_id=cve_id, score=epss, percentile=epss, as_of=AS_OF),
        kev=KevRecord(
            cve_id=cve_id,
            in_kev=kev,
            date_added=date(2024, 2, 1) if kev else None,
            as_of=AS_OF,
        ),
    )
    return EnrichedFinding(
        finding=finding,
        endpoint=endpoint,
        intel=(intel,),
        asset=AssetCriticality(
            endpoint_id=endpoint.endpoint_id,
            function=EndpointFunction.API_DATA,
            criticality=0.5,
            data_sensitivity=0.5,
            exposure=1.0,
        ),
        exploitability=ExploitabilityAssessment(finding_id=finding_id, exploit_feasibility=0.5),
        applicability=ApplicabilityAssessment(
            finding_id=finding_id, verdict=ApplicabilityVerdict.APPLICABLE, p_applicable=0.8
        ),
        likelihood=ExploitLikelihood(
            finding_id=finding_id,
            attacker="opportunistic",
            p_exploit=p_exploit,
            p_exploit_uncapped=p_exploit,
            horizon_days=90,
        ),
        impact=BusinessImpact(finding_id=finding_id, total=impact),
        remediation=RemediationCost(finding_id=finding_id, hours=4.0, cost=480.0),
        expected_loss=p_exploit * impact,
        as_of=AS_OF,
    )


def frame_of(
    items: list[EnrichedFinding], flags: ComponentFlags = ComponentFlags()
) -> FeatureFrame:
    return FeatureBuilder().build(items, {}, flags)


def order(frame: FeatureFrame, scores: np.ndarray) -> list[str]:
    """Finding ids in descending score, ties broken on id, as ``rank_scan`` does."""
    rows = list(zip((float(value) for value in scores), frame.finding_ids))
    return [finding_id for _, finding_id in sorted(rows, key=lambda row: (-row[0], row[1]))]


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", BASELINE_CLASSES, ids=lambda cls: cls.__name__)
def test_every_baseline_is_a_ranker_that_needs_no_fitting(cls) -> None:
    baseline = cls()
    assert isinstance(baseline, Ranker)
    assert baseline.requires_fit() is False


@pytest.mark.parametrize("cls", BASELINE_CLASSES, ids=lambda cls: cls.__name__)
def test_every_baseline_is_registered_under_its_own_name(cls) -> None:
    assert get_ranker(cls.name) is cls
    assert RANKERS[cls.name] is cls


def test_every_baseline_named_in_the_evaluation_protocol_exists() -> None:
    expected = {
        RankerName.CVSS_ONLY,
        RankerName.EPSS_ONLY,
        RankerName.KEV_FIRST,
        RankerName.SCANNER_SEVERITY,
        RankerName.EXPECTED_LOSS,
        RankerName.VMC_CHAIN,
        RankerName.RANDOM,
    }
    assert expected <= set(RANKERS)


@pytest.mark.parametrize("cls", BASELINE_CLASSES, ids=lambda cls: cls.__name__)
def test_fitting_a_baseline_is_a_no_op_that_returns_itself(cls) -> None:
    frame = frame_of([make_item("f_1")])
    baseline = cls()
    assert baseline.fit(frame, np.zeros(1)) is baseline


@pytest.mark.parametrize("cls", BASELINE_CLASSES, ids=lambda cls: cls.__name__)
def test_every_baseline_returns_one_finite_score_per_row(cls) -> None:
    frame = frame_of([make_item("f_1"), make_item("f_2"), make_item("f_3")])
    scores = cls().score(frame)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("cls", BASELINE_CLASSES, ids=lambda cls: cls.__name__)
def test_a_baseline_round_trips_through_save_and_load(cls, tmp_path) -> None:
    frame = frame_of([make_item("f_1"), make_item("f_2")])
    path = tmp_path / f"{cls.name.value}.json"
    baseline = cls()
    baseline.save(path)
    restored = cls.load(path)

    assert restored.params() == baseline.params()
    assert np.allclose(restored.score(frame), baseline.score(frame))


def test_a_parameterised_baseline_keeps_its_parameters_across_a_round_trip(tmp_path) -> None:
    """A reloaded random control with a different seed would silently be a different control."""
    frame = frame_of([make_item(f"f_{index}") for index in range(6)])
    for baseline in (RandomRanker(seed=99), VmcChainRanker(epss_threshold=0.5, cvss_threshold=8.0)):
        path = tmp_path / f"{baseline.name.value}_params.json"
        baseline.save(path)
        restored = type(baseline).load(path)
        assert restored.params() == baseline.params()
        assert np.allclose(restored.score(frame), baseline.score(frame))


# ---------------------------------------------------------------------------
# Individual orderings
# ---------------------------------------------------------------------------


def test_cvss_only_orders_by_base_score() -> None:
    frame = frame_of(
        [
            make_item("f_low", cvss=2.0, kev=True, epss=0.9),
            make_item("f_mid", cvss=6.5),
            make_item("f_high", cvss=9.8),
        ]
    )
    assert order(frame, CvssOnlyRanker().score(frame)) == ["f_high", "f_mid", "f_low"]


def test_epss_only_orders_by_probability_and_ignores_everything_else() -> None:
    frame = frame_of(
        [
            make_item("f_a", epss=0.01, cvss=10.0),
            make_item("f_b", epss=0.30, cvss=1.0),
            make_item("f_c", epss=0.75, cvss=4.0),
        ]
    )
    assert order(frame, EpssOnlyRanker().score(frame)) == ["f_c", "f_b", "f_a"]


def test_kev_first_puts_every_kev_finding_above_every_other_one() -> None:
    frame = frame_of(
        [
            make_item("f_kev_weak", kev=True, cvss=1.0),
            make_item("f_kev_strong", kev=True, cvss=8.0),
            make_item("f_plain_max", cvss=10.0),
            make_item("f_plain_mid", cvss=5.0),
        ]
    )
    assert order(frame, KevFirstRanker().score(frame)) == [
        "f_kev_strong",
        "f_kev_weak",
        "f_plain_max",
        "f_plain_mid",
    ]


def test_the_kev_tie_break_can_never_promote_a_finding_across_the_band() -> None:
    """CVSS 10.0 contributes 0.4, well short of the whole unit that separates the bands."""
    frame = frame_of(
        [make_item("f_kev", kev=True, cvss=0.0), make_item("f_plain", cvss=10.0)]
    )
    scores = KevFirstRanker().score(frame)
    assert order(frame, scores) == ["f_kev", "f_plain"]


def test_scanner_severity_orders_by_severity_then_confidence() -> None:
    frame = frame_of(
        [
            make_item("f_info", severity=ScannerSeverity.INFO, confidence=1.0),
            make_item("f_high_sure", severity=ScannerSeverity.HIGH, confidence=0.9),
            make_item("f_high_unsure", severity=ScannerSeverity.HIGH, confidence=0.1),
            make_item("f_critical", severity=ScannerSeverity.CRITICAL, confidence=0.2),
        ]
    )
    assert order(frame, ScannerSeverityRanker().score(frame)) == [
        "f_critical",
        "f_high_sure",
        "f_high_unsure",
        "f_info",
    ]


def test_expected_loss_orders_by_probability_times_impact() -> None:
    """The Gap 1 construct: a likely small loss can outrank an unlikely large one."""
    frame = frame_of(
        [
            make_item("f_likely_small", p_exploit=0.90, impact=100_000.0),   # 90,000
            make_item("f_unlikely_huge", p_exploit=0.01, impact=5_000_000.0),  # 50,000
            make_item("f_tiny", p_exploit=0.05, impact=1_000.0),               # 50
        ]
    )
    assert order(frame, ExpectedLossRanker().score(frame)) == [
        "f_likely_small",
        "f_unlikely_huge",
        "f_tiny",
    ]


# ---------------------------------------------------------------------------
# VMC chain: hand-worked
# ---------------------------------------------------------------------------


def test_vmc_chain_reproduces_a_hand_worked_example() -> None:
    """Bands are 2 / 1 / 0; the tie-break is ``0.5 * epss + 0.04 * cvss``.

    ==========  ===  =====  ====  ====  =======================  =====
    finding     KEV  EPSS   CVSS  band  tie-break                total
    ==========  ===  =====  ====  ====  =======================  =====
    f_kev       yes  0.01   4.0   2     0.005 + 0.160 = 0.165    2.165
    f_epss      no   0.50   3.0   2     0.250 + 0.120 = 0.370    2.370
    f_high      no   0.02   9.0   1     0.010 + 0.360 = 0.370    1.370
    f_mid       no   0.05   7.0   1     0.025 + 0.280 = 0.305    1.305
    f_low       no   0.01   2.0   0     0.005 + 0.080 = 0.085    0.085
    ==========  ===  =====  ====  ====  =======================  =====

    ``f_mid``'s EPSS of 0.05 sits below the 0.088 cut, so it stays in the severity band
    despite being the second most probable finding in the set.
    """
    frame = frame_of(
        [
            make_item("f_kev", kev=True, epss=0.01, cvss=4.0),
            make_item("f_epss", epss=0.50, cvss=3.0),
            make_item("f_high", epss=0.02, cvss=9.0),
            make_item("f_mid", epss=0.05, cvss=7.0),
            make_item("f_low", epss=0.01, cvss=2.0),
        ]
    )
    scores = dict(zip(frame.finding_ids, VmcChainRanker().score(frame)))

    assert scores["f_kev"] == pytest.approx(2.165)
    assert scores["f_epss"] == pytest.approx(2.370)
    assert scores["f_high"] == pytest.approx(1.370)
    assert scores["f_mid"] == pytest.approx(1.305)
    assert scores["f_low"] == pytest.approx(0.085)

    assert order(frame, VmcChainRanker().score(frame)) == [
        "f_epss",
        "f_kev",
        "f_high",
        "f_mid",
        "f_low",
    ]


def test_the_vmc_thresholds_are_the_published_ones() -> None:
    assert VMC_EPSS_THRESHOLD == pytest.approx(0.088)
    assert VMC_CVSS_THRESHOLD == pytest.approx(7.0)


def test_vmc_epss_threshold_is_inclusive_at_the_boundary() -> None:
    frame = frame_of(
        [
            make_item("f_at", epss=VMC_EPSS_THRESHOLD, cvss=1.0),
            make_item("f_just_below", epss=VMC_EPSS_THRESHOLD - 1e-4, cvss=10.0),
        ]
    )
    assert order(frame, VmcChainRanker().score(frame)) == ["f_at", "f_just_below"]


def test_vmc_cvss_threshold_is_inclusive_at_the_boundary() -> None:
    frame = frame_of(
        [
            make_item("f_at", cvss=VMC_CVSS_THRESHOLD),
            make_item("f_just_below", cvss=VMC_CVSS_THRESHOLD - 0.1),
        ]
    )
    assert order(frame, VmcChainRanker().score(frame)) == ["f_at", "f_just_below"]


def test_vmc_thresholds_can_be_overridden_for_a_sensitivity_analysis() -> None:
    frame = frame_of([make_item("f_a", epss=0.20, cvss=1.0), make_item("f_b", cvss=9.0)])
    strict = VmcChainRanker(epss_threshold=0.5, cvss_threshold=8.0)
    assert order(frame, strict.score(frame)) == ["f_b", "f_a"]


# ---------------------------------------------------------------------------
# Random control
# ---------------------------------------------------------------------------


def test_the_random_baseline_is_reproducible_under_its_seed() -> None:
    frame = frame_of([make_item(f"f_{index}") for index in range(12)])
    assert np.allclose(RandomRanker(seed=11).score(frame), RandomRanker(seed=11).score(frame))


def test_different_seeds_give_different_orderings() -> None:
    frame = frame_of([make_item(f"f_{index}") for index in range(12)])
    first = order(frame, RandomRanker(seed=11).score(frame))
    second = order(frame, RandomRanker(seed=12).score(frame))
    assert first != second


def test_the_random_score_follows_the_finding_rather_than_the_row_position() -> None:
    """Stability under re-ordering is what makes the control comparable across runs."""
    items = [make_item(f"f_{index}") for index in range(8)]
    forward = frame_of(items)
    backward = frame_of(list(reversed(items)))
    ranker = RandomRanker(seed=3)

    forward_scores = dict(zip(forward.finding_ids, ranker.score(forward)))
    backward_scores = dict(zip(backward.finding_ids, ranker.score(backward)))
    assert forward_scores == pytest.approx(backward_scores)


def test_random_scores_stay_inside_the_unit_interval() -> None:
    frame = frame_of([make_item(f"f_{index}") for index in range(50)])
    scores = RandomRanker(seed=5).score(frame)
    assert scores.min() >= 0.0
    assert scores.max() < 1.0


# ---------------------------------------------------------------------------
# Ablation cells where a baseline's own column is gone
# ---------------------------------------------------------------------------


def test_a_baseline_whose_column_the_ablation_dropped_scores_the_neutral_and_says_so() -> None:
    frame = frame_of(
        [make_item("f_a", epss=0.9), make_item("f_b", epss=0.1)],
        ComponentFlags(a=True, b=False, c=True),
    )
    baseline = EpssOnlyRanker()
    scores = baseline.score(frame)

    assert np.allclose(scores, NEUTRAL["b_epss"])
    assert any("b_epss" in warning for warning in baseline.warnings)


def test_baselines_reading_only_base_columns_survive_every_ablation_cell() -> None:
    items = [make_item("f_a", cvss=9.0), make_item("f_b", cvss=3.0)]
    for flags in ComponentFlags.all_cells():
        frame = frame_of(items, flags)
        assert order(frame, CvssOnlyRanker().score(frame)) == ["f_a", "f_b"]
