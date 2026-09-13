"""``RankManipulationDetector``: the four rank-manipulation detectors (DESIGN.md 3.8).

A detector that fires on everything is worthless, so every test here has a matching
negative: the clean finding, the finding just inside the threshold, the finding with a
version mismatch and no curated evidence to contradict. The adversarial corpus measures
false positives for exactly this reason and these tests are its unit-level counterpart.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

from vulnprio.core.config import PipelineConfig, SandboxConfig
from vulnprio.core.enums import (
    ApplicabilityVerdict,
    Component,
    CvssVersion,
    DetectorName,
    EndpointFunction,
    ExploitMaturity,
    ExploitSource,
    HttpMethod,
    LLMBackendKind,
    PrivilegeLevel,
    Provenance,
    RankerName,
    ScannerSeverity,
    ScoreSource,
    TrustTier,
    VersionMatch,
)
from vulnprio.core.interfaces import ManipulationDetector, Ranker
from vulnprio.core.models import (
    FEATURE_GROUPS,
    ApplicabilityAssessment,
    AssetCriticality,
    BusinessImpact,
    CvssRecord,
    Endpoint,
    EnrichedFinding,
    EpssRecord,
    ExploitEvidence,
    ExploitLikelihood,
    ExploitabilityAssessment,
    Explanation,
    FeatureFrame,
    Finding,
    KevRecord,
    LLMAudit,
    RemediationCost,
    TechComponent,
    TrustSummary,
    UntrustedText,
    VulnIntel,
)
from vulnprio.rank.compose import rank_scan
from vulnprio.rank.features import NEUTRAL, FeatureBuilder
from vulnprio.rank.rank_guard import (
    DEFAULT_DISPLACEMENT_MAD_MULTIPLIER,
    DEFAULT_MIN_RANK_DISPLACEMENT,
    EVIDENCE_SOURCE_PHRASE,
    NOT_APPLICABLE_P_THRESHOLD,
    RankManipulationDetector,
    neutralise_component_a,
    ordinal,
)

AS_OF = date(2024, 6, 1)
OBSERVED_AT = datetime(2024, 5, 1, 9, 0, 0)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_item(
    finding_id: str,
    scan_id: str = "scan_1",
    *,
    kev: bool = False,
    exploit: bool = False,
    cvss: float = 6.0,
    feasibility: float = 0.4,
    verdict: ApplicabilityVerdict = ApplicabilityVerdict.APPLICABLE,
    p_applicable: float = 0.8,
    version_match: VersionMatch = VersionMatch.UNKNOWN,
    p_exploit: float = 0.3,
    floor: float = 0.0,
    conflicts: tuple[str, ...] = (),
    canary: bool = False,
    audit_canary: bool = False,
    signals: int = 0,
    max_tier: TrustTier = TrustTier.SCANNER,
) -> EnrichedFinding:
    cve_id = f"CVE-2024-{1000 + abs(hash(finding_id)) % 8000}"
    component = TechComponent(vendor="apache", product="struts", version="2.5.12")
    endpoint = Endpoint(
        endpoint_id=f"ep_{finding_id}",
        app_id="app1",
        host="shop.example.com",
        url=f"https://shop.example.com/{finding_id}",
        path=f"/{finding_id}",
        method=HttpMethod.POST,
        auth_required=PrivilegeLevel.NONE,
        observed_tech=(component,),
    )
    finding = Finding(
        finding_id=finding_id,
        scan_id=scan_id,
        app_id="app1",
        endpoint_id=endpoint.endpoint_id,
        name="Injection",
        cwe_id=89,
        cve_ids=(cve_id,),
        scanner="zap",
        scanner_severity=ScannerSeverity.HIGH,
        scanner_confidence=0.8,
        description=UntrustedText(text="finding", provenance=Provenance.SCANNER_OUTPUT),
        affected_component=component,
        observed_at=OBSERVED_AT,
    )
    intel = VulnIntel(
        cve_id=cve_id,
        as_of=AS_OF,
        published=date(2024, 1, 10),
        cvss=(CvssRecord(version=CvssVersion.V31, source=ScoreSource.NVD, base_score=cvss),),
        epss=EpssRecord(cve_id=cve_id, score=0.2, percentile=0.8, as_of=AS_OF),
        kev=KevRecord(
            cve_id=cve_id,
            in_kev=kev,
            date_added=date(2024, 2, 1) if kev else None,
            as_of=AS_OF,
        ),
        exploits=(
            (
                ExploitEvidence(
                    source=ExploitSource.EXPLOIT_DB,
                    published=date(2024, 1, 20),
                    maturity=ExploitMaturity.FUNCTIONAL,
                    verified=True,
                ),
            )
            if exploit
            else ()
        ),
    )
    audit = (
        LLMAudit(
            backend=LLMBackendKind.HEURISTIC,
            model="heuristic",
            task="exploitability",
            canary_leaked=audit_canary,
            max_tier_used=max_tier,
        )
        if audit_canary
        else None
    )
    return EnrichedFinding(
        finding=finding,
        endpoint=endpoint,
        intel=(intel,),
        asset=AssetCriticality(
            endpoint_id=endpoint.endpoint_id,
            function=EndpointFunction.PAYMENT,
            criticality=0.6,
            data_sensitivity=0.6,
            exposure=1.0,
        ),
        exploitability=ExploitabilityAssessment(
            finding_id=finding_id,
            exploit_feasibility=feasibility,
            privilege_gained=PrivilegeLevel.ADMIN,
            audit=audit,
        ),
        applicability=ApplicabilityAssessment(
            finding_id=finding_id,
            verdict=verdict,
            p_applicable=p_applicable,
            version_match=version_match,
        ),
        likelihood=ExploitLikelihood(
            finding_id=finding_id,
            attacker="opportunistic",
            p_exploit=p_exploit,
            p_exploit_uncapped=p_exploit,
            horizon_days=90,
        ),
        impact=BusinessImpact(finding_id=finding_id, total=100_000.0),
        remediation=RemediationCost(finding_id=finding_id, hours=4.0, cost=480.0),
        expected_loss=p_exploit * 100_000.0,
        trust=TrustSummary(
            max_tier_used=max_tier,
            injection_signal_count=signals,
            conflicts=conflicts,
            floor_p_exploit=floor,
            canary_leaked=canary,
        ),
        as_of=AS_OF,
    )


def floor_conflict(effective: float, argued: float, cve: str = "CVE-2024-0001") -> str:
    """The exact conflict text ``TrustLedger.apply_floor`` writes when a floor really binds.

    Reproduced here rather than imported so that a change to the ledger's wording shows up
    as a failing test in the module that depends on it, instead of as a detector that has
    silently gone quiet.
    """
    return (
        f"p_exploit held at the tier<=1 floor {effective:.3f} "
        f"(untrusted evidence argued {argued:.3f}; {cve} is in CISA KEV)"
    )


def budget_conflict(feature: str = "feasibility") -> str:
    """The other kind of conflict the ledger writes: an influence-budget cap."""
    return (
        f"tier TARGET_CONTENT tried to move {feature} by -0.700; "
        "budget 0.150 applied -0.150"
    )


def explanation_for(item: EnrichedFinding, share: float) -> Explanation:
    return Explanation(finding_id=item.finding_id, untrusted_influence_share=share)


def frame_of(items: list[EnrichedFinding]) -> FeatureFrame:
    return FeatureBuilder().build(items, {})


class FeasibilityRanker(Ranker):
    """A ranker that reads nothing but Component A, so neutralising A reorders everything."""

    name = RankerName.RANDOM

    def fit(self, frame, relevance, sample_weight=None, seed=42):
        return self

    def score(self, frame: FeatureFrame) -> np.ndarray:
        return np.asarray(frame.X["a_exploit_feasibility"].to_numpy(), dtype=float)

    def requires_fit(self) -> bool:
        return False


class ApplicabilityRanker(Ranker):
    """Reads ``a_p_applicable``, whose neutral is 0.5, so an assessment can bury a finding."""

    name = RankerName.RANDOM

    def fit(self, frame, relevance, sample_weight=None, seed=42):
        return self

    def score(self, frame: FeatureFrame) -> np.ndarray:
        return np.asarray(frame.X["a_p_applicable"].to_numpy(), dtype=float)

    def requires_fit(self) -> bool:
        return False


class CvssRanker(Ranker):
    """A ranker that reads no Component A feature at all, so neutralising A changes nothing."""

    name = RankerName.CVSS_ONLY

    def fit(self, frame, relevance, sample_weight=None, seed=42):
        return self

    def score(self, frame: FeatureFrame) -> np.ndarray:
        return np.asarray(frame.X["cvss_base_max"].to_numpy(), dtype=float)

    def requires_fit(self) -> bool:
        return False


def ladder(n: int = 8) -> list[EnrichedFinding]:
    """Findings whose Component A feasibility runs opposite to their id order."""
    return [
        make_item(f"f_{index}", feasibility=0.1 * (index + 1), cvss=float(index + 1))
        for index in range(n)
    ]


@pytest.fixture
def detector() -> RankManipulationDetector:
    return RankManipulationDetector(PipelineConfig())


def detectors_of(alerts) -> set[DetectorName]:
    return {alert.detector for alert in alerts}


# ---------------------------------------------------------------------------
# Contract and the silent case
# ---------------------------------------------------------------------------


def test_the_detector_implements_the_frozen_interface(detector) -> None:
    assert isinstance(detector, ManipulationDetector)


def test_a_clean_finding_raises_nothing(detector) -> None:
    """Applicable, no canary, a modest untrusted share and a ranker that ignores Component A."""
    items = ladder()
    frame = frame_of(items)
    context = {
        "explanations": [explanation_for(item, 0.10) for item in items],
        "frame": frame,
        "ranker": CvssRanker(),
    }
    assert detector.detect(items, context) == []


def test_with_no_context_only_the_finding_s_own_evidence_can_fire(detector) -> None:
    """No explanations and no frame means detectors 1 and 2 have nothing to look at."""
    assert detector.detect(ladder()) == []


def test_a_missing_explanation_does_not_imply_innocence_or_guilt(detector) -> None:
    items = ladder(2)
    alerts = detector.detect(items, {"explanations": [explanation_for(items[0], 0.99)]})
    assert [alert.finding_id for alert in alerts] == ["f_0"]


# ---------------------------------------------------------------------------
# Detector 1: untrusted attribution share
# ---------------------------------------------------------------------------


def test_an_untrusted_share_above_the_configured_limit_fires(detector) -> None:
    item = make_item("f_hot", max_tier=TrustTier.TARGET_CONTENT)
    limit = SandboxConfig().max_untrusted_shap_share
    alerts = detector.detect([item], {"explanations": [explanation_for(item, limit + 0.2)]})

    assert detectors_of(alerts) == {DetectorName.INFLUENCE_BUDGET}
    assert alerts[0].finding_id == "f_hot"
    assert "55%" in alerts[0].message                       # the share itself
    assert "35%" in alerts[0].message                       # the limit it passed
    assert EVIDENCE_SOURCE_PHRASE[TrustTier.TARGET_CONTENT] in alerts[0].message


def test_a_share_at_the_limit_stays_silent(detector) -> None:
    item = make_item("f_edge")
    limit = SandboxConfig().max_untrusted_shap_share
    assert detector.detect([item], {"explanations": [explanation_for(item, limit)]}) == []


def test_the_share_limit_is_taken_from_the_sandbox_configuration() -> None:
    item = make_item("f_strict")
    strict = RankManipulationDetector(
        PipelineConfig(sandbox=SandboxConfig(max_untrusted_shap_share=0.05))
    )
    alerts = strict.detect([item], {"explanations": [explanation_for(item, 0.10)]})
    assert detectors_of(alerts) == {DetectorName.INFLUENCE_BUDGET}


def test_the_alert_severity_tracks_the_share(detector) -> None:
    item = make_item("f_hot")
    mild = detector.detect([item], {"explanations": [explanation_for(item, 0.40)]})[0]
    severe = detector.detect([item], {"explanations": [explanation_for(item, 0.95)]})[0]
    assert severe.severity > mild.severity


# ---------------------------------------------------------------------------
# Detector 2: displacement under neutralisation
# ---------------------------------------------------------------------------


def test_neutralising_component_a_resets_only_component_a(detector) -> None:
    frame = frame_of(ladder())
    neutral = neutralise_component_a(frame)

    assert neutral.feature_names == frame.feature_names
    assert neutral.finding_ids == frame.finding_ids
    for name in frame.feature_names:
        if FEATURE_GROUPS[name] is Component.A:
            assert (neutral.X[name] == NEUTRAL[name]).all(), name
        else:
            assert (neutral.X[name].to_numpy() == frame.X[name].to_numpy()).all(), name


def outlier_queue(*, buried: bool = False, n_peers: int = 9) -> list[EnrichedFinding]:
    """A scan where one finding's own assessment moves it and nobody else's moves them.

    Peers sit exactly on ``a_p_applicable``'s neutral of 0.5, so neutralising any one of
    them changes nothing and its displacement is zero. The outlier is far off the neutral,
    and its id is chosen so the tie-break carries it across the whole queue.
    """
    outlier_id, peer_ids = ("f_0", [f"f_{i}" for i in range(1, n_peers + 1)]) if buried else (
        "f_zz",
        [f"f_{i}" for i in range(1, n_peers + 1)],
    )
    return [
        make_item(outlier_id, p_applicable=0.0 if buried else 1.0),
        *[make_item(peer, p_applicable=0.5) for peer in peer_ids],
    ]


def test_one_finding_moved_far_beyond_its_scan_is_flagged(detector) -> None:
    items = outlier_queue()
    alerts = detector.detect(items, {"frame": frame_of(items), "ranker": ApplicabilityRanker()})

    assert detectors_of(alerts) == {DetectorName.DISPLACEMENT}
    assert [alert.finding_id for alert in alerts] == ["f_zz"]


def test_the_rank_delta_is_positive_when_untrusted_content_promoted_a_finding(detector) -> None:
    items = outlier_queue()
    alert = detector.detect(
        items, {"frame": frame_of(items), "ranker": ApplicabilityRanker()}
    )[0]

    assert alert.rank_delta == 9                 # 1st as assessed, 10th with its own A removed
    assert "9 places up the queue" in alert.message
    assert "would rank 10th rather than 1st" in alert.message


def test_the_rank_delta_is_negative_when_untrusted_content_buried_a_finding(detector) -> None:
    """``a_p_applicable``'s neutral is 0.5, so an assessment below it is a demotion."""
    items = outlier_queue(buried=True)
    alert = detector.detect(
        items, {"frame": frame_of(items), "ranker": ApplicabilityRanker()}
    )[0]

    assert alert.finding_id == "f_0"
    assert alert.rank_delta == -9                # 10th as assessed, 1st with its own A removed
    assert "9 places down the queue" in alert.message
    assert "would rank 1st rather than 10th" in alert.message


def test_a_scan_where_the_assessment_moves_everything_flags_nobody(detector) -> None:
    """The headline change, and the reason the old fixed threshold had to go.

    The ladder is a smooth ramp: every finding's own assessment moves it, by 0 through 7
    places. That is semantic assessment doing its job across a whole queue, not one
    finding being singled out. A fixed limit of three called six of the eight manipulated;
    against the scan's own distribution none of them is an outlier.
    """
    items = ladder()
    alerts = detector.detect(items, {"frame": frame_of(items), "ranker": FeasibilityRanker()})
    assert alerts == []


def test_a_finding_its_own_assessment_did_not_move_is_not_flagged(detector) -> None:
    """The regression the all-at-once counterfactual produced.

    ``f_0`` holds the *lowest* Component A feasibility in the ladder: its own assessment
    lifted it nowhere and it sits last either way. Neutralising the whole column at once
    reported it displaced by seven places, because every other finding collapsed to a tie
    and the id tie-break reshuffled around it. Per finding, it does not move at all.
    """
    items = ladder()
    actual, counterfactual, delta, _threshold = detector._displacements(
        frame_of(items), FeasibilityRanker()
    )["f_0"]
    assert (actual, counterfactual, delta) == (8, 8, 0)


def test_the_threshold_is_relative_to_the_scan_not_a_constant(detector) -> None:
    """The same displacement is an outlier in a quiet queue and unremarkable in a busy one."""
    quiet = outlier_queue()
    busy = ladder()

    quiet_moves = detector._displacements(frame_of(quiet), ApplicabilityRanker())
    busy_moves = detector._displacements(frame_of(busy), FeasibilityRanker())

    quiet_threshold = quiet_moves["f_zz"][3]
    busy_threshold = busy_moves["f_7"][3]
    assert quiet_threshold < busy_threshold
    assert abs(quiet_moves["f_zz"][2]) > quiet_threshold      # flagged
    assert abs(busy_moves["f_7"][2]) < busy_threshold         # not flagged


def test_the_absolute_floor_protects_a_scan_where_nothing_moves(detector) -> None:
    """Median and deviation are both zero there, so without a floor a two-place shift
    would be a towering outlier."""
    assert detector._scan_threshold([0, 0, 0, 0, 0]) == pytest.approx(
        float(DEFAULT_MIN_RANK_DISPLACEMENT)
    )
    assert detector._scan_threshold([]) == pytest.approx(float(DEFAULT_MIN_RANK_DISPLACEMENT))


def test_the_threshold_uses_a_robust_dispersion(detector) -> None:
    """Median plus a multiple of the scaled MAD, not mean plus standard deviation.

    The sample is small, skewed and contains the outliers being hunted; a mean would be
    dragged towards them and a standard deviation inflated by them, so the very finding
    that should stand out helps to hide itself.
    """
    displacements = [3, 3, 3, 3, 3, 3, 3, 30]
    median, mad = 3.0, 0.0
    assert detector._scan_threshold(displacements) == pytest.approx(
        max(float(DEFAULT_MIN_RANK_DISPLACEMENT), median + detector.mad_multiplier * 1.4826 * mad)
    )
    assert 30 > detector._scan_threshold(displacements)


def test_a_ranker_that_ignores_component_a_displaces_nothing(detector) -> None:
    items = ladder()
    alerts = detector.detect(items, {"frame": frame_of(items), "ranker": CvssRanker()})
    assert DetectorName.DISPLACEMENT not in detectors_of(alerts)


def test_the_displacement_sensitivity_is_configurable() -> None:
    items = outlier_queue()
    context = {"frame": frame_of(items), "ranker": ApplicabilityRanker()}
    assert RankManipulationDetector(min_rank_displacement=100).detect(items, context) == []
    assert RankManipulationDetector(mad_multiplier=0.0).detect(items, context)


def test_the_multiplier_comes_from_config_when_the_field_exists() -> None:
    """Reads ``RankingConfig.displacement_mad_multiplier`` the moment it is added."""
    assert RankManipulationDetector().mad_multiplier == pytest.approx(
        DEFAULT_DISPLACEMENT_MAD_MULTIPLIER
    )
    assert RankManipulationDetector(mad_multiplier=1.5).mad_multiplier == pytest.approx(1.5)


def test_the_defaults_are_the_documented_ones(detector) -> None:
    assert detector.min_rank_displacement == DEFAULT_MIN_RANK_DISPLACEMENT
    assert detector.mad_multiplier == pytest.approx(DEFAULT_DISPLACEMENT_MAD_MULTIPLIER)


def test_a_cell_without_component_a_has_nothing_to_neutralise(detector) -> None:
    from vulnprio.core.models import ComponentFlags

    items = ladder()
    frame = FeatureBuilder().build(items, {}, ComponentFlags(a=False, b=True, c=True))
    alerts = detector.detect(items, {"frame": frame, "ranker": CvssRanker()})
    assert DetectorName.DISPLACEMENT not in detectors_of(alerts)


def test_a_ranker_that_cannot_score_degrades_instead_of_crashing(detector) -> None:
    class Broken(Ranker):
        name = RankerName.RANDOM

        def fit(self, frame, relevance, sample_weight=None, seed=42):
            return self

        def score(self, frame):
            raise RuntimeError("booster is gone")

    items = ladder()
    assert detector.detect(items, {"frame": frame_of(items), "ranker": Broken()}) == []


# ---------------------------------------------------------------------------
# Detector 3: contradiction of tier <= 1 evidence
# ---------------------------------------------------------------------------


def test_denying_applicability_against_kev_and_a_version_match_fires(detector) -> None:
    """The sentence "KEV membership cannot be argued away" as an executable check."""
    item = make_item(
        "f_denied",
        kev=True,
        verdict=ApplicabilityVerdict.NOT_APPLICABLE,
        p_applicable=0.02,
        version_match=VersionMatch.MATCH,
    )
    alerts = detector.detect([item])
    assert detectors_of(alerts) == {DetectorName.CONSISTENCY}
    assert "known exploited vulnerabilities" in alerts[0].message
    assert "2%" in alerts[0].message


def test_a_verified_exploit_is_enough_on_its_own(detector) -> None:
    item = make_item(
        "f_denied",
        exploit=True,
        verdict=ApplicabilityVerdict.NOT_APPLICABLE,
        p_applicable=0.02,
        version_match=VersionMatch.MATCH,
    )
    alerts = detector.detect([item])
    assert detectors_of(alerts) == {DetectorName.CONSISTENCY}
    assert "working exploit code" in alerts[0].message


def test_a_low_probability_counts_as_denial_even_under_an_uncertain_verdict(detector) -> None:
    item = make_item(
        "f_quiet",
        kev=True,
        verdict=ApplicabilityVerdict.UNCERTAIN,
        p_applicable=NOT_APPLICABLE_P_THRESHOLD,
        version_match=VersionMatch.MATCH,
    )
    assert detectors_of(detector.detect([item])) == {DetectorName.CONSISTENCY}


def test_denying_applicability_with_a_version_mismatch_is_legitimate(detector) -> None:
    """Version evidence is authoritative: this is the assessment doing its job, not an attack."""
    item = make_item(
        "f_mismatch",
        kev=True,
        verdict=ApplicabilityVerdict.NOT_APPLICABLE,
        p_applicable=0.02,
        version_match=VersionMatch.MISMATCH,
    )
    assert detector.detect([item]) == []


def test_denying_applicability_with_no_curated_evidence_is_legitimate(detector) -> None:
    item = make_item(
        "f_plain",
        verdict=ApplicabilityVerdict.NOT_APPLICABLE,
        p_applicable=0.02,
        version_match=VersionMatch.MATCH,
    )
    assert detector.detect([item]) == []


def test_a_floor_the_trust_ledger_had_to_enforce_fires(detector) -> None:
    """The ledger recorded that it held the probability up, so something argued it down."""
    item = make_item(
        "f_floored",
        kev=True,
        p_exploit=0.50,
        floor=0.50,
        conflicts=(floor_conflict(effective=0.50, argued=0.10),),
        max_tier=TrustTier.REFERENCE_PAGE,
    )
    alerts = detector.detect([item])

    assert detectors_of(alerts) == {DetectorName.CONSISTENCY}
    assert "argued this finding's chance of being exploited downwards" in alerts[0].message
    assert "50%" in alerts[0].message


def test_a_probability_legitimately_below_the_nominal_floor_raises_nothing(detector) -> None:
    """The regression this detector was getting wrong on 18% of a real run.

    ``TrustSummary.floor_p_exploit`` is the *nominal* floor the curated evidence implies.
    ``TrustLedger.apply_floor`` enforces ``min(nominal floor, trusted reference)``, so an
    opportunistic attacker on a short horizon can legitimately score a KEV-listed finding
    at 0.353 against a nominal floor of 0.500 with no untrusted text involved anywhere.
    Nothing was argued away, the ledger recorded no conflict, and there is no breach.
    """
    item = make_item("f_legit", kev=True, p_exploit=0.353, floor=0.50, conflicts=())
    assert detector.detect([item]) == []


def test_a_probability_at_the_floor_is_fine(detector) -> None:
    item = make_item("f_held", kev=True, p_exploit=0.50, floor=0.50)
    assert detector.detect([item]) == []


def test_an_influence_budget_conflict_is_not_a_floor_breach(detector) -> None:
    """The ledger's other conflict record must not be mistaken for a floor being held."""
    item = make_item(
        "f_capped",
        kev=True,
        p_exploit=0.20,
        floor=0.50,
        conflicts=(budget_conflict(), budget_conflict("exposure")),
    )
    assert detector.detect([item]) == []


def test_the_floor_alert_carries_the_raw_ledger_record_for_the_research_view(detector) -> None:
    record = floor_conflict(effective=0.50, argued=0.10)
    item = make_item("f_floored", kev=True, p_exploit=0.50, floor=0.50, conflicts=(record,))
    detailed = detector.detect_detailed([item])

    assert [note for _, note in detailed] == [record]


# ---------------------------------------------------------------------------
# Detector 4: canary leak
# ---------------------------------------------------------------------------


def test_a_canary_leak_recorded_on_the_trust_summary_fires(detector) -> None:
    alerts = detector.detect([make_item("f_leak", canary=True)])
    assert detectors_of(alerts) == {DetectorName.CANARY}
    assert alerts[0].severity == 1.0


def test_a_canary_leak_recorded_only_on_an_audit_fires_too(detector) -> None:
    alerts = detector.detect([make_item("f_leak", audit_canary=True)])
    assert detectors_of(alerts) == {DetectorName.CANARY}


def test_no_canary_leak_means_no_canary_alert(detector) -> None:
    assert detector.detect([make_item("f_clean")]) == []


# ---------------------------------------------------------------------------
# Several detectors at once, and the path into the ranked queue
# ---------------------------------------------------------------------------


def test_a_thoroughly_manipulated_finding_trips_every_detector(detector) -> None:
    # The id sorts first and the assessment puts p_applicable far below its 0.5 neutral,
    # so removing Component A from this one finding alone carries it the length of the
    # queue while every peer, sitting on the neutral, does not move at all.
    manipulated = make_item(
        "f_0",
        kev=True,
        canary=True,
        feasibility=0.99,
        verdict=ApplicabilityVerdict.NOT_APPLICABLE,
        p_applicable=0.01,
        version_match=VersionMatch.MATCH,
        p_exploit=0.50,
        floor=0.50,
        conflicts=(floor_conflict(effective=0.50, argued=0.05),),
        max_tier=TrustTier.TARGET_CONTENT,
    )
    others = [make_item(f"f_{index}", p_applicable=0.5) for index in range(1, 9)]
    items = [manipulated, *others]
    frame = frame_of(items)

    alerts = detector.detect(
        items,
        {
            "explanations": [explanation_for(manipulated, 0.92)],
            "frame": frame,
            "ranker": ApplicabilityRanker(),
        },
    )
    mine = [alert for alert in alerts if alert.finding_id == "f_0"]
    assert detectors_of(mine) == {
        DetectorName.CANARY,
        DetectorName.INFLUENCE_BUDGET,
        DetectorName.DISPLACEMENT,
        DetectorName.CONSISTENCY,
    }


def test_detection_is_deterministic_and_ordered_by_input(detector) -> None:
    items = outlier_queue()
    context = {"frame": frame_of(items), "ranker": ApplicabilityRanker()}
    first = detector.detect(items, context)
    second = detector.detect(items, context)

    assert [alert.model_dump() for alert in first] == [alert.model_dump() for alert in second]
    order = [alert.finding_id for alert in first]
    assert order == sorted(order, key=lambda name: [item.finding_id for item in items].index(name))


def test_alerts_reach_the_ranked_queue_through_rank_scan(detector) -> None:
    items = outlier_queue()
    frame = frame_of(items)
    ranker = ApplicabilityRanker()
    alerts = detector.detect(items, {"frame": frame, "ranker": ranker})

    result = rank_scan(frame, ranker, items, {}, PipelineConfig(), alerts=alerts)
    flagged = {item.finding_id for item in result.items if item.manipulation_flag}
    assert flagged == {alert.finding_id for alert in alerts}


def test_alerts_already_on_a_finding_survive_into_the_queue() -> None:
    from vulnprio.core.models import ManipulationAlert

    existing = ManipulationAlert(
        finding_id="f_0", detector=DetectorName.PRE_LLM_PATTERN, message="pattern hit"
    )
    items = ladder(3)
    items[0] = items[0].model_copy(update={"alerts": (existing,)})
    frame = frame_of(items)

    result = rank_scan(frame, CvssRanker(), items, {}, PipelineConfig())
    first = next(item for item in result.items if item.finding_id == "f_0")
    assert first.alerts == (existing,)


# ---------------------------------------------------------------------------
# Alert text is what the report prints, so it is written for an operator
# ---------------------------------------------------------------------------

#: Vocabulary that belongs to the implementation, not to the person reading the report.
INTERNAL_VOCABULARY = (
    "Component A",
    "Component B",
    "Component C",
    "tier<=1",
    "tier <= 1",
    "p_exploit",
    "p_applicable",
    "SHAP",
    "TARGET_CONTENT",
    "REFERENCE_PAGE",
    "CURATED_FEED",
    "max_untrusted_shap_share",
    "neutralised",
    "rank_delta",
    "verdict",
)


def every_alert(detector: RankManipulationDetector) -> list:
    """One alert of every kind this detector can raise, for a vocabulary sweep."""
    manipulated = make_item(
        "f_0",
        kev=True,
        canary=True,
        verdict=ApplicabilityVerdict.NOT_APPLICABLE,
        p_applicable=0.01,
        version_match=VersionMatch.MATCH,
        p_exploit=0.50,
        floor=0.50,
        conflicts=(floor_conflict(effective=0.50, argued=0.05),),
        max_tier=TrustTier.REFERENCE_PAGE,
    )
    exploit_denied = make_item(
        "f_8",
        exploit=True,
        verdict=ApplicabilityVerdict.NOT_APPLICABLE,
        p_applicable=0.01,
        version_match=VersionMatch.MATCH,
    )
    others = [make_item(f"f_{index}", p_applicable=0.5) for index in range(1, 8)]
    items = [manipulated, exploit_denied, *others]
    return detector.detect(
        items,
        {
            "explanations": [explanation_for(manipulated, 0.92)],
            "frame": frame_of(items),
            "ranker": ApplicabilityRanker(),
        },
    )


def test_no_alert_message_uses_internal_vocabulary(detector) -> None:
    """``ManipulationAlert.message`` is rendered verbatim in the assessment report."""
    alerts = every_alert(detector)
    assert len(alerts) >= 5

    for alert in alerts:
        for term in INTERNAL_VOCABULARY:
            assert term.lower() not in alert.message.lower(), f"{term!r} in {alert.message!r}"


def test_every_alert_message_reads_as_a_sentence(detector) -> None:
    for alert in every_alert(detector):
        assert alert.message[0].isupper(), alert.message
        assert alert.message.endswith("."), alert.message
        assert len(alert.message) <= 400


def test_the_technical_form_is_still_available_for_the_research_view(detector) -> None:
    """Plain messages for the operator; the precise internal form alongside them."""
    detailed = RankManipulationDetector(PipelineConfig()).detect_detailed(
        [
            make_item(
                "f_9",
                canary=True,
                max_tier=TrustTier.REFERENCE_PAGE,
            )
        ]
    )
    alert, note = detailed[0]
    assert "canary" not in alert.message.lower()
    assert "canary leak" in note
    assert "REFERENCE_PAGE" in note


def test_detect_and_detect_detailed_agree_on_the_alerts(detector) -> None:
    items = outlier_queue()
    context = {"frame": frame_of(items), "ranker": ApplicabilityRanker()}
    plain = detector.detect(items, context)
    detailed = detector.detect_detailed(items, context)

    assert [alert.model_dump() for alert in plain] == [
        alert.model_dump() for alert, _ in detailed
    ]
    assert all(note for _, note in detailed)


def test_queue_positions_are_rendered_as_ordinals() -> None:
    assert [ordinal(value) for value in (1, 2, 3, 4, 11, 12, 13, 21, 22, 101)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "101st"
    ]
