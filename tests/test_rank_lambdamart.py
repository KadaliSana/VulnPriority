"""``LambdaMartRanker`` and ``CostSensitiveExploitHead`` (DESIGN.md 3.8, Gaps 5 and 7).

The headline test plants a signal - KEV membership decides relevance, CVSS is noise - and
asserts the learned ranker beats the CVSS-only baseline on NDCG@10. NDCG is computed by a
local helper rather than imported from ``vulnpriority.eval``, which is written in parallel.

The cost-sensitive probability head is tested here too, because Gap 7 pairs the two: the
ranker's impact-weighted pairs and the head's ``scale_pos_weight`` are the same correction
for the same rarity, applied to an ordering and to a probability respectively.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from vulnpriority.core.config import RankingConfig
from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    CvssVersion,
    EndpointFunction,
    ExploitMaturity,
    ExploitSource,
    HttpMethod,
    PrivilegeLevel,
    Provenance,
    RankerName,
    ScannerSeverity,
    ScoreSource,
    VersionMatch,
)
from vulnpriority.core.errors import RankerNotFittedError, VulnPriorityError
from vulnpriority.core.interfaces import ProbabilityModel, Ranker
from vulnpriority.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    BusinessImpact,
    ChainScore,
    ComponentFlags,
    CvssRecord,
    Endpoint,
    EnrichedFinding,
    EpssRecord,
    ExploitEvidence,
    ExploitLikelihood,
    ExploitabilityAssessment,
    FeatureFrame,
    Finding,
    KevRecord,
    RemediationCost,
    UntrustedText,
    VulnIntel,
)
from vulnpriority.core.registry import get_ranker
from vulnpriority.rank.baselines import CvssOnlyRanker
from vulnpriority.rank.features import FeatureBuilder
from vulnpriority.rank.lambdamart import LambdaMartRanker
from vulnpriority.rank.likelihood_head import CostSensitiveExploitHead

AS_OF = date(2024, 6, 1)
OBSERVED_AT = datetime(2024, 5, 1, 9, 0, 0)
SEED = 7


# ---------------------------------------------------------------------------
# NDCG, defined locally so this module does not depend on vulnpriority.eval
# ---------------------------------------------------------------------------


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Standard exponential-gain NDCG@k; 0.0 when the ideal ranking has no gain."""

    def dcg(grades: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(position + 2) for position, grade in enumerate(grades))

    actual = [int(relevance.get(finding_id, 0)) for finding_id in ranked_ids[:k]]
    ideal = sorted((int(relevance.get(finding_id, 0)) for finding_id in ranked_ids), reverse=True)[:k]
    best = dcg(ideal)
    return dcg(actual) / best if best > 0 else 0.0


def mean_ndcg(frame: FeatureFrame, scores: np.ndarray, relevance: dict[str, int], k: int) -> float:
    """Mean NDCG@k over query groups, which is how the protocol aggregates it."""
    per_scan: dict[str, list[tuple[float, str]]] = {}
    for index, finding_id in enumerate(frame.finding_ids):
        per_scan.setdefault(frame.group_ids[index], []).append((float(scores[index]), finding_id))

    values = []
    for rows in per_scan.values():
        ordered = [finding_id for _, finding_id in sorted(rows, key=lambda row: (-row[0], row[1]))]
        values.append(ndcg_at_k(ordered, relevance, k))
    return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Synthetic data with a planted signal
# ---------------------------------------------------------------------------


def _intel(cve_id: str, kev: bool, epss: float, cvss: float) -> VulnIntel:
    return VulnIntel(
        cve_id=cve_id,
        as_of=AS_OF,
        published=date(2024, 1, 10),
        cvss=(
            CvssRecord(
                version=CvssVersion.V31,
                source=ScoreSource.NVD,
                base_score=round(cvss, 1),
                submetrics={"AC": "L", "PR": "N", "UI": "N", "C": "H", "I": "H", "A": "H"},
            ),
        ),
        epss=EpssRecord(cve_id=cve_id, score=epss, percentile=epss, as_of=AS_OF),
        kev=KevRecord(cve_id=cve_id, in_kev=kev, date_added=date(2024, 2, 1) if kev else None, as_of=AS_OF),
        exploits=(
            (
                ExploitEvidence(
                    source=ExploitSource.EXPLOIT_DB,
                    published=date(2024, 1, 20),
                    maturity=ExploitMaturity.FUNCTIONAL,
                    verified=True,
                ),
            )
            if kev
            else ()
        ),
    )


def _enriched(
    finding_id: str,
    scan_id: str,
    *,
    kev: bool,
    epss: float,
    cvss: float,
    criticality: float,
    feasibility: float,
    p_exploit: float,
    impact: float,
) -> EnrichedFinding:
    cve_id = f"CVE-2024-{abs(hash(finding_id)) % 9000 + 1000}"
    endpoint = Endpoint(
        endpoint_id=f"ep_{finding_id}",
        app_id="app1",
        host="shop.example.com",
        url=f"https://shop.example.com/{finding_id}",
        path=f"/{finding_id}",
        method=HttpMethod.POST,
        auth_required=PrivilegeLevel.NONE,
        parameters=("q",),
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
        scanner_severity=ScannerSeverity.MEDIUM,
        scanner_confidence=0.5,
        description=UntrustedText(text="finding", provenance=Provenance.SCANNER_OUTPUT),
        observed_at=OBSERVED_AT,
    )
    return EnrichedFinding(
        finding=finding,
        endpoint=endpoint,
        intel=(_intel(cve_id, kev, epss, cvss),),
        asset=AssetCriticality(
            endpoint_id=endpoint.endpoint_id,
            function=EndpointFunction.API_DATA,
            criticality=criticality,
            data_sensitivity=criticality,
            exposure=1.0,
        ),
        exploitability=ExploitabilityAssessment(
            finding_id=finding_id,
            exploit_feasibility=feasibility,
            exploit_maturity=ExploitMaturity.POC,
            privilege_gained=PrivilegeLevel.USER,
            impact_c=0.5,
            impact_i=0.5,
            impact_a=0.5,
        ),
        applicability=ApplicabilityAssessment(
            finding_id=finding_id,
            verdict=ApplicabilityVerdict.APPLICABLE,
            p_applicable=0.8,
            version_match=VersionMatch.UNKNOWN,
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


def planted_dataset(
    n_scans: int = 10, per_scan: int = 24, seed: int = SEED
) -> tuple[list[EnrichedFinding], dict[str, ChainScore], dict[str, int]]:
    """KEV membership decides relevance.

    CVSS is drawn independently and is therefore pure noise, which is the point: the
    CVSS-only baseline has nothing to find. EPSS is drawn from overlapping ranges so that
    it is informative but not separating - otherwise the model could reach a perfect
    ordering through EPSS alone and never split on ``b_kev``, leaving the monotone
    constraint on KEV vacuous and untested.
    """
    rng = np.random.default_rng(seed)
    enriched: list[EnrichedFinding] = []
    chain: dict[str, ChainScore] = {}
    relevance: dict[str, int] = {}

    for scan in range(n_scans):
        scan_id = f"scan_{scan}"
        for index in range(per_scan):
            finding_id = f"f_{scan}_{index}"
            kev = bool(rng.random() < 0.25)
            item = _enriched(
                finding_id,
                scan_id,
                kev=kev,
                epss=float(rng.uniform(0.25, 0.65) if kev else rng.uniform(0.0, 0.45)),
                cvss=float(rng.uniform(3.0, 10.0)),
                criticality=float(rng.random()),
                feasibility=float(rng.random()),
                p_exploit=float(rng.uniform(0.01, 0.6)),
                impact=float(rng.uniform(1_000.0, 500_000.0)),
            )
            enriched.append(item)
            chain[finding_id] = ChainScore(
                finding_id=finding_id,
                reach_delta=float(rng.uniform(0.0, 50_000.0)),
                max_path_prob_to_target=float(rng.random()),
                n_paths_through=int(rng.integers(0, 6)),
                betweenness=float(rng.random()),
                hops_from_entry=int(rng.integers(0, 4)),
                privilege_gain=int(rng.integers(0, 3)),
            )
            relevance[finding_id] = 4 if kev else 0
    return enriched, chain, relevance


def split_frames(
    flags: ComponentFlags = ComponentFlags(),
    train_scans: int = 7,
) -> tuple[FeatureFrame, FeatureFrame, dict[str, int]]:
    """Time-ordered-ish split: the first ``train_scans`` scans train, the rest test."""
    enriched, chain, relevance = planted_dataset()
    builder = FeatureBuilder()
    train = [item for item in enriched if int(item.scan_id.split("_")[1]) < train_scans]
    test = [item for item in enriched if int(item.scan_id.split("_")[1]) >= train_scans]
    return builder.build(train, chain, flags), builder.build(test, chain, flags), relevance


def fast_config(**overrides) -> RankingConfig:
    """The production configuration with a smaller forest, so the suite stays quick.

    ``model_path`` is off unless a test asks for it. The shipped default points at a model
    inside the package, and a ranker that cannot fit adopts it - which is the right
    behaviour in production and the wrong one in a test, where it would make the result
    depend on whether somebody had run ``train-ranker`` on this machine. Tests that want
    that path build their own model and pass its path explicitly.
    """
    overrides.setdefault("model_path", None)
    return RankingConfig(n_estimators=120, max_depth=3, learning_rate=0.1, **overrides)


@pytest.fixture(scope="module")
def fitted() -> tuple[LambdaMartRanker, FeatureFrame, FeatureFrame, dict[str, int]]:
    train, test, relevance = split_frames()
    labels = np.array([relevance[finding_id] for finding_id in train.finding_ids], dtype=float)
    weights = np.array(
        [1.0 + math.log1p(value / 1000.0) for value in train.X["b_impact_log"]], dtype=float
    )
    ranker = LambdaMartRanker(fast_config()).fit(train, labels, weights, seed=SEED)
    return ranker, train, test, relevance


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_the_ranker_implements_the_frozen_interface_and_is_registered() -> None:
    ranker = LambdaMartRanker()
    assert isinstance(ranker, Ranker)
    assert ranker.name == RankerName.LAMBDAMART
    assert ranker.requires_fit() is True
    assert get_ranker(RankerName.LAMBDAMART) is LambdaMartRanker


def test_scoring_before_fitting_is_an_error() -> None:
    train, _, _ = split_frames()
    with pytest.raises(RankerNotFittedError):
        LambdaMartRanker().score(train)


def test_a_relevance_vector_of_the_wrong_length_is_rejected() -> None:
    train, _, _ = split_frames()
    with pytest.raises(VulnPriorityError):
        LambdaMartRanker(fast_config()).fit(train, np.zeros(3))


def test_scoring_a_frame_from_a_different_ablation_cell_is_refused(fitted) -> None:
    ranker, _, _, _ = fitted
    _, narrow, _ = split_frames(ComponentFlags(a=False, b=True, c=True))
    with pytest.raises(VulnPriorityError):
        ranker.score(narrow)


# ---------------------------------------------------------------------------
# The planted signal
# ---------------------------------------------------------------------------


def test_lambdamart_learns_the_planted_signal_and_beats_cvss_only(fitted) -> None:
    ranker, _, test, relevance = fitted

    learned = mean_ndcg(test, ranker.score(test), relevance, k=10)
    cvss = mean_ndcg(test, CvssOnlyRanker().score(test), relevance, k=10)

    assert ranker.used_fallback is False
    assert learned > 0.9, f"learned NDCG@10 was only {learned:.3f}"
    assert learned > cvss + 0.1, f"learned {learned:.3f} did not beat CVSS-only {cvss:.3f}"


def test_the_learned_ordering_puts_kev_findings_first(fitted) -> None:
    ranker, _, test, relevance = fitted
    scores = ranker.score(test)
    positives = [
        float(score)
        for score, finding_id in zip(scores, test.finding_ids)
        if relevance[finding_id] > 0
    ]
    negatives = [
        float(score)
        for score, finding_id in zip(scores, test.finding_ids)
        if relevance[finding_id] == 0
    ]
    assert min(positives) > max(negatives)


def test_fitting_is_reproducible_under_a_fixed_seed() -> None:
    train, test, relevance = split_frames()
    labels = np.array([relevance[finding_id] for finding_id in train.finding_ids], dtype=float)
    first = LambdaMartRanker(fast_config()).fit(train, labels, seed=SEED).score(test)
    second = LambdaMartRanker(fast_config()).fit(train, labels, seed=SEED).score(test)
    assert np.allclose(first, second)


# ---------------------------------------------------------------------------
# Monotone constraints
# ---------------------------------------------------------------------------


def _with_column(frame: FeatureFrame, column: str, value: float) -> FeatureFrame:
    """A copy of ``frame`` with one column forced to a constant."""
    X = frame.X.copy()
    X[column] = float(value)
    return FeatureFrame(
        X=X,
        finding_ids=list(frame.finding_ids),
        group_ids=list(frame.group_ids),
        flags=frame.flags,
    )


def test_the_monotone_vector_is_mapped_onto_the_present_columns(fitted) -> None:
    ranker, train, _, _ = fitted
    vector = ranker.monotone_vector()
    assert len(vector) == len(train.feature_names)
    constrained = {
        name for name, value in zip(train.feature_names, vector) if value == 1
    }
    assert constrained == {"b_kev", "b_epss", "c_reach_delta_log", "b_expected_loss_log"}


@pytest.mark.parametrize(
    "column", ["b_kev", "b_epss", "b_expected_loss_log", "c_reach_delta_log"]
)
def test_raising_a_constrained_feature_never_lowers_the_score(fitted, column) -> None:
    ranker, _, test, _ = fitted
    low = ranker.score(_with_column(test, column, float(test.X[column].min())))
    high = ranker.score(_with_column(test, column, float(test.X[column].max()) + 1.0))
    assert np.all(high >= low - 1e-6)


def test_switching_kev_on_never_demotes_a_finding(fitted) -> None:
    """The headline security property: an injection cannot argue KEV downward."""
    ranker, _, test, _ = fitted
    without = ranker.score(_with_column(test, "b_kev", 0.0))
    with_kev = ranker.score(_with_column(test, "b_kev", 1.0))
    assert np.all(with_kev >= without - 1e-6)
    assert float((with_kev - without).max()) > 0.0, "the constraint is vacuous here"


def test_no_monotone_constraint_sits_on_a_feature_untrusted_content_can_reach() -> None:
    """The rule that decides what may join the constraint set.

    A monotone constraint is a promise that the score will never move downwards on that
    input. On a curated-feed fact that is a safeguard - an injection cannot argue KEV
    membership down because it cannot reach ``b_kev`` at all. On a feature derived from a
    web page or from the target's own responses the same promise inverts into a
    guaranteed, model-independent lever: plant the input, and the score provably cannot
    fall. ``a_intel_corroborates_feeds`` is the tempting case and is excluded for exactly
    this reason; the exploitation evidence it corroborates is already constrained through
    ``b_kev``, where nothing untrusted can follow it.
    """
    from vulnpriority.core.enums import UNTRUSTED_TIERS
    from vulnpriority.rank.explain import FEATURE_TIER

    constrained = {name for name, value in RankingConfig().monotone.items() if value != 0}
    assert constrained

    for name in constrained:
        assert FEATURE_TIER[name] not in UNTRUSTED_TIERS, (
            f"{name} is constrained monotone but its evidence comes from "
            f"{FEATURE_TIER[name].name}, which untrusted content can author"
        )
    assert not any(name.startswith("a_intel_") for name in constrained)


def test_an_ablation_cell_without_component_b_carries_fewer_constraints() -> None:
    train, _, relevance = split_frames(ComponentFlags(a=True, b=False, c=True))
    labels = np.array([relevance[finding_id] for finding_id in train.finding_ids], dtype=float)
    ranker = LambdaMartRanker(fast_config()).fit(train, labels, seed=SEED)
    constrained = {
        name for name, value in zip(train.feature_names, ranker.monotone_vector()) if value == 1
    }
    assert constrained == {"c_reach_delta_log"}


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------


def test_per_row_weights_are_aggregated_to_the_per_group_weights_xgboost_requires(fitted) -> None:
    ranker, _, _, _ = fitted
    assert ranker.group_weight_aggregated is True


def test_weights_supplied_per_group_are_used_unchanged() -> None:
    train, _, relevance = split_frames()
    labels = np.array([relevance[finding_id] for finding_id in train.finding_ids], dtype=float)
    groups = train.group_sizes()
    ranker = LambdaMartRanker(fast_config()).fit(
        train, labels, np.full(groups.size, 2.0), seed=SEED
    )
    assert ranker.group_weight_aggregated is False
    assert ranker.used_fallback is False


def test_a_weight_vector_of_an_impossible_length_is_rejected() -> None:
    train, _, relevance = split_frames()
    labels = np.array([relevance[finding_id] for finding_id in train.finding_ids], dtype=float)
    with pytest.raises(VulnPriorityError):
        LambdaMartRanker(fast_config()).fit(train, labels, np.ones(5), seed=SEED)


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


def test_a_single_query_group_falls_back_to_expected_loss_rather_than_crashing() -> None:
    train, _, relevance = split_frames(train_scans=1)
    labels = np.array([relevance[finding_id] for finding_id in train.finding_ids], dtype=float)

    ranker = LambdaMartRanker(fast_config()).fit(train, labels, seed=SEED)

    assert ranker.used_fallback is True
    assert ranker.booster is None
    assert any("only 1 query group" in warning for warning in ranker.warnings)
    assert np.allclose(ranker.score(train), train.X["b_expected_loss_log"].to_numpy())


def test_constant_relevance_inside_every_group_falls_back_too() -> None:
    train, _, _ = split_frames()
    ranker = LambdaMartRanker(fast_config()).fit(
        train, np.zeros(len(train.finding_ids)), seed=SEED
    )
    assert ranker.used_fallback is True
    assert any("constant inside every query group" in warning for warning in ranker.warnings)
    assert np.allclose(ranker.score(train), train.X["b_expected_loss_log"].to_numpy())


def test_the_fallback_uses_the_best_column_the_ablation_cell_still_has() -> None:
    """With Component B off there is no expected loss column, so the chain column is next."""
    train, _, _ = split_frames(ComponentFlags(a=True, b=False, c=True), train_scans=1)
    ranker = LambdaMartRanker(fast_config()).fit(
        train, np.zeros(len(train.finding_ids)), seed=SEED
    )
    assert ranker.used_fallback is True
    assert np.allclose(ranker.score(train), train.X["c_reach_delta_log"].to_numpy())


def test_the_fallback_ranker_still_orders_findings_sensibly() -> None:
    train, _, _ = split_frames(train_scans=1)
    ranker = LambdaMartRanker(fast_config()).fit(
        train, np.zeros(len(train.finding_ids)), seed=SEED
    )
    scores = ranker.score(train)
    losses = train.X["b_expected_loss_log"].to_numpy()
    assert np.argmax(scores) == np.argmax(losses)


def test_a_fallback_ranker_has_no_shap_explanation_to_offer() -> None:
    train, _, _ = split_frames(train_scans=1)
    ranker = LambdaMartRanker(fast_config()).fit(
        train, np.zeros(len(train.finding_ids)), seed=SEED
    )
    assert ranker.explain(train) is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip_the_booster_and_the_column_list(fitted, tmp_path) -> None:
    ranker, _, test, _ = fitted
    path = tmp_path / "models" / "lambdamart.json"

    ranker.save(path)
    restored = LambdaMartRanker.load(path)

    assert restored.columns == ranker.columns
    assert restored.seed == ranker.seed
    assert restored.booster is not None
    assert np.allclose(restored.score(test), ranker.score(test))


def test_a_reloaded_ranker_keeps_its_monotone_constraints(fitted, tmp_path) -> None:
    ranker, _, test, _ = fitted
    path = tmp_path / "lambdamart.json"
    ranker.save(path)
    restored = LambdaMartRanker.load(path)

    without = restored.score(_with_column(test, "b_kev", 0.0))
    with_kev = restored.score(_with_column(test, "b_kev", 1.0))
    assert np.all(with_kev >= without - 1e-6)


def test_a_fallback_ranker_round_trips_as_a_fallback_ranker(tmp_path) -> None:
    train, _, _ = split_frames(train_scans=1)
    ranker = LambdaMartRanker(fast_config()).fit(
        train, np.zeros(len(train.finding_ids)), seed=SEED
    )
    path = tmp_path / "fallback.json"
    ranker.save(path)
    restored = LambdaMartRanker.load(path)

    assert restored.used_fallback is True
    assert restored.booster is None
    assert restored.warnings == ranker.warnings
    assert np.allclose(restored.score(train), ranker.score(train))


def test_loading_without_the_metadata_sidecar_is_an_error(tmp_path) -> None:
    with pytest.raises(VulnPriorityError):
        LambdaMartRanker.load(tmp_path / "absent.json")


# ---------------------------------------------------------------------------
# The cost-sensitive probability head (Gap 7)
# ---------------------------------------------------------------------------


def head_config(**overrides) -> RankingConfig:
    return RankingConfig(n_estimators=80, max_depth=3, learning_rate=0.1, **overrides)


def binary_labels(frame: FeatureFrame, relevance: dict[str, int]) -> np.ndarray:
    return np.array(
        [1 if relevance[finding_id] > 0 else 0 for finding_id in frame.finding_ids], dtype=int
    )


@pytest.fixture(scope="module")
def fitted_head() -> tuple[CostSensitiveExploitHead, FeatureFrame, FeatureFrame, dict[str, int]]:
    train, test, relevance = split_frames()
    head = CostSensitiveExploitHead(head_config()).fit(train, binary_labels(train, relevance), seed=SEED)
    return head, train, test, relevance


def test_the_head_implements_the_frozen_probability_interface(fitted_head) -> None:
    head, _, _, _ = fitted_head
    assert isinstance(head, ProbabilityModel)


def test_the_head_weights_the_minority_class_by_the_label_balance(fitted_head) -> None:
    head, train, _, relevance = fitted_head
    labels = binary_labels(train, relevance)
    positives = int(labels.sum())
    assert head.scale_pos_weight == pytest.approx((labels.size - positives) / positives)


def test_a_configured_scale_pos_weight_overrides_the_computed_one() -> None:
    train, _, relevance = split_frames()
    head = CostSensitiveExploitHead(head_config(head_scale_pos_weight=3.5))
    head.fit(train, binary_labels(train, relevance), seed=SEED)
    assert head.scale_pos_weight == pytest.approx(3.5)


def test_predicted_probabilities_are_one_dimensional_and_strictly_inside_the_unit_interval(
    fitted_head,
) -> None:
    head, _, test, _ = fitted_head
    probabilities = head.predict_proba(test)

    assert probabilities.shape == (len(test.finding_ids),)
    assert probabilities.min() > 0.0
    assert probabilities.max() < 1.0


def test_the_head_separates_exploited_findings_from_the_rest(fitted_head) -> None:
    head, _, test, relevance = fitted_head
    probabilities = head.predict_proba(test)
    labels = binary_labels(test, relevance)

    assert probabilities[labels == 1].mean() > probabilities[labels == 0].mean()


@pytest.mark.parametrize("method", ["isotonic", "sigmoid", "none"])
def test_every_calibration_method_produces_usable_probabilities(method) -> None:
    train, test, relevance = split_frames()
    head = CostSensitiveExploitHead(head_config(head_calibration=method))
    head.fit(train, binary_labels(train, relevance), seed=SEED)
    probabilities = head.predict_proba(test)

    assert head.calibration == method
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0.0) & (probabilities < 1.0)).all()


def test_the_calibration_method_can_be_overridden_per_instance() -> None:
    assert CostSensitiveExploitHead(calibration="sigmoid").calibration == "sigmoid"


def test_an_unknown_calibration_method_is_rejected() -> None:
    with pytest.raises(VulnPriorityError):
        CostSensitiveExploitHead(calibration="magic")


def test_the_head_is_reproducible_under_a_fixed_seed() -> None:
    train, test, relevance = split_frames()
    labels = binary_labels(train, relevance)
    first = CostSensitiveExploitHead(head_config()).fit(train, labels, seed=SEED).predict_proba(test)
    second = CostSensitiveExploitHead(head_config()).fit(train, labels, seed=SEED).predict_proba(test)
    assert np.allclose(first, second)


def test_a_single_class_training_set_yields_the_prior_rather_than_an_exception() -> None:
    """Rare events mean some folds have no positive at all; that is not a crash condition."""
    train, test, _ = split_frames()
    head = CostSensitiveExploitHead(head_config())
    head.fit(train, np.zeros(len(train.finding_ids)), seed=SEED)
    probabilities = head.predict_proba(test)

    assert head.constant_probability is not None
    assert any("single-class" in warning for warning in head.warnings)
    assert len(set(probabilities.tolist())) == 1


def test_graded_relevance_is_accepted_as_binary_ground_truth() -> None:
    train, test, relevance = split_frames()
    graded = np.array([relevance[finding_id] for finding_id in train.finding_ids], dtype=float)
    head = CostSensitiveExploitHead(head_config()).fit(train, graded, seed=SEED)
    assert head.constant_probability is None
    assert np.isfinite(head.predict_proba(test)).all()


def test_predicting_before_fitting_is_an_error() -> None:
    _, test, _ = split_frames()
    with pytest.raises(RankerNotFittedError):
        CostSensitiveExploitHead().predict_proba(test)


def test_predicting_on_a_different_ablation_cell_is_refused(fitted_head) -> None:
    head, _, _, _ = fitted_head
    _, narrow, _ = split_frames(ComponentFlags(a=False, b=True, c=True))
    with pytest.raises(VulnPriorityError):
        head.predict_proba(narrow)


def test_a_label_vector_of_the_wrong_length_is_rejected() -> None:
    train, _, _ = split_frames()
    with pytest.raises(VulnPriorityError):
        CostSensitiveExploitHead(head_config()).fit(train, np.zeros(4))


# ---------------------------------------------------------------------------
# Scoring with a model trained elsewhere
#
# Ranking one application cannot fit a model: one scan is one query group and a pairwise
# objective has no pair to learn from inside it. That is the ordinary operational case, and
# for a long time it silently produced an expected-loss ordering wearing the name
# "lambdamart". These tests pin the behaviour that replaced it.
# ---------------------------------------------------------------------------


def _trained_model(tmp_path) -> Path:
    """Fit on a corpus that can teach a ranking and save it. Returns the path."""
    train, _, relevance = split_frames()
    labels = np.array([relevance[finding_id] for finding_id in train.finding_ids], dtype=float)
    ranker = LambdaMartRanker(fast_config()).fit(train, labels, seed=SEED)
    assert ranker.used_fallback is False, "the fixture corpus must be able to train"
    path = tmp_path / "trained" / "lambdamart.ubj"
    ranker.save(path)
    return path


def test_a_single_scan_is_scored_by_a_model_trained_elsewhere(tmp_path) -> None:
    model_path = _trained_model(tmp_path)
    single, _, _ = split_frames(train_scans=1)

    ranker = LambdaMartRanker(fast_config(model_path=model_path)).fit(
        single, np.zeros(len(single.finding_ids)), seed=SEED
    )

    assert ranker.used_pretrained is True
    assert ranker.used_fallback is False
    assert ranker.booster is not None, "the ordering must come from XGBoost, not a column"
    assert ranker.fallback_reason == ""
    # And it is genuinely the model's opinion, not the expected-loss column under another name.
    assert not np.allclose(ranker.score(single), single.X["b_expected_loss_log"].to_numpy())


def test_a_pretrained_model_can_still_explain_itself(tmp_path) -> None:
    """A booster is a booster: SHAP works whether it was fitted here or loaded."""
    model_path = _trained_model(tmp_path)
    single, _, _ = split_frames(train_scans=1)
    ranker = LambdaMartRanker(fast_config(model_path=model_path)).fit(
        single, np.zeros(len(single.finding_ids)), seed=SEED
    )
    assert ranker.explain(single) is not None


def test_a_model_fitted_for_a_different_cell_is_refused_not_reshaped(tmp_path) -> None:
    """Scoring a frame against a booster trained on other columns is silently wrong.

    XGBoost will happily predict on a matrix with a different feature set, reading column
    three as whatever column three was during training. Refusing is the only safe answer,
    and the refusal says both shapes so the operator knows to re-train.
    """
    model_path = _trained_model(tmp_path)                       # all three components
    ablated, _, _ = split_frames(ComponentFlags(a=True, b=True, c=False), train_scans=1)

    ranker = LambdaMartRanker(fast_config(model_path=model_path)).fit(
        ablated, np.zeros(len(ablated.finding_ids)), seed=SEED
    )

    assert ranker.used_pretrained is False
    assert ranker.used_fallback is True
    assert any("cannot score this frame" in warning for warning in ranker.warnings)


def test_a_missing_model_file_is_not_an_error_just_a_fallback(tmp_path) -> None:
    single, _, _ = split_frames(train_scans=1)
    ranker = LambdaMartRanker(fast_config(model_path=tmp_path / "nothing-here.ubj")).fit(
        single, np.zeros(len(single.finding_ids)), seed=SEED
    )
    assert ranker.used_fallback is True
    assert ranker.used_pretrained is False
    assert ranker.fallback_reason == "only 1 query group in the training frame"


def test_a_corrupt_model_file_does_not_fail_the_run(tmp_path) -> None:
    """A bad cache costs the learned ordering, never the assessment."""
    path = tmp_path / "broken.ubj"
    path.write_bytes(b"not a booster")
    (tmp_path / "broken.ubj.meta.json").write_text('{"has_booster": true}', encoding="utf-8")

    single, _, _ = split_frames(train_scans=1)
    ranker = LambdaMartRanker(fast_config(model_path=path)).fit(
        single, np.zeros(len(single.finding_ids)), seed=SEED
    )
    assert ranker.used_fallback is True
    assert len(ranker.score(single)) == len(single.finding_ids)


def test_training_is_unaffected_by_a_model_already_on_disk(tmp_path) -> None:
    """A corpus that *can* train never reaches for the cache; it fits its own."""
    model_path = _trained_model(tmp_path)
    train, _, relevance = split_frames()
    labels = np.array([relevance[finding_id] for finding_id in train.finding_ids], dtype=float)

    ranker = LambdaMartRanker(fast_config(model_path=model_path)).fit(train, labels, seed=SEED)

    assert ranker.used_pretrained is False
    assert ranker.used_fallback is False
    assert ranker.booster is not None
