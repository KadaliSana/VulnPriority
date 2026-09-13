"""The 2^3 factorial ablation and the uncertainty machinery (Gap 5, Gap 4).

The arithmetic tests are the load-bearing ones: main effects and interactions are
computed on synthetic tables whose answers are known in closed form, so a sign error or a
wrong divisor cannot survive.

The end-to-end test uses a deliberately simple linear ranker defined here rather than
importing ``vulnprio.rank``: the ablation measures what the *components* contribute, and
a transparent learner makes it possible to assert that the measured effects match the
signal that was planted in the data.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from vulnprio.core.config import EvaluationConfig, PipelineConfig, RankingConfig
from vulnprio.core.enums import RankerName, SplitKind
from vulnprio.core.interfaces import Ranker
from vulnprio.core.models import (
    AblationTable,
    ComponentFlags,
    FeatureFrame,
    GroundTruthLabel,
    LabelSet,
    Split,
    feature_names_for,
)
from vulnprio.eval.ablation import (
    COMPONENT_KEYS,
    INTERACTION_KEYS,
    FullFactorialAblation,
    cell_means,
    contrast,
    interactions,
    main_effects,
)
from vulnprio.eval.bootstrap import (
    bootstrap_mean_ci,
    paired_bootstrap,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)

CELLS = ComponentFlags.all_cells()


def table(fn) -> dict[ComponentFlags, dict[str, float]]:
    """A synthetic cell table from a closed-form response surface."""
    return {
        flags: {"ndcg@10": fn(int(flags.a), int(flags.b), int(flags.c))} for flags in CELLS
    }


# ---------------------------------------------------------------------------
# Contrast coding
# ---------------------------------------------------------------------------


def test_contrast_coding_is_plus_or_minus_one_and_multiplies_for_interactions() -> None:
    on = ComponentFlags(a=True, b=False, c=True)
    assert contrast(on, "A") == 1.0
    assert contrast(on, "B") == -1.0
    assert contrast(on, "AB") == -1.0
    assert contrast(on, "AC") == 1.0
    assert contrast(on, "ABC") == -1.0


def test_the_design_has_eight_cells_balanced_on_every_factor() -> None:
    assert len(CELLS) == 8
    for key in COMPONENT_KEYS:
        assert sum(1 for flags in CELLS if contrast(flags, key) > 0) == 4


# ---------------------------------------------------------------------------
# Main effects: known answers
# ---------------------------------------------------------------------------


def test_main_effects_on_a_purely_additive_table() -> None:
    """y = 0.5 + 0.10a + 0.04b - 0.02c: the effects are the coefficients themselves."""
    means = table(lambda a, b, c: 0.5 + 0.10 * a + 0.04 * b - 0.02 * c)
    effects = main_effects(means)
    assert effects["A"]["ndcg@10"] == pytest.approx(0.10)
    assert effects["B"]["ndcg@10"] == pytest.approx(0.04)
    assert effects["C"]["ndcg@10"] == pytest.approx(-0.02)
    for key in INTERACTION_KEYS:
        assert interactions(means)[key]["ndcg@10"] == pytest.approx(0.0, abs=1e-12)


def test_a_main_effect_is_literally_the_mean_on_minus_the_mean_off() -> None:
    """The definition the report prints, checked against the contrast implementation."""
    rng = np.random.default_rng(4)
    means = {flags: {"ndcg@10": float(rng.random())} for flags in CELLS}
    effects = main_effects(means)
    for key in COMPONENT_KEYS:
        on = [values["ndcg@10"] for flags, values in means.items() if contrast(flags, key) > 0]
        off = [values["ndcg@10"] for flags, values in means.items() if contrast(flags, key) < 0]
        assert effects[key]["ndcg@10"] == pytest.approx(np.mean(on) - np.mean(off))


def test_main_effects_with_a_two_way_interaction_present() -> None:
    """y = 0.5 + 0.10a + 0.04b + 0.06ab.

    Averaged over b, A is worth 0.10 + 0.06/2 = 0.13 and B is worth 0.04 + 0.06/2 = 0.07.
    The AB interaction is half the coefficient, 0.03, by the +/-1 contrast identity.
    """
    means = table(lambda a, b, c: 0.5 + 0.10 * a + 0.04 * b + 0.06 * a * b)
    effects = main_effects(means)
    assert effects["A"]["ndcg@10"] == pytest.approx(0.13)
    assert effects["B"]["ndcg@10"] == pytest.approx(0.07)
    assert effects["C"]["ndcg@10"] == pytest.approx(0.0, abs=1e-12)

    combined = interactions(means)
    assert combined["AB"]["ndcg@10"] == pytest.approx(0.03)
    assert combined["AC"]["ndcg@10"] == pytest.approx(0.0, abs=1e-12)
    assert combined["BC"]["ndcg@10"] == pytest.approx(0.0, abs=1e-12)
    assert combined["ABC"]["ndcg@10"] == pytest.approx(0.0, abs=1e-12)


def test_the_two_way_interaction_equals_half_the_difference_of_conditional_effects() -> None:
    """AB = 1/2 * [ (effect of A given B on) - (effect of A given B off) ]."""
    means = table(lambda a, b, c: 0.5 + 0.10 * a + 0.04 * b + 0.06 * a * b - 0.03 * c)

    def mean_over_c(a: bool, b: bool) -> float:
        return float(
            np.mean(
                [
                    values["ndcg@10"]
                    for flags, values in means.items()
                    if flags.a is a and flags.b is b
                ]
            )
        )

    effect_given_b_on = mean_over_c(True, True) - mean_over_c(False, True)
    effect_given_b_off = mean_over_c(True, False) - mean_over_c(False, False)
    assert interactions(means)["AB"]["ndcg@10"] == pytest.approx(
        0.5 * (effect_given_b_on - effect_given_b_off)
    )


def test_a_pure_three_way_interaction_is_detected() -> None:
    """y = 0.4abc: only the ABC contrast should be large, and it equals 0.4/4."""
    means = table(lambda a, b, c: 0.4 * a * b * c)
    combined = interactions(means)
    assert combined["ABC"]["ndcg@10"] == pytest.approx(0.1)
    assert main_effects(means)["A"]["ndcg@10"] == pytest.approx(0.1)


def test_components_that_substitute_for_each_other_show_a_negative_interaction() -> None:
    """Each is worth 0.1 alone but only 0.12 together: they overlap."""
    means = table(lambda a, b, c: 0.5 + 0.10 * a + 0.10 * b - 0.08 * a * b)
    assert interactions(means)["AB"]["ndcg@10"] < 0.0
    assert interactions(means)["AB"]["ndcg@10"] == pytest.approx(-0.04)


def test_cell_means_average_over_seeds() -> None:
    per_cell = {flags: [{"ndcg@10": 0.5}, {"ndcg@10": 0.7}] for flags in CELLS}
    assert cell_means(per_cell)[CELLS[0]]["ndcg@10"] == pytest.approx(0.6)


def test_effects_over_an_empty_table_do_not_raise() -> None:
    assert main_effects({}) == {key: {} for key in COMPONENT_KEYS}
    assert interactions({}) == {key: {} for key in INTERACTION_KEYS}


# ---------------------------------------------------------------------------
# Paired bootstrap and Wilcoxon
# ---------------------------------------------------------------------------


def test_paired_bootstrap_interval_brackets_the_observed_difference() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(0.6, 0.05, 40)
    better = base + 0.05
    result = paired_bootstrap(better, base, iters=1000, seed=3)
    assert result.observed_diff == pytest.approx(0.05, abs=1e-9)
    assert result.ci_low <= result.observed_diff <= result.ci_high
    assert result.significant is True
    assert result.prob_a_better == pytest.approx(1.0)


def test_paired_bootstrap_reports_no_difference_as_no_difference() -> None:
    rng = np.random.default_rng(8)
    values = rng.normal(0.5, 0.1, 60)
    noisy = values + rng.normal(0.0, 0.1, 60)
    result = paired_bootstrap(values, noisy, iters=1000, seed=5)
    assert result.ci_low < 0.0 < result.ci_high
    assert result.significant is False


def test_pairing_is_what_makes_the_interval_informative() -> None:
    """Between-scan variance is huge; the paired difference is tiny and consistent."""
    rng = np.random.default_rng(13)
    difficulty = rng.normal(0.5, 0.25, 50)          # some scans are simply harder
    left = difficulty + 0.02
    right = difficulty
    paired = paired_bootstrap(left, right, iters=1000, seed=1)
    unpaired_width = np.subtract(*reversed(bootstrap_mean_ci(left, iters=1000, seed=1)))
    assert paired.significant is True
    assert (paired.ci_high - paired.ci_low) < unpaired_width


def test_paired_bootstrap_ci_is_seeded_and_reproducible() -> None:
    left = [0.1, 0.4, 0.5, 0.9, 0.3]
    right = [0.0, 0.2, 0.6, 0.7, 0.1]
    assert paired_bootstrap_ci(left, right, 200, 42) == paired_bootstrap_ci(left, right, 200, 42)
    assert paired_bootstrap_ci(left, right, 200, 42) != paired_bootstrap_ci(left, right, 200, 7)


def test_paired_bootstrap_survives_degenerate_input() -> None:
    assert paired_bootstrap_ci([], [], 100, 1) == (0.0, 0.0)
    assert paired_bootstrap_ci([0.5], [0.2], 100, 1) == pytest.approx((0.3, 0.3))
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1, 0.2], [0.1], 100, 1)


def test_wilcoxon_detects_a_consistent_shift_and_ignores_its_size() -> None:
    left = [0.51, 0.62, 0.43, 0.74, 0.55, 0.66, 0.37, 0.58, 0.49, 0.61, 0.53, 0.72]
    right = [value - 0.03 for value in left]
    result = wilcoxon_signed_rank(left, right)
    assert result.n_nonzero == len(left)
    assert result.statistic == 0.0                  # every difference has the same sign
    assert result.p_value < 0.01


def test_wilcoxon_on_identical_rankers_reports_no_evidence() -> None:
    values = [0.4, 0.5, 0.6, 0.7]
    result = wilcoxon_signed_rank(values, values)
    assert result.n_nonzero == 0
    assert result.p_value == 1.0                    # the case scipy refuses outright


def test_wilcoxon_matches_scipys_normal_approximation() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(6)
    left = rng.normal(0.6, 0.1, 30)
    right = left - rng.normal(0.02, 0.05, 30)
    mine = wilcoxon_signed_rank(left, right)
    theirs = scipy_stats.wilcoxon(left, right, method="approx", correction=True)
    assert mine.statistic == pytest.approx(float(theirs.statistic))
    assert mine.p_value == pytest.approx(float(theirs.pvalue), rel=1e-6)


# ---------------------------------------------------------------------------
# End to end over the eight cells
# ---------------------------------------------------------------------------


class LinearProbeRanker(Ranker):
    """A transparent stand-in for the learned ranker, defined here rather than imported.

    ``fit`` computes the per-column mean difference between positive and negative rows
    and ``score`` projects onto it - a one-step linear discriminant. It is deliberately
    simple, so that when the ablation reports "Component B is worth more than Component
    C", the claim can be traced to the signal planted in those columns rather than to a
    learner's idiosyncrasies.
    """

    name = RankerName.LAMBDAMART

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.columns: list[str] = []

    def fit(self, frame, relevance, sample_weight=None, seed: int = 42):
        matrix = frame.to_numpy()
        target = np.asarray(relevance, dtype=float) > 0
        self.columns = frame.feature_names
        if target.any() and (~target).any():
            self.weights = matrix[target].mean(axis=0) - matrix[~target].mean(axis=0)
        else:
            self.weights = np.zeros(matrix.shape[1])
        return self

    def score(self, frame) -> np.ndarray:
        if self.weights is None:
            raise AssertionError("score before fit")
        return frame.to_numpy() @ self.weights


def build_dataset(n_scans: int = 6, per_scan: int = 12, seed: int = 0):
    """Findings whose component columns carry planted, unequal amounts of label signal."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    labels: list[GroundTruthLabel] = []
    for scan_index in range(n_scans):
        for row_index in range(per_scan):
            finding_id = f"f_{scan_index}_{row_index}"
            exploited = row_index < 3                       # 25% positive
            rows.append(
                {
                    "finding_id": finding_id,
                    "scan_id": f"scan_{scan_index}",
                    "signal": 1.0 if exploited else 0.0,
                    "noise": rng.normal(0.0, 1.0),
                }
            )
            labels.append(
                GroundTruthLabel(
                    finding_id=finding_id,
                    exploited=exploited,
                    relevance_grade=4 if exploited else 0,
                )
            )
    label_set = LabelSet(observation_cutoff=date(2024, 6, 1), labels=tuple(labels))
    return rows, label_set


#: How much of the label each component's columns carry. B is the strongest, C is noise.
COMPONENT_SIGNAL = {"a_": 0.35, "b_": 0.90, "c_": 0.0, "": 0.05}


def make_frame(rows, flags: ComponentFlags, seed: int) -> FeatureFrame:
    rng = np.random.default_rng(1000 + seed)
    columns = feature_names_for(flags)
    data = {}
    for column in columns:
        prefix = column[:2] if column[:2] in COMPONENT_SIGNAL else ""
        strength = COMPONENT_SIGNAL[prefix]
        data[column] = [
            strength * row["signal"] + rng.normal(0.0, 1.0) for row in rows
        ]
    return FeatureFrame(
        X=pd.DataFrame(data, columns=columns),
        finding_ids=[row["finding_id"] for row in rows],
        group_ids=[row["scan_id"] for row in rows],
        flags=flags,
    )


@pytest.fixture
def ablation_config() -> PipelineConfig:
    return PipelineConfig(
        evaluation=EvaluationConfig(
            k_values=(5, 10), seeds=(1, 2), bootstrap_iters=64, n_folds=1
        ),
        ranking=RankingConfig(impact_weighted_pairs=False),
    )


@pytest.fixture
def dataset():
    return build_dataset()


def test_full_factorial_ablation_runs_every_cell_and_seed(dataset, ablation_config) -> None:
    rows, labels = dataset
    split = Split(
        kind=SplitKind.TIME_ORDERED,
        fold=0,
        train_scan_ids=("scan_0", "scan_1", "scan_2", "scan_3"),
        test_scan_ids=("scan_4", "scan_5"),
        train_end=date(2024, 3, 1),
        test_start=date(2024, 4, 1),
        gap_days=30,
    )
    train_rows = [row for row in rows if row["scan_id"] in split.train_scan_ids]
    test_rows = [row for row in rows if row["scan_id"] in split.test_scan_ids]

    def build_frame_fn(flags: ComponentFlags, fold: Split, seed: int):
        return (make_frame(train_rows, flags, seed), make_frame(test_rows, flags, seed))

    ablation = FullFactorialAblation(
        metrics=["ndcg@10", "recall@10"],
        ranker_factory=lambda flags, seed: LinearProbeRanker(),
    )
    result = ablation.run(build_frame_fn, labels, [split], ablation_config)

    assert isinstance(result, AblationTable)
    assert len(result.cells) == 8
    assert {cell.flags for cell in result.cells} == set(ComponentFlags.all_cells())
    for cell in result.cells:
        assert cell.n == 2 and cell.seeds == (1, 2)
        assert "ndcg@10" in cell.mean and "ndcg@10" in cell.std
        assert 0.0 <= cell.mean["ndcg@10"] <= 1.0

    assert set(result.main_effects) == set(COMPONENT_KEYS)
    assert set(result.interactions) == set(INTERACTION_KEYS)
    assert set(result.paired_ci) == set(COMPONENT_KEYS)
    for component in COMPONENT_KEYS:
        low, high = result.paired_ci[component]["ndcg@10"]
        assert low <= high


def test_the_ablation_recovers_the_planted_ordering_of_the_components(
    dataset, ablation_config
) -> None:
    """Component B's columns carry most of the label; C's carry none. The table says so."""
    rows, labels = dataset
    split = Split(
        kind=SplitKind.TIME_ORDERED,
        fold=0,
        train_scan_ids=("scan_0", "scan_1", "scan_2", "scan_3"),
        test_scan_ids=("scan_4", "scan_5"),
        train_end=date(2024, 3, 1),
        test_start=date(2024, 4, 1),
        gap_days=30,
    )
    train_rows = [row for row in rows if row["scan_id"] in split.train_scan_ids]
    test_rows = [row for row in rows if row["scan_id"] in split.test_scan_ids]

    def build_frame_fn(flags: ComponentFlags, fold: Split, seed: int):
        return (make_frame(train_rows, flags, seed), make_frame(test_rows, flags, seed))

    result = FullFactorialAblation(
        metrics=["ndcg@10"], ranker_factory=lambda flags, seed: LinearProbeRanker()
    ).run(build_frame_fn, labels, [split], ablation_config)

    effects = {key: result.main_effects[key]["ndcg@10"] for key in COMPONENT_KEYS}
    assert effects["B"] > effects["C"]
    assert effects["B"] > 0.0

    all_on = result.cell(ComponentFlags(a=True, b=True, c=True))
    all_off = result.cell(ComponentFlags(a=False, b=False, c=False))
    assert all_on is not None and all_off is not None
    assert all_on.mean["ndcg@10"] > all_off.mean["ndcg@10"]


def test_disabled_components_cannot_leak_through_the_feature_matrix() -> None:
    """The structural guarantee: an off cell has no column for that component at all."""
    rows, _ = build_dataset(n_scans=1, per_scan=4)
    frame = make_frame(rows, ComponentFlags(a=True, b=False, c=True), seed=1)
    assert not any(column.startswith("b_") for column in frame.feature_names)
    assert any(column.startswith("a_") for column in frame.feature_names)
    assert frame.feature_names == feature_names_for(ComponentFlags(a=True, b=False, c=True))

    with pytest.raises(ValueError):
        FeatureFrame(
            X=frame.X,
            finding_ids=frame.finding_ids,
            group_ids=frame.group_ids,
            flags=ComponentFlags(a=True, b=True, c=True),
        )


def test_ablation_retains_every_run_for_the_paired_bootstrap(dataset, ablation_config) -> None:
    rows, labels = dataset
    split = Split(
        kind=SplitKind.TIME_ORDERED,
        fold=0,
        train_scan_ids=("scan_0", "scan_1", "scan_2"),
        test_scan_ids=("scan_4", "scan_5"),
        train_end=date(2024, 3, 1),
        test_start=date(2024, 4, 1),
        gap_days=30,
    )
    rows_by_scan = {row["scan_id"]: None for row in rows}

    def build_frame_fn(flags: ComponentFlags, fold: Split, seed: int):
        train_rows = [row for row in rows if row["scan_id"] in fold.train_scan_ids]
        test_rows = [row for row in rows if row["scan_id"] in fold.test_scan_ids]
        return (make_frame(train_rows, flags, seed), make_frame(test_rows, flags, seed))

    ablation = FullFactorialAblation(
        metrics=["ndcg@10"], ranker_factory=lambda flags, seed: LinearProbeRanker()
    )
    ablation.run(build_frame_fn, labels, [split], ablation_config)

    assert len(ablation.runs) == 8 * 2                       # cells x seeds
    assert len(ablation.bundles) == 8 * 2                    # one fold each
    assert all(len(value) == 1 for value in ablation.runs.values())
    assert ("ABC", 1) in ablation.runs and ("none", 2) in ablation.runs
    assert rows_by_scan  # dataset sanity


def test_the_ablation_is_deterministic_across_repeated_runs(dataset, ablation_config) -> None:
    rows, labels = dataset
    split = Split(
        kind=SplitKind.TIME_ORDERED,
        fold=0,
        train_scan_ids=("scan_0", "scan_1", "scan_2", "scan_3"),
        test_scan_ids=("scan_4", "scan_5"),
        train_end=date(2024, 3, 1),
        test_start=date(2024, 4, 1),
        gap_days=30,
    )

    def build_frame_fn(flags: ComponentFlags, fold: Split, seed: int):
        train_rows = [row for row in rows if row["scan_id"] in fold.train_scan_ids]
        test_rows = [row for row in rows if row["scan_id"] in fold.test_scan_ids]
        return (make_frame(train_rows, flags, seed), make_frame(test_rows, flags, seed))

    def run_once() -> AblationTable:
        return FullFactorialAblation(
            metrics=["ndcg@10"], ranker_factory=lambda flags, seed: LinearProbeRanker()
        ).run(build_frame_fn, labels, [split], ablation_config)

    first, second = run_once(), run_once()
    assert first.main_effects == second.main_effects
    assert first.interactions == second.interactions
    assert first.paired_ci == second.paired_ci


# ---------------------------------------------------------------------------
# BenchmarkRunner: the "identical data" guarantee (Gap 4)
# ---------------------------------------------------------------------------


class RecordingRanker(Ranker):
    """A ranker that records exactly which rows it was shown, and scores by one column."""

    def __init__(self, name: RankerName, column: str | None = None) -> None:
        self.name = name
        self.column = column
        self.seen: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def fit(self, frame, relevance, sample_weight=None, seed: int = 42):
        self.seen.append(("fit", tuple(frame.finding_ids), tuple(frame.feature_names)))
        self.relevance = np.asarray(relevance, dtype=float)
        self.sample_weight = sample_weight
        return self

    def score(self, frame) -> np.ndarray:
        self.seen.append(("score", tuple(frame.finding_ids), tuple(frame.feature_names)))
        if self.column and self.column in frame.feature_names:
            return frame.X[self.column].to_numpy(dtype=float)
        return np.zeros(len(frame.finding_ids))


class ReversingRanker(Ranker):
    """The worst honest ordering: strictly the reverse of the frame's own row order.

    The planted datasets list the exploited findings first, so reversing puts every one of
    them at the bottom of the queue. It is the floor the oracle is measured against.
    """

    name = RankerName.RANDOM

    def fit(self, frame, relevance, sample_weight=None, seed: int = 42):
        return self

    def score(self, frame) -> np.ndarray:
        return np.arange(len(frame.finding_ids), dtype=float)


class OracleRanker(Ranker):
    """Ranks the exploited findings first. The ceiling every other ranker is measured against."""

    name = RankerName.EXPECTED_LOSS

    def __init__(self, positives: set[str]) -> None:
        self.positives = positives

    def fit(self, frame, relevance, sample_weight=None, seed: int = 42):
        return self

    def score(self, frame) -> np.ndarray:
        return np.asarray(
            [1.0 if finding_id in self.positives else 0.0 for finding_id in frame.finding_ids],
            dtype=float,
        )


@pytest.fixture
def benchmark_fold(dataset):
    from vulnprio.eval.benchmark import SplitFrames

    rows, labels = dataset
    split = Split(
        kind=SplitKind.TIME_ORDERED,
        fold=0,
        train_scan_ids=("scan_0", "scan_1", "scan_2", "scan_3"),
        test_scan_ids=("scan_4", "scan_5"),
        train_end=date(2024, 3, 1),
        test_start=date(2024, 4, 1),
        gap_days=30,
    )
    flags = ComponentFlags()
    train = make_frame([row for row in rows if row["scan_id"] in split.train_scan_ids], flags, 1)
    test = make_frame([row for row in rows if row["scan_id"] in split.test_scan_ids], flags, 1)
    return SplitFrames(split=split, train=train, test=test), labels


def test_benchmark_shows_every_ranker_byte_identical_data(benchmark_fold, ablation_config) -> None:
    """Gap 4's core claim: no ranker gets different rows, columns or preprocessing."""
    from vulnprio.eval.benchmark import BenchmarkRunner

    fold, labels = benchmark_fold
    first = RecordingRanker(RankerName.LAMBDAMART, "b_epss")
    second = RecordingRanker(RankerName.CVSS_ONLY, "cvss_base_max")
    BenchmarkRunner().run([fold], labels, [first, second], ablation_config)

    assert first.seen == second.seen
    assert [event for event, _, _ in first.seen] == ["fit", "score"]
    assert first.seen[0][1] == tuple(fold.train.finding_ids)
    assert first.seen[1][1] == tuple(fold.test.finding_ids)
    assert np.array_equal(first.relevance, second.relevance)


def test_benchmark_produces_one_bundle_per_ranker_with_the_full_metric_battery(
    benchmark_fold, ablation_config
) -> None:
    from vulnprio.core.enums import MetricName
    from vulnprio.eval.benchmark import BenchmarkRunner

    fold, labels = benchmark_fold
    bundles = BenchmarkRunner().run(
        [fold],
        labels,
        [RecordingRanker(RankerName.LAMBDAMART, "b_epss"), OracleRanker(labels.positives())],
        ablation_config,
    )
    assert len(bundles) == 2
    assert {bundle.ranker for bundle in bundles} == {
        RankerName.LAMBDAMART,
        RankerName.EXPECTED_LOSS,
    }
    keys = bundles[0].as_dict()
    for expected in ("ndcg@5", "ndcg@10", "precision@10", "recall@10", "risk_capture@10"):
        assert expected in keys
    for name in (
        MetricName.MAP,
        MetricName.MRR,
        MetricName.ROC_AUC,
        MetricName.PR_AUC,
        MetricName.MCC,
        MetricName.F1_MINORITY,
        MetricName.BALANCED_ACCURACY,
        MetricName.EFFICIENCY,
        MetricName.COVERAGE,
        MetricName.WORKLOAD_REDUCTION,
    ):
        assert bundles[0].get(name) is not None
    # no calibrated probabilities were supplied, so no calibration is claimed
    assert all(bundle.calibration is None for bundle in bundles)
    assert all(bundle.minority is not None for bundle in bundles)
    assert all(bundle.split == fold.split for bundle in bundles)


def test_the_oracle_ranker_beats_a_blind_one_on_every_ranking_metric(
    benchmark_fold, ablation_config
) -> None:
    from vulnprio.core.enums import MetricName
    from vulnprio.eval.benchmark import BenchmarkRunner

    fold, labels = benchmark_fold
    bundles = BenchmarkRunner().run(
        [fold],
        labels,
        [OracleRanker(labels.positives()), ReversingRanker()],
        ablation_config,
    )
    oracle = next(bundle for bundle in bundles if bundle.ranker == RankerName.EXPECTED_LOSS)
    blind = next(bundle for bundle in bundles if bundle.ranker == RankerName.RANDOM)

    assert oracle.get(MetricName.NDCG_AT_K, 5) == pytest.approx(1.0)
    assert oracle.get(MetricName.NDCG_AT_K, 5) > blind.get(MetricName.NDCG_AT_K, 5)
    assert oracle.get(MetricName.ROC_AUC) == pytest.approx(1.0)
    assert oracle.get(MetricName.MEAN_RANK_OF_EXPLOITED) < blind.get(
        MetricName.MEAN_RANK_OF_EXPLOITED
    )


def test_benchmark_retains_per_scan_values_for_the_paired_bootstrap(
    benchmark_fold, ablation_config
) -> None:
    """"Better on 2 of 2 scans" needs the pairs, not the fold mean."""
    from vulnprio.eval.benchmark import BenchmarkRunner

    fold, labels = benchmark_fold
    runner = BenchmarkRunner()
    runner.run(
        [fold],
        labels,
        [OracleRanker(labels.positives()), ReversingRanker()],
        ablation_config,
    )
    per_scan = runner.per_scan(RankerName.EXPECTED_LOSS, "ndcg@10")
    assert len(per_scan) == len(fold.split.test_scan_ids)

    left, right = runner.paired_values(RankerName.EXPECTED_LOSS, RankerName.RANDOM, "ndcg@10")
    assert len(left) == len(right) == len(fold.split.test_scan_ids)
    assert all(a >= b for a, b in zip(left, right))

    result = paired_bootstrap(left, right, iters=200, seed=1)
    assert result.observed_diff > 0.0


def test_a_ranker_that_ties_everything_still_gets_a_deterministic_order(
    benchmark_fold, ablation_config
) -> None:
    """Baselines tie heavily by design; the tie-break must not favour anyone."""
    from vulnprio.eval.benchmark import BenchmarkRunner, scan_order

    fold, labels = benchmark_fold
    scores = np.zeros(len(fold.test.finding_ids))
    scan_id = fold.test.group_ids[0]
    assert scan_order(fold.test, scores, scan_id) == [
        finding_id
        for finding_id, group in zip(fold.test.finding_ids, fold.test.group_ids)
        if group == scan_id
    ]

    bundles = BenchmarkRunner().run(
        [fold],
        labels,
        [RecordingRanker(RankerName.KEV_FIRST), RecordingRanker(RankerName.SCANNER_SEVERITY)],
        ablation_config,
    )
    assert bundles[0].as_dict() == bundles[1].as_dict()


def test_supplied_probabilities_are_what_unlock_calibration_metrics(
    benchmark_fold, ablation_config
) -> None:
    from vulnprio.eval.benchmark import BenchmarkRunner

    fold, labels = benchmark_fold
    positives = labels.positives()
    probabilities = {
        RankerName.EXPECTED_LOSS: {
            finding_id: 0.9 if finding_id in positives else 0.05
            for finding_id in fold.test.finding_ids
        }
    }
    bundle = BenchmarkRunner(probabilities=probabilities).run(
        [fold], labels, [OracleRanker(positives)], ablation_config
    )[0]
    assert bundle.calibration is not None
    assert bundle.calibration.n_bins == ablation_config.evaluation.calibration_bins
    assert sum(bundle.calibration.bin_count) == len(fold.test.finding_ids)
    assert bundle.minority is not None and bundle.minority.mcc == pytest.approx(1.0)


def test_expected_loss_is_recovered_from_the_feature_matrix_when_not_supplied(
    benchmark_fold, ablation_config
) -> None:
    """``b_expected_loss_log`` is log1p of the loss, so risk capture works without a side channel."""
    from vulnprio.core.enums import MetricName
    from vulnprio.eval.benchmark import BenchmarkRunner

    fold, labels = benchmark_fold
    assert "b_expected_loss_log" in fold.test.feature_names
    bundle = BenchmarkRunner().run(
        [fold], labels, [OracleRanker(labels.positives())], ablation_config
    )[0]
    assert 0.0 <= bundle.get(MetricName.RISK_CAPTURE_AT_K, 10) <= 1.0

    supplied = BenchmarkRunner(
        expected_loss={finding_id: 1.0 for finding_id in fold.test.finding_ids}
    ).run([fold], labels, [OracleRanker(labels.positives())], ablation_config)[0]
    # with flat losses risk capture is just the share of the queue inspected: there is no
    # monetary estimate left to capture, which is exactly the B-disabled ablation cell
    n_rows = len(fold.test.finding_ids) // len(fold.split.test_scan_ids)
    assert supplied.get(MetricName.RISK_CAPTURE_AT_K, 10) == pytest.approx(10 / n_rows)
