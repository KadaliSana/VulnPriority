"""``ReportBuilder``: the artefact a reviewer reads (DESIGN.md 3.9 and 4, Gap 4).

The report is where the protocol becomes checkable by someone who did not run it, so
these tests assert that every section and every figure is produced, that ``report.json``
carries every number the prose quotes, and that a partial run (metrics only, no ablation)
still yields a readable report rather than an exception.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib
import pytest

from vulnpriority.core.enums import (
    FeedMode,
    InjectionCategory,
    LLMBackendKind,
    MetricName,
    RankerName,
    SelectionMethod,
    SplitKind,
)
from vulnpriority.core.models import (
    AblationCell,
    AblationTable,
    AdversarialReport,
    CalibrationReport,
    ComponentFlags,
    MetricBundle,
    MetricValue,
    MinorityClassReport,
    RunManifest,
    SelectionResult,
    SimulationResult,
    Split,
)
from vulnpriority.eval.report import ReportArtifacts, ReportBuilder

SPLIT = Split(
    kind=SplitKind.TIME_ORDERED,
    fold=0,
    train_scan_ids=("scan_0", "scan_1", "scan_2"),
    test_scan_ids=("scan_3", "scan_4"),
    train_end=date(2024, 4, 1),
    test_start=date(2024, 5, 1),
    gap_days=30,
)


def bundle(ranker: RankerName, ndcg: float, calibrated: bool = False) -> MetricBundle:
    values = [
        MetricValue(name=MetricName.NDCG_AT_K, k=5, value=ndcg - 0.02, ci_low=ndcg - 0.08, ci_high=ndcg + 0.04, n=2),
        MetricValue(name=MetricName.NDCG_AT_K, k=10, value=ndcg, ci_low=ndcg - 0.05, ci_high=ndcg + 0.05, n=2),
        MetricValue(name=MetricName.RECALL_AT_K, k=10, value=ndcg * 0.9, ci_low=0.1, ci_high=0.9, n=2),
        MetricValue(name=MetricName.RISK_CAPTURE_AT_K, k=10, value=ndcg * 0.95, n=2),
        MetricValue(name=MetricName.MAP, value=ndcg * 0.8, n=2),
        MetricValue(name=MetricName.MRR, value=ndcg * 1.05 if ndcg < 0.9 else 0.95, n=2),
        MetricValue(name=MetricName.MEAN_RANK_OF_EXPLOITED, value=4.5, n=2),
        MetricValue(name=MetricName.KENDALL_TAU_VS_CVSS, value=0.21, n=2),
        MetricValue(name=MetricName.ROC_AUC, value=0.78, n=24),
        MetricValue(name=MetricName.PR_AUC, value=0.42, n=24),
        MetricValue(name=MetricName.MCC, value=0.33, n=24),
        MetricValue(name=MetricName.F1_MINORITY, value=0.39, n=24),
        MetricValue(name=MetricName.BALANCED_ACCURACY, value=0.66, n=24),
        MetricValue(name=MetricName.EFFICIENCY, value=0.25, n=2),
        MetricValue(name=MetricName.COVERAGE, value=0.80, n=2),
        MetricValue(name=MetricName.WORKLOAD_REDUCTION, value=0.90, n=2),
    ]
    return MetricBundle(
        ranker=ranker,
        flags=ComponentFlags(),
        split=SPLIT,
        seed=42,
        values=tuple(values),
        calibration=(
            CalibrationReport(
                brier=0.031,
                ece=0.042,
                n_bins=4,
                bin_confidence=(0.05, 0.3, 0.55, 0.9),
                bin_accuracy=(0.02, 0.35, 0.5, 0.8),
                bin_count=(120, 40, 12, 6),
            )
            if calibrated
            else None
        ),
        minority=MinorityClassReport(
            positive_rate=0.08,
            mcc=0.33,
            f1_positive=0.39,
            balanced_accuracy=0.66,
            threshold=0.5,
            per_class={
                "exploited": {"precision": 0.4, "recall": 0.38, "f1": 0.39, "support": 12.0},
                "not_exploited": {"precision": 0.95, "recall": 0.96, "f1": 0.95, "support": 138.0},
            },
        ),
        runtime_seconds=1.25,
    )


@pytest.fixture
def bundles() -> list[MetricBundle]:
    return [
        bundle(RankerName.LAMBDAMART, 0.71, calibrated=True),
        bundle(RankerName.EXPECTED_LOSS, 0.66),
        bundle(RankerName.CVSS_ONLY, 0.48),
        bundle(RankerName.RANDOM, 0.22),
    ]


@pytest.fixture
def ablation() -> AblationTable:
    cells = tuple(
        AblationCell(
            flags=flags,
            seeds=(42, 43),
            mean={"ndcg@10": 0.5 + 0.1 * flags.a + 0.05 * flags.b + 0.02 * flags.c},
            std={"ndcg@10": 0.01},
            n=2,
        )
        for flags in ComponentFlags.all_cells()
    )
    return AblationTable(
        cells=cells,
        main_effects={"A": {"ndcg@10": 0.10}, "B": {"ndcg@10": 0.05}, "C": {"ndcg@10": 0.02}},
        interactions={
            "AB": {"ndcg@10": 0.01},
            "AC": {"ndcg@10": 0.0},
            "BC": {"ndcg@10": -0.01},
            "ABC": {"ndcg@10": 0.0},
        },
        paired_ci={
            "A": {"ndcg@10": (0.06, 0.14)},
            "B": {"ndcg@10": (0.01, 0.09)},
            "C": {"ndcg@10": (-0.01, 0.05)},
        },
    )


@pytest.fixture
def simulations() -> list[SimulationResult]:
    return [
        SimulationResult(
            policy=RankerName.LAMBDAMART,
            weeks=4,
            capacity_hours_per_week=20.0,
            exposure_days_total=84.0,
            exposure_days_exploited=14.0,
            expected_loss_days=1_409_100.0,
            exploited_remediated_before_exploit=2,
            exploited_total=2,
            weekly_cumulative_exposure=(56.0, 84.0, 84.0, 84.0),
            reduction_vs_cvss=0.3333,
        ),
        SimulationResult(
            policy=RankerName.CVSS_ONLY,
            weeks=4,
            capacity_hours_per_week=20.0,
            exposure_days_total=126.0,
            exposure_days_exploited=28.0,
            expected_loss_days=2_807_700.0,
            exploited_remediated_before_exploit=0,
            exploited_total=2,
            weekly_cumulative_exposure=(14.0, 126.0, 126.0, 126.0),
            reduction_vs_cvss=0.0,
        ),
    ]


@pytest.fixture
def selections() -> list[SelectionResult]:
    return [
        SelectionResult(
            scan_id="scan_3",
            ranker=RankerName.LAMBDAMART,
            method=SelectionMethod.DP_EXACT,
            budget_hours=40.0,
            selected_ids=("f1", "f2", "f3"),
            total_hours=38.5,
            risk_captured=920_000.0,
            risk_capture_fraction=0.74,
            exploited_captured=3,
            exploited_total=4,
        )
    ]


@pytest.fixture
def adversarial() -> AdversarialReport:
    return AdversarialReport(
        backend=LLMBackendKind.HEURISTIC,
        n_cases=80,
        attack_success_rate=0.0,
        canary_leak_rate=0.0,
        detection_rate=0.92,
        false_positive_rate=0.05,
        mean_abs_rank_shift=0.4,
        max_abs_rank_shift=2,
        per_category={
            InjectionCategory.INSTRUCTION_OVERRIDE.value: {"detection_rate": 1.0, "n": 12.0},
            InjectionCategory.BENIGN_CONTROL.value: {"detection_rate": 0.05, "n": 20.0},
        },
    )


@pytest.fixture
def manifest() -> RunManifest:
    return RunManifest(
        run_id="run_test",
        created_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        config_hash="abc123def4567890",
        seeds=(42, 43, 44),
        dataset_hash="0f0f0f0f0f0f0f0f",
        package_versions={"numpy": "2.4.0", "xgboost": "3.0.0"},
        llm_backend=LLMBackendKind.HEURISTIC,
        llm_model="heuristic",
        feed_mode=FeedMode.OFFLINE,
        as_of=date(2024, 6, 1),
        command="vulnpriority run-all",
    )


# ---------------------------------------------------------------------------
# The expected files
# ---------------------------------------------------------------------------


def test_report_generation_writes_the_expected_files(
    tmp_path: Path, bundles, ablation, simulations, selections, adversarial, manifest
) -> None:
    artifacts = ReportBuilder().build(
        tmp_path / "report_out",
        bundles=bundles,
        ablation=ablation,
        simulations=simulations,
        selections=selections,
        adversarial=adversarial,
        manifest=manifest,
    )

    assert isinstance(artifacts, ReportArtifacts)
    assert artifacts.report_md.exists() and artifacts.report_md.name == "report.md"
    assert artifacts.report_json.exists() and artifacts.report_json.name == "report.json"
    assert artifacts.report_md.stat().st_size > 0

    names = {path.name for path in artifacts.figures}
    assert names == {
        "metric_comparison.png",
        "reliability.png",
        "ablation_main_effects.png",
        "exposure_curves.png",
    }
    for path in artifacts.figures:
        assert path.exists() and path.stat().st_size > 0
        assert path.parent.name == "figures"


def test_matplotlib_uses_the_headless_backend() -> None:
    assert matplotlib.get_backend().lower() == "agg"


# ---------------------------------------------------------------------------
# report.json carries every number
# ---------------------------------------------------------------------------


def test_report_json_is_machine_readable_and_complete(
    tmp_path: Path, bundles, ablation, simulations, selections, adversarial, manifest
) -> None:
    artifacts = ReportBuilder().build(
        tmp_path / "out",
        bundles=bundles,
        ablation=ablation,
        simulations=simulations,
        selections=selections,
        adversarial=adversarial,
        manifest=manifest,
    )
    payload = json.loads(artifacts.report_json.read_text(encoding="utf-8"))

    assert payload["manifest"]["config_hash"] == "abc123def4567890"
    assert payload["manifest"]["seeds"] == [42, 43, 44]
    assert set(payload["metrics"]) == {
        RankerName.LAMBDAMART.value,
        RankerName.EXPECTED_LOSS.value,
        RankerName.CVSS_ONLY.value,
        RankerName.RANDOM.value,
    }

    learned = payload["metrics"][RankerName.LAMBDAMART.value]["values"]
    assert learned["ndcg@10"]["value"] == pytest.approx(0.71)
    assert learned["ndcg@10"]["ci_low"] == pytest.approx(0.66)
    assert learned["mcc"]["value"] == pytest.approx(0.33)

    assert len(payload["bundles"]) == len(bundles)
    assert len(payload["ablation"]["cells"]) == 8
    assert payload["ablation"]["main_effects"]["A"]["ndcg@10"] == pytest.approx(0.10)
    assert len(payload["simulations"]) == 2
    assert len(payload["selections"]) == 1
    assert payload["adversarial"]["n_cases"] == 80
    assert len(payload["figures"]) == 4
    assert all(entry.startswith("figures/") for entry in payload["figures"])
    assert payload["protocol"]["folds"][0]["gap_days"] == 30


def test_report_json_never_contains_non_finite_numbers(tmp_path: Path, bundles) -> None:
    """NaN is not valid JSON; the report must never emit it."""
    artifacts = ReportBuilder().build(tmp_path / "out", bundles=bundles)
    text = artifacts.report_json.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)  # strict parse


# ---------------------------------------------------------------------------
# report.md is citable
# ---------------------------------------------------------------------------


def test_report_markdown_holds_the_tables_a_reader_would_cite(
    tmp_path: Path, bundles, ablation, simulations, selections, adversarial, manifest
) -> None:
    artifacts = ReportBuilder().build(
        tmp_path / "out",
        bundles=bundles,
        ablation=ablation,
        simulations=simulations,
        selections=selections,
        adversarial=adversarial,
        manifest=manifest,
    )
    text = artifacts.report_md.read_text(encoding="utf-8")

    for heading in (
        "# vulnpriority evaluation report",
        "## Reproducibility",
        "## Protocol",
        "## Ranking quality",
        "## Classification and calibration",
        "## Decision metrics",
        "## Component ablation",
        "## Selection under a remediation budget",
        "## Longitudinal simulation",
        "## Adversarial robustness",
        "## Figures",
    ):
        assert heading in text

    # every ranker appears, so the learned model is never shown without its baselines
    for ranker in (RankerName.LAMBDAMART, RankerName.CVSS_ONLY, RankerName.RANDOM):
        assert ranker.value in text

    assert "ndcg@10" in text and "0.710" in text
    assert "[0.660, 0.760]" in text                 # the interval is printed, not just the point
    assert "abc123def4567890" in text               # config hash, for reproducibility
    assert "Main effects" in text and "Interactions" in text
    assert "figures/metric_comparison.png" in text


def test_the_simulation_section_leads_with_prevention_and_labels_the_total(
    tmp_path: Path, simulations
) -> None:
    """Total exposure must never appear without the label saying what it measures."""
    artifacts = ReportBuilder().build(tmp_path / "out", simulations=simulations)
    text = artifacts.report_md.read_text(encoding="utf-8")

    assert "**Prevention.** Under `lambdamart`, 2 of 2" in text
    assert "against 0 of 2" in text and "`cvss_only`" in text
    assert "prevention rate" in text
    assert "exposure days (total, capacity-bound)" in text
    assert "capacity measurement, not a prioritisation measurement" in text
    assert "measured on exposure days carried by the confirmed-exploited findings" in text
    # the reduction column is described against the right quantity, never the total
    assert "reduction vs reference" in text


def test_the_capacity_context_is_printed_when_supplied(tmp_path: Path, simulations) -> None:
    from vulnpriority.eval.simulation import CapacityContext

    capacity = CapacityContext(
        weeks=26,
        capacity_hours_per_week=20.0,
        n_findings=901,
        n_clusters=712,
        n_exploited=145,
        backlog_hours=9_000.0,
    )
    artifacts = ReportBuilder().build(
        tmp_path / "out", simulations=simulations, capacity=capacity
    )
    text = artifacts.report_md.read_text(encoding="utf-8")
    assert "### Capacity context" in text
    assert "total hours available" in text and "520.0" in text
    assert "9000.0" in text
    assert "5.8%" in text                      # share of the backlog the budget can reach
    assert "most findings are never reached under any ordering" in text

    payload = json.loads(artifacts.report_json.read_text(encoding="utf-8"))
    summary = payload["simulation_summary"]
    assert summary["reduction_metric"] == "exposure_days_exploited"
    assert summary["capacity"]["reachable_fraction"] == pytest.approx(520 / 9000)
    assert summary["capacity"]["capacity_bound"] is True
    assert summary["best_by_prevention"] == "lambdamart"


def test_the_capacity_line_still_appears_without_a_capacity_context(
    tmp_path: Path, simulations
) -> None:
    """Weeks and budget are derivable from the results; the backlog is honestly missing."""
    artifacts = ReportBuilder().build(tmp_path / "out", simulations=simulations)
    text = artifacts.report_md.read_text(encoding="utf-8")
    assert "### Capacity context" in text
    assert "total hours available" in text
    assert "not supplied" in text


def test_the_random_control_is_labelled_as_a_control_in_the_report(tmp_path: Path) -> None:
    random_split = SPLIT.model_copy(update={"kind": SplitKind.RANDOM, "gap_days": 0})
    control = bundle(RankerName.LAMBDAMART, 0.93).model_copy(update={"split": random_split})
    artifacts = ReportBuilder().build(tmp_path / "out", bundles=[control])
    text = artifacts.report_md.read_text(encoding="utf-8")
    assert "Control condition present" in text
    assert "overstates" in text


def test_the_learned_ranker_is_listed_before_the_baselines(tmp_path: Path, bundles) -> None:
    artifacts = ReportBuilder().build(tmp_path / "out", bundles=bundles)
    payload = json.loads(artifacts.report_json.read_text(encoding="utf-8"))
    assert list(payload["metrics"])[0] == RankerName.LAMBDAMART.value


# ---------------------------------------------------------------------------
# Partial and empty runs
# ---------------------------------------------------------------------------


def test_a_metrics_only_run_still_produces_a_readable_report(tmp_path: Path, bundles) -> None:
    artifacts = ReportBuilder().build(tmp_path / "out", bundles=bundles)
    text = artifacts.report_md.read_text(encoding="utf-8")
    assert artifacts.report_json.exists()
    assert "_No ablation supplied._" in text
    assert "_No simulation results supplied._" in text
    assert "_No run manifest supplied._" in text
    names = {path.name for path in artifacts.figures}
    assert "metric_comparison.png" in names
    assert "reliability.png" in names                  # one bundle is calibrated
    assert "ablation_main_effects.png" not in names
    assert "exposure_curves.png" not in names


def test_an_empty_run_writes_both_files_without_raising(tmp_path: Path) -> None:
    artifacts = ReportBuilder().build(tmp_path / "empty")
    assert artifacts.report_md.exists() and artifacts.report_json.exists()
    assert artifacts.figures == ()
    payload = json.loads(artifacts.report_json.read_text(encoding="utf-8"))
    assert payload["metrics"] == {} and payload["bundles"] == []
    assert "_No metric bundles supplied._" in artifacts.report_md.read_text(encoding="utf-8")


def test_an_uncalibrated_run_skips_the_reliability_diagram(tmp_path: Path) -> None:
    artifacts = ReportBuilder().build(
        tmp_path / "out", bundles=[bundle(RankerName.CVSS_ONLY, 0.4)]
    )
    assert "reliability.png" not in {path.name for path in artifacts.figures}
    text = artifacts.report_md.read_text(encoding="utf-8")
    assert "| cvss_only |" in text


def test_rebuilding_into_the_same_directory_overwrites_cleanly(tmp_path: Path, bundles) -> None:
    target = tmp_path / "out"
    first = ReportBuilder().build(target, bundles=bundles)
    second = ReportBuilder(title="second pass").build(target, bundles=bundles)
    assert first.report_md == second.report_md
    assert "second pass" in second.report_md.read_text(encoding="utf-8")
    assert json.loads(second.report_json.read_text(encoding="utf-8"))["title"] == "second pass"


def test_the_output_directory_is_created_if_missing(tmp_path: Path, bundles) -> None:
    nested = tmp_path / "a" / "b" / "c"
    artifacts = ReportBuilder().build(nested, bundles=bundles)
    assert nested.exists() and (nested / "figures").is_dir()
    assert artifacts.as_dict()["report_md"].endswith("report.md")
