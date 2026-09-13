"""Run artifacts: every document survives a round trip through disk, validated both ways.

A resumed run reloads artifacts instead of recomputing them, and the manifest is the record
that lets a published number be reproduced. Both guarantees are only as good as the
save/load round trip, so each artifact type is written, read back, and compared on its
serialised form - not on ``is``, and not on a hand-picked subset of fields.

The enriched findings here come from the real ingest → assess → enrich stages on a small
generated world, so the round trip is exercised against the object graph the pipeline
actually produces (nested untrusted text, audits, trust ledgers, log-odds terms) rather than
against a hand-built stub that would not catch a serialisation gap.
"""

from __future__ import annotations

import importlib
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from vulnpriority.core.config import PipelineConfig, SyntheticConfig
from vulnpriority.core.enums import (
    DetectorName,
    FeedMode,
    InjectionCategory,
    LLMBackendKind,
    LabelSource,
    MetricName,
    PrivilegeLevel,
    RankerName,
    SelectionMethod,
    SplitKind,
    TrustTier,
)
from vulnpriority.core.models import (
    AblationCell,
    AblationTable,
    AdversarialOutcome,
    AdversarialReport,
    AttackGraphSummary,
    AttackPath,
    CalibrationReport,
    ChainScore,
    ComponentFlags,
    Explanation,
    FeatureContribution,
    FeatureFrame,
    GraphEdge,
    GraphNode,
    GroundTruthLabel,
    LabelSet,
    ManipulationAlert,
    MetricBundle,
    MetricValue,
    MinorityClassReport,
    RankedFinding,
    RankingResult,
    SelectionResult,
    SimulationResult,
    Split,
    feature_names_for,
)
from vulnpriority.pipeline.artifacts import (
    ARTIFACT_FILES,
    RunArtifacts,
    artifact_path,
    default_run_id,
    default_run_root,
    find_run_root,
    input_digest,
    flags_from_columns,
    load_ablation,
    load_adversarial,
    load_chain,
    load_enriched,
    load_explanations,
    load_features,
    load_labels,
    load_manifest,
    load_metrics,
    load_ranking,
    load_scans,
    load_selection,
    load_simulation,
    save_ablation,
    save_adversarial,
    save_chain,
    save_enriched,
    save_explanations,
    save_features,
    save_labels,
    save_manifest,
    save_metrics,
    save_ranking,
    save_scans,
    save_selection,
    save_simulation,
)
from vulnpriority.pipeline.runner import build_manifest, package_versions
from vulnpriority.pipeline.stages import assess_stage, enrich_stage, ingest_stage
from vulnpriority.synth.generator import SyntheticDataset

TINY = SyntheticConfig(
    seed=808,
    n_apps=2,
    scans_per_app=2,
    endpoints_per_app=(12, 14),
    findings_per_app=(8, 14),
)


# ---------------------------------------------------------------------------
# Fixtures: a small real run, offline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> SyntheticDataset:
    return SyntheticDataset.generate(TINY, tmp_path_factory.mktemp("artifact-data"))


@pytest.fixture(scope="module")
def config(dataset: SyntheticDataset, tmp_path_factory: pytest.TempPathFactory) -> PipelineConfig:
    """The default offline configuration, pointed at the generated feed fixtures."""
    base = PipelineConfig(synthetic=TINY, output_dir=tmp_path_factory.mktemp("artifact-runs"))
    feeds = base.feeds.model_copy(update={"fixture_dir": Path(dataset.fixture_dir)})
    return base.model_copy(update={"feeds": feeds})


@pytest.fixture(scope="module")
def pipeline_state(config: PipelineConfig, dataset: SyntheticDataset):
    """Scans and enriched findings from the real stages: no stubs in the round trip."""
    scans = ingest_stage(config, scans=dataset.scans)
    assessments = assess_stage(config, scans)
    enriched = enrich_stage(config, scans, assessments)
    return scans, enriched



def _require(*modules: str) -> None:
    """Skip when a component package written by another group is not importable.

    ``pytest.importorskip`` only skips a module that is genuinely *absent*; an ``ImportError``
    raised from *inside* a module that exists is re-raised. ``vulnpriority.rank.features``
    validates itself against ``FEATURE_SPECS`` at import time, which is the right behaviour
    for the ranking group's own tests and the wrong one here: these tests are about the
    pipeline wiring, and a component package mid-edit is not a failure of it.
    """
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as error:
            pytest.skip(f"{module} is not importable: {error}")



def _require_ranking_stack(enriched) -> None:
    """Skip when the ranking or selection package cannot yet build a frame from real rows.

    The precondition is stated as the thing the test actually needs - a feature matrix built
    from a real enriched finding - rather than as a blanket try/except around the assertions.
    An empty frame is not enough of a probe: a feature the builder declares but cannot
    populate only fails once there is a row to populate.
    """
    _require("vulnpriority.rank", "vulnpriority.select.knapsack")
    from vulnpriority.rank.features import FeatureBuilder

    try:
        FeatureBuilder().build(list(enriched)[:1], {}, ComponentFlags())
    except Exception as error:  # noqa: BLE001 - any failure here is a missing precondition
        pytest.skip(f"vulnpriority.rank cannot build a feature frame yet: {type(error).__name__}: {error}")


# ---------------------------------------------------------------------------
# Synthetic artifacts for the stages written by other groups
# ---------------------------------------------------------------------------


def _feature_frame(finding_ids: list[str], group_ids: list[str], flags: ComponentFlags) -> FeatureFrame:
    columns = feature_names_for(flags)
    data = {
        name: [round(0.01 * (row + column), 4) for row in range(len(finding_ids))]
        for column, name in enumerate(columns)
    }
    return FeatureFrame(
        X=pd.DataFrame(data, columns=columns),
        finding_ids=finding_ids,
        group_ids=group_ids,
        flags=flags,
    )


def _ranking(finding_ids: list[str], group_ids: list[str]) -> RankingResult:
    items = tuple(
        RankedFinding(
            finding_id=finding_id,
            scan_id=scan_id,
            rank=position + 1,
            score=1.0 - 0.01 * position,
            expected_loss=1000.0 * (position + 1),
            chain_adjusted_loss=1500.0 * (position + 1),
            p_exploit=min(0.99, 0.02 * (position + 1)),
            explanation=Explanation(
                finding_id=finding_id,
                base_value=0.5,
                top_contributions=(
                    FeatureContribution(
                        feature="b_kev",
                        value=1.0,
                        shap_value=0.42,
                        group=None,
                        tier=TrustTier.CURATED_FEED,
                    ),
                ),
                reason_codes=("in CISA KEV", "internet facing"),
                untrusted_influence_share=0.12,
            ),
            alerts=(
                (
                    ManipulationAlert(
                        finding_id=finding_id,
                        detector=DetectorName.DISPLACEMENT,
                        severity=0.6,
                        message="rank moved by 5 when untrusted tiers were neutralised",
                        rank_delta=5,
                    ),
                )
                if position == 0
                else ()
            ),
        )
        for position, (finding_id, scan_id) in enumerate(zip(finding_ids, group_ids))
    )
    return RankingResult(
        ranker=RankerName.LAMBDAMART,
        flags=ComponentFlags(),
        config_hash="deadbeefdeadbeef",
        seed=42,
        items=items,
    )


def _chain(finding_ids: list[str], scan_id: str):
    nodes = (
        GraphNode(node_id="state:internet:NONE", asset="internet", privilege=PrivilegeLevel.NONE, is_entry=True),
        GraphNode(
            node_id="state:host:ADMIN",
            asset="host",
            privilege=PrivilegeLevel.ADMIN,
            value=250_000.0,
            is_target=True,
        ),
    )
    edges = (
        GraphEdge(
            src="state:internet:NONE",
            dst="state:host:ADMIN",
            probability=0.21,
            finding_id=finding_ids[0],
            kind="exploit",
            tier=TrustTier.SCANNER,
        ),
    )
    graph = AttackGraphSummary(
        scan_id=scan_id,
        nodes=nodes,
        edges=edges,
        entry_node="state:internet:NONE",
        target_nodes=("state:host:ADMIN",),
        total_risk=52_500.0,
        top_paths=(
            AttackPath(
                nodes=("state:internet:NONE", "state:host:ADMIN"),
                finding_ids=(finding_ids[0],),
                probability=0.21,
                target_value=250_000.0,
            ),
        ),
        monotone_verified=True,
        rejected_untrusted_edges=3,
    )
    scores = {
        finding_id: ChainScore(
            finding_id=finding_id,
            reach_delta=1000.0 * (position + 1),
            max_path_prob_to_target=0.2,
            n_paths_through=position + 1,
            betweenness=0.05 * position,
            hops_from_entry=1,
            privilege_gain=2,
            is_chokepoint=position == 0,
            best_target="state:host:ADMIN",
        )
        for position, finding_id in enumerate(finding_ids)
    }
    return graph, scores


def _split(scan_ids: list[str]) -> Split:
    half = max(1, len(scan_ids) // 2)
    return Split(
        kind=SplitKind.TIME_ORDERED,
        fold=0,
        train_scan_ids=tuple(scan_ids[:half]),
        test_scan_ids=tuple(scan_ids[half:]),
        train_end=date(2024, 3, 1),
        test_start=date(2024, 3, 31),
        gap_days=30,
    )


def _metrics(scan_ids: list[str]) -> list[MetricBundle]:
    split = _split(scan_ids)
    return [
        MetricBundle(
            ranker=ranker,
            flags=ComponentFlags(),
            split=split,
            seed=42,
            values=(
                MetricValue(name=MetricName.NDCG_AT_K, k=10, value=0.71, ci_low=0.6, ci_high=0.8, n=24),
                MetricValue(name=MetricName.PR_AUC, value=0.33),
                MetricValue(name=MetricName.MCC, value=0.21),
            ),
            calibration=CalibrationReport(
                brier=0.04,
                ece=0.02,
                n_bins=10,
                bin_confidence=(0.1, 0.5, 0.9),
                bin_accuracy=(0.08, 0.48, 0.93),
                bin_count=(10, 20, 5),
            ),
            minority=MinorityClassReport(
                positive_rate=0.06,
                mcc=0.21,
                f1_positive=0.31,
                balanced_accuracy=0.62,
                threshold=0.5,
                per_class={"exploited": {"precision": 0.3, "recall": 0.4}},
            ),
            runtime_seconds=1.25,
        )
        for ranker in (RankerName.LAMBDAMART, RankerName.CVSS_ONLY)
    ]


def _labels(finding_ids: list[str]) -> LabelSet:
    return LabelSet(
        observation_cutoff=date(2024, 7, 1),
        labels=tuple(
            GroundTruthLabel(
                finding_id=finding_id,
                cve_id=f"CVE-2024-{20000 + position}",
                exploited=position % 5 == 0,
                relevance_grade=4 if position % 5 == 0 else 0,
                sources=(LabelSource.SYNTHETIC_ORACLE,),
                first_evidence_date=date(2024, 4, 1) if position % 5 == 0 else None,
                source_agreement=0.8,
            )
            for position, finding_id in enumerate(finding_ids)
        ),
    )


# ---------------------------------------------------------------------------
# Per-artifact round trips
# ---------------------------------------------------------------------------


def test_artifact_file_names_cover_the_design_contract() -> None:
    for name in ("scan", "enriched", "chain", "features", "ranking", "metrics", "manifest"):
        assert name in ARTIFACT_FILES
    assert ARTIFACT_FILES["scan"] == "scan.json"
    assert ARTIFACT_FILES["features"] == "features.csv"
    assert artifact_path("/tmp/run", "manifest").name == "manifest.json"

    with pytest.raises(Exception):
        artifact_path("/tmp/run", "not-an-artifact")


def test_scans_round_trip(tmp_path: Path, pipeline_state) -> None:
    scans, _enriched = pipeline_state
    save_scans(scans, tmp_path)
    reloaded = load_scans(tmp_path)

    assert len(reloaded) == len(scans)
    expected = sorted(scans, key=lambda scan: (scan.scanned_at, scan.scan_id))
    assert [scan.model_dump(mode="json") for scan in reloaded] == [
        scan.model_dump(mode="json") for scan in expected
    ]


def test_enriched_findings_round_trip(tmp_path: Path, pipeline_state) -> None:
    """The deepest object graph in the framework: assessments, audits, trust and terms."""
    _scans, enriched = pipeline_state
    assert enriched, "the enrich stage produced nothing to round trip"
    save_enriched(enriched, tmp_path)
    reloaded = load_enriched(tmp_path)

    assert [item.model_dump(mode="json") for item in reloaded] == [
        item.model_dump(mode="json") for item in enriched
    ]
    first = reloaded[0]
    assert first.likelihood.log_odds_terms == enriched[0].likelihood.log_odds_terms
    assert first.finding.description.sha256 == enriched[0].finding.description.sha256
    assert first.trust.max_tier_used == enriched[0].trust.max_tier_used


def test_chain_round_trips_graphs_and_scores(tmp_path: Path, pipeline_state) -> None:
    scans, enriched = pipeline_state
    ids = [item.finding_id for item in enriched[:4]]
    graph, scores = _chain(ids, scans[0].scan_id)
    save_chain([graph], {scans[0].scan_id: scores}, tmp_path)
    graphs, reloaded = load_chain(tmp_path)

    assert len(graphs) == 1
    assert graphs[0].model_dump(mode="json") == graph.model_dump(mode="json")
    assert set(reloaded) == {scans[0].scan_id}
    assert {key: value.model_dump(mode="json") for key, value in reloaded[scans[0].scan_id].items()} == {
        key: value.model_dump(mode="json") for key, value in scores.items()
    }


@pytest.mark.parametrize("flags", ComponentFlags.all_cells(), ids=lambda cell: cell.label())
def test_features_round_trip_in_every_ablation_cell(tmp_path: Path, flags: ComponentFlags) -> None:
    """The CSV recovers its ablation cell from the columns that are present."""
    finding_ids = [f"f_{index:04d}" for index in range(6)]
    group_ids = ["scan_a"] * 3 + ["scan_b"] * 3
    frame = _feature_frame(finding_ids, group_ids, flags)

    target = tmp_path / flags.label()
    target.mkdir()
    save_features(frame, target)
    reloaded = load_features(target)

    assert reloaded.flags == flags
    assert reloaded.finding_ids == finding_ids
    assert reloaded.group_ids == group_ids
    assert list(reloaded.X.columns) == feature_names_for(flags)
    pd.testing.assert_frame_equal(reloaded.X, frame.X.astype(float), check_dtype=False)
    assert list(reloaded.group_sizes()) == [3, 3]


def test_flags_are_recovered_from_column_sets() -> None:
    for flags in ComponentFlags.all_cells():
        assert flags_from_columns(feature_names_for(flags)) == flags
    assert flags_from_columns(["finding_id", "cvss_base_max"]) == ComponentFlags(
        a=False, b=False, c=False
    )


def test_ranking_round_trips_with_explanations_and_alerts(tmp_path: Path, pipeline_state) -> None:
    scans, enriched = pipeline_state
    ids = [item.finding_id for item in enriched[:4]]
    groups = [item.scan_id for item in enriched[:4]]
    ranking = _ranking(ids, groups)
    save_ranking(ranking, tmp_path)
    reloaded = load_ranking(tmp_path)

    assert reloaded.model_dump(mode="json") == ranking.model_dump(mode="json")
    assert reloaded.rank_of(ids[0]) == 1
    assert reloaded.items[0].manipulation_flag is True
    assert reloaded.items[0].explanation is not None
    assert reloaded.items[0].explanation.reason_codes == ("in CISA KEV", "internet facing")
    assert scans


def test_labels_round_trip_and_keep_the_cvss_prohibition(tmp_path: Path, pipeline_state) -> None:
    _scans, enriched = pipeline_state
    labels = _labels([item.finding_id for item in enriched[:10]])
    save_labels(labels, tmp_path)
    reloaded = load_labels(tmp_path)

    assert reloaded.model_dump(mode="json") == labels.model_dump(mode="json")
    assert reloaded.positives() == labels.positives()
    assert all(label.cvss_used_as_label is False for label in reloaded.labels)


def test_metrics_round_trip_with_calibration_and_minority(tmp_path: Path, pipeline_state) -> None:
    scans, _enriched = pipeline_state
    bundles = _metrics([scan.scan_id for scan in scans])
    save_metrics(bundles, tmp_path)
    reloaded = load_metrics(tmp_path)

    assert [bundle.model_dump(mode="json") for bundle in reloaded] == [
        bundle.model_dump(mode="json") for bundle in bundles
    ]
    assert reloaded[0].get(MetricName.NDCG_AT_K, 10) == pytest.approx(0.71)
    assert reloaded[0].calibration is not None and reloaded[0].calibration.ece == pytest.approx(0.02)
    assert reloaded[0].minority is not None
    assert "ndcg@10" in reloaded[0].as_dict()


def test_selection_and_simulation_round_trip(tmp_path: Path, pipeline_state) -> None:
    scans, enriched = pipeline_state
    selections = [
        SelectionResult(
            scan_id=scans[0].scan_id,
            ranker=RankerName.LAMBDAMART,
            method=SelectionMethod.DP_EXACT,
            budget_hours=40.0,
            selected_ids=tuple(item.finding_id for item in enriched[:3]),
            total_hours=12.5,
            risk_captured=125_000.0,
            risk_capture_fraction=0.62,
            exploited_captured=2,
            exploited_total=3,
        )
    ]
    simulations = [
        SimulationResult(
            policy=RankerName.LAMBDAMART,
            weeks=26,
            capacity_hours_per_week=20.0,
            exposure_days_total=1234.5,
            exposure_days_exploited=210.0,
            expected_loss_days=98_765.0,
            exploited_remediated_before_exploit=4,
            exploited_total=7,
            weekly_cumulative_exposure=(10.0, 25.0, 47.5),
            reduction_vs_cvss=-0.31,
        )
    ]
    save_selection(selections, tmp_path)
    save_simulation(simulations, tmp_path)

    assert [item.model_dump(mode="json") for item in load_selection(tmp_path)] == [
        item.model_dump(mode="json") for item in selections
    ]
    assert [item.model_dump(mode="json") for item in load_simulation(tmp_path)] == [
        item.model_dump(mode="json") for item in simulations
    ]


def test_ablation_and_adversarial_round_trip(tmp_path: Path) -> None:
    table = AblationTable(
        cells=tuple(
            AblationCell(flags=flags, seeds=(42, 43), mean={"ndcg@10": 0.5}, std={"ndcg@10": 0.02}, n=2)
            for flags in ComponentFlags.all_cells()
        ),
        main_effects={"A": {"ndcg@10": 0.04}},
        interactions={"A:C": {"ndcg@10": 0.01}},
        paired_ci={"A": {"ndcg@10": (0.01, 0.07)}},
    )
    report = AdversarialReport(
        backend=LLMBackendKind.HEURISTIC,
        corpus_version="v1",
        n_cases=80,
        attack_success_rate=0.0,
        canary_leak_rate=0.0,
        detection_rate=0.9,
        false_positive_rate=0.05,
        mean_abs_rank_shift=0.4,
        max_abs_rank_shift=2,
        per_category={"instruction_override": {"success_rate": 0.0}},
        outcomes=(
            AdversarialOutcome(
                case_id="case-001",
                category=InjectionCategory.INSTRUCTION_OVERRIDE,
                detected_pre_llm=True,
                delta_feasibility=0.02,
                rank_shift=1,
                passed=True,
                notes="redacted before the prompt was built",
            ),
        ),
    )
    save_ablation(table, tmp_path)
    save_adversarial(report, tmp_path)

    assert load_ablation(tmp_path).model_dump(mode="json") == table.model_dump(mode="json")
    assert load_ablation(tmp_path).cell(ComponentFlags()) is not None
    assert load_adversarial(tmp_path).model_dump(mode="json") == report.model_dump(mode="json")


def test_explanations_round_trip(tmp_path: Path) -> None:
    explanations = [
        Explanation(
            finding_id="f_0001",
            base_value=0.3,
            top_contributions=(
                FeatureContribution(feature="b_epss", value=0.4, shap_value=0.2, tier=TrustTier.CURATED_FEED),
                FeatureContribution(
                    feature="a_exploit_feasibility",
                    value=0.8,
                    shap_value=-0.1,
                    tier=TrustTier.REFERENCE_PAGE,
                ),
            ),
            reason_codes=("EPSS above the 90th percentile",),
            untrusted_influence_share=0.22,
        )
    ]
    save_explanations(explanations, tmp_path)
    assert [item.model_dump(mode="json") for item in load_explanations(tmp_path)] == [
        item.model_dump(mode="json") for item in explanations
    ]


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_manifest_records_everything_needed_to_reproduce(
    tmp_path: Path, config: PipelineConfig, pipeline_state, dataset: SyntheticDataset
) -> None:
    scans, _enriched = pipeline_state
    manifest = build_manifest(
        config,
        "run_test",
        scans=scans,
        dataset_digest=dataset.dataset_hash,
        command="vulnpriority run-all --synthetic",
        created_at=datetime(2024, 7, 1, 12, 0, 0),
    )
    save_manifest(manifest, tmp_path)
    reloaded = load_manifest(tmp_path)

    assert reloaded.model_dump(mode="json") == manifest.model_dump(mode="json")
    assert reloaded.config_hash == config.hash()
    assert reloaded.dataset_hash == dataset.dataset_hash
    assert config.seed in reloaded.seeds and config.ranking.seed in reloaded.seeds
    assert reloaded.command == "vulnpriority run-all --synthetic"
    # The manifest records what the run *resolved* to, not the switch position. A config
    # left on ``auto`` used to be written down as "auto (claude-sonnet-5)" on machines with
    # no Anthropic key, naming a model that was never called - the one thing a provenance
    # record must not do. ``offline.yaml`` pins both, so resolution is the identity here.
    assert reloaded.feed_mode in {config.feeds.mode, FeedMode.OFFLINE, FeedMode.LIVE_WITH_CACHE}
    assert reloaded.llm_backend != LLMBackendKind.AUTO
    assert reloaded.as_of == max(scan.scanned_at.date() for scan in scans)

    versions = reloaded.package_versions
    for name in ("python", "vulnpriority", "numpy", "pandas", "pydantic", "xgboost"):
        assert name in versions and versions[name]
    assert package_versions(("definitely-not-a-real-distribution",))[
        "definitely-not-a-real-distribution"
    ] == "not installed"


def test_the_config_hash_changes_when_the_config_does(config: PipelineConfig) -> None:
    other = config.model_copy(update={"seed": config.seed + 1})
    assert other.hash() != config.hash()
    assert default_run_id(other) != default_run_id(config)
    assert default_run_root(config, default_run_id(config)).name == default_run_id(config)


def test_two_different_scans_under_one_config_get_different_runs(
    config: PipelineConfig, tmp_path
) -> None:
    """The run id is the resume key, so it has to name the input as well as the settings.

    Without this the second of two scans run under one configuration lands in the first
    one's directory, reloads its artifacts and reports the first scan's findings as its
    own - a wrong answer delivered confidently, which is the worst failure mode a
    reproducibility mechanism can have.
    """
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text('{"findings": [{"name": "one"}]}', encoding="utf-8")
    second.write_text('{"findings": [{"name": "two"}]}', encoding="utf-8")

    assert default_run_id(config, [first]) != default_run_id(config, [second])
    # ...and the configuration still decides the rest of the identity.
    assert default_run_id(config, [first]).startswith(f"run_{config.hash()}")


def test_the_same_scan_resumes_itself_wherever_it_sits_on_disk(
    config: PipelineConfig, tmp_path
) -> None:
    """Identity is the content, not the path: a copied report is the same run."""
    original = tmp_path / "report.json"
    copy = tmp_path / "elsewhere" / "renamed.json"
    copy.parent.mkdir()
    payload = '{"findings": [{"name": "one"}]}'
    original.write_text(payload, encoding="utf-8")
    copy.write_text(payload, encoding="utf-8")
    assert default_run_id(config, [original]) == default_run_id(config, [copy])

    original.write_text('{"findings": [{"name": "changed"}]}', encoding="utf-8")
    assert default_run_id(config, [original]) != default_run_id(config, [copy])


def test_an_explicit_run_id_still_wins(config: PipelineConfig, tmp_path) -> None:
    named = config.model_copy(update={"run_id": "my-run"})
    scan = tmp_path / "a.json"
    scan.write_text("{}", encoding="utf-8")
    assert default_run_id(named, [scan]) == "my-run"


def test_a_run_with_no_input_keeps_the_bare_config_hash(config: PipelineConfig) -> None:
    """A synthetic run has no report to name, and its id must not change shape."""
    assert default_run_id(config) == f"run_{config.hash()}"
    assert input_digest() == ""


def test_find_run_root_locates_the_latest_run_of_a_configuration(
    config: PipelineConfig, tmp_path
) -> None:
    """``explain`` and ``manifest`` are not told which scan was analysed.

    They can no longer reconstruct the run id, but every run of one configuration shares the
    config-hash prefix, so the directory is still findable.
    """
    rooted = config.model_copy(update={"output_dir": tmp_path})
    older = tmp_path / f"run_{rooted.hash()}_aaaaaaaaaaaa"
    newer = tmp_path / f"run_{rooted.hash()}_bbbbbbbbbbbb"
    for path in (older, newer):
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
    import os, time
    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))

    assert find_run_root(rooted) == newer


def test_find_run_root_reports_the_exact_path_when_nothing_matches(
    config: PipelineConfig, tmp_path
) -> None:
    rooted = config.model_copy(update={"output_dir": tmp_path})
    assert find_run_root(rooted) == default_run_root(rooted, default_run_id(rooted))


# ---------------------------------------------------------------------------
# The whole RunArtifacts object
# ---------------------------------------------------------------------------


def test_run_artifacts_round_trip_everything(
    tmp_path: Path, config: PipelineConfig, pipeline_state, dataset: SyntheticDataset
) -> None:
    scans, enriched = pipeline_state
    ids = [item.finding_id for item in enriched]
    groups = [item.scan_id for item in enriched]
    graph, scores = _chain(ids[:4], scans[0].scan_id)

    root = tmp_path / "run_full"
    artifacts = RunArtifacts(
        run_id="run_full",
        root=root,
        config=config,
        scans=scans,
        enriched=enriched,
        graphs=[graph],
        chain_scores={scans[0].scan_id: scores},
        features=_feature_frame(ids, groups, ComponentFlags()),
        ranking=_ranking(ids, groups),
        explanations=[
            item.explanation for item in _ranking(ids, groups).items if item.explanation
        ],
        labels=_labels(ids),
        metrics=_metrics([scan.scan_id for scan in scans]),
        selection=[
            SelectionResult(
                scan_id=scans[0].scan_id,
                ranker=RankerName.LAMBDAMART,
                method=SelectionMethod.GREEDY_RATIO,
                budget_hours=40.0,
                selected_ids=tuple(ids[:2]),
                total_hours=8.0,
                risk_captured=1000.0,
                risk_capture_fraction=0.4,
            )
        ],
        simulation=[
            SimulationResult(
                policy=RankerName.EXPECTED_LOSS,
                weeks=26,
                capacity_hours_per_week=20.0,
                exposure_days_total=100.0,
            )
        ],
        manifest=build_manifest(
            config, "run_full", scans=scans, dataset_digest=dataset.dataset_hash, command="pytest"
        ),
    )
    artifacts.save()

    for name in ("config", "scan", "enriched", "chain", "features", "ranking", "labels",
                 "metrics", "selection", "simulation", "explanations", "manifest"):
        assert artifact_path(root, name).is_file(), f"{name} was not written"

    reloaded = RunArtifacts.load(root)

    assert reloaded.run_id == "run_full"
    assert reloaded.config is not None and reloaded.config.hash() == config.hash()
    assert [scan.scan_id for scan in reloaded.scans] == [
        scan.scan_id for scan in sorted(scans, key=lambda item: (item.scanned_at, item.scan_id))
    ]
    assert [item.model_dump(mode="json") for item in reloaded.enriched] == [
        item.model_dump(mode="json") for item in enriched
    ]
    assert reloaded.graphs[0].model_dump(mode="json") == graph.model_dump(mode="json")
    assert set(reloaded.chain_scores) == {scans[0].scan_id}
    assert reloaded.features is not None and reloaded.features.flags == ComponentFlags()
    assert reloaded.ranking is not None and len(reloaded.ranking.items) == len(ids)
    assert reloaded.labels is not None and reloaded.labels.positives() == artifacts.labels.positives()
    assert len(reloaded.metrics) == 2
    assert len(reloaded.selection) == 1 and len(reloaded.simulation) == 1
    assert reloaded.manifest is not None and reloaded.manifest.dataset_hash == dataset.dataset_hash

    summary = reloaded.summary()
    assert summary["scans"] == len(scans)
    assert summary["enriched"] == len(enriched)
    assert summary["feature_columns"] == len(feature_names_for(ComponentFlags()))
    assert "manifest" in summary["artifacts"]


# ---------------------------------------------------------------------------
# Budget-constrained selection is a comparison, not a single number (Gap 10)
# ---------------------------------------------------------------------------


def test_selection_table_groups_by_policy_and_mode() -> None:
    """One row per (policy, optimised-or-walked), with the share pooled over scans.

    The pooled denominator matters: averaging per-scan fractions would give a scan with
    three findings the same weight as one with three hundred.
    """
    from vulnpriority.pipeline.stages import selection_table

    def result(scan: str, ranker: RankerName, method: SelectionMethod, captured: float,
               fraction: float, caught: int) -> SelectionResult:
        return SelectionResult(
            scan_id=scan,
            ranker=ranker,
            method=method,
            budget_hours=40.0,
            selected_ids=("a", "b"),
            total_hours=30.0,
            risk_captured=captured,
            risk_capture_fraction=fraction,
            exploited_captured=caught,
            exploited_total=10,
        )

    rows = selection_table(
        [
            # scan_1 holds 1000 of risk, scan_2 holds 100.
            result("scan_1", RankerName.LAMBDAMART, SelectionMethod.DP_EXACT, 800.0, 0.8, 8),
            result("scan_2", RankerName.LAMBDAMART, SelectionMethod.DP_EXACT, 80.0, 0.8, 6),
            result("scan_1", RankerName.LAMBDAMART, SelectionMethod.RANK_PREFIX, 500.0, 0.5, 9),
            result("scan_2", RankerName.LAMBDAMART, SelectionMethod.RANK_PREFIX, 50.0, 0.5, 7),
            result("scan_1", RankerName.CVSS_ONLY, SelectionMethod.RANK_PREFIX, 300.0, 0.3, 2),
            result("scan_2", RankerName.CVSS_ONLY, SelectionMethod.RANK_PREFIX, 90.0, 0.9, 1),
        ]
    )

    keyed = {(row["policy"], row["optimised"]): row for row in rows}
    assert set(keyed) == {
        (RankerName.LAMBDAMART, True),
        (RankerName.LAMBDAMART, False),
        (RankerName.CVSS_ONLY, False),
    }

    optimised = keyed[(RankerName.LAMBDAMART, True)]
    assert optimised["risk_captured"] == pytest.approx(880.0)
    assert optimised["risk_capture_share"] == pytest.approx(880.0 / 1100.0)
    assert optimised["exploited_captured"] == 14 and optimised["exploited_total"] == 20
    assert optimised["method"] == ["dp_exact"]
    assert optimised["budget_used"] == pytest.approx(60.0 / 80.0)

    # The pooled share is not the mean of the fractions: cvss_only averages 0.6 per scan but
    # captures only 390 of the 1100 actually at risk.
    baseline = keyed[(RankerName.CVSS_ONLY, False)]
    assert baseline["risk_capture_share"] == pytest.approx(390.0 / 1100.0)
    assert baseline["risk_capture_share"] < 0.6

    assert [row["risk_capture_share"] for row in rows] == sorted(
        (row["risk_capture_share"] for row in rows), reverse=True
    )


def test_selection_table_survives_a_zero_value_scan() -> None:
    """A scan where nothing has value must not divide by zero or poison the pool."""
    from vulnpriority.pipeline.stages import selection_table

    rows = selection_table(
        [
            SelectionResult(
                scan_id="scan_empty",
                ranker=RankerName.CVSS_ONLY,
                method=SelectionMethod.RANK_PREFIX,
                budget_hours=40.0,
                risk_captured=0.0,
                risk_capture_fraction=0.0,
                exploited_total=0,
            )
        ]
    )
    assert len(rows) == 1
    assert rows[0]["risk_capture_share"] == 0.0
    assert rows[0]["exploited_share"] == 0.0


def test_select_stage_scores_every_baseline_on_identical_items(
    config: PipelineConfig, pipeline_state
) -> None:
    """The budget result is only interpretable against the baselines, so all of them run.

    This is the regression guard for a real defect: the stage used to emit one row for the
    learned ranker and nothing else, which cannot answer Gap 10 at all - "57% of risk in 38
    hours" means nothing until the reader sees what CVSS order captures in the same hours.
    """
    scans, enriched = pipeline_state
    _require_ranking_stack(enriched)
    from vulnpriority.pipeline.stages import rank_stage, select_stage, selection_policies

    _frame, ranking, _explanations = rank_stage(
        config, enriched, {}, ranker=RankerName.EXPECTED_LOSS
    )
    results = select_stage(config, enriched, ranking, chain_scores={}, labels=_labels(
        [item.finding_id for item in enriched]
    ))

    expected_policies = set(selection_policies(config, ranking))
    assert expected_policies >= {RankerName(name) for name in config.evaluation.baselines}
    assert {row.ranker for row in results} == expected_policies

    scan_ids = {item.scan_id for item in enriched}
    # One row per (scan, policy), plus a second row for the configured ranker walked as a
    # plain queue so ordering quality and cost-awareness can be read apart.
    assert len(results) == len(scan_ids) * (len(expected_policies) + 1)
    assert len({(row.scan_id, row.ranker, row.method) for row in results}) == len(results)

    # Baselines are orderings, not optimisers.
    for row in results:
        if row.ranker != ranking.ranker:
            assert row.method == SelectionMethod.RANK_PREFIX, row.ranker
    own = {row.method for row in results if row.ranker == ranking.ranker}
    assert own == {config.selection.method, SelectionMethod.RANK_PREFIX}

    # Identical budget, identical item pool and identical ground truth per scan: only the
    # ordering differs, which is the only way the comparison means anything.
    assert {row.budget_hours for row in results} == {config.selection.budget_hours}
    for scan_id in scan_ids:
        rows = [row for row in results if row.scan_id == scan_id]
        assert len({row.exploited_total for row in rows}) == 1
        pools = [
            row.risk_captured / row.risk_capture_fraction
            for row in rows
            if row.risk_capture_fraction > 0.0
        ]
        assert pools, f"{scan_id} captured nothing under any policy"
        assert max(pools) - min(pools) < 1e-6, "policies were scored against different values"
        for row in rows:
            assert row.total_hours <= row.budget_hours + 1e-9


def test_selection_results_round_trip_per_policy(tmp_path: Path, config: PipelineConfig,
                                                 pipeline_state) -> None:
    """Every policy's row survives the artifact round trip, not just the first."""
    _scans, enriched = pipeline_state
    _require_ranking_stack(enriched)
    from vulnpriority.pipeline.stages import rank_stage, select_stage

    _frame, ranking, _explanations = rank_stage(
        config, enriched, {}, ranker=RankerName.EXPECTED_LOSS
    )
    results = select_stage(config, enriched, ranking, chain_scores={})

    save_selection(results, tmp_path)
    reloaded = load_selection(tmp_path)
    assert [item.model_dump(mode="json") for item in reloaded] == [
        item.model_dump(mode="json") for item in results
    ]
    assert len({item.ranker for item in reloaded}) > 1


# ---------------------------------------------------------------------------
# Capacity context: the simulation's results are unreadable without it
# ---------------------------------------------------------------------------


def test_capacity_round_trips_through_the_simulation_artifact(tmp_path: Path) -> None:
    """``simulation.json`` carries the capacity, and it rebuilds into the report's type.

    "11 of 145 prevented" reads as a poor showing until the reader learns the budget could
    only ever reach a few percent of the backlog, so the context has to survive to the
    report - including across a resumed run, where it comes back off disk.
    """
    from vulnpriority.pipeline.artifacts import load_simulation_capacity, save_simulation
    from vulnpriority.pipeline.stages import CAPACITY_FIELDS, capacity_from_payload, capacity_payload

    _require("vulnpriority.eval.simulation")
    simulation = importlib.import_module("vulnpriority.eval.simulation")
    context = simulation.CapacityContext(
        weeks=26,
        capacity_hours_per_week=20.0,
        n_findings=901,
        n_clusters=310,
        n_exploited=145,
        backlog_hours=9_400.0,
    )
    payload = capacity_payload(context)
    assert set(payload) == set(CAPACITY_FIELDS)
    # Derived values are not persisted: they would be free to drift from their inputs.
    assert "reachable_fraction" not in payload and "capacity_bound" not in payload

    save_simulation(
        [SimulationResult(policy=RankerName.LAMBDAMART, weeks=26, capacity_hours_per_week=20.0)],
        tmp_path,
        payload,
    )
    reloaded = load_simulation_capacity(tmp_path)
    assert reloaded == payload

    rebuilt = capacity_from_payload(reloaded)
    assert rebuilt == context
    assert rebuilt.reachable_fraction == pytest.approx(520.0 / 9_400.0)
    assert rebuilt.capacity_bound is True


def test_capacity_helpers_degrade_rather_than_fabricate() -> None:
    """A missing or unrecognisable context yields ``None``, never a half-filled object."""
    from vulnpriority.pipeline.stages import capacity_from_payload, capacity_payload

    assert capacity_payload(None) is None
    assert capacity_payload(object()) is None
    assert capacity_from_payload(None) is None
    assert capacity_from_payload({"weeks": 26}) is None, "a partial payload must not rebuild"

    sentinel = object()
    assert capacity_from_payload(sentinel) is sentinel, "an object passes straight through"


def test_run_artifacts_keep_the_capacity_beside_the_results(tmp_path: Path) -> None:
    """The whole-run save/load carries the capacity, so a resumed report still has it."""
    root = tmp_path / "run_capacity"
    payload = {
        "weeks": 26,
        "capacity_hours_per_week": 20.0,
        "n_findings": 141,
        "n_clusters": 43,
        "n_exploited": 5,
        "backlog_hours": 470.05,
    }
    artifacts = RunArtifacts(
        run_id="run_capacity",
        root=root,
        simulation=[
            SimulationResult(policy=RankerName.CVSS_ONLY, weeks=26, capacity_hours_per_week=20.0)
        ],
        simulation_capacity=payload,
    )
    artifacts.save()
    assert RunArtifacts.load(root).simulation_capacity == payload


# ---------------------------------------------------------------------------
# Freshness: a resumed run must say so
# ---------------------------------------------------------------------------


def test_the_runner_reports_what_it_recomputed_and_what_it_reused(
    config: PipelineConfig, dataset: SyntheticDataset, tmp_path: Path
) -> None:
    """A code change with an unchanged config resumes the old answer; the run must say so.

    The run id is the configuration hash, so a *configuration* change can never resume stale
    numbers - it lands in a different directory. A *code* change can, and silently. This is
    the signal that stops someone quoting a cached result as a fresh measurement.
    """
    from vulnpriority.pipeline.runner import PipelineRunner

    settings = config.model_copy(update={"output_dir": tmp_path / "runs"})
    stages = ["enrich"]

    first = PipelineRunner(config=settings, dataset=dataset)
    first.run(settings, stages=stages)
    assert first.computed == ["ingest", "assess", "enrich"]
    assert first.reused == []

    second = PipelineRunner(config=settings, dataset=dataset)
    second.run(settings, stages=stages)
    assert second.computed == []
    assert second.reused == ["ingest", "assess", "enrich"], "a resumed run reported fresh work"

    forced = PipelineRunner(config=settings, dataset=dataset)
    forced.run(settings, stages=stages, resume=False)
    assert forced.reused == []
    assert forced.computed == ["ingest", "assess", "enrich"]


def test_assess_is_skipped_when_the_enrichment_is_already_on_disk(
    config: PipelineConfig, dataset: SyntheticDataset, tmp_path: Path
) -> None:
    """Component A has no artifact of its own, so recomputing it on resume is pure waste."""
    from vulnpriority.pipeline.runner import PipelineRunner

    settings = config.model_copy(update={"output_dir": tmp_path / "runs-assess"})
    PipelineRunner(config=settings, dataset=dataset).run(settings, stages=["enrich"])

    resumed = PipelineRunner(config=settings, dataset=dataset)
    resumed.run(settings, stages=["assess"])
    assert "assess" in resumed.reused and "assess" not in resumed.computed


def test_a_stage_runs_its_dependencies_not_its_predecessors() -> None:
    """Asking for one stage must pull in what it needs, not everything ahead of it.

    ``select`` sits after ``evaluate`` in the run order but does not depend on it, and used
    to fail on a small dataset with "no usable time-ordered fold" - a message about a stage
    the user never asked for. Dependency closure, not positional prefix.
    """
    from vulnpriority.pipeline.runner import STAGE_DEPENDENCIES, STAGE_ORDER, stage_closure

    assert set(STAGE_DEPENDENCIES) == set(STAGE_ORDER)

    closure = stage_closure(["select"])
    assert "rank" in closure and "label" in closure and "enrich" in closure
    assert "evaluate" not in closure, "select does not depend on the evaluation folds"
    assert "simulate" not in closure and "adversarial" not in closure

    # Adversarial needs the assessment path only: no ranker, no labels, no folds.
    assert stage_closure(["adversarial"]) == ("ingest", "assess", "enrich", "adversarial")

    # The research report is about the evaluation, so it genuinely needs all of it.
    assert set(stage_closure(["report"])) == set(STAGE_ORDER)

    # Every closure is ordered consistently with the declared run order.
    for name in STAGE_ORDER:
        closure = stage_closure([name])
        positions = [STAGE_ORDER.index(item) for item in closure]
        assert positions == sorted(positions)
        for dependency in STAGE_DEPENDENCIES[name]:
            assert dependency in closure


def test_loading_an_empty_directory_yields_an_empty_run(tmp_path: Path) -> None:
    """Resumption must cope with a run directory that holds nothing yet."""
    empty = tmp_path / "run_empty"
    empty.mkdir()
    artifacts = RunArtifacts.load(empty)

    assert artifacts.run_id == "run_empty"
    assert artifacts.scans == [] and artifacts.enriched == []
    assert artifacts.features is None and artifacts.ranking is None and artifacts.manifest is None
    assert artifacts.written() == ()
    assert artifacts.summary()["findings"] == 0


def test_saving_twice_is_idempotent(tmp_path: Path, config: PipelineConfig, pipeline_state) -> None:
    """A resumed run rewrites its artifacts; the bytes must not drift between saves."""
    scans, enriched = pipeline_state
    root = tmp_path / "run_twice"
    artifacts = RunArtifacts(run_id="run_twice", root=root, config=config, scans=scans, enriched=enriched)

    artifacts.save()
    first = {name: artifact_path(root, name).read_bytes() for name in artifacts.written()}
    RunArtifacts.load(root).save()
    second = {name: artifact_path(root, name).read_bytes() for name in artifacts.written()}

    assert first == second
