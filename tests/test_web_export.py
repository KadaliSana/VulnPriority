"""The results website: payload assembly and static export.

The site is a deliverable, so it is tested like one. These tests build a small run out of
core models directly, so they hold whether or not the full pipeline ran.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    AttackComplexity,
    EndpointFunction,
    ExploitMaturity,
    FeedMode,
    LLMBackendKind,
    MetricName,
    PrivilegeLevel,
    RankerName,
    SelectionMethod,
    SplitKind,
    TrustTier,
    UserInteraction,
    VersionMatch,
)
from vulnpriority.core.models import (
    AblationCell,
    AblationTable,
    AdversarialReport,
    ApplicabilityAssessment,
    AssetCriticality,
    AttackGraphSummary,
    AttackPath,
    BusinessImpact,
    CalibrationReport,
    ChainScore,
    ComponentFlags,
    EnrichedFinding,
    Explanation,
    ExploitLikelihood,
    ExploitabilityAssessment,
    FeatureContribution,
    GraphEdge,
    GraphNode,
    GroundTruthLabel,
    LabelSet,
    MetricBundle,
    MetricValue,
    MinorityClassReport,
    RankedFinding,
    RankingResult,
    RemediationCost,
    RunManifest,
    SelectionResult,
    SimulationResult,
    Split,
    TrustSummary,
)
from vulnpriority.web.exporter import build_dashboard, export_site
from vulnpriority.web.schema import DashboardData

AS_OF = date(2024, 6, 1)


def _enriched(scan, findings) -> list[EnrichedFinding]:
    out = []
    for index, finding in enumerate(findings):
        endpoint = scan.endpoint_by_id(finding.endpoint_id)
        assert endpoint is not None
        p_exploit = [0.62, 0.18, 0.03][index % 3]
        impact = [1_400_000.0, 220_000.0, 9_000.0][index % 3]
        out.append(
            EnrichedFinding(
                finding=finding,
                endpoint=endpoint,
                intel=(),
                asset=AssetCriticality(
                    endpoint_id=endpoint.endpoint_id,
                    function=[EndpointFunction.AUTH, EndpointFunction.ADMIN, EndpointFunction.STATIC_CONTENT][index % 3],
                    criticality=[0.9, 0.75, 0.1][index % 3],
                    data_sensitivity=[0.9, 0.6, 0.0][index % 3],
                    exposure=1.0,
                ),
                exploitability=ExploitabilityAssessment(
                    finding_id=finding.finding_id,
                    exploit_feasibility=[0.8, 0.4, 0.1][index % 3],
                    exploit_maturity=[ExploitMaturity.FUNCTIONAL, ExploitMaturity.POC, ExploitMaturity.UNKNOWN][index % 3],
                    attack_complexity=AttackComplexity.LOW,
                    user_interaction=UserInteraction.NONE,
                    privilege_gained=[PrivilegeLevel.SYSTEM, PrivilegeLevel.ADMIN, PrivilegeLevel.NONE][index % 3],
                ),
                applicability=ApplicabilityAssessment(
                    finding_id=finding.finding_id,
                    verdict=ApplicabilityVerdict.APPLICABLE,
                    p_applicable=0.9,
                    version_match=VersionMatch.MATCH,
                ),
                likelihood=ExploitLikelihood(
                    finding_id=finding.finding_id,
                    attacker="opportunistic",
                    p_exploit=p_exploit,
                    p_exploit_uncapped=p_exploit,
                    horizon_days=90,
                    log_odds_terms={"kev": 2.5, "epss_logit": 0.4},
                ),
                impact=BusinessImpact(
                    finding_id=finding.finding_id,
                    confidentiality=impact * 0.7,
                    integrity=impact * 0.2,
                    availability=impact * 0.1,
                    total=impact,
                ),
                remediation=RemediationCost(finding_id=finding.finding_id, hours=[8.0, 4.0, 1.0][index % 3], cost=960.0),
                expected_loss=p_exploit * impact,
                trust=TrustSummary(max_tier_used=TrustTier.REFERENCE_PAGE, injection_signal_count=index),
                as_of=AS_OF,
            )
        )
    return out


@pytest.fixture
def run(sample_scan):
    enriched = _enriched(sample_scan, sample_scan.findings)
    chain = {
        enriched[0].finding_id: ChainScore(
            finding_id=enriched[0].finding_id, reach_delta=310_000.0,
            max_path_prob_to_target=0.44, n_paths_through=3, hops_from_entry=1,
            privilege_gain=3, is_chokepoint=True, best_target="state:shop.example.com:SYSTEM",
        ),
        enriched[1].finding_id: ChainScore(
            finding_id=enriched[1].finding_id, reach_delta=12_000.0,
            max_path_prob_to_target=0.1, hops_from_entry=2,
        ),
    }
    ranking = RankingResult(
        ranker=RankerName.LAMBDAMART,
        flags=ComponentFlags(),
        seed=42,
        items=tuple(
            RankedFinding(
                finding_id=item.finding_id,
                scan_id=item.scan_id,
                rank=index + 1,
                score=3.0 - index,
                expected_loss=item.expected_loss,
                chain_adjusted_loss=item.expected_loss + (chain.get(item.finding_id).reach_delta if item.finding_id in chain else 0.0),
                p_exploit=item.likelihood.p_exploit,
                explanation=Explanation(
                    finding_id=item.finding_id,
                    base_value=0.5,
                    top_contributions=(
                        FeatureContribution(feature="b_kev", value=1.0, shap_value=0.8, tier=TrustTier.CURATED_FEED),
                        FeatureContribution(feature="a_exploit_feasibility", value=0.8, shap_value=-0.3, tier=TrustTier.REFERENCE_PAGE),
                    ),
                    reason_codes=("KEV-listed since 2024-02-01", "chain chokepoint"),
                    untrusted_influence_share=0.22,
                ),
            )
            for index, item in enumerate(enriched)
        ),
    )
    baselines = {
        "cvss_only": RankingResult(
            ranker=RankerName.CVSS_ONLY,
            items=tuple(
                RankedFinding(finding_id=item.finding_id, scan_id=item.scan_id, rank=len(enriched) - index, score=0.0)
                for index, item in enumerate(enriched)
            ),
        )
    }
    graph = AttackGraphSummary(
        scan_id=sample_scan.scan_id,
        nodes=(
            GraphNode(node_id="state:internet:NONE", asset="internet", privilege=PrivilegeLevel.NONE, is_entry=True),
            GraphNode(node_id="state:shop.example.com:USER", asset="shop.example.com", privilege=PrivilegeLevel.USER),
            GraphNode(node_id="state:shop.example.com:SYSTEM", asset="shop.example.com", privilege=PrivilegeLevel.SYSTEM, value=1_400_000.0, is_target=True),
        ),
        edges=(
            GraphEdge(src="state:internet:NONE", dst="state:shop.example.com:USER", probability=0.62, finding_id=enriched[0].finding_id),
            GraphEdge(src="state:shop.example.com:USER", dst="state:shop.example.com:SYSTEM", probability=0.7, finding_id=enriched[1].finding_id),
        ),
        entry_node="state:internet:NONE",
        target_nodes=("state:shop.example.com:SYSTEM",),
        total_risk=607_600.0,
        top_paths=(
            AttackPath(
                nodes=("state:internet:NONE", "state:shop.example.com:USER", "state:shop.example.com:SYSTEM"),
                finding_ids=(enriched[0].finding_id, enriched[1].finding_id),
                probability=0.434,
                target_value=1_400_000.0,
            ),
        ),
        monotone_verified=True,
        rejected_untrusted_edges=2,
    )
    split = Split(
        kind=SplitKind.TIME_ORDERED, fold=0, train_scan_ids=("s0",), test_scan_ids=(sample_scan.scan_id,),
        train_end=date(2024, 4, 1), test_start=date(2024, 5, 1), gap_days=30,
    )
    bundles = [
        MetricBundle(
            ranker=RankerName.LAMBDAMART, split=split,
            values=(
                MetricValue(name=MetricName.NDCG_AT_K, k=10, value=0.81, ci_low=0.74, ci_high=0.87),
                MetricValue(name=MetricName.PRECISION_AT_K, k=10, value=0.6),
            ),
            calibration=CalibrationReport(
                brier=0.09, ece=0.04, n_bins=5,
                bin_confidence=(0.1, 0.3, 0.5, 0.7, 0.9),
                bin_accuracy=(0.08, 0.33, 0.48, 0.75, 0.88),
                bin_count=(20, 14, 9, 6, 4),
            ),
            minority=MinorityClassReport(
                positive_rate=0.12, mcc=0.55, f1_positive=0.62, balanced_accuracy=0.78,
                per_class={"maturity=WEAPONIZED": {"f1": 0.5, "support": 4.0}},
            ),
        ),
        MetricBundle(
            ranker=RankerName.CVSS_ONLY, split=split,
            values=(
                MetricValue(name=MetricName.NDCG_AT_K, k=10, value=0.52, ci_low=0.44, ci_high=0.6),
                MetricValue(name=MetricName.PRECISION_AT_K, k=10, value=0.3),
            ),
        ),
    ]
    ablation = AblationTable(
        cells=(
            AblationCell(flags=ComponentFlags(), seeds=(42,), mean={"ndcg@10": 0.81}, std={"ndcg@10": 0.02}),
            AblationCell(flags=ComponentFlags(a=False), seeds=(42,), mean={"ndcg@10": 0.74}, std={"ndcg@10": 0.02}),
            AblationCell(flags=ComponentFlags(b=False), seeds=(42,), mean={"ndcg@10": 0.62}, std={"ndcg@10": 0.03}),
            AblationCell(flags=ComponentFlags(c=False), seeds=(42,), mean={"ndcg@10": 0.77}, std={"ndcg@10": 0.02}),
        ),
        main_effects={"ndcg@10": {"A": 0.07, "B": 0.19, "C": 0.04}},
        interactions={"ndcg@10": {"AB": 0.01, "AC": -0.002, "BC": 0.006, "ABC": 0.0}},
    )
    selections = [
        SelectionResult(
            scan_id=sample_scan.scan_id, ranker=RankerName.LAMBDAMART, method=SelectionMethod.DP_EXACT,
            budget_hours=16.0, selected_ids=(enriched[0].finding_id,), total_hours=8.0,
            risk_captured=868_000.0, risk_capture_fraction=0.82, exploited_captured=1, exploited_total=1,
        ),
        SelectionResult(
            scan_id=sample_scan.scan_id, ranker=RankerName.CVSS_ONLY, method=SelectionMethod.RANK_PREFIX,
            budget_hours=16.0, selected_ids=(enriched[2].finding_id,), total_hours=1.0,
            risk_captured=270.0, risk_capture_fraction=0.01, exploited_captured=0, exploited_total=1,
        ),
    ]
    simulations = [
        SimulationResult(
            policy=RankerName.LAMBDAMART, weeks=4, capacity_hours_per_week=20.0,
            exposure_days_total=140.0, exposure_days_exploited=12.0, expected_loss_days=91000.0,
            exploited_remediated_before_exploit=1, exploited_total=1,
            weekly_cumulative_exposure=(40.0, 80.0, 115.0, 140.0), reduction_vs_cvss=0.36,
        ),
        SimulationResult(
            policy=RankerName.CVSS_ONLY, weeks=4, capacity_hours_per_week=20.0,
            exposure_days_total=219.0, exposure_days_exploited=60.0, expected_loss_days=210000.0,
            exploited_remediated_before_exploit=0, exploited_total=1,
            weekly_cumulative_exposure=(55.0, 112.0, 168.0, 219.0),
        ),
    ]
    adversarial = AdversarialReport(
        backend=LLMBackendKind.HEURISTIC, corpus_version="v1", n_cases=80,
        attack_success_rate=0.0, canary_leak_rate=0.0, detection_rate=0.93, false_positive_rate=0.05,
        mean_abs_rank_shift=0.4, max_abs_rank_shift=1,
        per_category={"instruction_override": {"n": 15.0, "attack_success_rate": 0.0, "detection_rate": 1.0}},
    )
    labels = LabelSet(
        observation_cutoff=AS_OF,
        labels=(
            GroundTruthLabel(finding_id=enriched[0].finding_id, exploited=True, relevance_grade=4),
            GroundTruthLabel(finding_id=enriched[1].finding_id, exploited=False),
        ),
    )
    manifest = RunManifest(
        run_id="run_test", created_at=datetime(2024, 6, 1, 12, 0, 0), config_hash="abc123",
        seeds=(42,), dataset_hash="def456", llm_backend=LLMBackendKind.HEURISTIC,
        llm_model="heuristic", feed_mode=FeedMode.OFFLINE, as_of=AS_OF, command="pytest",
    )
    return dict(
        scans=[sample_scan], enriched=enriched, chain=chain, graphs=[graph], ranking=ranking,
        baseline_rankings=baselines, metrics=bundles, ablation=ablation, selections=selections,
        simulations=simulations, adversarial=adversarial, labels=labels, manifest=manifest,
    )


def test_dashboard_payload_is_complete(run) -> None:
    data = build_dashboard(**run)
    assert isinstance(data, DashboardData)
    assert data.summary.n_findings == 3
    assert data.summary.total_expected_loss > 0
    assert 0.0 <= data.summary.top_decile_loss_share <= 1.0
    assert len(data.gaps) == 10
    assert data.adversarial is not None and data.adversarial.attack_success_rate == 0.0


def test_findings_carry_ranking_explanation_and_baseline_comparison(run) -> None:
    data = build_dashboard(**run)
    top = data.findings[0]
    assert top.rank == 1
    assert top.reason_codes
    assert top.contributions and top.contributions[0].feature == "b_kev"
    assert "cvss_only" in top.ranks_by_policy
    assert top.exploited is True
    assert top.chain_delta > 0 and top.is_chokepoint


def test_graph_and_paths_are_exported(run) -> None:
    data = build_dashboard(**run)
    graph = data.graphs[0]
    assert graph.monotone_verified and graph.rejected_untrusted_edges == 2
    assert graph.top_paths and len(graph.top_paths[0].nodes) == 3
    assert graph.top_paths[0].expected_value > 0


def test_metrics_and_ablation_are_exported(run) -> None:
    data = build_dashboard(**run)
    learned = [m for m in data.metrics if m.ranker == "lambdamart" and m.metric == "ndcg"]
    assert learned and learned[0].value == pytest.approx(0.81)
    assert learned[0].ci_low is not None
    assert data.ablation.main_effects["ndcg@10"]["B"] == pytest.approx(0.19)
    assert len(data.ablation.cells) == 4
    assert data.calibration and data.calibration[0].mcc == pytest.approx(0.55)


def test_gap_rows_carry_evidence_from_this_run(run) -> None:
    data = build_dashboard(**run)
    by_id = {row.gap_id: row for row in data.gaps}
    assert "0.81" in by_id["Gap 1"].evidence and "0.52" in by_id["Gap 1"].evidence
    assert by_id["Gap 5"].evidence, "ablation main effects should appear as Gap 5 evidence"
    assert by_id["Gap 9"].evidence, "graph risk should appear as Gap 9 evidence"
    assert by_id["Gap 10"].evidence, "simulation exposure should appear as Gap 10 evidence"


def test_export_writes_a_self_contained_site(run, tmp_path: Path) -> None:
    data = build_dashboard(**run)
    out = export_site(data, tmp_path / "site")
    for name in ("index.html", "styles.css", "app.js", "data.js", "data.json"):
        assert (out / name).exists(), f"missing {name}"

    html = (out / "index.html").read_text(encoding="utf-8")
    # The page must not reach for the network: it has to work from file:// offline.
    assert "http://" not in html.replace('xmlns="http://www.w3.org/', "")
    assert "https://" not in html
    assert 'src="data.js"' in html

    # data.js is a plain assignment so it loads under file:// where fetch() is blocked.
    js = (out / "data.js").read_text(encoding="utf-8")
    assert js.startswith("window.VULNPRIORITY_DATA = ")
    payload = json.loads((out / "data.json").read_text(encoding="utf-8"))
    assert payload["summary"]["n_findings"] == 3
    DashboardData.model_validate(payload)


def test_empty_run_still_exports(tmp_path: Path) -> None:
    """A run that produced nothing must still yield a readable page, not a crash."""
    data = build_dashboard()
    out = export_site(data, tmp_path / "empty")
    assert (out / "data.js").exists()
    assert data.summary.n_findings == 0
    assert len(data.gaps) == 10
