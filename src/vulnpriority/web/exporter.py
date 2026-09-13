"""Turn a pipeline run into the website.

The site is deliberately static: one HTML file, one stylesheet, one script and one data
document. It opens from ``file://`` with no server and no network, which matters because the
framework's whole claim is that it runs offline and reproducibly. ``vulnpriority web --serve``
exists only for convenience.

The exporter is tolerant by design. A run that only ranked one scan produces a payload with
findings and nothing else, and the page hides the sections that have no data.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from vulnpriority.core.enums import Component, RankerName, TrustTier
from vulnpriority.core.money import DEFAULT_CURRENCY, format_money, format_money_compact
from vulnpriority.core.models import (
    AblationTable,
    AdversarialReport,
    AttackGraphSummary,
    ChainScore,
    EnrichedFinding,
    LabelSet,
    MetricBundle,
    RankedFinding,
    RankingResult,
    RunManifest,
    Scan,
    SelectionResult,
    SimulationResult,
)
from vulnpriority.web.schema import (
    DASHBOARD_SCHEMA_VERSION,
    DashboardData,
    WebAblation,
    WebAblationCell,
    WebAdversarial,
    WebCalibration,
    WebContribution,
    WebFinding,
    WebGapRow,
    WebGraph,
    WebGraphEdge,
    WebGraphNode,
    WebMeta,
    WebMetric,
    WebPath,
    WebScan,
    WebSelection,
    WebSimulation,
    WebSummary,
)

__all__ = ["build_dashboard", "export_site", "write_payload", "GAP_ROWS", "ASSET_DIR"]

ASSET_DIR = Path(__file__).resolve().parent / "assets"


# ---------------------------------------------------------------------------
# The traceability rows. These are prose, not measurements, so they live here
# rather than being recomputed per run; the evidence column is filled from the run.
# ---------------------------------------------------------------------------

GAP_ROWS: tuple[dict[str, Any], ...] = (
    {
        "gap_id": "Gap 1",
        "title": "Priority was defined five incompatible ways, and severity scores treat ordinal ratings as ratios.",
        "mitigation": "Priority is expected loss in money: probability of exploitation under an explicit attacker, times business impact, plus reachable-compromise contribution.",
        "modules": ["decision/expected_loss.py", "decision/impact.py", "attacker/model.py"],
        "tests": ["tests/test_decision_expected_loss.py", "tests/test_decision_impact.py"],
    },
    {
        "gap_id": "Gap 2",
        "title": "Frameworks model the vulnerability, not the adversary.",
        "mitigation": "The attacker is a configuration object with skill, resources, entry privilege, horizon and named weights; five presets ship and changing one re-prioritises without retraining.",
        "modules": ["attacker/model.py", "configs/attacker_models/"],
        "tests": ["tests/test_attacker_model.py"],
    },
    {
        "gap_id": "Gap 3",
        "title": "Models are trained and validated against CVSS labels that agree across sources only 65.9% of the time.",
        "mitigation": "CVSS can never be a label. Ground truth is KEV membership, exploit evidence, incidents or the oracle; labels are version- and source-aware.",
        "modules": ["eval/labels.py", "feeds/cvss_policy.py"],
        "tests": ["tests/test_eval_labels.py", "tests/test_feeds_cvss_policy.py"],
    },
    {
        "gap_id": "Gap 4",
        "title": "Incomparable metrics on incomparable data, with random rather than time-ordered splits.",
        "mitigation": "One frozen protocol: time-ordered splits with a gap, a fixed metric set, seven baselines on identical data, bootstrap intervals and a reproducible run manifest.",
        "modules": ["eval/splits.py", "eval/benchmark.py", "pipeline/runner.py"],
        "tests": ["tests/test_eval_splits.py", "tests/test_eval_metrics.py"],
    },
    {
        "gap_id": "Gap 5",
        "title": "Hybrid frameworks report end-to-end gains without attributing them to any component.",
        "mitigation": "Full factorial ablation over the three components across seeds, with main effects, interactions and paired intervals. Disabled components' features are dropped, not zeroed.",
        "modules": ["eval/ablation.py", "rank/features.py"],
        "tests": ["tests/test_eval_ablation.py", "tests/test_rank_features.py"],
    },
    {
        "gap_id": "Gap 6",
        "title": "Web applications are under-represented; code-level work concentrates on C and C++.",
        "mitigation": "Web-native throughout: ZAP, Burp and Nuclei ingestion, endpoint-level analysis, structural criticality from URL, method, authentication and response.",
        "modules": ["ingest/", "semantic/criticality.py"],
        "tests": ["tests/test_ingest_parsers.py", "tests/test_semantic_criticality.py"],
    },
    {
        "gap_id": "Gap 7",
        "title": "Models fail precisely on the rarest, most consequential classes.",
        "mitigation": "Cost-sensitive training weighted by monetary impact, and reporting that leads with Matthews correlation and per-class support rather than aggregate accuracy.",
        "modules": ["rank/likelihood_head.py", "eval/minority.py"],
        "tests": ["tests/test_eval_metrics.py", "tests/test_rank_lambdamart.py"],
    },
    {
        "gap_id": "Gap 8",
        "title": "Evidence rests on one or two organisations, and non-English sources are unexamined.",
        "mitigation": "Applications carry sectors, transfer is measured by leave-one-application-out, and both the token lexicon and the injection pattern library are multilingual.",
        "modules": ["eval/splits.py", "semantic/lexicon.py", "configs/sandbox/"],
        "tests": ["tests/test_eval_splits.py", "tests/test_sandbox_filter.py"],
    },
    {
        "gap_id": "Gap 9",
        "title": "Chaining sits outside the models, and business impact is almost never quantified in money.",
        "mitigation": "A directed multi-hop attack graph scores each finding by the reachable compromise it enables, in money, with monotonicity under patching provable rather than hoped for.",
        "modules": ["graph/attack_graph.py", "graph/chain_scorer.py"],
        "tests": ["tests/test_graph_build.py", "tests/test_graph_monotone.py"],
    },
    {
        "gap_id": "Gap 10",
        "title": "Evaluation stops at prediction; no study validates against remediation outcomes.",
        "mitigation": "Knapsack selection under a remediation-hour budget with unequal costs, and a longitudinal simulation measuring exposure days rather than classification accuracy.",
        "modules": ["select/knapsack.py", "eval/simulation.py"],
        "tests": ["tests/test_select_knapsack.py", "tests/test_eval_simulation.py"],
    },
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default  # NaN check without importing math


def _enum_value(value: Any, default: str = "") -> str:
    """Readable string for an enum member.

    String enums carry their label in ``value``; the integer enums (privilege, maturity)
    carry it in ``name``, and the page wants the word, not the ordinal.
    """
    if value is None:
        return default
    if isinstance(value, Enum):
        return value.value if isinstance(value.value, str) else value.name.lower()
    return str(value) or default


def _metric_key(name: Any, k: int | None) -> str:
    raw = _enum_value(name)
    return raw.replace("@k", "") if k is not None else raw


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def build_dashboard(
    *,
    scans: Sequence[Scan] = (),
    enriched: Sequence[EnrichedFinding] = (),
    chain: Mapping[str, ChainScore] | None = None,
    graphs: Sequence[AttackGraphSummary] = (),
    ranking: RankingResult | None = None,
    baseline_rankings: Mapping[str, RankingResult] | None = None,
    metrics: Sequence[MetricBundle] = (),
    ablation: AblationTable | None = None,
    selections: Sequence[SelectionResult] = (),
    simulations: Sequence[SimulationResult] = (),
    adversarial: AdversarialReport | None = None,
    labels: LabelSet | None = None,
    manifest: RunManifest | None = None,
    config: Any = None,
    generated_at: datetime | None = None,
) -> DashboardData:
    """Assemble the dashboard payload from whatever the run produced."""
    chain = dict(chain or {})
    baseline_rankings = dict(baseline_rankings or {})

    data = DashboardData()
    data.meta = _build_meta(manifest, config, generated_at)
    data.scans = [_build_scan(scan, enriched) for scan in scans]
    data.findings = _build_findings(enriched, chain, ranking, baseline_rankings, labels, selections)
    data.summary = _build_summary(scans, data.findings)
    data.graphs = [_build_graph(summary) for summary in graphs]
    data.metrics = _build_metrics(metrics)
    data.calibration = _build_calibration(metrics)
    data.ablation = _build_ablation(ablation)
    data.selections = [_build_selection(item) for item in selections]
    data.simulations = [_build_simulation(item) for item in simulations]
    data.adversarial = _build_adversarial(adversarial)
    data.gaps = _build_gaps(data)
    return data


def _currency_of(config: Any) -> str:
    """What this run's money figures are denominated in.

    ``ImpactModel.currency`` is the authority, so this asks the same resolver Component B
    asks -- inline override first, then the named preset -- rather than re-deriving it and
    risking a payload that says one thing while the numbers in it mean another. A run whose
    preset cannot be loaded at all falls back to the package default, which is what the
    numbers would have been computed with anyway.
    """
    try:
        return str(config.currency()) or DEFAULT_CURRENCY
    except Exception:
        return DEFAULT_CURRENCY


def _build_meta(manifest: RunManifest | None, config: Any, generated_at: datetime | None) -> WebMeta:
    from vulnpriority import __version__

    meta = WebMeta(
        schema_version=DASHBOARD_SCHEMA_VERSION,
        package_version=__version__,
        generated_at=generated_at or datetime.now(timezone.utc),
    )
    if manifest is not None:
        meta.run_id = manifest.run_id
        meta.config_hash = manifest.config_hash
        meta.dataset_hash = manifest.dataset_hash
        meta.llm_backend = _enum_value(manifest.llm_backend)
        meta.llm_model = manifest.llm_model
        meta.feed_mode = _enum_value(manifest.feed_mode)
        meta.seeds = list(manifest.seeds)
        meta.command = manifest.command
        meta.as_of = str(manifest.as_of or "")
    if config is not None:
        flags = config.flags()
        meta.components = {"a": flags.a, "b": flags.b, "c": flags.c}
        meta.attacker = getattr(config.component_b, "attacker_preset", "")
        meta.impact_model = getattr(config.component_b, "impact_preset", "")
        meta.currency = _currency_of(config)
        if not meta.llm_backend:
            meta.llm_backend = _enum_value(config.llm.backend)
            meta.llm_model = config.llm.model
        if not meta.feed_mode:
            meta.feed_mode = _enum_value(config.feeds.mode)
        if not meta.config_hash:
            meta.config_hash = config.hash()
    return meta


def _build_scan(scan: Scan, enriched: Sequence[EnrichedFinding]) -> WebScan:
    mine = [item for item in enriched if item.scan_id == scan.scan_id]
    return WebScan(
        scan_id=scan.scan_id,
        app_id=scan.app_id,
        app_name=scan.app_name,
        sector=scan.sector,
        scanned_at=scan.scanned_at.isoformat(),
        scanner=scan.scanner_name,
        n_endpoints=len(scan.endpoints),
        n_findings=len(scan.findings),
        total_expected_loss=sum(_f(item.expected_loss) for item in mine),
    )


def _rank_lookup(result: RankingResult | None) -> dict[str, RankedFinding]:
    if result is None:
        return {}
    return {item.finding_id: item for item in result.items}


def _guidance_lookup() -> Any:
    """``vulnpriority.report.guidance_for`` when that package is installed, else ``None``.

    Imported lazily and never required: the dashboard is complete without remediation prose,
    and a build without the report generator must still export a site.
    """
    try:
        from vulnpriority.report import guidance_for  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - absent or broken is the same thing here
        return None
    return guidance_for


def _remediation_summary(guidance_for: Any, finding: Any) -> str:
    """One line of fix advice for a finding, or "" when none is available."""
    if guidance_for is None:
        return ""
    try:
        try:
            guidance = guidance_for(finding.cwe_id, finding)
        except TypeError:
            guidance = guidance_for(finding.cwe_id)
    except Exception:  # noqa: BLE001 - advice is a nicety, not a contract
        return ""
    if guidance is None:
        return ""
    summary = getattr(guidance, "summary", None)
    if summary is None and isinstance(guidance, Mapping):
        summary = guidance.get("summary")
    return str(summary or "").strip()


def _build_findings(
    enriched: Sequence[EnrichedFinding],
    chain: Mapping[str, ChainScore],
    ranking: RankingResult | None,
    baselines: Mapping[str, RankingResult],
    labels: LabelSet | None,
    selections: Sequence[SelectionResult],
) -> list[WebFinding]:
    primary = _rank_lookup(ranking)
    guidance_for = _guidance_lookup()
    baseline_ranks: dict[str, dict[str, int]] = {}
    for name, result in baselines.items():
        for item in result.items:
            baseline_ranks.setdefault(item.finding_id, {})[_enum_value(name, name)] = item.rank

    selected: set[str] = set()
    for selection in selections:
        if selection.ranker == RankerName.LAMBDAMART or not selected:
            selected.update(selection.selected_ids)

    label_by_id = {label.finding_id: label for label in (labels.labels if labels else ())}

    out: list[WebFinding] = []
    for item in enriched:
        finding = item.finding
        ranked = primary.get(finding.finding_id)
        score = chain.get(finding.finding_id)
        intel = item.intel[0] if item.intel else None
        label = label_by_id.get(finding.finding_id)

        cvss_base = None
        cvss_version = ""
        if intel is not None and intel.cvss:
            best = max(intel.cvss, key=lambda record: (record.version.value, record.base_score))
            cvss_base = _f(best.base_score)
            cvss_version = _enum_value(best.version)

        web = WebFinding(
            finding_id=finding.finding_id,
            scan_id=finding.scan_id,
            app_id=finding.app_id,
            name=finding.name,
            description=getattr(finding.description, "text", "") or "",
            cwe_id=finding.cwe_id,
            cve_ids=list(finding.cve_ids),
            endpoint_path=item.endpoint.path,
            endpoint_method=_enum_value(item.endpoint.method),
            endpoint_function=_enum_value(item.asset.function),
            auth_required=int(item.endpoint.auth_required),
            cluster_size=finding.cluster_size,
            scanner_severity=_enum_value(finding.scanner_severity),
            cvss_base=cvss_base,
            cvss_version=cvss_version,
            epss=_f(intel.epss.score) if intel is not None and intel.epss else None,
            epss_percentile=_f(intel.epss.percentile) if intel is not None and intel.epss else None,
            kev=bool(intel is not None and intel.kev and intel.kev.in_kev),
            kev_ransomware=bool(intel is not None and intel.kev and intel.kev.known_ransomware_use),
            exploit_maturity=_enum_value(item.exploitability.exploit_maturity),
            exploit_count=len(intel.exploits) if intel is not None else 0,
            criticality=_f(item.asset.criticality),
            data_sensitivity=_f(item.asset.data_sensitivity),
            exposure=_f(item.asset.exposure),
            exploit_feasibility=_f(item.exploitability.exploit_feasibility),
            applicability=_enum_value(item.applicability.verdict),
            p_applicable=_f(item.applicability.p_applicable),
            version_match=_enum_value(item.applicability.version_match),
            p_exploit=_f(item.likelihood.p_exploit),
            impact=_f(item.impact.total),
            impact_confidentiality=_f(item.impact.confidentiality),
            impact_integrity=_f(item.impact.integrity),
            impact_availability=_f(item.impact.availability),
            impact_reputational=_f(item.impact.reputational),
            expected_loss=_f(item.expected_loss),
            likelihood_terms={k: _f(v) for k, v in item.likelihood.log_odds_terms.items()},
            chain_delta=_f(score.reach_delta) if score else 0.0,
            is_chokepoint=bool(score.is_chokepoint) if score else False,
            hops_from_entry=int(score.hops_from_entry) if score else 0,
            remediation_hours=_f(item.remediation.hours),
            remediation_summary=_remediation_summary(guidance_for, finding),
            max_tier_used=int(item.trust.max_tier_used),
            injection_signals=int(item.trust.injection_signal_count),
            selected_in_budget=finding.finding_id in selected,
        )
        web.chain_adjusted = _f(ranked.chain_adjusted_loss) if ranked else web.expected_loss + web.chain_delta
        if ranked is not None:
            web.rank = ranked.rank
            web.score = _f(ranked.score)
            web.alerts = [alert.message for alert in ranked.alerts]
            if ranked.explanation is not None:
                web.reason_codes = list(ranked.explanation.reason_codes)
                web.untrusted_influence_share = _f(ranked.explanation.untrusted_influence_share)
                web.contributions = [
                    WebContribution(
                        feature=c.feature,
                        value=_f(c.value),
                        shap=_f(c.shap_value),
                        group=_enum_value(c.group, "BASE") if c.group else "BASE",
                        tier=int(c.tier) if c.tier is not None else int(TrustTier.CURATED_FEED),
                    )
                    for c in ranked.explanation.top_contributions
                ]
        if not web.alerts:
            web.alerts = [alert.message for alert in item.alerts]
        web.ranks_by_policy = baseline_ranks.get(finding.finding_id, {})
        if label is not None:
            web.exploited = bool(label.exploited)
            web.relevance = int(label.relevance_grade)
        out.append(web)

    out.sort(key=lambda w: (w.rank if w.rank else 10**6, -w.expected_loss))
    return out


def _build_summary(scans: Sequence[Scan], findings: Sequence[WebFinding]) -> WebSummary:
    losses = sorted((f.expected_loss for f in findings), reverse=True)
    total = sum(losses)
    top_n = max(1, len(losses) // 10)
    return WebSummary(
        n_apps=len({scan.app_id for scan in scans}) or len({f.app_id for f in findings}),
        n_scans=len(scans) or len({f.scan_id for f in findings}),
        n_endpoints=sum(len(scan.endpoints) for scan in scans),
        n_findings=len(findings),
        n_clusters=len({(f.scan_id, f.name, f.cwe_id) for f in findings}),
        n_kev=sum(1 for f in findings if f.kev),
        n_exploited=sum(1 for f in findings if f.exploited),
        total_expected_loss=total,
        total_impact=sum(f.impact for f in findings),
        total_remediation_hours=sum(f.remediation_hours for f in findings),
        top_decile_loss_share=(sum(losses[:top_n]) / total) if total > 0 else 0.0,
    )


def _build_graph(summary: AttackGraphSummary) -> WebGraph:
    return WebGraph(
        scan_id=summary.scan_id,
        nodes=[
            WebGraphNode(
                id=node.node_id,
                asset=node.asset,
                privilege=int(node.privilege),
                value=_f(node.value),
                is_entry=node.is_entry,
                is_target=node.is_target,
            )
            for node in summary.nodes
        ],
        edges=[
            WebGraphEdge(
                src=edge.src,
                dst=edge.dst,
                probability=_f(edge.probability),
                finding_id=edge.finding_id,
                kind=edge.kind,
                tier=int(edge.tier),
            )
            for edge in summary.edges
        ],
        entry_node=summary.entry_node,
        target_nodes=list(summary.target_nodes),
        total_risk=_f(summary.total_risk),
        top_paths=[
            WebPath(
                nodes=list(path.nodes),
                finding_ids=list(path.finding_ids),
                probability=_f(path.probability),
                target_value=_f(path.target_value),
                expected_value=_f(path.expected_value),
            )
            for path in summary.top_paths
        ],
        monotone_verified=summary.monotone_verified,
        rejected_untrusted_edges=summary.rejected_untrusted_edges,
    )


def _build_metrics(bundles: Sequence[MetricBundle]) -> list[WebMetric]:
    out: list[WebMetric] = []
    for bundle in bundles:
        for value in bundle.values:
            out.append(
                WebMetric(
                    ranker=_enum_value(bundle.ranker),
                    metric=_metric_key(value.name, value.k),
                    k=value.k,
                    value=_f(value.value),
                    ci_low=None if value.ci_low is None else _f(value.ci_low),
                    ci_high=None if value.ci_high is None else _f(value.ci_high),
                    components=bundle.flags.label(),
                    split=_enum_value(bundle.split.kind),
                )
            )
    # One row per (ranker, metric): average across folds so the chart is readable.
    merged: dict[tuple[str, str, int | None], list[WebMetric]] = {}
    for metric in out:
        merged.setdefault((metric.ranker, metric.metric, metric.k), []).append(metric)
    final: list[WebMetric] = []
    for (ranker, name, k), group in merged.items():
        values = [m.value for m in group]
        lows = [m.ci_low for m in group if m.ci_low is not None]
        highs = [m.ci_high for m in group if m.ci_high is not None]
        final.append(
            WebMetric(
                ranker=ranker,
                metric=name,
                k=k,
                value=sum(values) / len(values),
                ci_low=min(lows) if lows else None,
                ci_high=max(highs) if highs else None,
                components=group[0].components,
                split=group[0].split,
            )
        )
    return final


def _build_calibration(bundles: Sequence[MetricBundle]) -> list[WebCalibration]:
    out: list[WebCalibration] = []
    for bundle in bundles:
        if bundle.calibration is None and bundle.minority is None:
            continue
        entry = WebCalibration(ranker=_enum_value(bundle.ranker))
        if bundle.calibration is not None:
            entry.brier = _f(bundle.calibration.brier)
            entry.ece = _f(bundle.calibration.ece)
            entry.bin_confidence = [_f(v) for v in bundle.calibration.bin_confidence]
            entry.bin_accuracy = [_f(v) for v in bundle.calibration.bin_accuracy]
            entry.bin_count = list(bundle.calibration.bin_count)
        if bundle.minority is not None:
            entry.mcc = _f(bundle.minority.mcc)
            entry.f1_positive = _f(bundle.minority.f1_positive)
            entry.balanced_accuracy = _f(bundle.minority.balanced_accuracy)
            entry.positive_rate = _f(bundle.minority.positive_rate)
            entry.per_class = {
                key: {k: _f(v) for k, v in value.items()} for key, value in bundle.minority.per_class.items()
            }
        out.append(entry)
    return out


def _build_ablation(table: AblationTable | None) -> WebAblation:
    if table is None:
        return WebAblation()
    return WebAblation(
        cells=[
            WebAblationCell(
                label=cell.flags.label(),
                a=cell.flags.a,
                b=cell.flags.b,
                c=cell.flags.c,
                mean={k: _f(v) for k, v in cell.mean.items()},
                std={k: _f(v) for k, v in cell.std.items()},
            )
            for cell in table.cells
        ],
        main_effects={k: {kk: _f(vv) for kk, vv in v.items()} for k, v in table.main_effects.items()},
        interactions={k: {kk: _f(vv) for kk, vv in v.items()} for k, v in table.interactions.items()},
        paired_ci={
            k: {kk: [_f(vv[0]), _f(vv[1])] for kk, vv in v.items()} for k, v in table.paired_ci.items()
        },
    )


def _build_selection(item: SelectionResult) -> WebSelection:
    return WebSelection(
        scan_id=item.scan_id,
        ranker=_enum_value(item.ranker),
        method=_enum_value(item.method),
        budget_hours=_f(item.budget_hours),
        n_selected=len(item.selected_ids),
        total_hours=_f(item.total_hours),
        risk_captured=_f(item.risk_captured),
        risk_capture_fraction=_f(item.risk_capture_fraction),
        exploited_captured=item.exploited_captured,
        exploited_total=item.exploited_total,
        selected_ids=list(item.selected_ids),
    )


def _build_simulation(item: SimulationResult) -> WebSimulation:
    return WebSimulation(
        policy=_enum_value(item.policy),
        weeks=item.weeks,
        capacity_hours_per_week=_f(item.capacity_hours_per_week),
        exposure_days_total=_f(item.exposure_days_total),
        exposure_days_exploited=_f(item.exposure_days_exploited),
        expected_loss_days=_f(item.expected_loss_days),
        exploited_remediated_before_exploit=item.exploited_remediated_before_exploit,
        exploited_total=item.exploited_total,
        weekly_cumulative_exposure=[_f(v) for v in item.weekly_cumulative_exposure],
        reduction_vs_reference=None if item.reduction_vs_cvss is None else _f(item.reduction_vs_cvss),
    )


def _build_adversarial(report: AdversarialReport | None) -> WebAdversarial | None:
    if report is None:
        return None
    return WebAdversarial(
        backend=_enum_value(report.backend),
        corpus_version=report.corpus_version,
        n_cases=report.n_cases,
        attack_success_rate=_f(report.attack_success_rate),
        canary_leak_rate=_f(report.canary_leak_rate),
        detection_rate=_f(report.detection_rate),
        false_positive_rate=_f(report.false_positive_rate),
        mean_abs_rank_shift=_f(report.mean_abs_rank_shift),
        max_abs_rank_shift=report.max_abs_rank_shift,
        per_category={k: {kk: _f(vv) for kk, vv in v.items()} for k, v in report.per_category.items()},
    )


def _build_gaps(data: DashboardData) -> list[WebGapRow]:
    """Fill the evidence column from the numbers this run actually produced."""
    evidence: dict[str, str] = {}

    def metric(ranker: str, name: str, k: int | None = None) -> float | None:
        for item in data.metrics:
            if item.ranker == ranker and item.metric == name and (k is None or item.k == k):
                return item.value
        return None

    learned = metric("lambdamart", "ndcg", 10)
    cvss = metric("cvss_only", "ndcg", 10)
    if learned is not None and cvss is not None:
        evidence["Gap 1"] = f"NDCG@10 {learned:.3f} against {cvss:.3f} for severity-first ordering."
        evidence["Gap 4"] = f"{len({m.ranker for m in data.metrics})} rankers evaluated on identical data."
    if data.findings:
        evidence["Gap 2"] = f"Attacker preset '{data.meta.attacker or 'default'}' applied to {len(data.findings)} findings."
        evidence["Gap 6"] = f"{data.summary.n_endpoints} web endpoints across {data.summary.n_apps} applications."
    if data.summary.n_exploited:
        evidence["Gap 3"] = f"{data.summary.n_exploited} findings labelled from confirmed exploitation, none from CVSS."
    effects = data.ablation.main_effects.get("ndcg@10") or next(iter(data.ablation.main_effects.values()), {})
    if effects:
        evidence["Gap 5"] = " ".join(f"{k} {v:+.3f}" for k, v in effects.items())
    for cal in data.calibration:
        if cal.mcc is not None:
            evidence["Gap 7"] = f"MCC {cal.mcc:.3f}, Brier {cal.brier:.3f}, expected calibration error {cal.ece:.3f}."
            break
    if data.graphs:
        risk = sum(g.total_risk for g in data.graphs)
        rejected = sum(g.rejected_untrusted_edges for g in data.graphs)
        evidence["Gap 9"] = (
            f"{format_money_compact(risk, data.meta.currency)} of reachable risk modelled; "
            f"{rejected} untrusted edges rejected."
        )
    if data.simulations:
        # Total exposure days is bounded below by remediation capacity, so it is nearly
        # identical under every policy and picking its minimum names whichever policy
        # happened to round down. Exposure on the findings that were actually exploited is
        # the quantity an ordering can move, so that is what the evidence line reports.
        best = min(data.simulations, key=lambda s: (s.exposure_days_exploited, -s.exploited_remediated_before_exploit))
        prevented = f"{best.exploited_remediated_before_exploit} of {best.exploited_total}" if best.exploited_total else "not labelled"
        evidence["Gap 10"] = (
            f"Policy '{best.policy}' left {best.exposure_days_exploited:,.0f} exposure days on findings that were "
            f"exploited, and remediated {prevented} of them before their first evidence date."
        )
    if data.scans:
        sectors = {s.sector for s in data.scans if s.sector}
        if sectors:
            evidence["Gap 8"] = f"Sectors covered: {', '.join(sorted(sectors))}."

    return [
        WebGapRow(
            gap_id=row["gap_id"],
            title=row["title"],
            mitigation=row["mitigation"],
            modules=list(row["modules"]),
            tests=list(row["tests"]),
            evidence=evidence.get(row["gap_id"], ""),
        )
        for row in GAP_ROWS
    ]


# ---------------------------------------------------------------------------
# writing the site
# ---------------------------------------------------------------------------


def write_payload(data: DashboardData, out_dir: Path) -> tuple[Path, Path]:
    """Write the payload twice: as a script for ``file://`` and as JSON for machines."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = data.model_dump(mode="json")
    text = json.dumps(payload, indent=1, ensure_ascii=False)

    json_path = out_dir / "data.json"
    json_path.write_text(text, encoding="utf-8")

    # A bare fetch() of data.json is blocked under file://, so the page loads a script instead.
    js_path = out_dir / "data.js"
    js_path.write_text(f"window.VULNPRIORITY_DATA = {text};\n", encoding="utf-8")
    return js_path, json_path


def export_site(data: DashboardData, out_dir: str | Path) -> Path:
    """Write the complete static site and return its directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copyfile(ASSET_DIR / name, out_dir / name)
    write_payload(data, out_dir)
    return out_dir
