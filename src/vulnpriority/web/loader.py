"""Rebuild the dashboard payload from a run directory on disk.

The pipeline writes its artifacts as validated JSON. This module reads whatever it finds and
hands it to the exporter, so a site can be produced from a run that finished hours ago, on a
machine that never re-runs the pipeline.

It is deliberately forgiving: a run that stopped after ranking has no metrics file, and that
is a normal state, not an error. Anything unreadable is skipped and named in ``skipped``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from vulnpriority.core.models import (
    AblationTable,
    AdversarialReport,
    AttackGraphSummary,
    ChainScore,
    EnrichedFinding,
    LabelSet,
    MetricBundle,
    RankingResult,
    RunManifest,
    Scan,
    SelectionResult,
    SimulationResult,
)
from vulnpriority.web.exporter import build_dashboard
from vulnpriority.web.schema import DashboardData

__all__ = ["load_run", "load_artifacts", "RunArtifactsOnDisk"]

T = TypeVar("T", bound=BaseModel)

#: Candidate file names per artifact, tried in order. The first that parses wins.
CANDIDATES: dict[str, tuple[str, ...]] = {
    "scans": ("scans.json", "scan.json", "ingest.json"),
    "enriched": ("enriched.json", "enrichment.json"),
    "chain": ("chain.json", "chain_scores.json"),
    "graphs": ("graphs.json", "graph.json", "attack_graph.json"),
    "ranking": ("ranking.json", "rank.json"),
    "baselines": ("baselines.json", "baseline_rankings.json"),
    "metrics": ("metrics.json", "evaluation.json", "benchmark.json"),
    "ablation": ("ablation.json",),
    "selections": ("selection.json", "selections.json", "knapsack.json"),
    "simulations": ("simulation.json", "simulations.json"),
    "adversarial": ("adversarial.json", "robustness.json"),
    "labels": ("labels.json", "ground_truth.json"),
    "manifest": ("manifest.json", "run_manifest.json"),
    "config": ("config.json", "resolved_config.json"),
}


class RunArtifactsOnDisk:
    """Everything a run directory yielded, with the parts that failed recorded."""

    def __init__(self) -> None:
        self.scans: list[Scan] = []
        self.enriched: list[EnrichedFinding] = []
        self.chain: dict[str, ChainScore] = {}
        self.graphs: list[AttackGraphSummary] = []
        self.ranking: RankingResult | None = None
        self.baselines: dict[str, RankingResult] = {}
        self.metrics: list[MetricBundle] = []
        self.ablation: AblationTable | None = None
        self.selections: list[SelectionResult] = []
        self.simulations: list[SimulationResult] = []
        self.adversarial: AdversarialReport | None = None
        self.labels: LabelSet | None = None
        self.manifest: RunManifest | None = None
        self.config: Any = None
        self.explanations: list[Any] = []
        self.skipped: list[str] = []


def _read(run_dir: Path, key: str) -> Any:
    for name in CANDIDATES[key]:
        path = run_dir / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    return None


def _many(model: type[T], payload: Any, skipped: list[str], label: str) -> list[T]:
    if payload is None:
        return []
    rows = payload if isinstance(payload, list) else payload.get("items", payload.get(label, []))
    out: list[T] = []
    for row in rows if isinstance(rows, list) else []:
        try:
            out.append(model.model_validate(row))
        except ValidationError:
            skipped.append(f"{label} row")
    return out


def _one(model: type[T], payload: Any, skipped: list[str], label: str) -> T | None:
    if payload is None:
        return None
    try:
        return model.model_validate(payload)
    except ValidationError:
        skipped.append(label)
        return None


def _from_pipeline(run_dir: Path) -> RunArtifactsOnDisk | None:
    """Load through the pipeline's own artifact reader, which owns the file layout.

    Guessing at wrapper keys is how a loader silently reports an empty run: the pipeline
    writes ``{"findings": [...]}`` into ``enriched.json`` and ``{"bundles": [...]}`` into
    ``metrics.json``, and a generic reader looking for ``items`` finds neither. Ask the
    module that wrote them instead, and keep the tolerant reader below for directories that
    did not come from this pipeline.
    """
    try:
        from vulnpriority.pipeline.artifacts import RunArtifacts
    except ImportError:
        return None
    try:
        loaded = RunArtifacts.load(run_dir)
    except Exception:  # noqa: BLE001 - any failure falls back to the tolerant reader
        return None

    art = RunArtifactsOnDisk()
    art.scans = list(loaded.scans or [])
    art.enriched = list(loaded.enriched or [])
    art.chain = dict(loaded.chain_scores or {})
    art.graphs = list(loaded.graphs or [])
    art.ranking = loaded.ranking
    art.baselines = dict(loaded.baseline_rankings or {})
    art.metrics = list(loaded.metrics or [])
    art.ablation = loaded.ablation
    art.selections = list(loaded.selection or [])
    art.simulations = list(loaded.simulation or [])
    art.adversarial = loaded.adversarial
    art.labels = loaded.labels
    art.manifest = loaded.manifest
    art.config = loaded.config
    art.explanations = list(loaded.explanations or [])
    if not art.scans and not art.enriched:
        return None
    return art


def load_artifacts(run_dir: str | Path) -> RunArtifactsOnDisk:
    """Read every artifact present in ``run_dir``."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"not a run directory: {run_dir}")

    native = _from_pipeline(run_dir)
    if native is not None:
        return native

    art = RunArtifactsOnDisk()

    scans_payload = _read(run_dir, "scans")
    if isinstance(scans_payload, dict) and "scan_id" in scans_payload:
        scans_payload = [scans_payload]
    art.scans = _many(Scan, scans_payload, art.skipped, "scans")

    art.enriched = _many(EnrichedFinding, _read(run_dir, "enriched"), art.skipped, "enriched")

    chain_payload = _read(run_dir, "chain")
    if isinstance(chain_payload, dict):
        for key, value in chain_payload.items():
            try:
                art.chain[key] = ChainScore.model_validate(value)
            except ValidationError:
                art.skipped.append("chain row")
    else:
        for score in _many(ChainScore, chain_payload, art.skipped, "chain"):
            art.chain[score.finding_id] = score

    graphs_payload = _read(run_dir, "graphs")
    if isinstance(graphs_payload, dict) and "scan_id" in graphs_payload:
        graphs_payload = [graphs_payload]
    art.graphs = _many(AttackGraphSummary, graphs_payload, art.skipped, "graphs")

    art.ranking = _one(RankingResult, _read(run_dir, "ranking"), art.skipped, "ranking")

    baselines_payload = _read(run_dir, "baselines")
    if isinstance(baselines_payload, dict):
        for key, value in baselines_payload.items():
            parsed = _one(RankingResult, value, art.skipped, f"baseline {key}")
            if parsed is not None:
                art.baselines[key] = parsed
    elif isinstance(baselines_payload, list):
        for value in baselines_payload:
            parsed = _one(RankingResult, value, art.skipped, "baseline")
            if parsed is not None:
                art.baselines[str(parsed.ranker.value)] = parsed

    art.metrics = _many(MetricBundle, _read(run_dir, "metrics"), art.skipped, "metrics")
    art.ablation = _one(AblationTable, _read(run_dir, "ablation"), art.skipped, "ablation")
    art.selections = _many(SelectionResult, _read(run_dir, "selections"), art.skipped, "selections")
    art.simulations = _many(SimulationResult, _read(run_dir, "simulations"), art.skipped, "simulations")
    art.adversarial = _one(AdversarialReport, _read(run_dir, "adversarial"), art.skipped, "adversarial")
    art.labels = _one(LabelSet, _read(run_dir, "labels"), art.skipped, "labels")
    art.manifest = _one(RunManifest, _read(run_dir, "manifest"), art.skipped, "manifest")

    config_payload = _read(run_dir, "config")
    if config_payload is None and art.manifest is not None and art.manifest.config:
        config_payload = art.manifest.config
    if config_payload is not None:
        try:
            from vulnpriority.core.config import PipelineConfig

            art.config = PipelineConfig.model_validate(config_payload)
        except ValidationError:
            art.skipped.append("config")
    return art


def load_run(run_dir: str | Path) -> DashboardData:
    """Read a run directory and assemble the dashboard payload."""
    art = load_artifacts(run_dir)
    data = build_dashboard(
        scans=art.scans,
        enriched=art.enriched,
        chain=art.chain,
        graphs=art.graphs,
        ranking=art.ranking,
        baseline_rankings=art.baselines,
        metrics=art.metrics,
        ablation=art.ablation,
        selections=art.selections,
        simulations=art.simulations,
        adversarial=art.adversarial,
        labels=art.labels,
        manifest=art.manifest,
        config=art.config,
    )
    if art.skipped:
        data.notes["skipped_artifacts"] = sorted(set(art.skipped))
    data.notes["run_dir"] = str(Path(run_dir))
    return data
