"""Run artifacts: the typed on-disk form of everything a run produces (DESIGN.md 3.11).

A run writes ``runs/<run_id>/`` and every file in it is a *validated* document: saving goes
through ``model_dump(mode="json")`` and loading goes back through ``model_validate``. That is
not ceremony. Three things depend on it:

* **Resumability.** ``PipelineRunner`` restarts a run by reloading the stages that already
  completed. A half-typed artifact would resume into a subtly different state than the one
  that was saved, which is the worst kind of irreproducibility because it is invisible.
* **Reproducibility (Gap 4).** ``manifest.json`` records the config hash, seeds, dataset
  hash and library versions. A number in the report can be traced back to the artifact that
  produced it and to the configuration that produced that.
* **Auditability.** ``enriched.json`` carries every ``log_odds_terms`` entry and every trust
  ledger decision, so a disputed ranking can be reconstructed from disk alone.

The feature matrix is the one artifact that is not JSON: it is a CSV, because that is what a
reviewer opens. Its ablation cell is recovered on load from the columns that are present -
``feature_names_for(flags)`` is total, so the mapping from column set to
:class:`~vulnprio.core.models.ComponentFlags` is exact rather than a guess.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from vulnprio.core.config import PROJECT_ROOT, PipelineConfig
from vulnprio.core.enums import Component
from vulnprio.core.errors import ConfigError
from vulnprio.core.models import (
    FEATURE_GROUPS,
    AblationTable,
    AdversarialReport,
    AttackGraphSummary,
    ChainScore,
    ComponentFlags,
    EnrichedFinding,
    Explanation,
    FeatureFrame,
    LabelSet,
    MetricBundle,
    RankingResult,
    RunManifest,
    Scan,
    SelectionResult,
    SimulationResult,
    feature_names_for,
)

__all__ = [
    "ARTIFACT_FILES",
    "RunArtifacts",
    "artifact_path",
    "flags_from_columns",
    "save_scans",
    "load_scans",
    "save_enriched",
    "load_enriched",
    "save_chain",
    "load_chain",
    "save_features",
    "load_features",
    "save_ranking",
    "load_ranking",
    "save_labels",
    "load_labels",
    "save_metrics",
    "load_metrics",
    "save_manifest",
    "load_manifest",
    "save_selection",
    "load_selection",
    "save_simulation",
    "load_simulation",
    "load_simulation_capacity",
    "save_ablation",
    "load_ablation",
    "save_adversarial",
    "load_adversarial",
    "save_explanations",
    "load_explanations",
    "default_run_id",
    "default_run_root",
    "input_digest",
    "find_run_root",
    "new_run_id",
]

#: Artifact name to file name. The seven names in DESIGN.md 3.11 come first; the rest are
#: the outputs of the stages that document lists but does not name a file for.
ARTIFACT_FILES: dict[str, str] = {
    "scan": "scan.json",
    "enriched": "enriched.json",
    "chain": "chain.json",
    "features": "features.csv",
    "ranking": "ranking.json",
    "metrics": "metrics.json",
    "manifest": "manifest.json",
    "labels": "labels.json",
    "selection": "selection.json",
    "simulation": "simulation.json",
    "ablation": "ablation.json",
    "adversarial": "adversarial.json",
    "explanations": "explanations.json",
    "config": "config.json",
}

#: Column names the feature CSV carries in front of the feature columns themselves.
_ID_COLUMNS: tuple[str, str] = ("finding_id", "group_id")


def artifact_path(root: str | Path, name: str) -> Path:
    """Absolute path of one artifact inside a run directory."""
    if name not in ARTIFACT_FILES:
        raise ConfigError(f"unknown artifact: {name!r}; known: {sorted(ARTIFACT_FILES)}")
    return Path(root) / ARTIFACT_FILES[name]


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Scans
# ---------------------------------------------------------------------------


def save_scans(scans: Sequence[Scan], root: str | Path) -> Path:
    """Write ``scan.json``: an envelope holding every scan of the run, date-ordered."""
    ordered = sorted(scans, key=lambda scan: (scan.scanned_at, scan.scan_id))
    return _write_json(
        artifact_path(root, "scan"),
        {"scans": [scan.model_dump(mode="json") for scan in ordered]},
    )


def load_scans(root: str | Path) -> list[Scan]:
    payload = _read_json(artifact_path(root, "scan"))
    records = payload.get("scans", payload) if isinstance(payload, dict) else payload
    return [Scan.model_validate(record) for record in records]


# ---------------------------------------------------------------------------
# Enriched findings
# ---------------------------------------------------------------------------


def save_enriched(enriched: Sequence[EnrichedFinding], root: str | Path) -> Path:
    return _write_json(
        artifact_path(root, "enriched"),
        {"findings": [item.model_dump(mode="json") for item in enriched]},
    )


def load_enriched(root: str | Path) -> list[EnrichedFinding]:
    payload = _read_json(artifact_path(root, "enriched"))
    records = payload.get("findings", payload) if isinstance(payload, dict) else payload
    return [EnrichedFinding.model_validate(record) for record in records]


# ---------------------------------------------------------------------------
# Attack graph
# ---------------------------------------------------------------------------


def save_chain(
    graphs: Sequence[AttackGraphSummary],
    scores: Mapping[str, Mapping[str, ChainScore]],
    root: str | Path,
) -> Path:
    """Write ``chain.json``: the per-scan graph summaries and the per-finding chain scores."""
    return _write_json(
        artifact_path(root, "chain"),
        {
            "graphs": [graph.model_dump(mode="json") for graph in graphs],
            "scores": {
                scan_id: {
                    finding_id: score.model_dump(mode="json")
                    for finding_id, score in sorted(per_scan.items())
                }
                for scan_id, per_scan in sorted(scores.items())
            },
        },
    )


def load_chain(
    root: str | Path,
) -> tuple[list[AttackGraphSummary], dict[str, dict[str, ChainScore]]]:
    payload = _read_json(artifact_path(root, "chain"))
    graphs = [AttackGraphSummary.model_validate(item) for item in payload.get("graphs", ())]
    scores = {
        scan_id: {
            finding_id: ChainScore.model_validate(record)
            for finding_id, record in per_scan.items()
        }
        for scan_id, per_scan in (payload.get("scores") or {}).items()
    }
    return graphs, scores


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def flags_from_columns(columns: Iterable[str]) -> ComponentFlags:
    """Recover the ablation cell from the feature columns that are present.

    A component is enabled exactly when any of its columns appears, because the feature
    builder drops a disabled component's columns wholesale (never zeroes them), so presence
    is a complete and unambiguous signal.
    """
    present = {str(column) for column in columns}
    groups = {FEATURE_GROUPS.get(column) for column in present}
    return ComponentFlags(
        a=Component.A in groups, b=Component.B in groups, c=Component.C in groups
    )


def save_features(frame: FeatureFrame, root: str | Path) -> Path:
    """Write ``features.csv`` with the identifier columns in front of the feature matrix."""
    path = artifact_path(root, "features")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = frame.X.copy()
    table.insert(0, "group_id", list(frame.group_ids))
    table.insert(0, "finding_id", list(frame.finding_ids))
    table.to_csv(path, index=False, lineterminator="\n")
    return path


def load_features(root: str | Path) -> FeatureFrame:
    """Read ``features.csv`` back into a validated :class:`FeatureFrame`."""
    path = artifact_path(root, "features")
    table = pd.read_csv(path, dtype={"finding_id": str, "group_id": str})
    finding_ids = [str(value) for value in table["finding_id"]]
    group_ids = [str(value) for value in table["group_id"]]
    matrix = table.drop(columns=list(_ID_COLUMNS))
    flags = flags_from_columns(matrix.columns)
    expected = feature_names_for(flags)
    missing = [name for name in expected if name not in matrix.columns]
    if missing:
        raise ConfigError(
            f"{path} is missing feature columns for cell {flags.label()}: {missing}"
        )
    matrix = matrix[expected].astype(float)
    return FeatureFrame(X=matrix, finding_ids=finding_ids, group_ids=group_ids, flags=flags)


# ---------------------------------------------------------------------------
# Single-model artifacts
# ---------------------------------------------------------------------------


def save_ranking(ranking: RankingResult, root: str | Path) -> Path:
    return _write_json(artifact_path(root, "ranking"), ranking.model_dump(mode="json"))


def load_ranking(root: str | Path) -> RankingResult:
    return RankingResult.model_validate(_read_json(artifact_path(root, "ranking")))


def save_labels(labels: LabelSet, root: str | Path) -> Path:
    return _write_json(artifact_path(root, "labels"), labels.model_dump(mode="json"))


def load_labels(root: str | Path) -> LabelSet:
    return LabelSet.model_validate(_read_json(artifact_path(root, "labels")))


def save_manifest(manifest: RunManifest, root: str | Path) -> Path:
    return _write_json(artifact_path(root, "manifest"), manifest.model_dump(mode="json"))


def load_manifest(root: str | Path) -> RunManifest:
    return RunManifest.model_validate(_read_json(artifact_path(root, "manifest")))


def save_ablation(table: AblationTable, root: str | Path) -> Path:
    return _write_json(artifact_path(root, "ablation"), table.model_dump(mode="json"))


def load_ablation(root: str | Path) -> AblationTable:
    return AblationTable.model_validate(_read_json(artifact_path(root, "ablation")))


def save_adversarial(report: AdversarialReport, root: str | Path) -> Path:
    return _write_json(artifact_path(root, "adversarial"), report.model_dump(mode="json"))


def load_adversarial(root: str | Path) -> AdversarialReport:
    return AdversarialReport.model_validate(_read_json(artifact_path(root, "adversarial")))


# ---------------------------------------------------------------------------
# Collection artifacts
# ---------------------------------------------------------------------------


def save_metrics(bundles: Sequence[MetricBundle], root: str | Path) -> Path:
    return _write_json(
        artifact_path(root, "metrics"),
        {"bundles": [bundle.model_dump(mode="json") for bundle in bundles]},
    )


def load_metrics(root: str | Path) -> list[MetricBundle]:
    payload = _read_json(artifact_path(root, "metrics"))
    records = payload.get("bundles", payload) if isinstance(payload, dict) else payload
    return [MetricBundle.model_validate(record) for record in records]


def save_selection(results: Sequence[SelectionResult], root: str | Path) -> Path:
    return _write_json(
        artifact_path(root, "selection"),
        {"results": [item.model_dump(mode="json") for item in results]},
    )


def load_selection(root: str | Path) -> list[SelectionResult]:
    payload = _read_json(artifact_path(root, "selection"))
    records = payload.get("results", payload) if isinstance(payload, dict) else payload
    return [SelectionResult.model_validate(record) for record in records]


def save_simulation(
    results: Sequence[SimulationResult],
    root: str | Path,
    capacity: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``simulation.json``: the per-policy results and the capacity they ran under.

    The capacity travels with the results because without it they cannot be read. "11 of 145
    prevented" looks like a poor showing until you know the budget could only ever reach a
    few percent of the backlog, and a reader who has only the results has no way to find
    that out.
    """
    return _write_json(
        artifact_path(root, "simulation"),
        {
            "results": [item.model_dump(mode="json") for item in results],
            "capacity": dict(capacity) if capacity else None,
        },
    )


def load_simulation(root: str | Path) -> list[SimulationResult]:
    payload = _read_json(artifact_path(root, "simulation"))
    records = payload.get("results", payload) if isinstance(payload, dict) else payload
    return [SimulationResult.model_validate(record) for record in records]


def load_simulation_capacity(root: str | Path) -> dict[str, Any] | None:
    """The capacity context stored beside the simulation results, when there is one."""
    payload = _read_json(artifact_path(root, "simulation"))
    if not isinstance(payload, dict):
        return None
    capacity = payload.get("capacity")
    return dict(capacity) if isinstance(capacity, dict) else None


def save_explanations(explanations: Sequence[Explanation], root: str | Path) -> Path:
    return _write_json(
        artifact_path(root, "explanations"),
        {"explanations": [item.model_dump(mode="json") for item in explanations]},
    )


def load_explanations(root: str | Path) -> list[Explanation]:
    payload = _read_json(artifact_path(root, "explanations"))
    records = payload.get("explanations", payload) if isinstance(payload, dict) else payload
    return [Explanation.model_validate(record) for record in records]


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------


class RunArtifacts(BaseModel):
    """Everything one run produced, in memory, with a typed path to and from disk.

    Every collection defaults to empty rather than to ``None`` where a stage always produces
    a sequence, and to ``None`` where a stage may legitimately not have run at all. That
    distinction is what lets :meth:`written` report which stages are actually complete.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    root: Path
    config: PipelineConfig | None = None
    scans: list[Scan] = Field(default_factory=list)
    enriched: list[EnrichedFinding] = Field(default_factory=list)
    graphs: list[AttackGraphSummary] = Field(default_factory=list)
    chain_scores: dict[str, dict[str, ChainScore]] = Field(default_factory=dict)
    features: FeatureFrame | None = None
    ranking: RankingResult | None = None
    baseline_rankings: dict[str, RankingResult] = Field(default_factory=dict)
    explanations: list[Explanation] = Field(default_factory=list)
    labels: LabelSet | None = None
    metrics: list[MetricBundle] = Field(default_factory=list)
    ablation: AblationTable | None = None
    selection: list[SelectionResult] = Field(default_factory=list)
    simulation: list[SimulationResult] = Field(default_factory=list)
    #: The capacity context the simulation ran under, as a JSON-safe mapping. Kept
    #: beside the results because the results cannot be interpreted without it.
    simulation_capacity: dict[str, Any] | None = None
    adversarial: AdversarialReport | None = None
    manifest: RunManifest | None = None
    dataset_root: Path | None = None

    # -- paths --------------------------------------------------------------

    def path(self, name: str) -> Path:
        return artifact_path(self.root, name)

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def written(self) -> tuple[str, ...]:
        """Artifact names present on disk, in the declared order."""
        return tuple(name for name in ARTIFACT_FILES if self.exists(name))

    # -- saving -------------------------------------------------------------

    def save(self, root: str | Path | None = None) -> Path:
        """Write every artifact this object actually holds. Returns the run directory."""
        target = Path(root) if root is not None else Path(self.root)
        target.mkdir(parents=True, exist_ok=True)
        if self.config is not None:
            _write_json(artifact_path(target, "config"), self.config.model_dump(mode="json"))
        if self.scans:
            save_scans(self.scans, target)
        if self.enriched:
            save_enriched(self.enriched, target)
        if self.graphs or self.chain_scores:
            save_chain(self.graphs, self.chain_scores, target)
        if self.features is not None:
            save_features(self.features, target)
        if self.ranking is not None:
            save_ranking(self.ranking, target)
        if self.explanations:
            save_explanations(self.explanations, target)
        if self.labels is not None:
            save_labels(self.labels, target)
        if self.metrics:
            save_metrics(self.metrics, target)
        if self.ablation is not None:
            save_ablation(self.ablation, target)
        if self.selection:
            save_selection(self.selection, target)
        if self.simulation:
            save_simulation(self.simulation, target, self.simulation_capacity)
        if self.adversarial is not None:
            save_adversarial(self.adversarial, target)
        if self.manifest is not None:
            save_manifest(self.manifest, target)
        self.root = target
        return target

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, root: str | Path, run_id: str | None = None) -> "RunArtifacts":
        """Reload whatever a previous run left in ``root``. Missing artifacts stay empty."""
        base = Path(root)
        artifacts = cls(run_id=run_id or base.name, root=base)
        if artifacts.exists("config"):
            artifacts.config = PipelineConfig.model_validate(
                _read_json(artifact_path(base, "config"))
            )
        if artifacts.exists("scan"):
            artifacts.scans = load_scans(base)
        if artifacts.exists("enriched"):
            artifacts.enriched = load_enriched(base)
        if artifacts.exists("chain"):
            artifacts.graphs, artifacts.chain_scores = load_chain(base)
        if artifacts.exists("features"):
            artifacts.features = load_features(base)
        if artifacts.exists("ranking"):
            artifacts.ranking = load_ranking(base)
        if artifacts.exists("explanations"):
            artifacts.explanations = load_explanations(base)
        if artifacts.exists("labels"):
            artifacts.labels = load_labels(base)
        if artifacts.exists("metrics"):
            artifacts.metrics = load_metrics(base)
        if artifacts.exists("ablation"):
            artifacts.ablation = load_ablation(base)
        if artifacts.exists("selection"):
            artifacts.selection = load_selection(base)
        if artifacts.exists("simulation"):
            artifacts.simulation = load_simulation(base)
            artifacts.simulation_capacity = load_simulation_capacity(base)
        if artifacts.exists("adversarial"):
            artifacts.adversarial = load_adversarial(base)
        if artifacts.exists("manifest"):
            artifacts.manifest = load_manifest(base)
        return artifacts

    # -- reporting ----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Counts for a CLI to print, cheap enough to call after every stage."""
        return {
            "run_id": self.run_id,
            "root": str(self.root),
            "scans": len(self.scans),
            "findings": sum(len(scan.findings) for scan in self.scans),
            "enriched": len(self.enriched),
            "graphs": len(self.graphs),
            "chain_scores": sum(len(item) for item in self.chain_scores.values()),
            "features": None if self.features is None else len(self.features.X),
            "feature_columns": None if self.features is None else len(self.features.X.columns),
            "ranked": 0 if self.ranking is None else len(self.ranking.items),
            "labels": 0 if self.labels is None else len(self.labels.labels),
            "positives": 0 if self.labels is None else len(self.labels.positives()),
            "metric_bundles": len(self.metrics),
            "selections": len(self.selection),
            "simulations": len(self.simulation),
            "adversarial_cases": 0 if self.adversarial is None else self.adversarial.n_cases,
            "artifacts": list(self.written()),
        }


def default_run_root(config: PipelineConfig, run_id: str) -> Path:
    """``<output_dir>/<run_id>``, resolved against the project root when relative."""
    base = Path(config.output_dir)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return base / run_id


def input_digest(
    scan_paths: Sequence[str | Path] | None = None,
    scans: Sequence[Any] | None = None,
    dataset_dir: str | Path | None = None,
) -> str:
    """Short digest of what a run was given to work on, or ``""`` when it was given nothing.

    Files are hashed by content, not by path: the same filename holding a different scan is
    a different run, and two copies of one report are the same run wherever they sit on
    disk. In-memory scans are identified by their ``scan_id``, which the ingest layer
    already derives from the application, the timestamp and the scanner.
    """
    digest = hashlib.sha256()
    seen_anything = False

    for path in scan_paths or ():
        candidate = Path(path)
        digest.update(b"\x00scan\x00")
        seen_anything = True
        try:
            with candidate.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
        except OSError:
            # Unreadable here is not this function's problem to report - the ingest stage
            # will raise a much better error. Fold the path in so two unreadable inputs
            # still produce two different run ids.
            digest.update(str(candidate.resolve()).encode("utf-8", "replace"))

    for scan in scans or ():
        digest.update(b"\x00id\x00")
        digest.update(str(getattr(scan, "scan_id", scan)).encode("utf-8", "replace"))
        seen_anything = True

    if dataset_dir is not None:
        digest.update(b"\x00dataset\x00")
        digest.update(str(Path(dataset_dir)).encode("utf-8", "replace"))
        seen_anything = True

    return digest.hexdigest()[:12] if seen_anything else ""


def default_run_id(
    config: PipelineConfig,
    scan_paths: Sequence[str | Path] | None = None,
    scans: Sequence[Any] | None = None,
    dataset_dir: str | Path | None = None,
) -> str:
    """Deterministic run id: the configuration *and* the input it was pointed at.

    Resumption works by artifact presence inside the run directory, so the run id is the
    resume key, and a resume key that omits the input is a correctness bug rather than an
    inconvenience: two different scans run under one config share a directory, and the
    second silently reports the first one's findings. Both halves are needed - the config
    hash so that re-running the same work resumes it, and the input digest so that "the
    same work" means the same work.

    An explicit ``config.run_id`` still wins outright: a caller who names a run has said
    which directory they mean.
    """
    if config.run_id:
        return config.run_id
    digest = input_digest(scan_paths, scans, dataset_dir)
    return f"run_{config.hash()}" if not digest else f"run_{config.hash()}_{digest}"


def find_run_root(config: PipelineConfig, run_id: str | None = None) -> Path:
    """Locate the run directory a command like ``explain`` or ``manifest`` should read.

    The exact directory when it exists, otherwise the most recently written run of the same
    configuration. Run ids carry an input digest (see :func:`default_run_id`), so a command
    that was not told which scan was analysed cannot reconstruct the id - but it can still
    find the run, because every run of one configuration shares the config-hash prefix.
    Returns the exact path when nothing matches, so the caller reports a missing directory
    rather than a confusing one.
    """
    exact = default_run_root(config, run_id or default_run_id(config))
    if exact.exists():
        return exact
    prefix = f"run_{config.hash()}"
    base = exact.parent
    if not base.is_dir():
        return exact
    candidates = [
        path
        for path in base.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and (path / "config.json").exists()
    ]
    if not candidates:
        return exact
    return max(candidates, key=lambda path: path.stat().st_mtime)


def new_run_id(prefix: str = "run") -> str:
    """Timestamped run id, for a caller that explicitly wants a fresh directory."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
