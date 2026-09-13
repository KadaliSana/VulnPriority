"""``PipelineRunner``: the stages, wired, resumable, and recorded (DESIGN.md 3.11, Gap 4).

The runner owns three things the stage functions deliberately do not:

*Order.* Stages run in dependency order and each one is handed the artifacts of the ones
before it. A caller can ask for any prefix or subset by name.

*Persistence and resumption.* Each completed stage is written to ``runs/<run_id>/``
immediately, and a re-run reloads whatever is already there instead of recomputing it. The
run id defaults to the configuration hash, so the same configuration resumes itself and a
changed configuration lands in a different directory rather than silently overwriting the
evidence for a published number.

*The manifest.* ``RunManifest`` records the configuration and its hash, every seed in play,
the dataset hash, the installed version of every library whose behaviour could move a
number, the LLM backend and model, the feed mode, the as-of date and the command line. That
record is what makes a result reproducible by someone who was not there, which is the whole
of Gap 4.
"""

from __future__ import annotations

import platform
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

from vulnprio.core.config import PipelineConfig
from vulnprio.core.enums import FeedMode, LLMBackendKind, RankerName
from vulnprio.core.errors import ConfigError
from vulnprio.core.hashing import dataset_hash
from vulnprio.core.models import ComponentFlags, RunManifest, Scan
from vulnprio.ingest.normalize import DEFAULT_SCANNED_AT
from vulnprio.pipeline import stages as stage_functions
from vulnprio.pipeline.artifacts import (
    RunArtifacts,
    default_run_id,
    default_run_root,
)
from vulnprio.pipeline.stages import StageUnavailableError

__all__ = [
    "STAGE_ORDER",
    "TRACKED_PACKAGES",
    "PipelineRunner",
    "build_manifest",
    "package_versions",
]

#: Dependency order. ``label`` is not in the CLI's command list because it has no output a
#: user asks for on its own, but it is a real stage: ranking and evaluation both need it.
STAGE_ORDER: tuple[str, ...] = (
    "ingest",
    "assess",
    "enrich",
    "chain",
    "label",
    "rank",
    "evaluate",
    "select",
    "simulate",
    "adversarial",
    "report",
)

#: What each stage actually needs, as opposed to what merely precedes it in the order above.
#: Asking for one stage runs the transitive closure of this, not a positional prefix: a
#: budget selection depends on a ranking and on labels, and has no business failing because
#: the dataset was too small to cut evaluation folds from.
STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "ingest": (),
    "assess": ("ingest",),
    "enrich": ("assess",),
    "chain": ("enrich",),
    "label": ("enrich",),
    "rank": ("enrich", "chain", "label"),
    "evaluate": ("rank",),
    "select": ("rank",),
    "simulate": ("rank",),
    "adversarial": ("enrich",),
    # The research report is about the evaluation, so it genuinely needs all of it.
    "report": ("evaluate", "select", "simulate", "adversarial"),
}


def stage_closure(names: Sequence[str]) -> tuple[str, ...]:
    """Every stage ``names`` depends on, in dependency order."""
    wanted: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name in wanted:
            continue
        if name not in STAGE_DEPENDENCIES:
            raise ConfigError(f"unknown stage {name!r}; known: {list(STAGE_ORDER)}")
        wanted.add(name)
        pending.extend(STAGE_DEPENDENCIES[name])
    return tuple(name for name in STAGE_ORDER if name in wanted)


#: Libraries whose version can move a published number, and which are therefore recorded.
TRACKED_PACKAGES: tuple[str, ...] = (
    "vulnprio",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "networkx",
    "pydantic",
    "shap",
    "typer",
    "httpx",
    "matplotlib",
    "pyyaml",
    "anthropic",
)


def package_versions(names: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed version of every tracked distribution, plus the interpreter itself.

    A package that is not installed is recorded as ``"not installed"`` rather than omitted:
    "shap was absent" is as much a fact about the run as "shap was 0.45.1".
    """
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
    }
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


_LOG = logging.getLogger(__name__)

def _resolved_assessor(config: Any) -> tuple[LLMBackendKind, str, FeedMode]:
    """What the run will actually use, not what the switch was left on.

    ``llm.backend`` defaults to ``auto`` and ``llm.model`` to an Anthropic-shaped string,
    because Anthropic was the first backend. Recording those verbatim produced manifests and
    reports reading ``auto (claude-sonnet-5)`` on machines with no Anthropic key at all,
    while the run was quietly and correctly using Gemini's free tier. A manifest exists so a
    number can be traced back to what produced it; naming a model that was never called is
    the one thing it must not do.

    Resolution can touch the network (it probes for reachability), so a failure here is
    never fatal: the configured values come back instead and are at least not a fiction the
    resolver invented.
    """
    configured = (
        LLMBackendKind(config.llm.backend),
        str(config.llm.model or ""),
        FeedMode(config.feeds.mode),
    )
    try:
        from vulnprio.core.resolve import resolve_run

        run = resolve_run(config)
        backend = LLMBackendKind(run.llm.resolved)
        feeds = FeedMode(run.feeds.resolved)
    except Exception as error:  # noqa: BLE001 - provenance must not fail a run
        _LOG.debug("could not resolve the assessment backend for the manifest: %s", error)
        return configured

    model = str(config.llm.model or "")
    if backend == LLMBackendKind.GEMINI:
        from vulnprio.llm.gemini_backend import resolve_model

        model = resolve_model(model)
    elif backend == LLMBackendKind.HEURISTIC:
        # No model answers; saying so beats naming one that did not.
        model = ""
    return backend, model, feeds


def build_manifest(
    config: PipelineConfig,
    run_id: str,
    *,
    scans: Sequence[Scan] = (),
    dataset_digest: str | None = None,
    command: str | None = None,
    created_at: datetime | None = None,
) -> RunManifest:
    """Everything needed to reproduce this run."""
    seeds = tuple(
        dict.fromkeys(
            [int(config.seed), int(config.ranking.seed), int(config.synthetic.seed)]
            + [int(value) for value in config.evaluation.seeds]
        )
    )
    digest = dataset_digest or dataset_hash(
        [scan.model_dump(mode="json") for scan in scans]
    )
    as_of: date | None = None
    if config.as_of:
        as_of = date.fromisoformat(str(config.as_of))
    elif scans:
        as_of = max(scan.scanned_at.date() for scan in scans)

    # ``DEFAULT_SCANNED_AT`` is the ingest layer's "this report carried no usable
    # timestamp" sentinel. It is a fine scan id input - constant, so parsing a file twice
    # gives the same id - and a catastrophic as-of date: every feed is then asked what was
    # known in 1970, which is nothing, and the run silently has no intelligence at all
    # while reporting a cut-off of 1970-01-01. Today is the honest reading of "we do not
    # know when this was scanned", and the substitution is logged rather than hidden.
    if as_of == DEFAULT_SCANNED_AT.date():
        as_of = date.today()
        _LOG.warning(
            "the scan(s) carry no usable timestamp, so intelligence is gathered as of %s "
            "rather than the %s placeholder ingest stamped on them",
            as_of.isoformat(),
            DEFAULT_SCANNED_AT.date().isoformat(),
        )

    resolved_backend, resolved_model, resolved_feeds = _resolved_assessor(config)

    return RunManifest(
        run_id=run_id,
        created_at=created_at or datetime.now(),
        config_hash=config.hash(),
        config=config.model_dump(mode="json"),
        seeds=seeds,
        dataset_hash=digest,
        package_versions=package_versions(),
        llm_backend=resolved_backend,
        llm_model=resolved_model,
        feed_mode=resolved_feeds,
        as_of=as_of,
        command=command if command is not None else " ".join(sys.argv),
    )


@dataclass
class PipelineRunner:
    """Runs the stages of DESIGN.md 3.11 and writes their artifacts.

    ``dataset`` is the synthetic world when there is one: it supplies the feed fixture
    directory, the scans and - importantly - the exploitation oracle, which is the only
    source of the counterfactual the longitudinal simulation needs.
    """

    config: PipelineConfig = field(default_factory=PipelineConfig)
    dataset: Any | None = None
    command: str | None = None
    strict: bool = True
    errors: dict[str, str] = field(default_factory=dict)
    #: Stages actually executed by the last :meth:`run`, and stages whose artifacts were
    #: reloaded instead. A code change with an unchanged config resumes the old answer,
    #: so the run has to be able to say which numbers are fresh.
    computed: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    #: Component A's output is not an artifact of its own (it is folded into ``enriched``),
    #: so it is memoised on the runner for the duration of one call.
    _assessments: list[Any] | None = field(default=None, repr=False)
    #: Per-policy rankings, shared by the budget comparison and the longitudinal simulation.
    #: Re-ranking the same frame twice would be wasteful and, worse, would leave the two
    #: comparisons free to disagree about what "the CVSS policy" ordered.
    _policies: dict[RankerName, Any] = field(default_factory=dict, repr=False)

    # -- identity -----------------------------------------------------------

    def run_id(self, config: PipelineConfig | None = None) -> str:
        return default_run_id(config or self.config)

    def run_root(self, config: PipelineConfig | None = None) -> Path:
        resolved = config or self.config
        return default_run_root(resolved, self.run_id(resolved))

    # -- the run ------------------------------------------------------------

    def run(
        self,
        config: PipelineConfig | None = None,
        stages: Sequence[str] | None = None,
        *,
        resume: bool = True,
        scan_paths: Sequence[str | Path] | None = None,
        scans: Sequence[Scan] | None = None,
        dataset_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        flags: ComponentFlags | None = None,
        save: bool = True,
    ) -> RunArtifacts:
        """Run ``stages`` (default: all of them) and return the artifacts they produced.

        Resumption is by artifact presence: a stage whose output file already exists in the
        run directory is reloaded rather than recomputed, which makes a long run
        interruptible and makes "re-run only the evaluation" a one-line change.
        """
        resolved = config or self.config
        if flags is not None:
            resolved = resolved.with_flags(flags)
        wanted = self._plan(stages)

        # The input is part of the run's identity, not just its configuration: the run id
        # is the resume key, and two different scans under one config must not share a
        # directory and reload each other's artifacts.
        run_id = default_run_id(
            resolved, scan_paths=scan_paths, scans=scans, dataset_dir=dataset_dir
        )
        root = Path(output_dir) / run_id if output_dir is not None else default_run_root(resolved, run_id)
        root.mkdir(parents=True, exist_ok=True)

        artifacts = RunArtifacts.load(root, run_id=run_id) if resume else RunArtifacts(
            run_id=run_id, root=root
        )
        artifacts.config = resolved
        artifacts.root = root
        self.errors = {}
        self._policies = {}
        self.computed = []
        self.reused = []

        dataset = self.dataset
        if dataset is not None and getattr(dataset, "root", None) is not None:
            artifacts.dataset_root = Path(dataset.root)

        for name in STAGE_ORDER:
            if name not in wanted:
                continue
            try:
                ran = self._run_stage(
                    name,
                    resolved,
                    artifacts,
                    resume=resume,
                    scan_paths=scan_paths,
                    scans=scans,
                    dataset_dir=dataset_dir,
                    dataset=dataset,
                )
                (self.computed if ran else self.reused).append(name)
            except StageUnavailableError as error:
                self.errors[name] = str(error)
                if self.strict:
                    raise
            if save:
                artifacts.save(root)

        if artifacts.manifest is None:
            artifacts.manifest = build_manifest(
                resolved,
                run_id,
                scans=artifacts.scans,
                dataset_digest=getattr(dataset, "dataset_hash", None),
                command=self.command,
            )
        if save:
            artifacts.save(root)
        return artifacts

    # -- internals ----------------------------------------------------------

    def _policy_rankings(
        self, config: PipelineConfig, artifacts: RunArtifacts, names: Sequence[RankerName]
    ) -> dict[RankerName, Any]:
        """Rankings for ``names``, computed once per run and cached on the runner.

        The produced ranking is used as-is for its own policy rather than recomputed, so the
        budget comparison scores exactly the queue the operator was shown.
        """
        if artifacts.ranking is not None:
            self._policies.setdefault(artifacts.ranking.ranker, artifacts.ranking)
        missing = [name for name in names if name not in self._policies]
        if missing:
            self._policies.update(
                stage_functions.policy_rankings(
                    config,
                    artifacts.enriched,
                    artifacts.chain_scores,
                    policies=missing,
                    labels=artifacts.labels,
                    flags=config.flags(),
                    frame=artifacts.features,
                )
            )
        if artifacts.ranking is not None:
            artifacts.baseline_rankings = {
                name.value: result
                for name, result in self._policies.items()
                if name != artifacts.ranking.ranker
            }
        return {name: self._policies[name] for name in names if name in self._policies}

    def _plan(self, stages: Sequence[str] | None) -> tuple[str, ...]:
        """Requested stages plus everything they depend on, in dependency order."""
        if stages is None:
            return STAGE_ORDER
        unknown = [name for name in stages if name not in STAGE_ORDER]
        if unknown:
            raise ConfigError(f"unknown stage(s) {unknown}; known: {list(STAGE_ORDER)}")
        return stage_closure(stages)

    def _run_stage(
        self,
        name: str,
        config: PipelineConfig,
        artifacts: RunArtifacts,
        *,
        resume: bool,
        scan_paths: Sequence[str | Path] | None,
        scans: Sequence[Scan] | None,
        dataset_dir: str | Path | None,
        dataset: Any | None,
    ) -> bool:
        """Run one stage. Returns True when it computed, False when it reused artifacts."""
        if name == "ingest":
            if resume and artifacts.scans:
                return False
            source_scans = list(scans or ())
            if not source_scans and dataset is not None:
                source_scans = list(getattr(dataset, "scans", ()) or ())
            source_dir = dataset_dir
            if source_dir is None and dataset is not None and getattr(dataset, "root", None):
                source_dir = Path(dataset.root)
            artifacts.scans = stage_functions.ingest_stage(
                config,
                scan_paths=scan_paths,
                scans=source_scans or None,
                dataset_dir=None if source_scans else source_dir,
            )
            return True

        if name == "assess":
            if resume and self._assessments is not None:
                return False
            if resume and artifacts.enriched:
                # Component A has no artifact of its own: its output is folded into
                # enriched.json. When that is already on disk the enrich stage will reuse
                # it, so recomputing the assessments here would only be thrown away.
                return False
            self._assessments = stage_functions.assess_stage(config, artifacts.scans)
            return True

        if name == "enrich":
            if resume and artifacts.enriched:
                return False
            if self._assessments is None:
                self._assessments = stage_functions.assess_stage(config, artifacts.scans)
            artifacts.enriched = stage_functions.enrich_stage(
                config, artifacts.scans, self._assessments
            )
            return True

        if name == "chain":
            if resume and (artifacts.graphs or artifacts.chain_scores):
                return False
            graphs, scores = stage_functions.chain_stage(
                config, artifacts.scans, artifacts.enriched
            )
            artifacts.graphs, artifacts.chain_scores = graphs, scores
            return True

        if name == "label":
            if resume and artifacts.labels is not None:
                return False
            artifacts.labels = stage_functions.label_stage(
                config,
                artifacts.scans,
                artifacts.enriched,
                oracle=getattr(dataset, "oracle", None),
            )
            return True

        if name == "rank":
            if resume and artifacts.ranking is not None and artifacts.features is not None:
                return False
            frame, ranking, explanations = stage_functions.rank_stage(
                config,
                artifacts.enriched,
                artifacts.chain_scores,
                labels=artifacts.labels,
                flags=config.flags(),
            )
            artifacts.features, artifacts.ranking = frame, ranking
            artifacts.explanations = list(explanations)
            return True

        if name == "evaluate":
            if resume and artifacts.metrics:
                return False
            if artifacts.labels is None:
                raise ConfigError("evaluate needs labels; run the label stage first")
            artifacts.metrics = stage_functions.evaluate_stage(
                config,
                artifacts.scans,
                artifacts.enriched,
                artifacts.labels,
                artifacts.features,
                chain_scores=artifacts.chain_scores,
                flags=config.flags(),
            )
            return True

        if name == "select":
            if resume and artifacts.selection:
                return False
            if artifacts.ranking is None:
                raise ConfigError("select needs a ranking; run the rank stage first")
            wanted = stage_functions.selection_policies(config, artifacts.ranking)
            artifacts.selection = stage_functions.select_stage(
                config,
                artifacts.enriched,
                artifacts.ranking,
                rankings=self._policy_rankings(config, artifacts, wanted),
                policies=wanted,
                chain_scores=artifacts.chain_scores,
                labels=artifacts.labels,
            )
            return True

        if name == "simulate":
            if resume and artifacts.simulation:
                return False
            if artifacts.ranking is None or artifacts.labels is None:
                raise ConfigError("simulate needs a ranking and labels")
            # Every policy is re-ranked over the one feature matrix the rank stage built, so
            # the comparison between policies varies only the policy - and the same cache
            # serves the budget comparison in the select stage.
            policies: dict[RankerName, Any] = self._policy_rankings(
                config, artifacts, list(config.simulation.policies)
            )
            policies.setdefault(artifacts.ranking.ranker, artifacts.ranking)
            simulator = stage_functions.build_simulator()
            artifacts.simulation = stage_functions.simulate_stage(
                config,
                artifacts.enriched,
                policies,
                artifacts.labels,
                oracle=getattr(dataset, "oracle", None),
                chain_scores=artifacts.chain_scores,
                simulator=simulator,
            )
            # The capacity the run was executed under travels with the results: without it
            # "11 of 145 prevented" cannot be told apart from a bad ordering.
            artifacts.simulation_capacity = stage_functions.capacity_payload(
                getattr(simulator, "capacity", None)
            )
            return True

        if name == "adversarial":
            if resume and artifacts.adversarial is not None:
                return False
            artifacts.adversarial = stage_functions.adversarial_stage(
                config, artifacts.scans, artifacts.enriched, ranking=artifacts.ranking
            )
            return True

        if name == "report":
            artifacts.manifest = artifacts.manifest or build_manifest(
                config,
                artifacts.run_id,
                scans=artifacts.scans,
                dataset_digest=getattr(dataset, "dataset_hash", None),
                command=self.command,
            )
            stage_functions.report_stage(
                config,
                artifacts.root,
                metrics=artifacts.metrics,
                ranking=artifacts.ranking,
                enriched=artifacts.enriched,
                ablation=artifacts.ablation,
                selection=artifacts.selection,
                simulation=artifacts.simulation,
                adversarial=artifacts.adversarial,
                manifest=artifacts.manifest,
                capacity=artifacts.simulation_capacity,
            )
            return True

        raise ConfigError(f"unhandled stage {name!r}")  # pragma: no cover - guarded by _plan
