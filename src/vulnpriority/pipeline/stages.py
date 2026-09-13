"""Pure stage functions: one step of the pipeline each (DESIGN.md 3.11).

Every function here takes explicit inputs plus the configuration and returns artifacts. None
of them reads global state, writes to disk, or knows what a run directory is - that is
:mod:`vulnpriority.pipeline.runner`'s job. Keeping the two apart is what lets the CLI run a
single stage on artifacts from a previous run, and what lets the ablation drive eight cells
through the same code without eight runners.

**Ablation flags are honoured here, at the source.** With ``A`` off the semantic assessor is
constructed with no backend and no sandbox, so every assessment degrades to its deterministic
structural form; with ``B`` off the enricher's own disabled path supplies neutral priors; with
``C`` off the attack graph is not built at all and no chain score exists to be consumed. In
every case the feature builder then *drops* that component's columns rather than zeroing
them, so a disabled component cannot leak through a constant.

Every component package - ``graph``, ``rank``, ``eval``, ``select``, ``adversarial`` - is
imported inside the function that needs it, so this module (and therefore the CLI) stays
importable regardless of which packages are present, and so nothing heavy is loaded to print
a help message. An import that fails is reported as :class:`StageUnavailableError` naming the
module, rather than surfacing somewhere deep and unattributable.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vulnpriority.core.config import (
    PipelineConfig,
    load_attacker_preset,
    load_impact_preset,
)
from vulnpriority.core.enums import RankerName, SelectionMethod
from vulnpriority.core.errors import ConfigError, VulnPriorityError
from vulnpriority.core.models import (
    AdversarialReport,
    ApplicabilityAssessment,
    AssetCriticality,
    AttackGraphSummary,
    AttackerModel,
    ChainScore,
    ComponentFlags,
    Endpoint,
    EnrichedFinding,
    ExploitabilityAssessment,
    Explanation,
    FeatureFrame,
    ImpactModel,
    LabelSet,
    MetricBundle,
    RankingResult,
    Scan,
    SelectionResult,
    SimulationResult,
    Split,
    VulnIntel,
)
from vulnpriority.decision.expected_loss import chain_adjusted_loss
from vulnpriority.enrich.enricher import ContextualEnricher
from vulnpriority.feeds.bundle import DefaultIntelAssembler, build_feed_bundle
from vulnpriority.ingest.correlate import FindingCorrelator
from vulnpriority.ingest.generic import parse_scan
from vulnpriority.llm.factory import build_llm_backend
from vulnpriority.sandbox.pipeline import Sandbox
from vulnpriority.semantic.assessor import AgenticSemanticAssessor

__all__ = [
    "StageUnavailableError",
    "ScanAssessment",
    "resolve_as_of",
    "attacker_for",
    "impact_model_for",
    "intel_by_cve",
    "build_ranker",
    "ingest_stage",
    "assess_stage",
    "enrich_stage",
    "chain_stage",
    "label_stage",
    "feature_stage",
    "rank_stage",
    "scoring_labels",
    "train_ranker_stage",
    "policy_rankings",
    "split_stage",
    "evaluate_stage",
    "ablate_stage",
    "select_stage",
    "selection_policies",
    "selection_table",
    "simulate_stage",
    "build_simulator",
    "capacity_payload",
    "capacity_from_payload",
    "CAPACITY_FIELDS",
    "adversarial_stage",
    "report_stage",
]


class StageUnavailableError(VulnPriorityError):
    """A stage's implementation module or entry point is not importable."""


def _import(module: str, *names: str) -> Any:
    """Import ``module`` (or the named attributes of it) with an attributable failure."""
    try:
        loaded = importlib.import_module(module)
    except ImportError as error:  # pragma: no cover - depends on the install
        raise StageUnavailableError(f"{module} is not importable: {error}") from error
    if not names:
        return loaded
    out = []
    for name in names:
        try:
            out.append(getattr(loaded, name))
        except AttributeError as error:  # pragma: no cover - depends on the install
            raise StageUnavailableError(f"{module} does not define {name!r}") from error
    return out[0] if len(out) == 1 else tuple(out)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def resolve_as_of(config: PipelineConfig, scan: Scan | None = None) -> date:
    """As-of date for a scan: the configured override, else the scan's own date.

    Defaulting to the scan date is what makes every feed lookup answer "what was known when
    this scan ran", which is the precondition for the time-ordered protocol of DESIGN.md 4
    meaning anything at all.
    """
    if config.as_of:
        return date.fromisoformat(str(config.as_of))
    if scan is not None:
        return scan.scanned_at.date()
    return datetime.now().date()


def attacker_for(config: PipelineConfig) -> AttackerModel:
    """The adversary this run models: the inline override, else the named preset."""
    if config.component_b.attacker is not None:
        return config.component_b.attacker
    try:
        return load_attacker_preset(config.component_b.attacker_preset)
    except ConfigError:
        return AttackerModel(name=config.component_b.attacker_preset)


def impact_model_for(config: PipelineConfig) -> ImpactModel:
    """The monetary impact model: the inline override, else the named preset.

    Kept as the name every stage already calls; the resolution itself lives on
    :meth:`PipelineConfig.impact_model` so that the CLI, the exporter and the report layer
    resolve the currency the same way this stage resolves the model.
    """
    return config.impact_model()


def intel_by_cve(enriched: Sequence[EnrichedFinding]) -> dict[str, VulnIntel]:
    """CVE id to the newest assembled intelligence across the run.

    ``LabelBuilder`` wants one record per CVE; a CVE that appears in scans from March and
    September was assembled twice, and the later assembly is the one that can carry the KEV
    listing, so it wins.
    """
    out: dict[str, VulnIntel] = {}
    for item in enriched:
        for record in item.intel:
            existing = out.get(record.cve_id)
            if existing is None or record.as_of > existing.as_of:
                out[record.cve_id] = record
    return out


def _flatten_chain(
    chain_scores: Mapping[str, Mapping[str, ChainScore]] | Mapping[str, ChainScore] | None,
) -> dict[str, ChainScore]:
    """Accept per-scan chain scores or an already-flat mapping; return the flat form."""
    flat: dict[str, ChainScore] = {}
    for key, value in (chain_scores or {}).items():
        if isinstance(value, ChainScore):
            flat[key] = value
        else:
            flat.update(dict(value))
    return flat


# ---------------------------------------------------------------------------
# 1. Ingest
# ---------------------------------------------------------------------------


def ingest_stage(
    config: PipelineConfig,
    *,
    scan_paths: Sequence[str | Path] | None = None,
    scans: Sequence[Scan] | None = None,
    dataset_dir: str | Path | None = None,
    correlate: bool = True,
) -> list[Scan]:
    """Parse scanner output into correlated :class:`Scan` objects, date-ordered.

    Accepts already-parsed scans (the synthetic generator hands them over directly), report
    paths in any supported format, or a synthetic dataset directory. Correlation runs by
    default because every downstream stage assumes ``dedup_key`` and ``cluster_size`` are
    populated: the framework ranks root causes, not alerts.
    """
    collected: list[Scan] = list(scans or ())

    if dataset_dir is not None:
        directory = Path(dataset_dir)
        scans_dir = directory / "scans" if (directory / "scans").is_dir() else directory
        collected.extend(parse_scan(path) for path in sorted(scans_dir.glob("*.json")))

    for path in scan_paths or ():
        source = Path(path)
        if source.is_dir():
            collected.extend(parse_scan(item) for item in sorted(source.glob("*.json")))
        else:
            collected.append(parse_scan(source))

    if not collected:
        raise ConfigError(
            "ingest_stage was given nothing to ingest: pass scans, scan_paths or dataset_dir"
        )

    if correlate:
        correlator = FindingCorrelator()
        collected = [correlator.correlate(scan) for scan in collected]

    seen: set[str] = set()
    unique: list[Scan] = []
    for scan in sorted(collected, key=lambda item: (item.scanned_at, item.scan_id)):
        if scan.scan_id in seen:
            continue
        seen.add(scan.scan_id)
        unique.append(scan)
    return unique


# ---------------------------------------------------------------------------
# 2. Assess (Component A)
# ---------------------------------------------------------------------------


@dataclass
class ScanAssessment:
    """Everything Component A and the feeds produced for one scan."""

    scan_id: str
    as_of: date
    intel: dict[str, tuple[VulnIntel, ...]] = field(default_factory=dict)       # by finding_id
    assets: dict[str, AssetCriticality] = field(default_factory=dict)           # by endpoint_id
    exploitability: dict[str, ExploitabilityAssessment] = field(default_factory=dict)
    applicability: dict[str, ApplicabilityAssessment] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)

    def intel_for(self, finding_id: str) -> tuple[VulnIntel, ...]:
        return self.intel.get(finding_id, ())


class _StaticAssembler:
    """An assembler over a fixed ``{cve_id: VulnIntel}`` mapping.

    Used by the adversarial stage, which supplies the mutated intel directly and must not
    re-read it from the feeds - the whole point of that stage is to vary the evidence.
    """

    def __init__(self, records: Mapping[str, VulnIntel]) -> None:
        self.records = dict(records)

    def assemble(self, cve_id: str, as_of: date, max_references: int = 8) -> VulnIntel | None:
        return self.records.get(str(cve_id).strip().upper())

    def assemble_many(
        self, cve_ids: Sequence[str], as_of: date, max_references: int = 8
    ) -> tuple[VulnIntel, ...]:
        found = (self.assemble(cve_id, as_of, max_references) for cve_id in cve_ids)
        return tuple(item for item in found if item is not None)


def _build_assessor(config: PipelineConfig) -> AgenticSemanticAssessor:
    """Component A's assessor, or its structural shadow when the component is disabled.

    With ``component_a.enabled`` false the assessor is built with neither a backend nor a
    sandbox. That is not a stub: every assessment falls back to the deterministic structural
    computation, which is exactly the A=off cell the 2^3 ablation is asking about.
    """
    if not config.component_a.enabled:
        return AgenticSemanticAssessor(backend=None, sandbox=None, config=config)
    sandbox = Sandbox(config.sandbox)
    backend = build_llm_backend(config, sandbox=sandbox)
    return AgenticSemanticAssessor(backend=backend, sandbox=sandbox, config=config)


def assess_stage(
    config: PipelineConfig,
    scans: Sequence[Scan],
    *,
    as_of: date | None = None,
    assembler: Any | None = None,
) -> list[ScanAssessment]:
    """Assemble intelligence and run Component A over every finding of every scan.

    One assembler and one assessor are shared across the whole call, so a CVE seen in five
    scans is fetched once and a root cause seen on forty endpoints is assessed once.
    """
    intel_assembler = assembler
    if intel_assembler is None:
        intel_assembler = DefaultIntelAssembler(build_feed_bundle(config), config)
    assessor = _build_assessor(config)

    results: list[ScanAssessment] = []
    for scan in scans:
        cutoff = as_of or resolve_as_of(config, scan)
        assessment = ScanAssessment(scan_id=scan.scan_id, as_of=cutoff)

        for endpoint in scan.endpoints:
            assessment.assets[endpoint.endpoint_id] = assessor.assess_asset(endpoint, scan)

        for finding in scan.findings:
            intel = (
                tuple(
                    intel_assembler.assemble_many(
                        tuple(finding.cve_ids), cutoff, config.feeds.max_references_per_cve
                    )
                )
                if finding.cve_ids
                else ()
            )
            assessment.intel[finding.finding_id] = intel
            endpoint = scan.endpoint_by_id(finding.endpoint_id)
            assessment.exploitability[finding.finding_id] = assessor.assess_exploitability(
                finding, intel, endpoint
            )
            assessment.applicability[finding.finding_id] = assessor.assess_applicability(
                finding, intel, scan.tech_stack
            )
        results.append(assessment)

    stats = assessor.stats.as_dict() if hasattr(assessor.stats, "as_dict") else {}
    for assessment in results:
        assessment.stats = dict(stats)
    return results


# ---------------------------------------------------------------------------
# 3. Enrich (Component B)
# ---------------------------------------------------------------------------


def _endpoint_of(scan: Scan, endpoint_id: str) -> Endpoint:
    endpoint = scan.endpoint_by_id(endpoint_id)
    if endpoint is None:
        raise ConfigError(
            f"scan {scan.scan_id} has a finding on endpoint {endpoint_id} that the scan "
            f"does not contain; the scan document is inconsistent"
        )
    return endpoint


def enrich_stage(
    config: PipelineConfig,
    scans: Sequence[Scan],
    assessments: Sequence[ScanAssessment],
    *,
    attacker: AttackerModel | None = None,
    impact_model: ImpactModel | None = None,
) -> list[EnrichedFinding]:
    """Component B: attacker likelihood, monetary impact, remediation cost, expected loss.

    When ``component_b.enabled`` is false the enricher's own disabled path returns a
    well-formed :class:`EnrichedFinding` built from neutral constants - a flat impact and a
    fixed prior probability - so the B=off ablation cell runs through identical downstream
    code. Those constants never reach a model, because the feature builder drops the B
    columns in that cell.
    """
    enricher = ContextualEnricher(config)
    adversary = attacker or attacker_for(config)
    impact = impact_model or impact_model_for(config)
    by_scan = {item.scan_id: item for item in assessments}

    enriched: list[EnrichedFinding] = []
    for scan in scans:
        assessment = by_scan.get(scan.scan_id)
        if assessment is None:
            raise ConfigError(f"no assessment was produced for scan {scan.scan_id}")
        for finding in scan.findings:
            endpoint = _endpoint_of(scan, finding.endpoint_id)
            enriched.append(
                enricher.enrich(
                    finding,
                    endpoint,
                    assessment.intel_for(finding.finding_id),
                    assessment.assets[endpoint.endpoint_id],
                    assessment.exploitability[finding.finding_id],
                    assessment.applicability[finding.finding_id],
                    adversary,
                    impact,
                    assessment.as_of,
                )
            )
    return enriched


# ---------------------------------------------------------------------------
# 4. Chain (Component C)
# ---------------------------------------------------------------------------


def chain_stage(
    config: PipelineConfig,
    scans: Sequence[Scan],
    enriched: Sequence[EnrichedFinding],
) -> tuple[list[AttackGraphSummary], dict[str, dict[str, ChainScore]]]:
    """Build the attack graph per scan and score each finding's reachability contribution.

    With ``component_c.enabled`` false nothing is built and nothing is scored: the C=off cell
    has no chain features at all, rather than chain features pinned to zero.
    """
    if not config.component_c.enabled:
        return [], {}

    scorer_class = _import("vulnpriority.graph.chain_scorer", "ReachabilityChainScorer")
    scorer = scorer_class(config=config, attacker=attacker_for(config))

    by_scan: dict[str, list[EnrichedFinding]] = {}
    for item in enriched:
        by_scan.setdefault(item.scan_id, []).append(item)

    graphs: list[AttackGraphSummary] = []
    scores: dict[str, dict[str, ChainScore]] = {}
    for scan in scans:
        graphs.append(scorer.build(scan, by_scan.get(scan.scan_id, [])))
        scores[scan.scan_id] = dict(scorer.score(scan.scan_id) or {})
    return graphs, scores


# ---------------------------------------------------------------------------
# 5. Labels
# ---------------------------------------------------------------------------


def label_stage(
    config: PipelineConfig,
    scans: Sequence[Scan],
    enriched: Sequence[EnrichedFinding],
    *,
    oracle: Any | None = None,
    observation_cutoff: date | None = None,
) -> LabelSet:
    """Build exploitation ground truth. CVSS is never a source (Gap 3).

    ``oracle`` is the synthetic :class:`~vulnpriority.synth.oracle.ExploitationOracle` when the
    run is on generated data; on real data it is ``None`` and the labels come from KEV and
    exploit evidence alone. Its first-evidence dates are passed rather than its events, so
    the label builder never sees the latent hazard that produced them.
    """
    builder_class = _import("vulnpriority.eval.labels", "LabelBuilder")
    builder = builder_class(policy=config.evaluation.label_policy)

    cutoff = observation_cutoff or max(
        (scan.scanned_at.date() for scan in scans), default=resolve_as_of(config)
    )
    events: Any = None
    if oracle is not None:
        dates = getattr(oracle, "first_evidence_dates", None)
        events = dates() if callable(dates) else oracle
    return builder.build(
        list(enriched),
        intel_by_cve(enriched),
        events,
        observation_cutoff=cutoff,
    )


# ---------------------------------------------------------------------------
# 6. Features and ranking
# ---------------------------------------------------------------------------


def feature_stage(
    config: PipelineConfig,
    enriched: Sequence[EnrichedFinding],
    chain_scores: Mapping[str, Any] | None = None,
    *,
    flags: ComponentFlags | None = None,
) -> FeatureFrame:
    """Build the feature matrix for one ablation cell.

    Columns are ``feature_names_for(flags)`` exactly - a disabled component's columns are
    absent, not zero - and :class:`FeatureFrame`'s own validator enforces that, so an
    ablation cell physically cannot carry a disabled component's signal.
    """
    builder_class = _import("vulnpriority.rank.features", "FeatureBuilder")
    cell = flags or config.flags()
    return builder_class().build(list(enriched), _flatten_chain(chain_scores), cell)


def build_ranker(config: PipelineConfig, name: RankerName | str | None = None, seed: int | None = None) -> Any:
    """Construct the named ranker from the registry that importing ``vulnpriority.rank`` fills.

    Each ranker takes what it actually needs - the learned model takes
    :class:`~vulnpriority.core.config.RankingConfig`, the random control takes a seed, the VMC
    chain baseline takes its two published thresholds, and the rest take nothing. Building
    them here rather than passing bare names downstream is what keeps a caller from handing
    a whole :class:`PipelineConfig` to a constructor that wanted a section of it.
    """
    _import("vulnpriority.rank")  # registers the built-ins
    get_ranker = _import("vulnpriority.core.registry", "get_ranker")
    resolved = RankerName(name or config.ranking.ranker)
    cls = get_ranker(resolved)
    parameters = inspect.signature(cls).parameters
    if "epss_threshold" in parameters and "cvss_threshold" in parameters:
        return cls(
            epss_threshold=float(config.evaluation.vmc_epss_threshold),
            cvss_threshold=float(config.evaluation.vmc_cvss_threshold),
        )
    if "config" in parameters:
        return cls(config.ranking)
    if "seed" in parameters:
        return cls(seed=int(seed if seed is not None else config.ranking.seed))
    return cls()


def _relevance_and_weight(
    frame: FeatureFrame,
    enriched: Sequence[EnrichedFinding],
    labels: LabelSet,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Graded relevance per row and the cost-sensitive sample weights of Gap 7.

    Weights are ``1 + log1p(impact / 1000)``: a false negative on a payment endpoint costs
    more than one on a stylesheet, and the pairwise objective is told so explicitly rather
    than left to infer it.
    """
    grades = labels.relevance()
    relevance = np.asarray(
        [float(grades.get(finding_id, 0)) for finding_id in frame.finding_ids], dtype=float
    )
    if not config.ranking.impact_weighted_pairs:
        return relevance, None
    impact = {item.finding_id: float(item.impact.total) for item in enriched}
    weights = np.asarray(
        [1.0 + math.log1p(max(0.0, impact.get(finding_id, 0.0)) / 1000.0) for finding_id in frame.finding_ids],
        dtype=float,
    )
    return relevance, weights


def rank_stage(
    config: PipelineConfig,
    enriched: Sequence[EnrichedFinding],
    chain_scores: Mapping[str, Any] | None = None,
    *,
    labels: LabelSet | None = None,
    flags: ComponentFlags | None = None,
    ranker: RankerName | str | None = None,
    seed: int | None = None,
    frame: FeatureFrame | None = None,
    explain: bool = True,
) -> tuple[FeatureFrame, RankingResult, list[Explanation]]:
    """Build features, fit or apply the ranker, and explain the result.

    A ranker that needs fitting is fitted on the labelled rows of this frame. When no labels
    exist at all - a fresh scan of a new application, the ordinary operational case - the run
    falls back to the expected-loss ordering, which is the decision-theoretic definition of
    priority from DESIGN.md 2 and is retained as a first-class ranker precisely so that this
    fallback is a principled answer rather than an apology.
    """
    rank_scan = _import("vulnpriority.rank.compose", "rank_scan")
    cell = flags or config.flags()
    flat = _flatten_chain(chain_scores)
    matrix = frame if frame is not None else feature_stage(config, enriched, flat, flags=cell)

    resolved_seed = int(seed if seed is not None else config.ranking.seed)
    model = build_ranker(config, ranker, resolved_seed)
    if model.requires_fit():
        if labels is None or not labels.labels:
            model = build_ranker(config, RankerName.EXPECTED_LOSS, resolved_seed)
        else:
            relevance, weights = _relevance_and_weight(matrix, enriched, labels, config)
            model.fit(matrix, relevance, weights, resolved_seed)

    explainer = None
    if explain:
        explainer = _explainer_for(config, model)
    alerts = _manipulation_alerts(config, enriched, matrix, model)

    result = rank_scan(
        frame=matrix,
        ranker=model,
        enriched=list(enriched),
        chain=flat,
        config=config,
        explainer=explainer,
        alerts=alerts,
    )
    explanations = [item.explanation for item in result.items if item.explanation is not None]
    return matrix, result, explanations


def train_ranker_stage(
    config: PipelineConfig,
    enriched: Sequence[EnrichedFinding],
    labels: LabelSet,
    chain_scores: Mapping[str, Any] | None = None,
    *,
    flags: ComponentFlags | None = None,
    seed: int | None = None,
) -> tuple[Any, FeatureFrame]:
    """Fit the learned ranker on a corpus that can actually teach one, and return it.

    Separate from :func:`rank_stage` because the two want opposite things. Ranking one
    application is inference: there is a single query group and nothing to learn, and the
    right behaviour is to score with a model fitted elsewhere. Training needs the opposite
    input - several scans and labels that vary within them - and fails loudly when it does
    not have it, because a ranker that quietly did not fit is the thing this whole change
    exists to stop.
    """
    cell = flags or config.flags()
    resolved_seed = int(seed if seed is not None else config.ranking.seed)
    matrix = feature_stage(config, enriched, chain_scores, flags=cell)

    if labels is None or not labels.labels:
        raise StageUnavailableError(
            "training a ranker needs confirmed-exploitation labels and this corpus has none"
        )

    model = build_ranker(config, RankerName.LAMBDAMART, resolved_seed)
    relevance, weights = _relevance_and_weight(matrix, enriched, labels, config)
    model.fit(matrix, relevance, weights, resolved_seed)

    if getattr(model, "used_fallback", False):
        raise StageUnavailableError(
            "the ranker did not fit: "
            + "; ".join(getattr(model, "warnings", ()) or ("no reason recorded",))
            + ". Training needs at least two scans with relevance that varies inside them."
        )
    return model, matrix


def _explainer_for(config: PipelineConfig, model: Any) -> Any | None:
    """A SHAP explainer when the model has a booster to attribute over, else ``None``."""
    try:
        explainer_class, lambdamart = _import("vulnpriority.rank.explain", "ShapExplainer"), _import(
            "vulnpriority.rank.lambdamart", "LambdaMartRanker"
        )
    except StageUnavailableError:  # pragma: no cover - depends on the install
        return None
    if not isinstance(model, lambdamart):
        return None
    try:
        return explainer_class(model, config=config.ranking)
    except Exception:  # pragma: no cover - shap is optional at run time
        return None


def _manipulation_alerts(
    config: PipelineConfig, enriched: Sequence[EnrichedFinding], frame: FeatureFrame, model: Any
) -> list[Any]:
    """Rank-manipulation alerts, or nothing when the detector is unavailable."""
    try:
        detector_class = _import("vulnpriority.rank.rank_guard", "RankManipulationDetector")
    except StageUnavailableError:  # pragma: no cover - depends on the install
        return []
    detector = detector_class(config)
    return list(detector.detect(list(enriched), {"frame": frame, "ranker": model}))


def policy_rankings(
    config: PipelineConfig,
    enriched: Sequence[EnrichedFinding],
    chain_scores: Mapping[str, Any] | None = None,
    *,
    policies: Sequence[RankerName] | None = None,
    labels: LabelSet | None = None,
    flags: ComponentFlags | None = None,
    frame: FeatureFrame | None = None,
) -> dict[RankerName, RankingResult]:
    """One ranking per policy, over one shared feature matrix.

    The simulation compares policies, so they must see identical rows and identical
    preprocessing; building the frame once here is what guarantees that rather than hoping
    for it.
    """
    cell = flags or config.flags()
    matrix = frame if frame is not None else feature_stage(config, enriched, chain_scores, flags=cell)
    wanted = tuple(policies if policies is not None else config.simulation.policies)

    out: dict[RankerName, RankingResult] = {}
    for policy in wanted:
        _matrix, result, _explanations = rank_stage(
            config,
            enriched,
            chain_scores,
            labels=labels,
            flags=cell,
            ranker=policy,
            frame=matrix,
            explain=False,
        )
        out[result.ranker] = result
    return out


# ---------------------------------------------------------------------------
# 7. Splits and evaluation
# ---------------------------------------------------------------------------


def split_stage(
    config: PipelineConfig,
    scans: Sequence[Scan],
    labels: LabelSet | None = None,
    *,
    seed: int | None = None,
) -> list[Split]:
    """Folds for the configured protocol: time-ordered with a gap, by default."""
    build_splitter = _import("vulnpriority.eval.splits", "build_splitter")
    splitter = build_splitter(config.evaluation, int(seed if seed is not None else config.seed))
    return list(splitter.split(list(scans), labels))


def _frames_for_splits(
    config: PipelineConfig,
    enriched: Sequence[EnrichedFinding],
    chain: Mapping[str, ChainScore],
    splits: Sequence[Split],
    flags: ComponentFlags,
) -> dict[Split, tuple[FeatureFrame, FeatureFrame]]:
    """Train and test matrices per fold, built once and shared by every ranker.

    Folds whose train or test side is empty are dropped rather than passed on: a metric over
    zero rows is not a small number, it is not a number.
    """
    by_scan: dict[str, list[EnrichedFinding]] = {}
    for item in enriched:
        by_scan.setdefault(item.scan_id, []).append(item)

    frames: dict[Split, tuple[FeatureFrame, FeatureFrame]] = {}
    for split in splits:
        train_rows = [row for scan_id in split.train_scan_ids for row in by_scan.get(scan_id, ())]
        test_rows = [row for scan_id in split.test_scan_ids for row in by_scan.get(scan_id, ())]
        if not train_rows or not test_rows:
            continue
        frames[split] = (
            feature_stage(config, train_rows, chain, flags=flags),
            feature_stage(config, test_rows, chain, flags=flags),
        )
    return frames


_LOG = logging.getLogger(__name__)


def scoring_labels(config: PipelineConfig, labels: LabelSet) -> LabelSet:
    """The label set a model may be *graded* against, as opposed to trained on.

    ``KEV`` and ``EXPLOIT_EVIDENCE`` are both accepted ground truth and feature columns, so
    a positive justified by nothing else tells the ranker the answer in its own input row.
    Scoring against those measures a lookup. :meth:`LabelSet.for_evaluation` demotes them to
    grade 0 - demotes, not drops, so the query group keeps its size and the model earns no
    credit rather than facing an easier ranking.

    Raises when nothing independent survives, because the alternative is emitting a number
    that looks like a result and is not one. A deployment whose only ground truth is KEV
    cannot honestly evaluate a ranker that reads KEV, and should be told so.
    """
    if not config.evaluation.exclude_circular_labels:
        return labels

    scoring = labels.for_evaluation()
    surviving = sum(1 for label in scoring.labels if label.relevance_grade > 0)
    removed = len(labels.circular_labels())
    if surviving == 0 and removed > 0:
        raise ConfigError(
            f"every one of the {removed} positive label(s) is justified only by KEV or "
            "exploit evidence, both of which the ranker reads as features. Scoring against "
            "them would measure a lookup rather than a prediction, so there is no honest "
            "metric to report here. Add an independent source (an incident record, or the "
            "synthetic oracle), or set evaluation.exclude_circular_labels=false and treat "
            "the numbers as optimistic."
        )
    if removed:
        _LOG.info(
            "evaluation: %d of %d positive label(s) were justified only by feature-visible "
            "sources and are not being scored; %d independent positive(s) remain",
            removed,
            removed + surviving,
            surviving,
        )
    return scoring


def evaluate_stage(
    config: PipelineConfig,
    scans: Sequence[Scan],
    enriched: Sequence[EnrichedFinding],
    labels: LabelSet,
    features: FeatureFrame | None = None,
    *,
    chain_scores: Mapping[str, Any] | None = None,
    splits: Sequence[Split] | None = None,
    flags: ComponentFlags | None = None,
    rankers: Sequence[RankerName] | None = None,
) -> list[MetricBundle]:
    """Run the model and every baseline over identical data, splits and metrics.

    The framework's ranker and its baselines see the same rows, the same preprocessing and
    the same folds, because the review's clearest finding is that published results are not
    comparable to each other.
    """
    runner_class = _import("vulnpriority.eval.benchmark", "BenchmarkRunner")
    cell = flags or config.flags()
    chain = _flatten_chain(chain_scores)
    # Splits are cut on the full label set: which scans land in which fold is a property of
    # the data, not of what we are allowed to score. Only the grading uses the filtered view.
    folds = list(splits) if splits is not None else split_stage(config, scans, labels)
    scoring = scoring_labels(config, labels)
    frames = _frames_for_splits(config, enriched, chain, folds, cell)
    if not frames:
        raise ConfigError(
            "no usable evaluation fold: every split had an empty train or test side. "
            "Generate more scans, or lower evaluation.min_train_scans / n_folds."
        )

    wanted = list(rankers) if rankers is not None else [
        config.ranking.ranker,
        *config.evaluation.baselines,
    ]
    seen: set[RankerName] = set()
    ordered = [item for item in wanted if not (item in seen or seen.add(item))]

    # Rankers are handed over as ``(name, instance)`` pairs built here rather than as names:
    # this module owns the configuration, so it is the only place that knows a LambdaMART
    # ranker takes ``RankingConfig`` while a baseline takes nothing at all.
    resolved = [(name, build_ranker(config, name)) for name in ordered]

    runner = runner_class(
        expected_loss={item.finding_id: float(item.expected_loss) for item in enriched}
    )
    if features is not None:
        # Keeping the operational frame is not required by the benchmark, but asserting the
        # cell matches catches a caller that evaluated one ablation cell against another's
        # features, which would silently invalidate every number below.
        if features.flags != cell:
            raise ConfigError(
                f"features were built for cell {features.flags.label()} but the evaluation "
                f"is running cell {cell.label()}"
            )
    return list(runner.run(frames, scoring, resolved, config))


def ablate_stage(
    config: PipelineConfig,
    scans: Sequence[Scan],
    enriched: Sequence[EnrichedFinding],
    labels: LabelSet,
    *,
    chain_scores: Mapping[str, Any] | None = None,
    splits: Sequence[Split] | None = None,
) -> Any:
    """The full 2^3 factorial over components A, B and C (Gap 5).

    Only the *features* vary across cells: the same learner, the same folds, the same labels.
    A cell with a component off has that component's columns dropped, which is what makes the
    main effect attributable to the component rather than to a different model.
    """
    ablation_class = _import("vulnpriority.eval.ablation", "FullFactorialAblation")
    chain = _flatten_chain(chain_scores)
    folds = list(splits) if splits is not None else split_stage(config, scans, labels)

    by_scan: dict[str, list[EnrichedFinding]] = {}
    for item in enriched:
        by_scan.setdefault(item.scan_id, []).append(item)

    def build_frame_fn(
        flags: ComponentFlags, split: Split, seed: int
    ) -> tuple[FeatureFrame, FeatureFrame]:
        train_rows = [row for scan_id in split.train_scan_ids for row in by_scan.get(scan_id, ())]
        test_rows = [row for scan_id in split.test_scan_ids for row in by_scan.get(scan_id, ())]
        return (
            feature_stage(config, train_rows, chain, flags=flags),
            feature_stage(config, test_rows, chain, flags=flags),
        )

    ablation = ablation_class(
        ranker=config.ranking.ranker,
        expected_loss={item.finding_id: float(item.expected_loss) for item in enriched},
        ranker_factory=lambda _flags, seed: (
            config.ranking.ranker,
            build_ranker(config, config.ranking.ranker, seed),
        ),
    )
    usable = [
        split
        for split in folds
        if any(scan_id in by_scan for scan_id in split.train_scan_ids)
        and any(scan_id in by_scan for scan_id in split.test_scan_ids)
    ]
    if not usable:
        raise ConfigError("no usable fold for the ablation: every split was empty on one side")
    # Same circularity, same fix: the ablation's main effects are NDCG differences, and a
    # component's apparent contribution would otherwise be measured partly against labels
    # that Component B's own KEV column defines.
    return ablation.run(build_frame_fn, scoring_labels(config, labels), usable, config)


# ---------------------------------------------------------------------------
# 8. Select
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SelectionSpine:
    """The parts of a selection item that do not depend on which policy is being scored."""

    finding_id: str
    value: float
    hours: float
    dedup_key: str | None
    exploited: bool


def _selection_spine(
    config: PipelineConfig,
    rows: Sequence[EnrichedFinding],
    chain: Mapping[str, ChainScore],
    positives: set[str],
) -> list[_SelectionSpine]:
    """Value and cost per finding, computed once and reused by every policy.

    Value is recomputed here from the enrichment and the chain rather than read off a
    ``RankedFinding``, so it provably cannot vary between policies: the only thing a policy
    changes is the *order*, and a budget comparison in which the policies were also scored
    against different values would measure nothing.
    """
    weight = float(config.component_c.chain_weight)
    spine: list[_SelectionSpine] = []
    for row in rows:
        reach = float(chain[row.finding_id].reach_delta) if row.finding_id in chain else 0.0
        spine.append(
            _SelectionSpine(
                finding_id=row.finding_id,
                value=chain_adjusted_loss(float(row.expected_loss), reach, weight),
                hours=float(row.remediation.hours),
                dedup_key=row.finding.dedup_key,
                exploited=row.finding_id in positives,
            )
        )
    return spine


def selection_policies(config: PipelineConfig, ranking: RankingResult | None = None) -> list[RankerName]:
    """The policies a budget comparison covers: the framework's ranker, then the baselines."""
    head = [ranking.ranker] if ranking is not None else [RankerName(config.ranking.ranker)]
    wanted = head + [RankerName(name) for name in config.evaluation.baselines]
    seen: set[RankerName] = set()
    return [name for name in wanted if not (name in seen or seen.add(name))]


def select_stage(
    config: PipelineConfig,
    enriched: Sequence[EnrichedFinding],
    ranking: RankingResult,
    *,
    rankings: Mapping[RankerName, RankingResult] | None = None,
    chain_scores: Mapping[str, Any] | None = None,
    labels: LabelSet | None = None,
    budget_hours: float | None = None,
    policies: Sequence[RankerName] | None = None,
) -> list[SelectionResult]:
    """0/1 knapsack under a remediation budget, per scan **and per policy** (Gap 10).

    Selection is per scan because a remediation budget is spent per application per cycle;
    one pooled knapsack would let a busy application crowd out a quiet but critical one.
    The value maximised is the chain-adjusted loss and the cost is charged once per
    ``dedup_key``, because fixing a root cause fixes every endpoint it appears on.

    **It is per policy because the comparison is the result.** "57% of risk captured in 38
    hours" says nothing on its own; it means something only against what sorting by CVSS
    captures in the same 38 hours, on the same items, at the same cost. So every baseline in
    ``EvaluationConfig.baselines`` is run alongside the configured ranker over an identical
    item set, an identical budget and an identical cost model.

    The two sides use different methods, and the asymmetry is deliberate rather than a
    handicap. A baseline is an *ordering*: a team handed a CVSS-sorted queue works down it
    until the sprint is full, which is exactly ``SelectionMethod.RANK_PREFIX``. The exact
    knapsack is available to the framework only because it attaches a value *and* a cost to
    every finding - that capability is part of what is being evaluated, so pretending the
    baselines have it would flatter them, and denying it to the framework would hide the
    contribution. ``SelectionResult.method`` records which one actually ran for each row.

    Because the asymmetry is real, it is also *measured*: the configured ranker is scored a
    second time under ``RANK_PREFIX`` as well. That second row is the like-for-like control,
    and the three-way comparison separates the two contributions a reviewer will otherwise
    conflate - ordering quality (this ranker under ``rank_prefix`` against a baseline under
    ``rank_prefix``) from cost-awareness (this ranker under the knapsack against itself
    under ``rank_prefix``). Without it, "we captured more risk" is not distinguishable from
    "we spent more of the budget".
    """
    select_under_budget, item_class = _import(
        "vulnpriority.select.knapsack", "select_under_budget", "SelectionItem"
    )
    method_enum = _import("vulnpriority.core.enums", "SelectionMethod")

    budget = float(budget_hours if budget_hours is not None else config.selection.budget_hours)
    chain = _flatten_chain(chain_scores)
    positives = labels.positives() if labels is not None else set()

    wanted = list(policies) if policies is not None else selection_policies(config, ranking)
    orderings: dict[RankerName, RankingResult] = dict(rankings or {})
    orderings[ranking.ranker] = ranking            # the produced ranking always wins
    missing = [name for name in wanted if name not in orderings]
    if missing:
        orderings.update(
            policy_rankings(
                config,
                enriched,
                chain,
                policies=missing,
                labels=labels,
                flags=ranking.flags,
            )
        )
    usable = [name for name in wanted if name in orderings]

    ranks_by_policy = {
        name: {entry.finding_id: entry.rank for entry in orderings[name].items}
        for name in usable
    }

    by_scan: dict[str, list[EnrichedFinding]] = {}
    for item in enriched:
        by_scan.setdefault(item.scan_id, []).append(item)

    results: list[SelectionResult] = []
    for scan_id in sorted(by_scan):
        spine = _selection_spine(config, by_scan[scan_id], chain, positives)
        if not spine:
            continue
        for policy in usable:
            ranks = ranks_by_policy[policy]
            items = [
                item_class(
                    finding_id=entry.finding_id,
                    value=entry.value,
                    hours=entry.hours,
                    dedup_key=entry.dedup_key,
                    rank=ranks.get(entry.finding_id),
                    exploited=entry.exploited,
                )
                for entry in spine
            ]
            methods = [method_enum.RANK_PREFIX]
            if policy == ranking.ranker and config.selection.method != method_enum.RANK_PREFIX:
                # The optimiser first, then the same ordering walked as a plain queue.
                methods.insert(0, config.selection.method)
            for method in methods:
                results.append(
                    select_under_budget(
                        items,
                        budget_hours=budget,
                        config=config,
                        scan_id=scan_id,
                        ranker=policy,
                        method=method,
                        labels=labels,
                    )
                )
    return results


def selection_table(results: Sequence[SelectionResult]) -> list[dict[str, Any]]:
    """Aggregate per-scan selections into one row per (policy, mode): the Gap 10 headline.

    Rows are keyed on the policy and on whether the budget was *optimised* or merely walked
    in rank order, never on the exact method name: ``dp_exact`` downgrading to
    ``greedy_ratio`` on one oversized scan is still the optimiser, and splitting that policy
    into two half-populated rows would make the shares incomparable. The method names that
    actually ran are reported alongside, so nothing is hidden by the grouping.

    Risk-captured share is pooled rather than averaged over scans - total captured over
    total available - so a scan with a hundred findings is not given the same weight as one
    with three. Every policy saw the same items on every scan, so the denominator is shared
    and the shares are directly comparable.
    """
    optimiser = SelectionMethod.RANK_PREFIX
    grouped: dict[tuple[RankerName, bool], list[SelectionResult]] = {}
    order: list[tuple[RankerName, bool]] = []
    available: dict[str, float] = {}
    for item in results:
        key = (item.ranker, item.method != optimiser)
        if key not in grouped:
            order.append(key)
        grouped.setdefault(key, []).append(item)
        if item.risk_capture_fraction > 0.0:
            total = item.risk_captured / item.risk_capture_fraction
            available[item.scan_id] = max(available.get(item.scan_id, 0.0), total)

    rows: list[dict[str, Any]] = []
    for policy, optimised in order:
        entries = grouped[(policy, optimised)]
        captured = sum(item.risk_captured for item in entries)
        pool = sum(available.get(item.scan_id, 0.0) for item in entries)
        exploited_caught = sum(item.exploited_captured for item in entries)
        exploited_total = sum(item.exploited_total for item in entries)
        budget = sum(item.budget_hours for item in entries)
        hours = sum(item.total_hours for item in entries)
        rows.append(
            {
                "policy": policy,
                "optimised": optimised,
                "method": sorted({item.method.value for item in entries}),
                "scans": len(entries),
                "items": sum(len(item.selected_ids) for item in entries),
                "hours": hours,
                "budget_hours": budget,
                "budget_used": (hours / budget) if budget > 0.0 else 0.0,
                "risk_captured": captured,
                "risk_capture_share": (captured / pool) if pool > 0.0 else 0.0,
                "exploited_captured": exploited_caught,
                "exploited_total": exploited_total,
                "exploited_share": (exploited_caught / exploited_total) if exploited_total else 0.0,
            }
        )
    rows.sort(key=lambda row: (-row["risk_capture_share"], row["policy"].value))
    return rows


# ---------------------------------------------------------------------------
# 9. Simulate
# ---------------------------------------------------------------------------

#: Constructor fields of ``eval.simulation.CapacityContext``. Its other attributes are
#: derived properties, so these are what a round trip has to carry.
CAPACITY_FIELDS: tuple[str, ...] = (
    "weeks",
    "capacity_hours_per_week",
    "n_findings",
    "n_clusters",
    "n_exploited",
    "backlog_hours",
)


def build_simulator() -> Any:
    """A fresh :class:`LongitudinalSimulator`, whose ``capacity`` the caller can read after."""
    return _import("vulnpriority.eval.simulation", "LongitudinalSimulator")()


def capacity_payload(capacity: Any | None) -> dict[str, Any] | None:
    """JSON-safe form of a capacity context, or ``None``.

    Only the constructor fields are stored: ``reachable_fraction`` and ``capacity_bound`` are
    computed from them, and persisting a derived value invites it to drift out of agreement
    with what it was derived from.
    """
    if capacity is None:
        return None
    if isinstance(capacity, Mapping):
        return {key: capacity[key] for key in CAPACITY_FIELDS if key in capacity} or None
    values = {
        key: getattr(capacity, key) for key in CAPACITY_FIELDS if hasattr(capacity, key)
    }
    return values or None


def capacity_from_payload(payload: Any | None) -> Any | None:
    """Rebuild a capacity context from :func:`capacity_payload`, for a resumed run.

    A context that is already an object passes straight through, and a payload that cannot
    be rebuilt returns ``None`` rather than a half-populated object: the report degrades to
    "not supplied", which is honest, where a fabricated backlog would not be.
    """
    if payload is None or not isinstance(payload, Mapping):
        return payload
    try:
        context_class = _import("vulnpriority.eval.simulation", "CapacityContext")
        return context_class(**{key: payload[key] for key in CAPACITY_FIELDS})
    except (StageUnavailableError, KeyError, TypeError):  # pragma: no cover - shape drift
        return None


def simulate_stage(
    config: PipelineConfig,
    enriched: Sequence[EnrichedFinding],
    rankings: Mapping[RankerName, RankingResult] | RankingResult,
    labels: LabelSet,
    *,
    oracle: Any | None = None,
    chain_scores: Mapping[str, Any] | None = None,
    simulator: Any | None = None,
) -> list[SimulationResult]:
    """26-week exposure simulation per policy: exposure days, not prediction accuracy.

    Pass a ``simulator`` from :func:`build_simulator` to keep hold of it: after the run it
    carries the :class:`CapacityContext` the results were produced under, and
    :func:`capacity_payload` turns that into the mapping the report and the artifacts store.

    The Gap 10 counterfactual - "would this have been exploited had we not fixed it by date
    D" - reaches the simulator through ``labels``: :func:`label_stage` writes the oracle's
    first-evidence date onto each :class:`~vulnpriority.core.models.GroundTruthLabel`, and the
    simulator compares it against the day the policy closed the finding. Routing it that way
    rather than handing the oracle over directly is deliberate: the simulator sees an
    observable date, never the latent hazard that produced it. ``oracle`` is still accepted
    and forwarded when the simulator declares a parameter for it.
    """
    policies: dict[RankerName, RankingResult]
    if isinstance(rankings, RankingResult):
        policies = {rankings.ranker: rankings}
    else:
        policies = dict(rankings)
    if not policies:
        raise ConfigError("simulate_stage was given no policy to simulate")

    simulator = simulator if simulator is not None else build_simulator()
    parameters = inspect.signature(simulator.run).parameters
    extra: dict[str, Any] = {}
    if oracle is not None and "oracle" in parameters:
        extra["oracle"] = oracle
    if chain_scores is not None and "chain" in parameters:
        extra["chain"] = _flatten_chain(chain_scores)
    return list(simulator.run(list(enriched), labels, policies, config, **extra))


# ---------------------------------------------------------------------------
# 10. Adversarial
# ---------------------------------------------------------------------------


def _pipeline_probe(config: PipelineConfig) -> Callable[[Scan, Mapping[str, VulnIntel]], dict[str, Any]]:
    """The callable the adversarial evaluator runs once clean and once per payload.

    It is the real Component A and Component B path - sandbox, guarded backend, influence
    budgets and all - over whichever scan and intel it is handed. The ordering it returns is
    the expected-loss ordering, which needs no fitted model and is therefore identical
    between the clean and injected runs except for what the payload actually moved.
    """

    def probe(scan: Scan, intel: Mapping[str, VulnIntel]) -> dict[str, Any]:
        assessments = assess_stage(
            config, [scan], assembler=_StaticAssembler(intel), as_of=resolve_as_of(config, scan)
        )
        enriched = enrich_stage(config, [scan], assessments)
        assessment = assessments[0]
        ordered = sorted(
            enriched, key=lambda item: (-float(item.expected_loss), item.finding_id)
        )
        canary_leaked = any(item.trust.canary_leaked for item in enriched)
        return {
            "order": tuple(item.finding_id for item in ordered),
            "feasibility": {
                finding_id: float(value.exploit_feasibility)
                for finding_id, value in assessment.exploitability.items()
            },
            "criticality": {
                item.finding_id: float(item.asset.criticality) for item in enriched
            },
            "p_exploit": {
                item.finding_id: float(item.likelihood.p_exploit) for item in enriched
            },
            "canary_leaked": canary_leaked,
            "alerts": tuple(alert for item in enriched for alert in item.alerts),
        }

    return probe


def adversarial_stage(
    config: PipelineConfig,
    scans: Sequence[Scan],
    enriched: Sequence[EnrichedFinding],
    *,
    ranking: RankingResult | None = None,
    corpus_path: str | Path | None = None,
    scan: Scan | None = None,
) -> AdversarialReport:
    """Clean run against injected run over the injection corpus (DESIGN.md 3.10).

    One scan is used as the fixture - the same application, the same CVEs, once clean and
    once per payload - because the report compares *cases*, and varying the application at
    the same time would confound the comparison.
    """
    evaluator_class = _import("vulnpriority.adversarial.evaluator", "AdversarialEvaluator")
    load_corpus = _import("vulnpriority.adversarial.corpus", "load_corpus")

    fixture = scan or (scans[0] if scans else None)
    if fixture is None:
        raise ConfigError("adversarial_stage needs at least one scan to use as the fixture")

    version, cases = load_corpus(corpus_path or config.adversarial.corpus_path)
    intel = intel_by_cve([item for item in enriched if item.scan_id == fixture.scan_id])
    evaluator = evaluator_class(
        fixture,
        intel,
        sandbox_config=config.sandbox,
        backend=config.llm.backend,
    )
    return evaluator.run(
        _pipeline_probe(config), cases, config.adversarial, corpus_version=version
    )


# ---------------------------------------------------------------------------
# 11. Report
# ---------------------------------------------------------------------------


def report_stage(
    config: PipelineConfig,
    output_dir: str | Path,
    *,
    metrics: Sequence[MetricBundle] = (),
    ranking: RankingResult | None = None,
    enriched: Sequence[EnrichedFinding] = (),
    ablation: Any | None = None,
    selection: Sequence[SelectionResult] = (),
    simulation: Sequence[SimulationResult] = (),
    adversarial: AdversarialReport | None = None,
    manifest: Any | None = None,
    capacity: Any | None = None,
) -> list[Path]:
    """Write ``report.md``, ``report.json`` and the figures, returning the paths written.

    ``capacity`` is the simulation's :class:`CapacityContext` (or the mapping
    :func:`capacity_payload` produced from it). Without it the report can only say "backlog:
    not supplied", and a reader seeing "11 of 145 prevented" has no way to learn that the
    budget could never reach more than a few percent of the work - which makes a real result
    look like a poor one.
    """
    builder_class = _import("vulnpriority.eval.report", "ReportBuilder")
    artifacts = builder_class(currency=impact_model_for(config).currency).build(
        Path(output_dir),
        bundles=list(metrics),
        ablation=ablation,
        simulations=list(simulation),
        selections=list(selection),
        adversarial=adversarial,
        manifest=manifest,
        capacity=capacity_from_payload(capacity),
    )
    written = [Path(artifacts.report_md), Path(artifacts.report_json)]
    written.extend(Path(figure) for figure in getattr(artifacts, "figures", ()))
    return written
