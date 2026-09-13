"""``ReportBuilder``: the artefact a reviewer reads (DESIGN.md 3.9 and 4, Gap 4).

Three outputs, written side by side into one directory:

``report.md``
    Tables a reader can cite, with the protocol stated above them so no number is
    quotable without its conditions: which split produced it, how many scans it averages,
    and what its confidence interval is. Every headline figure is printed next to the
    baselines it must beat, because a metric without a baseline is not a result.
``report.json``
    The same content, machine readable and complete, so a downstream tool (or a reviewer
    re-deriving a number) never has to parse the prose.
``figures/*.png``
    Matplotlib figures on the ``Agg`` backend: written to disk, never shown.

The builder is deliberately tolerant: any section whose input is absent is skipped with a
line saying so, because the CLI's ``evaluate``, ``ablate``, ``select``, ``simulate`` and
``adversarial`` commands can each be run alone and must each produce a readable report.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

# Non-interactive backend: figures are written to disk and `plt.show()` is never called.
matplotlib.use("Agg")

import numpy as np

from vulnpriority.core.enums import MetricName, RankerName
from vulnpriority.core.money import DEFAULT_CURRENCY, format_money
from vulnpriority.core.models import (
    AblationTable,
    AdversarialReport,
    CalibrationReport,
    MetricBundle,
    RunManifest,
    SelectionResult,
    SimulationResult,
)
from vulnpriority.eval.ablation import COMPONENT_KEYS, INTERACTION_KEYS
from vulnpriority.eval.simulation import prevention_rate, summarise

__all__ = [
    "ReportArtifacts",
    "ReportBuilder",
    "HEADLINE_METRICS",
    "CLASSIFICATION_METRICS",
    "DECISION_METRICS",
]

#: Ranking metrics printed in the headline comparison table, in order.
HEADLINE_METRICS: tuple[str, ...] = (
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "precision@10",
    "recall@10",
    "risk_capture@10",
    "map",
    "mrr",
    "mean_rank_of_exploited",
    "kendall_tau_vs_cvss",
)

#: Classification metrics printed in the second table, in order.
CLASSIFICATION_METRICS: tuple[str, ...] = (
    MetricName.ROC_AUC.value,
    MetricName.PR_AUC.value,
    MetricName.MCC.value,
    MetricName.F1_MINORITY.value,
    MetricName.BALANCED_ACCURACY.value,
)

#: Decision metrics printed in the third table, in order.
DECISION_METRICS: tuple[str, ...] = (
    MetricName.EFFICIENCY.value,
    MetricName.COVERAGE.value,
    MetricName.WORKLOAD_REDUCTION.value,
)


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths of everything :meth:`ReportBuilder.build` wrote."""

    output_dir: Path
    report_md: Path
    report_json: Path
    figures: tuple[Path, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "report_md": str(self.report_md),
            "report_json": str(self.report_json),
            "figures": [str(path) for path in self.figures],
        }


class ReportBuilder:
    """Assembles the report from whatever evaluation artefacts are available."""

    def __init__(
        self,
        title: str = "vulnpriority evaluation report",
        currency: str = DEFAULT_CURRENCY,
    ) -> None:
        self.title = title
        #: What the money columns in this report are denominated in. Supplied by the
        #: caller from ``ImpactModel.currency``; the report never infers it from a field
        #: name, because no field name carries one.
        self.currency = currency

    def build(
        self,
        output_dir: str | Path,
        bundles: Sequence[MetricBundle] = (),
        ablation: AblationTable | None = None,
        simulations: Sequence[SimulationResult] = (),
        selections: Sequence[SelectionResult] = (),
        adversarial: AdversarialReport | None = None,
        manifest: RunManifest | None = None,
        capacity: Any | None = None,
    ) -> ReportArtifacts:
        """Write the report. ``capacity`` is the simulator's ``CapacityContext``.

        Passing it is what lets the simulation section state the share of the backlog the
        budget could ever reach, without which a prevention count cannot be read. It is
        optional so that a metrics-only run still produces a report.
        """
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        figure_dir = directory / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)

        by_ranker = _aggregate_bundles(bundles)
        figures = self._figures(figure_dir, bundles, by_ranker, ablation, simulations)

        payload = {
            "title": self.title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest.model_dump(mode="json") if manifest is not None else None,
            "protocol": _protocol(bundles),
            "metrics": {
                ranker: {
                    "n_folds": summary["n_folds"],
                    "values": summary["values"],
                    "runtime_seconds": summary["runtime_seconds"],
                    "minority": summary["minority"],
                    "calibration": summary["calibration"],
                }
                for ranker, summary in by_ranker.items()
            },
            "bundles": [bundle.model_dump(mode="json") for bundle in bundles],
            "ablation": ablation.model_dump(mode="json") if ablation is not None else None,
            "simulations": [item.model_dump(mode="json") for item in simulations],
            "simulation_summary": (
                summarise(list(simulations), capacity) if simulations else None
            ),
            "selections": [item.model_dump(mode="json") for item in selections],
            "adversarial": adversarial.model_dump(mode="json") if adversarial is not None else None,
            "figures": [str(path.relative_to(directory)).replace("\\", "/") for path in figures],
        }

        report_json = directory / "report.json"
        report_json.write_text(
            json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8"
        )

        report_md = directory / "report.md"
        report_md.write_text(
            self._markdown(
                by_ranker,
                bundles,
                ablation,
                simulations,
                selections,
                adversarial,
                manifest,
                figures,
                directory,
                capacity,
            ),
            encoding="utf-8",
        )
        return ReportArtifacts(
            output_dir=directory,
            report_md=report_md,
            report_json=report_json,
            figures=tuple(figures),
        )

    # -- markdown ----------------------------------------------------------

    def _markdown(
        self,
        by_ranker: Mapping[str, dict[str, Any]],
        bundles: Sequence[MetricBundle],
        ablation: AblationTable | None,
        simulations: Sequence[SimulationResult],
        selections: Sequence[SelectionResult],
        adversarial: AdversarialReport | None,
        manifest: RunManifest | None,
        figures: Sequence[Path],
        directory: Path,
        capacity: Any | None = None,
    ) -> str:
        lines: list[str] = [f"# {self.title}", ""]
        lines += [
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}.",
            "",
            "Every number below is produced by one protocol: time-ordered splits with a gap "
            "buffer, confirmed-exploitation labels (never CVSS), and identical data, "
            "preprocessing and metrics for every ranker. Intervals are paired bootstrap "
            "percentile intervals over per-scan values.",
            "",
        ]

        lines += self._reproducibility(manifest)
        lines += self._protocol_section(bundles)
        lines += self._ranking_section(by_ranker)
        lines += self._classification_section(by_ranker)
        lines += self._decision_section(by_ranker)
        lines += self._ablation_section(ablation)
        lines += self._selection_section(selections)
        lines += self._simulation_section(simulations, capacity)
        lines += self._adversarial_section(adversarial)

        lines += ["## Figures", ""]
        if figures:
            for path in figures:
                relative = str(path.relative_to(directory)).replace("\\", "/")
                lines.append(f"![{path.stem}]({relative})")
                lines.append("")
        else:
            lines += ["_No figures: no metric bundles were supplied._", ""]
        return "\n".join(lines) + "\n"

    def _reproducibility(self, manifest: RunManifest | None) -> list[str]:
        lines = ["## Reproducibility", ""]
        if manifest is None:
            return lines + ["_No run manifest supplied._", ""]
        rows = [
            ["run id", manifest.run_id],
            ["created at", str(manifest.created_at)],
            ["config hash", manifest.config_hash],
            ["dataset hash", manifest.dataset_hash or "-"],
            ["seeds", ", ".join(str(seed) for seed in manifest.seeds) or "-"],
            ["LLM backend", f"{manifest.llm_backend.value} ({manifest.llm_model or 'n/a'})"],
            ["feed mode", manifest.feed_mode.value],
            ["as of", str(manifest.as_of) if manifest.as_of else "-"],
            ["command", manifest.command or "-"],
        ]
        lines += _table(["item", "value"], rows)
        if manifest.package_versions:
            lines += ["", "Library versions:", ""]
            lines += _table(
                ["package", "version"],
                [[name, version] for name, version in sorted(manifest.package_versions.items())],
            )
        return lines + [""]

    def _protocol_section(self, bundles: Sequence[MetricBundle]) -> list[str]:
        lines = ["## Protocol", ""]
        protocol = _protocol(bundles)
        if not protocol["folds"]:
            return lines + ["_No metric bundles supplied._", ""]
        rows = [
            [
                str(fold["fold"]),
                fold["kind"],
                str(fold["n_train"]),
                str(fold["n_test"]),
                str(fold["train_end"]),
                str(fold["test_start"]),
                str(fold["gap_days"]),
                fold["held_out_app_id"] or "-",
            ]
            for fold in protocol["folds"]
        ]
        lines += _table(
            ["fold", "kind", "train scans", "test scans", "train end", "test start", "gap days", "held-out app"],
            rows,
        )
        if any(fold["kind"] == "random" for fold in protocol["folds"]):
            lines += [
                "",
                "> **Control condition present.** Folds marked `random` are the unrealistic "
                "control (Gap 4): they leak future information and near-duplicate scans across "
                "the split. Their numbers quantify what random splitting overstates and are "
                "never a result.",
            ]
        return lines + [""]

    def _ranking_section(self, by_ranker: Mapping[str, dict[str, Any]]) -> list[str]:
        lines = ["## Ranking quality", ""]
        if not by_ranker:
            return lines + ["_No metric bundles supplied._", ""]
        present = [key for key in HEADLINE_METRICS if _any_metric(by_ranker, key)]
        rows = [
            [ranker] + [_fmt_with_ci(summary["values"].get(key)) for key in present]
            for ranker, summary in by_ranker.items()
        ]
        lines += _table(["ranker"] + list(present), rows)
        lines += [
            "",
            "`mean_rank_of_exploited` is in positions and lower is better; every other column "
            "is in [0, 1] or [-1, 1] and higher is better, except `kendall_tau_vs_cvss`, which "
            "is descriptive: a value near 1 means the ranking has reproduced CVSS and added "
            "nothing.",
            "",
        ]
        return lines

    def _classification_section(self, by_ranker: Mapping[str, dict[str, Any]]) -> list[str]:
        lines = ["## Classification and calibration", ""]
        if not by_ranker:
            return lines + ["_No metric bundles supplied._", ""]
        rows = []
        for ranker, summary in by_ranker.items():
            row = [ranker] + [
                _fmt(summary["values"].get(key, {}).get("value")) for key in CLASSIFICATION_METRICS
            ]
            calibration = summary["calibration"]
            row += [
                _fmt(calibration["brier"]) if calibration else "-",
                _fmt(calibration["ece"]) if calibration else "-",
            ]
            minority = summary["minority"]
            row += [_fmt(minority["positive_rate"]) if minority else "-"]
            rows.append(row)
        lines += _table(
            ["ranker"] + list(CLASSIFICATION_METRICS) + ["brier", "ece", "positive rate"], rows
        )
        lines += [
            "",
            "MCC and minority-class F1 are the metrics of record for the rare positive class "
            "(Gap 7): on a positive rate this low, accuracy and ROC-AUC flatter a model that "
            "has learned nothing. Brier and ECE are blank for a ranker with no calibrated "
            "probability head, because a ranking score is not a probability.",
            "",
        ]
        return lines

    def _decision_section(self, by_ranker: Mapping[str, dict[str, Any]]) -> list[str]:
        lines = ["## Decision metrics", ""]
        if not by_ranker:
            return lines + ["_No metric bundles supplied._", ""]
        rows = [
            [ranker] + [_fmt(summary["values"].get(key, {}).get("value")) for key in DECISION_METRICS]
            for ranker, summary in by_ranker.items()
        ]
        lines += _table(["ranker"] + list(DECISION_METRICS), rows)
        lines += [
            "",
            "Efficiency and coverage follow Shimizu and Hashimoto: efficiency is the share of "
            "the selected set that is confirmed-exploited, coverage the share of all "
            "confirmed-exploited findings the selection retains. They trade against each other "
            "and workload reduction is only meaningful beside coverage.",
            "",
        ]
        return lines

    def _ablation_section(self, ablation: AblationTable | None) -> list[str]:
        lines = ["## Component ablation (2^3 factorial)", ""]
        if ablation is None or not ablation.cells:
            return lines + ["_No ablation supplied._", ""]
        metrics = sorted({key for cell in ablation.cells for key in cell.mean})
        headline = [key for key in HEADLINE_METRICS if key in metrics][:4] or metrics[:4]

        rows = []
        for cell in ablation.cells:
            row = [
                cell.flags.label(),
                "on" if cell.flags.a else "off",
                "on" if cell.flags.b else "off",
                "on" if cell.flags.c else "off",
                str(cell.n),
            ]
            row += [
                f"{_fmt(cell.mean.get(key))} +/- {_fmt(cell.std.get(key))}" for key in headline
            ]
            rows.append(row)
        lines += _table(["cell", "A", "B", "C", "runs"] + headline, rows)

        lines += ["", "### Main effects", ""]
        effect_rows = []
        for component in COMPONENT_KEYS:
            effects = ablation.main_effects.get(component, {})
            intervals = ablation.paired_ci.get(component, {})
            row = [component]
            for key in headline:
                value = effects.get(key)
                interval = intervals.get(key)
                row.append(
                    _fmt(value)
                    if interval is None
                    else f"{_fmt(value)} [{_fmt(interval[0])}, {_fmt(interval[1])}]"
                )
            effect_rows.append(row)
        lines += _table(["component"] + headline, effect_rows)
        lines += [
            "",
            "A main effect is the mean difference with the component enabled minus disabled, "
            "averaged over every configuration of the other two components. Intervals are "
            "paired bootstrap intervals over cells paired by other-factor configuration and "
            "seed; an interval spanning zero means the component's contribution is within noise.",
            "",
            "### Interactions",
            "",
        ]
        interaction_rows = [
            [key] + [_fmt(ablation.interactions.get(key, {}).get(metric)) for metric in headline]
            for key in INTERACTION_KEYS
        ]
        lines += _table(["effect"] + headline, interaction_rows)
        lines += [
            "",
            "A positive two-way interaction means the components are complementary (each is "
            "worth more in the other's presence); a negative one means they are substitutes.",
            "",
        ]
        return lines

    def _selection_section(self, selections: Sequence[SelectionResult]) -> list[str]:
        lines = ["## Selection under a remediation budget", ""]
        if not selections:
            return lines + ["_No selection results supplied._", ""]
        rows = [
            [
                item.scan_id,
                item.ranker.value,
                item.method.value,
                _fmt(item.budget_hours, 1),
                str(len(item.selected_ids)),
                _fmt(item.total_hours, 1),
                _money(item.risk_captured, self.currency),
                _fmt(item.risk_capture_fraction),
                f"{item.exploited_captured}/{item.exploited_total}",
            ]
            for item in selections
        ]
        lines += _table(
            [
                "scan",
                "ranker",
                "method",
                "budget h",
                "selected",
                "hours",
                "risk captured",
                "risk fraction",
                "exploited caught",
            ],
            rows,
        )
        return lines + [""]

    def _simulation_section(
        self, simulations: Sequence[SimulationResult], capacity: Any | None = None
    ) -> list[str]:
        lines = ["## Longitudinal simulation", ""]
        if not simulations:
            return lines + ["_No simulation results supplied._", ""]

        lines += _capacity_lines(simulations, capacity)
        lines += _prevention_headline(simulations)

        rows = [
            [
                item.policy.value,
                f"{item.exploited_remediated_before_exploit}/{item.exploited_total}",
                f"{prevention_rate(item) * 100:.1f}%",
                _fmt(item.exposure_days_exploited, 1),
                "-" if item.reduction_vs_cvss is None else f"{item.reduction_vs_cvss * 100:+.1f}%",
                _money(item.expected_loss_days, self.currency),
                _fmt(item.exposure_days_total, 1),
            ]
            for item in simulations
        ]
        lines += _table(
            [
                "policy",
                "prevented",
                "prevention rate",
                "exposure days (exploited)",
                "reduction vs reference",
                "expected-loss days",
                "exposure days (total, capacity-bound)",
            ],
            rows,
        )
        lines += [
            "",
            "**`prevented`** counts confirmed-exploited findings closed before their first "
            "exploitation evidence date: the counterfactual that says the ordering would have "
            "stopped something. It is the simulation's headline outcome, and it is only "
            "interpretable against the capacity line above.",
            "",
            "**`reduction vs reference` is measured on exposure days carried by the "
            "confirmed-exploited findings**, not on total exposure. That is the part of the "
            "outcome a remediation ordering controls, and it is what Gap 10 asks about. "
            "Exploited exposure is preferred over expected-loss days for the headline because "
            "it is ground truth rather than the framework's own estimate - expected loss can be "
            "reduced by believing one's own `p_exploit` harder, while an exploited finding was "
            "exploited whatever the model thought.",
            "",
            "**`exposure days (total)` is a capacity measurement, not a prioritisation "
            "measurement.** Every finding the budget never reaches accrues the full horizon "
            "under every policy, so on a backlog larger than the budget the total is bounded "
            "below by capacity and is expected to be near-identical across policies. A reader "
            "who sees 145,000 days against 147,000 is looking at the size of the budget, not at "
            "a defeat for the ordering that produced the second number; the columns to the left "
            "are where the orderings differ.",
            "",
        ]
        return lines

    def _adversarial_section(self, adversarial: AdversarialReport | None) -> list[str]:
        lines = ["## Adversarial robustness", ""]
        if adversarial is None:
            return lines + ["_No adversarial report supplied._", ""]
        rows = [
            ["cases", str(adversarial.n_cases)],
            ["attack success rate", _fmt(adversarial.attack_success_rate)],
            ["canary leak rate", _fmt(adversarial.canary_leak_rate)],
            ["detection rate", _fmt(adversarial.detection_rate)],
            ["false positive rate (benign controls)", _fmt(adversarial.false_positive_rate)],
            ["mean absolute rank shift", _fmt(adversarial.mean_abs_rank_shift, 2)],
            ["max absolute rank shift", str(adversarial.max_abs_rank_shift)],
        ]
        lines += _table(["measure", "value"], rows)
        if adversarial.per_category:
            lines += ["", "Per category:", ""]
            categories = sorted(adversarial.per_category)
            columns = sorted({key for values in adversarial.per_category.values() for key in values})
            lines += _table(
                ["category"] + columns,
                [
                    [category] + [_fmt(adversarial.per_category[category].get(key)) for key in columns]
                    for category in categories
                ],
            )
        return lines + [""]

    # -- figures -----------------------------------------------------------

    def _figures(
        self,
        figure_dir: Path,
        bundles: Sequence[MetricBundle],
        by_ranker: Mapping[str, dict[str, Any]],
        ablation: AblationTable | None,
        simulations: Sequence[SimulationResult],
    ) -> list[Path]:
        paths: list[Path] = []
        for path in (
            _figure_metric_comparison(figure_dir, by_ranker),
            _figure_reliability(figure_dir, bundles),
            _figure_ablation(figure_dir, ablation),
            _figure_exposure(figure_dir, simulations),
        ):
            if path is not None:
                paths.append(path)
        return paths


# ---------------------------------------------------------------------------
# Aggregation and formatting
# ---------------------------------------------------------------------------


def _aggregate_bundles(bundles: Sequence[MetricBundle]) -> dict[str, dict[str, Any]]:
    """Fold means per ranker, keeping the interval and the fold count.

    Bundles are grouped by ranker and averaged over folds; the interval reported is the
    mean of the folds' own bootstrap intervals, which is the conservative reading when a
    metric is stable across folds and visibly wide when it is not.
    """
    grouped: dict[str, list[MetricBundle]] = {}
    for bundle in bundles:
        grouped.setdefault(bundle.ranker.value, []).append(bundle)

    out: dict[str, dict[str, Any]] = {}
    for ranker, items in grouped.items():
        values: dict[str, dict[str, Any]] = {}
        keys = {value.key for item in items for value in item.values}
        for key in sorted(keys):
            points, lows, highs, counts = [], [], [], []
            for item in items:
                for value in item.values:
                    if value.key != key or not math.isfinite(value.value):
                        continue
                    points.append(value.value)
                    counts.append(value.n or 0)
                    if value.ci_low is not None and value.ci_high is not None:
                        lows.append(value.ci_low)
                        highs.append(value.ci_high)
            if not points:
                continue
            values[key] = {
                "value": float(np.mean(points)),
                "ci_low": float(np.mean(lows)) if lows else None,
                "ci_high": float(np.mean(highs)) if highs else None,
                "n": int(sum(counts)),
                "folds": len(points),
            }
        calibration = next((item.calibration for item in items if item.calibration), None)
        minority = next((item.minority for item in items if item.minority), None)
        out[ranker] = {
            "n_folds": len(items),
            "values": values,
            "runtime_seconds": float(np.mean([item.runtime_seconds for item in items])),
            "calibration": calibration.model_dump(mode="json") if calibration else None,
            "minority": minority.model_dump(mode="json") if minority else None,
        }
    # Stable, readable order: learned ranker first, then baselines alphabetically.
    ordering = [name.value for name in RankerName]
    return dict(sorted(out.items(), key=lambda kv: ordering.index(kv[0]) if kv[0] in ordering else 99))


def _protocol(bundles: Sequence[MetricBundle]) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for bundle in bundles:
        key = (bundle.split.kind.value, bundle.split.fold)
        if key in seen:
            continue
        seen.add(key)
        folds.append(
            {
                "fold": bundle.split.fold,
                "kind": bundle.split.kind.value,
                "n_train": len(bundle.split.train_scan_ids),
                "n_test": len(bundle.split.test_scan_ids),
                "train_end": str(bundle.split.train_end),
                "test_start": str(bundle.split.test_start),
                "gap_days": bundle.split.gap_days,
                "held_out_app_id": bundle.split.held_out_app_id,
            }
        )
    return {
        "folds": sorted(folds, key=lambda item: (item["kind"], item["fold"])),
        "rankers": sorted({bundle.ranker.value for bundle in bundles}),
        "flags": sorted({bundle.flags.label() for bundle in bundles}),
    }


def _any_metric(by_ranker: Mapping[str, dict[str, Any]], key: str) -> bool:
    return any(key in summary["values"] for summary in by_ranker.values())


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _money(value: float | None, currency: str = DEFAULT_CURRENCY) -> str:
    """A table cell: grouped digits and the run's symbol, never a scale word.

    Every row in a column has to be written at the same scale or the column cannot be
    compared down its length, which is the whole reason to put it in a table.
    """
    if value is None or not math.isfinite(float(value)):
        return "-"
    return format_money(value, currency)


def _fmt_with_ci(entry: Mapping[str, Any] | None, digits: int = 3) -> str:
    if not entry:
        return "-"
    text = _fmt(entry.get("value"), digits)
    low, high = entry.get("ci_low"), entry.get("ci_high")
    if low is None or high is None:
        return text
    return f"{text} [{_fmt(low, digits)}, {_fmt(high, digits)}]"


def _capacity_lines(simulations: Sequence[SimulationResult], capacity: Any | None) -> list[str]:
    """The budget the simulation ran under, so its numbers can be read at all.

    ``capacity`` is a :class:`~vulnpriority.eval.simulation.CapacityContext` when the caller
    has one. Without it, weeks, hours per week and total hours available are still derived
    from the results themselves; only the backlog size and the reachable share are missing,
    and the report says so rather than omitting the line.
    """
    first = simulations[0]
    # The context is the authority on the budget when it is supplied; the results carry the
    # same two numbers and stand in for it when it is not.
    weeks = capacity.weeks if capacity is not None else first.weeks
    per_week = capacity.capacity_hours_per_week if capacity is not None else first.capacity_hours_per_week
    rows = [
        ["weeks simulated", str(weeks)],
        ["capacity per week (hours)", _fmt(per_week, 1)],
        ["total hours available", _fmt(weeks * per_week, 1)],
    ]
    note = ""
    if capacity is not None:
        rows += [
            ["remediation backlog (hours, one charge per root-cause cluster)", _fmt(capacity.backlog_hours, 1)],
            ["findings / root-cause clusters", f"{capacity.n_findings} / {capacity.n_clusters}"],
            ["confirmed-exploited findings", str(capacity.n_exploited)],
            ["share of the backlog the budget could ever reach", f"{capacity.reachable_fraction * 100:.1f}%"],
        ]
        if capacity.capacity_bound:
            note = (
                f"The budget covers {capacity.reachable_fraction * 100:.1f}% of the backlog, so "
                "most findings are never reached under any ordering and total exposure is "
                "bounded below by capacity. Read the prevention and exploited-exposure columns, "
                "not the total."
            )
    else:
        rows.append(
            ["remediation backlog", "not supplied (pass the simulator's CapacityContext)"]
        )

    lines = ["### Capacity context", ""] + _table(["item", "value"], rows)
    if note:
        lines += ["", f"> {note}"]
    return lines + [""]


def _reference_of(simulations: Sequence[SimulationResult]) -> SimulationResult | None:
    """The reference policy, recovered from the results themselves.

    A policy compared against itself reports a reduction of exactly zero, which identifies
    the reference without the caller having to pass its name. When that is ambiguous the
    caller gets ``None`` and the headline falls back to the weakest policy, which is still
    an honest comparison - both were simulated on the same backlog and the same budget.
    """
    zero = [item for item in simulations if item.reduction_vs_cvss == 0.0]
    return zero[0] if len(zero) == 1 else None


def _prevention_headline(simulations: Sequence[SimulationResult]) -> list[str]:
    """One sentence naming the best policy's prevention count against the reference."""
    usable = [item for item in simulations if item.exploited_total > 0]
    if not usable:
        return []
    best = max(usable, key=prevention_rate)
    reference = _reference_of(usable) or min(usable, key=prevention_rate)
    if best.policy == reference.policy:
        return [
            f"**Prevention.** Under `{best.policy.value}`, "
            f"{best.exploited_remediated_before_exploit} of {best.exploited_total} "
            f"confirmed-exploited findings ({prevention_rate(best) * 100:.1f}%) were remediated "
            "before their first exploitation evidence date.",
            "",
        ]
    return [
        f"**Prevention.** Under `{best.policy.value}`, "
        f"{best.exploited_remediated_before_exploit} of {best.exploited_total} "
        f"confirmed-exploited findings ({prevention_rate(best) * 100:.1f}%) were remediated "
        "before their first exploitation evidence date, against "
        f"{reference.exploited_remediated_before_exploit} of {reference.exploited_total} "
        f"({prevention_rate(reference) * 100:.1f}%) under `{reference.policy.value}` - on the "
        "same backlog and the same weekly capacity.",
        "",
    ]


def _cell(value: Any) -> str:
    """One markdown table cell, with pipes escaped so a value cannot break the table."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    body = [list(row) for row in rows]
    lines = ["| " + " | ".join(_cell(header) for header in headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in body:
        cells = [_cell(cell) for cell in row] + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(cells[: len(headers)]) + " |")
    return lines


# ---------------------------------------------------------------------------
# Figures (Agg backend; nothing is ever shown)
# ---------------------------------------------------------------------------


def _figure_metric_comparison(
    figure_dir: Path, by_ranker: Mapping[str, dict[str, Any]]
) -> Path | None:
    """Bar chart of the headline ranking metrics per ranker, with confidence intervals."""
    from matplotlib import pyplot as plt

    metrics = [key for key in ("ndcg@10", "recall@10", "risk_capture@10", "map") if _any_metric(by_ranker, key)]
    rankers = [name for name in by_ranker if any(key in by_ranker[name]["values"] for key in metrics)]
    if not metrics or not rankers:
        return None

    positions = np.arange(len(rankers), dtype=float)
    width = 0.8 / len(metrics)
    figure, axes = plt.subplots(figsize=(max(6.0, 1.6 * len(rankers)), 4.2))
    for index, metric in enumerate(metrics):
        heights, errors_low, errors_high = [], [], []
        for ranker in rankers:
            entry = by_ranker[ranker]["values"].get(metric)
            value = float(entry["value"]) if entry else 0.0
            heights.append(value)
            low = entry.get("ci_low") if entry else None
            high = entry.get("ci_high") if entry else None
            errors_low.append(max(0.0, value - float(low)) if low is not None else 0.0)
            errors_high.append(max(0.0, float(high) - value) if high is not None else 0.0)
        axes.bar(
            positions + index * width - 0.4 + width / 2,
            heights,
            width=width,
            yerr=[errors_low, errors_high],
            capsize=3,
            label=metric,
        )
    axes.set_xticks(positions)
    axes.set_xticklabels(rankers, rotation=30, ha="right")
    axes.set_ylabel("metric value")
    axes.set_title("Ranking quality by ranker (bootstrap intervals over scans)")
    axes.legend(fontsize="small")
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = figure_dir / "metric_comparison.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _figure_reliability(figure_dir: Path, bundles: Sequence[MetricBundle]) -> Path | None:
    """Reliability diagram for every ranker that supplied calibrated probabilities."""
    from matplotlib import pyplot as plt

    calibrated: list[tuple[str, CalibrationReport]] = []
    for bundle in bundles:
        if bundle.calibration is not None and any(bundle.calibration.bin_count):
            if bundle.ranker.value not in {name for name, _ in calibrated}:
                calibrated.append((bundle.ranker.value, bundle.calibration))
    if not calibrated:
        return None

    figure, axes = plt.subplots(figsize=(5.0, 5.0))
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    for name, report in calibrated:
        xs = [
            confidence
            for confidence, count in zip(report.bin_confidence, report.bin_count)
            if count > 0
        ]
        ys = [
            accuracy for accuracy, count in zip(report.bin_accuracy, report.bin_count) if count > 0
        ]
        axes.plot(xs, ys, marker="o", label=f"{name} (ECE {report.ece:.3f})")
    axes.set_xlabel("predicted probability")
    axes.set_ylabel("observed exploitation frequency")
    axes.set_title("Reliability diagram")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.legend(fontsize="small")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    path = figure_dir / "reliability.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _figure_ablation(figure_dir: Path, ablation: AblationTable | None) -> Path | None:
    """Main effects of Components A, B and C with their paired bootstrap intervals."""
    from matplotlib import pyplot as plt

    if ablation is None or not ablation.main_effects:
        return None
    metrics = sorted({key for effects in ablation.main_effects.values() for key in effects})
    preferred = [key for key in HEADLINE_METRICS if key in metrics][:3] or metrics[:3]
    if not preferred:
        return None

    positions = np.arange(len(COMPONENT_KEYS), dtype=float)
    width = 0.8 / len(preferred)
    figure, axes = plt.subplots(figsize=(6.0, 4.0))
    for index, metric in enumerate(preferred):
        heights, low_errors, high_errors = [], [], []
        for component in COMPONENT_KEYS:
            value = float(ablation.main_effects.get(component, {}).get(metric, 0.0))
            heights.append(value)
            interval = ablation.paired_ci.get(component, {}).get(metric)
            if interval is None:
                low_errors.append(0.0)
                high_errors.append(0.0)
            else:
                low_errors.append(max(0.0, value - float(interval[0])))
                high_errors.append(max(0.0, float(interval[1]) - value))
        axes.bar(
            positions + index * width - 0.4 + width / 2,
            heights,
            width=width,
            yerr=[low_errors, high_errors],
            capsize=3,
            label=metric,
        )
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_xticks(positions)
    axes.set_xticklabels([f"Component {key}" for key in COMPONENT_KEYS])
    axes.set_ylabel("main effect (on minus off)")
    axes.set_title("Component main effects (2^3 factorial)")
    axes.legend(fontsize="small")
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = figure_dir / "ablation_main_effects.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _figure_exposure(figure_dir: Path, simulations: Sequence[SimulationResult]) -> Path | None:
    """Cumulative exposure-day curves, one line per remediation policy."""
    from matplotlib import pyplot as plt

    usable = [item for item in simulations if item.weekly_cumulative_exposure]
    if not usable:
        return None
    figure, axes = plt.subplots(figsize=(6.5, 4.2))
    for item in usable:
        weeks = np.arange(1, len(item.weekly_cumulative_exposure) + 1)
        axes.plot(weeks, item.weekly_cumulative_exposure, marker="", label=item.policy.value)
    axes.set_xlabel("week")
    axes.set_ylabel("cumulative exposure days")
    axes.set_title("Cumulative exposure under a fixed weekly remediation capacity")
    axes.legend(fontsize="small")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    path = figure_dir / "exposure_curves.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
