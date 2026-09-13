"""Full factorial ablation over Components A, B and C (DESIGN.md 3.9, Gap 5).

Hybrid frameworks routinely report an end-to-end gain and attribute it to the whole
architecture. That is not evidence about any component: the gain may come entirely from
one of them, or - and this is the interesting case - from an *interaction*, where the
semantic assessment is only worth anything once the attack graph exists to use it.

This module runs the complete ``2^3`` design: all eight on/off combinations of Components
A, B and C, each over every seed, and reports the effects with the standard factorial
contrast arithmetic.

With the ``+/-1`` coding ``x_A = +1`` when A is enabled and ``-1`` when it is not, and
``y(cell)`` the mean metric value in a cell, over the eight cells::

    main effect of A   = sum(x_A * y) / 4  =  mean(y | A on) - mean(y | A off)
    interaction AB     = sum(x_A * x_B * y) / 4
                       = (1/2) * [ (effect of A | B on) - (effect of A | B off) ]
    interaction ABC    = sum(x_A * x_B * x_C * y) / 4

The divisor is ``2^(k-1) = 4``, which is what makes a main effect read directly as "the
metric is this much higher with the component than without it, averaged over every
configuration of the other two". Each effect additionally carries a paired bootstrap
interval computed over the cells paired by their other-factor configuration and seed, so
a component whose contribution is within noise is visibly within noise.

Disabled components do not merely have their features zeroed: ``FeatureFrame`` validates
its columns against ``feature_names_for(flags)``, so an off cell physically cannot carry
the component's signal through a constant column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vulnprio.core.config import PipelineConfig
from vulnprio.core.enums import Component, RankerName
from vulnprio.core.models import (
    AblationCell,
    AblationTable,
    ComponentFlags,
    FeatureFrame,
    LabelSet,
    MetricBundle,
    Split,
)
from vulnprio.eval.benchmark import BenchmarkRunner, SplitFrames
from vulnprio.eval.bootstrap import paired_bootstrap

__all__ = [
    "COMPONENT_KEYS",
    "INTERACTION_KEYS",
    "FullFactorialAblation",
    "cell_means",
    "main_effects",
    "interactions",
    "contrast",
    "flag_value",
    "component_of",
    "table_to_rows",
    "BuildFrameFn",
]

#: Main-effect keys, in report order.
COMPONENT_KEYS: tuple[str, ...] = ("A", "B", "C")

#: Two- and three-way interaction keys, in report order.
INTERACTION_KEYS: tuple[str, ...] = ("AB", "AC", "BC", "ABC")


def flag_value(flags: ComponentFlags, component: str) -> bool:
    """Whether ``component`` (``"A"``, ``"B"`` or ``"C"``) is enabled in ``flags``."""
    return {"A": flags.a, "B": flags.b, "C": flags.c}[component.upper()]


def contrast(flags: ComponentFlags, key: str) -> float:
    """The ``+/-1`` contrast coefficient of a cell for an effect key.

    ``"A"`` gives ``+1`` when A is on; ``"AB"`` gives the product of A's and B's
    coefficients, which is what makes the interaction contrast orthogonal to both main
    effects.
    """
    value = 1.0
    for component in key.upper():
        value *= 1.0 if flag_value(flags, component) else -1.0
    return value


def cell_means(
    bundles_by_cell: Mapping[ComponentFlags, Sequence[Mapping[str, float]]]
) -> dict[ComponentFlags, dict[str, float]]:
    """Mean metric values per cell over that cell's seeds."""
    out: dict[ComponentFlags, dict[str, float]] = {}
    for flags, runs in bundles_by_cell.items():
        keys = sorted({key for run in runs for key in run})
        out[flags] = {
            key: float(np.mean([run[key] for run in runs if key in run]))
            for key in keys
            if any(key in run for run in runs)
        }
    return out


def _effect(means: Mapping[ComponentFlags, Mapping[str, float]], key: str, metric: str) -> float:
    """One factorial contrast, divided by ``2^(k-1)`` where ``k`` is the factor count."""
    usable = [(flags, values[metric]) for flags, values in means.items() if metric in values]
    if not usable:
        return 0.0
    total = sum(contrast(flags, key) * value for flags, value in usable)
    # A complete 2^3 design has 8 cells and divisor 4; a partial one is normalised by
    # half its cell count so the effect keeps its "difference of averages" reading.
    return float(total / max(1.0, len(usable) / 2.0))


def main_effects(
    means: Mapping[ComponentFlags, Mapping[str, float]]
) -> dict[str, dict[str, float]]:
    """``{"A": {metric: effect}}``: mean with the component on minus mean with it off.

    Averaged over every configuration of the other two components, which is what makes
    it a *main* effect rather than a one-off comparison of two pipelines.
    """
    metrics = sorted({metric for values in means.values() for metric in values})
    return {
        component: {metric: _effect(means, component, metric) for metric in metrics}
        for component in COMPONENT_KEYS
    }


def interactions(
    means: Mapping[ComponentFlags, Mapping[str, float]]
) -> dict[str, dict[str, float]]:
    """Two- and three-way interaction effects, same contrast arithmetic.

    A positive ``AB`` means A and B are *complementary*: A is worth more when B is present
    than when it is absent. A negative one means they are substitutes, each partly doing
    the other's job - which is itself a finding worth reporting, and one an end-to-end
    comparison cannot see at all.
    """
    metrics = sorted({metric for values in means.values() for metric in values})
    return {
        key: {metric: _effect(means, key, metric) for metric in metrics}
        for key in INTERACTION_KEYS
    }


#: ``build_frame_fn(flags, split, seed) -> (train frame, test frame)``.
BuildFrameFn = Callable[[ComponentFlags, Split, int], tuple[FeatureFrame, FeatureFrame]]


@dataclass
class FullFactorialAblation:
    """Runs the ``2^3`` design and assembles an :class:`AblationTable`.

    ``ranker`` names the model under ablation - the ablation asks what the *components*
    contribute, so the same learner is used in every cell and only its inputs change.
    ``metrics`` restricts the table to the metric keys worth reporting; ``None`` keeps
    every metric the benchmark produced.
    """

    ranker: Any = RankerName.LAMBDAMART
    metrics: Sequence[str] | None = None
    expected_loss: Mapping[str, float] | None = None
    #: Optional ``(flags, seed) -> ranker`` factory, so every cell gets a fresh model.
    ranker_factory: Callable[[ComponentFlags, int], Any] | None = None
    #: ``(cell label, seed) -> {metric key: value}``, retained for the paired bootstrap.
    runs: dict[tuple[str, int], dict[str, float]] = field(default_factory=dict)
    bundles: list[MetricBundle] = field(default_factory=list)

    def run(
        self,
        build_frame_fn: BuildFrameFn,
        labels: LabelSet,
        splits: Sequence[Split],
        config: PipelineConfig,
    ) -> AblationTable:
        """Evaluate every cell of the design over every seed.

        ``build_frame_fn`` is the caller's feature pipeline: given the cell's component
        flags, a split and a seed it returns the train and test matrices for that cell.
        Keeping it a callback is what lets the ablation drive the *whole* pipeline -
        including re-running Components A, B and C - rather than just masking columns of
        a single pre-built matrix, which would be a much weaker experiment.
        """
        seeds = tuple(config.evaluation.seeds) or (config.seed,)
        self.runs = {}
        self.bundles = []
        by_cell: dict[ComponentFlags, list[dict[str, float]]] = {}
        seeds_by_cell: dict[ComponentFlags, list[int]] = {}

        for flags in ComponentFlags.all_cells():
            for seed in seeds:
                cell_config = config.with_flags(flags).model_copy(
                    update={"seed": seed, "ranking": config.ranking.model_copy(update={"seed": seed})}
                )
                folds = [
                    SplitFrames(split=split, train=train, test=test)
                    for split in splits
                    for train, test in [build_frame_fn(flags, split, seed)]
                ]
                model = (
                    self.ranker_factory(flags, seed) if self.ranker_factory else self.ranker
                )
                runner = BenchmarkRunner(expected_loss=self.expected_loss)
                bundles = runner.run(folds, labels, [model], cell_config)
                self.bundles.extend(bundles)

                values = _mean_over_folds(bundles, self.metrics)
                by_cell.setdefault(flags, []).append(values)
                seeds_by_cell.setdefault(flags, []).append(seed)
                self.runs[(flags.label(), seed)] = values

        means = cell_means(by_cell)
        cells = tuple(
            AblationCell(
                flags=flags,
                seeds=tuple(seeds_by_cell[flags]),
                mean=means[flags],
                std={
                    key: float(np.std([run[key] for run in by_cell[flags] if key in run]))
                    for key in means[flags]
                },
                n=len(by_cell[flags]),
            )
            for flags in sorted(by_cell, key=lambda item: item.label())
        )
        return AblationTable(
            cells=cells,
            main_effects=main_effects(means),
            interactions=interactions(means),
            paired_ci=self.paired_intervals(by_cell, seeds_by_cell, config),
        )

    def paired_intervals(
        self,
        by_cell: Mapping[ComponentFlags, Sequence[Mapping[str, float]]],
        seeds_by_cell: Mapping[ComponentFlags, Sequence[int]],
        config: PipelineConfig,
    ) -> dict[str, dict[str, tuple[float, float]]]:
        """Paired bootstrap interval for each main effect.

        The pairing is what makes the interval tight enough to be informative: a cell
        with A on is paired with the cell that differs *only* in A - same other
        components, same seed - so seed-to-seed variance and the difficulty of the
        underlying data cancel out of every pair.
        """
        out: dict[str, dict[str, tuple[float, float]]] = {}
        for component in COMPONENT_KEYS:
            pairs: list[tuple[dict[str, float], dict[str, float]]] = []
            for flags, runs in by_cell.items():
                if not flag_value(flags, component):
                    continue
                partner = flags.model_copy(update={component.lower(): False})
                if partner not in by_cell:
                    continue
                partner_by_seed = dict(zip(seeds_by_cell[partner], by_cell[partner]))
                for seed, run in zip(seeds_by_cell[flags], runs):
                    if seed in partner_by_seed:
                        pairs.append((run, partner_by_seed[seed]))
            if not pairs:
                continue
            metrics = sorted({key for on, off in pairs for key in set(on) & set(off)})
            intervals: dict[str, tuple[float, float]] = {}
            for metric in metrics:
                on_values = [on[metric] for on, off in pairs if metric in on and metric in off]
                off_values = [off[metric] for on, off in pairs if metric in on and metric in off]
                result = paired_bootstrap(
                    on_values,
                    off_values,
                    iters=config.evaluation.bootstrap_iters,
                    seed=config.seed,
                )
                intervals[metric] = (result.ci_low, result.ci_high)
            out[component] = intervals
        return out


def _mean_over_folds(
    bundles: Sequence[MetricBundle], metrics: Sequence[str] | None
) -> dict[str, float]:
    """Mean of each metric across the folds of one cell/seed run."""
    collected: dict[str, list[float]] = {}
    for bundle in bundles:
        for key, value in bundle.as_dict().items():
            if metrics is not None and key not in metrics:
                continue
            if np.isfinite(value):
                collected.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in collected.items()}


def component_of(key: str) -> Component | None:
    """The ``Component`` a single-letter effect key names, or ``None`` for interactions."""
    mapping = {"A": Component.A, "B": Component.B, "C": Component.C}
    return mapping.get(key.upper()) if len(key) == 1 else None


def table_to_rows(table: AblationTable) -> list[dict[str, Any]]:
    """Flatten an :class:`AblationTable` into report rows, one per cell."""
    rows: list[dict[str, Any]] = []
    for cell in table.cells:
        row: dict[str, Any] = {
            "cell": cell.flags.label(),
            "A": cell.flags.a,
            "B": cell.flags.b,
            "C": cell.flags.c,
            "n": cell.n,
            "seeds": list(cell.seeds),
        }
        row.update({key: value for key, value in sorted(cell.mean.items())})
        rows.append(row)
    return rows
