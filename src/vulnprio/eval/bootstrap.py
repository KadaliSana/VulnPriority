"""Uncertainty: paired bootstrap intervals and the Wilcoxon signed-rank test (Gap 4).

Almost every reported improvement in this literature is a single number with no interval
attached, which makes "0.71 versus 0.68" unfalsifiable. The protocol therefore compares
rankers **paired by scan**: the same scan, the same features, the same labels, two
orderings. Pairing removes between-scan variance - some scans are simply easier - which
is where most of the apparent noise lives.

Two complementary statements:

* :func:`paired_bootstrap_ci` gives a confidence interval on the *mean difference*,
  answering "how big is the gain and how sure are we of its size".
* :func:`wilcoxon_signed_rank` gives a distribution-free p-value on the *median*
  difference, answering "could this have come from no difference at all". It is
  insensitive to the handful of scans where one ranker wins enormously, which a mean can
  ride on.

Both are seeded and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "BootstrapDiff",
    "WilcoxonResult",
    "paired_bootstrap_ci",
    "paired_bootstrap",
    "wilcoxon_signed_rank",
    "bootstrap_mean_ci",
]


@dataclass(frozen=True)
class BootstrapDiff:
    """Result of a paired bootstrap over per-scan metric values."""

    observed_diff: float
    ci_low: float
    ci_high: float
    prob_a_better: float
    n_pairs: int
    iters: int
    seed: int
    alpha: float = 0.05

    @property
    def ci(self) -> tuple[float, float]:
        return (self.ci_low, self.ci_high)

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero - the only claim the protocol will make."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    n_nonzero: int
    z: float = 0.0


def paired_bootstrap_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    iters: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile confidence interval for ``mean(a) - mean(b)`` over paired observations.

    Resamples *scan indices* with replacement ``iters`` times, keeping each scan's pair
    of values together, and returns the ``alpha/2`` and ``1 - alpha/2`` percentiles of
    the resampled mean difference. Pairing is the whole point: resampling the two vectors
    independently would inflate the interval with between-scan variance that cancels.

    Degenerate input (empty, or a single pair) returns ``(d, d)`` for the observed
    difference ``d`` rather than raising, so one-scan folds do not abort a run.
    """
    result = paired_bootstrap(values_a, values_b, iters=iters, seed=seed, alpha=alpha)
    return result.ci


def paired_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    iters: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapDiff:
    """:func:`paired_bootstrap_ci` with the point estimate and win probability attached.

    ``prob_a_better`` is the fraction of bootstrap replicates in which A's mean exceeds
    B's - a directly readable "in 97% of resamples the learned ranker won".
    """
    left = np.asarray(list(values_a), dtype=float)
    right = np.asarray(list(values_b), dtype=float)
    if left.shape != right.shape:
        raise ValueError(f"paired bootstrap needs equal-length inputs: {left.shape} vs {right.shape}")
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    n = len(left)
    observed = float(np.mean(left - right)) if n else 0.0
    if n < 2 or iters < 1:
        return BootstrapDiff(observed, observed, observed, 1.0 if observed > 0 else 0.0, n, 0, seed, alpha)

    rng = np.random.default_rng(seed)
    differences = left - right
    draws = rng.integers(0, n, size=(int(iters), n))
    means = differences[draws].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    return BootstrapDiff(
        observed_diff=observed,
        ci_low=float(low),
        ci_high=float(high),
        prob_a_better=float(np.mean(means > 0.0)),
        n_pairs=n,
        iters=int(iters),
        seed=seed,
        alpha=alpha,
    )


def bootstrap_mean_ci(
    values: Sequence[float], iters: int = 500, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile interval for the mean of one metric across scans.

    Used for the error bars on the metric comparison figure. Fewer than two finite
    values returns ``(v, v)``.
    """
    sample = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(sample) == 0:
        return (0.0, 0.0)
    if len(sample) < 2 or iters < 1:
        value = float(sample.mean())
        return (value, value)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sample), size=(int(iters), len(sample)))
    means = sample[draws].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    return (float(low), float(high))


def wilcoxon_signed_rank(values_a: Iterable[float], values_b: Iterable[float]) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test over paired per-scan metric values.

    The statistic is built from the differences ``d_i = a_i - b_i``:

    1. drop the zero differences (ties between the two rankers on that scan);
    2. rank ``|d_i|`` ascending, sharing mean ranks among ties;
    3. ``W+`` is the rank sum of the positive differences, ``W-`` of the negative ones,
       and the reported statistic is ``min(W+, W-)``.

    The p-value uses the normal approximation with a continuity correction and the
    standard tie correction::

        mu    = n(n+1)/4
        sigma = sqrt( n(n+1)(2n+1)/24 - sum_t (t^3 - t)/48 )
        z     = (W+ - mu +/- 0.5) / sigma

    The approximation is what makes this dependency-free and deterministic; with the
    protocol's fold counts ``n`` is in the dozens (one value per test scan), where the
    approximation is good. All-zero differences (two rankers that agree on every scan)
    return ``p = 1.0`` rather than raising, which is the correct reading and is exactly
    the case scipy refuses.
    """
    left = np.asarray(list(values_a), dtype=float)
    right = np.asarray(list(values_b), dtype=float)
    if left.shape != right.shape:
        raise ValueError(f"wilcoxon needs equal-length inputs: {left.shape} vs {right.shape}")
    finite = np.isfinite(left) & np.isfinite(right)
    differences = (left - right)[finite]
    nonzero = differences[differences != 0.0]
    n = len(nonzero)
    if n == 0:
        return WilcoxonResult(statistic=0.0, p_value=1.0, n_nonzero=0, z=0.0)

    magnitude = np.abs(nonzero)
    order = np.argsort(magnitude, kind="mergesort")
    ordered = magnitude[order]
    ranks = np.empty(n, dtype=float)
    tie_correction = 0.0
    index = 0
    while index < n:
        end = index
        while end + 1 < n and ordered[end + 1] == ordered[index]:
            end += 1
        size = end - index + 1
        ranks[order[index : end + 1]] = (index + end) / 2.0 + 1.0
        if size > 1:
            tie_correction += size**3 - size
        index = end + 1

    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    statistic = min(w_plus, w_minus)

    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction / 48.0
    if variance <= 0.0:
        return WilcoxonResult(statistic=statistic, p_value=1.0, n_nonzero=n, z=0.0)
    difference = w_plus - mean
    correction = 0.5 if difference > 0 else (-0.5 if difference < 0 else 0.0)
    z = (difference - correction) / math.sqrt(variance)
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return WilcoxonResult(
        statistic=statistic, p_value=float(min(1.0, max(0.0, p_value))), n_nonzero=n, z=float(z)
    )
