"""The seven baseline rankers (DESIGN.md 3.8 and section 4).

A learned ranker that is not measured against the orderings practitioners actually use is
an unfalsifiable claim, so every baseline here runs on the identical
:class:`FeatureFrame`, in the identical harness, with the identical metrics. None of them
fits anything: they are closed-form functions of columns that are already in the frame,
which is also why they are honest - a baseline that needed training would quietly be a
different experiment.

Scoring convention: higher is remediated sooner, and tie-breaks are explicit and bounded
so that a banded baseline (KEV-first, VMC chain) can never let a tie-break cross a band.
Every tie-break term is documented at its constant.

============================  ============================================================
ranker                        ordering
============================  ============================================================
``cvss_only``                 CVSS base score, the industry default and the thing Gap 3 says
                              must never also be the label
``epss_only``                 EPSS probability alone
``kev_first``                 KEV members first, CVSS within each band
``scanner_severity``          the scanner's own severity, confidence as tie-break
``expected_loss``             P(exploit) x impact: the decision-theoretic ordering of Gap 1
``vmc_chain``                 Shimizu and Hashimoto: KEV or EPSS >= 0.088, then CVSS >= 7.0
``random``                    seeded uniform noise, the floor every other number is above
============================  ============================================================

When a baseline's column is absent because its component is switched off in an ablation
cell, the baseline scores the documented neutral for that column rather than failing: the
cell is then measuring "this baseline has nothing to say here", which is the correct
reading, and :attr:`Baseline.warnings` records that it happened.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from vulnpriority.core.enums import RankerName
from vulnpriority.core.interfaces import Ranker
from vulnpriority.core.models import FeatureFrame
from vulnpriority.core.registry import register_ranker
from vulnpriority.rank.features import NEUTRAL

__all__ = [
    "VMC_EPSS_THRESHOLD",
    "VMC_CVSS_THRESHOLD",
    "VMC_BAND_EXPLOITED",
    "VMC_BAND_SEVERE",
    "VMC_BAND_REST",
    "KEV_BAND",
    "Baseline",
    "CvssOnlyRanker",
    "EpssOnlyRanker",
    "KevFirstRanker",
    "ScannerSeverityRanker",
    "ExpectedLossRanker",
    "VmcChainRanker",
    "RandomRanker",
]

_LOG = logging.getLogger(__name__)

#: Shimizu and Hashimoto's EPSS cut: the probability above which a vulnerability joins the
#: "treat as exploited" band. Mirrors ``EvaluationConfig.vmc_epss_threshold``.
VMC_EPSS_THRESHOLD: float = 0.088

#: Their CVSS cut: the conventional "high" boundary. Mirrors ``EvaluationConfig.vmc_cvss_threshold``.
VMC_CVSS_THRESHOLD: float = 7.0

#: Band scores. Kept a whole unit apart so no tie-break can promote across a band.
VMC_BAND_EXPLOITED: float = 2.0
VMC_BAND_SEVERE: float = 1.0
VMC_BAND_REST: float = 0.0

#: KEV-first band separation, same reasoning.
KEV_BAND: float = 1.0

#: Tie-break weights. ``0.5 * epss + 0.04 * cvss`` lives in ``[0, 0.9]`` because EPSS is a
#: probability and CVSS is capped at 10, so it orders within a band and never leaves it.
_TIE_EPSS: float = 0.5
_TIE_CVSS: float = 0.04

#: Scanner confidence tie-break: at most 0.5, half a severity step.
_TIE_CONFIDENCE: float = 0.5


class Baseline(Ranker):
    """Shared behaviour: no fitting, column access with documented neutral fallback."""

    name: RankerName = RankerName.RANDOM

    def __init__(self) -> None:
        """Baselines carry no state beyond the warnings they raise about missing columns."""
        self.warnings: tuple[str, ...] = ()

    def fit(
        self,
        frame: FeatureFrame,
        relevance: np.ndarray,
        sample_weight: np.ndarray | None = None,
        seed: int = 42,
    ) -> "Baseline":
        """No-op: a baseline is a fixed function of the frame, so there is nothing to learn."""
        return self

    def requires_fit(self) -> bool:
        """False for every baseline; the benchmark runner uses this to skip training."""
        return False

    def params(self) -> dict[str, float | int]:
        """Constructor arguments that define this baseline, for persistence.

        Empty for the baselines that are pure functions of the frame; the banded and the
        random baselines override it so a saved run can be reproduced exactly.
        """
        return {}

    def save(self, path: str | Path) -> None:
        """Record which baseline this is and the parameters that define its ordering."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ranker": self.name.value, "params": self.params()}
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Baseline":
        """Reconstruct a baseline from its recorded parameters."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(str(target))
        payload = json.loads(target.read_text(encoding="utf-8"))
        return cls(**payload.get("params", {}))

    # -- helpers ------------------------------------------------------------

    def column(self, frame: FeatureFrame, name: str) -> np.ndarray:
        """A column as floats, or its documented neutral when the ablation dropped it."""
        if name in frame.X.columns:
            return np.asarray(frame.X[name].to_numpy(), dtype=float)
        message = (
            f"{self.name.value}: column {name!r} is absent in ablation cell "
            f"{frame.flags.label()}; scoring its neutral value {NEUTRAL[name]}"
        )
        if message not in self.warnings:
            self.warnings = self.warnings + (message,)
            _LOG.warning("%s", message)
        return np.full(len(frame.finding_ids), NEUTRAL[name], dtype=float)


@register_ranker(RankerName.CVSS_ONLY)
class CvssOnlyRanker(Baseline):
    """CVSS base score alone: what most tools ship, and the ordering Gap 3 indicts."""

    name: RankerName = RankerName.CVSS_ONLY

    def score(self, frame: FeatureFrame) -> np.ndarray:
        """Highest CVSS base score across the finding's CVEs."""
        return self.column(frame, "cvss_base_max")


@register_ranker(RankerName.EPSS_ONLY)
class EpssOnlyRanker(Baseline):
    """EPSS probability alone: exploitation likelihood with no notion of consequence."""

    name: RankerName = RankerName.EPSS_ONLY

    def score(self, frame: FeatureFrame) -> np.ndarray:
        """EPSS score, percentile used only to break exact ties."""
        return self.column(frame, "b_epss") + 1e-6 * self.column(frame, "b_epss_percentile")


@register_ranker(RankerName.KEV_FIRST)
class KevFirstRanker(Baseline):
    """Everything on the CISA KEV catalogue first, CVSS order within each band."""

    name: RankerName = RankerName.KEV_FIRST

    def score(self, frame: FeatureFrame) -> np.ndarray:
        """``KEV_BAND`` for KEV membership plus a bounded CVSS tie-break."""
        return KEV_BAND * self.column(frame, "b_kev") + _TIE_CVSS * self.column(
            frame, "cvss_base_max"
        )


@register_ranker(RankerName.SCANNER_SEVERITY)
class ScannerSeverityRanker(Baseline):
    """The scanner's own severity: the ordering an unassisted analyst inherits."""

    name: RankerName = RankerName.SCANNER_SEVERITY

    def score(self, frame: FeatureFrame) -> np.ndarray:
        """Severity ordinal plus at most half a step of scanner confidence."""
        return self.column(frame, "scanner_severity_ord") + _TIE_CONFIDENCE * self.column(
            frame, "scanner_confidence"
        )


@register_ranker(RankerName.EXPECTED_LOSS)
class ExpectedLossRanker(Baseline):
    """The decision-theoretic ordering: ``P(exploit) x impact`` (Gap 1).

    Retained as a first-class baseline on purpose. If the learned ranker cannot beat the
    construct it was trained to approximate, that is a result worth reporting rather than
    a bug worth hiding.
    """

    name: RankerName = RankerName.EXPECTED_LOSS

    def score(self, frame: FeatureFrame) -> np.ndarray:
        """``log1p(expected_loss)``: monotone in expected loss, so the order is exact."""
        return self.column(frame, "b_expected_loss_log")


@register_ranker(RankerName.VMC_CHAIN)
class VmcChainRanker(Baseline):
    """Shimizu and Hashimoto's vulnerability management chain.

    Three bands, evaluated in order:

    1. **exploited** (score 2): on the CISA KEV catalogue, or ``EPSS >= 0.088``;
    2. **severe** (score 1): not in band 1, but ``CVSS >= 7.0``;
    3. **rest** (score 0).

    Within a band, ``0.5 * epss + 0.04 * cvss`` orders the findings and is bounded by 0.9
    so it can never promote a finding into the band above. The framework reports
    efficiency and coverage against this ordering because it is the published baseline
    the review's chain-aware argument is set against.
    """

    name: RankerName = RankerName.VMC_CHAIN

    def __init__(
        self,
        epss_threshold: float = VMC_EPSS_THRESHOLD,
        cvss_threshold: float = VMC_CVSS_THRESHOLD,
    ) -> None:
        """Thresholds default to the published cuts and to ``EvaluationConfig``'s values."""
        super().__init__()
        self.epss_threshold = float(epss_threshold)
        self.cvss_threshold = float(cvss_threshold)

    def params(self) -> dict[str, float | int]:
        """The two band cuts, so a sensitivity analysis round-trips."""
        return {"epss_threshold": self.epss_threshold, "cvss_threshold": self.cvss_threshold}

    def score(self, frame: FeatureFrame) -> np.ndarray:
        """Band score plus the bounded within-band tie-break."""
        kev = self.column(frame, "b_kev")
        epss = self.column(frame, "b_epss")
        cvss = self.column(frame, "cvss_base_max")

        exploited = (kev >= 0.5) | (epss >= self.epss_threshold)
        severe = (~exploited) & (cvss >= self.cvss_threshold)
        bands = np.where(
            exploited, VMC_BAND_EXPLOITED, np.where(severe, VMC_BAND_SEVERE, VMC_BAND_REST)
        )
        return bands + _TIE_EPSS * epss + _TIE_CVSS * cvss


@register_ranker(RankerName.RANDOM)
class RandomRanker(Baseline):
    """Seeded uniform noise: the control every other ordering has to beat.

    The score is a hash of ``(seed, finding_id)`` rather than a draw from a stream, so it
    is stable under re-ordering, batching and re-runs. Two rankers with the same seed
    always agree; different seeds give independent orderings.
    """

    name: RankerName = RankerName.RANDOM

    def __init__(self, seed: int = 42) -> None:
        """``seed`` selects the permutation; the same seed always gives the same order."""
        super().__init__()
        self.seed = int(seed)

    def params(self) -> dict[str, float | int]:
        """The seed, without which a reloaded control would be a different control."""
        return {"seed": self.seed}

    def score(self, frame: FeatureFrame) -> np.ndarray:
        """A deterministic uniform draw in ``[0, 1)`` per finding id."""
        return np.array(
            [self._uniform(finding_id) for finding_id in frame.finding_ids], dtype=float
        )

    def _uniform(self, finding_id: str) -> float:
        """Hash ``(seed, finding_id)`` into ``[0, 1)`` with 53 bits of resolution."""
        digest = hashlib.blake2b(
            f"{self.seed}:{finding_id}".encode("utf-8"), digest_size=8
        ).digest()
        return (int.from_bytes(digest, "big") >> 11) / float(1 << 53)
