"""Competitor rankers re-implemented from the literature review (Section 2 of the paper).

The seven baselines in :mod:`vulnpriority.rank.baselines` are the orderings practitioners
use. These are the orderings *researchers* proposed, one or more per paradigm the review
classifies, re-implemented so they run on the identical :class:`FeatureFrame`, folds,
labels and metrics as everything else (Gap 4). None of the original systems is public in a
form that runs on web-scanner output, so each is an adaptation: the decision logic and the
inputs are the paper's, the columns are the nearest ones this framework already computes,
and a competitor is never handed an input its paper does not assume. In particular no
competitor sees Component A's intel features, Component B's attacker probability or
expected loss, or Component C's monetary reachability.

==================  ==========  ===========================================================
ranker              source      ordering
==================  ==========  ===========================================================
``ssvc``            [11]        CISA SSVC deployer tree: exploitation, exposure,
                                automatable, mission impact -> Act / Attend / Track* / Track
``cavp``            [53]        CVSS temporal score: base x exploit-code-maturity x
                                remediation-level x report-confidence
``albanese``        [39]        exponential-decay exploitation likelihood x exposure factor
``vulnscore``       [60]        EPSS percentile, CVSS and asset criticality, equally weighted
``context_xgb``     [45]        context-aware XGBoost classifier, ranked by P(exploited)
``vpss_rf``         [15]        random forest over CVSS, relevance, protection level and
                                asset criticality
``licality``        [30]        neural likelihood x CVSS criticality
``topology_epss``   [66]        EPSS-weighted structural importance in the attack graph
==================  ==========  ===========================================================

The learned competitors (``context_xgb``, ``vpss_rf``, ``licality``) are trained on each
fold's training side exactly like LambdaMART, on binary "exploited" targets because that is
what their papers predict. If a training fold has a single class they fall back to the
paper's closed-form prior rather than failing.
"""

from __future__ import annotations

import numpy as np

from vulnpriority.core.enums import ExploitMaturity, RankerName
from vulnpriority.core.models import FeatureFrame
from vulnpriority.core.registry import register_ranker
from vulnpriority.rank.baselines import Baseline

__all__ = [
    "SsvcRanker",
    "CavpRanker",
    "AlbaneseRanker",
    "VulnScoreRanker",
    "ContextXgbRanker",
    "VpssRandomForestRanker",
    "LicalityRanker",
    "TopologyEpssRanker",
    "LITERATURE_RANKERS",
]

#: CVSS v3.1 temporal multipliers for Exploit Code Maturity, indexed by ExploitMaturity.
#: UNKNOWN maps to "Not Defined" (1.0), as the specification requires.
_ECM = {
    ExploitMaturity.UNKNOWN: 1.0,
    ExploitMaturity.UNPROVEN: 0.91,
    ExploitMaturity.POC: 0.94,
    ExploitMaturity.FUNCTIONAL: 0.97,
    ExploitMaturity.WEAPONIZED: 1.0,
}
#: Report Confidence: "Confirmed" when every CVSS source agrees, else "Reasonable".
_RC_CONFIRMED, _RC_REASONABLE = 1.0, 0.96


class _Closed(Baseline):
    """A closed-form competitor: no fitting, same column access as the baselines."""


class _Learned(Baseline):
    """A competitor that trains on the fold's training side."""

    #: Columns the competitor's paper assumes are available.
    columns: tuple[str, ...] = ()

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.seed = int(seed)
        self._model = None

    def requires_fit(self) -> bool:
        return True

    def params(self) -> dict[str, float | int]:
        return {"seed": self.seed}

    def matrix(self, frame: FeatureFrame) -> np.ndarray:
        return np.column_stack([self.column(frame, name) for name in self.columns])

    def fit(self, frame, relevance, sample_weight=None, seed: int = 42):
        y = (np.asarray(relevance, dtype=float) > 0.0).astype(int)
        self._model = None
        if len(np.unique(y)) < 2:
            return self
        self._model = self.make_model(int(seed))
        self._model.fit(self.matrix(frame), y)
        return self

    def score(self, frame: FeatureFrame) -> np.ndarray:
        if self._model is None:
            return self.prior(frame)
        return self._model.predict_proba(self.matrix(frame))[:, 1]

    def make_model(self, seed: int):  # pragma: no cover - abstract
        raise NotImplementedError

    def prior(self, frame: FeatureFrame) -> np.ndarray:
        """Closed-form fallback when a fold cannot be trained: CVSS order."""
        return self.column(frame, "cvss_base_max")


# ---------------------------------------------------------------- statistical / analytical


@register_ranker(RankerName.CAVP)
class CavpRanker(_Closed):
    """Jung, Li and Bechor [53]: CVSS temporal score from exploit and report evidence.

    CAVP derives Exploit Code Maturity, Remediation Level and Report Confidence from NVD
    references. Here maturity comes from the curated exploit feeds, confidence from CVSS
    source agreement, and remediation level is "Not Defined" because scanner output carries
    no patch-availability signal.
    """

    name: RankerName = RankerName.CAVP

    def score(self, frame: FeatureFrame) -> np.ndarray:
        base = self.column(frame, "cvss_base_max")
        maturity = self.column(frame, "b_exploit_maturity_feed_ord").round().clip(0, 4)
        ecm = np.array([_ECM[ExploitMaturity(int(m))] for m in maturity])
        agree = self.column(frame, "cvss_source_agreement")
        rc = np.where(agree >= 0.999, _RC_CONFIRMED, _RC_REASONABLE)
        # Round up to one decimal as the specification does; the tie-break keeps the
        # unrounded value so two findings with equal temporal scores stay in base order.
        temporal = np.ceil(base * ecm * rc * 10.0) / 10.0
        return temporal + 1e-3 * base


@register_ranker(RankerName.ALBANESE)
class AlbaneseRanker(_Closed):
    """Albanese, Iganibo and Adebiyi [39]: exponential-decay likelihood x exposure factor.

    Likelihood grows with the CVSS exploitability sub-metrics and exploit availability and
    decays with vulnerability age; the exposure factor is the CVSS impact sub-score.
    """

    name: RankerName = RankerName.ALBANESE

    #: Decay half-life of exploitation interest, in days.
    half_life_days: float = 365.0

    def score(self, frame: FeatureFrame) -> np.ndarray:
        exploitability = (
            self.column(frame, "cvss_ac_low")
            + self.column(frame, "cvss_pr_none")
            + self.column(frame, "cvss_ui_none")
        ) / 3.0
        exploit = (self.column(frame, "b_exploit_count") > 0).astype(float)
        age = self.column(frame, "vuln_age_days").clip(0, None)
        rate = 1.0 + 2.0 * exploitability + 3.0 * exploit
        decay = np.exp(-np.log(2.0) * age / self.half_life_days)
        likelihood = 1.0 - np.exp(-rate * (0.25 + 0.75 * decay))
        exposure = (
            self.column(frame, "cvss_c_high")
            + self.column(frame, "cvss_i_high")
            + self.column(frame, "cvss_a_high")
        ) / 3.0
        return likelihood * (0.1 + exposure) + 1e-3 * self.column(frame, "cvss_base_max")


# ---------------------------------------------------------------- hybrid, rule-based


@register_ranker(RankerName.SSVC)
class SsvcRanker(_Closed):
    """CISA SSVC deployer decision tree [11].

    Decision points: Exploitation (none / PoC / active), Exposure (small / controlled /
    open), Automatable (yes / no) and Mission and Well-being impact (low / medium / high),
    mapped to the four SSVC outcomes Act (3), Attend (2), Track* (1) and Track (0). CVSS
    orders findings within an outcome, as CISA's guidance suggests.
    """

    name: RankerName = RankerName.SSVC

    def score(self, frame: FeatureFrame) -> np.ndarray:
        kev = self.column(frame, "b_kev") >= 0.5
        poc = self.column(frame, "b_exploit_count") > 0
        exploitation = np.where(kev, 2, np.where(poc, 1, 0))
        auth = self.column(frame, "auth_required_ord")
        exposure = np.where(auth <= 0, 2, np.where(auth <= 1, 1, 0))
        automatable = (
            (self.column(frame, "cvss_ac_low") >= 0.5)
            & (self.column(frame, "cvss_pr_none") >= 0.5)
            & (self.column(frame, "cvss_ui_none") >= 0.5)
        )
        criticality = self.column(frame, "a_asset_criticality")
        impact = np.where(criticality >= 0.67, 2, np.where(criticality >= 0.34, 1, 0))

        outcome = np.zeros(len(frame.finding_ids))
        for i in range(len(outcome)):
            e, x, a, m = exploitation[i], exposure[i], bool(automatable[i]), impact[i]
            if e == 2:
                outcome[i] = 3 if (a or x == 2 or m == 2) else 2
            elif e == 1:
                outcome[i] = 2 if (m == 2 and (a or x == 2)) else 1
            else:
                outcome[i] = 1 if (m == 2 and a and x == 2) else 0
        return outcome + 0.04 * self.column(frame, "cvss_base_max")


@register_ranker(RankerName.VULNSCORE)
class VulnScoreRanker(_Closed):
    """Alqahtani and Almukaynizi [60]: EPSS percentile, CVSS and asset criticality.

    VulnScore combines the three with user-defined weights; equal weights are used here,
    the neutral choice when no organisation has tuned them.
    """

    name: RankerName = RankerName.VULNSCORE

    def score(self, frame: FeatureFrame) -> np.ndarray:
        return (
            self.column(frame, "b_epss_percentile")
            + self.column(frame, "cvss_base_max") / 10.0
            + self.column(frame, "a_asset_criticality")
        ) / 3.0


# ---------------------------------------------------------------- graph-based


@register_ranker(RankerName.TOPOLOGY_EPSS)
class TopologyEpssRanker(_Closed):
    """Tiță et al. [66]: EPSS-weighted structural importance in the attack graph.

    Tiță et al. rank by Kirchhoff effective resistance with EPSS edge conductances. The
    structural part is approximated by the finding's betweenness and best path probability
    in the same attack graph Component C builds; unlike Component C it carries no monetary
    asset value, which is the distinction the paper's comparison is meant to test.
    """

    name: RankerName = RankerName.TOPOLOGY_EPSS

    def score(self, frame: FeatureFrame) -> np.ndarray:
        conductance = 0.05 + self.column(frame, "b_epss")
        structure = self.column(frame, "c_betweenness") + self.column(frame, "c_max_path_prob")
        return conductance * (0.1 + structure)


# ---------------------------------------------------------------- learned


_CVSS = ("cvss_base_max", "cvss_ac_low", "cvss_pr_none", "cvss_ui_none",
         "cvss_c_high", "cvss_i_high", "cvss_a_high")


@register_ranker(RankerName.CONTEXT_XGB)
class ContextXgbRanker(_Learned):
    """Sevimli Deniz and Koca [45]: context-aware XGBoost exploitation classifier.

    CVSS, EPSS, asset context and vulnerability age, the four feature groups of their
    ablation, ranked by predicted probability for Top-K selection.
    """

    name: RankerName = RankerName.CONTEXT_XGB
    columns = _CVSS + ("b_epss", "a_asset_criticality", "a_exposure", "a_data_sensitivity",
                       "vuln_age_days")

    def make_model(self, seed: int):
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=seed, n_jobs=1,
                             eval_metric="logloss", tree_method="hist")

    def prior(self, frame: FeatureFrame) -> np.ndarray:
        return self.column(frame, "b_epss") + 0.01 * self.column(frame, "cvss_base_max")


@register_ranker(RankerName.VPSS_RF)
class VpssRandomForestRanker(_Learned):
    """Hore et al. [15]: random-forest Vulnerability Priority Scoring System.

    CVSS plus the three organisational inputs VPSS adds: relevance (applicability),
    protection level (authentication in front of the endpoint) and asset criticality.
    """

    name: RankerName = RankerName.VPSS_RF
    columns = _CVSS + ("a_p_applicable", "auth_required_ord", "a_asset_criticality")

    def make_model(self, seed: int):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed,
                                      n_jobs=1, class_weight="balanced")


@register_ranker(RankerName.LICALITY)
class LicalityRanker(_Learned):
    """Zeng et al. [30]: neural exploitation likelihood x CVSS criticality.

    LICALITY learns likelihood with a neural network over vulnerability features and takes
    criticality from the CVSS impact vector through probabilistic logic; the product orders
    the queue.
    """

    name: RankerName = RankerName.LICALITY
    columns = _CVSS + ("b_epss", "b_exploit_count", "vuln_age_days", "scanner_severity_ord")

    def make_model(self, seed: int):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16),
                             max_iter=800, alpha=1e-3, random_state=seed))

    def score(self, frame: FeatureFrame) -> np.ndarray:
        likelihood = super().score(frame)
        criticality = (
            self.column(frame, "cvss_c_high")
            + self.column(frame, "cvss_i_high")
            + self.column(frame, "cvss_a_high")
        ) / 3.0
        return likelihood * (0.25 + criticality)


LITERATURE_RANKERS: tuple[RankerName, ...] = (
    RankerName.SSVC,
    RankerName.CAVP,
    RankerName.ALBANESE,
    RankerName.VULNSCORE,
    RankerName.CONTEXT_XGB,
    RankerName.VPSS_RF,
    RankerName.LICALITY,
    RankerName.TOPOLOGY_EPSS,
)
