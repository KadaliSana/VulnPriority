"""``CostSensitiveExploitHead``: a calibrated P(exploit) head (DESIGN.md 3.8, Gap 7).

The ranker gives an ordering; this gives a number. The evaluation protocol asks for
Brier score, expected calibration error and reliability bins, none of which a LambdaMART
margin can supply, and the selection layer wants a probability it can multiply by money.

Two properties matter, and both follow from the fact that exploitation is rare:

**Cost sensitivity.** With a few percent positives, an unweighted classifier learns to
say "no". ``scale_pos_weight`` is set to ``n_negative / n_positive`` when
``RankingConfig.head_scale_pos_weight`` leaves it unset, which is the standard correction
and makes the minority class carry the same total weight as the majority. Gap 7 asks for
exactly this, alongside MCC and minority-class F1 in the evaluation.

**Calibration.** A cost-weighted classifier is deliberately mis-calibrated - that is what
the weighting does - so the raw score is monotone but not a probability. The head fits
the classifier on one slice and a calibrator on a held-out slice: isotonic regression
(free-form, monotone, the default) or a Platt sigmoid (two parameters, better on small
samples). The held-out slice is stratified on the label so the calibrator sees positives,
and it is carved by a seeded shuffle so the whole thing is reproducible.

Degenerate inputs are handled rather than raised on: a single-class training set yields
the constant prior, and a held-out slice with only one class skips calibration and says
so in :attr:`warnings`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

from vulnpriority.core.config import RankingConfig
from vulnpriority.core.errors import RankerNotFittedError, VulnPriorityError
from vulnpriority.core.interfaces import ProbabilityModel
from vulnpriority.core.models import FeatureFrame

__all__ = ["CALIBRATION_METHODS", "CostSensitiveExploitHead"]

_LOG = logging.getLogger(__name__)

#: Accepted values of ``RankingConfig.head_calibration``.
CALIBRATION_METHODS: tuple[str, ...] = ("isotonic", "sigmoid", "none")

#: Probability clip. Exactly 0 or 1 makes log loss infinite and claims a certainty no
#: model over this evidence has earned.
_P_CLIP: tuple[float, float] = (1e-6, 1.0 - 1e-6)


class CostSensitiveExploitHead(ProbabilityModel):
    """Cost-weighted gradient-boosted classifier with held-out probability calibration."""

    def __init__(
        self,
        config: RankingConfig | None = None,
        *,
        calibration: str | None = None,
        holdout_fraction: float = 0.25,
    ) -> None:
        """``calibration`` overrides ``RankingConfig.head_calibration`` for one instance."""
        self.config: RankingConfig = config if config is not None else RankingConfig()
        method = (calibration if calibration is not None else self.config.head_calibration).lower()
        if method not in CALIBRATION_METHODS:
            raise VulnPriorityError(
                f"unknown calibration method {method!r}; expected one of {CALIBRATION_METHODS}"
            )
        self.calibration: str = method
        self.holdout_fraction: float = float(holdout_fraction)
        self.columns: list[str] = []
        self.seed: int = int(self.config.seed)
        self.scale_pos_weight: float = 1.0
        self.warnings: tuple[str, ...] = ()
        self.constant_probability: float | None = None
        self._model: XGBClassifier | None = None
        self._calibrator: IsotonicRegression | LogisticRegression | None = None

    # -- fitting ------------------------------------------------------------

    def fit(self, frame: FeatureFrame, y: np.ndarray, seed: int = 42) -> "CostSensitiveExploitHead":
        """Fit the classifier and its calibrator. ``y`` is binary exploitation ground truth."""
        labels = self._as_binary(y, len(frame.finding_ids))
        self.columns = list(frame.feature_names)
        self.seed = int(seed)
        self.warnings = ()
        self.constant_probability = None
        self._model = None
        self._calibrator = None

        positives = int(labels.sum())
        negatives = int(labels.size - positives)
        if positives == 0 or negatives == 0:
            prior = float(np.clip(positives / max(1, labels.size), *_P_CLIP))
            self.constant_probability = prior
            self._warn(
                f"training labels are single-class ({positives} positive of {labels.size}); "
                f"predicting the constant prior {prior:.4f}"
            )
            return self

        self.scale_pos_weight = (
            float(self.config.head_scale_pos_weight)
            if self.config.head_scale_pos_weight is not None
            else negatives / positives
        )

        train_index, holdout_index = self._holdout_split(labels)
        model = XGBClassifier(**self._xgb_params())
        model.fit(frame.X.iloc[train_index], labels[train_index], verbose=False)
        self._model = model

        if self.calibration != "none":
            self._fit_calibrator(
                self._raw_scores(frame.X.iloc[holdout_index]), labels[holdout_index]
            )
        return self

    # -- prediction ---------------------------------------------------------

    def predict_proba(self, frame: FeatureFrame) -> np.ndarray:
        """Calibrated ``P(exploit)`` per row, as a one-dimensional array in (0, 1).

        One dimension, not two: the framework only ever asks for the positive class, and
        the calibration report, the expected-loss ordering and the minority metrics all
        consume this shape directly.
        """
        if not self.columns:
            raise RankerNotFittedError("CostSensitiveExploitHead.predict_proba called before fit")
        if list(frame.feature_names) != self.columns:
            raise VulnPriorityError(
                "feature columns changed since fit: "
                f"fitted on {len(self.columns)} columns, predicting {len(frame.feature_names)}"
            )
        rows = len(frame.finding_ids)
        if self.constant_probability is not None:
            return np.full(rows, self.constant_probability, dtype=float)

        raw = self._raw_scores(frame.X)
        if self._calibrator is None:
            return np.clip(raw, *_P_CLIP)
        return np.clip(self._apply_calibrator(raw), *_P_CLIP)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _as_binary(y: np.ndarray, rows: int) -> np.ndarray:
        """Coerce labels to 0/1, accepting graded relevance as "positive when non-zero"."""
        values = np.asarray(y).reshape(-1)
        if values.shape[0] != rows:
            raise VulnPriorityError(f"y has {values.shape[0]} entries for {rows} rows")
        return (values.astype(float) > 0.0).astype(int)

    def _xgb_params(self) -> dict[str, Any]:
        """Classifier hyper-parameters, sharing the ranker's tree settings."""
        config = self.config
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "min_child_weight": config.min_child_weight,
            "reg_lambda": config.reg_lambda,
            "tree_method": config.tree_method,
            "n_jobs": config.n_jobs,
            "random_state": self.seed,
            "scale_pos_weight": self.scale_pos_weight,
            "verbosity": 0,
        }

    def _holdout_split(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Seeded, label-stratified train/calibration split.

        Falls back to training on everything (and calibrating on everything, which the
        caller is warned about) when the positive class is too small to stratify.
        """
        index = np.arange(labels.size)
        if self.calibration == "none":
            return index, index
        positives = int(labels.sum())
        negatives = int(labels.size - positives)
        holdout = int(round(self.holdout_fraction * labels.size))
        if holdout < 2 or min(positives, negatives) < 2 or holdout > labels.size - 2:
            self._warn(
                "training set is too small to hold out a stratified calibration slice; "
                "calibrating in-sample, which understates calibration error"
            )
            return index, index
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=holdout, random_state=self.seed
        )
        train_index, holdout_index = next(splitter.split(index.reshape(-1, 1), labels))
        return train_index, holdout_index

    def _raw_scores(self, X: Any) -> np.ndarray:
        """Uncalibrated positive-class probabilities from the booster."""
        if self._model is None:  # pragma: no cover - guarded by predict_proba
            raise RankerNotFittedError("no classifier fitted")
        return np.asarray(self._model.predict_proba(X), dtype=float)[:, 1]

    def _fit_calibrator(self, raw: np.ndarray, labels: np.ndarray) -> None:
        """Fit isotonic or Platt calibration on the held-out slice."""
        if np.unique(labels).size < 2:
            self._warn(
                "calibration slice contains a single class; leaving probabilities uncalibrated"
            )
            return
        if self.calibration == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(raw, labels.astype(float))
            self._calibrator = calibrator
            return
        logit = np.log(np.clip(raw, *_P_CLIP) / (1.0 - np.clip(raw, *_P_CLIP)))
        calibrator = LogisticRegression(solver="lbfgs", C=1e10, max_iter=1000)
        calibrator.fit(logit.reshape(-1, 1), labels)
        self._calibrator = calibrator

    def _apply_calibrator(self, raw: np.ndarray) -> np.ndarray:
        """Map uncalibrated scores through the fitted calibrator."""
        if isinstance(self._calibrator, IsotonicRegression):
            return np.asarray(self._calibrator.predict(raw), dtype=float)
        logit = np.log(np.clip(raw, *_P_CLIP) / (1.0 - np.clip(raw, *_P_CLIP)))
        assert self._calibrator is not None  # narrowed by predict_proba
        return np.asarray(self._calibrator.predict_proba(logit.reshape(-1, 1)), dtype=float)[:, 1]

    def _warn(self, message: str) -> None:
        """Record a degradation on the model and in the log, never silently."""
        self.warnings = self.warnings + (message,)
        _LOG.warning("%s", message)
