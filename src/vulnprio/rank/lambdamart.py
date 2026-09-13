"""``LambdaMartRanker``: the learned ordering (DESIGN.md 3.8, Architecture item 3).

XGBoost's ``rank:ndcg`` objective with scans as query groups. Three design decisions are
worth stating plainly because they are what make the learned ranker defensible rather
than merely accurate:

**Monotone constraints.** ``RankingConfig.monotone`` forces the score to be
non-decreasing in KEV membership, EPSS, expected loss and chain contribution. This is a
correctness constraint first and a security control second: it is not merely implausible
for a model to rank a KEV-listed finding lower *because* it is KEV-listed, it is now
impossible, which bounds what a prompt injection reaching those features can achieve. The
constraint vector is built against the columns actually present, so an ablation cell that
drops Component B simply carries fewer constraints rather than failing to fit.

The rule for what may join that set: **a monotone constraint belongs only on a feature no
untrusted tier can reach.** Every current member is a curated feed fact or the framework's
own arithmetic over operator configuration - CISA, FIRST, the attacker model, the attack
graph - and none of them can be moved by text the target application or a web page
authored. That is precisely what makes the constraint a safeguard. Put one on a feature an
attacker can set and the guarantee inverts: a constraint is a *promise* the model will
never score downwards on that input, so on attacker-reachable input it becomes a
guaranteed, model-independent lever. This is why the ``a_intel_*`` group carries no
constraints despite ``a_intel_corroborates_feeds`` looking like a natural candidate - it is
derived from ``REFERENCE_PAGE`` material, and the exploitation evidence it corroborates is
already constrained through ``b_kev``, where an injection cannot follow it.

**Degenerate input does not crash and does not lie.** A single query group, or labels
that are constant within every group, gives LambdaMART no pair to learn from; XGBoost
happily fits and returns an all-zero score, which would silently become an arbitrary
ordering. Assessing one application is exactly that case, and it is the ordinary
operational one: one scan is one query group, so nothing can be learned from it no matter
how many findings it holds. That is arithmetic, not a shortcoming of the implementation.

**So a model trained elsewhere scores it.** Ranking is fitted once, on a corpus that has
several scans and confirmed-exploitation labels, and persisted; a run over a single scan
loads that booster and *scores* with it. Scoring needs neither labels nor query groups, so
this is genuinely the learned model doing the ordering rather than a stand-in wearing its
name. The persisted column list must match the frame exactly - a booster fitted for a
different ablation cell would otherwise score the wrong features silently - and a mismatch
is refused rather than worked around.

Only when there is no compatible trained model does the ranker fall back to the
decision-theoretic ordering - expected loss, the construct of Gap 1 - and in that case
:attr:`used_fallback` is set, a warning is recorded, and every layer above reports it. The
one thing that must never happen is the framework presenting expected-loss ordering as a
learned ranking, which is what it used to do.

**Weights are per query group.** DESIGN.md specifies impact-weighted pairs, and XGBoost's
ranking objective accepts exactly one weight per query group. ``fit`` therefore takes the
per-row weights the caller computed and aggregates them by mean within each group,
recording that it did so; a caller that already has per-group weights may pass those
instead and they are used unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from vulnprio.core.config import RankingConfig
from vulnprio.core.enums import RankerName
from vulnprio.core.errors import RankerNotFittedError, VulnprioError
from vulnprio.core.interfaces import Ranker
from vulnprio.core.models import Explanation, FeatureFrame
from vulnprio.core.registry import register_ranker

__all__ = ["FALLBACK_COLUMNS", "LambdaMartRanker"]

_LOG = logging.getLogger(__name__)

#: Columns tried, in order, when the learned model cannot be used. ``b_expected_loss_log``
#: first because it *is* the construct definition of priority; the rest are progressively
#: weaker stand-ins for ablation cells where Components B and C are switched off.
FALLBACK_COLUMNS: tuple[str, ...] = (
    "b_expected_loss_log",
    "b_p_exploit_attacker",
    "c_reach_delta_log",
    "cvss_base_max",
    "scanner_severity_ord",
)


@register_ranker(RankerName.LAMBDAMART)
class LambdaMartRanker(Ranker):
    """LambdaMART over the :class:`FeatureFrame`, grouped by scan."""

    name: RankerName = RankerName.LAMBDAMART

    def __init__(self, config: RankingConfig | None = None) -> None:
        """``config`` supplies every hyper-parameter; the defaults are DESIGN.md's."""
        self.config: RankingConfig = config if config is not None else RankingConfig()
        self.columns: list[str] = []
        self.seed: int = int(self.config.seed)
        self.warnings: tuple[str, ...] = ()
        self.used_fallback: bool = False
        #: True when the booster doing the scoring was fitted on another corpus and loaded
        #: rather than fitted on this frame. The ordering is still the learned model's; the
        #: distinction matters for provenance, not for whether XGBoost ran.
        self.used_pretrained: bool = False
        #: Why the learned model could not be fitted here, when it could not. Empty
        #: otherwise. Carried up into ``RankingResult`` so a reader is never left guessing
        #: which of the two orderings produced the queue in front of them.
        self.fallback_reason: str = ""
        self.group_weight_aggregated: bool = False
        self._model: xgb.XGBRanker | None = None

    # -- fitting ------------------------------------------------------------

    def fit(
        self,
        frame: FeatureFrame,
        relevance: np.ndarray,
        sample_weight: np.ndarray | None = None,
        seed: int = 42,
    ) -> "LambdaMartRanker":
        """Fit on graded relevance (0-4), grouping rows by scan.

        ``sample_weight`` may be per row (aggregated to per group by mean, as XGBoost's
        ranking objective requires exactly one weight per query) or already per group.
        Degenerate input is not an error: it sets :attr:`used_fallback` and a warning.
        """
        labels = np.asarray(relevance, dtype=float).reshape(-1)
        if labels.shape[0] != len(frame.finding_ids):
            raise VulnprioError(
                f"relevance has {labels.shape[0]} entries for {len(frame.finding_ids)} rows"
            )

        self.columns = list(frame.feature_names)
        self.seed = int(seed)
        self.warnings = ()
        self.used_fallback = False
        self.used_pretrained = False
        self.fallback_reason = ""
        self.group_weight_aggregated = False

        groups = frame.group_sizes()
        reason = self._degenerate_reason(labels, groups)
        if reason is not None:
            # Nothing can be learned from this frame, which is the ordinary case for a
            # single application. Score it with a model that was trained where learning was
            # possible; only if there is none does the ordering stop being the model's.
            if self._adopt_trained_model(reason):
                return self
            self._fall_back(reason)
            return self

        weights = self._group_weights(sample_weight, groups)
        model = xgb.XGBRanker(**self._xgb_params())
        model.fit(
            frame.X,
            self._as_grades(labels),
            group=groups,
            sample_weight=weights,
            verbose=False,
        )
        self._model = model
        return self

    # -- scoring ------------------------------------------------------------

    def score(self, frame: FeatureFrame) -> np.ndarray:
        """Raw ranking scores, one per row; higher means remediate sooner.

        Not a probability and not comparable across scans - LambdaMART optimises the
        ordering within a query group, so only the within-scan order is meaningful.
        """
        if not self.columns:
            raise RankerNotFittedError("LambdaMartRanker.score called before fit")
        self._check_columns(frame)
        if self._model is None:
            return self._fallback_score(frame)
        return np.asarray(self._model.predict(frame.X), dtype=float)

    def explain(self, frame: FeatureFrame, top_n: int = 5) -> list[Explanation] | None:
        """SHAP explanations for the fitted booster, or ``None`` when running on fallback."""
        if self._model is None:
            return None
        from vulnprio.rank.explain import ShapExplainer  # local import: avoids a cycle

        return ShapExplainer(self, top_n=top_n).explain(frame)

    def requires_fit(self) -> bool:
        """True: this is the one ranker in the framework that learns."""
        return True

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the booster to ``path`` and its metadata to ``path + '.meta.json'``.

        The column list travels with the booster because a frame built for a different
        ablation cell would otherwise be scored against the wrong features silently.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {
            "ranker": self.name.value,
            "columns": self.columns,
            "seed": self.seed,
            "warnings": list(self.warnings),
            "used_fallback": self.used_fallback,
            "used_pretrained": self.used_pretrained,
            "fallback_reason": self.fallback_reason,
            "group_weight_aggregated": self.group_weight_aggregated,
            "config": self.config.model_dump(mode="json"),
            "has_booster": self._model is not None,
        }
        if self._model is not None:
            self._model.save_model(str(target))
        elif target.exists():
            target.unlink()
        self._meta_path(target).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LambdaMartRanker":
        """Reconstruct a saved ranker: booster, column list and configuration."""
        target = Path(path)
        meta_path = cls._meta_path(target)
        if not meta_path.exists():
            raise VulnprioError(f"no ranker metadata beside {target} (expected {meta_path})")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        ranker = cls(RankingConfig.model_validate(meta.get("config", {})))
        ranker.columns = list(meta.get("columns", []))
        ranker.seed = int(meta.get("seed", ranker.config.seed))
        ranker.warnings = tuple(meta.get("warnings", ()))
        ranker.used_fallback = bool(meta.get("used_fallback", False))
        ranker.used_pretrained = bool(meta.get("used_pretrained", False))
        ranker.fallback_reason = str(meta.get("fallback_reason", ""))
        ranker.group_weight_aggregated = bool(meta.get("group_weight_aggregated", False))
        if meta.get("has_booster", False):
            if not target.exists():
                raise VulnprioError(f"ranker metadata claims a booster but {target} is missing")
            model = xgb.XGBRanker(**ranker._xgb_params())
            model.load_model(str(target))
            ranker._model = model
        return ranker

    # -- introspection ------------------------------------------------------

    @property
    def booster(self) -> xgb.Booster | None:
        """The fitted booster, or ``None`` when the ranker is on the fallback path."""
        return None if self._model is None else self._model.get_booster()

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has run, whether or not a booster was learned."""
        return bool(self.columns)

    def monotone_vector(self, columns: list[str] | None = None) -> tuple[int, ...]:
        """``RankingConfig.monotone`` projected onto the columns actually present."""
        names = columns if columns is not None else self.columns
        return tuple(int(self.config.monotone.get(name, 0)) for name in names)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _meta_path(target: Path) -> Path:
        return target.with_name(target.name + ".meta.json")

    def _xgb_params(self) -> dict[str, Any]:
        """Estimator keyword arguments, with the monotone vector for the current columns."""
        config = self.config
        params: dict[str, Any] = {
            "objective": config.objective,
            "eval_metric": config.eval_metric,
            "lambdarank_pair_method": config.lambdarank_pair_method,
            "lambdarank_num_pair_per_sample": config.lambdarank_num_pair_per_sample,
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
            "verbosity": 0,
        }
        vector = self.monotone_vector()
        if vector and any(vector):
            params["monotone_constraints"] = "(" + ",".join(str(value) for value in vector) + ")"
        return params

    @staticmethod
    def _as_grades(labels: np.ndarray) -> np.ndarray:
        """Relevance as the non-negative integers ``rank:ndcg`` expects."""
        return np.clip(np.rint(labels), 0, None).astype(int)

    def _degenerate_reason(self, labels: np.ndarray, groups: np.ndarray) -> str | None:
        """Why LambdaMART cannot learn here, or ``None`` when it can.

        Two conditions, both of which leave the objective with no ranking pair: fewer than
        two query groups, and constant relevance inside every group.
        """
        if groups.size < 2:
            return f"only {int(groups.size)} query group in the training frame"
        offset = 0
        varied = 0
        for size in groups:
            block = labels[offset : offset + int(size)]
            offset += int(size)
            if block.size and float(np.max(block)) > float(np.min(block)):
                varied += 1
        if varied == 0:
            return "relevance is constant inside every query group"
        return None

    def _adopt_trained_model(self, reason: str) -> bool:
        """Load a previously fitted booster to score this frame, if one fits it.

        Returns True when the ordering will be the learned model's after all. "Fits" means
        the saved column list is *identical* to this frame's, in order: a booster trained
        with Component B present scores a B-ablated frame as though the missing columns were
        something else entirely, and silently. A mismatch is therefore a refusal with a
        message naming both shapes, not a best effort.
        """
        path = getattr(self.config, "model_path", None)
        if not path:
            return False
        target = Path(path)
        if not target.exists() and not self._meta_path(target).exists():
            return False
        try:
            trained = type(self).load(target)
        except Exception as error:  # noqa: BLE001 - a bad cache must not fail a run
            _LOG.warning("could not load the trained ranker at %s: %s", target, error)
            return False
        if trained._model is None:
            return False
        if list(trained.columns) != list(self.columns):
            message = (
                f"the trained ranker at {target} was fitted on {len(trained.columns)} "
                f"feature(s) and this run builds {len(self.columns)}, so it cannot score "
                "this frame; re-train it for this component configuration"
            )
            self.warnings = self.warnings + (message,)
            _LOG.warning("%s", message)
            return False

        self._model = trained._model
        self.used_pretrained = True
        self.used_fallback = False
        message = (
            f"nothing can be learned from this frame ({reason}), so the ranking is scored "
            f"with the model trained at {target}"
        )
        self.warnings = self.warnings + (message,)
        _LOG.info("%s", message)
        return True

    def _fall_back(self, reason: str) -> None:
        """Record the degenerate condition and arm the expected-loss ordering."""
        message = (
            f"LambdaMART cannot learn a ranking ({reason}) and no trained model is "
            "available to score it; falling back to expected-loss ordering"
        )
        self._model = None
        self.used_fallback = True
        self.fallback_reason = reason
        self.warnings = self.warnings + (message,)
        _LOG.warning("%s", message)

    def _group_weights(
        self, sample_weight: np.ndarray | None, groups: np.ndarray
    ) -> np.ndarray | None:
        """Per-group weights from per-row weights (mean), or the caller's own per-group array."""
        if sample_weight is None:
            return None
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weights.size == groups.size:
            return weights
        total = int(groups.sum())
        if weights.size != total:
            raise VulnprioError(
                f"sample_weight has {weights.size} entries; expected {total} rows "
                f"or {groups.size} groups"
            )
        aggregated = np.empty(groups.size, dtype=float)
        offset = 0
        for index, size in enumerate(groups):
            block = weights[offset : offset + int(size)]
            offset += int(size)
            aggregated[index] = float(block.mean()) if block.size else 1.0
        self.group_weight_aggregated = True
        _LOG.debug(
            "aggregated %d per-row sample weights into %d per-group weights",
            weights.size,
            groups.size,
        )
        return aggregated

    def _check_columns(self, frame: FeatureFrame) -> None:
        """Refuse to score a frame whose columns are not the ones the model was fitted on."""
        if list(frame.feature_names) != self.columns:
            raise VulnprioError(
                "feature columns changed since fit: "
                f"fitted on {len(self.columns)} columns, scoring {len(frame.feature_names)}"
            )

    def _fallback_score(self, frame: FeatureFrame) -> np.ndarray:
        """Expected-loss ordering, used when the learned model could not be fitted."""
        for column in FALLBACK_COLUMNS:
            if column in frame.X.columns:
                return np.asarray(frame.X[column].to_numpy(), dtype=float)
        return np.zeros(len(frame.finding_ids), dtype=float)
