"""Shrink model output toward the deterministic baseline when the two disagree.

A language model that disagrees mildly with the heuristic is probably reading evidence
the lexicon cannot; a model that disagrees wildly is more likely to have been argued
into it by the text it was asked to describe. Rather than choose, this guard interpolates:
beyond ``max_divergence`` the model value is pulled ``consistency_shrinkage`` of the way
back to the baseline, so a successful injection buys a bounded amount of movement no
matter how extreme the number it produced, and honest disagreement survives intact.

The shrinkage is recorded (``consistency_shrunk``, ``divergence``) so evaluation can
report how often, and how far, the model was overruled.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from vulnpriority.core.config import LLMConfig
from vulnpriority.core.interfaces import LLMResult
from vulnpriority.core.models import LLMAudit

__all__ = ["ConsistencyGuard", "ConsistencyOutcome"]

TModel = TypeVar("TModel", bound=BaseModel)


class ConsistencyOutcome(BaseModel):
    """What the guard did, in numbers the audit can carry."""

    model_config = {"frozen": True}

    divergence: float = 0.0
    shrunk: bool = False
    fields_shrunk: tuple[str, ...] = ()


def _numeric_fields(model: BaseModel) -> dict[str, float]:
    """Float fields of a model. Bools are excluded: they are categorical, not scalar."""
    out: dict[str, float] = {}
    for name, value in model.__dict__.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, float):
            out[name] = value
    return out


class ConsistencyGuard:
    """Compare a model answer with the heuristic baseline field by field."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        max_divergence: float | None = None,
        shrinkage: float | None = None,
    ) -> None:
        """``config`` supplies the defaults; explicit arguments win, for experiments."""
        config = config or LLMConfig()
        self.max_divergence = config.max_divergence if max_divergence is None else float(max_divergence)
        self.shrinkage = config.consistency_shrinkage if shrinkage is None else float(shrinkage)

    def reconcile(self, model_output: TModel, baseline: TModel) -> tuple[TModel, ConsistencyOutcome]:
        """Return a possibly shrunken copy of ``model_output`` plus what was done.

        Only fields present and numeric in both models take part. ``divergence`` is the
        largest absolute disagreement observed, whether or not it exceeded the threshold,
        because the untouched value is still worth reporting.
        """
        model_values = _numeric_fields(model_output)
        baseline_values = _numeric_fields(baseline)

        updates: dict[str, float] = {}
        shrunk_fields: list[str] = []
        worst = 0.0
        for name, model_value in model_values.items():
            if name not in baseline_values:
                continue
            baseline_value = baseline_values[name]
            gap = abs(model_value - baseline_value)
            worst = max(worst, gap)
            if gap > self.max_divergence:
                updates[name] = round(model_value + self.shrinkage * (baseline_value - model_value), 6)
                shrunk_fields.append(name)

        outcome = ConsistencyOutcome(
            divergence=round(worst, 6),
            shrunk=bool(updates),
            fields_shrunk=tuple(sorted(shrunk_fields)),
        )
        if not updates:
            return model_output, outcome
        return model_output.model_copy(update=updates), outcome

    def apply_to_audit(self, audit: LLMAudit, outcome: ConsistencyOutcome) -> LLMAudit:
        """Fold an outcome into an audit record."""
        return audit.model_copy(
            update={
                "consistency_shrunk": outcome.shrunk,
                "divergence": max(audit.divergence, outcome.divergence),
            }
        )

    def apply(self, result: LLMResult, baseline: LLMResult) -> LLMResult:
        """Reconcile a whole :class:`LLMResult` against the baseline result."""
        parsed, outcome = self.reconcile(result.parsed, baseline.parsed)
        return LLMResult(
            parsed=parsed,
            raw_text=result.raw_text,
            audit=self.apply_to_audit(result.audit, outcome),
        )
