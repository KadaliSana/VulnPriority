"""Stages 5-7 of the sandbox: constrain what the model is allowed to have said.

Everything before this module tries to stop an instruction reaching the model. This module
assumes one got through anyway and asks a different question: of the things the model could
have been talked into saying, which ones can actually change a number in the pipeline?

* A value outside its bounds is clamped, not rejected, and the clamp is recorded - the
  alternative (a hard failure) hands an attacker a denial-of-service.
* A claim without a quotable span from the sanitized input is dropped, because a model that
  cannot point at the evidence has invented it.
* A reference to a finding outside the call's scope is a rejection, because there is no
  legitimate reason for one finding's assessment to name another.
* A canary in the output ends the call immediately.
* Whatever survives may still only move a feature as far as its tier's influence budget
  allows, and may never argue a value below a floor established by tier <= 1 evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any, Collection, Iterable, Mapping, Sequence

from pydantic import BaseModel, ValidationError

from vulnpriority.core.config import SandboxConfig
from vulnpriority.core.enums import UNTRUSTED_TIERS, InjectionCategory, TrustTier
from vulnpriority.core.errors import (
    CanaryLeakError,
    InfluenceBudgetExceeded,
    SchemaRejectedError,
)
from vulnpriority.core.models import InjectionSignal
from vulnpriority.sandbox.canary import canary_in_output

__all__ = [
    "OutputGuard",
    "clamp_influence",
    "assert_within_budget",
    "numeric_bounds",
    "span_matches",
]

_MAX_SNIPPET = 200
_WHITESPACE = re.compile(r"\s+")
_FINDING_REF = re.compile(r"\bf_[0-9A-Za-z]{2,}\b")
_STRICT_EPSILON = 1e-9


def _normalize_span(text: str) -> str:
    """Whitespace-normalised, case-folded form used for substring verification."""
    return _WHITESPACE.sub(" ", text).strip().casefold()


def span_matches(span: str, sanitized_inputs: Sequence[str]) -> bool:
    """True when ``span`` is a literal substring of one sanitized input.

    Only whitespace and case are normalised. Anything looser would let a model paraphrase
    its way to an "evidence" quote, which is exactly the failure this check exists for.
    """
    candidate = _normalize_span(span)
    if not candidate:
        return False
    return any(candidate in _normalize_span(source) for source in sanitized_inputs)


def numeric_bounds(schema: type[BaseModel]) -> dict[str, tuple[float | None, float | None, bool, bool]]:
    """Extract ``(low, high, low_is_strict, high_is_strict)`` per numeric field.

    Read from pydantic's own field metadata rather than a hand-maintained table, so the
    bounds the guard enforces can never drift from the bounds the schema declares.
    """
    bounds: dict[str, tuple[float | None, float | None, bool, bool]] = {}
    for name, field in schema.model_fields.items():
        low: float | None = None
        high: float | None = None
        low_strict = False
        high_strict = False
        for marker in field.metadata:
            ge = getattr(marker, "ge", None)
            gt = getattr(marker, "gt", None)
            le = getattr(marker, "le", None)
            lt = getattr(marker, "lt", None)
            if ge is not None:
                low, low_strict = float(ge), False
            if gt is not None:
                low, low_strict = float(gt), True
            if le is not None:
                high, high_strict = float(le), False
            if lt is not None:
                high, high_strict = float(lt), True
        if low is not None or high is not None:
            bounds[name] = (low, high, low_strict, high_strict)
    return bounds


def _max_lengths(schema: type[BaseModel]) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for name, field in schema.model_fields.items():
        for marker in field.metadata:
            max_length = getattr(marker, "max_length", None)
            if max_length is not None:
                lengths[name] = int(max_length)
    return lengths


def clamp_influence(
    base_value: float,
    proposed_value: float,
    tier: TrustTier,
    config: SandboxConfig,
    corroborated: bool = False,
    *,
    floor: float | None = None,
    lower: float = 0.0,
    upper: float = 1.0,
) -> tuple[float, float]:
    """Apply the influence budget to one proposed feature change.

    ``base_value`` is what the deterministic heuristic computed; ``proposed_value`` is what
    the untrusted-evidence path wants instead. The tier's budget caps how far the proposal
    may move the base, a corroborated proposal (two independent sources agreeing) gets the
    larger ``corroborated_budget``, and the result is clipped to the feature's own range.

    ``floor`` is the value implied by tier <= 1 evidence - KEV membership, a version match
    from a curated feed. Unless ``allow_downgrade_below_floor`` is set, an untrusted tier
    may not push the value below it: a blog post does not get to argue away CISA KEV.

    Returns ``(permitted_value, applied_delta)`` where ``applied_delta`` is measured from
    ``base_value``, so ``TrustSummary.influence_used`` can record it directly.
    """
    budget = (
        float(config.corroborated_budget)
        if corroborated
        else float(config.influence_budget.get(TrustTier(tier), 0.0))
    )
    budget = max(0.0, budget)

    delta = float(proposed_value) - float(base_value)
    if delta > budget:
        delta = budget
    elif delta < 0.0:
        # Deflation is the dangerous direction. An attacker who talks a real vulnerability
        # down leaves it unpatched; one who talks a harmless finding up only wastes effort.
        # The budget is therefore asymmetric for untrusted tiers, which is where the
        # adversarial corpus's surviving attacks were.
        down_budget = budget
        if TrustTier(tier) in UNTRUSTED_TIERS and not corroborated:
            down_budget = budget * float(config.deflation_budget_factor)
        if delta < -down_budget:
            delta = -down_budget

    permitted = float(base_value) + delta
    permitted = min(max(permitted, lower), upper)

    if floor is not None and not config.allow_downgrade_below_floor and TrustTier(tier) in UNTRUSTED_TIERS:
        permitted = max(permitted, float(floor))

    return permitted, permitted - float(base_value)


def assert_within_budget(
    applied_delta: float, tier: TrustTier, config: SandboxConfig, feature: str = "feature"
) -> None:
    """Raise :class:`InfluenceBudgetExceeded` when a delta breaks the tier's budget.

    Callers that compute a delta by another route (an enricher combining several signals)
    use this to assert the invariant the clamp above establishes by construction.
    """
    budget = float(config.influence_budget.get(TrustTier(tier), 0.0))
    if abs(applied_delta) > budget + 1e-9:
        raise InfluenceBudgetExceeded(
            f"{feature}: tier {TrustTier(tier).name} moved the value by {applied_delta:+.4f}, "
            f"budget is {budget:.4f}"
        )


class OutputGuard:
    """Validates, clamps and evidence-checks structured model output."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _as_dict(raw_obj: Any) -> dict[str, Any]:
        if isinstance(raw_obj, BaseModel):
            return dict(raw_obj.model_dump())
        if isinstance(raw_obj, Mapping):
            return dict(raw_obj)
        raise SchemaRejectedError(f"model output is not an object: {type(raw_obj).__name__}")

    @staticmethod
    def _serialise(data: Any) -> str:
        try:
            return json.dumps(data, default=str, ensure_ascii=False)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return str(data)

    @staticmethod
    def _signal(pattern_id: str, category: InjectionCategory, snippet: str, tier: TrustTier) -> InjectionSignal:
        return InjectionSignal(
            pattern_id=pattern_id,
            category=category,
            snippet=snippet[:_MAX_SNIPPET],
            tier=tier,
        )

    # -- the guard ---------------------------------------------------------

    def validate(
        self,
        raw_obj: Any,
        schema: type[BaseModel],
        sanitized_inputs: Sequence[str],
        canary: str = "",
        *,
        in_scope_ids: Collection[str] | None = None,
        tier: TrustTier = TrustTier.REFERENCE_PAGE,
    ) -> tuple[BaseModel, list[InjectionSignal]]:
        """Turn raw model output into a validated schema instance plus its signals.

        Raises :class:`CanaryLeakError` when the canary leaked and
        :class:`SchemaRejectedError` when the output cannot be repaired into the schema or
        refers to a finding outside ``in_scope_ids``. ``SchemaRejectedError`` is the
        retryable failure: the caller may re-prompt, then fall back to the heuristic.
        """
        signals: list[InjectionSignal] = []
        data = self._as_dict(raw_obj)
        serialised = self._serialise(data)

        # 1. Canary: if the model echoed it, nothing else about this reply is trustworthy.
        if canary and canary_in_output(serialised, canary):
            raise CanaryLeakError(
                "canary token appeared in model output; the untrusted content reached the model"
            )

        fields = schema.model_fields

        # 2. Drop keys the schema does not declare (the frozen models forbid extras anyway,
        #    and an injected extra key is schema smuggling by definition).
        for key in [key for key in data if key not in fields]:
            signals.append(
                self._signal("schema_extra_field", InjectionCategory.SCHEMA_SMUGGLING, str(key), tier)
            )
            data.pop(key, None)

        # 3. Truncate over-long strings instead of failing the whole call.
        for name, max_length in _max_lengths(schema).items():
            value = data.get(name)
            if isinstance(value, str) and len(value) > max_length:
                signals.append(
                    self._signal("schema_string_truncated", InjectionCategory.SCHEMA_SMUGGLING, name, tier)
                )
                data[name] = value[:max_length]

        # 4. Clamp numerics to the schema's declared bounds.
        for name, (low, high, low_strict, high_strict) in numeric_bounds(schema).items():
            if name not in data:
                continue
            value = data[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            clamped = numeric
            if low is not None:
                breached = clamped <= low if low_strict else clamped < low
                if breached:
                    clamped = low + _STRICT_EPSILON if low_strict else low
            if high is not None:
                breached = clamped >= high if high_strict else clamped > high
                if breached:
                    clamped = high - _STRICT_EPSILON if high_strict else high
            if clamped != numeric:
                signals.append(
                    self._signal(
                        "numeric_out_of_bounds",
                        InjectionCategory.NUMERIC_OVERFLOW,
                        f"{name}={numeric!r} clamped to {clamped!r}",
                        tier,
                    )
                )
                data[name] = type(value)(clamped) if isinstance(value, int) and clamped.is_integer() else clamped

        # 5. Evidence spans must be quotable from the sanitized input.
        if self.config.require_evidence_spans and "evidence_spans" in fields:
            spans = data.get("evidence_spans") or ()
            if isinstance(spans, (str, bytes)):
                spans = [spans]
            kept: list[str] = []
            for span in spans:
                text = str(span)
                if span_matches(text, sanitized_inputs):
                    kept.append(text)
                else:
                    signals.append(
                        self._signal(
                            "evidence_span_unverified",
                            InjectionCategory.FAKE_EVIDENCE_INFLATE,
                            text,
                            tier,
                        )
                    )
            data["evidence_spans"] = tuple(kept)

        # 6. Cross-finding references are never legitimate.
        if in_scope_ids is not None:
            self._reject_out_of_scope(data, in_scope_ids)

        # 7. Finally, the schema itself decides.
        try:
            parsed = schema.model_validate(data)
        except ValidationError as exc:
            raise SchemaRejectedError(f"model output failed schema validation: {exc}") from exc
        return parsed, signals

    @staticmethod
    def _reject_out_of_scope(data: Mapping[str, Any], in_scope_ids: Collection[str]) -> None:
        scope = set(in_scope_ids)
        declared = data.get("finding_id")
        if isinstance(declared, str) and declared and declared not in scope:
            raise SchemaRejectedError(
                f"model output claims finding_id {declared!r}, which is not in scope for this call"
            )
        for key, value in data.items():
            for text in _iter_strings(value):
                for reference in _FINDING_REF.findall(text):
                    if reference not in scope:
                        raise SchemaRejectedError(
                            f"model output field {key!r} references out-of-scope finding {reference!r}"
                        )


def _iter_strings(value: Any) -> Iterable[str]:
    """Yield every string inside a nested output value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_strings(item)
