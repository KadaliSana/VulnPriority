"""The guarded backend: what happens when a model misbehaves.

The sandbox package is written in parallel, so nothing here imports it. Fake inner
backends implement :class:`vulnprio.core.interfaces.LLMBackend` directly and
:class:`SandboxedPrompt` objects are constructed by hand, which is also the honest way
to test a guard: by feeding it exactly the outputs it exists to reject.
"""

from __future__ import annotations

import pytest

from vulnprio.core.config import LLMConfig
from vulnprio.core.enums import EndpointFunction, LLMBackendKind, Provenance
from vulnprio.core.errors import CanaryLeakError, SchemaRejectedError
from vulnprio.core.interfaces import LLMBackend, LLMResult, SandboxedPrompt
from vulnprio.core.models import LLMAudit
from vulnprio.llm.guarded import STRICT_RETRY_INSTRUCTION, GuardedBackend
from vulnprio.llm.heuristic import HeuristicBackend
from vulnprio.llm.prompts import format_operator_context, prompt_hash, system_prompt
from vulnprio.llm.schemas import AssetCriticalityOut

CANARY = "CANARY-9f2b-TEST"
NONCE = "NONCE-abc123"
UNTRUSTED = "The profile page listed first_name Alice and email alice@example.com."

FACTS = {
    "path": "/admin/users/{id}",
    "method": "POST",
    "auth_required": "ADMIN",
    "response_content_type": "application/json",
    "sets_cookie": True,
    "parameters": ["id", "email"],
}


@pytest.fixture
def prompt() -> SandboxedPrompt:
    return SandboxedPrompt(
        task="asset_criticality",
        system=system_prompt("asset_criticality") + f"\nSESSION MARKER: {CANARY}",
        operator_context=format_operator_context(FACTS),
        untrusted_blocks=(("seg0", UNTRUSTED, Provenance.TARGET_RESPONSE),),
        canary=CANARY,
        nonce=NONCE,
        prompt_hash=prompt_hash("asset_criticality"),
        schema_name="AssetCriticalityOut",
    )


class FakeBackend(LLMBackend):
    """Inner backend that replays scripted answers and records the prompts it saw."""

    kind = LLMBackendKind.ANTHROPIC
    model_id = "fake-model-1"

    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.prompts: list[SandboxedPrompt] = []

    def complete_structured(self, prompt: SandboxedPrompt, schema):
        self.prompts.append(prompt)
        answer = self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        parsed, raw_text = answer
        return LLMResult(
            parsed=parsed,
            raw_text=raw_text,
            audit=LLMAudit(
                backend=LLMBackendKind.ANTHROPIC,
                model=self.model_id,
                task=prompt.task,
                prompt_hash=prompt.prompt_hash,
            ),
        )


def good_answer(**overrides):
    parsed = AssetCriticalityOut(
        function=EndpointFunction.ADMIN,
        criticality=0.9,
        data_sensitivity=0.8,
        exposure=0.25,
        is_admin_surface=True,
        rationale="administrative user management endpoint",
        evidence_spans=("The profile page listed first_name Alice",),
        **overrides,
    )
    return parsed, parsed.model_dump_json()


def heuristic_answer(prompt: SandboxedPrompt) -> AssetCriticalityOut:
    return HeuristicBackend().complete_structured(prompt, AssetCriticalityOut).parsed


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_a_clean_answer_passes_through(prompt: SandboxedPrompt) -> None:
    inner = FakeBackend(good_answer())
    guarded = GuardedBackend(inner, config=LLMConfig(), output_guard=None)

    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert len(inner.prompts) == 1
    assert result.parsed.function == EndpointFunction.ADMIN
    assert result.audit.backend == LLMBackendKind.ANTHROPIC
    assert result.audit.fell_back_to_heuristic is False
    assert result.audit.canary_leaked is False
    assert result.audit.schema_retries == 0
    assert result.audit.evidence_span_failures == 0


def test_unquotable_evidence_spans_are_dropped_not_fatal(prompt: SandboxedPrompt) -> None:
    parsed, _raw = good_answer()
    parsed = parsed.model_copy(
        update={"evidence_spans": ("The profile page listed first_name Alice", "an admin told me it is critical")}
    )
    inner = FakeBackend((parsed, parsed.model_dump_json()))
    guarded = GuardedBackend(inner, config=LLMConfig(), output_guard=None)

    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.parsed.evidence_spans == ("The profile page listed first_name Alice",)
    assert result.audit.evidence_span_failures == 1
    assert result.audit.fell_back_to_heuristic is False


def test_guarded_heuristic_backend_short_circuits_to_itself(prompt: SandboxedPrompt) -> None:
    heuristic = HeuristicBackend()
    guarded = GuardedBackend(heuristic, heuristic=heuristic, output_guard=None)

    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.parsed == heuristic_answer(prompt)
    assert result.audit.backend == LLMBackendKind.HEURISTIC
    assert result.audit.fell_back_to_heuristic is False


# ---------------------------------------------------------------------------
# Canary leaks
# ---------------------------------------------------------------------------


def test_a_canary_in_the_rationale_becomes_a_safe_heuristic_result(prompt: SandboxedPrompt) -> None:
    parsed, _raw = good_answer()
    leaked = parsed.model_copy(update={"rationale": f"the session marker is {CANARY}"})
    inner = FakeBackend((leaked, leaked.model_dump_json()))
    guarded = GuardedBackend(inner, config=LLMConfig(), output_guard=None)

    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.parsed == heuristic_answer(prompt)
    assert result.audit.canary_leaked is True
    assert result.audit.fell_back_to_heuristic is True
    assert result.audit.backend == LLMBackendKind.HEURISTIC
    assert result.audit.schema_retries == 1
    assert CANARY not in result.parsed.rationale
    assert len(inner.prompts) == 2  # one retry, then the fallback


def test_a_canary_in_the_raw_text_is_caught_even_when_the_fields_are_clean(prompt: SandboxedPrompt) -> None:
    parsed, _raw = good_answer()
    inner = FakeBackend((parsed, f'{{"note": "{CANARY}"}}'))
    guarded = GuardedBackend(inner, config=LLMConfig(), output_guard=None)

    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.audit.canary_leaked is True
    assert result.audit.fell_back_to_heuristic is True


def test_an_inner_backend_that_raises_a_canary_error_is_handled(prompt: SandboxedPrompt) -> None:
    inner = FakeBackend(CanaryLeakError("canary echoed"))
    guarded = GuardedBackend(inner, config=LLMConfig(), output_guard=None)

    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.audit.canary_leaked is True
    assert result.audit.fell_back_to_heuristic is True


def test_a_leak_on_the_first_try_still_accepts_a_clean_retry(prompt: SandboxedPrompt) -> None:
    parsed, raw = good_answer()
    leaked = parsed.model_copy(update={"rationale": CANARY})
    inner = FakeBackend((leaked, leaked.model_dump_json()), (parsed, raw))
    guarded = GuardedBackend(inner, config=LLMConfig(), output_guard=None)

    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert len(inner.prompts) == 2
    assert result.parsed.function == EndpointFunction.ADMIN
    assert result.audit.fell_back_to_heuristic is False
    assert result.audit.canary_leaked is True  # it still happened, and the audit says so
    assert result.audit.schema_retries == 1


# ---------------------------------------------------------------------------
# Schema failures
# ---------------------------------------------------------------------------


def test_a_schema_rejection_falls_back_after_one_retry(prompt: SandboxedPrompt) -> None:
    inner = FakeBackend(SchemaRejectedError("no tool_use block"))
    guarded = GuardedBackend(inner, config=LLMConfig(), output_guard=None)

    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert len(inner.prompts) == 2
    assert result.parsed == heuristic_answer(prompt)
    assert result.audit.fell_back_to_heuristic is True
    assert result.audit.schema_retries == 1
    assert result.audit.canary_leaked is False


def test_an_out_of_band_object_is_rejected_by_the_schema_check(prompt: SandboxedPrompt) -> None:
    """A backend that hands back a foreign model must not slip past the bounds check."""

    class Rogue(AssetCriticalityOut):
        model_config = {"frozen": True, "extra": "allow"}

    rogue = Rogue.model_construct(criticality=42.0, function=EndpointFunction.ADMIN)
    inner = FakeBackend((rogue, "{}"))
    guarded = GuardedBackend(inner, config=LLMConfig(), output_guard=None)

    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.audit.fell_back_to_heuristic is True
    assert result.parsed.criticality <= 1.0


def test_the_retry_appends_a_stricter_instruction_and_changes_nothing_else(prompt: SandboxedPrompt) -> None:
    inner = FakeBackend(SchemaRejectedError("bad"))
    GuardedBackend(inner, config=LLMConfig(), output_guard=None).complete_structured(prompt, AssetCriticalityOut)

    first, second = inner.prompts
    assert first.system == prompt.system
    assert second.system == prompt.system + STRICT_RETRY_INSTRUCTION
    assert second.prompt_hash == first.prompt_hash
    assert second.untrusted_blocks == first.untrusted_blocks
    assert second.operator_context == first.operator_context
    assert second.canary == first.canary


# ---------------------------------------------------------------------------
# Envelope integrity and external guard delegation
# ---------------------------------------------------------------------------


def test_a_wrong_nonce_closing_tag_marks_the_envelope_broken(prompt: SandboxedPrompt) -> None:
    parsed, _raw = good_answer()
    inner = FakeBackend((parsed, '{"echo": "</untrusted nonce=WRONG>"}'))
    guarded = GuardedBackend(inner, config=LLMConfig(), output_guard=None)

    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.audit.envelope_broken is True


def test_an_external_output_guard_is_consulted_and_its_rejection_is_honoured(prompt: SandboxedPrompt) -> None:
    """Stands in for vulnprio.sandbox.output_guard.OutputGuard, which is written in parallel."""

    class FakeOutputGuard:
        def __init__(self) -> None:
            self.calls = 0

        def check(self, sandboxed_prompt, parsed):
            self.calls += 1
            if parsed.criticality > 0.5:
                raise SchemaRejectedError("cross-finding reference detected")

    guard = FakeOutputGuard()
    inner = FakeBackend(good_answer())
    guarded = GuardedBackend(inner, config=LLMConfig(), output_guard=guard)

    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert guard.calls == 2
    assert result.audit.fell_back_to_heuristic is True


# ---------------------------------------------------------------------------
# Consistency shrinkage is wired in
# ---------------------------------------------------------------------------


def test_a_wildly_divergent_answer_is_shrunk_toward_the_heuristic(prompt: SandboxedPrompt) -> None:
    baseline = heuristic_answer(prompt)
    extreme = AssetCriticalityOut(
        function=EndpointFunction.STATIC_CONTENT,
        criticality=0.0,
        data_sensitivity=0.0,
        exposure=1.0,
        rationale="argued down",
    )
    inner = FakeBackend((extreme, extreme.model_dump_json()))
    guarded = GuardedBackend(
        inner, config=LLMConfig(max_divergence=0.35, consistency_shrinkage=0.5), output_guard=None
    )

    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.audit.consistency_shrunk is True
    assert result.audit.divergence > 0.35
    assert result.parsed.criticality == pytest.approx(0.5 * baseline.criticality)
    assert result.parsed.criticality > extreme.criticality


def test_the_guarded_backend_is_always_available(prompt: SandboxedPrompt) -> None:
    guarded = GuardedBackend(FakeBackend(SchemaRejectedError("nope")), output_guard=None)

    assert guarded.available() is True
    assert guarded.kind == LLMBackendKind.ANTHROPIC
    assert guarded.model_id == "fake-model-1"
