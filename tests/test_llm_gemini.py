"""The Gemini backend: request shape, schema conversion, and what a free tier does to you.

Nothing here needs a key or a socket. The SDK client is a stub that records what it was
asked for and replays a scripted reply, which is the only honest way to assert a request
shape: the thing under test is what the framework *sends*, and a live call would tell us
about Google's servers rather than about this code.

The rate-limit tests carry the most weight. A free quota is not an edge case -- it is the
normal operating condition of the configuration this backend exists to enable -- so "429
degrades one finding and never aborts the scan" is asserted directly rather than assumed.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from vulnprio.core.config import LLMConfig
from vulnprio.core.enums import (
    EndpointFunction,
    ExploitMaturity,
    LLMBackendKind,
    PrivilegeLevel,
    Provenance,
)
from vulnprio.core.errors import ConfigError
from vulnprio.core.interfaces import SandboxedPrompt
from vulnprio.llm.gemini_backend import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_KEY_ENV,
    GeminiBackend,
    gemini_schema_for,
    is_rate_limited,
    resolve_key_env,
    resolve_model,
)
from vulnprio.llm.guarded import GuardedBackend
from vulnprio.llm.heuristic import HeuristicBackend
from vulnprio.llm.openai_compatible import RateLimiter
from vulnprio.llm.prompts import format_operator_context, prompt_hash, system_prompt
from vulnprio.llm.schemas import (
    ApplicabilityOut,
    AssetCriticalityOut,
    ExploitabilityOut,
)

CANARY = "CANARY-gemini-7f31"
NONCE = "NONCE-gem-01"
UNTRUSTED = "The admin console at /admin/users lists every account and its email address."

FACTS = {
    "path": "/admin/users",
    "method": "POST",
    "auth_required": "ADMIN",
    "response_content_type": "application/json",
    "sets_cookie": True,
    "parameters": ["id", "email"],
}

GOOD_PAYLOAD = {
    "confidence": 0.8,
    "rationale": "administrative user management surface",
    "evidence_spans": ["The admin console at /admin/users lists every account"],
    "function": "admin",
    "criticality": 0.9,
    "data_sensitivity": 0.8,
    "exposure": 0.3,
    "is_auth_boundary": False,
    "is_admin_surface": True,
}


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeUsage:
    def __init__(self, prompt_tokens: int = 31, output_tokens: int = 17) -> None:
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = output_tokens


class FakeResponse:
    """What ``generate_content`` returns: ``.text`` plus usage, as the SDK provides."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = FakeUsage()


class FakeQuotaError(Exception):
    """A Gemini quota rejection, in the shape the SDK raises it."""

    def __init__(self, message: str = "RESOURCE_EXHAUSTED: quota exceeded", code: int = 429) -> None:
        super().__init__(message)
        self.code = code


class FakeModels:
    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeClient:
    def __init__(self, *replies) -> None:
        self.models = FakeModels(*replies)


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


def config(**overrides) -> LLMConfig:
    """A Gemini config that never touches the on-disk cache."""
    base = {"backend": LLMBackendKind.GEMINI, "use_cache": False}
    base.update(overrides)
    return LLMConfig(**base)


def backend(*replies, cfg: LLMConfig | None = None, **kwargs) -> GeminiBackend:
    delays: list[float] = []
    created = GeminiBackend(
        cfg or config(),
        client=FakeClient(*replies),
        sleep=delays.append,
        **kwargs,
    )
    created.slept = delays  # type: ignore[attr-defined]
    return created


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------


def test_schema_conversion_bounds_unit_floats():
    converted = gemini_schema_for(AssetCriticalityOut)
    criticality = converted["properties"]["criticality"]
    assert criticality["type"] == "NUMBER"
    assert criticality["minimum"] == 0.0
    assert criticality["maximum"] == 1.0


def test_schema_conversion_renders_string_enums_as_closed_sets():
    function = gemini_schema_for(AssetCriticalityOut)["properties"]["function"]
    assert function["type"] == "STRING"
    assert set(function["enum"]) == {member.value for member in EndpointFunction}


def test_schema_conversion_renders_int_enums_as_bounded_integers():
    """The case the SDK's own converter cannot express.

    ``types.Schema.enum`` is ``list[str]``, so an ``IntEnum`` has to become a bounded
    ``INTEGER``. That is exact here because these enumerations are contiguous from zero:
    every integer in the range is a member and nothing outside the set is admitted.
    """
    properties = gemini_schema_for(ExploitabilityOut)["properties"]

    maturity = properties["exploit_maturity"]
    assert maturity["type"] == "INTEGER"
    assert (maturity["minimum"], maturity["maximum"]) == (
        float(min(ExploitMaturity)),
        float(max(ExploitMaturity)),
    )
    assert "0, 1, 2, 3, 4" in maturity["description"]

    privileges = properties["privileges_required"]
    assert privileges["type"] == "INTEGER"
    assert privileges["maximum"] == float(max(PrivilegeLevel))


def test_schema_conversion_requires_every_property():
    """Nothing is optional: an omitted field returns a default no caller can detect."""
    for schema in (AssetCriticalityOut, ExploitabilityOut, ApplicabilityOut):
        converted = gemini_schema_for(schema)
        assert converted["required"] == list(converted["properties"])


def _valid_asset_payload() -> dict:
    """A minimal payload the asset schema accepts, for testing what it refuses."""
    return {
        "confidence": 0.8,
        "rationale": "An authenticated administrative endpoint.",
        "evidence_spans": [],
        "function": "admin",
        "criticality": 0.7,
        "data_sensitivity": 0.6,
        "exposure": 0.5,
        "is_auth_boundary": False,
        "is_admin_surface": True,
    }


def test_schema_conversion_emits_no_additional_properties_field():
    """The API rejects the field outright, so it must not be sent -- verified live.

    ``types.Schema`` carries ``additional_properties``, which is what this test originally
    asserted. The REST surface does not: a request carrying it comes back
    ``400 INVALID_ARGUMENT ... Unknown name "additional_properties" at
    'generation_config.response_schema'``, so emitting it made every structured call fail.

    Nothing is lost. The object is closed in practice because every property is ``required``
    and none is optional, and extras are forbidden for real on the way back, where the reply
    is validated against the pydantic model whose config is ``extra="forbid"`` -- which is
    the check that actually stops schema smuggling. The test below pins that.
    """
    converted = gemini_schema_for(AssetCriticalityOut)
    assert "additional_properties" not in converted
    assert converted["required"] == list(converted["properties"])


def test_extras_are_still_refused_where_it_counts():
    """The guarantee lives in the return-path model, not in what we asked the provider for."""
    with pytest.raises(ValidationError):
        AssetCriticalityOut.model_validate(
            {**_valid_asset_payload(), "smuggled_priority": 1.0}
        )


def test_schema_conversion_caps_a_free_form_map_without_pinning_its_value_type():
    """``preconditions_met`` keeps its size cap; its value type cannot be sent.

    The value type would ride in ``additional_properties``, which the API refuses (see
    above), so the map goes over open and the ``dict[str, bool]`` annotation is enforced
    on the way back by ``model_validate``. The cap survives because ``max_properties`` is
    a field the API does accept.
    """
    preconditions = gemini_schema_for(ApplicabilityOut)["properties"]["preconditions_met"]
    assert preconditions["type"] == "OBJECT"
    assert "additional_properties" not in preconditions
    assert preconditions["max_properties"] == 8


def test_schema_conversion_caps_string_and_array_lengths():
    converted = gemini_schema_for(AssetCriticalityOut)
    assert converted["properties"]["rationale"]["max_length"] == 600
    spans = converted["properties"]["evidence_spans"]
    assert spans["max_items"] == 5
    assert spans["items"]["max_length"] == 200


@pytest.mark.parametrize("schema", [AssetCriticalityOut, ExploitabilityOut, ApplicabilityOut])
def test_schema_conversion_is_accepted_by_the_sdk(schema):
    """The conversion must survive the SDK's own validation, not merely look plausible."""
    types = pytest.importorskip("google.genai.types")
    validated = types.Schema.model_validate(gemini_schema_for(schema))
    assert validated.type is not None
    types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=gemini_schema_for(schema)
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_missing_key_is_a_config_error_at_construction(monkeypatch):
    """A key that is not there is a configuration fault, not a mid-scan surprise."""
    monkeypatch.delenv(DEFAULT_KEY_ENV, raising=False)
    with pytest.raises(ConfigError) as raised:
        GeminiBackend(config())
    assert DEFAULT_KEY_ENV in str(raised.value)


def test_injected_client_needs_no_key(monkeypatch):
    monkeypatch.delenv(DEFAULT_KEY_ENV, raising=False)
    assert backend(FakeResponse("{}")).available() is True


def test_key_env_defaults_to_gemini_and_stays_overridable():
    assert resolve_key_env(LLMConfig.model_fields["api_key_env"].default) == DEFAULT_KEY_ENV
    assert resolve_key_env("MY_OWN_KEY") == "MY_OWN_KEY"


def test_model_defaults_to_free_tier_flash_and_stays_overridable():
    assert resolve_model(LLMConfig.model_fields["model"].default) == DEFAULT_GEMINI_MODEL
    assert resolve_model("gemini-2.5-pro") == "gemini-2.5-pro"


def test_explicit_model_wins_over_the_free_tier_default():
    assert backend(FakeResponse("{}"), cfg=config(model="gemini-2.5-pro")).model_id == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_request_carries_model_mime_type_and_schema(prompt):
    gemini = backend(FakeResponse(json.dumps(GOOD_PAYLOAD)))
    gemini.complete_structured(prompt, AssetCriticalityOut)

    sent = gemini.client.models.calls[0]
    assert sent["model"] == DEFAULT_GEMINI_MODEL
    config_sent = sent["config"]
    assert config_sent.response_mime_type == "application/json"
    assert config_sent.response_schema is not None
    assert config_sent.temperature == 0.0


def test_request_declares_no_tools(prompt):
    """This backend judges evidence it was handed. It must not fetch any of its own."""
    gemini = backend(FakeResponse(json.dumps(GOOD_PAYLOAD)))
    gemini.complete_structured(prompt, AssetCriticalityOut)
    assert getattr(gemini.client.models.calls[0]["config"], "tools", None) in (None, [])


def test_request_puts_the_system_prompt_in_system_instruction(prompt):
    gemini = backend(FakeResponse(json.dumps(GOOD_PAYLOAD)))
    gemini.complete_structured(prompt, AssetCriticalityOut)
    assert gemini.client.models.calls[0]["config"].system_instruction == prompt.system


def test_request_body_carries_operator_context_and_untrusted_text(prompt):
    gemini = backend(FakeResponse(json.dumps(GOOD_PAYLOAD)))
    gemini.complete_structured(prompt, AssetCriticalityOut)
    contents = gemini.client.models.calls[0]["contents"]
    assert UNTRUSTED in contents
    assert "/admin/users" in contents


def test_retry_appends_a_stricter_instruction_without_changing_the_first_turn(prompt):
    gemini = backend(FakeResponse("not json at all"), FakeResponse(json.dumps(GOOD_PAYLOAD)))
    gemini.complete_structured(prompt, AssetCriticalityOut)

    first, second = gemini.client.models.calls[0]["contents"], gemini.client.models.calls[1]["contents"]
    assert "previous reply was rejected" not in first
    assert "previous reply was rejected" in second


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------


def test_structured_reply_is_parsed_and_audited(prompt):
    gemini = backend(FakeResponse(json.dumps(GOOD_PAYLOAD)))
    result = gemini.complete_structured(prompt, AssetCriticalityOut)

    assert result.parsed.function is EndpointFunction.ADMIN
    assert result.parsed.criticality == pytest.approx(0.9)
    assert result.parsed.is_admin_surface is True
    assert result.audit.backend is LLMBackendKind.GEMINI
    assert result.audit.model == DEFAULT_GEMINI_MODEL
    assert result.audit.fell_back_to_heuristic is False
    assert (result.audit.input_tokens, result.audit.output_tokens) == (31, 17)


def test_reply_is_read_from_candidates_when_there_is_no_text_accessor(prompt):
    """A response that only carries candidates still parses."""

    class PartsOnly:
        text = None
        usage_metadata = FakeUsage()

        class _Part:
            def __init__(self, text): self.text = text

        class _Content:
            def __init__(self, parts): self.parts = parts

        class _Candidate:
            def __init__(self, content): self.content = content

        def __init__(self, body):
            self.candidates = [self._Candidate(self._Content([self._Part(body)]))]

    gemini = backend(PartsOnly(json.dumps(GOOD_PAYLOAD)))
    assert gemini.complete_structured(prompt, AssetCriticalityOut).parsed.is_admin_surface is True


def test_out_of_range_value_is_rejected_not_clamped_through(prompt):
    """A number outside [0, 1] fails validation; it never reaches the caller as-is."""
    poisoned = {**GOOD_PAYLOAD, "criticality": 9.5}
    gemini = backend(FakeResponse(json.dumps(poisoned)))
    result = gemini.complete_structured(prompt, AssetCriticalityOut)

    assert result.audit.fell_back_to_heuristic is True
    assert result.parsed.criticality <= 1.0


def test_extra_field_is_rejected_rather_than_passed_through(prompt):
    """Schema smuggling fails validation instead of quietly succeeding."""
    smuggled = {**GOOD_PAYLOAD, "p_exploit_override": 1.0}
    gemini = backend(FakeResponse(json.dumps(smuggled)))
    result = gemini.complete_structured(prompt, AssetCriticalityOut)

    assert result.audit.fell_back_to_heuristic is True
    assert not hasattr(result.parsed, "p_exploit_override")


def test_malformed_json_falls_back_rather_than_raising(prompt):
    gemini = backend(FakeResponse("I'm afraid I can't do that."))
    result = gemini.complete_structured(prompt, AssetCriticalityOut)
    assert result.audit.fell_back_to_heuristic is True


def test_a_bad_reply_is_retried_before_falling_back(prompt):
    gemini = backend(FakeResponse("{"), FakeResponse(json.dumps(GOOD_PAYLOAD)))
    result = gemini.complete_structured(prompt, AssetCriticalityOut)

    assert len(gemini.client.models.calls) == 2
    assert result.audit.fell_back_to_heuristic is False
    assert result.parsed.is_admin_surface is True


# ---------------------------------------------------------------------------
# Rate limiting: the free tier's normal operating condition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        FakeQuotaError(),
        Exception("429 Too Many Requests"),
        Exception("RESOURCE_EXHAUSTED"),
    ],
)
def test_rate_limit_is_recognised_in_every_shape_it_arrives_in(exc):
    assert is_rate_limited(exc) is True


def test_an_ordinary_failure_is_not_mistaken_for_a_rate_limit():
    assert is_rate_limited(ValueError("invalid argument")) is False


def test_rate_limit_backs_off_then_succeeds(prompt):
    gemini = backend(FakeQuotaError(), FakeResponse(json.dumps(GOOD_PAYLOAD)))
    result = gemini.complete_structured(prompt, AssetCriticalityOut)

    assert len(gemini.client.models.calls) == 2
    assert gemini.slept and gemini.slept[0] > 0.0  # it actually waited
    assert result.audit.fell_back_to_heuristic is False
    assert result.parsed.is_admin_surface is True


def test_backoff_grows_between_attempts(prompt):
    gemini = backend(FakeQuotaError(), cfg=config(max_retries=3, retry_backoff_s=2.0))
    gemini.complete_structured(prompt, AssetCriticalityOut)
    assert gemini.slept == [2.0, 4.0, 8.0]


def test_backoff_is_capped(prompt):
    gemini = backend(FakeQuotaError(), cfg=config(max_retries=4, retry_backoff_s=10.0, max_backoff_s=20.0))
    gemini.complete_structured(prompt, AssetCriticalityOut)
    assert max(gemini.slept) <= 20.0


def test_a_spent_quota_degrades_the_finding_and_never_aborts_the_scan(prompt):
    """The whole point. A 429 must produce an answer, not an exception."""
    gemini = backend(FakeQuotaError(), cfg=config(max_retries=1))
    result = gemini.complete_structured(prompt, AssetCriticalityOut)

    assert isinstance(result.parsed, AssetCriticalityOut)
    assert result.audit.fell_back_to_heuristic is True


def test_the_fallback_audit_is_truthful_about_who_answered(prompt):
    """A heuristic answer must not be reported as a model answer."""
    gemini = backend(FakeQuotaError(), cfg=config(max_retries=0))
    result = gemini.complete_structured(prompt, AssetCriticalityOut)
    heuristic = HeuristicBackend().complete_structured(prompt, AssetCriticalityOut)

    assert result.audit.fell_back_to_heuristic is True
    assert result.audit.backend is LLMBackendKind.HEURISTIC
    assert result.parsed == heuristic.parsed


def test_fallback_can_be_switched_off_for_a_run_that_must_not_degrade_silently(prompt):
    gemini = backend(FakeQuotaError(), cfg=config(fallback_to_heuristic=False, max_retries=0))
    with pytest.raises(ConfigError):
        gemini.complete_structured(prompt, AssetCriticalityOut)


def test_provider_supplied_retry_delay_wins_over_the_exponential_guess(prompt):
    """The provider knows when its window reopens; a guess that undershoots earns a 429."""
    quota = FakeQuotaError("429 quota exceeded, 'retryDelay': '37s'")
    gemini = backend(quota, cfg=config(max_retries=1, retry_backoff_s=2.0, max_backoff_s=60.0))
    gemini.complete_structured(prompt, AssetCriticalityOut)
    assert gemini.slept == [37.0]


# ---------------------------------------------------------------------------
# Client-side pacing
# ---------------------------------------------------------------------------


def test_pacing_is_off_by_default():
    assert RateLimiter(0).min_interval_s == 0.0


def test_pacing_spaces_requests_to_the_configured_rate():
    """15 requests a minute is one every four seconds; 0.1s in, 3.9s are still owed."""
    #                 first wait  second wait  post-sleep restamp
    clock = iter([0.0, 0.1, 4.0])
    waited: list[float] = []
    limiter = RateLimiter(15, monotonic=lambda: next(clock), sleep=waited.append)

    assert limiter.min_interval_s == pytest.approx(4.0)
    limiter.wait()  # the first request never waits
    limiter.wait()
    assert waited == [pytest.approx(3.9)]


def test_pacing_is_applied_to_every_call(prompt):
    waits: list[float] = []
    gemini = backend(
        FakeResponse(json.dumps(GOOD_PAYLOAD)),
        cfg=config(requests_per_minute=30),
        limiter=RateLimiter(30, monotonic=iter([0.0, 0.0, 0.0]).__next__, sleep=waits.append),
    )
    gemini.complete_structured(prompt, AssetCriticalityOut)
    assert gemini.limiter.rpm == 30


# ---------------------------------------------------------------------------
# Under the guard
# ---------------------------------------------------------------------------


def test_guarded_gemini_rejects_a_canary_leak(prompt):
    """A new backend does not get a new path into the score."""
    leaked = {**GOOD_PAYLOAD, "rationale": f"admin surface, marker {CANARY}"}
    guarded = GuardedBackend(
        inner=backend(FakeResponse(json.dumps(leaked))),
        heuristic=HeuristicBackend(),
        config=config(),
        output_guard=None,
    )
    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.audit.canary_leaked is True
    assert result.audit.fell_back_to_heuristic is True
    assert CANARY not in result.parsed.rationale


def test_guarded_gemini_drops_an_unquotable_evidence_span(prompt):
    """A claim the model cannot quote loses its evidence, not the whole answer."""
    invented = {**GOOD_PAYLOAD, "evidence_spans": ["a sentence that was never shown to it"]}
    guarded = GuardedBackend(
        inner=backend(FakeResponse(json.dumps(invented))),
        heuristic=HeuristicBackend(),
        config=config(),
        output_guard=None,
    )
    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.parsed.evidence_spans == ()
    assert result.audit.evidence_span_failures == 1


def test_guarded_gemini_still_revalidates_the_schema(prompt):
    """The bounds are the security contract and are checked again after the backend."""

    class LiesAboutBounds(GeminiBackend):
        def complete_structured(self, prompt, schema):
            result = super().complete_structured(prompt, schema)
            forged = result.parsed.model_copy(update={"criticality": 4.2})
            return result.model_copy(update={"parsed": forged})

    inner = LiesAboutBounds(
        config(), client=FakeClient(FakeResponse(json.dumps(GOOD_PAYLOAD))), sleep=lambda _s: None
    )
    guarded = GuardedBackend(
        inner=inner, heuristic=HeuristicBackend(), config=config(), output_guard=None
    )
    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.parsed.criticality <= 1.0
    assert result.audit.fell_back_to_heuristic is True


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_a_cached_answer_costs_no_request(prompt, tmp_path):
    cached = config(use_cache=True, cache_dir=tmp_path)
    first = backend(FakeResponse(json.dumps(GOOD_PAYLOAD)), cfg=cached)
    first.complete_structured(prompt, AssetCriticalityOut)

    second = backend(FakeResponse(json.dumps(GOOD_PAYLOAD)), cfg=cached)
    result = second.complete_structured(prompt, AssetCriticalityOut)

    assert second.client.models.calls == []
    assert result.audit.cached is True
    assert result.parsed.is_admin_surface is True
