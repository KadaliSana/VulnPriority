"""The OpenAI-compatible backend: one wire format, many providers, no new dependency.

The transport is a stub with a ``post`` method, so every test asserts what the framework
sends and how it treats what comes back. No key, no socket, no ``openai`` package.

Two themes carry most of the weight. The first is that naming an endpoint is mandatory and
naming a model is mandatory, because a default for either would be the framework making a
decision that belongs to the operator. The second is that a weak provider must *degrade* --
a reply that ignores every JSON instruction becomes a heuristic answer with an audit that
says so, never a corrupted score.
"""

from __future__ import annotations

import json

import pytest

from vulnpriority.core.config import LLMConfig
from vulnpriority.core.enums import EndpointFunction, LLMBackendKind, Provenance
from vulnpriority.core.errors import ConfigError
from vulnpriority.core.interfaces import SandboxedPrompt
from vulnpriority.llm.guarded import GuardedBackend
from vulnpriority.llm.heuristic import HeuristicBackend
from vulnpriority.llm.openai_compatible import (
    DEFAULT_KEY_ENV,
    OpenAICompatibleBackend,
    backoff_delay,
    extract_json_object,
    is_local_endpoint,
    openai_json_schema,
    retry_after_seconds,
)
from vulnpriority.llm.prompts import format_operator_context, prompt_hash, system_prompt
from vulnpriority.llm.schemas import ApplicabilityOut, AssetCriticalityOut, ExploitabilityOut

CANARY = "CANARY-oai-4d20"
NONCE = "NONCE-oai-01"
UNTRUSTED = "The billing endpoint /api/checkout accepts a card_number parameter."

FACTS = {"path": "/api/checkout", "method": "POST", "parameters": ["card_number", "amount"]}

GROQ = "https://api.groq.com/openai/v1"
OLLAMA = "http://localhost:11434/v1"

GOOD_PAYLOAD = {
    "confidence": 0.75,
    "rationale": "payment surface handling card data",
    "evidence_spans": ["The billing endpoint /api/checkout accepts a card_number parameter."],
    "function": "payment",
    "criticality": 0.95,
    "data_sensitivity": 0.95,
    "exposure": 0.8,
    "is_auth_boundary": False,
    "is_admin_surface": False,
}


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeHTTPResponse:
    def __init__(self, status_code=200, body=None, headers=None, text="") -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body


def completion(content: str, prompt_tokens: int = 40, completion_tokens: int = 25) -> FakeHTTPResponse:
    """A normal 200 from a chat-completions endpoint."""
    return FakeHTTPResponse(
        body={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    )


def rate_limited(retry_after: str | None = None) -> FakeHTTPResponse:
    headers = {"retry-after": retry_after} if retry_after else {}
    return FakeHTTPResponse(status_code=429, headers=headers, text="rate limit exceeded")


class FakeTransport:
    """Records every POST and replays scripted responses."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.posts: list[dict] = []

    def post(self, url, json=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        reply = self.responses[min(len(self.posts) - 1, len(self.responses) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply


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


@pytest.fixture(autouse=True)
def _a_key_exists(monkeypatch):
    """Most tests need *a* key; the ones about its absence delete it themselves."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
    monkeypatch.setenv(DEFAULT_KEY_ENV, "sk-test-key")


def config(**overrides) -> LLMConfig:
    base = {
        "backend": LLMBackendKind.OPENAI_COMPATIBLE,
        "base_url": GROQ,
        "model": "openai/gpt-oss-20b",
        "api_key_env": "GROQ_API_KEY",
        "use_cache": False,
    }
    base.update(overrides)
    return LLMConfig(**base)


def backend(*responses, cfg: LLMConfig | None = None, **kwargs) -> OpenAICompatibleBackend:
    delays: list[float] = []
    created = OpenAICompatibleBackend(
        cfg or config(), client=FakeTransport(*responses), sleep=delays.append, **kwargs
    )
    created.slept = delays  # type: ignore[attr-defined]
    return created


# ---------------------------------------------------------------------------
# Construction: what must be named, and what may be absent
# ---------------------------------------------------------------------------


def test_a_missing_base_url_is_a_config_error():
    """No default endpoint: choosing whose servers see the scan is the operator's call."""
    with pytest.raises(ConfigError) as raised:
        OpenAICompatibleBackend(LLMConfig(backend=LLMBackendKind.OPENAI_COMPATIBLE))
    assert "base_url" in str(raised.value)


def test_the_base_url_error_names_concrete_free_options():
    """Discoverability is the point: the error should be enough to act on."""
    with pytest.raises(ConfigError) as raised:
        OpenAICompatibleBackend(LLMConfig(backend=LLMBackendKind.OPENAI_COMPATIBLE))
    message = str(raised.value)
    assert "groq.com" in message and "openrouter.ai" in message and "11434" in message


def test_an_unset_model_is_a_config_error():
    """The Anthropic default model is meaningless on every OpenAI-compatible provider."""
    with pytest.raises(ConfigError) as raised:
        OpenAICompatibleBackend(
            LLMConfig(backend=LLMBackendKind.OPENAI_COMPATIBLE, base_url=GROQ)
        )
    assert "llm.model" in str(raised.value)


def test_a_missing_key_for_a_remote_endpoint_is_a_config_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigError) as raised:
        OpenAICompatibleBackend(config())
    assert "GROQ_API_KEY" in str(raised.value)


def test_key_env_falls_back_to_openai_api_key_when_left_at_the_anthropic_default():
    created = backend(completion("{}"), cfg=config(api_key_env=LLMConfig.model_fields["api_key_env"].default))
    assert created.api_key_env == DEFAULT_KEY_ENV


@pytest.mark.parametrize("url", [OLLAMA, "http://127.0.0.1:1234/v1", "http://localhost:8000/v1"])
def test_a_local_server_needs_no_key(monkeypatch, url):
    """Ollama, LM Studio and vLLM serve an unauthenticated endpoint on the operator's box."""
    monkeypatch.delenv(DEFAULT_KEY_ENV, raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    created = OpenAICompatibleBackend(config(base_url=url, model="llama3.2"), client=FakeTransport())
    assert created.local is True
    assert "Authorization" not in created.headers()


def test_a_keyless_local_server_omits_the_header_entirely(monkeypatch, prompt):
    """Absent, not empty: a blank bearer is rejected by some local servers."""
    monkeypatch.delenv(DEFAULT_KEY_ENV, raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    local = OpenAICompatibleBackend(
        config(base_url=OLLAMA, model="llama3.2"),
        client=FakeTransport(completion(json.dumps(GOOD_PAYLOAD))),
    )
    local.complete_structured(prompt, AssetCriticalityOut)
    assert "Authorization" not in local.client.posts[0]["headers"]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (OLLAMA, True),
        ("http://localhost:1234/v1", True),
        ("http://127.0.0.1:8000/v1", True),
        (GROQ, False),
        ("https://openrouter.ai/api/v1", False),
    ],
)
def test_local_endpoint_detection(url, expected):
    assert is_local_endpoint(url) is expected


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_request_goes_to_chat_completions_under_the_configured_base_url(prompt):
    created = backend(completion(json.dumps(GOOD_PAYLOAD)))
    created.complete_structured(prompt, AssetCriticalityOut)
    assert created.client.posts[0]["url"] == f"{GROQ}/chat/completions"


def test_a_trailing_slash_on_the_base_url_does_not_double_up(prompt):
    created = backend(completion(json.dumps(GOOD_PAYLOAD)), cfg=config(base_url=GROQ + "/"))
    created.complete_structured(prompt, AssetCriticalityOut)
    assert created.client.posts[0]["url"] == f"{GROQ}/chat/completions"


def test_request_carries_the_bearer_token(prompt):
    created = backend(completion(json.dumps(GOOD_PAYLOAD)))
    created.complete_structured(prompt, AssetCriticalityOut)
    assert created.client.posts[0]["headers"]["Authorization"] == "Bearer gsk-test-key"


def test_request_carries_model_messages_and_bounds(prompt):
    created = backend(completion(json.dumps(GOOD_PAYLOAD)))
    created.complete_structured(prompt, AssetCriticalityOut)

    body = created.client.posts[0]["json"]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["temperature"] == 0.0
    assert body["stream"] is False
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == prompt.system
    assert UNTRUSTED in body["messages"][1]["content"]


def test_auto_json_mode_asks_for_json_and_states_the_schema(prompt):
    """The widest-supported combination: JSON mode for anyone who honours it, schema for all."""
    created = backend(completion(json.dumps(GOOD_PAYLOAD)))
    created.complete_structured(prompt, AssetCriticalityOut)

    body = created.client.posts[0]["json"]
    assert body["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in body["messages"][1]["content"]


def test_json_schema_mode_sends_the_schema_to_the_provider(prompt):
    created = backend(completion(json.dumps(GOOD_PAYLOAD)), cfg=config(json_mode="json_schema"))
    created.complete_structured(prompt, AssetCriticalityOut)

    response_format = created.client.posts[0]["json"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "record_asset_criticality"
    assert "criticality" in response_format["json_schema"]["schema"]["properties"]


def test_prompt_mode_sends_no_response_format_at_all(prompt):
    """The documented fallback for a provider that 400s on ``response_format``."""
    created = backend(completion(json.dumps(GOOD_PAYLOAD)), cfg=config(json_mode="prompt"))
    created.complete_structured(prompt, AssetCriticalityOut)

    body = created.client.posts[0]["json"]
    assert "response_format" not in body
    assert "JSON Schema" in body["messages"][1]["content"]


def test_json_object_mode_omits_the_schema_from_the_prompt(prompt):
    created = backend(completion(json.dumps(GOOD_PAYLOAD)), cfg=config(json_mode="json_object"))
    created.complete_structured(prompt, AssetCriticalityOut)

    body = created.client.posts[0]["json"]
    assert body["response_format"] == {"type": "json_object"}
    assert "JSON Schema" not in body["messages"][1]["content"]


# ---------------------------------------------------------------------------
# Schema rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schema", [AssetCriticalityOut, ExploitabilityOut, ApplicabilityOut])
def test_every_property_is_required(schema):
    document = openai_json_schema(schema)
    assert set(document["required"]) == set(document["properties"])


def test_extra_properties_are_forbidden():
    """``extra="forbid"`` reaching the provider is what blocks schema smuggling early."""
    assert openai_json_schema(AssetCriticalityOut)["additionalProperties"] is False


def test_int_enums_survive_as_integer_enums():
    """Unlike Gemini, the OpenAI shape expresses an integer enum directly."""
    document = openai_json_schema(ExploitabilityOut)
    maturity = document["$defs"]["ExploitMaturity"]
    assert maturity["enum"] == [0, 1, 2, 3, 4]


def test_unit_bounds_reach_the_provider():
    criticality = openai_json_schema(AssetCriticalityOut)["properties"]["criticality"]
    assert (criticality["minimum"], criticality["maximum"]) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------


def test_structured_reply_is_parsed_and_audited(prompt):
    created = backend(completion(json.dumps(GOOD_PAYLOAD)))
    result = created.complete_structured(prompt, AssetCriticalityOut)

    assert result.parsed.function is EndpointFunction.PAYMENT
    assert result.parsed.data_sensitivity == pytest.approx(0.95)
    assert result.audit.backend is LLMBackendKind.OPENAI_COMPATIBLE
    assert result.audit.model == "openai/gpt-oss-20b"
    assert (result.audit.input_tokens, result.audit.output_tokens) == (40, 25)
    assert result.audit.fell_back_to_heuristic is False


def test_a_fenced_reply_is_still_parsed(prompt):
    """Weak models wrap JSON in markdown. Tolerant about packaging, strict about content."""
    fenced = "```json\n" + json.dumps(GOOD_PAYLOAD) + "\n```"
    result = backend(completion(fenced)).complete_structured(prompt, AssetCriticalityOut)
    assert result.parsed.function is EndpointFunction.PAYMENT


def test_a_reply_with_a_preamble_is_still_parsed(prompt):
    chatty = "Sure! Here is the assessment:\n" + json.dumps(GOOD_PAYLOAD)
    result = backend(completion(chatty)).complete_structured(prompt, AssetCriticalityOut)
    assert result.parsed.function is EndpointFunction.PAYMENT


def test_a_list_of_content_parts_is_joined(prompt):
    """Some providers return content as a list of typed parts rather than a string."""
    body = {
        "choices": [{"message": {"content": [{"type": "text", "text": json.dumps(GOOD_PAYLOAD)}]}}],
        "usage": {},
    }
    result = backend(FakeHTTPResponse(body=body)).complete_structured(prompt, AssetCriticalityOut)
    assert result.parsed.function is EndpointFunction.PAYMENT


@pytest.mark.parametrize(
    "text",
    ["", "I cannot help with that.", "{", "[1, 2, 3", "no json here at all"],
)
def test_a_malformed_reply_is_rejected_not_passed_through(prompt, text):
    result = backend(completion(text), cfg=config(max_retries=0)).complete_structured(
        prompt, AssetCriticalityOut
    )
    assert result.audit.fell_back_to_heuristic is True
    assert isinstance(result.parsed, AssetCriticalityOut)


def test_an_out_of_range_number_is_rejected(prompt):
    poisoned = {**GOOD_PAYLOAD, "exposure": 7.0}
    result = backend(completion(json.dumps(poisoned)), cfg=config(max_retries=0)).complete_structured(
        prompt, AssetCriticalityOut
    )
    assert result.audit.fell_back_to_heuristic is True
    assert result.parsed.exposure <= 1.0


def test_a_smuggled_field_is_rejected(prompt):
    smuggled = {**GOOD_PAYLOAD, "override_rank": 1}
    result = backend(completion(json.dumps(smuggled)), cfg=config(max_retries=0)).complete_structured(
        prompt, AssetCriticalityOut
    )
    assert result.audit.fell_back_to_heuristic is True


def test_a_server_error_falls_back_rather_than_raising(prompt):
    created = backend(FakeHTTPResponse(status_code=500, text="upstream exploded"), cfg=config(max_retries=0))
    result = created.complete_structured(prompt, AssetCriticalityOut)
    assert result.audit.fell_back_to_heuristic is True


def test_a_bad_reply_is_retried_with_a_stricter_instruction(prompt):
    created = backend(completion("nonsense"), completion(json.dumps(GOOD_PAYLOAD)))
    result = created.complete_structured(prompt, AssetCriticalityOut)

    assert len(created.client.posts) == 2
    first = created.client.posts[0]["json"]["messages"][1]["content"]
    second = created.client.posts[1]["json"]["messages"][1]["content"]
    assert "previous reply was rejected" not in first
    assert "previous reply was rejected" in second
    assert result.audit.fell_back_to_heuristic is False


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_a_429_backs_off_then_succeeds(prompt):
    created = backend(rate_limited(), completion(json.dumps(GOOD_PAYLOAD)))
    result = created.complete_structured(prompt, AssetCriticalityOut)

    assert len(created.client.posts) == 2
    assert created.slept == [2.0]
    assert result.audit.fell_back_to_heuristic is False


def test_retry_after_wins_over_the_exponential_guess(prompt):
    created = backend(rate_limited("11"), cfg=config(max_retries=1))
    created.complete_structured(prompt, AssetCriticalityOut)
    assert created.slept == [11.0]


def test_a_spent_quota_degrades_the_finding_and_never_aborts_the_scan(prompt):
    created = backend(rate_limited(), cfg=config(max_retries=1))
    result = created.complete_structured(prompt, AssetCriticalityOut)

    assert isinstance(result.parsed, AssetCriticalityOut)
    assert result.audit.fell_back_to_heuristic is True
    assert result.audit.backend is LLMBackendKind.HEURISTIC


def test_the_fallback_answer_is_the_heuristic_answer(prompt):
    created = backend(rate_limited(), cfg=config(max_retries=0))
    result = created.complete_structured(prompt, AssetCriticalityOut)
    assert result.parsed == HeuristicBackend().complete_structured(prompt, AssetCriticalityOut).parsed


@pytest.mark.parametrize(
    ("attempt", "expected"), [(0, 2.0), (1, 4.0), (2, 8.0), (5, 60.0)]
)
def test_backoff_is_exponential_and_capped(attempt, expected):
    assert backoff_delay(attempt, base_s=2.0, max_s=60.0) == expected


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "30"}, 30.0),
        ({"Retry-After": "5"}, 5.0),
        ({"retry-after": "2.5s"}, 2.5),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_retry_after_parsing(headers, expected):
    assert retry_after_seconds(headers) == expected


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ("```\n{\"a\": 1}\n```", {"a": 1}),
        ('Here you go: {"a": 1} -- hope that helps', {"a": 1}),
    ],
)
def test_extract_json_object_tolerates_packaging(text, expected):
    assert extract_json_object(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "no object", "{unclosed"])
def test_extract_json_object_refuses_nonsense(text):
    with pytest.raises(ValueError):
        extract_json_object(text)


# ---------------------------------------------------------------------------
# Under the guard
# ---------------------------------------------------------------------------


def test_guarded_backend_rejects_a_canary_leak(prompt):
    leaked = {**GOOD_PAYLOAD, "rationale": f"payment page ({CANARY})"}
    guarded = GuardedBackend(
        inner=backend(completion(json.dumps(leaked))),
        heuristic=HeuristicBackend(),
        config=config(),
        output_guard=None,
    )
    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.audit.canary_leaked is True
    assert result.audit.fell_back_to_heuristic is True
    assert CANARY not in result.parsed.rationale


def test_guarded_backend_drops_an_unquotable_evidence_span(prompt):
    invented = {**GOOD_PAYLOAD, "evidence_spans": ["text the model was never shown"]}
    guarded = GuardedBackend(
        inner=backend(completion(json.dumps(invented))),
        heuristic=HeuristicBackend(),
        config=config(),
        output_guard=None,
    )
    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.parsed.evidence_spans == ()
    assert result.audit.evidence_span_failures == 1


def test_guarded_backend_keeps_a_good_answer_intact(prompt):
    guarded = GuardedBackend(
        inner=backend(completion(json.dumps(GOOD_PAYLOAD))),
        heuristic=HeuristicBackend(),
        config=config(),
        output_guard=None,
    )
    result = guarded.complete_structured(prompt, AssetCriticalityOut)

    assert result.parsed.function is EndpointFunction.PAYMENT
    assert result.audit.fell_back_to_heuristic is False
    assert result.parsed.evidence_spans == tuple(GOOD_PAYLOAD["evidence_spans"])


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_a_cached_answer_costs_no_request(prompt, tmp_path):
    cached = config(use_cache=True, cache_dir=tmp_path)
    backend(completion(json.dumps(GOOD_PAYLOAD)), cfg=cached).complete_structured(
        prompt, AssetCriticalityOut
    )

    second = backend(completion(json.dumps(GOOD_PAYLOAD)), cfg=cached)
    result = second.complete_structured(prompt, AssetCriticalityOut)

    assert second.client.posts == []
    assert result.audit.cached is True
