"""Gemini backend: structured output on a free tier, with the rate limit as a design input.

``gemini-3.5-flash-lite`` has a no-cost quota, which makes this the backend that lets someone
run the whole framework -- model-backed assessment and all -- without a bill. That single
fact drives everything else in this module.

**Structured output.** ``GenerateContentConfig(response_mime_type="application/json",
response_schema=...)``, so the bounds are enforced by the provider and an out-of-range
answer usually never gets generated. Whatever does come back is still ``model_validate``\\ d
here and still goes through the sandbox output guard afterwards.

**The schema is converted here rather than handed over.** The SDK accepts a pydantic class
directly, and for two of the three task schemas that works. It does *not* work for
:class:`~vulnpriority.llm.schemas.ExploitabilityOut`: ``types.Schema.enum`` is typed
``list[str]``, and ``ExploitMaturity``/``PrivilegeLevel`` are ``IntEnum``, so the SDK's own
converter raises ``ValidationError`` on ``enum`` values of ``0..4``. Passing the class
through would therefore fail on exactly one of the three assessments the framework makes.
:func:`gemini_schema_for` does the conversion instead and renders a contiguous ``IntEnum``
as a bounded ``INTEGER`` with its members spelled out in the description -- which is
exact, because every integer in the range *is* a member.

**Rate limits are an outcome, not an exception.** A free quota will be hit. The backend
paces itself client-side (:class:`~vulnpriority.llm.openai_compatible.RateLimiter`), treats a
429 as a first-class result, backs off within the configured retry budget, and if the quota
is still shut falls back to the heuristic with ``fell_back_to_heuristic`` set -- exactly as
the Anthropic backend does for its own failures. A rate limit degrades one finding. It must
never abort a scan.

The SDK is imported lazily inside the constructor so the package stays importable, and the
offline suite stays runnable, on a machine with no ``google-genai`` and no key.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from pydantic import ValidationError

from vulnpriority.core.config import LLMConfig
from vulnpriority.core.enums import LLMBackendKind
from vulnpriority.core.errors import ConfigError
from vulnpriority.core.interfaces import LLMBackend, LLMResult, SandboxedPrompt
from vulnpriority.core.models import InjectionSignal, LLMAudit
from vulnpriority.core.registry import register_backend
from vulnpriority.llm.cache import LLMResponseCache, mark_cached
from vulnpriority.llm.heuristic import HeuristicBackend
from vulnpriority.llm.openai_compatible import RateLimited, RateLimiter, backoff_delay
from vulnpriority.llm.prompts import render_task
from vulnpriority.llm.schemas import BoundedOut, task_for_schema

__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_KEY_ENV",
    "GeminiBackend",
    "gemini_schema_for",
    "is_rate_limited",
    "resolve_key_env",
    "resolve_model",
    "retry_delay_of",
]

#: Free-tier default. Flash is the model the free quota is generous with, and it is also
#: the one that carries free Google Search grounding for :mod:`vulnpriority.intel`.
#: Pinned rather than an alias. ``gemini-flash-latest`` never goes stale, but it floats:
#: asked for it, the API served ``gemini-3.8-flash``, and a framework whose whole argument
#: is reproducibility cannot have the model change underneath a recorded run. The cost of
#: pinning is that a pin eventually dies -- ``gemini-2.5-flash`` now 404s with "no longer
#: available to new users" -- so :data:`FALLBACK_GEMINI_MODEL` catches that case loudly.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

#: Used only when the pinned model is retired, so a run degrades to the current flash model
#: with a recorded note instead of failing outright. Whatever actually served the request is
#: reported by the response's ``model_version`` and belongs in the run manifest.
FALLBACK_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_KEY_ENV = "GEMINI_API_KEY"


def resolve_model(configured: str | None, default: str = DEFAULT_GEMINI_MODEL) -> str:
    """``configured``, unless it is still the Anthropic-shaped default from ``LLMConfig``.

    ``LLMConfig.model`` defaults to a Claude id because Anthropic was the first backend.
    Reading an untouched default as "not chosen for me" means switching a run to Gemini is
    one line of YAML instead of three, and an explicitly set model always wins. Compared
    against the field default rather than a literal so this cannot drift.
    """
    anthropic_default = LLMConfig.model_fields["model"].default
    return default if (not configured or configured == anthropic_default) else configured


def resolve_key_env(configured: str | None, default: str = DEFAULT_KEY_ENV) -> str:
    """Same rule for ``api_key_env``: an untouched ``ANTHROPIC_API_KEY`` means this one."""
    anthropic_default = LLMConfig.model_fields["api_key_env"].default
    return default if (not configured or configured == anthropic_default) else configured


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------

#: JSON Schema type name -> the SDK's ``types.Type`` name.
_TYPE_NAMES: dict[str, str] = {
    "object": "OBJECT",
    "array": "ARRAY",
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "null": "NULL",
}


def _resolve_ref(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Inline a ``$ref``, and collapse pydantic's ``allOf: [{$ref}]`` wrapper.

    Pydantic emits a field whose type is an enum *and* which has a default as
    ``{"allOf": [{"$ref": "#/$defs/Foo"}], "default": ...}``. Gemini has no ``$defs``
    indirection worth using here and the schemas are small, so everything is inlined.
    """
    merged = dict(node)
    for _ in range(8):  # bounded: these schemas nest shallowly and cycles must not hang
        wrapper = merged.pop("allOf", None)
        if isinstance(wrapper, list) and len(wrapper) == 1 and isinstance(wrapper[0], dict):
            merged = {**wrapper[0], **merged}
            continue
        ref = merged.pop("$ref", None)
        if not ref:
            break
        name = str(ref).rsplit("/", 1)[-1]
        target = defs.get(name)
        if not isinstance(target, dict):
            break
        merged = {**target, **merged}
    return merged


def _convert(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """One JSON Schema node -> one Gemini ``Schema`` dict, in the SDK's field naming."""
    node = _resolve_ref(node, defs)
    json_type = node.get("type")
    enum_values = node.get("enum")
    out: dict[str, Any] = {}

    if enum_values:
        strings = [v for v in enum_values if isinstance(v, str)]
        if len(strings) == len(enum_values):
            # A closed string set is exactly what Gemini's ``enum`` expresses.
            out["type"] = "STRING"
            out["enum"] = strings
        else:
            # An IntEnum. ``types.Schema.enum`` is ``list[str]``, so a bounded INTEGER is
            # the faithful rendering: these enumerations are contiguous from zero, so
            # every integer in range is a member and nothing out of the set is admitted.
            numbers = [int(v) for v in enum_values if isinstance(v, (int, float))]
            out["type"] = "INTEGER"
            if numbers:
                out["minimum"] = float(min(numbers))
                out["maximum"] = float(max(numbers))
                listed = ", ".join(str(n) for n in sorted(numbers))
                description = str(node.get("description") or "").strip()
                out["description"] = (
                    f"{description} Allowed values: {listed}." if description
                    else f"Allowed values: {listed}."
                )
        return out

    out["type"] = _TYPE_NAMES.get(str(json_type), "STRING")

    if description := node.get("description"):
        out["description"] = str(description)
    for source, target in (("minimum", "minimum"), ("maximum", "maximum")):
        if (value := node.get(source)) is not None:
            out[target] = float(value)
    if (value := node.get("maxLength")) is not None:
        out["max_length"] = int(value)
    if (value := node.get("minLength")) is not None:
        out["min_length"] = int(value)

    if out["type"] == "ARRAY":
        if (value := node.get("maxItems")) is not None:
            out["max_items"] = int(value)
        if (value := node.get("minItems")) is not None:
            out["min_items"] = int(value)
        items = node.get("items")
        out["items"] = _convert(items, defs) if isinstance(items, dict) else {"type": "STRING"}

    if out["type"] == "OBJECT":
        properties = node.get("properties")
        if isinstance(properties, dict) and properties:
            out["properties"] = {name: _convert(child, defs) for name, child in properties.items()}
            # Every property required. Nothing is optional in an assessment: a field the
            # model is allowed to omit comes back as a default the caller cannot tell
            # apart from a considered answer.
            out["required"] = list(properties)
            out["property_ordering"] = list(properties)
            # No ``additional_properties`` here. ``types.Schema`` carries the field, but the
            # REST surface rejects it outright -- "Unknown name \"additional_properties\"
            # at 'generation_config.response_schema'" -- so emitting it turns every request
            # into a 400. It costs nothing to omit: a response_schema with `required` set is
            # already closed in practice, and the reply is validated against the pydantic
            # model here afterwards, which forbids extras for real.
        else:
            # A free-form map such as ``preconditions_met: dict[str, bool]``. Its value
            # type is still pinned; only the keys are open, and the count is capped.
            # The value type of a free-form map cannot be pinned for the same reason:
            # the field the SDK would carry it in is not accepted by the API. The map stays
            # open and the pydantic model checks the value types on the way back.
            if (value := node.get("maxProperties")) is not None:
                out["max_properties"] = int(value)
    return out


def gemini_schema_for(schema: type[BoundedOut]) -> dict[str, Any]:
    """``response_schema`` for a bounded task schema, as a plain dict.

    A dict rather than a ``types.Schema`` so this module needs no SDK import to build a
    request shape, and so the whole conversion is testable with nothing installed. The SDK
    validates it into a ``types.Schema`` on the way out, which is asserted in the suite.
    """
    document = schema.model_json_schema()
    defs = document.get("$defs") or {}
    converted = _convert(document, defs)
    converted.setdefault("description", (schema.__doc__ or "").strip().split("\n")[0])
    return converted


# ---------------------------------------------------------------------------
# Rate-limit recognition
# ---------------------------------------------------------------------------


def is_rate_limited(exc: BaseException) -> bool:
    """True when ``exc`` is the provider saying "too many requests".

    Recognised structurally rather than by SDK class, because the SDK is optional, the
    error type is injectable in tests, and a quota rejection arrives in more than one
    shape: ``ClientError(code=429)``, an HTTP status on a nested response, or -- for a
    daily quota -- a ``RESOURCE_EXHAUSTED`` status with no numeric code in reach.
    """
    for attribute in ("code", "status_code"):
        try:
            if int(getattr(exc, attribute, 0) or 0) == 429:
                return True
        except (TypeError, ValueError):
            pass
    response = getattr(exc, "response", None)
    try:
        if int(getattr(response, "status_code", 0) or 0) == 429:
            return True
    except (TypeError, ValueError):
        pass
    text = f"{type(exc).__name__}: {exc}".upper()
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "RATE_LIMIT" in text


def retry_delay_of(exc: BaseException) -> float | None:
    """The ``retryDelay`` Google attaches to a quota error, in seconds, when present.

    A 429 from the Gemini API carries a ``RetryInfo`` detail saying when the window
    reopens. Honouring it beats guessing: an exponential backoff that undershoots earns a
    second 429 and spends another attempt from the budget.
    """
    for attribute in ("details", "response_json", "body"):
        payload = getattr(exc, attribute, None)
        found = _retry_delay_in(payload)
        if found is not None:
            return found
    return _retry_delay_in(str(exc))


def _retry_delay_in(payload: Any) -> float | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        marker = payload.find("retryDelay")
        if marker < 0:
            return None
        tail = payload[marker : marker + 80]
        digits = "".join(ch for ch in tail.split(":", 1)[-1] if ch.isdigit() or ch == ".")
        try:
            return float(digits) if digits else None
        except ValueError:
            return None
    if isinstance(payload, dict):
        if (value := payload.get("retryDelay")) is not None:
            try:
                return float(str(value).rstrip("s"))
            except ValueError:
                return None
        for child in payload.values():
            if (found := _retry_delay_in(child)) is not None:
                return found
        return None
    if isinstance(payload, (list, tuple)):
        for child in payload:
            if (found := _retry_delay_in(child)) is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


@register_backend(LLMBackendKind.GEMINI)
class GeminiBackend(LLMBackend):
    """Calls the Gemini API for one bounded assessment at a time."""

    kind = LLMBackendKind.GEMINI

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: Any | None = None,
        heuristic: LLMBackend | None = None,
        cache: LLMResponseCache | None = None,
        sleep: Callable[[float], None] | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        """Build the backend, failing immediately when it cannot possibly work.

        A missing key is a configuration error rather than a runtime surprise halfway
        through a scan, so it is raised here. ``client`` is injectable so the test suite
        can exercise call shape, pacing, the 429 path and the fallback with no key and no
        network.
        """
        self.config = config or LLMConfig()
        self.model_id = resolve_model(self.config.model)
        self.api_key_env = resolve_key_env(self.config.api_key_env)
        self.heuristic = heuristic or HeuristicBackend()
        self.cache = cache if cache is not None else LLMResponseCache(
            self.config.cache_dir, enabled=self.config.use_cache
        )
        self._sleep = sleep or (lambda _seconds: None)
        self.limiter = limiter if limiter is not None else RateLimiter(
            self.config.requests_per_minute, sleep=self._sleep
        )
        self._api_key = os.environ.get(self.api_key_env, "")
        #: Every request this backend issued, for tests and for a live-run audit.
        self.requests: list[dict[str, Any]] = []
        #: ``google.genai.types`` when the SDK is present, else ``None``. Only used to
        #: turn the request dict into a real config object before it goes out.
        self._types: Any = None

        if client is not None:
            self.client = client
            self._types = _import_types()
            return
        if not self._api_key:
            raise ConfigError(
                f"Gemini backend requires an API key in ${self.api_key_env}; get a free one "
                "at https://aistudio.google.com/apikey, set it, inject a client, or use the "
                "heuristic backend."
            )
        try:
            from google import genai  # noqa: PLC0415 - lazy: the package is optional
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ConfigError(
                "the 'google-genai' package is not installed; install vulnpriority[gemini]"
            ) from exc
        self._types = _import_types()
        self.client = genai.Client(api_key=self._api_key)

    # -- LLMBackend --------------------------------------------------------

    def available(self) -> bool:
        """True when a client is present; a key-less injected client still counts."""
        return self.client is not None

    def complete_structured(self, prompt: SandboxedPrompt, schema: type[BoundedOut]) -> LLMResult:
        """One assessment: cache, then up to ``max_retries + 1`` attempts, then fallback."""
        task = prompt.task or task_for_schema(schema)

        cached = self.cache.load(prompt.prompt_hash, schema)
        if cached is not None:
            audit = mark_cached(self._audit(prompt, task, schema_retries=0))
            return LLMResult(parsed=cached, raw_text=cached.model_dump_json(), audit=audit)

        attempts = max(1, self.config.max_retries + 1)
        schema_retries = 0
        rate_limited = False
        last_error: Exception | None = None
        started = time.perf_counter()

        for attempt in range(attempts):
            try:
                response = self._call(prompt, schema, task, strict=attempt > 0)
                parsed, raw_text = self._extract(response, schema)
            except RateLimited as exc:
                rate_limited = True
                last_error = exc
                if attempt + 1 < attempts:
                    self._sleep(
                        backoff_delay(
                            attempt,
                            self.config.retry_backoff_s,
                            self.config.max_backoff_s,
                            exc.retry_after,
                        )
                    )
                continue
            except Exception as exc:  # noqa: BLE001 - every other failure retries alike
                last_error = exc
                if isinstance(exc, ValidationError):
                    schema_retries += 1
                if attempt + 1 < attempts:
                    self._sleep(0.0)
                continue

            audit = self._audit(
                prompt,
                task,
                schema_retries=schema_retries,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                usage=getattr(response, "usage_metadata", None),
            )
            self.cache.store(prompt.prompt_hash, schema.__name__, raw_text, task=task)
            return LLMResult(parsed=parsed, raw_text=raw_text, audit=audit)

        if not self.config.fallback_to_heuristic:
            raise ConfigError(
                f"Gemini call for task {task!r} failed after {attempts} attempt(s): {last_error}"
            )
        return self._fallback(
            prompt, schema, schema_retries=schema_retries, rate_limited=rate_limited, started=started
        )

    # -- request -----------------------------------------------------------

    def _user_message(self, prompt: SandboxedPrompt, schema: type[BoundedOut], strict: bool) -> str:
        """Render the user turn; ``strict`` adds a reminder used only on a retry."""
        untrusted = "\n\n".join(text for _segment_id, text, _provenance in prompt.untrusted_blocks)
        body = render_task(prompt.task, prompt.operator_context, untrusted, schema.__name__)
        if strict:
            body += (
                "\n\nThe previous reply was rejected. Return one JSON object matching the "
                "response schema exactly, supply every field within its bounds, add no "
                "field that is not in the schema, and quote evidence spans verbatim."
            )
        return body

    def generation_config(self, prompt: SandboxedPrompt, schema: type[BoundedOut]) -> dict[str, Any]:
        """The ``GenerateContentConfig`` payload.

        Note what is here and what is not. ``response_mime_type`` plus ``response_schema``
        is the whole structured-output mechanism; there are no tools, because this backend
        judges evidence it was handed and must not fetch any of its own. Temperature is
        pinned to the configured 0 so an evaluation's numbers do not move between runs.
        """
        return {
            "system_instruction": prompt.system,
            "temperature": float(self.config.temperature),
            "max_output_tokens": int(self.config.max_tokens),
            "response_mime_type": "application/json",
            "response_schema": gemini_schema_for(schema),
        }

    def _call(self, prompt: SandboxedPrompt, schema: type[BoundedOut], task: str, strict: bool) -> Any:
        """Issue one paced request. A 429 becomes :class:`RateLimited`, never a raw error."""
        payload = self.generation_config(prompt, schema)
        request = {
            "model": self.model_id,
            "contents": self._user_message(prompt, schema, strict),
            "config": payload,
        }
        self.requests.append(request)
        self.limiter.wait()
        config: Any = payload
        if self._types is not None:
            # Validated locally so a malformed schema is a build-time error here rather
            # than a 400 on the wire that costs an attempt from the retry budget.
            config = self._types.GenerateContentConfig(**payload)
        try:
            return self.client.models.generate_content(
                model=request["model"], contents=request["contents"], config=config
            )
        except Exception as exc:  # noqa: BLE001 - re-raised, or narrowed to RateLimited
            if is_rate_limited(exc):
                raise RateLimited(
                    f"Gemini rate limit (429) for task {task!r}: {exc}",
                    retry_after=retry_delay_of(exc),
                ) from exc
            raise

    # -- response ----------------------------------------------------------

    @staticmethod
    def _response_text(response: Any) -> str:
        """The model's JSON text, from ``response.text`` or by walking the parts.

        ``.text`` is the SDK's convenience accessor and is what a real response provides;
        the parts walk covers a response object that only carries candidates, which is the
        shape a hand-built test double and a partially blocked response both have.
        """
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        collected: list[str] = []
        for candidate in getattr(response, "candidates", None) or ():
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or ():
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text:
                    collected.append(part_text)
        joined = "".join(collected)
        if not joined.strip():
            raise ValueError("no text in the Gemini response")
        return joined

    @classmethod
    def _extract(cls, response: Any, schema: type[BoundedOut]) -> tuple[BoundedOut, str]:
        """Parse and validate. Anything that does not validate is a failed attempt."""
        parsed = schema.model_validate(json.loads(cls._response_text(response)))
        return parsed, parsed.model_dump_json()

    # -- fallback and audit ------------------------------------------------

    def _fallback(
        self,
        prompt: SandboxedPrompt,
        schema: type[BoundedOut],
        schema_retries: int,
        rate_limited: bool,
        started: float,
    ) -> LLMResult:
        """The heuristic answer, with an audit that says the model did not produce it.

        This is the path a free tier will actually take on a busy scan, so it is the one
        that has to be honest: ``backend`` reports HEURISTIC because the heuristic is what
        answered, and ``fell_back_to_heuristic`` is set so the report, the consistency
        guard and the manipulation detector all see a degraded finding rather than a
        confident one.
        """
        fallback = self.heuristic.complete_structured(prompt, schema)
        audit = fallback.audit.model_copy(
            update={
                "fell_back_to_heuristic": True,
                "schema_retries": schema_retries + (1 if rate_limited else 0),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        return LLMResult(parsed=fallback.parsed, raw_text=fallback.raw_text, audit=audit)

    def _audit(
        self,
        prompt: SandboxedPrompt,
        task: str,
        schema_retries: int = 0,
        latency_ms: float = 0.0,
        usage: Any | None = None,
    ) -> LLMAudit:
        signals: list[InjectionSignal] = []
        for report in prompt.reports:
            signals.extend(report.signals)
        return LLMAudit(
            backend=LLMBackendKind.GEMINI,
            model=self.model_id,
            task=task,
            prompt_hash=prompt.prompt_hash,
            schema_retries=schema_retries,
            signals=tuple(signals),
            max_tier_used=prompt.max_tier_used,
            input_tokens=max(0, int(getattr(usage, "prompt_token_count", 0) or 0)),
            output_tokens=max(0, int(getattr(usage, "candidates_token_count", 0) or 0)),
            latency_ms=max(0.0, latency_ms),
        )


def _import_types() -> Any:
    """``google.genai.types``, or ``None``. Absence is never fatal at import time."""
    try:  # pragma: no cover - depends on the environment
        from google.genai import types  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    return types
