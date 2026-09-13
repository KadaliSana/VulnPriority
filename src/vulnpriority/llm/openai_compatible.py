"""One backend for every provider that speaks OpenAI chat completions.

Groq, OpenRouter, Together, Cerebras, a local Ollama, LM Studio and vLLM are not seven
integrations. They are one request shape -- ``POST {base_url}/chat/completions`` with a
``messages`` array and a bearer token -- behind seven hostnames. So this is one backend
plus a ``base_url``, and the way to reach a free Groq key or a model running on the
operator's own laptop is to name the endpoint, not to add a class.

Written against plain :mod:`httpx`, which the framework already depends on. Adding the
``openai`` package to reach an endpoint that is defined by its wire format would be a
dependency bought for nothing.

**No default endpoint, and no default model.** Both are required configuration. A default
host would be the framework quietly choosing whose servers text derived from someone's
private scan is sent to, and a default model would be a guess that is wrong on six of the
seven providers above. Missing either is a :class:`~vulnpriority.core.errors.ConfigError` at
construction, not a surprise halfway through a scan.

**Structured output degrades, it never corrupts.** ``json_mode`` picks how hard the
provider is asked to produce JSON (see :class:`~vulnpriority.core.config.LLMConfig`), and the
weakest setting is the schema written into the prompt with no API-level enforcement at
all. That is safe because enforcement was never the API's job here: whatever comes back
is parsed, ``model_validate``\\ d against the bounded schema, and then put through the
sandbox output guard. A provider that ignores every JSON instruction produces a rejected
answer and a heuristic fallback with an audit that says so -- a degraded finding, never a
corrupted score.

This module also holds the free-tier plumbing shared with
:mod:`vulnpriority.llm.gemini_backend` -- :class:`RateLimiter` and the 429 backoff -- because
it is the one that needs no optional SDK to import.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from vulnpriority.core.config import LLMConfig
from vulnpriority.core.enums import LLMBackendKind
from vulnpriority.core.errors import ConfigError
from vulnpriority.core.interfaces import LLMBackend, LLMResult, SandboxedPrompt
from vulnpriority.core.models import InjectionSignal, LLMAudit
from vulnpriority.core.registry import register_backend
from vulnpriority.llm.cache import LLMResponseCache, mark_cached
from vulnpriority.llm.heuristic import HeuristicBackend
from vulnpriority.llm.prompts import render_task
from vulnpriority.llm.schemas import BoundedOut, task_for_schema

__all__ = [
    "DEFAULT_KEY_ENV",
    "LOCAL_HOSTS",
    "RateLimiter",
    "RateLimited",
    "OpenAICompatibleBackend",
    "backoff_delay",
    "extract_json_object",
    "is_local_endpoint",
    "openai_json_schema",
    "retry_after_seconds",
    "schema_instruction",
]

#: Consulted when ``api_key_env`` is left at its Anthropic-shaped default.
DEFAULT_KEY_ENV = "OPENAI_API_KEY"

#: Hosts allowed to run without a key. Ollama, LM Studio and vLLM serve an
#: OpenAI-compatible endpoint on the operator's own machine with no authentication, and
#: refusing to talk to them would rule out the one option that is free *and* sends nothing
#: anywhere. A remote host with no key is still a configuration error: an unauthenticated
#: request to someone else's server is a mistake, not a deployment style.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"})


# ---------------------------------------------------------------------------
# Free-tier plumbing: pacing, and what to do about a 429
# ---------------------------------------------------------------------------


class RateLimited(Exception):
    """A provider said 429. Carries the delay it asked for, when it named one."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    """Client-side pacing: never issue more than ``rpm`` requests per minute.

    Pacing is cheaper than recovering. A 429 costs a wasted round trip, a backoff, and --
    on a tight free tier -- often a second 429; spacing the requests costs only the wait
    that was going to happen anyway. A scan issues three model calls per finding, so on a
    15-requests-per-minute quota an unpaced run hits the wall inside the first five
    findings.

    ``rpm <= 0`` disables pacing entirely, which is the default and what the paid paths
    and the whole test suite use. ``monotonic`` and ``sleep`` are injected so tests can
    assert the delays without spending them.
    """

    def __init__(
        self,
        rpm: int = 0,
        *,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.rpm = max(0, int(rpm))
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
        self._last: float | None = None
        #: Every wait this limiter imposed, in seconds. Diagnostics, and test evidence.
        self.waits: list[float] = []

    @property
    def min_interval_s(self) -> float:
        return 0.0 if self.rpm <= 0 else 60.0 / float(self.rpm)

    def wait(self) -> float:
        """Block until the next request is due. Returns how long that took."""
        interval = self.min_interval_s
        now = self._monotonic()
        if interval <= 0.0 or self._last is None:
            self._last = now
            return 0.0
        delay = max(0.0, self._last + interval - now)
        if delay > 0.0:
            self._sleep(delay)
            self.waits.append(delay)
        self._last = self._monotonic()
        return delay


def backoff_delay(attempt: int, base_s: float, max_s: float, retry_after: float | None = None) -> float:
    """Seconds to wait before retry ``attempt`` (0-based).

    Exponential from ``base_s``, capped at ``max_s``. A provider-supplied ``retry_after``
    always wins, because the provider knows when its window reopens and a guess that
    undershoots earns another 429.
    """
    if retry_after is not None and retry_after >= 0.0:
        return min(float(retry_after), max(float(max_s), float(retry_after)))
    return min(max(0.0, float(base_s)) * (2.0**max(0, int(attempt))), max(0.0, float(max_s)))


def retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    """``Retry-After`` as seconds, or ``None`` when absent or not a plain number.

    Only the delta-seconds form is honoured. The HTTP-date form is legal but no free-tier
    provider uses it, and mis-parsing a date into a multi-hour sleep inside a scan would
    be far worse than falling back to the exponential guess.
    """
    if not headers:
        return None
    raw = None
    for name in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        value = headers.get(name) if hasattr(headers, "get") else None
        if value:
            raw = value
            break
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip().rstrip("s"))
    except ValueError:
        return None
    return seconds if seconds >= 0.0 else None


# ---------------------------------------------------------------------------
# Schema handling
# ---------------------------------------------------------------------------


def openai_json_schema(schema: type[BoundedOut]) -> dict[str, Any]:
    """JSON Schema for ``schema`` with every property required, at every level.

    Pydantic marks nothing required here because every bounded field carries a default.
    That is right for validation and wrong for a request: a provider told a field is
    optional will omit it, and the framework then gets a defaulted answer it cannot tell
    apart from a considered one. Structured-output modes also generally insist on it.

    ``extra="forbid"`` on :class:`~vulnpriority.llm.schemas.BoundedOut` already puts
    ``additionalProperties: false`` on the object, which is what stops schema smuggling
    at the provider rather than only at validation.
    """
    document = schema.model_json_schema()

    def require_all(node: Any) -> None:
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        if isinstance(properties, dict) and properties:
            node["required"] = list(properties)
            for child in properties.values():
                require_all(child)
        for key in ("items", "additionalProperties", "not"):
            require_all(node.get(key))
        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            for child in node.get(key) or ():
                require_all(child)
        for child in (node.get("$defs") or {}).values():
            require_all(child)

    require_all(document)
    return document


def schema_instruction(schema: type[BoundedOut]) -> str:
    """The schema written into the prompt, for providers with no JSON mode worth trusting.

    The documented fallback. It is an instruction, not a guarantee, and it is paired with
    strict validation on the way out rather than relied upon.
    """
    return (
        "\n\nReturn a single JSON object and nothing else: no prose, no markdown, no code "
        "fence. It must validate against this JSON Schema exactly, with every property "
        "present and no property that is not listed:\n\n"
        + json.dumps(openai_json_schema(schema), sort_keys=True, indent=1)
    )


def extract_json_object(text: str) -> Any:
    """Parse the one JSON object in a reply, tolerating the wrappers weak models add.

    Tolerant about *packaging* -- a ```` ```json ```` fence, a leading "Here is the
    assessment:" -- and not at all tolerant about content: the result still has to be
    valid JSON and still has to survive ``model_validate``. Raises ``ValueError`` when
    there is no object to be found, which the caller treats as a failed attempt.
    """
    body = (text or "").strip()
    if not body:
        raise ValueError("empty response body")
    if body.startswith("```"):
        body = body.split("\n", 1)[-1] if "\n" in body else ""
        end = body.rfind("```")
        if end >= 0:
            body = body[:end]
        body = body.strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the response")
    return json.loads(body[start : end + 1])


def is_local_endpoint(base_url: str) -> bool:
    """True when ``base_url`` points at the operator's own machine."""
    host = (urlsplit(base_url).hostname or "").lower()
    return host in LOCAL_HOSTS or host.endswith(".local")


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


@register_backend(LLMBackendKind.OPENAI_COMPATIBLE)
class OpenAICompatibleBackend(LLMBackend):
    """Calls ``POST {base_url}/chat/completions`` for one bounded assessment at a time."""

    kind = LLMBackendKind.OPENAI_COMPATIBLE

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: Any | None = None,
        heuristic: LLMBackend | None = None,
        cache: LLMResponseCache | None = None,
        sleep: Callable[[float], None] | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        """Fail here, loudly, when the configuration cannot possibly work.

        ``client`` is an injected :class:`httpx.Client` (or anything with a compatible
        ``post``), which is how the test suite asserts request shape, the 429 path and the
        fallback without a key and without a socket.
        """
        self.config = config or LLMConfig()
        self.base_url = str(self.config.base_url or "").rstrip("/")
        if not self.base_url:
            raise ConfigError(
                "the openai_compatible backend needs llm.base_url, for example "
                "https://api.groq.com/openai/v1 (Groq), https://openrouter.ai/api/v1 "
                "(OpenRouter) or http://localhost:11434/v1 (a local Ollama). There is no "
                "default: naming the endpoint is the operator's decision."
            )

        anthropic_model_default = LLMConfig.model_fields["model"].default
        if not self.config.model or self.config.model == anthropic_model_default:
            raise ConfigError(
                "the openai_compatible backend needs an explicit llm.model, for example "
                "'llama-3.3-70b-versatile' on Groq or 'llama3.2' on a local Ollama; "
                f"llm.model is still the Anthropic default {anthropic_model_default!r}."
            )
        self.model_id = self.config.model

        anthropic_key_default = LLMConfig.model_fields["api_key_env"].default
        self.api_key_env = (
            DEFAULT_KEY_ENV
            if self.config.api_key_env == anthropic_key_default
            else self.config.api_key_env
        )
        self._api_key = os.environ.get(self.api_key_env, "")
        self.local = is_local_endpoint(self.base_url)
        if not self._api_key and not self.local:
            raise ConfigError(
                f"the openai_compatible backend requires an API key in ${self.api_key_env} "
                f"for {self.base_url}; set it, set llm.api_key_env to the variable you use, "
                "point llm.base_url at a local server (Ollama, LM Studio, vLLM), or use the "
                "heuristic backend."
            )

        self.heuristic = heuristic or HeuristicBackend()
        self.cache = cache if cache is not None else LLMResponseCache(
            self.config.cache_dir, enabled=self.config.use_cache
        )
        self._sleep = sleep or (lambda _seconds: None)
        self.limiter = limiter if limiter is not None else RateLimiter(
            self.config.requests_per_minute, sleep=self._sleep
        )
        #: Every request body this backend sent, for tests and for a live-run audit.
        self.requests: list[dict[str, Any]] = []
        self._owns_client = client is None
        self.client = client if client is not None else httpx.Client(timeout=self.config.timeout_s)

    # -- LLMBackend --------------------------------------------------------

    def available(self) -> bool:
        return self.client is not None

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def headers(self) -> dict[str, str]:
        """Request headers. The bearer is simply absent on a keyless local server."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

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
                payload = self._call(prompt, schema, task, strict=attempt > 0)
                parsed, raw_text = self._extract(payload, schema)
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
            except Exception as exc:  # noqa: BLE001 - every failure mode retries alike
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
                usage=payload.get("usage") if isinstance(payload, dict) else None,
            )
            self.cache.store(prompt.prompt_hash, schema.__name__, raw_text, task=task)
            return LLMResult(parsed=parsed, raw_text=raw_text, audit=audit)

        if not self.config.fallback_to_heuristic:
            raise ConfigError(
                f"{self.base_url} call for task {task!r} failed after {attempts} "
                f"attempt(s): {last_error}"
            )
        return self._fallback(
            prompt, schema, schema_retries=schema_retries, rate_limited=rate_limited, started=started
        )

    # -- request -----------------------------------------------------------

    def _user_message(self, prompt: SandboxedPrompt, schema: type[BoundedOut], strict: bool) -> str:
        """Render the user turn; ``strict`` adds a reminder used only on a retry."""
        untrusted = "\n\n".join(text for _segment_id, text, _provenance in prompt.untrusted_blocks)
        body = render_task(prompt.task, prompt.operator_context, untrusted, schema.__name__)
        if self.config.json_mode in ("auto", "prompt"):
            body += schema_instruction(schema)
        if strict:
            body += (
                "\n\nThe previous reply was rejected. Emit one JSON object and nothing "
                "else, supply every field within its bounds, add no field that is not in "
                "the schema, and quote evidence spans verbatim."
            )
        return body

    def _response_format(self, schema: type[BoundedOut], task: str) -> dict[str, Any] | None:
        """``response_format`` for the configured ``json_mode``, or ``None`` for prompt-only."""
        mode = self.config.json_mode
        if mode == "prompt":
            return None
        if mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": f"record_{task}",
                    "schema": openai_json_schema(schema),
                },
            }
        return {"type": "json_object"}

    def build_request(
        self, prompt: SandboxedPrompt, schema: type[BoundedOut], task: str, strict: bool = False
    ) -> dict[str, Any]:
        """The request body. Exposed so a test can assert the shape without a socket."""
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": self._user_message(prompt, schema, strict)},
            ],
            "max_tokens": int(self.config.max_tokens),
            "temperature": float(self.config.temperature),
            "stream": False,
        }
        response_format = self._response_format(schema, task)
        if response_format is not None:
            body["response_format"] = response_format
        return body

    def _call(
        self, prompt: SandboxedPrompt, schema: type[BoundedOut], task: str, strict: bool
    ) -> dict[str, Any]:
        """Issue one paced request. A 429 becomes :class:`RateLimited`, never a raw error."""
        body = self.build_request(prompt, schema, task, strict=strict)
        self.requests.append(body)
        self.limiter.wait()
        response = self.client.post(self.url, json=body, headers=self.headers())
        status = int(getattr(response, "status_code", 0) or 0)
        if status == 429:
            raise RateLimited(
                f"{self.base_url} rate limit (429) for task {task!r}",
                retry_after=retry_after_seconds(getattr(response, "headers", None)),
            )
        if status >= 400:
            raise ValueError(f"{self.base_url} returned HTTP {status}: {self._text_of(response)[:300]}")
        return response.json()

    @staticmethod
    def _text_of(response: Any) -> str:
        text = getattr(response, "text", "")
        return text if isinstance(text, str) else ""

    # -- response ----------------------------------------------------------

    @staticmethod
    def _message_content(payload: Any) -> str:
        """The assistant text, tolerating the list-of-parts shape some providers return."""
        choices = (payload or {}).get("choices") if isinstance(payload, dict) else None
        if not choices:
            raise ValueError("no choices in the chat-completions response")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            return "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        return str(content or "")

    @classmethod
    def _extract(cls, payload: Any, schema: type[BoundedOut]) -> tuple[BoundedOut, str]:
        """Parse and validate. Anything that does not validate is a failed attempt."""
        parsed = schema.model_validate(extract_json_object(cls._message_content(payload)))
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

        A rate limit degrades a finding; it never aborts a scan.
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
        usage: Mapping[str, Any] | None = None,
    ) -> LLMAudit:
        signals: list[InjectionSignal] = []
        for report in prompt.reports:
            signals.extend(report.signals)
        usage = usage or {}
        return LLMAudit(
            backend=LLMBackendKind.OPENAI_COMPATIBLE,
            model=self.model_id,
            task=task,
            prompt_hash=prompt.prompt_hash,
            schema_retries=schema_retries,
            signals=tuple(signals),
            max_tier_used=prompt.max_tier_used,
            input_tokens=max(0, int(usage.get("prompt_tokens", 0) or 0)),
            output_tokens=max(0, int(usage.get("completion_tokens", 0) or 0)),
            latency_ms=max(0.0, latency_ms),
        )

    def close(self) -> None:
        """Close the HTTP client, but only one this backend created."""
        if self._owns_client and hasattr(self.client, "close"):
            self.client.close()
