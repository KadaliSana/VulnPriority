"""Anthropic backend: structured output via a forced tool call.

Structured output is obtained by declaring the bounded pydantic schema as a tool and
forcing the model to call it (``tool_choice={"type": "tool", "name": ...}``). That is
deliberate: it moves schema enforcement to the provider, so an answer that would violate
the bounds usually never gets generated, and anything that slips through still faces
``model_validate`` here and the sandbox output guard afterwards.

Temperature is pinned to 0 and the model id is pinned in config, because an evaluation
whose numbers move between runs cannot support a claim. The SDK is imported lazily
inside the constructor so that the whole package remains importable -- and the offline
test suite remains runnable -- on a machine with no ``anthropic`` installed and no key.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from pydantic import ValidationError

from vulnprio.core.config import LLMConfig
from vulnprio.core.enums import LLMBackendKind
from vulnprio.core.errors import ConfigError
from vulnprio.core.interfaces import LLMBackend, LLMResult, SandboxedPrompt
from vulnprio.core.models import InjectionSignal, LLMAudit
from vulnprio.core.registry import register_backend
from vulnprio.llm.cache import LLMResponseCache, mark_cached
from vulnprio.llm.heuristic import HeuristicBackend
from vulnprio.llm.prompts import render_task
from vulnprio.llm.schemas import BoundedOut, task_for_schema

__all__ = ["AnthropicBackend", "tool_spec_for"]


def tool_spec_for(schema: type[BoundedOut], task: str) -> dict[str, Any]:
    """Tool declaration carrying the schema's JSON Schema as its input shape."""
    return {
        "name": f"record_{task}",
        "description": (
            f"Record the {task.replace('_', ' ')} assessment. Every field is bounded; "
            "quote evidence spans verbatim from the supplied data."
        ),
        "input_schema": schema.model_json_schema(),
    }


@register_backend(LLMBackendKind.ANTHROPIC)
class AnthropicBackend(LLMBackend):
    """Calls the Anthropic Messages API for one bounded assessment at a time."""

    kind = LLMBackendKind.ANTHROPIC

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: Any | None = None,
        heuristic: LLMBackend | None = None,
        cache: LLMResponseCache | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Build the backend, failing immediately when it cannot possibly work.

        A missing key is a configuration error rather than a runtime surprise halfway
        through a scan, so it is raised here. ``client`` is injectable so the test suite
        can exercise call shape, retries and fallback without a network or a key.
        """
        self.config = config or LLMConfig()
        self.model_id = self.config.model
        self.heuristic = heuristic or HeuristicBackend()
        self.cache = cache if cache is not None else LLMResponseCache(
            self.config.cache_dir, enabled=self.config.use_cache
        )
        self._sleep = sleep or (lambda _seconds: None)
        self._api_key = os.environ.get(self.config.api_key_env, "")

        if client is not None:
            self.client = client
            return
        if not self._api_key:
            raise ConfigError(
                f"Anthropic backend requires an API key in ${self.config.api_key_env}; "
                "set it, inject a client, or use the heuristic backend."
            )
        try:
            import anthropic  # noqa: PLC0415 - lazy: the package is optional
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ConfigError("the 'anthropic' package is not installed") from exc
        self.client = anthropic.Anthropic(
            api_key=self._api_key,
            timeout=self.config.timeout_s,
            max_retries=0,  # retries are handled here so they are visible in the audit
        )

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
        last_error: Exception | None = None
        started = time.perf_counter()

        for attempt in range(attempts):
            try:
                response = self._call(prompt, schema, task, strict=attempt > 0)
                parsed, raw_text = self._extract(response, schema)
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
                usage=getattr(response, "usage", None),
            )
            self.cache.store(prompt.prompt_hash, schema.__name__, raw_text, task=task)
            return LLMResult(parsed=parsed, raw_text=raw_text, audit=audit)

        if not self.config.fallback_to_heuristic:
            raise ConfigError(
                f"Anthropic call for task {task!r} failed after {attempts} attempts: {last_error}"
            )

        fallback = self.heuristic.complete_structured(prompt, schema)
        audit = fallback.audit.model_copy(
            update={
                "fell_back_to_heuristic": True,
                "schema_retries": schema_retries,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        return LLMResult(parsed=fallback.parsed, raw_text=fallback.raw_text, audit=audit)

    # -- internals ---------------------------------------------------------

    def _user_message(self, prompt: SandboxedPrompt, schema: type[BoundedOut], strict: bool) -> str:
        """Render the user turn; ``strict`` adds a reminder used only on a retry."""
        untrusted = "\n\n".join(text for _segment_id, text, _provenance in prompt.untrusted_blocks)
        body = render_task(prompt.task, prompt.operator_context, untrusted, schema.__name__)
        if strict:
            body += (
                "\n\nThe previous reply was rejected. Call the tool exactly once, supply every "
                "field within its bounds, add no field that is not in the schema, and quote "
                "evidence spans verbatim."
            )
        return body

    def _call(self, prompt: SandboxedPrompt, schema: type[BoundedOut], task: str, strict: bool) -> Any:
        tool = tool_spec_for(schema, task)
        return self.client.messages.create(
            model=self.model_id,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=prompt.system,
            messages=[{"role": "user", "content": self._user_message(prompt, schema, strict)}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
        )

    @staticmethod
    def _extract(response: Any, schema: type[BoundedOut]) -> tuple[BoundedOut, str]:
        """Pull the forced tool call out of a Messages response and validate it."""
        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if block_type != "tool_use":
                continue
            payload = getattr(block, "input", None)
            if payload is None and isinstance(block, dict):
                payload = block.get("input")
            parsed = schema.model_validate(payload)
            return parsed, parsed.model_dump_json()
        raise ValueError("no tool_use block in the Anthropic response")

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
            backend=LLMBackendKind.ANTHROPIC,
            model=self.model_id,
            task=task,
            prompt_hash=prompt.prompt_hash,
            schema_retries=schema_retries,
            signals=tuple(signals),
            max_tier_used=prompt.max_tier_used,
            input_tokens=max(0, int(getattr(usage, "input_tokens", 0) or 0)),
            output_tokens=max(0, int(getattr(usage, "output_tokens", 0) or 0)),
            latency_ms=max(0.0, latency_ms),
        )
