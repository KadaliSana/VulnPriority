"""Construction of the backend the rest of the framework uses.

There is exactly one supported way to obtain a backend, and it always returns a
:class:`~vulnpriority.llm.guarded.GuardedBackend`. Wrapping here rather than at the call site
is what makes "Component A may only use the guarded backend" an enforceable property
instead of a convention.
"""

from __future__ import annotations

from typing import Any

from vulnpriority.core.config import LLMConfig, PipelineConfig
from vulnpriority.core.enums import LLMBackendKind
from vulnpriority.core.errors import ConfigError
from vulnpriority.core.interfaces import LLMBackend
from vulnpriority.core.resolve import concrete_llm
from vulnpriority.llm.anthropic_backend import AnthropicBackend
from vulnpriority.llm.cache import LLMResponseCache
from vulnpriority.llm.gemini_backend import GeminiBackend
from vulnpriority.llm.guarded import GuardedBackend
from vulnpriority.llm.heuristic import HeuristicBackend
from vulnpriority.llm.openai_compatible import OpenAICompatibleBackend

__all__ = ["build_llm_backend", "build_inner_backend"]


def _llm_config(config: PipelineConfig | LLMConfig | None) -> LLMConfig:
    if config is None:
        return LLMConfig()
    if isinstance(config, PipelineConfig):
        return config.llm
    if isinstance(config, LLMConfig):
        return config
    raise ConfigError(f"cannot build an LLM backend from {type(config).__name__}")


def build_inner_backend(
    config: PipelineConfig | LLMConfig | None = None,
    client: Any | None = None,
) -> LLMBackend:
    """The unguarded backend named by the configuration.

    Exposed for tests and for the adversarial evaluator, which needs to compare guarded
    against unguarded behaviour. Production code calls :func:`build_llm_backend`.
    """
    # ``AUTO`` -- the default -- resolves here, once: the first backend whose credentials
    # are present, cost-ascending, and the heuristic when none is. This is the only place
    # backend choice is dispatched on, so the resolution cannot disagree with itself.
    llm = concrete_llm(_llm_config(config))
    if llm.backend == LLMBackendKind.HEURISTIC:
        return HeuristicBackend()

    # Every remote backend is built the same way: the same config, the same injected
    # client hook, the same heuristic to fall back to and the same response cache. The
    # differences between providers belong inside the backends, not here.
    remote: dict[LLMBackendKind, type] = {
        LLMBackendKind.ANTHROPIC: AnthropicBackend,
        LLMBackendKind.GEMINI: GeminiBackend,
        LLMBackendKind.OPENAI_COMPATIBLE: OpenAICompatibleBackend,
    }
    backend_class = remote.get(llm.backend)
    if backend_class is None:
        raise ConfigError(f"unsupported LLM backend: {llm.backend}")
    return backend_class(
        llm,
        client=client,
        heuristic=HeuristicBackend(),
        cache=LLMResponseCache(llm.cache_dir, enabled=llm.use_cache),
    )


def build_llm_backend(
    config: PipelineConfig | LLMConfig | None = None,
    sandbox: Any | None = None,
    client: Any | None = None,
) -> LLMBackend:
    """Guarded backend for ``config``.

    ``sandbox`` is optional: when a :class:`~vulnpriority.sandbox.pipeline.Sandbox` is passed
    and exposes an ``output_guard``, that guard is used instead of the one the guarded
    backend would import for itself, so a run's sandbox configuration (bounds, evidence
    requirements) governs the output side too.

    The wrapping is unconditional and applies to every member of
    :class:`~vulnpriority.core.enums.LLMBackendKind`, present and future. That is the point of
    building backends here rather than at the call site: adding a provider adds a model,
    never a second path into the score. Sandbox, canary, envelope integrity, schema
    validation, evidence-span verification and the influence budget apply identically
    whichever model answered, and ``test_llm_factory_guards_every_backend`` asserts it for
    every enum member so a new one cannot be added unguarded.
    """
    llm = _llm_config(config)
    inner = build_inner_backend(llm, client=client)
    return GuardedBackend(
        inner=inner,
        heuristic=HeuristicBackend(),
        config=llm,
        output_guard=getattr(sandbox, "output_guard", None),
    )
