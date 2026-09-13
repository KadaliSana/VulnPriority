"""LLM backend layer (DESIGN 3.4).

Importing this package registers the built-in backends in
:data:`vulnprio.core.registry.LLM_BACKENDS`. Component A must obtain its backend from
:func:`build_llm_backend`, which always returns a guarded one.
"""

from __future__ import annotations

from vulnprio.llm.anthropic_backend import AnthropicBackend, tool_spec_for
from vulnprio.llm.cache import LLMResponseCache, mark_cached
from vulnprio.llm.consistency import ConsistencyGuard, ConsistencyOutcome
from vulnprio.llm.factory import build_inner_backend, build_llm_backend
from vulnprio.llm.gemini_backend import (
    DEFAULT_GEMINI_MODEL,
    GeminiBackend,
    gemini_schema_for,
)
from vulnprio.llm.guarded import STRICT_RETRY_INSTRUCTION, GuardedBackend, OutputGuardLike
from vulnprio.llm.heuristic import (
    HeuristicBackend,
    sanitized_text_of,
    score_applicability,
    score_asset_criticality,
    score_exploitability,
    strip_imperative_sentences,
)
from vulnprio.llm.openai_compatible import (
    OpenAICompatibleBackend,
    RateLimited,
    RateLimiter,
    openai_json_schema,
)
from vulnprio.llm.prompts import (
    OPERATOR_FACT_KEYS,
    SYSTEM_PROMPTS,
    TASK_TEMPLATES,
    TASKS,
    format_operator_context,
    parse_operator_context,
    prompt_hash,
    prompt_hashes,
    render_task,
    system_prompt,
)
from vulnprio.llm.schemas import (
    ApplicabilityOut,
    AssetCriticalityOut,
    BoundedOut,
    ExploitabilityOut,
    SCHEMA_FOR_TASK,
    schema_for_task,
    task_for_schema,
    to_applicability,
    to_asset_criticality,
    to_exploitability,
)

__all__ = [
    "AnthropicBackend",
    "ApplicabilityOut",
    "AssetCriticalityOut",
    "BoundedOut",
    "ConsistencyGuard",
    "ConsistencyOutcome",
    "DEFAULT_GEMINI_MODEL",
    "ExploitabilityOut",
    "GeminiBackend",
    "GuardedBackend",
    "HeuristicBackend",
    "LLMResponseCache",
    "OPERATOR_FACT_KEYS",
    "OpenAICompatibleBackend",
    "OutputGuardLike",
    "RateLimited",
    "RateLimiter",
    "SCHEMA_FOR_TASK",
    "STRICT_RETRY_INSTRUCTION",
    "SYSTEM_PROMPTS",
    "TASKS",
    "TASK_TEMPLATES",
    "build_inner_backend",
    "build_llm_backend",
    "format_operator_context",
    "gemini_schema_for",
    "mark_cached",
    "openai_json_schema",
    "parse_operator_context",
    "prompt_hash",
    "prompt_hashes",
    "render_task",
    "sanitized_text_of",
    "schema_for_task",
    "score_applicability",
    "score_asset_criticality",
    "score_exploitability",
    "strip_imperative_sentences",
    "system_prompt",
    "task_for_schema",
    "to_applicability",
    "to_asset_criticality",
    "to_exploitability",
    "tool_spec_for",
]
