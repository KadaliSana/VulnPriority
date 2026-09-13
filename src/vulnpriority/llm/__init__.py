"""LLM backend layer (DESIGN 3.4).

Importing this package registers the built-in backends in
:data:`vulnpriority.core.registry.LLM_BACKENDS`. Component A must obtain its backend from
:func:`build_llm_backend`, which always returns a guarded one.
"""

from __future__ import annotations

from vulnpriority.llm.anthropic_backend import AnthropicBackend, tool_spec_for
from vulnpriority.llm.cache import LLMResponseCache, mark_cached
from vulnpriority.llm.consistency import ConsistencyGuard, ConsistencyOutcome
from vulnpriority.llm.factory import build_inner_backend, build_llm_backend
from vulnpriority.llm.gemini_backend import (
    DEFAULT_GEMINI_MODEL,
    GeminiBackend,
    gemini_schema_for,
)
from vulnpriority.llm.guarded import STRICT_RETRY_INSTRUCTION, GuardedBackend, OutputGuardLike
from vulnpriority.llm.heuristic import (
    HeuristicBackend,
    sanitized_text_of,
    score_applicability,
    score_asset_criticality,
    score_exploitability,
    strip_imperative_sentences,
)
from vulnpriority.llm.openai_compatible import (
    OpenAICompatibleBackend,
    RateLimited,
    RateLimiter,
    openai_json_schema,
)
from vulnpriority.llm.prompts import (
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
from vulnpriority.llm.schemas import (
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
