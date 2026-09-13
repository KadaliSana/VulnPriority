"""``concrete_intel`` picks the search provider a run can actually pay for.

The interesting case is the Gemini path. ``search_provider`` defaults to ``anthropic``,
which a Gemini-only run has no key for, so it has to become something else; which of the
two remaining providers it becomes depends on whether a Parallel key is in the
environment. The autouse fixture in ``conftest`` strips every provider key, so each test
here puts back exactly the ones it means to test with.
"""

from __future__ import annotations

import pytest

from vulnpriority.core.config import IntelConfig
from vulnpriority.core.resolve import LLMBackendKind, Resolution, concrete_intel


def live_resolution() -> Resolution:
    """The resolution a run gets when intel resolved to live searching."""
    return Resolution(
        switch="intel",
        configured="auto",
        resolved="live",
        automatic=True,
        live=True,
        reason="searching with the gemini backend on the free tier",
    )


def test_gemini_run_with_a_parallel_key_searches_with_parallel(monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")
    settled = concrete_intel(IntelConfig(), live_resolution(), LLMBackendKind.GEMINI)
    assert settled.search_provider == "parallel"


def test_gemini_run_without_a_parallel_key_falls_back_to_grounding(monkeypatch):
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    settled = concrete_intel(IntelConfig(), live_resolution(), LLMBackendKind.GEMINI)
    assert settled.search_provider == "gemini"


def test_a_blank_parallel_key_is_not_a_key(monkeypatch):
    """An exported-but-empty variable is the shape a broken ``.env`` leaves behind."""
    monkeypatch.setenv("PARALLEL_API_KEY", "   ")
    settled = concrete_intel(IntelConfig(), live_resolution(), LLMBackendKind.GEMINI)
    assert settled.search_provider == "gemini"


def test_an_explicit_provider_is_left_alone(monkeypatch):
    """``configs/free-gemini.yaml`` pins ``gemini`` to stay free; Parallel bills per call."""
    monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")
    pinned = IntelConfig(search_provider="gemini")
    settled = concrete_intel(pinned, live_resolution(), LLMBackendKind.GEMINI)
    assert settled.search_provider == "gemini"


@pytest.mark.parametrize("backend", [LLMBackendKind.ANTHROPIC, None])
def test_only_the_gemini_path_reroutes(monkeypatch, backend):
    """A run that can use the Anthropic server tools keeps them, Parallel key or not."""
    monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")
    settled = concrete_intel(IntelConfig(), live_resolution(), backend)
    assert settled.search_provider == "anthropic"
