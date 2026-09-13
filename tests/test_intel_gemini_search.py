"""Phase-1 request shape for the Gemini grounded-search provider.

``GeminiSearchProvider`` had no tests; these cover the one thing about its request that
is easy to regress silently, because getting it wrong costs a log line rather than a
failure.
"""

from __future__ import annotations

import pytest

from vulnpriority.core.config import IntelConfig
from vulnpriority.intel.gemini_search import GeminiSearchProvider


@pytest.fixture
def phase_one_config() -> dict:
    """The ``config`` block of a phase-1 request, built with no key and no client."""
    provider = GeminiSearchProvider(IntelConfig())
    return provider.request_kwargs(IntelConfig(), "CVE-2026-0001 exploit")["config"]


def test_phase_one_disables_automatic_function_calling(phase_one_config):
    """Grounding is server-side, so there is no local callable for the SDK to invoke.

    Any ``tools`` at all routes ``models.generate_content`` through the SDK's automatic
    function calling path, which logs advice to use ``Chat.send_message`` instead. The
    call works either way; this keeps a scan's logs free of advice that does not apply.
    """
    assert phase_one_config["automatic_function_calling"] == {"disable": True}


def test_the_sdk_accepts_the_config_block_as_written(phase_one_config):
    """The dict is handed to the SDK unparsed, so a wrong key would surface as a 400."""
    types = pytest.importorskip("google.genai.types")

    settled = types.GenerateContentConfig(**phase_one_config)

    assert settled.automatic_function_calling.disable is True
    assert settled.tools, "phase one is a grounded search; it must carry the search tool"
