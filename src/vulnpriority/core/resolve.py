"""Run-time resolution of the ``auto`` switches, and the reasons behind each one.

Three settings can say ``auto``: :attr:`FeedsConfig.mode`, :attr:`LLMConfig.backend` and
:attr:`IntelConfig.mode`/``enabled``. ``auto`` means *prefer the real thing, degrade to
offline, and say which happened*. It is the default, so an operator who has a key and a
network gets a live assessment without configuring one, and an operator who has neither
gets exactly the offline run this framework has always produced.

Three rules make that safe rather than merely convenient.

**An explicit setting is never overridden.** ``auto`` resolves; ``offline`` means offline,
``anthropic`` means Anthropic and fails loudly if its key is missing. Silent substitution
for a value someone actually wrote would make configuration meaningless.

**Resolution never touches the configuration that gets hashed.** ``PipelineConfig.hash()``
feeds the run id, so a config mutated by what happened to be in the environment would make
run directories environment-dependent and reproducibility a lie. Resolution returns a
separate :class:`RunResolution`; the config it describes is untouched.

**Degradation is loud.** Every resolution carries a sentence saying what ran and what would
change it, and that sentence reaches the run manifest, the ``run-all`` summary, the API and
the report. A reader must never have to guess whether a report came from a live run or a
fixture run. The pattern is :class:`~vulnpriority.core.models.AsOfVerdict`'s: a machine-readable
decision next to the reason a person reads.

**The research protocol still wins.** ``research_mode`` forces intelligence offline no
matter what keys are present, because an evaluation that quietly acquired today's internet
because a key happened to be exported would stop being reproducible -- which is the exact
failure the as-of discipline exists to prevent.
"""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from vulnpriority.core.enums import FeedMode, LLMBackendKind

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from vulnpriority.core.config import FeedsConfig, IntelConfig, LLMConfig, PipelineConfig

__all__ = [
    "BACKEND_PROBE_ORDER",
    "PROBE_HOST",
    "PROBE_PORT",
    "PROBE_TIMEOUT_S",
    "SEARCH_CAPABLE_BACKENDS",
    "Resolution",
    "RunResolution",
    "concrete_feeds",
    "concrete_intel",
    "concrete_llm",
    "network_is_reachable",
    "reset_network_probe",
    "resolve_backend",
    "resolve_feed_mode",
    "resolve_intel",
    "resolve_run",
]

#: What ``auto`` tries, in order, and why this order.
#:
#: Cost ascending, because cost is the constraint that makes a free tier worth having. A
#: configured ``base_url`` comes first because writing one down is already a deliberate
#: choice -- and because a local server (Ollama, LM Studio, vLLM) is both free and the only
#: option where scan-derived text never leaves the machine. Gemini's free tier comes next,
#: then Anthropic, which always costs money.
#:
#: A user who wants a specific backend sets ``llm.backend`` and this list is not consulted.
BACKEND_PROBE_ORDER: tuple[tuple[LLMBackendKind, str], ...] = (
    (LLMBackendKind.OPENAI_COMPATIBLE, "llm.base_url"),
    (LLMBackendKind.GEMINI, "GEMINI_API_KEY"),
    (LLMBackendKind.ANTHROPIC, "ANTHROPIC_API_KEY"),
)

#: Backends that can search the internet. Intelligence gathering needs one: the
#: OpenAI-compatible providers expose no search tool, so a run on Groq or a local Ollama
#: gets model-backed assessment and no exploit intelligence, which is the honest outcome
#: rather than an empty result dressed up as a search.
SEARCH_CAPABLE_BACKENDS: frozenset[LLMBackendKind] = frozenset(
    {LLMBackendKind.ANTHROPIC, LLMBackendKind.GEMINI}
)

#: Reachability probe target. The NVD host is what the feed layer actually needs, so
#: probing it answers the question being asked rather than a proxy for it.
PROBE_HOST = "nvd.nist.gov"
PROBE_PORT = 443
#: Deliberately short. A probe is a fast check for a capability, not a retry loop, and a
#: run that was going to be offline anyway must not pay seconds to discover it.
PROBE_TIMEOUT_S = 1.5

#: Probe result for this process. ``auto`` feeds resolve once per run, never per CVE.
_NETWORK: bool | None = None


def reset_network_probe() -> None:
    """Forget the cached probe result. For tests, and for a long-lived server process."""
    global _NETWORK
    _NETWORK = None


def network_is_reachable(timeout_s: float = PROBE_TIMEOUT_S) -> bool:
    """One cached TCP probe of the feed host.

    Cached for the life of the process because the answer is a property of the machine, not
    of the CVE being looked up, and because probing per lookup would add a round trip to
    every one of them. A failure of any kind means "not reachable": there is no error here
    worth distinguishing, since every one of them ends in the same offline run.
    """
    global _NETWORK
    if _NETWORK is None:
        try:
            with socket.create_connection((PROBE_HOST, PROBE_PORT), timeout=timeout_s):
                _NETWORK = True
        except OSError:
            _NETWORK = False
    return _NETWORK


class Resolution(BaseModel):
    """What one ``auto`` switch became, and why.

    ``configured`` and ``resolved`` are the machine-readable pair; ``reason`` is the
    sentence a person reads. Keeping both is the point: a log line nobody can act on and a
    status code nobody can interpret are equally useless.
    """

    model_config = ConfigDict(frozen=True)

    switch: str
    configured: str
    resolved: str
    automatic: bool = False
    live: bool = False
    reason: str = ""

    def line(self) -> str:
        """One-line summary in the register the CLI and the report both use."""
        return f"{self.switch}: {self.resolved}" + (f" ({self.reason})" if self.reason else "")


class RunResolution(BaseModel):
    """Every ``auto`` switch for one run, resolved together."""

    model_config = ConfigDict(frozen=True)

    feeds: Resolution
    llm: Resolution
    intel: Resolution

    @property
    def all(self) -> tuple[Resolution, ...]:
        return (self.feeds, self.llm, self.intel)

    @property
    def automatic(self) -> bool:
        """True when at least one switch was resolved rather than configured."""
        return any(item.automatic for item in self.all)

    @property
    def feed_mode(self) -> FeedMode:
        return FeedMode(self.feeds.resolved)

    @property
    def backend(self) -> LLMBackendKind:
        return LLMBackendKind(self.llm.resolved)

    @property
    def intel_enabled(self) -> bool:
        return self.intel.live

    def lines(self) -> tuple[str, ...]:
        return tuple(item.line() for item in self.all)

    def notes(self) -> dict[str, str]:
        """``switch -> reason``, for a JSON payload or a report table."""
        return {item.switch: item.reason for item in self.all}


# ---------------------------------------------------------------------------
# The three switches
# ---------------------------------------------------------------------------


def _key_present(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def resolve_feed_mode(feeds: "FeedsConfig", probe: bool = True) -> Resolution:
    """``FeedsConfig.mode``, resolving ``auto`` against network reachability.

    ``probe`` exists so a caller that already knows the answer -- or a test that must not
    touch a socket -- can skip the check.
    """
    configured = FeedMode(feeds.mode)
    if configured is not FeedMode.AUTO:
        return Resolution(
            switch="feeds",
            configured=configured.value,
            resolved=configured.value,
            automatic=False,
            live=configured is not FeedMode.OFFLINE,
            reason="set explicitly in the configuration",
        )

    if probe and network_is_reachable():
        return Resolution(
            switch="feeds",
            configured="auto",
            resolved=FeedMode.LIVE_WITH_CACHE.value,
            automatic=True,
            live=True,
            reason=f"{PROBE_HOST} is reachable, so feeds are live and cached on disk",
        )
    return Resolution(
        switch="feeds",
        configured="auto",
        resolved=FeedMode.OFFLINE.value,
        automatic=True,
        live=False,
        reason=(
            f"no network ({PROBE_HOST}:{PROBE_PORT} unreachable); serving the recorded "
            "fixture corpus instead"
        ),
    )


def resolve_backend(llm: "LLMConfig") -> Resolution:
    """``LLMConfig.backend``, resolving ``auto`` to the first usable provider.

    The order is :data:`BACKEND_PROBE_ORDER` and it is cost-ascending. Explicitly setting
    ``llm.backend`` skips all of this, including the case where the chosen backend's key is
    missing: that is a configuration error the backend itself raises, and turning it into a
    silent downgrade to the heuristic would hide a broken deployment behind a plausible
    number.
    """
    configured = LLMBackendKind(llm.backend)
    if configured is not LLMBackendKind.AUTO:
        return Resolution(
            switch="llm",
            configured=configured.value,
            resolved=configured.value,
            automatic=False,
            live=configured is not LLMBackendKind.HEURISTIC,
            reason="set explicitly in the configuration",
        )

    from vulnpriority.core.config import LLMConfig  # noqa: PLC0415 - local, avoids a cycle

    default_key_env = LLMConfig.model_fields["api_key_env"].default
    default_model = LLMConfig.model_fields["model"].default
    tried: list[str] = []

    for kind, requirement in BACKEND_PROBE_ORDER:
        if kind is LLMBackendKind.OPENAI_COMPATIBLE:
            if not llm.base_url:
                continue
            if not llm.model or llm.model == default_model:
                tried.append("llm.base_url is set but llm.model is not")
                continue
            from vulnpriority.llm.openai_compatible import (  # noqa: PLC0415 - local
                DEFAULT_KEY_ENV,
                is_local_endpoint,
            )

            key_env = DEFAULT_KEY_ENV if llm.api_key_env == default_key_env else llm.api_key_env
            if is_local_endpoint(llm.base_url):
                return _chose(
                    kind,
                    f"llm.base_url points at a local server ({llm.base_url}); "
                    "no key needed and nothing leaves this machine",
                )
            if _key_present(key_env):
                return _chose(kind, f"llm.base_url is set and ${key_env} is present")
            tried.append(f"${key_env} is not set for {llm.base_url}")
            continue

        key_env = requirement
        if _key_present(key_env):
            free = " (free tier)" if kind is LLMBackendKind.GEMINI else ""
            return _chose(kind, f"${key_env} is present{free}")
        tried.append(f"${key_env} is not set")

    return Resolution(
        switch="llm",
        configured="auto",
        resolved=LLMBackendKind.HEURISTIC.value,
        automatic=True,
        live=False,
        reason=(
            "no model credentials found ("
            + "; ".join(tried)
            + "); using the deterministic heuristic assessor"
        ),
    )


def _chose(kind: LLMBackendKind, reason: str) -> Resolution:
    return Resolution(
        switch="llm",
        configured="auto",
        resolved=kind.value,
        automatic=True,
        live=True,
        reason=reason,
    )


def resolve_intel(
    intel: "IntelConfig", backend: Resolution, feeds: Resolution | None = None
) -> Resolution:
    """``IntelConfig.mode``/``enabled``, resolving ``auto`` against the resolved backend.

    Intelligence is never on without a backend that can actually search, and never on under
    ``research_mode``. Both refusals are stated rather than implied: "off" and "off because
    the only key present was for a backend with no search tool" are different facts, and the
    second one tells the operator what to change.
    """
    configured_mode = str(intel.mode)
    configured_enabled = intel.enabled
    explicit_mode = configured_mode != "auto"
    explicit_enabled = configured_enabled != "auto"

    def verdict(resolved: str, live: bool, reason: str, automatic: bool = True) -> Resolution:
        return Resolution(
            switch="intel",
            configured=configured_mode if explicit_mode else "auto",
            resolved=resolved,
            automatic=automatic,
            live=live,
            reason=reason,
        )

    # An explicit off, either way round, is final.
    if explicit_enabled and not configured_enabled:
        return verdict("offline", False, "disabled in the configuration", automatic=False)
    if explicit_mode and configured_mode == "offline":
        return verdict(
            "offline", False, "offline in the configuration; serving the recorded corpus",
            automatic=False,
        )

    if intel.research_mode:
        return verdict(
            "offline",
            False,
            "research protocol: live search is refused so the evaluation stays reproducible",
        )

    resolved_backend = LLMBackendKind(backend.resolved)
    if resolved_backend not in SEARCH_CAPABLE_BACKENDS:
        if resolved_backend is LLMBackendKind.HEURISTIC:
            detail = backend.reason or "no model credentials found"
            return verdict("offline", False, f"no searching backend available: {detail}")
        return verdict(
            "offline",
            False,
            f"the {resolved_backend.value} backend has no web-search tool; "
            "set GEMINI_API_KEY or ANTHROPIC_API_KEY to gather exploit intelligence",
        )

    if feeds is not None and not feeds.live and feeds.automatic:
        return verdict("offline", False, "no network; serving the recorded intel corpus")

    free = " on the free tier" if resolved_backend is LLMBackendKind.GEMINI else ""
    return verdict(
        "live",
        True,
        f"searching with the {resolved_backend.value} backend{free}",
        automatic=not (explicit_mode and configured_mode == "live"),
    )


# ---------------------------------------------------------------------------
# Concretisers: a section with every ``auto`` replaced by what it resolved to
# ---------------------------------------------------------------------------
#
# These return copies. Nothing here mutates the configuration the caller hashed, so a run
# id stays a function of what was written down rather than of what happened to be in the
# environment when it ran.


def concrete_feeds(feeds: "FeedsConfig", resolution: Resolution | None = None) -> "FeedsConfig":
    """``feeds`` with ``mode`` settled. Already-explicit configuration copies unchanged."""
    if FeedMode(feeds.mode) is not FeedMode.AUTO:
        return feeds
    resolved = resolution or resolve_feed_mode(feeds)
    return feeds.model_copy(update={"mode": FeedMode(resolved.resolved)})


def concrete_llm(llm: "LLMConfig", resolution: Resolution | None = None) -> "LLMConfig":
    """``llm`` with ``backend`` settled."""
    if LLMBackendKind(llm.backend) is not LLMBackendKind.AUTO:
        return llm
    resolved = resolution or resolve_backend(llm)
    return llm.model_copy(update={"backend": LLMBackendKind(resolved.resolved)})


def concrete_intel(
    intel: "IntelConfig",
    resolution: Resolution,
    backend: LLMBackendKind | None = None,
) -> "IntelConfig":
    """``intel`` with ``enabled``, ``mode`` and ``search_provider`` settled.

    ``search_provider`` follows the resolved backend, so a run that found only a Gemini key
    searches with Google Search grounding rather than trying the Anthropic server tools it
    has no key for -- unless a Parallel key is also present, in which case the dedicated
    Search API wins. An explicitly configured provider is left alone.
    """
    update: dict[str, object] = {}
    if intel.enabled == "auto":
        update["enabled"] = resolution.live
    if intel.mode == "auto":
        update["mode"] = "live" if resolution.live else "offline"
    if backend is LLMBackendKind.GEMINI and intel.search_provider == "anthropic":
        # "anthropic" is the field default rather than a stated preference; a run that
        # resolved to Gemini has no Anthropic key to search with. Prefer Parallel when its
        # key is there: it is pure retrieval, so phase 1 is search and nothing else, and it
        # spends no part of the free tier's per-model request budget, which the extraction
        # step needs. Without that key, Google Search grounding needs no second credential.
        update["search_provider"] = (
            "parallel" if _key_present(intel.parallel_api_key_env) else "gemini"
        )
    return intel.model_copy(update=update) if update else intel


def resolve_run(config: "PipelineConfig", probe: bool = True) -> RunResolution:
    """Resolve every ``auto`` switch for one run, in dependency order.

    Feeds first because the network probe is shared, backend second, intelligence last
    because it depends on both. ``config`` is read and never written: the caller keeps the
    configuration it hashed, and gets the resolution beside it.
    """
    feeds = resolve_feed_mode(config.feeds, probe=probe)
    backend = resolve_backend(config.llm)
    intel = resolve_intel(config.intel, backend, feeds)
    return RunResolution(feeds=feeds, llm=backend, intel=intel)
