"""Feed bundle construction and the default intel assembler.

``build_feed_bundle`` is the single place where :class:`FeedMode` turns into concrete
clients, so "offline by default" is enforced by construction rather than by discipline:
in ``OFFLINE`` mode no live client is ever instantiated.

``DefaultIntelAssembler`` merges the five feeds into one :class:`VulnIntel` per CVE and lets
that model's own validator be the final arbiter of temporal leakage - the assembler does not
quietly repair a feed that returns future-dated evidence, it fails.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

from pydantic import ValidationError

from vulnprio.core.config import FeedsConfig, PipelineConfig
from vulnprio.core.enums import FeedMode
from vulnprio.core.errors import ConfigError, TemporalLeakageError
from vulnprio.core.interfaces import FeedBundle, IntelAssembler
from vulnprio.core.models import ReferenceDoc, VulnIntel
from vulnprio.core.resolve import concrete_feeds
from vulnprio.feeds.base import resolve_fixture_dir, wrap_with_cache
from vulnprio.feeds.cache import FileCacheStore
from vulnprio.feeds.epss import EpssFixtureFeed, EpssLiveFeed
from vulnprio.feeds.exploitdb import ExploitFixtureFeed, ExploitLiveFeed
from vulnprio.feeds.kev import KevFixtureFeed, KevLiveFeed
from vulnprio.feeds.nvd import NvdFixtureFeed, NvdLiveFeed
from vulnprio.feeds.references import ReferenceFixtureFetcher, ReferenceLiveFetcher

__all__ = [
    "DefaultIntelAssembler",
    "build_feed_bundle",
    "build_fixture_bundle",
    "build_live_bundle",
    "feeds_config_of",
]


def feeds_config_of(config: FeedsConfig | PipelineConfig | None) -> FeedsConfig:
    """Accept either the whole pipeline config or just its feeds section."""
    if config is None:
        return FeedsConfig()
    if isinstance(config, PipelineConfig):
        return config.feeds
    if isinstance(config, FeedsConfig):
        return config
    raise ConfigError(f"expected FeedsConfig or PipelineConfig, got {type(config).__name__}")


def _resolve(path: Any) -> Any:
    """Resolve a possibly relative config path against the project root."""
    return resolve_fixture_dir(path)


def build_fixture_bundle(config: FeedsConfig | PipelineConfig | None = None) -> FeedBundle:
    """Every feed served from ``data/fixtures/feeds``. No HTTP client is created."""
    feeds = feeds_config_of(config)
    fixture_dir = _resolve(feeds.fixture_dir)
    return FeedBundle(
        nvd=NvdFixtureFeed(fixture_dir),
        epss=EpssFixtureFeed(fixture_dir),
        kev=KevFixtureFeed(fixture_dir),
        exploits=ExploitFixtureFeed(fixture_dir),
        references=ReferenceFixtureFetcher(
            fixture_dir,
            allowlist=feeds.reference_host_allowlist,
            char_budget=feeds.reference_char_budget,
        ),
        mode=FeedMode.OFFLINE,
    )


def build_live_bundle(
    config: FeedsConfig | PipelineConfig | None = None, *, with_cache: bool = True
) -> FeedBundle:
    """Live clients, optionally wrapped in the on-disk cache (``live_with_cache``)."""
    feeds = feeds_config_of(config)
    api_key = os.environ.get(feeds.nvd_api_key_env) or None
    timeout = feeds.http_timeout_s
    clients = {
        "nvd": NvdLiveFeed(api_key=api_key, timeout_s=timeout),
        "epss": EpssLiveFeed(timeout_s=timeout),
        "kev": KevLiveFeed(timeout_s=timeout),
        "exploits": ExploitLiveFeed(timeout_s=timeout),
        "references": ReferenceLiveFetcher(
            allowlist=feeds.reference_host_allowlist,
            max_bytes=feeds.max_reference_bytes,
            char_budget=feeds.reference_char_budget,
            timeout_s=timeout,
        ),
    }
    if not with_cache:
        return FeedBundle(**clients, mode=FeedMode.LIVE)
    store = FileCacheStore(_resolve(feeds.cache_dir), feeds.cache_ttl_days)
    cached = {name: wrap_with_cache(client, store) for name, client in clients.items()}
    return FeedBundle(**cached, mode=FeedMode.LIVE_WITH_CACHE)


def build_feed_bundle(config: FeedsConfig | PipelineConfig | None = None) -> FeedBundle:
    """Build the bundle the configuration asks for.

    ``AUTO`` -- the default -- resolves here, once, against a short reachability probe:
    ``LIVE_WITH_CACHE`` when the network is up and ``OFFLINE`` when it is not. This is the
    only place feed mode is dispatched on, which is what keeps the probe to one per run
    rather than one per CVE. The live modes exist so a real deployment can refresh
    intelligence, and ``LIVE_WITH_CACHE`` is what makes such a run replayable afterwards.

    An unrecognised mode still raises rather than defaulting to a live path: failing loudly
    is the right behaviour for a switch that decides whether sockets get opened.
    """
    feeds = concrete_feeds(feeds_config_of(config))
    if feeds.mode == FeedMode.OFFLINE:
        return build_fixture_bundle(feeds)
    if feeds.mode == FeedMode.LIVE:
        return build_live_bundle(feeds, with_cache=False)
    if feeds.mode == FeedMode.LIVE_WITH_CACHE:
        return build_live_bundle(feeds, with_cache=True)
    raise ConfigError(f"unsupported feed mode: {feeds.mode}")


class DefaultIntelAssembler(IntelAssembler):
    """Merges NVD, EPSS, KEV, exploit evidence and reference pages into one ``VulnIntel``."""

    def __init__(
        self,
        bundle: FeedBundle,
        config: FeedsConfig | PipelineConfig | None = None,
    ) -> None:
        self.bundle = bundle
        self.config = feeds_config_of(config)

    # -- reference handling -------------------------------------------------

    def _fetch_references(
        self, skeleton: VulnIntel | None, as_of: date, max_references: int
    ) -> tuple[ReferenceDoc, ...]:
        """Replace NVD's URL stubs with fetched bodies, skipping what cannot be retrieved.

        A stub whose page is refused (host not allowed, byte cap, fetch error) is dropped
        rather than kept empty: an empty document would enter the sandbox as a real piece of
        evidence and dilute the reference features.
        """
        if skeleton is None or max_references <= 0:
            return ()
        limit = min(int(max_references), int(self.config.max_references_per_cve))
        if limit <= 0:
            return ()
        docs: list[ReferenceDoc] = []
        for stub in skeleton.references:
            if len(docs) >= limit:
                break
            fetched = self.bundle.references.get(stub.url, as_of)
            if fetched is None:
                continue
            tags = fetched.tags or stub.tags
            docs.append(fetched.model_copy(update={"tags": tags}) if tags != fetched.tags else fetched)
        return tuple(docs)

    # -- assembly -----------------------------------------------------------

    def assemble(self, cve_id: str, as_of: date, max_references: int = 8) -> VulnIntel:
        """One CVE's intelligence as of ``as_of``.

        Every feed is queried with the same cut-off; the resulting model is validated by
        :class:`VulnIntel`, whose ``_no_future_leakage`` validator is the contract that a
        misbehaving feed is caught by. That failure is re-raised as
        :class:`TemporalLeakageError` so callers can distinguish it from a schema bug.
        """
        normalised = str(cve_id).strip().upper()
        skeleton = self.bundle.nvd.get(normalised, as_of)
        epss = self.bundle.epss.get(normalised, as_of)
        kev = self.bundle.kev.get(normalised, as_of)
        exploits = tuple(self.bundle.exploits.get(normalised, as_of) or ())
        references = self._fetch_references(skeleton, as_of, max_references)

        fields: dict[str, Any] = {
            "cve_id": normalised,
            "as_of": as_of,
            "epss": epss,
            "kev": kev,
            "exploits": exploits,
            "references": references,
        }
        if skeleton is not None:
            last_modified = skeleton.last_modified
            if last_modified is not None and last_modified > as_of:
                last_modified = None
            fields.update(
                description=skeleton.description,
                published=skeleton.published,
                last_modified=last_modified,
                cvss=skeleton.cvss,
                affected=skeleton.affected,
            )
        try:
            return VulnIntel(**fields)
        except ValidationError as exc:
            raise TemporalLeakageError(
                f"assembling {normalised} as of {as_of.isoformat()} produced future-dated intel: {exc}"
            ) from exc

    def assemble_many(
        self, cve_ids: tuple[str, ...], as_of: date, max_references: int = 8
    ) -> tuple[VulnIntel, ...]:
        """Assemble several CVEs, preserving the caller's order and dropping duplicates."""
        seen: set[str] = set()
        out: list[VulnIntel] = []
        for cve_id in cve_ids:
            normalised = str(cve_id).strip().upper()
            if not normalised or normalised in seen:
                continue
            seen.add(normalised)
            out.append(self.assemble(normalised, as_of, max_references))
        return tuple(out)
