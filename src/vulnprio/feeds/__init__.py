"""External vulnerability intelligence, as-of dated and offline by default (DESIGN.md 3.2).

Public surface:

* feed clients - ``Nvd*``, ``Epss*``, ``Kev*``, ``Exploit*``, ``Reference*`` - each with a
  fixture implementation (checked-in data, no sockets) and a live implementation (httpx,
  real API shapes);
* :func:`build_feed_bundle` / :class:`DefaultIntelAssembler`, which compose them into one
  :class:`~vulnprio.core.models.VulnIntel` per CVE;
* :func:`select_cvss` and :func:`submetric_flags`, the CVSS selection policy (Gap 3);
* :class:`FileCacheStore`, which makes a live run replayable.
"""

from __future__ import annotations

from vulnprio.feeds.base import (
    CachingEpssFeed,
    CachingExploitFeed,
    CachingFeed,
    CachingKevFeed,
    CachingNvdFeed,
    CachingReferenceFetcher,
    FixtureFeed,
    LiveFeedBase,
    as_of_guard,
    clear_fixture_cache,
    load_fixture_records,
    parse_feed_date,
    parse_feed_datetime,
    require_not_future,
    resolve_fixture_dir,
    wrap_with_cache,
)
from vulnprio.feeds.bundle import (
    DefaultIntelAssembler,
    build_feed_bundle,
    build_fixture_bundle,
    build_live_bundle,
    feeds_config_of,
)
from vulnprio.feeds.cache import CacheEntry, FileCacheStore, utc_now
from vulnprio.feeds.cvss_policy import (
    CVSS_VERSION_ORDER,
    SOURCE_PRIORITY,
    SUBMETRIC_FLAG_NAMES,
    cvss_features,
    cvss_version_ordinal,
    parse_cvss_vector,
    select_cvss,
    source_agreement,
    submetric_flags,
)
from vulnprio.feeds.epss import EPSS_API_URL, EpssFixtureFeed, EpssLiveFeed, parse_epss_rows
from vulnprio.feeds.exploitdb import (
    EXPLOITDB_CSV_URL,
    ExploitFixtureFeed,
    ExploitLiveFeed,
    classify_maturity,
    extract_cve_codes,
)
from vulnprio.feeds.kev import KEV_CATALOG_URL, KevFixtureFeed, KevLiveFeed, kev_record_from_entry
from vulnprio.feeds.nvd import (
    NVD_API_URL,
    NvdFixtureFeed,
    NvdLiveFeed,
    parse_nvd_configurations,
    parse_nvd_cve,
    parse_nvd_cvss,
    reference_urls,
)
from vulnprio.feeds.references import (
    DEFAULT_ALLOWLIST,
    LiveReferenceFetcher,
    ReferenceFixtureFetcher,
    ReferenceLiveFetcher,
    detect_language,
    extract_title,
    html_to_text,
    is_allowed_host,
)

__all__ = [
    # cache
    "CacheEntry",
    "FileCacheStore",
    "utc_now",
    # base
    "FixtureFeed",
    "LiveFeedBase",
    "CachingFeed",
    "CachingNvdFeed",
    "CachingEpssFeed",
    "CachingKevFeed",
    "CachingExploitFeed",
    "CachingReferenceFetcher",
    "wrap_with_cache",
    "as_of_guard",
    "require_not_future",
    "parse_feed_date",
    "parse_feed_datetime",
    "resolve_fixture_dir",
    "load_fixture_records",
    "clear_fixture_cache",
    # nvd
    "NVD_API_URL",
    "NvdFixtureFeed",
    "NvdLiveFeed",
    "parse_nvd_cve",
    "parse_nvd_cvss",
    "parse_nvd_configurations",
    "reference_urls",
    # epss
    "EPSS_API_URL",
    "EpssFixtureFeed",
    "EpssLiveFeed",
    "parse_epss_rows",
    # kev
    "KEV_CATALOG_URL",
    "KevFixtureFeed",
    "KevLiveFeed",
    "kev_record_from_entry",
    # exploits
    "EXPLOITDB_CSV_URL",
    "ExploitFixtureFeed",
    "ExploitLiveFeed",
    "classify_maturity",
    "extract_cve_codes",
    # references
    "DEFAULT_ALLOWLIST",
    "ReferenceFixtureFetcher",
    "ReferenceLiveFetcher",
    "LiveReferenceFetcher",
    "html_to_text",
    "extract_title",
    "detect_language",
    "is_allowed_host",
    # cvss policy
    "CVSS_VERSION_ORDER",
    "SOURCE_PRIORITY",
    "SUBMETRIC_FLAG_NAMES",
    "select_cvss",
    "source_agreement",
    "submetric_flags",
    "cvss_features",
    "cvss_version_ordinal",
    "parse_cvss_vector",
    # bundle
    "build_feed_bundle",
    "build_fixture_bundle",
    "build_live_bundle",
    "feeds_config_of",
    "feeds_config_of",
    "DefaultIntelAssembler",
]
