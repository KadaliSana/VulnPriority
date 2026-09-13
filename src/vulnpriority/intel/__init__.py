"""Internet exploit intelligence: the agentic retrieval layer (Architecture item 1).

The literature review this framework implements proposes "an AI agent-based vulnerability
prioritization framework that automates the retrieval and interpretation of references and
exploit information published on the internet". This package is that retrieval and
interpretation. Everything else in the framework reasons over evidence it was handed; this
is the part that goes and finds it.

It runs in two phases, because the API makes one impossible: citations and
``output_config.format`` cannot coexist in a single request. Phase one searches and reads
with citations and no schema; phase two turns the sanitized result into bounded numbers
with a schema and no tools. :mod:`vulnpriority.intel.anthropic_search` explains the request
shapes; :mod:`vulnpriority.intel.agent` explains the as-of discipline, which is the subtlest
thing here.

:mod:`vulnpriority.intel.parallel_search` is the third way to do phase one and the only one
where the split is not forced: Parallel is a search API with no model attached, so there is
no citations-versus-schema conflict to work around, retrieval is retrieval and extraction is
extraction. It costs $1 per 1,000 searches at one search per finding, which is what makes
live intelligence affordable after Gemini's grounding quota turned out to be far tighter
than its generation quota. The price of having no model in phase one is that there is no
researcher narrative and therefore no report summary on that path; the ranking features are
unaffected, because they come from phase two either way.

Three rules from DESIGN.md hold without exception:

* **Untrusted by default.** Every retrieved byte is
  :class:`~vulnpriority.core.models.UntrustedText` at tier ``REFERENCE_PAGE`` and reaches a
  model only through :mod:`vulnpriority.sandbox.pipeline`. Phase two runs under the existing
  ``GuardedBackend``, so the canary, envelope, schema and evidence-span checks are the
  same ones Component A already relies on rather than a second implementation of them.
* **As-of dated.** Every document is stamped with ``retrieved_at``, and
  :func:`~vulnpriority.intel.agent.judge_as_of` decides whether searching now can honestly
  answer a question about this scan's date.
* **Offline by default.** ``IntelConfig.enabled`` is False and ``mode`` is ``offline``.
  The default agent serves from ``data/fixtures/intel/searches.json`` and cannot reach the
  network. No test needs a key.

What the layer produces is not a paragraph in a report. It is a set of features --
:data:`~vulnpriority.intel.models.INTEL_FEATURE_NAMES` -- that the learned ranker consumes
alongside CVSS, EPSS and KEV. The model extracts; the ranker decides. The summary exists
so an analyst can read why, and it is labelled as model-written wherever it appears.
"""

from __future__ import annotations

from vulnpriority.intel.agent import (
    INTEL_TASK,
    ExploitIntelAgent,
    IntelBaselineBackend,
    compare_with_feeds,
    fuse_into_exploitability,
    judge_as_of,
    resolve_intel_config,
    scan_age_days,
)
from vulnpriority.intel.cache import IntelCache, intel_cache_key
from vulnpriority.intel.gemini_search import GeminiSearchProvider, grounding_documents
from vulnpriority.intel.models import (
    EXPLOIT_INTEL_JSON_SCHEMA,
    INTEL_FEATURE_NAMES,
    AgreementAxis,
    AsOfStatus,
    AsOfVerdict,
    ExploitIntelOut,
    FeedAgreement,
    IntelCitation,
    IntelConfig,
    IntelDocument,
    IntelGather,
    IntelQuery,
    IntelQueryKind,
    IntelRemedy,
    IntelResult,
    IntelSourceKind,
    IntelSummary,
    IntelUsage,
    neutral_intel_features,
)
from vulnpriority.intel.offline import FixtureSearchProvider, RecordingProvider
from vulnpriority.intel.parallel_search import (
    PARALLEL_ENDPOINT,
    PARALLEL_MODES,
    PARALLEL_PRICE_PER_REQUEST_USD,
    ParallelSearchProvider,
    parallel_documents,
    search_cost_usd,
)
from vulnpriority.intel.provider import (
    BaseSearchProvider,
    NullSearchProvider,
    SearchProvider,
    build_search_provider,
    make_document,
)
from vulnpriority.intel.queries import (
    PHASE1_SYSTEM,
    PHASE2_SYSTEM,
    build_queries,
    classify_source,
    dedupe_documents,
    known_urls,
    normalize_url,
)
from vulnpriority.intel.summarize import summarize_intel

__all__ = [
    "AgreementAxis",
    "AsOfStatus",
    "AsOfVerdict",
    "BaseSearchProvider",
    "EXPLOIT_INTEL_JSON_SCHEMA",
    "ExploitIntelAgent",
    "ExploitIntelOut",
    "FeedAgreement",
    "FixtureSearchProvider",
    "GeminiSearchProvider",
    "INTEL_FEATURE_NAMES",
    "INTEL_TASK",
    "IntelBaselineBackend",
    "IntelCache",
    "IntelCitation",
    "IntelConfig",
    "IntelDocument",
    "IntelGather",
    "IntelQuery",
    "IntelQueryKind",
    "IntelRemedy",
    "IntelResult",
    "IntelSourceKind",
    "IntelSummary",
    "IntelUsage",
    "NullSearchProvider",
    "PARALLEL_ENDPOINT",
    "PARALLEL_MODES",
    "PARALLEL_PRICE_PER_REQUEST_USD",
    "PHASE1_SYSTEM",
    "PHASE2_SYSTEM",
    "ParallelSearchProvider",
    "RecordingProvider",
    "SearchProvider",
    "build_queries",
    "build_search_provider",
    "classify_source",
    "compare_with_feeds",
    "dedupe_documents",
    "fuse_into_exploitability",
    "grounding_documents",
    "intel_cache_key",
    "judge_as_of",
    "known_urls",
    "make_document",
    "neutral_intel_features",
    "normalize_url",
    "parallel_documents",
    "resolve_intel_config",
    "scan_age_days",
    "search_cost_usd",
    "summarize_intel",
]
