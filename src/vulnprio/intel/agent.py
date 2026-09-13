"""The two-phase exploit-intelligence agent.

    phase 1   search and read the internet, with citations, no schema
    sandbox   every retrieved byte through vulnprio.sandbox, tier REFERENCE_PAGE
    phase 2   convert the sanitized material into bounded numbers, with a schema,
              no tools, no citations, through the existing GuardedBackend
    fuse      cap the movement at the reference-page influence budget, compare the
              claims against curated-feed evidence, and emit features

**As-of discipline -- the subtlest failure mode in this feature.** Every other feed in
this framework is as-of dated: ask NVD about a 2024 scan and you get what NVD knew in
2024. A live web search has no such property. It returns *today's* internet, including the
proof-of-concept published last week and the CISA advisory issued last month.

Whether that is a problem depends entirely on which of two runs is happening, and
conflating them serves neither:

*The operational run.* Someone points the tool at an application they are authorised to
test, it scans now, and the scan's date is now. Live search is not leakage here -- it is
the entire point. Current internet, current scan, current answer. The guard stays silent
and needs no flag to do so.

*The research run.* A historical scan, or a dataset of scans over time, replayed for the
time-ordered evaluation. Searching today's internet for a scan dated last year imports
knowledge that did not exist at the time: the model appears to predict exploitation
because it read the news about it, the time-ordered split quietly stops meaning anything,
and every number downstream is invalid in a way no other test would catch.

So the discriminator is the *age of the scan*, not a mode flag someone has to remember to
set. :func:`judge_as_of` returns an :class:`~vulnprio.intel.models.AsOfVerdict` carrying a
status, a machine-readable remedy and a sentence. A scan inside ``max_scan_age_days``
proceeds silently. A stale scan in the interactive path is not simply refused -- it comes
back with ``IntelRemedy.RESCAN_TARGET``, so the web application can offer a "Re-scan this
target" button rather than telling an operator no, and an explicit
``allow_anachronistic_search`` lets them search anyway with the mismatch stamped on the
result. A stale scan under ``research_mode`` is refused outright with
``IntelRemedy.USE_FIXTURES``, because there the correct action is a recorded corpus.

Either way the provenance is stamped: every :class:`~vulnprio.intel.models.IntelDocument`
carries ``retrieved_at``, and every result carries the scan date it was gathered for and
whether the two agreed, so a report read months later shows which it was.

**Nothing here ever raises into a scan.** No key, no network, a server-tool error, a
schema rejection, a canary leak: each produces an :class:`IntelResult` with ``errors``
populated, no extraction, and neutral features. One finding's failed web research must not
be able to abort a scan.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from random import Random
from typing import Any, Sequence

from vulnprio.core.config import PipelineConfig
from vulnprio.core.enums import (
    AttackComplexity,
    ExploitMaturity,
    LLMBackendKind,
    Provenance,
    TrustTier,
)
from vulnprio.core.hashing import config_hash
from vulnprio.core.interfaces import LLMBackend, LLMResult, SandboxedPrompt
from vulnprio.core.models import Finding, LLMAudit, UntrustedText, VulnIntel
from vulnprio.core.resolve import concrete_intel, resolve_backend, resolve_intel
from vulnprio.intel.cache import IntelCache, intel_cache_key
from vulnprio.intel.models import (
    MAX_CLAIMED_VERSIONS,
    MAX_EXPLOIT_URLS,
    AgreementAxis,
    AsOfStatus,
    AsOfVerdict,
    ExploitIntelOut,
    FeedAgreement,
    IntelConfig,
    IntelRemedy,
    IntelDocument,
    IntelGather,
    IntelResult,
    IntelSourceKind,
    IntelUsage,
)
from vulnprio.intel.offline import FixtureSearchProvider
from vulnprio.intel.provider import (
    BaseSearchProvider,
    NullSearchProvider,
    build_search_provider,
)
from vulnprio.intel.queries import (
    PHASE1_SYSTEM,
    PHASE2_SYSTEM,
    build_queries,
    dedupe_documents,
    known_urls,
    phase1_instruction,
    phase2_context,
)
from vulnprio.intel.summarize import summarize_intel
from vulnprio.llm.guarded import GuardedBackend
from vulnprio.llm.heuristic import sanitized_text_of, strip_imperative_sentences
from vulnprio.sandbox.pipeline import Sandbox, build_sandboxed_prompt
from vulnprio.semantic.criticality import apply_budget, clamp01, heuristic_audit
from vulnprio.semantic.cpe_match import parse_version, version_in_range
from vulnprio.semantic.exploitability import extract_maturity, fuse_maturity, structured_maturity

__all__ = [
    "INTEL_TASK",
    "PROMPT_FINGERPRINT",
    "IntelBaselineBackend",
    "ExploitIntelAgent",
    "as_of_is_anachronistic",
    "compare_with_feeds",
    "fuse_into_exploitability",
    "judge_as_of",
    "resolve_intel_config",
    "scan_age_days",
]

#: Task name carried on the phase-2 prompt and its audit record.
INTEL_TASK = "exploit_intel"

#: Hash of both frozen system prompts. Part of the cache key, so editing a prompt
#: invalidates cached intelligence instead of silently serving answers to a question that
#: is no longer being asked.
PROMPT_FINGERPRINT = config_hash({"phase1": PHASE1_SYSTEM, "phase2": PHASE2_SYSTEM})


# ---------------------------------------------------------------------------
# Deterministic phase-2 baseline
# ---------------------------------------------------------------------------

_ACTIVE_EXPLOITATION = re.compile(
    r"(?i)exploited\s+in\s+the\s+wild|active(?:ly)?\s+exploit(?:ed|ation)|"
    r"in[\s-]the[\s-]wild\s+exploitation|observed\s+exploitation|"
    r"under\s+active\s+attack|mass\s+exploitation"
)
#: A *denial* that exploitation is happening, which is a claim about the world. Written
#: narrowly on purpose: "no report of exploitation was located in this search" is a
#: statement about the search, not about the vulnerability, and treating it as a denial
#: would make every thorough negative result look like a contradiction of CISA.
_EXPLOITATION_DENIAL = re.compile(
    r"(?i)\b(?:is|are|was|were|has|have)\s+not\s+(?:been\s+)?(?:actively\s+)?exploited\b"
    r"|\bnot\s+(?:known\s+to\s+be\s+|being\s+)?exploited\s+in\s+the\s+wild\b"
    r"|\bno\s+known\s+exploitation\b"
    r"|\bnever\s+been\s+exploited\b"
    r"|\bthere\s+is\s+no\s+(?:evidence|indication|sign)\s+of\s+(?:active\s+)?exploitation\b"
    r"|\bnot\s+exploitable\s+in\s+practice\b"
)
_EXPLOIT_URL = re.compile(
    r"(?i)\bhttps?://(?:www\.)?(?:github\.com|gitlab\.com|exploit-db\.com|"
    r"www\.exploit-db\.com|packetstormsecurity\.com)/[^\s<>\"')\]]+"
)
#: Versions stated to be *affected*. "Fixed in 2.5.22" and "upgrade to 2.17.1" name the
#: remedy, not the vulnerability, and letting them into a field called
#: ``affected_versions_claimed`` would make an advisory that names its own patch look like
#: it contradicted the NVD range.
_VERSION_CLAIM = re.compile(
    r"(?i)\b(?:affect(?:s|ed|ing)?|prior\s+to|before|earlier\s+than|through|up\s+to|"
    r"version[s]?|vulnerable)\b[^.\n]{0,60}?(\d+\.\d+(?:\.\d+){0,2}(?:\.x)?)"
)
_FIX_CONTEXT = re.compile(
    r"(?i)\b(?:fixed|patched|resolved|addressed|corrected)\s+in\b|\bupgrade\s+to\b|"
    r"\bupdate\s+to\b|\brelease[ds]?\s+as\b"
)
_HIGH_COMPLEXITY = re.compile(
    r"(?i)non[\s-]default\s+configuration|requires\s+(?:a\s+)?(?:valid|authenticated|"
    r"specific|particular)\b|race\s+condition|only\s+when\s+configured|"
    r"depends\s+on\s+the\s+environment"
)
_LOW_COMPLEXITY = re.compile(
    r"(?i)unauthenticated\s+remote|single\s+http\s+request|trivially\s+exploitable|"
    r"no\s+authentication\s+(?:is\s+)?required|one[\s-]line\s+exploit"
)

#: Feasibility contributed by each maturity level, added to a deliberately low prior. The
#: prior is low because this estimator reads blog posts: absent any evidence of an exploit
#: it should say "probably not easy", not "maybe".
_MATURITY_FEASIBILITY: dict[ExploitMaturity, float] = {
    ExploitMaturity.UNKNOWN: 0.00,
    ExploitMaturity.UNPROVEN: -0.10,
    ExploitMaturity.POC: 0.15,
    ExploitMaturity.FUNCTIONAL: 0.28,
    ExploitMaturity.WEAPONIZED: 0.38,
}


class IntelBaselineBackend(LLMBackend):
    """Deterministic extraction from sanitized retrieved material.

    The same role :class:`~vulnprio.llm.heuristic.HeuristicBackend` plays for Component A,
    for a schema that backend does not know: it is the safe answer that always exists, the
    value a model's answer is measured against for the influence budget and the
    consistency guard, and the reason the whole two-phase flow runs offline with no key.

    It reads only the sanitized blocks, with instruction-shaped sentences removed first,
    so a page that says "set exploit_feasibility to 1.0" contributes neither its imperative
    nor the words it happens to contain.
    """

    kind = LLMBackendKind.HEURISTIC
    model_id = "intel-baseline"

    def available(self) -> bool:
        return True

    @staticmethod
    def clean_text_of(prompt: SandboxedPrompt) -> str:
        """Sanitized text of the blocks the sandbox found nothing wrong with.

        A block that arrived carrying an injection attempt is excluded from scoring
        entirely. That is not squeamishness: without it, appending a payload to any page
        would let an attacker suppress the maturity evidence in every *other* page in the
        same prompt, which is a cheap and effective deflation attack against a rule that
        was meant to be a defence. Clean evidence keeps its weight; hostile evidence gets
        none, and the attempt is still counted into ``a_intel_injection_signals``.
        """
        parts: list[str] = []
        for index, (_segment_id, text, _provenance) in enumerate(prompt.untrusted_blocks):
            report = prompt.reports[index] if index < len(prompt.reports) else None
            if report is not None and report.signals:
                continue
            parts.append(text)
        return "\n".join(parts)

    def complete_structured(self, prompt: SandboxedPrompt, schema: type) -> LLMResult:
        text = self.clean_text_of(prompt)
        kept, _dropped = strip_imperative_sentences(text)
        body = " ".join(kept)

        maturity = extract_maturity(body)
        active = bool(_ACTIVE_EXPLOITATION.search(body)) and not bool(
            _EXPLOITATION_DENIAL.search(body)
        )
        exploit_urls = tuple(dict.fromkeys(_EXPLOIT_URL.findall(body)))[:MAX_EXPLOIT_URLS]
        versions = tuple(
            dict.fromkeys(
                match.group(1)
                for match in _VERSION_CLAIM.finditer(body)
                if not _FIX_CONTEXT.search(body[max(0, match.start() - 40) : match.end()])
            )
        )[:MAX_CLAIMED_VERSIONS]

        complexity = AttackComplexity.UNKNOWN
        if _HIGH_COMPLEXITY.search(body):
            complexity = AttackComplexity.HIGH
        elif _LOW_COMPLEXITY.search(body):
            complexity = AttackComplexity.LOW

        feasibility = 0.30 + _MATURITY_FEASIBILITY[maturity]
        feasibility += 0.12 if active else 0.0
        feasibility += 0.08 if exploit_urls else 0.0
        feasibility += {AttackComplexity.LOW: 0.08, AttackComplexity.HIGH: -0.10}.get(
            complexity, 0.0
        )
        feasibility = clamp01(feasibility)

        signals = sum(len(report.signals) for report in prompt.reports)
        # Only blocks the sandbox found nothing wrong with raise confidence. A page that
        # arrived with an injection attempt attached must not be able to make the framework
        # *more* sure of anything merely by existing, which is what counting every block
        # would let it do.
        clean_blocks = sum(1 for report in prompt.reports if not report.signals)
        confidence = 0.15 + min(0.45, 0.10 * clean_blocks)
        confidence += 0.10 if maturity != ExploitMaturity.UNKNOWN else 0.0
        confidence -= 0.05 * min(3, signals)
        confidence = clamp01(confidence)

        spans = self._spans(text, kept, maturity, active, exploit_urls)
        rationale = (
            f"deterministic: blocks={len(prompt.untrusted_blocks)} clean={clean_blocks} "
            f"maturity={maturity.name} "
            f"active_claim={active} exploit_urls={len(exploit_urls)} "
            f"versions={len(versions)} signals={signals}"
        )[:600]

        parsed = ExploitIntelOut(
            exploit_maturity=maturity,
            exploit_feasibility=feasibility,
            attack_complexity=complexity,
            preconditions=(),
            affected_versions_claimed=versions,
            public_exploit_urls=exploit_urls,
            active_exploitation_claimed=active,
            confidence=confidence,
            rationale=rationale,
            evidence_spans=spans,
        )
        return LLMResult(
            parsed=parsed,
            raw_text=parsed.model_dump_json(),
            audit=LLMAudit(
                backend=LLMBackendKind.HEURISTIC,
                model=self.model_id,
                task=prompt.task or INTEL_TASK,
                prompt_hash=prompt.prompt_hash,
                signals=tuple(
                    signal for report in prompt.reports for signal in report.signals
                ),
                max_tier_used=prompt.max_tier_used if prompt.reports else TrustTier.OPERATOR,
            ),
        )

    @staticmethod
    def _spans(
        text: str,
        sentences: Sequence[str],
        maturity: ExploitMaturity,
        active: bool,
        exploit_urls: Sequence[str],
    ) -> tuple[str, ...]:
        """Up to three verbatim sentences that carried the claims.

        Verbatim matters: the output guard drops any span that is not a literal substring
        of what the model was shown, and the baseline must satisfy the same rule it is
        holding a model to.
        """
        wanted: list[str] = []
        for sentence in sentences:
            if len(wanted) >= 3:
                break
            relevant = (
                (active and _ACTIVE_EXPLOITATION.search(sentence))
                or (exploit_urls and _EXPLOIT_URL.search(sentence))
                or (
                    maturity != ExploitMaturity.UNKNOWN
                    and extract_maturity(sentence) == maturity
                )
            )
            if relevant:
                span = sentence[:200]
                if span and span in text and span not in wanted:
                    wanted.append(span)
        return tuple(wanted)


# ---------------------------------------------------------------------------
# As-of guard
# ---------------------------------------------------------------------------


def scan_age_days(as_of: date, today: date) -> int:
    """How stale the scan is, in days. Never negative: a future scan is not stale."""
    return max(0, (today - as_of).days)


def as_of_is_anachronistic(as_of: date, today: date, tolerance_days: int = 7) -> bool:
    """True when the scan is old enough that today's internet is a different question."""
    return scan_age_days(as_of, today) > max(0, int(tolerance_days))


def judge_as_of(
    as_of: date,
    today: date,
    config: IntelConfig,
    *,
    live: bool,
) -> AsOfVerdict:
    """Decide whether live search may run, and what to do when it may not.

    Offline runs are never judged: the fixture corpus carries the retrieval dates it was
    recorded with, so replaying it answers the question it was recorded for.
    """
    age = scan_age_days(as_of, today)
    common = {
        "as_of": as_of,
        "evaluated_at": today,
        "age_days": age,
        "max_age_days": int(config.max_scan_age_days),
        "research_mode": bool(config.research_mode),
    }
    if not live:
        return AsOfVerdict(
            status=AsOfStatus.NOT_APPLICABLE,
            remedy=IntelRemedy.NONE,
            search_allowed=True,
            message="offline corpus; retrieval dates are recorded per document",
            **common,
        )
    if age <= int(config.max_scan_age_days):
        return AsOfVerdict(
            status=AsOfStatus.CURRENT,
            remedy=IntelRemedy.NONE,
            search_allowed=True,
            message=(
                f"scan is {age} day(s) old; live intelligence describes the same moment"
            ),
            **common,
        )

    if config.research_mode:
        return AsOfVerdict(
            status=AsOfStatus.REFUSED,
            remedy=IntelRemedy.USE_FIXTURES,
            search_allowed=False,
            message=(
                f"live search refused: this scan is {age} day(s) old and the run is the "
                "time-ordered evaluation protocol, where searching today's internet would "
                "import knowledge that did not exist at the scan date and invalidate the "
                "split. Replay a recorded intel corpus instead."
            ),
            **common,
        )

    if config.allow_anachronistic_search:
        return AsOfVerdict(
            status=AsOfStatus.OVERRIDDEN,
            remedy=IntelRemedy.RESCAN_TARGET,
            search_allowed=True,
            message=(
                f"scan is {age} day(s) old and live search was run anyway by configuration: "
                "the intelligence reflects today, not the scan date. Re-scan the target to "
                "make the two agree."
            ),
            **common,
        )

    return AsOfVerdict(
        status=AsOfStatus.STALE,
        remedy=IntelRemedy.RESCAN_TARGET,
        search_allowed=False,
        message=(
            f"this scan is {age} day(s) old, so live exploit intelligence would describe "
            "today rather than the scan. Re-scan the target so the assessment and the "
            "intelligence agree, or set allow_anachronistic_search to search anyway."
        ),
        **common,
    )


# ---------------------------------------------------------------------------
# Comparison against curated feeds
# ---------------------------------------------------------------------------


def _feed_exploit_maturity(intel: Sequence[VulnIntel]) -> ExploitMaturity:
    best = ExploitMaturity.UNKNOWN
    for item in intel:
        for exploit in item.exploits:
            if int(exploit.maturity) > int(best):
                best = exploit.maturity
    return best


def _upper_bound(product: Any) -> str | None:
    """The highest version an affected range covers, if it names one."""
    return product.version_end_including or product.version_end_excluding or None


def _is_probable_fix_version(version: str, ranges: Sequence[Any]) -> bool:
    """True when a claimed version sits above every affected range.

    Advisories name the release that fixes the issue in the same sentence as the ones it
    affects, and a deterministic reader of prose cannot always tell the two apart. Version
    order can: nothing above the top of every affected range is a claim about what is
    vulnerable.
    """
    bounds = [_upper_bound(product) for product in ranges]
    known = [parse_version(bound) for bound in bounds if bound]
    if not known:
        return False
    return parse_version(version) > max(known)


def compare_with_feeds(
    extraction: ExploitIntelOut | None,
    documents: Sequence[IntelDocument],
    intel: Sequence[VulnIntel],
    sanitized_text: str = "",
) -> FeedAgreement:
    """Compare what the internet said with what the curated feeds already establish.

    This is the layer's most valuable output. "A proof-of-concept exists" means something
    quite different depending on whether CISA also lists the CVE as exploited, and a
    ranking model can only learn that difference if it is handed the difference rather
    than the raw claim. Three axes are checked independently, and the two summary flags are
    not mutually exclusive: material can corroborate exploitation while contradicting the
    affected-version range, and collapsing that into one number would discard the case
    most worth seeing.
    """
    if extraction is None and not documents:
        return FeedAgreement(reason="no retrieved material to compare")

    reasons: list[str] = []
    claims_active = bool(extraction.active_exploitation_claimed) if extraction else False
    denies_active = bool(_EXPLOITATION_DENIAL.search(sanitized_text)) if sanitized_text else False

    # --- axis 1: exploitation, against CISA KEV -------------------------------
    kev_records = [item.kev for item in intel if item.kev is not None]
    if not kev_records:
        kev = AgreementAxis.UNCHECKED
    else:
        in_kev = any(record.in_kev for record in kev_records)
        if in_kev and claims_active:
            kev = AgreementAxis.AGREE
            reasons.append("retrieved material corroborates KEV exploitation")
        elif in_kev and denies_active:
            kev = AgreementAxis.CONTRADICT
            reasons.append("retrieved material denies exploitation that KEV records")
        elif in_kev:
            kev = AgreementAxis.SILENT
        elif claims_active:
            # The feed checked and the CVE is absent from the catalogue, so this is an
            # unverified in-the-wild claim rather than new corroborated information.
            kev = AgreementAxis.CONTRADICT
            reasons.append("active exploitation claimed but the CVE is not listed in KEV")
        else:
            kev = AgreementAxis.SILENT

    # --- axis 2: exploit artefacts, against the exploit feeds ------------------
    feed_maturity = _feed_exploit_maturity(intel)
    claimed_maturity = extraction.exploit_maturity if extraction else ExploitMaturity.UNKNOWN
    poc_urls = bool(extraction.public_exploit_urls) if extraction else False
    poc_documents = any(document.source_kind == IntelSourceKind.POC for document in documents)
    if feed_maturity == ExploitMaturity.UNKNOWN:
        exploit_records = AgreementAxis.UNCHECKED
    elif claimed_maturity == ExploitMaturity.UNKNOWN and not (poc_urls or poc_documents):
        exploit_records = AgreementAxis.SILENT
    elif int(claimed_maturity) >= int(feed_maturity) or poc_urls or poc_documents:
        exploit_records = AgreementAxis.AGREE
        reasons.append("retrieved material corroborates the exploit records")
    elif int(claimed_maturity) <= int(ExploitMaturity.UNPROVEN):
        exploit_records = AgreementAxis.CONTRADICT
        reasons.append("retrieved material calls the issue unproven despite an exploit record")
    else:
        exploit_records = AgreementAxis.AGREE

    # --- axis 3: affected versions, against the NVD ranges ---------------------
    ranges = [product for item in intel for product in item.affected]
    claimed_versions = tuple(extraction.affected_versions_claimed) if extraction else ()
    if not ranges or not claimed_versions:
        affected_versions = AgreementAxis.UNCHECKED
    else:
        inside = [
            version
            for version in claimed_versions
            if any(version_in_range(version, product) for product in ranges)
        ]
        # A version above every affected range is the advisory naming its own patch, not a
        # disagreement about what is vulnerable. Counting it as a contradiction would make
        # every well-written advisory look like it was arguing with NVD.
        outside = [
            version
            for version in claimed_versions
            if version not in inside and not _is_probable_fix_version(version, ranges)
        ]
        if inside:
            affected_versions = AgreementAxis.AGREE
            reasons.append("claimed affected versions fall inside the NVD ranges")
        elif outside:
            affected_versions = AgreementAxis.CONTRADICT
            reasons.append(
                f"claimed affected versions fall outside every NVD range: {outside[:3]}"
            )
        else:
            affected_versions = AgreementAxis.SILENT

    axes = (kev, exploit_records, affected_versions)
    return FeedAgreement(
        kev=kev,
        exploit_records=exploit_records,
        affected_versions=affected_versions,
        corroborates=AgreementAxis.AGREE in axes,
        contradicts=AgreementAxis.CONTRADICT in axes,
        reason="; ".join(reasons)[:400] or "retrieved material adds nothing the feeds settle",
    )


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class ExploitIntelAgent:
    """Runs the two phases for one finding at a time and keeps the running bill."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        provider: BaseSearchProvider | None = None,
        extractor: LLMBackend | None = None,
        *,
        sandbox: Sandbox | None = None,
        cache: IntelCache | None = None,
        today: date | None = None,
        now: datetime | None = None,
    ) -> None:
        """``provider`` and ``extractor`` are injectable; both default to offline.

        Defaulting to the fixture provider and the deterministic extractor is what makes
        the offline claim structural rather than a matter of remembering to configure it:
        constructing this agent with no arguments cannot reach the network.
        """
        self.config = config or PipelineConfig()
        #: What the ``auto`` switches settled on, and why. Recorded so a skipped gather can
        #: say "no searching backend available: $GEMINI_API_KEY is not set" rather than the
        #: unactionable "intel gathering is disabled".
        configured_intel = resolve_intel_config(self.config)
        backend = resolve_backend(self.config.llm)
        self.resolution = resolve_intel(configured_intel, backend)
        #: Kept so a per-call config override can be resolved the same way this one was.
        self.backend_kind = LLMBackendKind(backend.resolved)
        self.intel_config = concrete_intel(
            configured_intel, self.resolution, backend=self.backend_kind
        )
        self.sandbox = sandbox or Sandbox(self.config.sandbox)
        self.baseline = IntelBaselineBackend()
        self.provider = provider if provider is not None else self._default_provider()
        self.extractor = extractor
        self.cache = cache if cache is not None else IntelCache(
            self.intel_config.cache_dir, enabled=self.intel_config.use_cache
        )
        self._today = today
        self._now = now
        self.total_usage = IntelUsage()
        self.results: list[IntelResult] = []

    # -- construction helpers ---------------------------------------------

    def _default_provider(self) -> BaseSearchProvider:
        """The provider for the resolved configuration.

        ``self.intel_config`` is already concrete by the time this runs, so ``auto`` never
        reaches the routing below: an unresolved mode could only ever resolve *towards* a
        live provider by accident, which is the one direction that must not happen by
        accident.
        """
        return build_search_provider(self.intel_config)

    def _guarded(self) -> GuardedBackend:
        """The phase-2 backend: the extractor under the framework's own output guard.

        Offline, inner *is* the baseline, which makes ``GuardedBackend`` take its
        single-pass branch: the deterministic answer still goes through the canary,
        envelope, schema and evidence-span checks, so the offline path exercises the same
        guards the live path does rather than a weaker imitation of them.
        """
        inner: LLMBackend = self.extractor or self.baseline
        from vulnprio.core.config import LLMConfig  # noqa: PLC0415 - local, avoids a cycle

        llm_config = LLMConfig(
            model=self.intel_config.model,
            max_retries=self.intel_config.max_retries,
            timeout_s=self.intel_config.timeout_s,
            api_key_env=self.intel_config.api_key_env,
        )
        return GuardedBackend(inner=inner, heuristic=self.baseline, config=llm_config)

    def today(self) -> date:
        return self._today or datetime.now(timezone.utc).date()

    # -- main entry point --------------------------------------------------

    def gather(
        self,
        finding: Finding,
        intel: Sequence[VulnIntel] = (),
        as_of: date | None = None,
        config: IntelConfig | None = None,
    ) -> IntelResult:
        """Gather, extract, compare and summarise for one finding.

        Never raises. Every failure path returns a result whose ``errors`` say what went
        wrong and whose features are neutral.
        """
        # A per-call override has to go through the same resolution the constructor applied.
        # Without this, ``auto`` reaches ``_gather`` as the literal string, which is truthy,
        # so a caller asking for the default configuration would silently get a live run.
        intel_config = self.intel_config if config is None else concrete_intel(
            config, self.resolution, backend=self.backend_kind
        )
        as_of = as_of or self.today()
        result = self._gather(finding, tuple(intel), as_of, intel_config)
        self.total_usage = self.total_usage + result.usage
        self.results.append(result)
        return result

    def _empty(
        self,
        finding: Finding,
        intel: Sequence[VulnIntel],
        as_of: date,
        *,
        queries: Sequence[Any] = (),
        errors: Sequence[str] = (),
        skipped_reason: str = "",
        verdict: AsOfVerdict | None = None,
    ) -> IntelResult:
        return IntelResult(
            finding_id=finding.finding_id,
            cve_id=(finding.cve_ids[0] if finding.cve_ids else None),
            as_of=as_of,
            queries=tuple(queries),
            errors=tuple(errors),
            skipped_reason=skipped_reason,
            as_of_verdict=verdict or AsOfVerdict(),
            provider=getattr(self.provider, "name", ""),
            model=self.intel_config.model,
        )

    def _gather(
        self,
        finding: Finding,
        intel: tuple[VulnIntel, ...],
        as_of: date,
        intel_config: IntelConfig,
    ) -> IntelResult:
        # Settle ``auto`` here rather than trusting the caller to have done it. It is a
        # string sentinel, so any truthiness test that sees it unresolved reads "auto" as
        # "yes" and silently starts a live run -- the one failure mode of a tri-state flag.
        # This is the single point where ``enabled`` is read, so normalising here covers
        # every route in: the constructor, a per-call override, and a directly assigned
        # ``intel_config``.
        if not isinstance(intel_config.enabled, bool):
            intel_config = concrete_intel(
                intel_config, self.resolution, backend=self.backend_kind
            )

        if not intel_config.enabled:
            return self._empty(
                finding,
                intel,
                as_of,
                skipped_reason=self.resolution.reason or "intel gathering is disabled",
            )

        queries = build_queries(finding, intel, intel_config)
        if not queries:
            return self._empty(
                finding, intel, as_of, skipped_reason="no searchable identifiers"
            )

        cve_id = queries[0].cve_id or (finding.cve_ids[0] if finding.cve_ids else None)
        key = intel_cache_key(
            finding_id=finding.finding_id,
            cve_id=cve_id,
            queries=queries,
            as_of=as_of,
            model=intel_config.model,
            prompt_hash=PROMPT_FINGERPRINT,
            provider=getattr(self.provider, "name", ""),
            schema_name=ExploitIntelOut.__name__,
        )
        cached = self.cache.load(key)
        if cached is not None:
            return cached

        errors: list[str] = []
        live = intel_config.mode == "live"
        verdict = judge_as_of(as_of, self.today(), intel_config, live=live)
        if not verdict.search_allowed:
            # Not an error: it is an actionable state with a named remedy, and a caller
            # that renders it as a failure is doing its user a disservice.
            return self._empty(
                finding,
                intel,
                as_of,
                queries=queries,
                skipped_reason=verdict.message,
                verdict=verdict,
            )
        if verdict.status == AsOfStatus.OVERRIDDEN:
            errors.append(verdict.message)

        gathered = self._run_phase1(finding, intel, queries, intel_config)
        errors.extend(gathered.errors)

        documents = dedupe_documents(
            gathered.documents, known_urls(intel), intel_config.max_documents
        )
        if not documents and not gathered.narrative:
            result = self._empty(
                finding, intel, as_of, queries=queries, errors=errors, verdict=verdict
            )
            result = result.model_copy(update={"usage": gathered.usage})
            self.cache.store(key, result)
            return result

        extraction, audit, prompt, sanitized = self._run_phase2(
            finding, intel, documents, gathered, intel_config
        )
        if audit is not None and audit.canary_leaked:
            errors.append("canary leaked in the extraction reply; the answer was discarded")
        if audit is not None and audit.fell_back_to_heuristic and self.extractor is not None:
            errors.append("extraction fell back to the deterministic baseline")

        signals = len(audit.signals) if audit is not None else 0
        # The deterministic baseline provably never read a block the sandbox flagged, so
        # its claims are not gated by the signal count. A model's answer might have been
        # influenced by one, so it is.
        model_authored = (
            audit is not None
            and audit.backend != LLMBackendKind.HEURISTIC
            and not audit.fell_back_to_heuristic
        )
        fused, influence = self._fuse(
            extraction, intel, signals if model_authored else 0, intel_config
        )

        agreement = compare_with_feeds(fused, documents, intel, sanitized)
        poc_urls = tuple(
            dict.fromkeys(
                [
                    *(fused.public_exploit_urls if fused else ()),
                    *(
                        document.url
                        for document in documents
                        if document.source_kind == IntelSourceKind.POC
                    ),
                ]
            )
        )[:MAX_EXPLOIT_URLS]

        summary = None
        if intel_config.summarize:
            summary = summarize_intel(
                gathered,
                documents,
                config=intel_config,
                model=gathered.model or intel_config.model,
                prompt_hash=PROMPT_FINGERPRINT,
                recorded=(getattr(self.provider, "name", "") == "fixture"),
                now=self._now,
            )
            if summary is None and gathered.narrative:
                errors.append("summary refused: no citation pointed at a retrieved page")

        usage = gathered.usage
        if audit is not None:
            usage = usage + IntelUsage(
                input_tokens=audit.input_tokens,
                output_tokens=audit.output_tokens,
                calls=1 if audit.backend == LLMBackendKind.ANTHROPIC else 0,
            )

        result = IntelResult(
            finding_id=finding.finding_id,
            cve_id=cve_id,
            as_of=as_of,
            queries=queries,
            documents=documents,
            extraction=fused,
            summary=summary,
            agreement=agreement,
            as_of_verdict=verdict,
            public_exploit_urls=poc_urls,
            active_exploitation_claimed=bool(fused.active_exploitation_claimed) if fused else False,
            usage=usage,
            errors=tuple(errors),
            provider=gathered.provider or getattr(self.provider, "name", ""),
            model=gathered.model or intel_config.model,
            injection_signals=signals,
            max_tier_used=prompt.max_tier_used if prompt is not None else TrustTier.OPERATOR,
            influence_used=influence,
            canary_leaked=bool(audit.canary_leaked) if audit is not None else False,
            fell_back_to_baseline=bool(audit.fell_back_to_heuristic) if audit is not None else False,
        )
        self.cache.store(key, result)
        return result

    # -- phase 1 -----------------------------------------------------------

    def _run_phase1(
        self,
        finding: Finding,
        intel: Sequence[VulnIntel],
        queries: Sequence[Any],
        intel_config: IntelConfig,
    ) -> IntelGather:
        instruction = phase1_instruction(finding, intel, queries, known_urls(intel))
        try:
            return self.provider.gather(queries, intel_config, instruction=instruction)
        except Exception as exc:  # noqa: BLE001 - a provider must not abort a scan
            return IntelGather(
                errors=(f"search provider failed: {type(exc).__name__}: {exc}",),
                provider=getattr(self.provider, "name", ""),
            )

    # -- phase 2 -----------------------------------------------------------

    def _untrusted_segments(
        self, documents: Sequence[IntelDocument], gathered: IntelGather
    ) -> tuple[UntrustedText, ...]:
        """Retrieved pages, then the researcher prose, all at tier REFERENCE_PAGE.

        The narrative is model output rather than a fetched page, but it is a restatement
        of fetched pages and must not be trusted above the pages it restates. Giving it a
        gentler tier would create exactly the laundering path the sandbox exists to close.
        """
        segments = [document.snippet for document in documents]
        if gathered.narrative:
            segments.append(
                UntrustedText(
                    text=gathered.narrative,
                    provenance=Provenance.REFERENCE_PAGE,
                    source_url=None,
                )
            )
        return tuple(segments)

    def _run_phase2(
        self,
        finding: Finding,
        intel: Sequence[VulnIntel],
        documents: Sequence[IntelDocument],
        gathered: IntelGather,
        intel_config: IntelConfig,
    ) -> tuple[ExploitIntelOut | None, LLMAudit | None, SandboxedPrompt | None, str]:
        """Sanitise, prompt, extract. Returns (output, audit, prompt, sanitized text)."""
        segments = self._untrusted_segments(documents, gathered)
        context = phase2_context(finding, intel, documents, bool(gathered.narrative))
        seed = config_hash({"f": finding.finding_id, "u": [s.sha256 for s in segments]})
        prompt = build_sandboxed_prompt(
            task=INTEL_TASK,
            system=PHASE2_SYSTEM,
            operator_context=context,
            untrusted=segments,
            config=self.config.sandbox,
            rng=Random(int(seed[:8], 16)),
            schema_name=ExploitIntelOut.__name__,
            sandbox=self.sandbox,
        )
        sanitized = sanitized_text_of(prompt)
        try:
            result = self._guarded().complete_structured(prompt, ExploitIntelOut)
        except Exception as exc:  # noqa: BLE001 - never abort a scan
            # The deterministic answer always exists, so a failing extractor degrades to it
            # rather than to nothing. Returning None here would make a model outage look
            # identical to "the internet said nothing", which are not the same state.
            return self._baseline_only(prompt, sanitized, reason=type(exc).__name__)

        parsed = getattr(result, "parsed", None)
        audit = getattr(result, "audit", None)
        if not isinstance(parsed, ExploitIntelOut):
            return self._baseline_only(prompt, sanitized, reason="unusable_reply")
        if audit is not None and audit.canary_leaked:
            # The model's reply is discarded outright: a leaked canary means the retrieved
            # content reached the instruction channel and nothing that reply says can be
            # trusted. The deterministic reading of the same pages still stands, and the
            # leak is recorded on the result for the manipulation detector.
            baseline, baseline_audit, _prompt, _text = self._baseline_only(
                prompt, sanitized, reason="canary_leak"
            )
            return (
                baseline,
                baseline_audit.model_copy(update={"canary_leaked": True}),
                prompt,
                sanitized,
            )
        return parsed, audit, prompt, sanitized

    def _baseline_only(
        self, prompt: SandboxedPrompt, sanitized: str, reason: str
    ) -> tuple[ExploitIntelOut | None, LLMAudit | None, SandboxedPrompt, str]:
        """The deterministic extraction, with an audit saying why the model was not used."""
        try:
            fallback = self.baseline.complete_structured(prompt, ExploitIntelOut)
        except Exception:  # noqa: BLE001 - the last line of defence returns nothing
            return None, heuristic_audit(INTEL_TASK, reason=reason), prompt, sanitized
        audit = fallback.audit.model_copy(
            update={
                "task": f"{INTEL_TASK}:{reason}",
                "fell_back_to_heuristic": True,
                "prompt_hash": prompt.prompt_hash,
            }
        )
        parsed = fallback.parsed if isinstance(fallback.parsed, ExploitIntelOut) else None
        return parsed, audit, prompt, sanitized

    # -- fusion ------------------------------------------------------------

    def _fuse(
        self,
        extraction: ExploitIntelOut | None,
        intel: Sequence[VulnIntel],
        signals: int,
        intel_config: IntelConfig,
    ) -> tuple[ExploitIntelOut | None, dict[str, float]]:
        """Cap the extraction at the reference-page influence budget.

        Feasibility is measured against a neutral 0.5 rather than against the deterministic
        baseline, because this whole layer *is* reference-page evidence: the number it
        contributes must sit within one budget of no-evidence, exactly as a reference page
        does elsewhere in the framework. Maturity is capped by
        :func:`~vulnprio.semantic.exploitability.fuse_maturity` against the curated-feed
        floor, so a page can raise it by at most one step and can never lower it -- and
        cannot raise it at all when the sandbox flagged an injection attempt.
        """
        if extraction is None:
            return None, {}

        budget = float(
            self.config.sandbox.influence_budget.get(TrustTier.REFERENCE_PAGE, 0.35)
        )
        feasibility, delta = apply_budget(0.5, float(extraction.exploit_feasibility), budget)

        floor = structured_maturity(intel)
        maturity = fuse_maturity(floor, extraction.exploit_maturity, injection_signals=signals)

        fused = extraction.model_copy(
            update={"exploit_feasibility": feasibility, "exploit_maturity": maturity}
        )
        # Key order is sorted so the JSON a result serialises to is stable. The cache
        # writes with sorted keys, and a warm result that differed from its cold original
        # only by dict ordering would break the byte-identity guarantee for no reason.
        influence = dict(
            sorted(
                {
                    "exploit_feasibility": round(abs(delta), 6),
                    "budget": round(budget, 6),
                }.items()
            )
        )
        return fused, influence


def fuse_into_exploitability(
    assessment: Any,
    result: IntelResult | None,
    config: PipelineConfig | None = None,
) -> Any:
    """Fold retrieved intelligence into Component A's exploitability assessment.

    This lives here rather than in :mod:`vulnprio.semantic.exploitability` so that the
    trust reasoning stays in one package and the consumer's change is three lines. It
    deliberately re-applies the reference-page influence budget rather than trusting the
    agent to have done it: this is a second entry point into a score, and a second entry
    point that skipped the budget would be exactly the unbudgeted path ADR-002 forbids.

    The budget is asymmetric, following ``SandboxConfig.deflation_budget_factor``. Talking
    a real vulnerability *down* leaves it unpatched; talking a harmless one up only wastes
    remediation effort. The two are not equally dangerous and are not given equal room.

    ``privileges_required``, ``privilege_gained`` and the CIA impacts are untouched: they
    are attack-graph structure and curated-feed facts, and internet prose does not get to
    move either.
    """
    if result is None or result.extraction is None:
        return assessment

    config = config or PipelineConfig()
    sandbox_config = config.sandbox
    budget = float(sandbox_config.influence_budget.get(TrustTier.REFERENCE_PAGE, 0.35))
    extraction = result.extraction

    proposed = clamp01(float(extraction.exploit_feasibility))
    if proposed < assessment.exploit_feasibility:
        budget *= float(sandbox_config.deflation_budget_factor)
    feasibility, _delta = apply_budget(
        float(assessment.exploit_feasibility), proposed, budget
    )

    maturity = fuse_maturity(
        assessment.exploit_maturity,
        extraction.exploit_maturity,
        injection_signals=result.injection_signals,
    )

    complexity = assessment.attack_complexity
    if complexity == AttackComplexity.UNKNOWN:
        complexity = extraction.attack_complexity

    preconditions = list(dict.fromkeys([*assessment.preconditions, *extraction.preconditions]))[:8]
    spans = tuple(
        dict.fromkeys([*assessment.evidence_spans, *extraction.evidence_spans])
    )[:12]
    note = (
        f" | intel: {len(result.documents)} page(s), "
        f"corroborates={result.agreement.corroborates} "
        f"contradicts={result.agreement.contradicts} "
        f"signals={result.injection_signals}"
    )
    return assessment.model_copy(
        update={
            "exploit_feasibility": feasibility,
            "exploit_maturity": maturity,
            "attack_complexity": complexity,
            "preconditions": tuple(preconditions),
            "evidence_spans": spans,
            "rationale": (assessment.rationale + note)[:600],
        }
    )


def resolve_intel_config(config: PipelineConfig | IntelConfig | None) -> IntelConfig:
    """The intel section of a pipeline config, or defaults when it has none yet.

    ``core/config.py`` is frozen; until an ``intel`` field lands there this keeps the
    package usable and keeps the default -- disabled, offline -- in one place.
    """
    if config is None:
        return IntelConfig()
    if isinstance(config, IntelConfig):
        return config
    section = getattr(config, "intel", None)
    if isinstance(section, IntelConfig):
        return section
    if isinstance(section, dict):
        return IntelConfig.model_validate(section)
    return IntelConfig()
