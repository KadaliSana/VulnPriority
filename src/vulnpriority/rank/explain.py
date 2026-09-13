"""Why this finding is where it is (DESIGN.md 3.8, Goal 5).

Two things are being produced here and they serve different readers, and - the point of
the split - they have different dependencies.

**Attributions** are for the analyst who wants the model's arithmetic: exact SHAP values
over the fitted booster, tagged with the component that owns each feature and the trust
tier that feature's evidence came from. Because SHAP is additive, the tagging supports
the one number this module exists to compute -
``untrusted_influence_share = sum |shap| over Component A features / sum |shap| over all
features`` - which is how much of a finding's position was argued by content the target
application or the internet authored. ``RankManipulationDetector`` reads it, and
``SandboxConfig.max_untrusted_shap_share`` is the line it must not cross.

**Reason codes** are for everyone else, and they are deliberately not generated text.
Every code is a template in :data:`REASON_TEMPLATES` filled from structured fields -
dates, enum members, numbers - so an explanation cannot be the attack. Model prose
(``rationale``, ``evidence_spans``) and untrusted text (``UntrustedText.text``) are never
read by this module at all; the one place a target-influenced string can reach a code is
an observed software version, and that is passed through :func:`safe_token`, which strips
it to a version-shaped character set and truncates it.

Crucially, **reason codes need no model**. "KEV-listed since 2024-02-01" and "chain
chokepoint: removes $277,755 of reachable risk" are statements about the evidence, not
about the ranker, so :func:`evidence_reason_codes` and :class:`EvidenceExplainer` produce
them from an :class:`EnrichedFinding` and a :class:`ChainScore` alone. They were once
reachable only through :class:`ShapExplainer`, which needs a fitted booster, and that was a
real defect: the interactive path - one scan, no labelled history, which is how most people
will actually use this - takes the ranker's degenerate-input fallback and so explained
nothing, while a research run over two dozen scans explained everything. Use
:func:`build_explainer` to get the best explainer a given ranker can support;
``rank_scan`` falls back to the evidence-only one on its own, so a queue never comes back
with nothing to say.

The whole module is deterministic: the same booster and the same frame give byte-identical
explanations, because the ablation and the adversarial evaluation both diff explanations
across runs and any nondeterminism would show up as an attack.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import shap

from vulnpriority.core.config import RankingConfig
from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    Component,
    ExploitMaturity,
    PrivilegeLevel,
    TrustTier,
    VersionMatch,
)
from vulnpriority.core.errors import RankerNotFittedError
from vulnpriority.core.money import DEFAULT_CURRENCY, format_money
from vulnpriority.core.models import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    ChainScore,
    EnrichedFinding,
    Explanation,
    FeatureContribution,
    FeatureFrame,
)
from vulnpriority.rank.features import usable_intel

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from vulnpriority.rank.lambdamart import LambdaMartRanker

__all__ = [
    "FEATURE_TIER",
    "REASON_TEMPLATES",
    "DEFAULT_MAX_REASON_CODES",
    "safe_token",
    "feature_tier",
    "evidence_reason_codes",
    "EvidenceExplainer",
    "ShapExplainer",
    "build_explainer",
]

#: Trust tier of the evidence behind each feature.
#:
#: Component A features default to ``TARGET_CONTENT`` - the least trusted tier - because
#: without the finding in hand there is no way to know how far down the semantic
#: assessment reached; when the :class:`EnrichedFinding` *is* available, the per-finding
#: ``TrustSummary.max_tier_used`` replaces the default and is usually more favourable.
#: Component C features are ``SCANNER`` because the graph builder refuses edges from any
#: tier above it. The Component B features split: feed facts are ``CURATED_FEED``, while
#: the attacker, impact and cost models are operator configuration.
FEATURE_TIER: dict[str, TrustTier] = {
    "cvss_base_max": TrustTier.CURATED_FEED,
    "cvss_version_ord": TrustTier.CURATED_FEED,
    "cvss_source_agreement": TrustTier.CURATED_FEED,
    "cvss_ac_low": TrustTier.CURATED_FEED,
    "cvss_pr_none": TrustTier.CURATED_FEED,
    "cvss_ui_none": TrustTier.CURATED_FEED,
    "cvss_c_high": TrustTier.CURATED_FEED,
    "cvss_i_high": TrustTier.CURATED_FEED,
    "cvss_a_high": TrustTier.CURATED_FEED,
    "scanner_severity_ord": TrustTier.SCANNER,
    "scanner_confidence": TrustTier.SCANNER,
    "cwe_owasp_top10": TrustTier.CURATED_FEED,
    "vuln_age_days": TrustTier.CURATED_FEED,
    "cluster_size": TrustTier.SCANNER,
    "auth_required_ord": TrustTier.SCANNER,
    "method_state_changing": TrustTier.SCANNER,
    "param_count": TrustTier.SCANNER,
    "a_asset_criticality": TrustTier.TARGET_CONTENT,
    "a_data_sensitivity": TrustTier.TARGET_CONTENT,
    "a_exposure": TrustTier.TARGET_CONTENT,
    "a_function_ord": TrustTier.TARGET_CONTENT,
    "a_is_admin_surface": TrustTier.TARGET_CONTENT,
    "a_is_auth_boundary": TrustTier.TARGET_CONTENT,
    "a_exploit_feasibility": TrustTier.TARGET_CONTENT,
    "a_exploit_maturity_ord": TrustTier.TARGET_CONTENT,
    "a_attack_complexity_high": TrustTier.TARGET_CONTENT,
    "a_privileges_required_ord": TrustTier.TARGET_CONTENT,
    "a_user_interaction_required": TrustTier.TARGET_CONTENT,
    "a_impact_cia_mean": TrustTier.TARGET_CONTENT,
    "a_privilege_gained_ord": TrustTier.TARGET_CONTENT,
    "a_p_applicable": TrustTier.TARGET_CONTENT,
    "a_version_match_ord": TrustTier.TARGET_CONTENT,
    "a_confidence": TrustTier.TARGET_CONTENT,
    "a_injection_signals": TrustTier.SCANNER,
    # Retrieved intelligence is REFERENCE_PAGE by construction - ``IntelDocument`` refuses
    # a snippet with any other provenance. The signal count is the sandbox's own
    # measurement of those pages rather than a claim from them, so it is SCANNER, matching
    # ``a_injection_signals``.
    "a_intel_documents": TrustTier.REFERENCE_PAGE,
    "a_intel_public_exploit_urls": TrustTier.REFERENCE_PAGE,
    "a_intel_active_exploitation": TrustTier.REFERENCE_PAGE,
    "a_intel_confidence": TrustTier.REFERENCE_PAGE,
    "a_intel_corroborates_feeds": TrustTier.REFERENCE_PAGE,
    "a_intel_contradicts_feeds": TrustTier.REFERENCE_PAGE,
    "a_intel_injection_signals": TrustTier.SCANNER,
    "b_epss": TrustTier.CURATED_FEED,
    "b_epss_percentile": TrustTier.CURATED_FEED,
    "b_kev": TrustTier.CURATED_FEED,
    "b_kev_ransomware": TrustTier.CURATED_FEED,
    "b_kev_age_days": TrustTier.CURATED_FEED,
    "b_exploit_count": TrustTier.CURATED_FEED,
    "b_exploit_maturity_feed_ord": TrustTier.CURATED_FEED,
    "b_exploit_verified": TrustTier.CURATED_FEED,
    "b_p_exploit_attacker": TrustTier.OPERATOR,
    "b_impact_log": TrustTier.OPERATOR,
    "b_expected_loss_log": TrustTier.OPERATOR,
    "b_remediation_hours": TrustTier.OPERATOR,
    "c_reach_delta_log": TrustTier.SCANNER,
    "c_max_path_prob": TrustTier.SCANNER,
    "c_n_paths_through": TrustTier.SCANNER,
    "c_betweenness": TrustTier.SCANNER,
    "c_hops_from_entry": TrustTier.SCANNER,
    "c_privilege_gain": TrustTier.SCANNER,
    "c_is_chokepoint": TrustTier.SCANNER,
}

# The frozen contract owns the feature list, and a feature missing from the table above
# would silently be attributed to the most untrusted tier. Caught at import, like the
# neutral and documentation tables in ``features``.
_UNTIERED = [name for name in FEATURE_NAMES if name not in FEATURE_TIER]
if _UNTIERED:  # pragma: no cover - import-time contract check
    raise ImportError(f"vulnpriority.rank.explain has no trust tier for {_UNTIERED}")

#: Every sentence this module can emit. Listed together so the set of things an
#: explanation may say is reviewable in one place and provably finite.
REASON_TEMPLATES: dict[str, str] = {
    "kev_dated": "KEV-listed since {date}",
    "kev_undated": "KEV-listed (catalogue entry carries no date)",
    "kev_ransomware": "KEV entry records ransomware campaign use",
    "epss": "EPSS {score:.3f}, {percentile:.0%} of all CVEs",
    "exploit_verified": "verified {maturity} exploit code in a curated index",
    "exploit_unverified": "{count} public exploit record(s), highest maturity {maturity}",
    "chokepoint": "chain chokepoint: removes {reach} of reachable risk",
    "chain_contribution": "chain contribution: unlocks {reach} of reachable risk",
    "privilege_gain": "grants {gained} on success, {hops} hop(s) from the attacker's entry point",
    "version_mismatch": "version mismatch: not applicable to observed {version}",
    "version_match": "version match confirmed against observed {version}",
    "not_applicable": "assessed not applicable to this deployment",
    "expected_loss": "expected loss {loss} = P(exploit) {p:.2f} x impact {impact}",
    "exposure": "internet-facing and reachable without credentials",
    "admin_surface": "administrative surface",
    "cluster": "same root cause on {count} endpoints in this scan",
    "cvss": "CVSS {score:.1f} ({version})",
    "scanner": "scanner reported {severity} at {confidence:.0%} confidence",
    "untrusted_influence": "untrusted content drove {share:.0%} of this score",
    "injection_signals": "{count} injection signal(s) raised while assessing this finding",
    "canary": "canary token leaked during assessment: treat this score as compromised",
    "feature_up": "{feature} = {value:.3g} raised the score",
    "feature_down": "{feature} = {value:.3g} lowered the score",
    # Said out loud rather than left as an empty panel. The interactive path - one scan,
    # no labelled history - cannot train a ranker, so there is no attribution to show; a
    # reader deserves to know that is a known limit and not a broken screen.
    "no_factor_breakdown": (
        "Ranked on the evidence above; the per-factor breakdown needs a model trained on "
        "labelled scan history, which this run did not have."
    ),
}

#: How many reason codes an explanation carries before it stops being read.
DEFAULT_MAX_REASON_CODES: int = 6

#: Characters allowed to survive from a target-influenced version string.
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._+\-]+")

#: Longest version-shaped token a reason code will quote.
_SAFE_TOKEN_CHARS: int = 40


def safe_token(value: str | None) -> str:
    """Reduce a target-influenced identifier to a version-shaped token.

    Observed software versions are the only externally-authored strings a reason code may
    contain. Anything outside ``[A-Za-z0-9._+-]`` is collapsed to a single ``-`` and the
    result is truncated, so a response header cannot smuggle punctuation, markup or an
    instruction into an explanation that a human or another model later reads.
    """
    if not value:
        return "unknown"
    cleaned = _SAFE_TOKEN.sub("-", str(value)).strip("-")
    return (cleaned[:_SAFE_TOKEN_CHARS] or "unknown")


def feature_tier(feature: str, enriched: EnrichedFinding | None = None) -> TrustTier:
    """Trust tier for a feature, specialised to a finding when one is available."""
    tier = FEATURE_TIER.get(feature, TrustTier.TARGET_CONTENT)
    if enriched is not None and FEATURE_GROUPS.get(feature) is Component.A:
        return enriched.trust.max_tier_used
    return tier


class ShapExplainer:
    """SHAP attributions plus templated reason codes for a fitted LambdaMART ranker."""

    def __init__(
        self,
        ranker: "LambdaMartRanker",
        *,
        top_n: int | None = None,
        max_reason_codes: int = DEFAULT_MAX_REASON_CODES,
        config: RankingConfig | None = None,
        currency: str = DEFAULT_CURRENCY,
    ) -> None:
        """``ranker`` must be fitted with a booster; a fallback ranker has nothing to explain."""
        booster = ranker.booster
        if booster is None:
            raise RankerNotFittedError(
                "ShapExplainer needs a fitted booster; the ranker is on the fallback path"
            )
        settings = config if config is not None else ranker.config
        self.ranker = ranker
        self.config: RankingConfig = settings
        self.top_n: int = int(top_n if top_n is not None else settings.explain_top_n)
        self.max_reason_codes: int = int(max_reason_codes)
        #: What the money figures in the reason codes are denominated in.
        self.currency: str = currency
        self._explainer = shap.TreeExplainer(booster)

    # -- attributions -------------------------------------------------------

    @property
    def base_value(self) -> float:
        """The booster's expected output: the score before any feature moves it."""
        value = self._explainer.expected_value
        return float(np.asarray(value).reshape(-1)[0])

    def shap_values(self, frame: FeatureFrame) -> np.ndarray:
        """SHAP value matrix, shaped ``(rows, features)`` and aligned with the frame."""
        values = np.asarray(self._explainer.shap_values(frame.X), dtype=float)
        if values.ndim == 3:  # pragma: no cover - multi-output boosters are not used here
            values = values[..., 0]
        return values

    def untrusted_share(self, shap_row: np.ndarray, columns: Sequence[str]) -> float:
        """Share of absolute attribution carried by Component A features.

        Zero when nothing moved the score: a row whose every attribution is zero is not
        "entirely untrusted", it is unexplained, and reporting 0 keeps the guard from
        firing on an empty explanation.
        """
        magnitudes = np.abs(np.asarray(shap_row, dtype=float))
        total = float(magnitudes.sum())
        if total <= 0.0:
            return 0.0
        untrusted = float(
            sum(
                magnitude
                for magnitude, name in zip(magnitudes, columns)
                if FEATURE_GROUPS.get(name) is Component.A
            )
        )
        return float(min(1.0, max(0.0, untrusted / total)))

    # -- explanations -------------------------------------------------------

    def explain(
        self,
        frame: FeatureFrame,
        enriched: Sequence[EnrichedFinding] | Mapping[str, EnrichedFinding] | None = None,
        chain: Mapping[str, ChainScore] | None = None,
        top_n: int | None = None,
    ) -> list[Explanation]:
        """One :class:`Explanation` per row, in frame order.

        ``enriched`` and ``chain`` are optional: without them the reason codes fall back
        to the feature-level templates, which is all the information a raw frame carries.
        With them, the evidence-level codes ("KEV-listed since ...") are produced first.
        """
        lookup = _as_lookup(enriched)
        chain_map: Mapping[str, ChainScore] = chain or {}
        columns = list(frame.feature_names)
        values = self.shap_values(frame)
        matrix = frame.X.to_numpy(dtype=float)
        limit = int(top_n if top_n is not None else self.top_n)
        base = self.base_value

        explanations: list[Explanation] = []
        for index, finding_id in enumerate(frame.finding_ids):
            item = lookup.get(finding_id)
            share = self.untrusted_share(values[index], columns)
            contributions = self._top_contributions(
                values[index], matrix[index], columns, item, limit
            )
            explanations.append(
                Explanation(
                    finding_id=finding_id,
                    base_value=base,
                    top_contributions=contributions,
                    reason_codes=self._reason_codes(
                        item, chain_map.get(finding_id), contributions, share
                    ),
                    untrusted_influence_share=share,
                )
            )
        return explanations

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _top_contributions(
        shap_row: np.ndarray,
        value_row: np.ndarray,
        columns: Sequence[str],
        enriched: EnrichedFinding | None,
        limit: int,
    ) -> tuple[FeatureContribution, ...]:
        """The ``limit`` largest attributions by magnitude, ties broken by column order."""
        order = sorted(
            range(len(columns)),
            key=lambda index: (-abs(float(shap_row[index])), index),
        )
        return tuple(
            FeatureContribution(
                feature=columns[index],
                value=float(value_row[index]),
                shap_value=float(shap_row[index]),
                group=FEATURE_GROUPS.get(columns[index]),
                tier=feature_tier(columns[index], enriched),
            )
            for index in order[: max(0, limit)]
        )
    def _reason_codes(
        self,
        enriched: EnrichedFinding | None,
        chain: ChainScore | None,
        contributions: Sequence[FeatureContribution],
        share: float,
    ) -> tuple[str, ...]:
        """Evidence codes, then the features that actually moved the score."""
        codes = evidence_reason_codes(enriched, chain, share, currency=self.currency)
        codes.extend(_feature_codes(contributions))
        return _dedupe(codes, self.max_reason_codes)


class EvidenceExplainer:
    """Reason codes with no model behind them (DESIGN.md 3.8, Goal 5).

    Everything an operator most wants to read - "KEV-listed since 2024-02-01", "chain
    chokepoint: removes $277,755 of reachable risk", "version mismatch: not applicable to
    observed 2.5.12" - is a statement about the *evidence*, not about the ranker. None of
    it needs a fitted booster, and tying it to one was a real defect: the interactive path,
    where somebody scans their own application once and looks at the queue, has no training
    history by definition, so the most common real run explained nothing while a research
    run over two dozen scans explained everything. Exactly backwards.

    This explainer therefore produces the same reason codes :class:`ShapExplainer`
    produces, minus the per-feature attributions, for any ranker at all - a baseline, a
    LambdaMART on its degenerate-input fallback, or no model whatsoever.

    Two fields are deliberately left empty rather than approximated. ``top_contributions``
    is empty because without SHAP there are no attributions, and ``base_value`` with it.
    ``untrusted_influence_share`` stays at 0.0 because it is *defined* as a share of
    absolute SHAP mass; substituting a different quantity on the same field would silently
    re-scale the threshold ``RankManipulationDetector`` compares it against and manufacture
    alerts, which is the failure mode this framework keeps having to design out. Instead
    the missing breakdown is stated outright, in one line, so an empty factor panel reads
    as a known limit rather than as a bug.
    """

    def __init__(
        self,
        *,
        max_reason_codes: int = DEFAULT_MAX_REASON_CODES,
        notice: str = REASON_TEMPLATES["no_factor_breakdown"],
        currency: str = DEFAULT_CURRENCY,
    ) -> None:
        """``notice`` is the single honest line explaining why there is no breakdown."""
        self.max_reason_codes: int = int(max_reason_codes)
        self.notice: str = notice
        #: What the money figures in the reason codes are denominated in.
        self.currency: str = currency

    def explain(
        self,
        frame: FeatureFrame,
        enriched: Sequence[EnrichedFinding] | Mapping[str, EnrichedFinding] | None = None,
        chain: Mapping[str, ChainScore] | None = None,
        top_n: int | None = None,
    ) -> list[Explanation]:
        """One :class:`Explanation` per row, in frame order, carrying reason codes only.

        Signature-compatible with :meth:`ShapExplainer.explain` so ``rank_scan`` and the
        pipeline can hold either without caring which. ``top_n`` is accepted and ignored:
        there are no contributions to take a top of.
        """
        lookup = _as_lookup(enriched)
        chain_map: Mapping[str, ChainScore] = chain or {}

        explanations: list[Explanation] = []
        for finding_id in frame.finding_ids:
            item = lookup.get(finding_id)
            if item is None:
                continue
            codes = evidence_reason_codes(
                item, chain_map.get(finding_id), 0.0, currency=self.currency
            )
            explanations.append(
                Explanation(
                    finding_id=finding_id,
                    base_value=0.0,
                    top_contributions=(),
                    reason_codes=self._with_notice(codes),
                    untrusted_influence_share=0.0,
                )
            )
        return explanations

    def _with_notice(self, codes: list[str]) -> tuple[str, ...]:
        """Trim to leave room for the notice, so it is never the code that gets cut."""
        if not self.notice:
            return _dedupe(codes, self.max_reason_codes)
        kept = _dedupe(codes, max(0, self.max_reason_codes - 1))
        return _dedupe([*kept, self.notice], self.max_reason_codes)


def build_explainer(
    ranker: object,
    *,
    top_n: int | None = None,
    config: RankingConfig | None = None,
    max_reason_codes: int = DEFAULT_MAX_REASON_CODES,
    currency: str = DEFAULT_CURRENCY,
) -> "ShapExplainer | EvidenceExplainer":
    """The best explainer this ranker can support, never ``None``.

    A fitted LambdaMART gets the full SHAP treatment; anything else - a baseline, or a
    LambdaMART that fell back to expected-loss ordering because there was one query group
    and no labelled history - gets evidence-only reason codes rather than nothing.
    """
    booster = getattr(ranker, "booster", None)
    if booster is None:
        return EvidenceExplainer(max_reason_codes=max_reason_codes, currency=currency)
    try:
        return ShapExplainer(
            ranker,  # type: ignore[arg-type]
            top_n=top_n,
            max_reason_codes=max_reason_codes,
            config=config,
            currency=currency,
        )
    except RankerNotFittedError:  # pragma: no cover - booster present but unusable
        return EvidenceExplainer(max_reason_codes=max_reason_codes, currency=currency)


# ---------------------------------------------------------------------------
# Reason codes. Model-free by construction: every one of these reads structured
# evidence, and none of them touches a booster, a rationale or any untrusted text.
# ---------------------------------------------------------------------------


def evidence_reason_codes(
    enriched: EnrichedFinding | None,
    chain: ChainScore | None = None,
    untrusted_share: float = 0.0,
    *,
    currency: str = DEFAULT_CURRENCY,
) -> list[str]:
    """Every reason code derivable from the evidence alone, in reading order.

    A canary leak leads unconditionally. It is not one consideration among several: it says
    the assessment behind every other code on the list may be fabricated.

    Uncapped and not de-duplicated - the caller decides how many it can show, and whether
    it has model-derived codes to append.
    """
    codes: list[str] = []
    if enriched is not None:
        if enriched.trust.canary_leaked:
            codes.append(REASON_TEMPLATES["canary"])
        codes.extend(_evidence_codes(enriched, chain, currency))
    codes.extend(_risk_codes(enriched, untrusted_share))
    return codes


def _dedupe(codes: Sequence[str], limit: int) -> tuple[str, ...]:
    """First occurrence wins, capped at ``limit``, order preserved."""
    unique: list[str] = []
    for code in codes:
        if code and code not in unique:
            unique.append(code)
    return tuple(unique[: max(0, limit)])


def _as_lookup(
    enriched: Sequence[EnrichedFinding] | Mapping[str, EnrichedFinding] | None,
) -> dict[str, EnrichedFinding]:
    """Accept either a sequence or an id-keyed mapping of enriched findings."""
    if enriched is None:
        return {}
    if isinstance(enriched, Mapping):
        return dict(enriched)
    return {item.finding_id: item for item in enriched}


def _evidence_codes(
    enriched: EnrichedFinding,
    chain: ChainScore | None,
    currency: str = DEFAULT_CURRENCY,
) -> list[str]:
    """Codes derived from structured evidence. No model prose is read here."""
    codes: list[str] = []
    records = usable_intel(enriched)

    for record in records:
        kev = record.kev
        if kev is None or not kev.in_kev:
            continue
        if kev.date_added is not None and kev.date_added <= enriched.as_of:
            codes.append(REASON_TEMPLATES["kev_dated"].format(date=kev.date_added.isoformat()))
        elif kev.date_added is None:
            codes.append(REASON_TEMPLATES["kev_undated"])
        if kev.known_ransomware_use:
            codes.append(REASON_TEMPLATES["kev_ransomware"])
        break

    epss = [record.epss for record in records if record.epss is not None]
    if epss:
        best = max(epss, key=lambda item: item.score)
        codes.append(REASON_TEMPLATES["epss"].format(score=best.score, percentile=best.percentile))

    codes.extend(_exploit_codes(records, enriched))
    codes.extend(_chain_codes(enriched, chain, currency))
    codes.extend(_applicability_codes(enriched))

    if enriched.expected_loss > 0.0:
        codes.append(
            REASON_TEMPLATES["expected_loss"].format(
                loss=format_money(enriched.expected_loss, currency),
                p=enriched.likelihood.p_exploit,
                impact=format_money(enriched.impact.total, currency),
            )
        )
    if enriched.asset.exposure >= 1.0:
        codes.append(REASON_TEMPLATES["exposure"])
    if enriched.asset.is_admin_surface:
        codes.append(REASON_TEMPLATES["admin_surface"])
    if enriched.finding.cluster_size > 1:
        codes.append(REASON_TEMPLATES["cluster"].format(count=int(enriched.finding.cluster_size)))
    return codes


def _exploit_codes(records: Sequence[Any], enriched: EnrichedFinding) -> list[str]:
    """Curated exploit evidence, verified entries reported ahead of unverified ones."""
    exploits = [
        exploit
        for record in records
        for exploit in record.exploits
        if exploit.published is None or exploit.published <= enriched.as_of
    ]
    if not exploits:
        return []
    best = max(exploits, key=lambda item: (item.verified, int(item.maturity)))
    maturity = ExploitMaturity(best.maturity).name.lower()
    if best.verified:
        return [REASON_TEMPLATES["exploit_verified"].format(maturity=maturity)]
    return [REASON_TEMPLATES["exploit_unverified"].format(count=len(exploits), maturity=maturity)]


def _chain_codes(
    enriched: EnrichedFinding,
    chain: ChainScore | None,
    currency: str = DEFAULT_CURRENCY,
) -> list[str]:
    """Attack-graph position, when Component C produced a score for this finding."""
    if chain is None:
        return []
    codes: list[str] = []
    if chain.is_chokepoint:
        codes.append(
            REASON_TEMPLATES["chokepoint"].format(
                reach=format_money(chain.reach_delta, currency)
            )
        )
    elif chain.reach_delta > 0.0:
        codes.append(
            REASON_TEMPLATES["chain_contribution"].format(
                reach=format_money(chain.reach_delta, currency)
            )
        )
    if chain.privilege_gain > 0:
        gained = PrivilegeLevel(enriched.exploitability.privilege_gained).name.lower()
        codes.append(
            REASON_TEMPLATES["privilege_gain"].format(
                gained=gained, hops=int(chain.hops_from_entry)
            )
        )
    return codes


def _applicability_codes(enriched: EnrichedFinding) -> list[str]:
    """Version evidence and the applicability verdict, as structured facts only."""
    codes: list[str] = []
    applicability = enriched.applicability
    observed = enriched.finding.affected_component
    version = observed.version if observed is not None else None
    if version is None:
        for component in enriched.endpoint.observed_tech:
            if component.version:
                version = component.version
                break
    token = safe_token(version)

    if applicability.version_match == VersionMatch.MISMATCH:
        codes.append(REASON_TEMPLATES["version_mismatch"].format(version=token))
    elif applicability.version_match == VersionMatch.MATCH and version is not None:
        codes.append(REASON_TEMPLATES["version_match"].format(version=token))
    if applicability.verdict == ApplicabilityVerdict.NOT_APPLICABLE:
        codes.append(REASON_TEMPLATES["not_applicable"])
    return codes


def _risk_codes(enriched: EnrichedFinding | None, share: float) -> list[str]:
    """Codes about the explanation itself: how much of it rests on untrusted content."""
    codes: list[str] = []
    if share > 0.0:
        codes.append(REASON_TEMPLATES["untrusted_influence"].format(share=share))
    if enriched is None:
        return codes
    if enriched.trust.canary_leaked:
        codes.append(REASON_TEMPLATES["canary"])
    if enriched.trust.injection_signal_count > 0:
        codes.append(
            REASON_TEMPLATES["injection_signals"].format(
                count=int(enriched.trust.injection_signal_count)
            )
        )
    return codes


def _feature_codes(contributions: Sequence[FeatureContribution]) -> list[str]:
    """Codes naming the features that actually moved the score. Needs SHAP."""
    codes: list[str] = []
    for contribution in contributions:
        if not math.isfinite(contribution.shap_value) or contribution.shap_value == 0.0:
            continue
        key = "feature_up" if contribution.shap_value > 0 else "feature_down"
        codes.append(
            REASON_TEMPLATES[key].format(feature=contribution.feature, value=contribution.value)
        )
    return codes
