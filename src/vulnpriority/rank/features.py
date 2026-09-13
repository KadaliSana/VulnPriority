"""Feature construction for the learning-to-rank stack (DESIGN.md 3.8, Gap 5).

``FeatureBuilder`` turns the object graph produced by Components A, B and C into the
numeric matrix the ranker consumes. Two properties are load-bearing and both are enforced
here rather than trusted:

1. **Ablation columns are dropped, not zeroed.** ``feature_names_for(flags)`` decides the
   column list. A disabled component contributes no column at all, so a model trained in
   the ``A-off`` cell cannot rediscover Component A through a constant, and a reviewer can
   see from the frame's width which cell produced a number.
2. **Query groups are scans and must be contiguous.** ``XGBRanker`` group sizes are
   positional, so the builder re-orders rows so that every scan's findings are adjacent,
   preserving first-appearance order of scans and input order within a scan.

Missing intelligence is never guessed at. Every "we do not know" case resolves to the
documented neutral in :data:`NEUTRAL`, applied identically everywhere, so that "no CVE"
and "a CVE with no EPSS snapshot" produce the same row and neither is silently scored as
contested evidence.

Feature derivation table
========================

Every name below is exactly one entry of ``FEATURE_SPECS`` in ``core/models.py``, in that
order. "usable intel" means a :class:`VulnIntel` record whose ``as_of`` is not after the
finding's ``as_of`` and whose CVE is one of the finding's CVEs; "driving CVE" is the
usable record with the highest CVSS base score (ties broken on CVE id), because the
attacker picks the worst of a finding's CVEs.

===============================  =======  ==========================================================
feature                          group    derivation
===============================  =======  ==========================================================
cvss_base_max                    BASE     max base score over the driving CVE's records; 0.0 if none
cvss_version_ord                 BASE     ``cvss_version_ordinal`` of the selected record (2.0->0 .. 4.0->3)
cvss_source_agreement            BASE     ``feeds.cvss_policy.source_agreement``; 1.0 when absent
cvss_ac_low                      BASE     selected record's AC:L flag (v2 AC:L), 0/1
cvss_pr_none                     BASE     selected record's PR:N flag (v2 Au:N), 0/1
cvss_ui_none                     BASE     selected record's UI:N flag (v2: always 1), 0/1
cvss_c_high                      BASE     selected record's C:H / VC:H flag (v2 C:C), 0/1
cvss_i_high                      BASE     selected record's I:H / VI:H flag (v2 I:C), 0/1
cvss_a_high                      BASE     selected record's A:H / VA:H flag (v2 A:C), 0/1
scanner_severity_ord             BASE     index in :data:`SEVERITY_ORDER` (info 0 .. critical 4)
scanner_confidence               BASE     ``Finding.scanner_confidence`` verbatim, already in [0, 1]
cwe_owasp_top10                  BASE     1.0 when the CWE is in :data:`OWASP_TOP10_CWES` (2021 mapping)
vuln_age_days                    BASE     ``as_of - min(published)`` over usable intel, in days; 0.0 if unknown
cluster_size                     BASE     ``Finding.cluster_size``: endpoints sharing this root cause
auth_required_ord                BASE     ``PrivilegeLevel`` IntEnum value of ``Endpoint.auth_required``
method_state_changing            BASE     1.0 for POST/PUT/PATCH/DELETE (``ingest.normalize``)
param_count                      BASE     ``len(Endpoint.parameters)``
a_asset_criticality              A        ``AssetCriticality.criticality``
a_data_sensitivity               A        ``AssetCriticality.data_sensitivity``
a_exposure                       A        ``AssetCriticality.exposure``
a_function_ord                   A        ``attacker.likelihood.function_ordinal`` of the endpoint function
a_is_admin_surface               A        ``AssetCriticality.is_admin_surface``, 0/1
a_is_auth_boundary               A        ``AssetCriticality.is_auth_boundary``, 0/1
a_exploit_feasibility            A        ``ExploitabilityAssessment.exploit_feasibility``
a_exploit_maturity_ord           A        ``ExploitMaturity`` IntEnum value of the assessed maturity
a_attack_complexity_high         A        1.0 when the assessed complexity is HIGH
a_privileges_required_ord        A        ``PrivilegeLevel`` IntEnum value of the assessed requirement
a_user_interaction_required      A        1.0 when the assessed interaction is REQUIRED
a_impact_cia_mean                A        ``(impact_c + impact_i + impact_a) / 3``
a_privilege_gained_ord           A        ``PrivilegeLevel`` IntEnum value of the post-condition
a_p_applicable                   A        ``ApplicabilityAssessment.p_applicable``
a_version_match_ord              A        :data:`VERSION_MATCH_ORDER` (mismatch 0, unknown 1, match 2)
a_confidence                     A        mean of the three assessments' ``confidence`` fields
a_injection_signals              A        ``TrustSummary.injection_signal_count`` (a count, not a rate)
a_intel_documents                A        ``log1p`` of the retrieved document count
a_intel_public_exploit_urls      A        distinct public exploit / proof-of-concept URLs found
a_intel_active_exploitation      A        retrieved material claims active exploitation, 0/1
a_intel_confidence               A        the extraction's own confidence, 0-1
a_intel_corroborates_feeds       A        retrieved material agrees with curated feed evidence, 0/1
a_intel_contradicts_feeds        A        retrieved material contradicts it, 0/1
a_intel_injection_signals        A        injection signals raised across the retrieved documents
b_epss                           B        max EPSS score over usable intel; 0.0 when no snapshot
b_epss_percentile                B        percentile of the record supplying ``b_epss``; 0.0 when none
b_kev                            B        1.0 when in CISA KEV with ``date_added <= as_of``
b_kev_ransomware                 B        1.0 when that KEV entry records ransomware campaign use
b_kev_age_days                   B        ``as_of - date_added`` in days; 0.0 when not in KEV
b_exploit_count                  B        number of exploit records published on or before ``as_of``
b_exploit_maturity_feed_ord      B        highest feed-attested ``ExploitMaturity`` IntEnum value
b_exploit_verified               B        1.0 when any as-of exploit record is ``verified``
b_p_exploit_attacker             B        ``ExploitLikelihood.p_exploit`` from the attacker model
b_impact_log                 B        ``log1p(BusinessImpact.total)``
b_expected_loss_log              B        ``log1p(EnrichedFinding.expected_loss)``
b_remediation_hours              B        ``RemediationCost.hours``
c_reach_delta_log                C        ``log1p(ChainScore.reach_delta)``
c_max_path_prob                  C        ``ChainScore.max_path_prob_to_target``
c_n_paths_through                C        ``ChainScore.n_paths_through``
c_betweenness                    C        ``ChainScore.betweenness``
c_hops_from_entry                C        ``ChainScore.hops_from_entry``
c_privilege_gain                 C        ``ChainScore.privilege_gain`` (privilege levels climbed)
c_is_chokepoint                  C        ``ChainScore.is_chokepoint``, 0/1
===============================  =======  ==========================================================

Monetary features are ``log1p``-scaled because impact spans four orders of magnitude and
an unscaled money column makes every split threshold live in the tail. Ordinal features
use the ``IntEnum`` value directly so that the monotone constraints in
``RankingConfig.monotone`` mean what they say. Counts and day-counts are left raw: trees
are scale-free, and a raw day count is what an analyst can check by hand. The one counted
exception is ``a_intel_documents``, which ``vulnpriority.intel`` specifies as ``log1p`` because
the retrieval budget makes it the one count that varies over orders of magnitude between
runs; the other intel counts are capped by the intel package's own limits and stay raw.

The seven ``a_intel_*`` features are how what the agent read on the internet reaches the
*ranking* rather than only the report. The review's architecture is explicit that the model
is a feature-extraction layer and the learned ranker decides, so retrieved intelligence
arrives here as bounded numbers to be weighed against CVSS, EPSS and KEV - never as a
priority the model assigned. They sit in Component A's group, so an ablation cell with A
switched off drops them entirely.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from vulnpriority.attacker.likelihood import function_ordinal
from vulnpriority.core.errors import VulnPriorityError
from vulnpriority.core.enums import (
    AttackComplexity,
    EndpointFunction,
    ExploitMaturity,
    ScannerSeverity,
    UserInteraction,
    VersionMatch,
)
from vulnpriority.core.models import (
    FEATURE_NAMES,
    ChainScore,
    ComponentFlags,
    EnrichedFinding,
    FeatureFrame,
    VulnIntel,
    feature_names_for,
)
from vulnpriority.feeds.cvss_policy import cvss_features
from vulnpriority.ingest.normalize import method_is_state_changing

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Imported for annotations alone. ``vulnpriority.intel`` reaches the ranking through the
    # duck-typed ``IntelResult.feature_values()`` contract, so this package stays
    # importable whether or not the intel layer is installed, configured or working.
    from vulnpriority.intel.models import IntelResult

__all__ = [
    "SEVERITY_ORDER",
    "VERSION_MATCH_ORDER",
    "OWASP_TOP10_CWES",
    "INTEL_FEATURES",
    "INTEL_LOG_SCALED",
    "NEUTRAL",
    "FEATURE_DOC",
    "severity_ordinal",
    "version_match_ordinal",
    "usable_intel",
    "neutral_row",
    "FeatureBuilder",
]

#: ``ScannerSeverity`` is a string enum, so its rank has to be stated explicitly.
SEVERITY_ORDER: tuple[ScannerSeverity, ...] = (
    ScannerSeverity.INFO,
    ScannerSeverity.LOW,
    ScannerSeverity.MEDIUM,
    ScannerSeverity.HIGH,
    ScannerSeverity.CRITICAL,
)

#: Ordered so the feature increases with "this finding really does apply here". UNKNOWN
#: sits between the two verdicts rather than beside either: it is genuinely intermediate.
VERSION_MATCH_ORDER: dict[VersionMatch, float] = {
    VersionMatch.MISMATCH: 0.0,
    VersionMatch.UNKNOWN: 1.0,
    VersionMatch.MATCH: 2.0,
}

#: CWE ids mapped to the OWASP Top 10 (2021). Membership is a coarse but real signal that
#: a web application finding belongs to a class the industry already prioritises, and it
#: is deliberately a *feature*, never a label (Gap 3).
OWASP_TOP10_CWES: frozenset[int] = frozenset(
    {
        # A01 Broken Access Control
        22, 23, 35, 59, 200, 201, 219, 264, 275, 276, 284, 285, 352, 359, 377, 402,
        425, 441, 497, 538, 540, 548, 552, 566, 601, 639, 651, 668, 706, 862, 863,
        913, 922, 1275,
        # A02 Cryptographic Failures
        261, 296, 310, 319, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331,
        335, 336, 337, 338, 340, 347, 523, 720, 757, 759, 760, 780, 818, 916,
        # A03 Injection
        20, 74, 75, 77, 78, 79, 80, 83, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98,
        99, 100, 113, 116, 138, 184, 470, 471, 564, 610, 643, 644, 652, 917,
        # A04 Insecure Design
        73, 183, 209, 213, 235, 256, 257, 266, 269, 280, 311, 312, 313, 316, 419,
        430, 434, 444, 451, 472, 501, 522, 525, 539, 579, 598, 602, 642, 646, 650,
        653, 656, 657, 799, 807, 840, 841, 927, 1021, 1173,
        # A05 Security Misconfiguration
        2, 11, 13, 15, 16, 260, 315, 520, 526, 537, 541, 547, 611, 614, 756, 776,
        942, 1004, 1032, 1174,
        # A06 Vulnerable and Outdated Components
        937, 1035, 1104,
        # A07 Identification and Authentication Failures
        255, 259, 287, 288, 290, 294, 295, 297, 300, 302, 304, 306, 307, 346, 384,
        521, 613, 620, 640, 798, 940, 1216,
        # A08 Software and Data Integrity Failures
        345, 353, 426, 494, 502, 565, 784, 829, 830, 915,
        # A09 Security Logging and Monitoring Failures
        117, 223, 532, 778,
        # A10 Server-Side Request Forgery
        918,
    }
)

#: The Component A features contributed by ``vulnpriority.intel``, in emission order. Mirrors
#: ``vulnpriority.intel.models.INTEL_FEATURE_NAMES``; a test asserts the two agree, so the two
#: packages cannot drift apart silently.
INTEL_FEATURES: tuple[str, ...] = (
    "a_intel_documents",
    "a_intel_public_exploit_urls",
    "a_intel_active_exploitation",
    "a_intel_confidence",
    "a_intel_corroborates_feeds",
    "a_intel_contradicts_feeds",
    "a_intel_injection_signals",
)

#: Intel features this layer ``log1p``-scales. ``IntelResult.feature_values`` returns raw
#: counts and names this one as the feature layer's responsibility.
INTEL_LOG_SCALED: frozenset[str] = frozenset({"a_intel_documents"})

#: The single documented neutral per feature, used whenever the evidence behind it is
#: absent. The rule is "absence of evidence scores as absence, not as contested": a
#: finding with no CVE gets ``b_epss = 0`` and ``b_kev = 0``, not a mid-scale guess, so it
#: is never promoted by ignorance. The one exception is ``cvss_source_agreement``, which
#: is 1.0 when there is nothing to disagree about - the value
#: ``feeds.cvss_policy.select_cvss`` itself returns for an empty record set.
NEUTRAL: dict[str, float] = {
    "cvss_base_max": 0.0,
    "cvss_version_ord": 0.0,
    "cvss_source_agreement": 1.0,
    "cvss_ac_low": 0.0,
    "cvss_pr_none": 0.0,
    "cvss_ui_none": 0.0,
    "cvss_c_high": 0.0,
    "cvss_i_high": 0.0,
    "cvss_a_high": 0.0,
    "scanner_severity_ord": 0.0,
    "scanner_confidence": 0.5,
    "cwe_owasp_top10": 0.0,
    "vuln_age_days": 0.0,
    "cluster_size": 1.0,
    "auth_required_ord": 0.0,
    "method_state_changing": 0.0,
    "param_count": 0.0,
    "a_asset_criticality": 0.0,
    "a_data_sensitivity": 0.0,
    "a_exposure": 0.0,
    "a_function_ord": function_ordinal(EndpointFunction.UNKNOWN),
    "a_is_admin_surface": 0.0,
    "a_is_auth_boundary": 0.0,
    "a_exploit_feasibility": 0.0,
    "a_exploit_maturity_ord": 0.0,
    "a_attack_complexity_high": 0.0,
    "a_privileges_required_ord": 0.0,
    "a_user_interaction_required": 0.0,
    "a_impact_cia_mean": 0.0,
    "a_privilege_gained_ord": 0.0,
    "a_p_applicable": 0.5,
    "a_version_match_ord": VERSION_MATCH_ORDER[VersionMatch.UNKNOWN],
    "a_confidence": 0.5,
    "a_injection_signals": 0.0,
    # Retrieved intelligence: every one of these is 0.0 when intelligence gathering is
    # disabled, unavailable, or ran and found nothing. Those three are deliberately the
    # same state - the framework learned nothing from the internet about this finding -
    # and a finding that was never searched for must not be scored differently from one
    # that was searched for and turned up empty.
    "a_intel_documents": 0.0,
    "a_intel_public_exploit_urls": 0.0,
    "a_intel_active_exploitation": 0.0,
    "a_intel_confidence": 0.0,
    "a_intel_corroborates_feeds": 0.0,
    "a_intel_contradicts_feeds": 0.0,
    "a_intel_injection_signals": 0.0,
    "b_epss": 0.0,
    "b_epss_percentile": 0.0,
    "b_kev": 0.0,
    "b_kev_ransomware": 0.0,
    "b_kev_age_days": 0.0,
    "b_exploit_count": 0.0,
    "b_exploit_maturity_feed_ord": 0.0,
    "b_exploit_verified": 0.0,
    "b_p_exploit_attacker": 0.0,
    "b_impact_log": 0.0,
    "b_expected_loss_log": 0.0,
    "b_remediation_hours": 0.0,
    "c_reach_delta_log": 0.0,
    "c_max_path_prob": 0.0,
    "c_n_paths_through": 0.0,
    "c_betweenness": 0.0,
    "c_hops_from_entry": 0.0,
    "c_privilege_gain": 0.0,
    "c_is_chokepoint": 0.0,
}

#: One-line derivation per feature, machine-readable companion to the table above. The
#: report builder prints it beside the SHAP summary so a reader never has to guess what a
#: column means.
FEATURE_DOC: dict[str, str] = {
    "cvss_base_max": "highest CVSS base score across the driving CVE's records",
    "cvss_version_ord": "CVSS version ordinal of the record chosen by the selection policy",
    "cvss_source_agreement": "1 - normalised spread of base scores across scoring sources",
    "cvss_ac_low": "chosen CVSS record asserts low attack complexity",
    "cvss_pr_none": "chosen CVSS record asserts no privileges required",
    "cvss_ui_none": "chosen CVSS record asserts no user interaction",
    "cvss_c_high": "chosen CVSS record asserts high confidentiality impact",
    "cvss_i_high": "chosen CVSS record asserts high integrity impact",
    "cvss_a_high": "chosen CVSS record asserts high availability impact",
    "scanner_severity_ord": "scanner's own severity as an ordinal, info 0 to critical 4",
    "scanner_confidence": "scanner's own confidence in the finding",
    "cwe_owasp_top10": "CWE is mapped to the OWASP Top 10 (2021)",
    "vuln_age_days": "days between the earliest CVE publication and the as-of date",
    "cluster_size": "endpoints in this scan sharing the finding's root cause",
    "auth_required_ord": "privilege level the endpoint demands before responding",
    "method_state_changing": "HTTP verb mutates server state",
    "param_count": "number of observed request parameters on the endpoint",
    "a_asset_criticality": "Component A criticality of the endpoint, inferred from structure",
    "a_data_sensitivity": "Component A sensitivity of the data the endpoint handles",
    "a_exposure": "Component A exposure: 1.0 is anonymously reachable from the internet",
    "a_function_ord": "ordinal of the inferred endpoint function",
    "a_is_admin_surface": "endpoint was inferred to be an administrative surface",
    "a_is_auth_boundary": "endpoint was inferred to be an authentication boundary",
    "a_exploit_feasibility": "Component A judgement of how feasible exploitation is here",
    "a_exploit_maturity_ord": "Component A exploit maturity ordinal",
    "a_attack_complexity_high": "Component A judged attack complexity high",
    "a_privileges_required_ord": "privilege the attacker must already hold",
    "a_user_interaction_required": "exploitation needs a victim to act",
    "a_impact_cia_mean": "mean of the assessed confidentiality, integrity and availability impact",
    "a_privilege_gained_ord": "privilege the attacker holds after exploitation",
    "a_p_applicable": "probability the finding actually applies to the observed application",
    "a_version_match_ord": "version evidence: mismatch 0, unknown 1, match 2",
    "a_confidence": "mean confidence across the three Component A assessments",
    "a_injection_signals": "injection signals the sandbox raised while assessing this finding",
    "a_intel_documents": "log1p of the pages retrieved about this finding from the internet",
    "a_intel_public_exploit_urls": "distinct public exploit or proof-of-concept URLs found",
    "a_intel_active_exploitation": "retrieved material claims the vulnerability is being exploited",
    "a_intel_confidence": "how confident the extraction was in what it read",
    "a_intel_corroborates_feeds": "retrieved material agrees with the curated feed evidence",
    "a_intel_contradicts_feeds": "retrieved material contradicts the curated feed evidence",
    "a_intel_injection_signals": "injection signals raised across the retrieved pages",
    "b_epss": "highest EPSS probability across the finding's CVEs as of the cut-off",
    "b_epss_percentile": "EPSS percentile of the record supplying b_epss",
    "b_kev": "CVE is on the CISA Known Exploited Vulnerabilities catalogue",
    "b_kev_ransomware": "KEV entry records known ransomware campaign use",
    "b_kev_age_days": "days since the CVE was added to KEV",
    "b_exploit_count": "public exploit records published on or before the cut-off",
    "b_exploit_maturity_feed_ord": "highest exploit maturity attested by curated feeds",
    "b_exploit_verified": "at least one curated exploit record is marked verified",
    "b_p_exploit_attacker": "attacker model's P(exploit) over the horizon",
    "b_impact_log": "log1p of the estimated monetary impact",
    "b_expected_loss_log": "log1p of P(exploit) x impact, the priority construct itself",
    "b_remediation_hours": "estimated engineering hours to remediate this root cause",
    "c_reach_delta_log": "log1p of the reachable risk this finding unlocks in the attack graph",
    "c_max_path_prob": "highest path probability from the entry node through this finding",
    "c_n_paths_through": "enumerated attack paths that traverse this finding",
    "c_betweenness": "betweenness centrality of the finding's edge in the attack graph",
    "c_hops_from_entry": "hops from the attacker's entry state to this finding",
    "c_privilege_gain": "privilege levels climbed by exploiting this finding",
    "c_is_chokepoint": "removing this finding removes a large share of reachable risk",
}

# The frozen contract owns the feature list; these two tables must track it exactly. A
# missing entry is a silent KeyError at build time deep in a loop, so it is caught here,
# at import, where the message can say which feature was forgotten.
_MISSING_NEUTRALS = [name for name in FEATURE_NAMES if name not in NEUTRAL]
_MISSING_DOCS = [name for name in FEATURE_NAMES if name not in FEATURE_DOC]
if _MISSING_NEUTRALS or _MISSING_DOCS:  # pragma: no cover - import-time contract check
    raise ImportError(
        "vulnpriority.rank.features is out of step with FEATURE_SPECS: "
        f"missing neutrals {_MISSING_NEUTRALS}, missing docs {_MISSING_DOCS}"
    )
_EXTRA = sorted((set(NEUTRAL) | set(FEATURE_DOC)) - set(FEATURE_NAMES))
if _EXTRA:  # pragma: no cover - import-time contract check
    raise ImportError(f"vulnpriority.rank.features documents unknown features: {_EXTRA}")


def severity_ordinal(severity: ScannerSeverity) -> float:
    """Rank of a scanner severity, ``info`` 0 through ``critical`` 4."""
    try:
        return float(SEVERITY_ORDER.index(severity))
    except ValueError:  # pragma: no cover - unreachable while ScannerSeverity is closed
        return NEUTRAL["scanner_severity_ord"]


def version_match_ordinal(match: VersionMatch) -> float:
    """Ordinal for version evidence: mismatch 0, unknown 1, match 2."""
    return VERSION_MATCH_ORDER.get(match, VERSION_MATCH_ORDER[VersionMatch.UNKNOWN])


def usable_intel(enriched: EnrichedFinding) -> list[VulnIntel]:
    """Intel records for this finding's CVEs that are not dated after its ``as_of``.

    The as-of filter is applied here rather than trusted from upstream because temporal
    leakage is the failure mode that quietly inflates every published result (Gap 4).
    """
    as_of = enriched.as_of
    wanted = {cve.upper() for cve in enriched.finding.cve_ids}
    records: list[VulnIntel] = []
    for record in enriched.intel:
        if record.as_of > as_of:
            continue
        if wanted and record.cve_id.upper() not in wanted:
            continue
        records.append(record)
    return records


def neutral_row(columns: Iterable[str] | None = None) -> dict[str, float]:
    """The all-neutral feature row, restricted to ``columns`` when given.

    Used by the rank guard to neutralise Component A columns and re-score, and by tests
    that need a row with no evidence in it at all.
    """
    names = list(columns) if columns is not None else list(NEUTRAL)
    return {name: NEUTRAL[name] for name in names}


def _finite(value: float, neutral: float) -> float:
    """Coerce a computed value to a finite float, falling back to its documented neutral.

    Nothing in the framework should produce a NaN, but a feature matrix with one in it
    fails deep inside XGBoost with an unhelpful message, so the guard is cheap insurance.
    """
    number = float(value)
    return number if math.isfinite(number) else float(neutral)


def _days_between(start: date, end: date) -> float:
    """Non-negative day count from ``start`` to ``end``."""
    return float(max(0, (end - start).days))


class FeatureBuilder:
    """Builds the :class:`FeatureFrame` the rankers consume.

    Stateless and deterministic: the same enriched findings, chain scores and flags always
    produce the same frame, byte for byte, which is what lets the ablation attribute a
    metric difference to the component rather than to the run.
    """

    def __init__(self) -> None:
        """No configuration: the column list is owned by ``FEATURE_SPECS`` and the flags."""

    # -- public API ---------------------------------------------------------

    def build(
        self,
        enriched: Sequence[EnrichedFinding],
        chain: Mapping[str, ChainScore],
        flags: ComponentFlags = ComponentFlags(),
        intel: Mapping[str, "IntelResult"] | None = None,
    ) -> FeatureFrame:
        """Build the feature matrix for an ablation cell.

        ``chain`` maps ``finding_id`` to its :class:`ChainScore`; findings missing from it
        (Component C disabled, or a finding that creates no graph edge) take the neutral
        chain row.

        Retrieved intelligence arrives either way. ``EnrichedFinding.retrieved_intel``
        is the home it belongs in - it is a per-finding Component A product like the three
        assessments beside it, and putting it there is what lets the explainer and the rank
        guard see it without every call site threading a parallel argument. The ``intel``
        mapping is the transitional path and wins where both are present, so a caller that
        has results in hand before the enricher folds them in is not blocked. Absent both,
        every ``a_intel_*`` column takes its neutral, which is the same row a finding gets
        when intelligence gathering ran and found nothing.

        Rows are re-ordered so every scan's findings are contiguous, which is what
        ``FeatureFrame.group_sizes`` and ``XGBRanker`` require.
        """
        columns = feature_names_for(flags)
        ordered = self._group_by_scan(enriched)
        rows = [
            self._row(
                item,
                chain.get(item.finding_id),
                self._intel_for(item, intel),
                columns,
            )
            for item in ordered
        ]

        frame = pd.DataFrame(rows, columns=columns, dtype=float)
        if not rows:
            frame = pd.DataFrame({name: pd.Series(dtype=float) for name in columns})
        return FeatureFrame(
            X=frame,
            finding_ids=[item.finding_id for item in ordered],
            group_ids=[item.scan_id for item in ordered],
            flags=flags,
        )

    def row_for(
        self,
        enriched: EnrichedFinding,
        chain: ChainScore | None = None,
        flags: ComponentFlags = ComponentFlags(),
        intel: "IntelResult | None" = None,
    ) -> dict[str, float]:
        """One feature row as a name-to-value mapping, for inspection and for tests."""
        columns = feature_names_for(flags)
        resolved = intel if intel is not None else self._intel_for(enriched, None)
        return dict(zip(columns, self._row(enriched, chain, resolved, columns)))

    @staticmethod
    def _intel_for(
        enriched: EnrichedFinding,
        intel: Mapping[str, "IntelResult"] | None,
    ) -> "IntelResult | None":
        """Retrieved intelligence for one finding, explicit mapping first.

        The attribute is ``intel_result``. It was read as ``retrieved_intel`` through a
        defensive ``getattr`` that was meant to tolerate an older model and instead
        swallowed a name that has never existed: every finding carrying intelligence
        returned ``None`` here, so all seven ``a_intel_*`` columns sat at their neutral no
        matter what the retrieval layer found. The suite did not catch it because its one
        test supplies intelligence through the explicit ``intel`` mapping, which is the
        path this line is not on.

        Accessed directly now. If the field is ever renamed again, that must be an
        ``AttributeError`` at the first call rather than a column of quiet zeros.
        """
        if intel is not None and enriched.finding_id in intel:
            return intel[enriched.finding_id]
        return enriched.intel_result

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _group_by_scan(enriched: Sequence[EnrichedFinding]) -> list[EnrichedFinding]:
        """Stable regrouping: scans in first-appearance order, findings in input order."""
        buckets: dict[str, list[EnrichedFinding]] = {}
        for item in enriched:
            buckets.setdefault(item.scan_id, []).append(item)
        return [item for bucket in buckets.values() for item in bucket]

    def _row(
        self,
        enriched: EnrichedFinding,
        chain: ChainScore | None,
        intel: "IntelResult | None",
        columns: Sequence[str],
    ) -> list[float]:
        """Compute every feature for one finding, then project onto the cell's columns."""
        values: dict[str, float] = {}
        values.update(self._base_features(enriched))
        values.update(self._component_a_features(enriched))
        values.update(self._intel_features(intel))
        values.update(self._component_b_features(enriched))
        values.update(self._component_c_features(chain))
        return [_finite(values[name], NEUTRAL[name]) for name in columns]

    # -- BASE ---------------------------------------------------------------

    def _base_features(self, enriched: EnrichedFinding) -> dict[str, float]:
        """Scanner facts and curated CVSS: everything that needs no component at all."""
        finding = enriched.finding
        endpoint = enriched.endpoint
        records = usable_intel(enriched)

        features = dict(self._cvss_features(records))
        features["scanner_severity_ord"] = severity_ordinal(finding.scanner_severity)
        features["scanner_confidence"] = float(finding.scanner_confidence)
        features["cwe_owasp_top10"] = (
            1.0 if finding.cwe_id is not None and int(finding.cwe_id) in OWASP_TOP10_CWES else 0.0
        )
        features["vuln_age_days"] = self._vuln_age_days(records, enriched.as_of)
        features["cluster_size"] = float(finding.cluster_size)
        features["auth_required_ord"] = float(int(endpoint.auth_required))
        features["method_state_changing"] = 1.0 if method_is_state_changing(endpoint.method) else 0.0
        features["param_count"] = float(len(endpoint.parameters))
        return features

    @staticmethod
    def _cvss_features(records: Sequence[VulnIntel]) -> dict[str, float]:
        """CVSS features from the driving CVE, or the documented neutrals when there is none.

        The driving CVE is the one with the highest base score: a finding that cites three
        CVEs is as urgent as its worst, and mixing scores from different CVEs into one
        ``source_agreement`` would report a disagreement that no analyst made.
        """
        with_cvss = [record for record in records if record.cvss]
        if not with_cvss:
            return {
                name: NEUTRAL[name]
                for name in (
                    "cvss_base_max",
                    "cvss_version_ord",
                    "cvss_source_agreement",
                    "cvss_ac_low",
                    "cvss_pr_none",
                    "cvss_ui_none",
                    "cvss_c_high",
                    "cvss_i_high",
                    "cvss_a_high",
                )
            }
        driving = max(
            with_cvss,
            key=lambda record: (max(item.base_score for item in record.cvss), record.cve_id),
        )
        return cvss_features(driving.cvss)

    @staticmethod
    def _vuln_age_days(records: Sequence[VulnIntel], as_of: date) -> float:
        """Days since the earliest publication of any of the finding's CVEs."""
        published = [record.published for record in records if record.published is not None]
        if not published:
            return NEUTRAL["vuln_age_days"]
        return _days_between(min(published), as_of)

    # -- Component A --------------------------------------------------------

    @staticmethod
    def _component_a_features(enriched: EnrichedFinding) -> dict[str, float]:
        """Everything the agentic semantic assessment produced for this finding."""
        asset = enriched.asset
        exploitability = enriched.exploitability
        applicability = enriched.applicability
        confidences = (asset.confidence, exploitability.confidence, applicability.confidence)
        return {
            "a_asset_criticality": float(asset.criticality),
            "a_data_sensitivity": float(asset.data_sensitivity),
            "a_exposure": float(asset.exposure),
            "a_function_ord": function_ordinal(asset.function),
            "a_is_admin_surface": 1.0 if asset.is_admin_surface else 0.0,
            "a_is_auth_boundary": 1.0 if asset.is_auth_boundary else 0.0,
            "a_exploit_feasibility": float(exploitability.exploit_feasibility),
            "a_exploit_maturity_ord": float(int(exploitability.exploit_maturity)),
            "a_attack_complexity_high": (
                1.0 if exploitability.attack_complexity == AttackComplexity.HIGH else 0.0
            ),
            "a_privileges_required_ord": float(int(exploitability.privileges_required)),
            "a_user_interaction_required": (
                1.0 if exploitability.user_interaction == UserInteraction.REQUIRED else 0.0
            ),
            "a_impact_cia_mean": float(exploitability.impact_cia_mean),
            "a_privilege_gained_ord": float(int(exploitability.privilege_gained)),
            "a_p_applicable": float(applicability.p_applicable),
            "a_version_match_ord": version_match_ordinal(applicability.version_match),
            "a_confidence": float(sum(confidences) / len(confidences)),
            "a_injection_signals": float(enriched.trust.injection_signal_count),
        }

    # -- Component A, retrieved intelligence --------------------------------

    @staticmethod
    def _intel_features(intel: "IntelResult | None") -> dict[str, float]:
        """Bounded numbers extracted from what the agent read on the internet.

        The derivation is deliberately thin: ``vulnpriority.intel`` owns the extraction and
        exposes it through ``IntelResult.feature_values()``, and this layer only applies
        the scaling it declared as the feature layer's responsibility. Duplicating the
        derivation here would let the two drift, and the intel package is the one that
        knows how it counted.

        No intel for this finding means every column takes its neutral - the same row the
        intel package itself returns when gathering ran and found nothing. Disabled,
        unavailable and empty are one state and must be indistinguishable in the matrix.
        """
        if intel is None:
            return {name: NEUTRAL[name] for name in INTEL_FEATURES}

        raw = dict(intel.feature_values())
        missing = [name for name in INTEL_FEATURES if name not in raw]
        if missing:  # pragma: no cover - contract check against a package in flux
            raise VulnPriorityError(
                f"IntelResult.feature_values() omitted {missing}; "
                "vulnpriority.intel and vulnpriority.rank.features disagree about INTEL_FEATURES"
            )
        return {
            name: (
                math.log1p(max(0.0, float(raw[name])))
                if name in INTEL_LOG_SCALED
                else float(raw[name])
            )
            for name in INTEL_FEATURES
        }

    # -- Component B --------------------------------------------------------

    def _component_b_features(self, enriched: EnrichedFinding) -> dict[str, float]:
        """Curated threat intelligence plus the attacker, impact and cost models."""
        records = usable_intel(enriched)
        as_of = enriched.as_of
        epss, percentile = self._best_epss(records, as_of)
        in_kev, ransomware, kev_added = self._kev_state(records, as_of)
        count, maturity, verified = self._exploit_state(records, as_of)

        return {
            "b_epss": epss,
            "b_epss_percentile": percentile,
            "b_kev": 1.0 if in_kev else 0.0,
            "b_kev_ransomware": 1.0 if ransomware else 0.0,
            "b_kev_age_days": (
                _days_between(kev_added, as_of) if in_kev and kev_added is not None
                else NEUTRAL["b_kev_age_days"]
            ),
            "b_exploit_count": float(count),
            "b_exploit_maturity_feed_ord": float(int(maturity)),
            "b_exploit_verified": 1.0 if verified else 0.0,
            "b_p_exploit_attacker": float(enriched.likelihood.p_exploit),
            "b_impact_log": math.log1p(max(0.0, float(enriched.impact.total))),
            "b_expected_loss_log": math.log1p(max(0.0, float(enriched.expected_loss))),
            "b_remediation_hours": float(enriched.remediation.hours),
        }

    @staticmethod
    def _best_epss(records: Sequence[VulnIntel], as_of: date) -> tuple[float, float]:
        """``(score, percentile)`` of the highest as-of EPSS snapshot; neutrals when none."""
        snapshots = [
            record.epss for record in records if record.epss is not None and record.epss.as_of <= as_of
        ]
        if not snapshots:
            return NEUTRAL["b_epss"], NEUTRAL["b_epss_percentile"]
        best = max(snapshots, key=lambda item: (item.score, item.percentile, item.cve_id))
        return float(best.score), float(best.percentile)

    @staticmethod
    def _kev_state(
        records: Sequence[VulnIntel], as_of: date
    ) -> tuple[bool, bool, date | None]:
        """``(in_kev, ransomware, earliest date_added)`` honouring ``date_added <= as_of``."""
        in_kev = False
        ransomware = False
        added: date | None = None
        for record in records:
            kev = record.kev
            if kev is None or not kev.in_kev:
                continue
            if kev.date_added is not None and kev.date_added > as_of:
                continue
            in_kev = True
            ransomware = ransomware or kev.known_ransomware_use
            if kev.date_added is not None and (added is None or kev.date_added < added):
                added = kev.date_added
        return in_kev, ransomware, added

    @staticmethod
    def _exploit_state(
        records: Sequence[VulnIntel], as_of: date
    ) -> tuple[int, ExploitMaturity, bool]:
        """``(count, highest maturity, any verified)`` over as-of exploit evidence."""
        count = 0
        maturity = ExploitMaturity.UNKNOWN
        verified = False
        for record in records:
            for exploit in record.exploits:
                if exploit.published is not None and exploit.published > as_of:
                    continue
                count += 1
                if exploit.maturity > maturity:
                    maturity = exploit.maturity
                verified = verified or exploit.verified
        return count, maturity, verified

    # -- Component C --------------------------------------------------------

    @staticmethod
    def _component_c_features(chain: ChainScore | None) -> dict[str, float]:
        """Attack-graph position; a finding with no chain score takes the neutral row."""
        if chain is None:
            return {
                name: NEUTRAL[name]
                for name in (
                    "c_reach_delta_log",
                    "c_max_path_prob",
                    "c_n_paths_through",
                    "c_betweenness",
                    "c_hops_from_entry",
                    "c_privilege_gain",
                    "c_is_chokepoint",
                )
            }
        return {
            "c_reach_delta_log": math.log1p(max(0.0, float(chain.reach_delta))),
            "c_max_path_prob": float(chain.max_path_prob_to_target),
            "c_n_paths_through": float(chain.n_paths_through),
            "c_betweenness": float(chain.betweenness),
            "c_hops_from_entry": float(chain.hops_from_entry),
            "c_privilege_gain": float(chain.privilege_gain),
            "c_is_chokepoint": 1.0 if chain.is_chokepoint else 0.0,
        }
