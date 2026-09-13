"""``FeatureBuilder``: the feature matrix and the ablation contract (DESIGN.md 3.8).

Everything here is built directly from ``vulnpriority.core.models`` rather than by running
Components A, B and C, because those are written in parallel and the feature contract has
to hold independently of them.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import pytest

from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    AttackComplexity,
    Component,
    CvssVersion,
    EndpointFunction,
    ExploitMaturity,
    ExploitSource,
    HttpMethod,
    PrivilegeLevel,
    Provenance,
    ScannerSeverity,
    ScoreSource,
    UserInteraction,
    VersionMatch,
)
from vulnpriority.core.models import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    ApplicabilityAssessment,
    AssetCriticality,
    BusinessImpact,
    ChainScore,
    ComponentFlags,
    CvssRecord,
    Endpoint,
    EnrichedFinding,
    EpssRecord,
    ExploitEvidence,
    ExploitLikelihood,
    ExploitabilityAssessment,
    Finding,
    KevRecord,
    RemediationCost,
    TechComponent,
    TrustSummary,
    UntrustedText,
    VulnIntel,
    feature_names_for,
)
from vulnpriority.intel.models import (
    INTEL_FEATURE_NAMES,
    ExploitIntelOut,
    FeedAgreement,
    IntelDocument,
    IntelResult,
    IntelSourceKind,
)
from vulnpriority.rank.features import (
    FEATURE_DOC,
    INTEL_FEATURES,
    NEUTRAL,
    OWASP_TOP10_CWES,
    FeatureBuilder,
    neutral_row,
    severity_ordinal,
    version_match_ordinal,
)

AS_OF = date(2024, 6, 1)
OBSERVED_AT = datetime(2024, 5, 1, 9, 0, 0)


# ---------------------------------------------------------------------------
# Object factories: core models only, no component packages
# ---------------------------------------------------------------------------


def make_endpoint(
    endpoint_id: str = "ep_1",
    *,
    method: HttpMethod = HttpMethod.POST,
    auth: PrivilegeLevel = PrivilegeLevel.USER,
    parameters: tuple[str, ...] = ("username", "password"),
) -> Endpoint:
    return Endpoint(
        endpoint_id=endpoint_id,
        app_id="app1",
        host="shop.example.com",
        url=f"https://shop.example.com/{endpoint_id}",
        path=f"/{endpoint_id}",
        method=method,
        auth_required=auth,
        parameters=parameters,
        observed_tech=(TechComponent(vendor="apache", product="struts", version="2.5.12"),),
    )


def make_finding(
    finding_id: str = "f_1",
    scan_id: str = "scan_1",
    *,
    endpoint_id: str = "ep_1",
    cwe_id: int | None = 89,
    cve_ids: tuple[str, ...] = ("CVE-2024-0001",),
    severity: ScannerSeverity = ScannerSeverity.HIGH,
    confidence: float = 0.9,
    cluster_size: int = 3,
) -> Finding:
    return Finding(
        finding_id=finding_id,
        scan_id=scan_id,
        app_id="app1",
        endpoint_id=endpoint_id,
        name="SQL Injection",
        cwe_id=cwe_id,
        cve_ids=cve_ids,
        scanner="zap",
        scanner_plugin_id="40018",
        scanner_severity=severity,
        scanner_confidence=confidence,
        description=UntrustedText(text="injection", provenance=Provenance.SCANNER_OUTPUT),
        affected_component=TechComponent(vendor="apache", product="struts", version="2.5.12"),
        observed_at=OBSERVED_AT,
        dedup_key="dk_1",
        cluster_size=cluster_size,
    )


def make_intel(
    cve_id: str = "CVE-2024-0001",
    *,
    as_of: date = AS_OF,
    base_scores: tuple[float, ...] = (9.8,),
    epss: float | None = 0.42,
    percentile: float = 0.97,
    in_kev: bool = True,
    kev_added: date | None = date(2024, 2, 1),
    ransomware: bool = False,
    exploits: int = 1,
    maturity: ExploitMaturity = ExploitMaturity.FUNCTIONAL,
    verified: bool = True,
    published: date | None = date(2024, 1, 10),
) -> VulnIntel:
    return VulnIntel(
        cve_id=cve_id,
        as_of=as_of,
        published=published,
        cvss=tuple(
            CvssRecord(
                version=CvssVersion.V31,
                source=source,
                base_score=score,
                vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                submetrics={"AV": "N", "AC": "L", "PR": "N", "UI": "N", "C": "H", "I": "H", "A": "H"},
            )
            for score, source in zip(base_scores, (ScoreSource.NVD, ScoreSource.CNA, ScoreSource.OTHER))
        ),
        epss=None if epss is None else EpssRecord(cve_id=cve_id, score=epss, percentile=percentile, as_of=as_of),
        kev=KevRecord(
            cve_id=cve_id,
            in_kev=in_kev,
            date_added=kev_added,
            known_ransomware_use=ransomware,
            as_of=as_of,
        ),
        exploits=tuple(
            ExploitEvidence(
                source=ExploitSource.EXPLOIT_DB,
                published=date(2024, 1, 20),
                maturity=maturity,
                verified=verified,
            )
            for _ in range(exploits)
        ),
    )


def make_enriched(
    finding: Finding | None = None,
    endpoint: Endpoint | None = None,
    intel: tuple[VulnIntel, ...] = (),
    *,
    criticality: float = 0.8,
    sensitivity: float = 0.7,
    exposure: float = 1.0,
    function: EndpointFunction = EndpointFunction.AUTH,
    admin: bool = False,
    auth_boundary: bool = True,
    feasibility: float = 0.7,
    a_maturity: ExploitMaturity = ExploitMaturity.POC,
    complexity: AttackComplexity = AttackComplexity.LOW,
    privileges_required: PrivilegeLevel = PrivilegeLevel.NONE,
    interaction: UserInteraction = UserInteraction.NONE,
    impact_cia: tuple[float, float, float] = (0.9, 0.6, 0.3),
    privilege_gained: PrivilegeLevel = PrivilegeLevel.ADMIN,
    p_applicable: float = 0.9,
    verdict: ApplicabilityVerdict = ApplicabilityVerdict.APPLICABLE,
    version_match: VersionMatch = VersionMatch.MATCH,
    confidences: tuple[float, float, float] = (0.6, 0.8, 0.4),
    p_exploit: float = 0.55,
    impact: float = 250_000.0,
    hours: float = 8.0,
    expected_loss: float | None = None,
    signals: int = 0,
    as_of: date = AS_OF,
    trust: TrustSummary | None = None,
) -> EnrichedFinding:
    finding = finding if finding is not None else make_finding()
    endpoint = endpoint if endpoint is not None else make_endpoint()
    loss = expected_loss if expected_loss is not None else p_exploit * impact
    return EnrichedFinding(
        finding=finding,
        endpoint=endpoint,
        intel=intel,
        asset=AssetCriticality(
            endpoint_id=endpoint.endpoint_id,
            function=function,
            criticality=criticality,
            data_sensitivity=sensitivity,
            exposure=exposure,
            is_auth_boundary=auth_boundary,
            is_admin_surface=admin,
            confidence=confidences[0],
        ),
        exploitability=ExploitabilityAssessment(
            finding_id=finding.finding_id,
            exploit_feasibility=feasibility,
            exploit_maturity=a_maturity,
            attack_complexity=complexity,
            privileges_required=privileges_required,
            user_interaction=interaction,
            impact_c=impact_cia[0],
            impact_i=impact_cia[1],
            impact_a=impact_cia[2],
            privilege_gained=privilege_gained,
            confidence=confidences[1],
        ),
        applicability=ApplicabilityAssessment(
            finding_id=finding.finding_id,
            verdict=verdict,
            p_applicable=p_applicable,
            version_match=version_match,
            confidence=confidences[2],
        ),
        likelihood=ExploitLikelihood(
            finding_id=finding.finding_id,
            attacker="opportunistic",
            p_exploit=p_exploit,
            p_exploit_uncapped=p_exploit,
            horizon_days=90,
        ),
        impact=BusinessImpact(finding_id=finding.finding_id, total=impact),
        remediation=RemediationCost(finding_id=finding.finding_id, hours=hours, cost=hours * 120.0),
        expected_loss=loss,
        trust=trust if trust is not None else TrustSummary(injection_signal_count=signals),
        as_of=as_of,
    )


def make_chain(
    finding_id: str = "f_1",
    *,
    reach: float = 12_400.0,
    prob: float = 0.4,
    paths: int = 3,
    betweenness: float = 0.25,
    hops: int = 2,
    gain: int = 2,
    chokepoint: bool = True,
) -> ChainScore:
    return ChainScore(
        finding_id=finding_id,
        reach_delta=reach,
        max_path_prob_to_target=prob,
        n_paths_through=paths,
        betweenness=betweenness,
        hops_from_entry=hops,
        privilege_gain=gain,
        is_chokepoint=chokepoint,
    )


@pytest.fixture
def builder() -> FeatureBuilder:
    return FeatureBuilder()


@pytest.fixture
def one() -> tuple[list[EnrichedFinding], dict[str, ChainScore]]:
    """One fully populated finding with intel and a chain score."""
    enriched = make_enriched(intel=(make_intel(),))
    return [enriched], {enriched.finding_id: make_chain(enriched.finding_id)}


# ---------------------------------------------------------------------------
# Column contract: the ablation drops columns, it does not zero them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flags", ComponentFlags.all_cells(), ids=lambda f: f.label())
def test_every_ablation_cell_has_exactly_its_own_columns(builder, one, flags) -> None:
    enriched, chain = one
    frame = builder.build(enriched, chain, flags)

    assert frame.feature_names == feature_names_for(flags)
    assert list(frame.X.columns) == feature_names_for(flags)

    present = set(frame.feature_names)
    for name, group in FEATURE_GROUPS.items():
        if group is None or group in flags.enabled():
            assert name in present, f"{name} should be present in cell {flags.label()}"
        else:
            assert name not in present, f"{name} leaked into cell {flags.label()}"


def test_the_full_cell_carries_every_declared_feature(builder, one) -> None:
    enriched, chain = one
    frame = builder.build(enriched, chain, ComponentFlags(a=True, b=True, c=True))
    assert frame.feature_names == list(FEATURE_NAMES)
    assert len(frame.feature_names) == len(set(FEATURE_NAMES))


def test_disabled_components_are_dropped_rather_than_zeroed(builder, one) -> None:
    """A zeroed column would still be a column, and a model could learn the pattern."""
    enriched, chain = one
    off = builder.build(enriched, chain, ComponentFlags(a=False, b=False, c=True))
    assert not [name for name in off.feature_names if name.startswith(("a_", "b_"))]
    assert [name for name in off.feature_names if name.startswith("c_")]


# ---------------------------------------------------------------------------
# Grouping: scans must be contiguous because XGBoost group sizes are positional
# ---------------------------------------------------------------------------


def _interleaved() -> list[EnrichedFinding]:
    """Findings deliberately supplied in scan order 1, 2, 1, 2, 3, 1."""
    plan = [("scan_1", 3), ("scan_2", 2), ("scan_3", 1)]
    pattern = ["scan_1", "scan_2", "scan_1", "scan_2", "scan_3", "scan_1"]
    counters = {scan: 0 for scan, _ in plan}
    out: list[EnrichedFinding] = []
    for scan_id in pattern:
        counters[scan_id] += 1
        finding = make_finding(f"f_{scan_id}_{counters[scan_id]}", scan_id)
        out.append(make_enriched(finding=finding))
    return out


def test_rows_are_regrouped_so_each_scan_is_contiguous(builder) -> None:
    frame = builder.build(_interleaved(), {})

    seen: list[str] = []
    for group_id in frame.group_ids:
        if not seen or seen[-1] != group_id:
            assert group_id not in seen, f"{group_id} appears in two separate blocks"
            seen.append(group_id)
    assert seen == ["scan_1", "scan_2", "scan_3"]
    assert list(frame.group_sizes()) == [3, 2, 1]


def test_regrouping_preserves_input_order_inside_a_scan(builder) -> None:
    frame = builder.build(_interleaved(), {})
    assert frame.finding_ids == [
        "f_scan_1_1",
        "f_scan_1_2",
        "f_scan_1_3",
        "f_scan_2_1",
        "f_scan_2_2",
        "f_scan_3_1",
    ]


def test_group_sizes_sum_to_the_row_count(builder) -> None:
    frame = builder.build(_interleaved(), {})
    assert int(frame.group_sizes().sum()) == len(frame.X)


# ---------------------------------------------------------------------------
# Finiteness
# ---------------------------------------------------------------------------


def test_every_feature_is_finite_on_a_populated_finding(builder, one) -> None:
    enriched, chain = one
    frame = builder.build(enriched, chain)
    values = frame.to_numpy()
    assert values.shape == (1, len(FEATURE_NAMES))
    assert bool((~(values == values)).sum() == 0)          # no NaN
    assert all(math.isfinite(value) for value in values.ravel())


def test_every_feature_is_finite_on_an_evidence_free_finding(builder) -> None:
    """No CVE, no intel, no chain score: the row must still be numeric and finite."""
    bare = make_enriched(
        finding=make_finding("f_bare", cwe_id=None, cve_ids=()),
        impact=0.0,
        expected_loss=0.0,
    )
    frame = builder.build([bare], {})
    assert all(math.isfinite(value) for value in frame.to_numpy().ravel())


def test_an_empty_input_gives_an_empty_but_valid_frame(builder) -> None:
    frame = builder.build([], {})
    assert frame.feature_names == list(FEATURE_NAMES)
    assert len(frame.X) == 0
    assert list(frame.group_sizes()) == []


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------


def test_base_features_come_from_the_scanner_and_the_cvss_policy(builder, one) -> None:
    enriched, chain = one
    row = builder.build(enriched, chain).X.iloc[0]

    assert row["cvss_base_max"] == pytest.approx(9.8)
    assert row["cvss_version_ord"] == pytest.approx(2.0)          # 3.1 is the third version
    assert row["cvss_source_agreement"] == pytest.approx(1.0)     # one source, nothing to disagree
    for flag in ("cvss_ac_low", "cvss_pr_none", "cvss_ui_none", "cvss_c_high", "cvss_i_high", "cvss_a_high"):
        assert row[flag] == pytest.approx(1.0)
    assert row["scanner_severity_ord"] == pytest.approx(severity_ordinal(ScannerSeverity.HIGH))
    assert row["scanner_severity_ord"] == pytest.approx(3.0)
    assert row["scanner_confidence"] == pytest.approx(0.9)
    assert row["cwe_owasp_top10"] == pytest.approx(1.0)           # CWE-89 is A03 Injection
    assert row["vuln_age_days"] == pytest.approx((AS_OF - date(2024, 1, 10)).days)
    assert row["cluster_size"] == pytest.approx(3.0)
    assert row["auth_required_ord"] == pytest.approx(float(PrivilegeLevel.USER))
    assert row["method_state_changing"] == pytest.approx(1.0)     # POST
    assert row["param_count"] == pytest.approx(2.0)


def test_cvss_source_disagreement_lowers_the_agreement_feature(builder) -> None:
    enriched = make_enriched(intel=(make_intel(base_scores=(9.8, 5.8)),))
    row = builder.build([enriched], {}).X.iloc[0]
    assert row["cvss_base_max"] == pytest.approx(9.8)
    assert row["cvss_source_agreement"] == pytest.approx(1.0 - 4.0 / 10.0)


def test_a_cwe_outside_the_owasp_mapping_scores_zero(builder) -> None:
    unmapped = next(index for index in range(4000, 4100) if index not in OWASP_TOP10_CWES)
    enriched = make_enriched(finding=make_finding(cwe_id=unmapped))
    assert builder.build([enriched], {}).X.iloc[0]["cwe_owasp_top10"] == pytest.approx(0.0)


def test_component_a_features_mirror_the_assessments(builder, one) -> None:
    enriched, chain = one
    row = builder.build(enriched, chain).X.iloc[0]

    assert row["a_asset_criticality"] == pytest.approx(0.8)
    assert row["a_data_sensitivity"] == pytest.approx(0.7)
    assert row["a_exposure"] == pytest.approx(1.0)
    assert row["a_is_admin_surface"] == pytest.approx(0.0)
    assert row["a_is_auth_boundary"] == pytest.approx(1.0)
    assert row["a_exploit_feasibility"] == pytest.approx(0.7)
    assert row["a_exploit_maturity_ord"] == pytest.approx(float(ExploitMaturity.POC))
    assert row["a_attack_complexity_high"] == pytest.approx(0.0)
    assert row["a_privileges_required_ord"] == pytest.approx(0.0)
    assert row["a_user_interaction_required"] == pytest.approx(0.0)
    assert row["a_impact_cia_mean"] == pytest.approx((0.9 + 0.6 + 0.3) / 3.0)
    assert row["a_privilege_gained_ord"] == pytest.approx(float(PrivilegeLevel.ADMIN))
    assert row["a_p_applicable"] == pytest.approx(0.9)
    assert row["a_version_match_ord"] == pytest.approx(version_match_ordinal(VersionMatch.MATCH))
    assert row["a_confidence"] == pytest.approx((0.6 + 0.8 + 0.4) / 3.0)
    assert row["a_injection_signals"] == pytest.approx(0.0)


def test_version_match_ordinal_orders_mismatch_below_match() -> None:
    assert (
        version_match_ordinal(VersionMatch.MISMATCH)
        < version_match_ordinal(VersionMatch.UNKNOWN)
        < version_match_ordinal(VersionMatch.MATCH)
    )


def test_injection_signals_reach_the_frame(builder) -> None:
    enriched = make_enriched(signals=4)
    assert builder.build([enriched], {}).X.iloc[0]["a_injection_signals"] == pytest.approx(4.0)


def test_component_b_features_mirror_the_feeds_and_the_models(builder, one) -> None:
    enriched, chain = one
    row = builder.build(enriched, chain).X.iloc[0]

    assert row["b_epss"] == pytest.approx(0.42)
    assert row["b_epss_percentile"] == pytest.approx(0.97)
    assert row["b_kev"] == pytest.approx(1.0)
    assert row["b_kev_ransomware"] == pytest.approx(0.0)
    assert row["b_kev_age_days"] == pytest.approx((AS_OF - date(2024, 2, 1)).days)
    assert row["b_exploit_count"] == pytest.approx(1.0)
    assert row["b_exploit_maturity_feed_ord"] == pytest.approx(float(ExploitMaturity.FUNCTIONAL))
    assert row["b_exploit_verified"] == pytest.approx(1.0)
    assert row["b_p_exploit_attacker"] == pytest.approx(0.55)
    assert row["b_impact_log"] == pytest.approx(math.log1p(250_000.0))
    assert row["b_expected_loss_log"] == pytest.approx(math.log1p(0.55 * 250_000.0))
    assert row["b_remediation_hours"] == pytest.approx(8.0)


def test_ransomware_flagged_kev_entries_raise_their_own_feature(builder) -> None:
    enriched = make_enriched(intel=(make_intel(ransomware=True),))
    assert builder.build([enriched], {}).X.iloc[0]["b_kev_ransomware"] == pytest.approx(1.0)


def test_the_highest_epss_across_several_cves_wins(builder) -> None:
    """The attacker picks the easiest of a finding's CVEs, so the feature takes the max."""
    finding = make_finding(cve_ids=("CVE-2024-0001", "CVE-2024-0002"))
    intel = (
        make_intel("CVE-2024-0001", epss=0.10, percentile=0.50, base_scores=(7.0,)),
        make_intel("CVE-2024-0002", epss=0.80, percentile=0.99, base_scores=(5.0,)),
    )
    row = builder.build([make_enriched(finding=finding, intel=intel)], {}).X.iloc[0]
    assert row["b_epss"] == pytest.approx(0.80)
    assert row["b_epss_percentile"] == pytest.approx(0.99)
    assert row["cvss_base_max"] == pytest.approx(7.0)      # the driving CVE is the worst-scored


def test_component_c_features_are_log_scaled_where_they_are_monetary(builder, one) -> None:
    enriched, chain = one
    row = builder.build(enriched, chain).X.iloc[0]

    assert row["c_reach_delta_log"] == pytest.approx(math.log1p(12_400.0))
    assert row["c_max_path_prob"] == pytest.approx(0.4)
    assert row["c_n_paths_through"] == pytest.approx(3.0)
    assert row["c_betweenness"] == pytest.approx(0.25)
    assert row["c_hops_from_entry"] == pytest.approx(2.0)
    assert row["c_privilege_gain"] == pytest.approx(2.0)
    assert row["c_is_chokepoint"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Retrieved exploit intelligence (vulnpriority.intel) reaching the ranker
# ---------------------------------------------------------------------------


def intel_result(
    finding_id: str = "f_1",
    *,
    documents: int = 0,
    exploit_urls: tuple[str, ...] = (),
    active: bool = False,
    confidence: float = 0.5,
    corroborates: bool = False,
    contradicts: bool = False,
    signals: int = 0,
) -> "IntelResult":
    """An ``IntelResult`` with exactly the retrieved evidence a test needs."""
    return IntelResult(
        finding_id=finding_id,
        cve_id="CVE-2024-0001",
        as_of=AS_OF,
        documents=tuple(
            IntelDocument(
                url=f"https://example.test/doc{index}",
                snippet=UntrustedText(
                    text=f"retrieved page {index}", provenance=Provenance.REFERENCE_PAGE
                ),
                retrieved_at=datetime(2024, 5, 20, 12, 0, 0),
                source_kind=IntelSourceKind.WRITEUP,
            )
            for index in range(documents)
        ),
        extraction=(
            ExploitIntelOut(confidence=confidence, active_exploitation_claimed=active)
            if documents or exploit_urls
            else None
        ),
        agreement=FeedAgreement(corroborates=corroborates, contradicts=contradicts),
        public_exploit_urls=exploit_urls,
        active_exploitation_claimed=active,
        injection_signals=signals,
    )


def test_the_intel_feature_list_agrees_with_the_intel_package() -> None:
    """Two packages, one ordered contract; drift here would misalign whole columns."""
    assert INTEL_FEATURES == INTEL_FEATURE_NAMES
    assert all(FEATURE_GROUPS[name] is Component.A for name in INTEL_FEATURES)


def test_retrieved_intelligence_reaches_the_feature_matrix(builder) -> None:
    enriched = make_enriched()
    result = intel_result(
        enriched.finding_id,
        documents=4,
        exploit_urls=("https://example.test/poc1", "https://example.test/poc2"),
        active=True,
        confidence=0.75,
        corroborates=True,
        signals=3,
    )
    row = builder.build([enriched], {}, ComponentFlags(), {enriched.finding_id: result}).X.iloc[0]

    assert row["a_intel_documents"] == pytest.approx(math.log1p(4.0))
    assert row["a_intel_public_exploit_urls"] == pytest.approx(2.0)
    assert row["a_intel_active_exploitation"] == pytest.approx(1.0)
    assert row["a_intel_confidence"] == pytest.approx(0.75)
    assert row["a_intel_corroborates_feeds"] == pytest.approx(1.0)
    assert row["a_intel_contradicts_feeds"] == pytest.approx(0.0)
    assert row["a_intel_injection_signals"] == pytest.approx(3.0)


def test_the_document_count_is_the_one_intel_count_that_is_log_scaled(builder) -> None:
    """``IntelResult.feature_values`` returns raw counts and names this one as ours."""
    enriched = make_enriched()
    urls = tuple(f"https://example.test/poc{index}" for index in range(5))
    result = intel_result(enriched.finding_id, documents=7, exploit_urls=urls)
    row = builder.row_for(enriched, None, ComponentFlags(), result)

    assert row["a_intel_documents"] == pytest.approx(math.log1p(7.0))
    assert row["a_intel_public_exploit_urls"] == pytest.approx(5.0)      # raw, capped upstream
    assert row["a_intel_injection_signals"] == pytest.approx(0.0)


def test_a_finding_with_no_intel_and_one_whose_intel_found_nothing_are_identical(builder) -> None:
    """Disabled, unavailable and empty are one state: the framework learned nothing.

    A finding that was never searched for must not be scored differently from one that was
    searched for and turned up empty. Neither is evidence of absence, and a ranker handed a
    distinction that does not exist will happily learn it.
    """
    never_searched = make_enriched(finding=make_finding("f_a"))
    searched_empty = make_enriched(finding=make_finding("f_b"))
    empty = intel_result("f_b")

    assert empty.found_anything is False
    without = builder.row_for(never_searched, None, ComponentFlags(), None)
    found_nothing = builder.row_for(searched_empty, None, ComponentFlags(), empty)

    assert without == pytest.approx(found_nothing)
    for name in INTEL_FEATURES:
        assert found_nothing[name] == pytest.approx(NEUTRAL[name]), name


def test_intel_absent_from_the_mapping_takes_the_neutral_row(builder) -> None:
    items = [make_enriched(finding=make_finding("f_a")), make_enriched(finding=make_finding("f_b"))]
    mapping = {"f_a": intel_result("f_a", documents=3, active=True)}
    frame = builder.build(items, {}, ComponentFlags(), mapping)

    covered, uncovered = frame.X.iloc[0], frame.X.iloc[1]
    assert covered["a_intel_documents"] > 0.0
    for name in INTEL_FEATURES:
        assert uncovered[name] == pytest.approx(NEUTRAL[name]), name


def test_corroboration_and_contradiction_are_separate_columns(builder) -> None:
    """Not opposite ends of one axis: they are different evidence about different things.

    A page agreeing with CISA and a page insisting a KEV entry is a false positive are not
    the same claim with opposite signs, and material can corroborate on exploitation while
    contradicting on affected versions at the same time. One signed feature could not
    express that, and collapsing them would discard exactly the distinction this layer
    exists to provide.
    """
    assert "a_intel_corroborates_feeds" in FEATURE_NAMES
    assert "a_intel_contradicts_feeds" in FEATURE_NAMES

    enriched = make_enriched()

    def row(corroborates: bool, contradicts: bool) -> dict[str, float]:
        result = intel_result(
            enriched.finding_id,
            documents=2,
            corroborates=corroborates,
            contradicts=contradicts,
        )
        return builder.row_for(enriched, None, ComponentFlags(), result)

    neither = row(False, False)
    agrees = row(True, False)
    disagrees = row(False, True)
    both = row(True, True)

    assert (agrees["a_intel_corroborates_feeds"], agrees["a_intel_contradicts_feeds"]) == (1.0, 0.0)
    assert (disagrees["a_intel_corroborates_feeds"], disagrees["a_intel_contradicts_feeds"]) == (0.0, 1.0)
    # The case a single signed axis could not represent at all.
    assert (both["a_intel_corroborates_feeds"], both["a_intel_contradicts_feeds"]) == (1.0, 1.0)
    assert (neither["a_intel_corroborates_feeds"], neither["a_intel_contradicts_feeds"]) == (0.0, 0.0)
    assert len({
        (r["a_intel_corroborates_feeds"], r["a_intel_contradicts_feeds"])
        for r in (neither, agrees, disagrees, both)
    }) == 4


def test_intel_columns_drop_with_component_a(builder) -> None:
    """They are Component A output, so the A-off ablation cell must not carry them."""
    enriched = make_enriched()
    result = intel_result(enriched.finding_id, documents=5, active=True)
    frame = builder.build(
        [enriched], {}, ComponentFlags(a=False, b=True, c=True), {enriched.finding_id: result}
    )
    assert not [name for name in frame.feature_names if name.startswith("a_intel_")]


def test_intel_features_are_finite_and_documented(builder) -> None:
    enriched = make_enriched()
    result = intel_result(enriched.finding_id, documents=3, signals=2, active=True)
    row = builder.row_for(enriched, None, ComponentFlags(), result)

    assert all(math.isfinite(row[name]) for name in INTEL_FEATURES)
    assert all(FEATURE_DOC[name].strip() for name in INTEL_FEATURES)


def test_injection_signals_from_retrieved_pages_are_their_own_feature(builder) -> None:
    """Kept apart from the sandbox's count for the assessment prompts themselves."""
    enriched = make_enriched(signals=4)
    result = intel_result(enriched.finding_id, documents=2, signals=9)
    row = builder.row_for(enriched, None, ComponentFlags(), result)

    assert row["a_injection_signals"] == pytest.approx(4.0)
    assert row["a_intel_injection_signals"] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Missing evidence resolves to the documented neutral, consistently
# ---------------------------------------------------------------------------


def test_a_finding_with_no_intel_takes_the_documented_neutrals(builder) -> None:
    bare = make_enriched(finding=make_finding("f_bare", cwe_id=None, cve_ids=()))
    row = builder.build([bare], {}).X.iloc[0]
    for name in (
        "cvss_base_max",
        "cvss_version_ord",
        "cvss_source_agreement",
        "cvss_ac_low",
        "vuln_age_days",
        "b_epss",
        "b_epss_percentile",
        "b_kev",
        "b_kev_ransomware",
        "b_kev_age_days",
        "b_exploit_count",
        "b_exploit_maturity_feed_ord",
        "b_exploit_verified",
    ):
        assert row[name] == pytest.approx(NEUTRAL[name]), name


def test_absent_evidence_is_never_scored_as_contested(builder) -> None:
    """``cvss_source_agreement`` is the one neutral that is 1.0, matching ``select_cvss``."""
    assert NEUTRAL["cvss_source_agreement"] == pytest.approx(1.0)
    assert all(NEUTRAL[name] == 0.0 for name in ("b_epss", "b_kev", "cvss_base_max"))


def test_a_finding_with_no_chain_score_takes_the_neutral_chain_row(builder) -> None:
    row = builder.build([make_enriched()], {}).X.iloc[0]
    for name in (name for name in FEATURE_NAMES if name.startswith("c_")):
        assert row[name] == pytest.approx(NEUTRAL[name]), name


def test_intel_dated_after_the_as_of_cut_off_is_ignored(builder) -> None:
    """Temporal leakage is the failure mode that silently inflates every result (Gap 4)."""
    future = make_intel(as_of=date(2024, 7, 1), kev_added=date(2024, 6, 20))
    row = builder.build([make_enriched(intel=(future,), as_of=AS_OF)], {}).X.iloc[0]
    assert row["b_kev"] == pytest.approx(0.0)
    assert row["b_epss"] == pytest.approx(0.0)
    assert row["cvss_base_max"] == pytest.approx(0.0)


def test_intel_for_a_different_cve_is_ignored(builder) -> None:
    other = make_intel("CVE-1999-9999", base_scores=(10.0,), epss=0.99)
    row = builder.build([make_enriched(intel=(other,))], {}).X.iloc[0]
    assert row["cvss_base_max"] == pytest.approx(0.0)
    assert row["b_epss"] == pytest.approx(0.0)


def test_neutral_row_projects_onto_the_requested_columns() -> None:
    columns = feature_names_for(ComponentFlags(a=True, b=False, c=False))
    row = neutral_row(columns)
    assert list(row) == columns
    assert row["a_p_applicable"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Determinism and documentation
# ---------------------------------------------------------------------------


def test_building_the_same_input_twice_gives_identical_frames(builder, one) -> None:
    enriched, chain = one
    first = builder.build(enriched, chain)
    second = builder.build(enriched, chain)
    assert first.finding_ids == second.finding_ids
    assert first.group_ids == second.group_ids
    assert (first.to_numpy() == second.to_numpy()).all()


def test_every_feature_has_a_neutral_and_a_documented_derivation() -> None:
    assert set(FEATURE_DOC) == set(FEATURE_NAMES)
    assert set(NEUTRAL) == set(FEATURE_NAMES)
    assert all(FEATURE_DOC[name].strip() for name in FEATURE_NAMES)


def test_row_for_matches_the_built_frame(builder, one) -> None:
    enriched, chain = one
    item = enriched[0]
    row = builder.row_for(item, chain[item.finding_id])
    frame_row = builder.build(enriched, chain).X.iloc[0]
    assert list(row) == list(frame_row.index)
    for name, value in row.items():
        assert value == pytest.approx(float(frame_row[name])), name


def test_component_grouping_matches_the_frozen_spec() -> None:
    """Feature prefixes and ``FEATURE_GROUPS`` must agree, or an ablation silently leaks."""
    for name, group in FEATURE_GROUPS.items():
        if name.startswith("a_"):
            assert group is Component.A, name
        elif name.startswith("b_"):
            assert group is Component.B, name
        elif name.startswith("c_"):
            assert group is Component.C, name
        else:
            assert group is None, name
