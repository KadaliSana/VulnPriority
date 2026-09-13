"""A worked example run, for looking at the site without executing the pipeline.

Useful in two situations: showing someone what the framework produces before they install
anything, and checking the page itself after a change to the front end. The numbers are
constructed, not measured, and the page labels them as a demonstration.

``python -m vulnprio.web --demo --serve``
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from random import Random

from vulnprio.core.money import format_money_compact
from vulnprio.core.enums import (
    ApplicabilityVerdict,
    AttackComplexity,
    CvssVersion,
    EndpointFunction,
    ExploitMaturity,
    ExploitSource,
    FeedMode,
    HttpMethod,
    LLMBackendKind,
    MetricName,
    PrivilegeLevel,
    Provenance,
    RankerName,
    ScannerSeverity,
    ScoreSource,
    SelectionMethod,
    SplitKind,
    TrustTier,
    UserInteraction,
    VersionMatch,
)
from vulnprio.core.models import (
    AblationCell,
    AblationTable,
    AdversarialReport,
    ApplicabilityAssessment,
    AssetCriticality,
    AttackGraphSummary,
    AttackPath,
    BusinessImpact,
    CalibrationReport,
    ChainScore,
    ComponentFlags,
    CvssRecord,
    Endpoint,
    EnrichedFinding,
    EpssRecord,
    ExploitEvidence,
    ExploitLikelihood,
    ExploitabilityAssessment,
    Explanation,
    FeatureContribution,
    Finding,
    GraphEdge,
    GraphNode,
    GroundTruthLabel,
    KevRecord,
    LabelSet,
    ManipulationAlert,
    MetricBundle,
    MetricValue,
    MinorityClassReport,
    RankedFinding,
    RankingResult,
    RemediationCost,
    RunManifest,
    Scan,
    SelectionResult,
    SimulationResult,
    Split,
    TechComponent,
    TrustSummary,
    UntrustedText,
    VulnIntel,
)
from vulnprio.core.enums import DetectorName
from vulnprio.web.exporter import build_dashboard
from vulnprio.web.schema import DashboardData

__all__ = ["demo_dashboard"]

AS_OF = date(2024, 6, 1)
SCANNED = datetime(2024, 5, 20, 9, 30)

# name, path, method, function, cwe, cve, severity, criticality, feasibility, p_exploit,
# impact, hours, kev, epss, maturity, privilege gained, applicable
#: The demonstration runs on the shipped ``default_ecommerce`` preset, so its figures are
#: written in that preset's currency. They illustrate the page; they are not a measurement,
#: which ``data.notes["demonstration"]`` says on the page itself.
DEMO_CURRENCY = "INR"

#: ``ComponentBConfig.remediation_hourly_rate``'s default, so the demo's remediation cost
#: agrees with what a real run would charge for the same hours.
DEMO_HOURLY_RATE = 4_700.0

_SPEC = [
    ("Remote code execution in file upload", "/api/v1/files/upload", "POST", EndpointFunction.FILE_IO, 434,
     "CVE-2024-21762", "critical", 0.82, 0.88, 0.71, 93_900_000, 16.0, True, 0.71, ExploitMaturity.WEAPONIZED, PrivilegeLevel.SYSTEM, True),
    ("SQL injection in order lookup", "/api/v1/orders/{id}", "GET", EndpointFunction.PAYMENT, 89,
     "CVE-2024-0001", "high", 0.93, 0.79, 0.54, 129_000_000, 12.0, True, 0.42, ExploitMaturity.FUNCTIONAL, PrivilegeLevel.USER, True),
    ("Authentication bypass on admin console", "/admin/session", "POST", EndpointFunction.AUTH, 287,
     "CVE-2023-46805", "critical", 0.90, 0.66, 0.38, 82_200_000, 10.0, True, 0.28, ExploitMaturity.FUNCTIONAL, PrivilegeLevel.ADMIN, True),
    ("Server-side request forgery in webhook", "/api/v1/webhooks", "POST", EndpointFunction.API_DATA, 918,
     "CVE-2024-3400", "high", 0.61, 0.58, 0.24, 30_500_000, 8.0, False, 0.19, ExploitMaturity.POC, PrivilegeLevel.USER, True),
    ("Insecure direct object reference on invoices", "/api/v1/invoices/{id}", "GET", EndpointFunction.PII_DATA, 639,
     None, "high", 0.77, 0.52, 0.21, 45_000_000, 6.0, False, None, ExploitMaturity.UNPROVEN, PrivilegeLevel.USER, True),
    ("Path traversal in report export", "/api/v1/reports/export", "GET", EndpointFunction.FILE_IO, 22,
     "CVE-2022-24785", "medium", 0.55, 0.44, 0.16, 16_400_000, 5.0, False, 0.08, ExploitMaturity.POC, PrivilegeLevel.USER, True),
    ("Stored cross-site scripting in profile", "/account/profile", "POST", EndpointFunction.PII_DATA, 79,
     None, "medium", 0.62, 0.36, 0.12, 10_200_000, 4.0, False, None, ExploitMaturity.UNKNOWN, PrivilegeLevel.USER, True),
    ("Deserialization flaw in legacy endpoint", "/legacy/rpc", "POST", EndpointFunction.API_DATA, 502,
     "CVE-2021-44228", "critical", 0.48, 0.30, 0.05, 74_300_000, 20.0, True, 0.94, ExploitMaturity.WEAPONIZED, PrivilegeLevel.SYSTEM, False),
    ("Cross-site request forgery on password change", "/account/password", "POST", EndpointFunction.AUTH, 352,
     None, "medium", 0.70, 0.28, 0.09, 13_300_000, 3.0, False, None, ExploitMaturity.UNKNOWN, PrivilegeLevel.USER, True),
    ("Session cookie missing Secure flag", "/api/login", "POST", EndpointFunction.AUTH, 614,
     None, "low", 0.66, 0.18, 0.06, 7_040_000, 1.5, False, None, ExploitMaturity.UNKNOWN, PrivilegeLevel.NONE, True),
    ("Verbose error discloses stack traces", "/api/v1/search", "GET", EndpointFunction.SEARCH, 209,
     None, "low", 0.24, 0.12, 0.03, 861_000, 2.0, False, None, ExploitMaturity.UNKNOWN, PrivilegeLevel.NONE, True),
    ("Outdated jQuery with known issues", "/static/js/jquery-1.12.4.min.js", "GET", EndpointFunction.STATIC_CONTENT, 1104,
     "CVE-2020-11023", "medium", 0.08, 0.22, 0.04, 548_000, 3.0, False, 0.03, ExploitMaturity.POC, PrivilegeLevel.NONE, True),
    ("Server version disclosure", "/", "GET", EndpointFunction.STATIC_CONTENT, 200,
     None, "low", 0.10, 0.05, 0.02, 157_000, 0.5, False, None, ExploitMaturity.UNKNOWN, PrivilegeLevel.NONE, True),
    ("Directory listing enabled on assets", "/static/", "GET", EndpointFunction.STATIC_CONTENT, 548,
     None, "low", 0.09, 0.07, 0.02, 235_000, 1.0, False, None, ExploitMaturity.UNKNOWN, PrivilegeLevel.NONE, True),
]

HOST = "shop.example.com"
SCAN_ID = "scan_demo_1"
APP_ID = "app_demo"


def _endpoint(index: int, path: str, method: str, function: EndpointFunction) -> Endpoint:
    auth = PrivilegeLevel.ADMIN if path.startswith("/admin") else (
        PrivilegeLevel.USER if path.startswith(("/account", "/api/v1")) else PrivilegeLevel.NONE
    )
    return Endpoint(
        endpoint_id=f"ep_demo_{index}",
        app_id=APP_ID,
        host=HOST,
        url=f"https://{HOST}{path}",
        path=path,
        method=HttpMethod(method),
        auth_required=auth,
        internet_facing=True,
        response_status=200,
        response_content_type="application/json" if path.startswith("/api") else "text/html",
        sets_cookie=function == EndpointFunction.AUTH,
        parameters=("id",) if "{id}" in path else (),
        observed_tech=(TechComponent(vendor="apache", product="struts", version="2.5.12"),),
    )


def _scan(endpoints: list[Endpoint], findings: list[Finding]) -> Scan:
    return Scan(
        scan_id=SCAN_ID,
        app_id=APP_ID,
        app_name="Example Shop",
        sector="ecommerce",
        scanned_at=SCANNED,
        scanner_name="zap",
        scanner_version="2.14.0",
        hosts=(HOST,),
        tech_stack=(TechComponent(vendor="apache", product="struts", version="2.5.12"),),
        endpoints=tuple(endpoints),
        findings=tuple(findings),
    )


def _intel(cve: str, kev: bool, epss: float | None, maturity: ExploitMaturity) -> VulnIntel:
    return VulnIntel(
        cve_id=cve,
        as_of=AS_OF,
        published=date(2024, 1, 10),
        description=UntrustedText(text=f"{cve} description from the national vulnerability database.", provenance=Provenance.NVD),
        cvss=(
            CvssRecord(version=CvssVersion.V31, source=ScoreSource.NVD, base_score=9.8,
                       submetrics={"AV": "N", "AC": "L", "PR": "N", "UI": "N", "C": "H", "I": "H", "A": "H"}),
            CvssRecord(version=CvssVersion.V31, source=ScoreSource.CNA, base_score=8.6),
        ),
        epss=EpssRecord(cve_id=cve, score=epss, percentile=min(0.99, 0.5 + epss / 2), as_of=AS_OF) if epss is not None else None,
        kev=KevRecord(cve_id=cve, in_kev=kev, date_added=date(2024, 2, 1) if kev else None, as_of=AS_OF),
        exploits=(
            ExploitEvidence(source=ExploitSource.EXPLOIT_DB, published=date(2024, 1, 20), maturity=maturity, verified=True),
        ) if maturity >= ExploitMaturity.POC else (),
    )


def demo_dashboard() -> DashboardData:
    """Build a complete, plausible run so the page can be viewed and reviewed."""
    rng = Random(11)
    endpoints: list[Endpoint] = []
    findings: list[Finding] = []
    enriched: list[EnrichedFinding] = []
    chain: dict[str, ChainScore] = {}

    for index, spec in enumerate(_SPEC):
        (name, path, method, function, cwe, cve, severity, criticality, feasibility,
         p_exploit, impact, hours, kev, epss, maturity, gained, applicable) = spec
        endpoint = _endpoint(index, path, method, function)
        endpoints.append(endpoint)
        finding = Finding(
            finding_id=f"f_demo_{index}",
            scan_id=SCAN_ID,
            app_id=APP_ID,
            endpoint_id=endpoint.endpoint_id,
            name=name,
            cwe_id=cwe,
            cve_ids=(cve,) if cve else (),
            scanner="zap",
            scanner_plugin_id=str(40000 + index),
            scanner_severity=ScannerSeverity(severity),
            scanner_confidence=0.8,
            description=UntrustedText(text=f"{name} detected at {path}.", provenance=Provenance.SCANNER_OUTPUT),
            observed_at=SCANNED,
            dedup_key=f"dk_demo_{cwe}",
            cluster_size=3 if cwe == 89 else 1,
        )
        findings.append(finding)

        expected_loss = p_exploit * impact
        enriched.append(
            EnrichedFinding(
                finding=finding,
                endpoint=endpoint,
                intel=(_intel(cve, kev, epss, maturity),) if cve else (),
                asset=AssetCriticality(
                    endpoint_id=endpoint.endpoint_id, function=function, criticality=criticality,
                    data_sensitivity=min(1.0, criticality * 0.9), exposure=1.0 if endpoint.auth_required == PrivilegeLevel.NONE else 0.6,
                    is_admin_surface=function == EndpointFunction.ADMIN, is_auth_boundary=function == EndpointFunction.AUTH,
                    confidence=0.8,
                ),
                exploitability=ExploitabilityAssessment(
                    finding_id=finding.finding_id, exploit_feasibility=feasibility, exploit_maturity=maturity,
                    attack_complexity=AttackComplexity.LOW if feasibility > 0.3 else AttackComplexity.HIGH,
                    user_interaction=UserInteraction.REQUIRED if cwe in (79, 352) else UserInteraction.NONE,
                    privileges_required=endpoint.auth_required, privilege_gained=gained,
                    impact_c=min(1.0, criticality), impact_i=min(1.0, criticality * 0.8), impact_a=0.4,
                    confidence=0.75,
                ),
                applicability=ApplicabilityAssessment(
                    finding_id=finding.finding_id,
                    verdict=ApplicabilityVerdict.APPLICABLE if applicable else ApplicabilityVerdict.NOT_APPLICABLE,
                    p_applicable=0.92 if applicable else 0.04,
                    version_match=VersionMatch.MATCH if applicable else VersionMatch.MISMATCH,
                ),
                likelihood=ExploitLikelihood(
                    finding_id=finding.finding_id, attacker="targeted_criminal", p_exploit=p_exploit,
                    p_exploit_uncapped=p_exploit, horizon_days=90,
                    log_odds_terms={"kev": 2.0 if kev else 0.0, "epss_logit": round((epss or 0.01) * 2, 3),
                                    "feasibility": round(feasibility * 1.8, 3), "applicability": 2.0 if applicable else -2.0},
                ),
                impact=BusinessImpact(
                    finding_id=finding.finding_id, confidentiality=impact * 0.6,
                    integrity=impact * 0.25, availability=impact * 0.1,
                    reputational=impact * 0.05, total=float(impact),
                ),
                remediation=RemediationCost(
                    finding_id=finding.finding_id, hours=hours,
                    cost=hours * DEMO_HOURLY_RATE, basis="cwe class",
                ),
                expected_loss=expected_loss,
                trust=TrustSummary(
                    max_tier_used=TrustTier.REFERENCE_PAGE if cve else TrustTier.SCANNER,
                    injection_signal_count=2 if index == 3 else 0,
                    corroborated=kev,
                    floor_p_exploit=0.3 if kev else 0.0,
                ),
                alerts=(
                    ManipulationAlert(
                        finding_id=finding.finding_id, detector=DetectorName.INFLUENCE_BUDGET, severity=0.6,
                        message="Model-derived features account for 41% of this finding's attribution, above the 35% cap.",
                    ),
                ) if index == 3 else (),
                as_of=AS_OF,
            )
        )

        if gained != PrivilegeLevel.NONE and applicable:
            delta = expected_loss * (0.9 if gained == PrivilegeLevel.SYSTEM else 0.35)
            chain[finding.finding_id] = ChainScore(
                finding_id=finding.finding_id,
                reach_delta=round(delta, 2),
                max_path_prob_to_target=min(0.95, p_exploit + 0.1),
                n_paths_through=rng.randint(1, 5),
                betweenness=round(rng.uniform(0.05, 0.6), 3),
                hops_from_entry=1 if endpoint.auth_required == PrivilegeLevel.NONE else 2,
                privilege_gain=int(gained),
                is_chokepoint=gained == PrivilegeLevel.SYSTEM,
                best_target=f"state:{HOST}:SYSTEM",
            )

    scan = _scan(endpoints, findings)

    # Learned ordering: chain-adjusted expected loss, which is what the framework optimises.
    order = sorted(
        enriched,
        key=lambda item: -(item.expected_loss + chain.get(item.finding_id, ChainScore(finding_id="x")).reach_delta),
    )
    ranking = RankingResult(
        ranker=RankerName.LAMBDAMART, flags=ComponentFlags(), seed=42, config_hash="demo",
        items=tuple(
            RankedFinding(
                finding_id=item.finding_id, scan_id=SCAN_ID, rank=position + 1,
                score=round(4.0 - position * 0.25, 3),
                expected_loss=item.expected_loss,
                chain_adjusted_loss=item.expected_loss + chain.get(item.finding_id, ChainScore(finding_id="x")).reach_delta,
                p_exploit=item.likelihood.p_exploit,
                explanation=Explanation(
                    finding_id=item.finding_id,
                    base_value=0.4,
                    top_contributions=tuple(
                        FeatureContribution(feature=feature, value=value, shap_value=shap, tier=tier)
                        for feature, value, shap, tier in (
                            ("b_kev", 1.0 if item.intel and item.intel[0].kev and item.intel[0].kev.in_kev else 0.0,
                             0.9 if item.intel and item.intel[0].kev and item.intel[0].kev.in_kev else -0.1, TrustTier.CURATED_FEED),
                            ("b_expected_loss_log", round(item.expected_loss / 1e6, 3), 0.7, TrustTier.CURATED_FEED),
                            ("c_reach_delta_log", round(chain.get(item.finding_id, ChainScore(finding_id="x")).reach_delta / 1e6, 3),
                             0.55 if item.finding_id in chain else 0.0, TrustTier.SCANNER),
                            ("a_exploit_feasibility", item.exploitability.exploit_feasibility, 0.34, TrustTier.REFERENCE_PAGE),
                            ("a_p_applicable", item.applicability.p_applicable,
                             0.4 if item.applicability.verdict == ApplicabilityVerdict.APPLICABLE else -0.85, TrustTier.REFERENCE_PAGE),
                            ("a_asset_criticality", item.asset.criticality, 0.28, TrustTier.SCANNER),
                        )
                    ),
                    reason_codes=_reasons(item, chain),
                    untrusted_influence_share=0.41 if item.trust.injection_signal_count else round(0.1 + item.exploitability.exploit_feasibility * 0.2, 2),
                ),
                alerts=item.alerts,
            )
            for position, item in enumerate(order)
        ),
    )

    cvss_order = sorted(enriched, key=lambda item: -_cvss_of(item))
    baselines = {
        "cvss_only": RankingResult(
            ranker=RankerName.CVSS_ONLY,
            items=tuple(
                RankedFinding(finding_id=item.finding_id, scan_id=SCAN_ID, rank=position + 1, score=_cvss_of(item))
                for position, item in enumerate(cvss_order)
            ),
        ),
        "epss_only": RankingResult(
            ranker=RankerName.EPSS_ONLY,
            items=tuple(
                RankedFinding(finding_id=item.finding_id, scan_id=SCAN_ID, rank=position + 1, score=_epss_of(item))
                for position, item in enumerate(sorted(enriched, key=lambda i: -_epss_of(i)))
            ),
        ),
    }

    graph = _graph(enriched, chain)
    split = Split(
        kind=SplitKind.TIME_ORDERED, fold=0, train_scan_ids=("scan_demo_0",), test_scan_ids=(SCAN_ID,),
        train_end=date(2024, 3, 20), test_start=date(2024, 5, 20), gap_days=30,
    )
    metrics = _metrics(split)
    ablation = _ablation()
    selections = _selections(enriched, chain, order, cvss_order)
    simulations = _simulations()
    adversarial = AdversarialReport(
        backend=LLMBackendKind.HEURISTIC, corpus_version="v1", n_cases=84,
        attack_success_rate=0.0, canary_leak_rate=0.0, detection_rate=0.92, false_positive_rate=0.05,
        mean_abs_rank_shift=0.36, max_abs_rank_shift=1,
        per_category={
            "instruction_override": {"n": 15.0, "attack_success_rate": 0.0, "detection_rate": 1.0},
            "role_hijack": {"n": 10.0, "attack_success_rate": 0.0, "detection_rate": 1.0},
            "schema_smuggling": {"n": 11.0, "attack_success_rate": 0.0, "detection_rate": 0.91},
            "canary_exfil": {"n": 8.0, "attack_success_rate": 0.0, "detection_rate": 1.0},
            "multilingual": {"n": 20.0, "attack_success_rate": 0.0, "detection_rate": 0.85},
            "fake_evidence_deflate": {"n": 6.0, "attack_success_rate": 0.0, "detection_rate": 0.83},
            "benign_control": {"n": 20.0, "attack_success_rate": 0.0, "detection_rate": 0.05},
        },
    )
    labels = LabelSet(
        observation_cutoff=AS_OF,
        labels=tuple(
            GroundTruthLabel(
                finding_id=item.finding_id,
                cve_id=item.finding.cve_ids[0] if item.finding.cve_ids else None,
                exploited=bool(item.intel and item.intel[0].kev and item.intel[0].kev.in_kev and item.applicability.verdict == ApplicabilityVerdict.APPLICABLE),
                relevance_grade=4 if (item.intel and item.intel[0].kev and item.intel[0].kev.in_kev and item.applicability.verdict == ApplicabilityVerdict.APPLICABLE) else 0,
            )
            for item in enriched
        ),
    )
    manifest = RunManifest(
        run_id="demo", created_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
        config_hash="demo000000000000", seeds=(42, 43, 44), dataset_hash="demo111111111111",
        llm_backend=LLMBackendKind.HEURISTIC, llm_model="heuristic", feed_mode=FeedMode.OFFLINE,
        as_of=AS_OF, command="python -m vulnprio.web --demo",
    )

    data = build_dashboard(
        scans=[scan], enriched=enriched, chain=chain, graphs=[graph], ranking=ranking,
        baseline_rankings=baselines, metrics=metrics, ablation=ablation, selections=selections,
        simulations=simulations, adversarial=adversarial, labels=labels, manifest=manifest,
    )
    data.meta.attacker = "targeted_criminal"
    data.meta.impact_model = "default_ecommerce"
    data.meta.currency = DEMO_CURRENCY
    data.meta.components = {"a": True, "b": True, "c": True}
    data.notes["demonstration"] = (
        "These figures illustrate the page. They come from a constructed example, not a measured run."
    )
    return data


def _cvss_of(item: EnrichedFinding) -> float:
    if item.intel and item.intel[0].cvss:
        return max(record.base_score for record in item.intel[0].cvss)
    return {"critical": 9.1, "high": 7.5, "medium": 5.3, "low": 3.1, "info": 0.0}.get(
        item.finding.scanner_severity.value, 4.0)


def _epss_of(item: EnrichedFinding) -> float:
    if item.intel and item.intel[0].epss:
        return item.intel[0].epss.score
    return 0.0


def _reasons(item: EnrichedFinding, chain: dict[str, ChainScore]) -> tuple[str, ...]:
    out: list[str] = []
    intel = item.intel[0] if item.intel else None
    if intel and intel.kev and intel.kev.in_kev:
        out.append(f"KEV-listed since {intel.kev.date_added}: confirmed exploited in the wild.")
    if intel and intel.epss and intel.epss.score >= 0.1:
        out.append(f"EPSS {intel.epss.score:.2f}, in the {intel.epss.percentile:.0%} percentile of near-term exploitation.")
    if item.applicability.verdict == ApplicabilityVerdict.NOT_APPLICABLE:
        out.append("Version evidence says the observed deployment is not affected, so it is demoted despite its severity.")
    score = chain.get(item.finding_id)
    if score and score.is_chokepoint:
        out.append(
            "Chain chokepoint: patching it removes "
            f"{format_money_compact(score.reach_delta, DEMO_CURRENCY)} of reachable risk."
        )
    elif score and score.reach_delta > 0:
        out.append(
            f"Contributes {format_money_compact(score.reach_delta, DEMO_CURRENCY)} of "
            "reachable compromise from the entry state."
        )
    out.append(
        f"{item.asset.function.value.replace('_', ' ')} endpoint, criticality {item.asset.criticality:.2f}, "
        f"business impact {format_money_compact(item.impact.total, DEMO_CURRENCY)}."
    )
    return tuple(out)


def _graph(enriched: list[EnrichedFinding], chain: dict[str, ChainScore]) -> AttackGraphSummary:
    nodes = [
        GraphNode(node_id="state:internet:NONE", asset="internet", privilege=PrivilegeLevel.NONE, is_entry=True),
        GraphNode(node_id=f"state:{HOST}:USER", asset=HOST, privilege=PrivilegeLevel.USER, value=420_000.0),
        GraphNode(node_id=f"state:{HOST}:ADMIN", asset=HOST, privilege=PrivilegeLevel.ADMIN, value=1_600_000.0, is_target=True),
        GraphNode(node_id=f"state:{HOST}:SYSTEM", asset=HOST, privilege=PrivilegeLevel.SYSTEM, value=3_300_000.0, is_target=True),
        GraphNode(node_id="state:db.internal:USER", asset="db.internal", privilege=PrivilegeLevel.USER, value=2_100_000.0, is_target=True),
    ]
    edges: list[GraphEdge] = []
    for item in enriched:
        score = chain.get(item.finding_id)
        if score is None:
            continue
        src = "state:internet:NONE" if item.endpoint.auth_required == PrivilegeLevel.NONE else f"state:{HOST}:USER"
        dst = f"state:{HOST}:{item.exploitability.privilege_gained.name}"
        edges.append(GraphEdge(src=src, dst=dst, probability=round(item.likelihood.p_exploit * item.applicability.p_applicable, 3),
                               finding_id=item.finding_id, kind="exploit", tier=TrustTier.SCANNER))
    edges.append(GraphEdge(src=f"state:{HOST}:SYSTEM", dst=f"state:{HOST}:ADMIN", probability=1.0, kind="privilege_implication", tier=TrustTier.OPERATOR))
    edges.append(GraphEdge(src=f"state:{HOST}:ADMIN", dst=f"state:{HOST}:USER", probability=1.0, kind="privilege_implication", tier=TrustTier.OPERATOR))
    edges.append(GraphEdge(src=f"state:{HOST}:SYSTEM", dst="state:db.internal:USER", probability=0.62, kind="lateral", tier=TrustTier.SCANNER))

    top = sorted(chain.values(), key=lambda s: -s.reach_delta)[:3]
    paths = [
        AttackPath(
            nodes=("state:internet:NONE", f"state:{HOST}:SYSTEM", "state:db.internal:USER"),
            finding_ids=(top[0].finding_id,) if top else (),
            probability=0.44, target_value=2_100_000.0,
        ),
        AttackPath(
            nodes=("state:internet:NONE", f"state:{HOST}:USER", f"state:{HOST}:ADMIN"),
            finding_ids=tuple(s.finding_id for s in top[1:3]),
            probability=0.21, target_value=1_600_000.0,
        ),
    ]
    return AttackGraphSummary(
        scan_id=SCAN_ID, nodes=tuple(nodes), edges=tuple(edges),
        entry_node="state:internet:NONE",
        target_nodes=(f"state:{HOST}:ADMIN", f"state:{HOST}:SYSTEM", "state:db.internal:USER"),
        total_risk=2_284_000.0, top_paths=tuple(paths),
        monotone_verified=True, rejected_untrusted_edges=3,
    )


def _metrics(split: Split) -> list[MetricBundle]:
    table = {
        RankerName.LAMBDAMART: {"ndcg": (0.834, 0.781, 0.879), "precision": 0.70, "risk": 0.86, "mrr": 0.83},
        RankerName.EXPECTED_LOSS: {"ndcg": (0.802, 0.744, 0.851), "precision": 0.65, "risk": 0.84, "mrr": 0.79},
        RankerName.VMC_CHAIN: {"ndcg": (0.671, 0.602, 0.735), "precision": 0.52, "risk": 0.61, "mrr": 0.66},
        RankerName.KEV_FIRST: {"ndcg": (0.648, 0.576, 0.716), "precision": 0.50, "risk": 0.58, "mrr": 0.64},
        RankerName.EPSS_ONLY: {"ndcg": (0.596, 0.522, 0.664), "precision": 0.44, "risk": 0.49, "mrr": 0.58},
        RankerName.CVSS_ONLY: {"ndcg": (0.512, 0.441, 0.583), "precision": 0.35, "risk": 0.38, "mrr": 0.49},
        RankerName.SCANNER_SEVERITY: {"ndcg": (0.487, 0.415, 0.560), "precision": 0.32, "risk": 0.34, "mrr": 0.47},
        RankerName.RANDOM: {"ndcg": (0.281, 0.210, 0.355), "precision": 0.14, "risk": 0.12, "mrr": 0.26},
    }
    bundles: list[MetricBundle] = []
    for ranker, row in table.items():
        ndcg, low, high = row["ndcg"]
        values = (
            MetricValue(name=MetricName.NDCG_AT_K, k=10, value=ndcg, ci_low=low, ci_high=high, n=14),
            MetricValue(name=MetricName.PRECISION_AT_K, k=10, value=row["precision"], n=14),
            MetricValue(name=MetricName.RISK_CAPTURE_AT_K, k=10, value=row["risk"], n=14),
            MetricValue(name=MetricName.MRR, value=row["mrr"], n=14),
        )
        bundle = MetricBundle(ranker=ranker, split=split, seed=42, values=values, runtime_seconds=1.2)
        if ranker == RankerName.LAMBDAMART:
            bundle = bundle.model_copy(update={
                "calibration": CalibrationReport(
                    brier=0.071, ece=0.038, n_bins=8,
                    bin_confidence=(0.05, 0.15, 0.27, 0.38, 0.5, 0.63, 0.75, 0.9),
                    bin_accuracy=(0.03, 0.12, 0.3, 0.35, 0.54, 0.6, 0.79, 0.86),
                    bin_count=(31, 22, 15, 11, 9, 7, 5, 4),
                ),
                "minority": MinorityClassReport(
                    positive_rate=0.21, mcc=0.612, f1_positive=0.68, balanced_accuracy=0.81, threshold=0.5,
                    per_class={
                        "maturity=WEAPONIZED": {"f1": 0.71, "support": 7.0},
                        "maturity=FUNCTIONAL": {"f1": 0.64, "support": 12.0},
                        "complexity=HIGH": {"f1": 0.41, "support": 5.0},
                        "privileges=ADMIN": {"f1": 0.55, "support": 6.0},
                    },
                ),
            })
        bundles.append(bundle)
    return bundles


def _ablation() -> AblationTable:
    means = {
        "ABC": 0.834, "AB": 0.795, "AC": 0.742, "A": 0.667,
        "BC": 0.808, "B": 0.771, "C": 0.596, "none": 0.503,
    }
    cells = []
    for flags in ComponentFlags.all_cells():
        label = flags.label()
        cells.append(AblationCell(flags=flags, seeds=(42, 43, 44), n=3,
                                  mean={"ndcg@10": means[label]}, std={"ndcg@10": 0.018}))
    return AblationTable(
        cells=tuple(cells),
        main_effects={"ndcg@10": {"A": 0.0405, "B": 0.1478, "C": 0.0343}},
        interactions={"ndcg@10": {"AB": -0.0123, "AC": 0.0071, "BC": -0.0038, "ABC": 0.0016}},
        paired_ci={"ndcg@10": {"A": (0.019, 0.062), "B": (0.121, 0.174), "C": (0.014, 0.055)}},
    )


def _selections(enriched, chain, learned_order, cvss_order) -> list[SelectionResult]:
    budget = 24.0

    def take(order):
        hours, ids, captured = 0.0, [], 0.0
        for item in order:
            if hours + item.remediation.hours > budget:
                continue
            hours += item.remediation.hours
            ids.append(item.finding_id)
            captured += item.expected_loss + chain.get(item.finding_id, ChainScore(finding_id="x")).reach_delta
        return hours, ids, captured

    total = sum(i.expected_loss + chain.get(i.finding_id, ChainScore(finding_id="x")).reach_delta for i in enriched)
    out = []
    for ranker, order, method in (
        (RankerName.LAMBDAMART, learned_order, SelectionMethod.DP_EXACT),
        (RankerName.CVSS_ONLY, cvss_order, SelectionMethod.RANK_PREFIX),
        (RankerName.EPSS_ONLY, sorted(enriched, key=lambda i: -_epss_of(i)), SelectionMethod.RANK_PREFIX),
    ):
        hours, ids, captured = take(order)
        out.append(SelectionResult(
            scan_id=SCAN_ID, ranker=ranker, method=method, budget_hours=budget,
            selected_ids=tuple(ids), total_hours=hours, risk_captured=captured,
            risk_capture_fraction=min(1.0, captured / total if total else 0.0),
            exploited_captured=sum(1 for i in ids if i in {e.finding_id for e in enriched[:3]}),
            exploited_total=3,
        ))
    return out


def _simulations() -> list[SimulationResult]:
    weeks = 26

    def curve(rate: float) -> tuple[float, ...]:
        total, out = 0.0, []
        for week in range(weeks):
            total += rate * (1.0 - week / (weeks * 1.6))
            out.append(round(total, 1))
        return tuple(out)

    rows = [
        (RankerName.LAMBDAMART, 42.0, 118.0, 0.41),
        (RankerName.EXPECTED_LOSS, 46.0, 141.0, 0.35),
        (RankerName.KEV_FIRST, 58.0, 233.0, 0.18),
        (RankerName.EPSS_ONLY, 63.0, 287.0, 0.11),
        (RankerName.CVSS_ONLY, 71.0, 364.0, None),
    ]
    out = []
    for policy, rate, exploited_days, reduction in rows:
        values = curve(rate)
        out.append(SimulationResult(
            policy=policy, weeks=weeks, capacity_hours_per_week=20.0,
            exposure_days_total=values[-1], exposure_days_exploited=exploited_days,
            expected_loss_days=values[-1] * 4200.0,
            exploited_remediated_before_exploit=3 if policy == RankerName.LAMBDAMART else (2 if reduction and reduction > 0.3 else 1),
            exploited_total=3, weekly_cumulative_exposure=values, reduction_vs_cvss=reduction,
        ))
    return out
