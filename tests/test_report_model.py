"""The assessment report: structure, content discipline and the remediation knowledge base.

These tests build a :class:`~vulnprio.web.schema.DashboardData` directly, so they hold
whether or not the pipeline, the demo payload or the website exist. What they are really
checking is the report's three promises: every section appears when the data is there,
no section appears when it is not, and no number appears that the payload did not supply.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vulnprio.core.money import (
    DEFAULT_CURRENCY,
    currency_symbol,
    format_money,
    format_money_compact,
)
from vulnprio.report import (
    GENERIC_REMEDIATION,
    REMEDIATIONS,
    AssessmentReport,
    ReportOptions,
    build_report,
    build_report_model,
    guidance_for,
    known_cwes,
)
from vulnprio.report.model import RESEARCH_SECTIONS
from vulnprio.web.schema import (
    DashboardData,
    WebAdversarial,
    WebContribution,
    WebFinding,
    WebGapRow,
    WebGraph,
    WebMeta,
    WebPath,
    WebScan,
    WebSelection,
    WebSummary,
)

#: Every payload in this module is built from a default ``WebMeta``, so this is what
#: its money figures are denominated in and what the report must format them as.
CURRENCY = DEFAULT_CURRENCY

GENERATED_AT = datetime(2024, 6, 2, 8, 30, tzinfo=timezone.utc)

#: Every CWE the brief requires the knowledge base to cover.
REQUIRED_CWES = (
    79, 89, 22, 78, 94, 502, 918, 287, 306, 352, 434, 611, 639, 200, 209, 269,
    614, 693, 798, 863, 1004, 1104, 1275, 548, 598, 601, 650, 942, 16, 319,
)

ALL_SECTIONS = (
    "header",
    "executive-summary",
    "fix-first",
    "remediation-plan",
    "attack-chains",
    "finding-detail",
    "methodology",
    "assurance",
    "appendix",
)


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def make_finding(index: int, **overrides) -> WebFinding:
    """One finding with plausible, internally consistent numbers."""
    p_exploit = round(0.62 - index * 0.04, 4)
    impact = 900_000.0 - index * 50_000.0
    expected = round(p_exploit * impact, 2)
    chain_delta = round(expected * 0.5, 2) if index < 3 else 0.0
    finding = WebFinding(
        finding_id=f"f{index}",
        scan_id="scan_1",
        app_id="app1",
        name=f"Finding {index}",
        cwe_id=[89, 79, 434, 918, 22, 287][index % 6],
        cve_ids=[f"CVE-2024-{1000 + index}"],
        endpoint_path=f"/api/v1/resource{index}",
        endpoint_method="POST",
        endpoint_function="api_data",
        auth_required=index % 3,
        cluster_size=1 + (index % 2),
        scanner_severity=["critical", "high", "medium", "low"][index % 4],
        cvss_base=9.8 - index * 0.3,
        cvss_version="3.1",
        cvss_source_agreement=0.62 if index == 0 else 1.0,
        epss=round(0.5 - index * 0.03, 4),
        epss_percentile=round(0.9 - index * 0.02, 4),
        kev=index < 2,
        kev_ransomware=index == 0,
        exploit_maturity="weaponized" if index == 0 else "functional",
        exploit_count=2 if index < 3 else 0,
        criticality=0.8,
        data_sensitivity=0.7,
        exposure=0.6,
        exploit_feasibility=0.75,
        applicability="applicable",
        p_applicable=0.9,
        version_match="match",
        p_exploit=p_exploit,
        impact=impact,
        expected_loss=expected,
        chain_delta=chain_delta,
        chain_adjusted=expected + chain_delta,
        is_chokepoint=index == 0,
        hops_from_entry=1 + index % 3,
        remediation_hours=2.0 + index,
        rank=index + 1,
        score=5.0 - index * 0.2,
        reason_codes=[
            f"REASON-{index}-A: expected loss drives this position.",
            f"REASON-{index}-B: catalogue membership recorded.",
        ],
        contributions=[
            WebContribution(feature="b_kev", value=1.0, shap=0.8, group="B", tier=1),
            WebContribution(feature="a_exposure", value=0.6, shap=0.3, group="A", tier=3),
        ],
        untrusted_influence_share=0.2,
        max_tier_used=3,
        injection_signals=1 if index == 0 else 0,
        exploited=index < 2,
        relevance=4 if index < 2 else 0,
        selected_in_budget=False,
    )
    return finding.model_copy(update=overrides) if overrides else finding


def findings_only_payload(count: int = 6) -> DashboardData:
    """A run that ingested, enriched and ranked one scan. No evaluation of any kind."""
    findings = [make_finding(index) for index in range(count)]
    total = sum(item.expected_loss for item in findings)
    losses = sorted((item.expected_loss for item in findings), reverse=True)
    top_decile = sum(losses[: max(1, len(losses) // 10)]) / total if total else 0.0
    return DashboardData(
        meta=WebMeta(
            run_id="run_partial",
            generated_at=GENERATED_AT,
            config_hash="cfg0123456789",
            package_version="0.1.0",
            llm_backend="heuristic",
            llm_model="heuristic",
            feed_mode="offline",
            attacker="opportunistic",
            impact_model="default_ecommerce",
            as_of="2024-06-01",
            components={"a": True, "b": True, "c": False},
        ),
        summary=WebSummary(
            n_apps=1,
            n_scans=1,
            n_endpoints=11,
            n_findings=len(findings),
            n_kev=sum(1 for item in findings if item.kev),
            n_exploited=sum(1 for item in findings if item.exploited),
            total_expected_loss=total,
            total_impact=sum(item.impact for item in findings),
            total_remediation_hours=sum(item.remediation_hours for item in findings),
            top_decile_loss_share=top_decile,
        ),
        scans=[
            WebScan(
                scan_id="scan_1",
                app_id="app1",
                app_name="Example Shop",
                sector="ecommerce",
                scanned_at="2024-05-20T09:30:00",
                scanner="zap",
                n_endpoints=11,
                n_findings=len(findings),
                total_expected_loss=total,
            )
        ],
        findings=findings,
    )


APPLICATIONS = (
    ("app1", "Northwind Commerce"),
    ("app2", "Carelink Records"),
    ("app3", "Opsgrid Platform"),
)
SCAN_DATES = ("2024-03-15T09:00:00", "2024-04-14T09:00:00", "2024-05-14T09:00:00")


def multi_scan_payload(
    *, apps: int = 3, scans_per_app: int = 3, findings_per_scan: int = 4, dated: bool = True
) -> DashboardData:
    """A research-shaped run: the same applications scanned repeatedly over time.

    This is what the pipeline actually produces, and it is the shape that breaks a report
    written for one scan: ``rank`` repeats once per scan, the same weakness appears in every
    application, and a finding nobody fixed is present in every scan of its application.
    """
    scans: list[WebScan] = []
    findings: list[WebFinding] = []
    for app_index, (app_id, app_name) in enumerate(APPLICATIONS[:apps]):
        for scan_index in range(scans_per_app):
            scan_id = f"scan_{app_id}_{scan_index}"
            scans.append(
                WebScan(
                    scan_id=scan_id,
                    app_id=app_id,
                    app_name=app_name,
                    sector="ecommerce",
                    scanned_at=SCAN_DATES[scan_index] if dated else "",
                    scanner="zap",
                    n_endpoints=18,
                    n_findings=findings_per_scan,
                )
            )
            for rank in range(1, findings_per_scan + 1):
                loss = 900_000.0 - app_index * 7_000 - rank * 30_000 - scan_index * 100
                findings.append(
                    make_finding(
                        rank - 1,
                        finding_id=f"f_{app_id}_{scan_index}_{rank}",
                        scan_id=scan_id,
                        app_id=app_id,
                        name=f"Weakness {rank}",
                        rank=rank,
                        expected_loss=loss,
                        chain_adjusted=loss,
                        chain_delta=0.0,
                        selected_in_budget=False,
                    )
                )
    return DashboardData(
        meta=WebMeta(
            run_id="run_multi",
            generated_at=GENERATED_AT,
            config_hash="cfg_multi",
            package_version="0.1.0",
            llm_backend="heuristic",
            feed_mode="offline",
            attacker="opportunistic",
            impact_model="default_ecommerce",
            as_of="2024-06-01",
            components={"a": True, "b": True, "c": True},
        ),
        # Deliberately the totals over EVERY scan, as the exporter computes them.
        summary=WebSummary(
            n_apps=apps,
            n_scans=len(scans),
            n_endpoints=sum(scan.n_endpoints for scan in scans),
            n_findings=len(findings),
            n_kev=sum(1 for item in findings if item.kev),
            total_expected_loss=sum(item.expected_loss for item in findings),
            total_impact=sum(item.impact for item in findings),
            total_remediation_hours=sum(item.remediation_hours for item in findings),
            top_decile_loss_share=0.12,
        ),
        scans=scans,
        findings=findings,
    )


def full_payload(count: int = 6) -> DashboardData:
    """Everything a complete run produces, so every section has something to say."""
    data = findings_only_payload(count)
    data.meta.components = {"a": True, "b": True, "c": True}
    data.graphs = [
        WebGraph(
            scan_id="scan_1",
            entry_node="state:internet:NONE",
            target_nodes=["state:shop.example.com:ADMIN", "state:db.internal:USER"],
            total_risk=1_250_000.0,
            monotone_verified=True,
            rejected_untrusted_edges=2,
            top_paths=[
                WebPath(
                    nodes=[
                        "state:internet:NONE",
                        "state:shop.example.com:SYSTEM",
                        "state:db.internal:USER",
                    ],
                    finding_ids=["f0"],
                    probability=0.4,
                    target_value=800_000.0,
                    expected_value=320_000.0,
                ),
                WebPath(
                    nodes=[
                        "state:internet:NONE",
                        "state:shop.example.com:USER",
                        "state:shop.example.com:ADMIN",
                    ],
                    finding_ids=["f1", "f2"],
                    probability=0.2,
                    target_value=500_000.0,
                    expected_value=100_000.0,
                ),
            ],
        )
    ]
    for item in data.findings[:3]:
        item.selected_in_budget = True
    selected = [item.finding_id for item in data.findings if item.selected_in_budget]
    data.selections = [
        WebSelection(
            scan_id="scan_1",
            ranker="lambdamart",
            method="dp_exact",
            budget_hours=12.0,
            n_selected=len(selected),
            total_hours=sum(
                item.remediation_hours for item in data.findings if item.finding_id in selected
            ),
            risk_captured=1_100_000.0,
            risk_capture_fraction=0.58,
            exploited_captured=2,
            exploited_total=2,
            selected_ids=selected,
        )
    ]
    data.adversarial = WebAdversarial(
        backend="heuristic",
        corpus_version="v1",
        n_cases=84,
        attack_success_rate=0.0,
        canary_leak_rate=0.0,
        detection_rate=0.93,
        false_positive_rate=0.05,
        mean_abs_rank_shift=0.4,
        max_abs_rank_shift=1,
    )
    data.gaps = [
        WebGapRow(
            gap_id="Gap 1",
            title="No validated construct definition of priority.",
            mitigation="Priority is expected loss in money.",
            modules=["decision/expected_loss.py"],
            tests=["tests/test_decision_expected_loss.py"],
            evidence="Expected loss computed for every finding.",
        )
    ]
    data.notes = {"caveat": "Constructed for a test."}
    return data


# ---------------------------------------------------------------------------
# Sections appear, and are omitted
# ---------------------------------------------------------------------------


def test_full_payload_produces_every_section() -> None:
    report = build_report_model(full_payload())
    present = [section.id for section in report.sections]
    assert present == list(ALL_SECTIONS)
    for section in report.sections:
        assert not section.is_empty


def test_full_payload_reaches_every_required_subsection() -> None:
    report = build_report_model(full_payload())
    for section_id in (
        "scope",
        "residual-risk",
        "evidence-weighting",
        "adversarial",
        "run-notes",
        "appendix-findings",
    ):
        assert report.has(section_id), f"missing subsection {section_id}"


def test_partial_payload_omits_the_sections_with_no_data() -> None:
    report = build_report_model(findings_only_payload())
    present = [section.id for section in report.sections]

    assert "remediation-plan" in report.omitted_sections
    assert "attack-chains" in report.omitted_sections
    assert "remediation-plan" not in present
    assert "attack-chains" not in present

    # What a findings-only run can still say, it says.
    assert {"header", "executive-summary", "fix-first", "finding-detail", "methodology"} <= set(
        present
    )
    # An omitted section leaves no heading behind.
    markdown = build_report(findings_only_payload())
    assert "The remediation plan" not in markdown
    assert "Attack chains" not in markdown
    # ...and it says so rather than printing an empty number.
    assert "No remediation budget was configured" in markdown


def test_partial_payload_omits_traceability_but_keeps_the_finding_list() -> None:
    report = build_report_model(findings_only_payload())
    assert report.has("appendix-findings")
    assert not report.has("appendix-traceability")


def test_empty_run_still_renders() -> None:
    data = DashboardData()
    report = build_report_model(data)

    assert report.sections, "an empty run must still produce a document"
    assert report.has("header")
    assert report.has("executive-summary")
    assert "fix-first" in report.omitted_sections
    assert "finding-detail" in report.omitted_sections
    assert "appendix" in report.omitted_sections

    markdown = build_report(data)
    assert "The scan recorded no findings" in markdown
    assert markdown.strip()
    assert build_report(data, "html").strip()
    AssessmentReport.model_validate_json(build_report(data, "json"))


def test_unknown_format_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown report format"):
        build_report(full_payload(), "pdf")


def test_accepts_the_dictionary_form_of_the_payload() -> None:
    data = full_payload()
    from_model = build_report(data)
    from_dict = build_report(data.model_dump(mode="json"))
    assert from_model == from_dict


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------


def test_executive_summary_numbers_match_the_payload() -> None:
    data = full_payload()
    report = build_report_model(data)
    summary = report.section("executive-summary")
    assert summary is not None
    text = summary.text()

    assert report.numbers["n_findings"] == data.summary.n_findings
    assert report.numbers["total_expected_loss"] == pytest.approx(
        data.summary.total_expected_loss
    )
    assert report.numbers["n_kev"] == data.summary.n_kev

    # The prose carries the same figures, formatted.
    assert f"{data.summary.n_findings:,} issues" in text
    assert format_money_compact(data.summary.total_expected_loss, CURRENCY) in text
    assert f"{data.summary.top_decile_loss_share * 100:.0f}%" in text

    # Residual risk is the payload's own arithmetic, not an invention.
    selected = {item for item in data.selections[0].selected_ids}
    retired = sum(f.expected_loss for f in data.findings if f.finding_id in selected)
    deferred = sum(f.expected_loss for f in data.findings if f.finding_id not in selected)
    assert report.numbers["retired_expected_loss"] == pytest.approx(retired)
    assert report.numbers["deferred_expected_loss"] == pytest.approx(deferred)
    assert format_money_compact(retired, CURRENCY) in text
    assert format_money_compact(deferred, CURRENCY) in text


def test_headline_numbers_describe_the_rows_the_report_prints() -> None:
    """A stale summary block never wins over the findings the document actually lists."""
    data = full_payload(4)
    data.summary.n_findings = 999
    data.summary.total_expected_loss = 42.0
    data.summary.n_apps = 7

    report = build_report_model(data)
    assert report.numbers["n_findings"] == 4
    assert report.numbers["total_expected_loss"] == pytest.approx(
        sum(item.expected_loss for item in data.findings)
    )
    assert "999" not in report.section("executive-summary").text()
    # And the derived counts stay consistent with each other.
    assert report.title == "Security assessment: Example Shop"


def test_top_table_rows_match_the_findings() -> None:
    data = full_payload()
    report = build_report_model(data)
    section = report.section("fix-first")
    assert section is not None
    table = section.tables[0]
    assert table.columns == [
        "Rank",
        "Finding",
        "Endpoint",
        "Expected loss",
        "P(exploit)",
        "Remediation",
    ]
    first = data.findings[0]
    assert table.rows[0][0] == str(first.rank)
    assert table.rows[0][1] == first.name
    assert table.rows[0][3] == format_money(first.expected_loss, CURRENCY)


def test_nothing_is_invented_when_evidence_is_absent() -> None:
    bare = WebFinding(
        finding_id="bare",
        scan_id="scan_1",
        name="Bare finding",
        endpoint_path="/thing",
        endpoint_method="GET",
        expected_loss=10.0,
        impact=100.0,
        p_exploit=0.1,
        remediation_hours=1.0,
        rank=1,
    )
    data = DashboardData(findings=[bare])
    text = build_report_model(data).text()

    assert "No CVSS base score was carried for this finding." in text
    assert "No EPSS forecast was available for this finding." in text
    assert "not listed in the CISA Known Exploited Vulnerabilities catalogue" in text
    assert "Exploit maturity is unknown" in text
    assert "Applicability was not assessed" in text
    # No manipulation testing means it says so, rather than printing a rate of zero.
    assert "was not measured in this run" in text
    assert "no figure for them is reported here" in text


def test_missing_values_never_print_as_zero() -> None:
    data = DashboardData(
        meta=WebMeta(),
        findings=[
            WebFinding(
                finding_id="x",
                name="X",
                endpoint_path="/x",
                endpoint_method="GET",
                expected_loss=5.0,
                remediation_hours=1.0,
            )
        ],
    )
    report = build_report_model(data)
    header = report.section("header")
    assert header is not None
    values = {row[0]: row[1] for row in header.tables[0].rows}
    assert values["Run id"] == "not recorded"
    assert values["Configuration hash"] == "not recorded"
    assert values["Attacker model in force"] == "not recorded"
    assert values["Report generated"] == "not recorded"


def test_estimates_are_marked_as_estimates() -> None:
    text = build_report_model(full_payload()).text()
    assert "modelled estimate" in text
    assert "The money figures are estimates." in text
    assert "The probabilities are estimates." in text
    assert "No exploitation was attempted." in text


CHAIN_WEIGHT_ROW = "Weight given to what a finding opens up"


def test_chain_weight_is_derived_from_the_payload_not_assumed() -> None:
    report = build_report_model(full_payload())
    methodology = report.section("methodology")
    assert methodology is not None
    rows = {row[0]: row[1] for row in methodology.tables[0].rows}
    assert rows[CHAIN_WEIGHT_ROW].startswith("1.00 (derived from this run")

    # Make the relationship inconsistent and it refuses to state a weight.
    data = full_payload()
    data.findings[0].chain_adjusted = data.findings[0].expected_loss + 3.0 * (
        data.findings[0].chain_delta
    )
    rows = {
        row[0]: row[1]
        for row in build_report_model(data).section("methodology").tables[0].rows
    }
    assert "not consistently derivable" in rows[CHAIN_WEIGHT_ROW]


# ---------------------------------------------------------------------------
# Detail cap
# ---------------------------------------------------------------------------


def test_detail_is_capped_and_the_omission_is_stated() -> None:
    data = full_payload(count=12)
    report = build_report_model(data, options=ReportOptions(max_detail_findings=4))

    assert report.detail_shown == 4
    assert report.detail_omitted == 8
    detail = report.section("finding-detail")
    assert detail is not None
    assert len(detail.subsections) == 4
    assert "8 findings are omitted from this section" in detail.text()
    assert "capped at 4 entries" in detail.text()

    # The appendix still lists all of them.
    appendix = report.section("appendix-findings")
    assert appendix is not None
    assert len(appendix.tables[0].rows) == 12


def test_detail_threshold_excludes_and_explains() -> None:
    data = full_payload(count=6)
    threshold = data.findings[2].expected_loss
    report = build_report_model(
        data, options=ReportOptions(min_detail_expected_loss=threshold)
    )
    assert report.detail_shown == 3
    assert report.detail_omitted == 3
    assert "expected-loss threshold for detail" in report.section("finding-detail").text()


# ---------------------------------------------------------------------------
# More than one scan: ranks are not comparable, applications must be named,
# and repeated scans of one application must not be counted twice
# ---------------------------------------------------------------------------


def test_single_scan_payload_is_unchanged() -> None:
    """One scan: rank is the ranker's own answer, and the target is named."""
    data = full_payload(4)
    report = build_report_model(data)

    assert report.title == "Security assessment: Example Shop"
    table = report.section("fix-first").tables[0]
    assert "Application" not in table.columns
    assert [row[0] for row in table.rows] == [str(item.rank) for item in data.findings]
    assert not report.section("scope").tables


def test_multi_scan_rows_are_numbered_sequentially() -> None:
    """Twenty-four scans hold twenty-four findings ranked 1; the report must not print that."""
    data = multi_scan_payload()
    per_scan_ranks = [item.rank for item in data.findings]
    assert per_scan_ranks.count(1) > 1, "the fixture must reproduce the repeated rank"

    report = build_report_model(data)
    table = report.section("fix-first").tables[0]
    positions = [row[0] for row in table.rows]
    assert positions == [str(index) for index in range(1, len(table.rows) + 1)]
    assert len(set(positions)) == len(positions)

    # The whole queue is numbered, not just the top table.
    appendix = report.section("appendix-findings").tables[0]
    assert [row[0] for row in appendix.rows] == [
        str(index) for index in range(1, len(appendix.rows) + 1)
    ]


def test_multi_scan_orders_by_priority_across_applications() -> None:
    data = multi_scan_payload()
    report = build_report_model(data)
    losses = [
        float(row[4].replace(currency_symbol(CURRENCY), "").replace(",", ""))
        for row in report.section("fix-first").tables[0].rows
    ]
    assert losses == sorted(losses, reverse=True)


def test_multi_scan_ordering_falls_back_to_expected_loss() -> None:
    """With no chain contribution carried, the ordering uses expected loss."""
    data = multi_scan_payload()
    for item in data.findings:
        item.chain_adjusted = 0.0
    report = build_report_model(data)
    ordered = [item.expected_loss for item in report_findings(report, data)]
    assert ordered == sorted(ordered, reverse=True)


def report_findings(report: AssessmentReport, data: DashboardData) -> list[WebFinding]:
    """The findings the report kept, in the order it put them in.

    Read off the detail subsections, whose ids carry the finding id. The fixtures are small
    enough that the detail cap never bites, and the assertion below keeps it that way.
    """
    detail = report.section("finding-detail")
    assert detail is not None
    assert report.detail_omitted == 0, "fixture outgrew the detail cap; this helper would lie"
    by_id = {item.finding_id: item for item in data.findings}
    return [by_id[section.id.removeprefix("finding-")] for section in detail.subsections]


def test_multi_application_tables_name_the_application() -> None:
    """'Fix the deserialization flaw' is not actionable when eight services have one."""
    data = multi_scan_payload()
    report = build_report_model(data)

    for section_id in ("fix-first", "appendix-findings"):
        table = report.section(section_id).tables[0]
        assert table.columns[1] == "Application"
        labels = {row[1] for row in table.rows}
        assert labels <= {name for _, name in APPLICATIONS}
        assert len(labels) > 1

    # And each finding's own entry says which application it belongs to.
    detail = report.section("finding-detail").subsections[0]
    identity = {row[0]: row[1] for row in detail.tables[0].rows}
    assert identity["Application"] in {name for _, name in APPLICATIONS}
    assert "(" in detail.title and identity["Application"] in detail.title


def test_multi_application_title_is_counted_not_concatenated() -> None:
    data = multi_scan_payload()
    report = build_report_model(data)

    assert report.title == "Security assessment: 3 applications"
    for _, name in APPLICATIONS:
        assert name not in report.title
    # The run-identity row points at the scope section rather than listing eight names.
    identity = {row[0]: row[1] for row in report.section("header").tables[0].rows}
    assert identity["Target"] in {
        "Northwind Commerce, Carelink Records and Opsgrid Platform",
        "3 applications, listed in the scope section",
    }


def test_title_counts_scans_when_several_survive_per_application() -> None:
    data = multi_scan_payload(dated=False)
    report = build_report_model(data)
    assert report.title == "Security assessment: 3 applications, 9 scans"


def test_scope_lists_the_applications_deduplicated() -> None:
    data = multi_scan_payload()
    report = build_report_model(data)
    scope = report.section("scope")
    assert scope.tables, "a multi-application report must list its applications"
    rows = scope.tables[0].rows
    assert [row[0] for row in rows] == [name for _, name in APPLICATIONS]
    assert len(rows) == len(APPLICATIONS), "one row per application, not one per scan"
    for row in rows:
        assert row[1].startswith("1 (of 3; earlier ones superseded)")


def test_the_latest_scan_of_each_application_is_used() -> None:
    """A research run needs the time series; an assessment describes the current state."""
    data = multi_scan_payload()
    report = build_report_model(data)

    kept = report_findings(report, data)
    kept_scans = {item.scan_id for item in kept}
    assert kept_scans == {f"scan_{app_id}_2" for app_id, _ in APPLICATIONS}

    # Nothing is counted three times.
    assert report.numbers["n_findings"] == len(APPLICATIONS) * 4
    assert report.numbers["n_findings"] < data.summary.n_findings
    latest_loss = sum(
        item.expected_loss for item in data.findings if item.scan_id in kept_scans
    )
    assert report.numbers["total_expected_loss"] == pytest.approx(latest_loss)
    assert report.numbers["total_expected_loss"] < data.summary.total_expected_loss

    # And it says so, rather than silently pruning.
    scope = report.section("scope").text()
    assert "most recent scan of each" in scope
    assert "6 earlier scans in the run are not included" in scope
    assert "over the included scans only" in scope


def test_repeated_scans_without_dates_are_not_pruned_on_a_guess() -> None:
    """No date means no way to know which is current, so nothing is dropped and it says so."""
    data = multi_scan_payload(dated=False)
    report = build_report_model(data)

    assert report.numbers["n_findings"] == data.summary.n_findings
    scope = report.section("scope").text()
    assert "carry no usable date" in scope
    assert "counted more than once" in scope
    assert "most recent scan of each" not in scope


def test_per_scan_rank_stays_available_in_the_detail_entry() -> None:
    data = multi_scan_payload()
    report = build_report_model(data)
    detail = report.section("finding-detail").subsections[0]
    identity = {row[0]: row[1] for row in detail.tables[0].rows}
    assert identity["Rank within its own scan"].startswith("1 (")
    assert "not the same thing" in identity["Rank within its own scan"]


def test_a_single_application_scanned_repeatedly_needs_no_application_column() -> None:
    data = multi_scan_payload(apps=1)
    report = build_report_model(data)
    assert report.title == "Security assessment: Northwind Commerce"
    assert "Application" not in report.section("fix-first").tables[0].columns


# ---------------------------------------------------------------------------
# The research material is opt-in
# ---------------------------------------------------------------------------

#: Words that belong in a paper and not in the document an engineer is handed.
RESEARCH_VOCABULARY = (
    "Component A",
    "Components enabled",
    "NDCG",
    "ndcg",
    "ablation",
    "LambdaMART",
    "Shapley",
    "literature",
    "monotone",
    "baseline",
    "synthetic oracle",
    "trust tier",
    "Trust tier",
    "corpus",
)


def test_research_material_is_absent_by_default() -> None:
    report = build_report_model(full_payload())

    for section_id in RESEARCH_SECTIONS:
        assert not report.has(section_id), f"{section_id} should be opt-in"
        assert section_id in report.omitted_sections

    # The operator's document keeps the appendix, but only the part they can act on.
    assert report.has("appendix-findings")
    assert report.options.include_research is False


def test_the_default_document_has_no_research_vocabulary() -> None:
    markdown = build_report(full_payload())
    found = [word for word in RESEARCH_VOCABULARY if word in markdown]
    assert found == [], f"research vocabulary leaked into the default report: {found}"


def test_research_material_comes_back_when_asked_for() -> None:
    report = build_report_model(full_payload(), options=ReportOptions(include_research=True))

    for section_id in RESEARCH_SECTIONS:
        assert report.has(section_id), f"{section_id} should be present when opted in"
        assert section_id not in report.omitted_sections

    markdown = build_report(full_payload(), options=ReportOptions(include_research=True))
    assert "LambdaMART" in markdown
    assert "Gap 1" in markdown
    assert "weakness classes" in markdown
    assert "Trust tiers, in the framework" in markdown
    assert "Components enabled" in markdown


def test_research_subsections_absent_from_the_payload_are_still_recorded() -> None:
    """Opting in does not invent a section the run produced no data for."""
    data = full_payload()
    data.gaps = []
    report = build_report_model(data, options=ReportOptions(include_research=True))
    assert not report.has("appendix-traceability")
    assert "appendix-traceability" in report.omitted_sections


def test_the_feature_attribution_table_is_research_only() -> None:
    data = full_payload(count=2)
    default = build_report_model(data)
    opted_in = build_report_model(data, options=ReportOptions(include_research=True))

    finding_id = data.findings[0].finding_id
    assert default.section(f"rank-{finding_id}").tables == []
    assert opted_in.section(f"rank-{finding_id}").tables


def test_the_methodology_survives_in_plain_words() -> None:
    """The engineer still gets the formula and the settings, without the machinery."""
    report = build_report_model(full_payload())
    methodology = report.section("methodology")
    assert methodology is not None
    text = methodology.text()

    # The one plain sentence leads.
    assert methodology.paragraphs[0].startswith("Priority here is one number: the chance that")
    assert "`expected loss = chance of exploitation x cost if exploited`" in text
    assert "Why this is not severity order." in text

    # This run's settings are still on the page.
    rows = {row[0]: row[1] for row in methodology.tables[0].rows}
    assert rows["Attacker assumed"] == "opportunistic"
    assert rows["Cost model used"] == "default_ecommerce"
    assert rows["Intelligence as of"] == "2024-06-01"

    # Trust is described in words, not in tier numbers.
    weighting = report.section("evidence-weighting")
    assert weighting is not None
    assert "may not be neutral" in weighting.text()
    assert weighting.tables == []


def test_include_research_does_not_change_the_operator_sections() -> None:
    data = full_payload()
    default = build_report_model(data)
    opted_in = build_report_model(data, options=ReportOptions(include_research=True))
    for section_id in ("executive-summary", "fix-first", "remediation-plan", "assurance"):
        assert default.section(section_id).text() == opted_in.section(section_id).text()


# ---------------------------------------------------------------------------
# Reason codes only: no model-authored prose about any finding
# ---------------------------------------------------------------------------


def test_finding_text_uses_reason_codes_and_nothing_else() -> None:
    data = full_payload()
    report = build_report_model(data)
    by_id = {item.finding_id: item for item in data.findings}

    for finding_id, finding in by_id.items():
        section = report.section(f"rank-{finding_id}")
        if section is None:
            continue
        assert section.bullets == finding.reason_codes

    # And the "why it is here" block under the top table is the same list.
    top = data.findings[0]
    why = report.section(f"why-{top.finding_id}")
    assert why is not None
    assert why.bullets == top.reason_codes


def test_findings_with_no_reason_codes_fall_back_to_their_own_numbers() -> None:
    data = full_payload(count=2)
    data.findings[0].reason_codes = []
    report = build_report_model(data)
    bullets = report.section(f"rank-{data.findings[0].finding_id}").bullets
    assert bullets
    # Every fallback bullet quotes a figure that is in the payload.
    assert any("$" in bullet or "EPSS" in bullet or "catalogue" in bullet for bullet in bullets)
    assert not any(bullet.startswith("REASON-") for bullet in bullets)


def test_the_package_imports_no_language_model() -> None:
    """No model authors any part of this document, and the imports prove it."""
    package = Path(__file__).resolve().parents[1] / "src" / "vulnprio" / "report"
    forbidden = ("anthropic", "openai", "vulnprio.llm", "vulnprio.sandbox", "vulnprio.semantic")
    pattern = re.compile(r"^\s*(?:from|import)\s+\S+", re.MULTILINE)
    for path in sorted(package.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for statement in pattern.findall(source):
            for name in forbidden:
                assert name not in statement, f"{path.name} imports {name}"

    # Importing the package here and inspecting sys.modules would be order-dependent:
    # sys.modules is process-global, so any earlier test that legitimately imported the
    # SDK makes this fail for a reason that has nothing to do with the report package.
    # Ask a clean interpreter instead, which is the question we actually mean.
    import subprocess
    import sys

    probe = subprocess.run(
        [sys.executable, "-c", "import sys, vulnprio.report; print('anthropic' in sys.modules)"],
        capture_output=True, text=True, timeout=120,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "False", "importing vulnprio.report pulled in a language model SDK"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_payload_produces_byte_identical_output() -> None:
    first, second = full_payload(), full_payload()
    assert build_report(first, "md") == build_report(second, "md")
    assert build_report(first, "html") == build_report(second, "html")
    assert build_report(first, "json") == build_report(second, "json")


def test_repeated_calls_on_one_payload_are_identical() -> None:
    data = full_payload()
    runs = [build_report(data, "md") for _ in range(3)]
    assert len(set(runs)) == 1


# ---------------------------------------------------------------------------
# Remediation knowledge base
# ---------------------------------------------------------------------------


def test_every_required_cwe_has_guidance() -> None:
    missing = [cwe for cwe in REQUIRED_CWES if cwe not in REMEDIATIONS]
    assert missing == [], f"knowledge base is missing {missing}"


@pytest.mark.parametrize("cwe_id", sorted(REMEDIATIONS))
def test_every_entry_is_complete(cwe_id: int) -> None:
    entry = REMEDIATIONS[cwe_id]
    assert entry.cwe_id == cwe_id
    assert entry.weakness
    assert entry.summary
    assert 3 <= len(entry.steps) <= 6, f"CWE-{cwe_id} has {len(entry.steps)} steps"
    assert all(step.strip() for step in entry.steps)
    assert entry.verification
    assert entry.typical_effort
    assert entry.do_not.lower().startswith("do not")
    assert entry.source
    assert not entry.generic


def test_guidance_for_returns_the_curated_entry() -> None:
    entry = guidance_for(89)
    assert entry.cwe_id == 89
    assert not entry.generic
    assert "parameter" in " ".join(entry.steps).lower()


def test_guidance_for_an_unknown_cwe_is_clearly_generic() -> None:
    entry = guidance_for(999_999)
    assert entry.generic is True
    assert entry.summary.startswith("This is generic guidance.")
    assert "CWE-999999" in entry.summary
    assert entry.cwe_id == 999_999
    assert entry.typical_effort == "Not estimated: no playbook is held for this weakness."


def test_guidance_for_no_cwe_is_generic_too() -> None:
    entry = guidance_for(None)
    assert entry.generic is True
    assert entry is GENERIC_REMEDIATION or entry.summary == GENERIC_REMEDIATION.summary


def test_guidance_context_comes_from_the_finding_only() -> None:
    finding = make_finding(0, cluster_size=4)
    entry = guidance_for(finding.cwe_id, finding)
    assert finding.endpoint_path in entry.context
    assert "4 alerts" in entry.context
    # The guidance itself is unchanged by the finding.
    assert entry.steps == REMEDIATIONS[finding.cwe_id].steps


def test_generic_guidance_is_flagged_in_the_report() -> None:
    data = full_payload(count=1)
    data.findings[0].cwe_id = 999_999
    text = build_report_model(data).text()
    assert "This guidance is generic." in text
    assert "fell back to the generic one" not in text  # it did not
    assert "CWE-999999" in text


def test_known_cwes_is_sorted_and_matches_the_table() -> None:
    assert known_cwes() == tuple(sorted(REMEDIATIONS))
    assert len(known_cwes()) >= len(REQUIRED_CWES)


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["md", "html", "json"])
def test_five_hundred_findings_render_in_well_under_a_second(fmt: str) -> None:
    data = full_payload(count=500)
    start = time.perf_counter()
    output = build_report(data, fmt)
    elapsed = time.perf_counter() - start
    assert output
    assert elapsed < 1.0, f"{fmt} took {elapsed:.3f}s for 500 findings"
