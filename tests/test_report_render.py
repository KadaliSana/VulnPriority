"""Rendering the assessment report: Markdown, HTML and JSON.

The report is a deliverable, so it is tested like one. A table that breaks when a value
contains a pipe, or a page that quietly fetches a font from the internet, would each defeat
the point of the artefact: the Markdown is meant to survive being pasted into an issue
tracker, and the HTML is meant to open and print on a machine with no network.

The payloads here are built by hand and deliberately include hostile values - pipes, angle
brackets, quotes and newlines - because those are what a scanner actually reports.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from vulnprio.report import (
    AssessmentReport,
    ReportOptions,
    build_report,
    build_report_model,
    render_html,
    render_markdown,
)
from vulnprio.report.render_md import markdown_table
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

GENERATED_AT = datetime(2024, 6, 2, 8, 30, tzinfo=timezone.utc)

#: Values a scanner really does emit, and every one of them can break a naive renderer.
HOSTILE_NAME = 'SQL injection via "id" | OR 1=1 <script>alert(1)</script>'
HOSTILE_PATH = "/search?q=a|b&c=<d>"
HOSTILE_REASON = "Reason with a | pipe, a <tag> and an & ampersand."


def _finding(index: int, **overrides) -> WebFinding:
    finding = WebFinding(
        finding_id=f"f{index}",
        scan_id="scan_1",
        app_id="app1",
        name=f"Finding {index}",
        cwe_id=[89, 79, 918, 614][index % 4],
        cve_ids=[f"CVE-2024-{2000 + index}"],
        endpoint_path=f"/api/{index}",
        endpoint_method="GET",
        endpoint_function="api_data",
        scanner_severity="high",
        cvss_base=8.1,
        cvss_version="3.1",
        cvss_source_agreement=0.9,
        epss=0.3,
        epss_percentile=0.8,
        kev=index == 0,
        exploit_maturity="functional",
        exploit_count=1,
        criticality=0.7,
        exposure=0.5,
        exploit_feasibility=0.6,
        applicability="applicable",
        p_applicable=0.85,
        version_match="match",
        p_exploit=0.4,
        impact=500_000.0 - index * 1_000,
        expected_loss=200_000.0 - index * 400,
        chain_delta=1_000.0,
        chain_adjusted=201_000.0 - index * 400,
        remediation_hours=3.0,
        rank=index + 1,
        reason_codes=[f"Reason {index}."],
        contributions=[WebContribution(feature="b_kev", value=1.0, shap=0.5, group="B", tier=1)],
        untrusted_influence_share=0.1,
        max_tier_used=2,
    )
    return finding.model_copy(update=overrides) if overrides else finding


def payload(count: int = 4, *, hostile: bool = False) -> DashboardData:
    findings = [_finding(index) for index in range(count)]
    if hostile:
        findings[0] = findings[0].model_copy(
            update={
                "name": HOSTILE_NAME,
                "endpoint_path": HOSTILE_PATH,
                "reason_codes": [HOSTILE_REASON, "Second\nline reason."],
            }
        )
    findings[0].selected_in_budget = True
    total = sum(item.expected_loss for item in findings)
    return DashboardData(
        meta=WebMeta(
            run_id="run_render",
            generated_at=GENERATED_AT,
            config_hash="cfg777",
            package_version="0.1.0",
            llm_backend="heuristic",
            feed_mode="offline",
            attacker="opportunistic",
            impact_model="default_ecommerce",
            as_of="2024-06-01",
            components={"a": True, "b": True, "c": True},
        ),
        summary=WebSummary(
            n_apps=1,
            n_scans=1,
            n_endpoints=9,
            n_findings=len(findings),
            n_kev=1,
            total_expected_loss=total,
            total_impact=sum(item.impact for item in findings),
            total_remediation_hours=sum(item.remediation_hours for item in findings),
            top_decile_loss_share=0.25,
        ),
        scans=[
            WebScan(
                scan_id="scan_1",
                app_id="app1",
                app_name="Example | Shop <beta>" if hostile else "Example Shop",
                sector="ecommerce",
                scanned_at="2024-05-20T09:30:00",
                scanner="zap",
                n_endpoints=9,
                n_findings=len(findings),
                total_expected_loss=total,
            )
        ],
        findings=findings,
        graphs=[
            WebGraph(
                scan_id="scan_1",
                entry_node="state:internet:NONE",
                target_nodes=["state:shop.example.com:ADMIN"],
                total_risk=750_000.0,
                monotone_verified=True,
                top_paths=[
                    WebPath(
                        nodes=["state:internet:NONE", "state:shop.example.com:ADMIN"],
                        finding_ids=["f0"],
                        probability=0.3,
                        target_value=400_000.0,
                        expected_value=120_000.0,
                    )
                ],
            )
        ],
        selections=[
            WebSelection(
                scan_id="scan_1",
                ranker="lambdamart",
                method="dp_exact",
                budget_hours=6.0,
                n_selected=1,
                total_hours=3.0,
                risk_captured=201_000.0,
                risk_capture_fraction=0.4,
                selected_ids=["f0"],
            )
        ],
        adversarial=WebAdversarial(
            backend="heuristic", corpus_version="v1", n_cases=84, detection_rate=0.9
        ),
        gaps=[
            WebGapRow(
                gap_id="Gap 9",
                title="Chaining sits outside the models | and impact is not in money.",
                mitigation="A directed multi-hop attack graph.",
                modules=["graph/attack_graph.py"],
                evidence="$750,000 of reachable risk modelled.",
            )
        ],
        notes={"caveat": "Constructed for a test."},
    )


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def unescaped_pipes(line: str) -> int:
    """Pipes that actually delimit a cell: a backslash-escaped pipe does not."""
    count = 0
    index = 0
    while index < len(line):
        if line[index] == "\\":
            index += 2
            continue
        if line[index] == "|":
            count += 1
        index += 1
    return count


def table_blocks(markdown: str) -> list[list[str]]:
    """Every run of consecutive table lines in the document."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


SEPARATOR = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+$")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_markdown_tables_are_well_formed() -> None:
    markdown = build_report(payload())
    blocks = table_blocks(markdown)
    assert blocks, "the report should contain tables"
    for block in blocks:
        assert len(block) >= 3, f"table with no rows: {block}"
        assert SEPARATOR.match(block[1]), f"bad separator row: {block[1]!r}"
        widths = {unescaped_pipes(line) for line in block}
        assert len(widths) == 1, f"ragged table: {block[:3]}"
        assert widths.pop() >= 2


def test_hostile_values_do_not_break_a_table() -> None:
    markdown = build_report(payload(hostile=True))
    for block in table_blocks(markdown):
        widths = {unescaped_pipes(line) for line in block}
        assert len(widths) == 1, f"a hostile value split a row: {block[:3]}"
    # The pipe survives, escaped, rather than being dropped.
    assert "\\|" in markdown
    assert "OR 1=1" in markdown


def test_markdown_escapes_every_pipe_inside_a_cell() -> None:
    markdown = build_report(payload(hostile=True))
    for block in table_blocks(markdown):
        for line in block[2:]:
            cells = re.split(r"(?<!\\)\|", line)[1:-1]
            for cell in cells:
                # A cell may contain an escaped pipe but never a bare one.
                assert not re.search(r"(?<!\\)\|", cell)


def test_markdown_carries_no_html() -> None:
    """The renderer emits none, and it neutralises the HTML a scanner puts in its own text."""
    markdown = build_report(payload(hostile=True))
    for tag in ("div", "table", "tr", "td", "script", "style", "br", "/script"):
        assert not re.search(rf"(?<!\\)<{re.escape(tag)}\b", markdown), tag
    assert not re.search(r"(?<!\\)&[A-Za-z#][A-Za-z0-9]*;", markdown)
    # Escaped, not deleted: the reader still sees the payload the scanner reported.
    assert "\\<script>alert(1)\\</script>" in markdown


def test_markdown_flattens_newlines_inside_cells() -> None:
    markdown = build_report(payload(hostile=True))
    for block in table_blocks(markdown):
        for line in block:
            assert "\n" not in line


def test_markdown_has_no_heading_without_content() -> None:
    """An omitted section leaves no heading; a present one always has something under it."""
    markdown = build_report(payload())
    lines = markdown.splitlines()
    headings = [
        (index, len(line) - len(line.lstrip("#")))
        for index, line in enumerate(lines)
        if line.startswith("#")
    ]
    for position, (index, level) in enumerate(headings):
        # Content counts until the next heading at the same or a higher level.
        end = len(lines)
        for later_index, later_level in headings[position + 1 :]:
            if later_level <= level:
                end = later_index
                break
        body = [line for line in lines[index + 1 : end] if line.strip()]
        assert body, f"empty section: {lines[index]!r}"


def test_multi_application_tables_stay_well_formed() -> None:
    """The Application column appears in several tables; none of them may go ragged."""
    data = payload(count=4)
    data.scans.append(
        WebScan(
            scan_id="scan_2",
            app_id="app2",
            app_name="Second | Service",
            scanned_at="2024-05-21T09:30:00",
            scanner="zap",
            n_endpoints=4,
            n_findings=1,
        )
    )
    data.findings.append(
        _finding(9, finding_id="f9", scan_id="scan_2", app_id="app2", rank=1)
    )
    markdown = build_report(data)

    for block in table_blocks(markdown):
        widths = {unescaped_pipes(line) for line in block}
        assert len(widths) == 1, f"ragged table: {block[:3]}"
    assert "| Application |" in markdown
    assert "Second \\| Service" in markdown
    # Two findings ranked 1 in their own scans, numbered 1 and 2 in the document.
    assert "Security assessment: 2 applications" in markdown


def test_markdown_table_helper_pads_short_rows() -> None:
    lines = markdown_table(["a", "b", "c"], [["1"], ["1", "2", "3", "4"]])
    assert unescaped_pipes(lines[0]) == 4
    assert all(unescaped_pipes(line) == 4 for line in lines)


def test_markdown_starts_with_the_title_and_lists_contents() -> None:
    report = build_report_model(payload())
    markdown = render_markdown(report)
    assert markdown.startswith(f"# {report.title}\n")
    assert "**Contents**" in markdown
    for section in report.sections:
        assert f"- {section.title}" in markdown
    assert markdown.endswith("\n")
    assert "\n\n\n" not in markdown


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def test_html_is_self_contained() -> None:
    html = build_report(payload(hostile=True), "html")
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html
    # Nothing is fetched: no scripts, no external references of any kind.
    assert "<script" not in html.lower()
    assert "src=" not in html.lower()
    assert "http://" not in html
    assert "https://" not in html
    assert "@import" not in html
    assert "url(" not in html
    assert "<link" not in html.lower()
    assert "<iframe" not in html.lower()
    # The only href is an internal anchor.
    for href in re.findall(r'href="([^"]*)"', html):
        assert href.startswith("#"), href


def test_html_escapes_hostile_content() -> None:
    html = build_report(payload(hostile=True), "html")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Example | Shop &lt;beta&gt;" in html


def test_html_renders_the_tables_and_the_contents() -> None:
    report = build_report_model(payload())
    html = render_html(report)
    assert html.count("<table>") == sum(
        1
        for section in report.sections
        for candidate in section.walk()
        for table in candidate.tables
        if not table.is_empty
    )
    assert '<nav class="toc">' in html
    for section in report.sections:
        assert f'id="{section.id}"' in html
        assert f'href="#{section.id}"' in html
    assert html.rstrip().endswith("</html>")


def test_html_carries_a_print_stylesheet() -> None:
    html = build_report(payload(), "html")
    assert "@media print" in html
    assert "break-inside" in html


def test_html_omits_sections_with_no_data() -> None:
    minimal = DashboardData(findings=[_finding(0)])
    html = build_report(minimal, "html")
    assert "The remediation plan" not in html
    assert "Attack chains" not in html
    assert "Executive summary" in html


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def test_json_round_trips_through_the_model() -> None:
    data = payload()
    text = build_report(data, "json")
    restored = AssessmentReport.model_validate_json(text)
    assert restored == build_report_model(data)
    assert restored.model_dump_json(indent=2) == text


def test_json_carries_the_structure_not_the_prose_only() -> None:
    report = AssessmentReport.model_validate_json(build_report(payload(), "json"))
    assert [section.id for section in report.sections]
    assert report.numbers["n_findings"] == 4
    detail = report.section("finding-detail")
    assert detail is not None
    assert detail.subsections
    assert detail.subsections[0].tables


def test_json_and_markdown_describe_the_same_report() -> None:
    data = payload()
    from_json = AssessmentReport.model_validate_json(build_report(data, "json"))
    assert render_markdown(from_json) == build_report(data, "md")


# ---------------------------------------------------------------------------
# Format selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias,expected", [("md", "md"), ("markdown", "md"), ("MD", "md")])
def test_format_aliases(alias: str, expected: str) -> None:
    assert build_report(payload(), alias) == build_report(payload(), expected)


def test_options_reach_the_renderers() -> None:
    data = payload(count=6)
    markdown = build_report(data, options=ReportOptions(top_table_rows=2, max_detail_findings=1))
    report = build_report_model(
        data, options=ReportOptions(top_table_rows=2, max_detail_findings=1)
    )
    assert len(report.section("fix-first").tables[0].rows) == 2
    assert report.detail_shown == 1
    assert "5 findings are omitted from this section" in markdown


def test_title_override() -> None:
    markdown = build_report(payload(), title="Q2 assessment for Example Shop")
    assert markdown.startswith("# Q2 assessment for Example Shop\n")


# ---------------------------------------------------------------------------
# Presentation for a host page: the theme it is in, and the furniture it already has
# ---------------------------------------------------------------------------


def _root_tag(html: str) -> str:
    """The document's ``<html ...>`` tag. The stylesheet names every selector it supports,
    so asking whether a *document* is embedded means reading the root element, not grepping
    the whole file."""
    match = re.search(r"<html[^>]*>", html)
    assert match, "every render must produce a root element"
    return match.group(0)


def test_a_host_page_can_pin_the_palette() -> None:
    """The results page has its own light/dark control; the report has to agree with it.

    Without this the report only ever reads ``prefers-color-scheme``, so a reader who has
    chosen light on a dark system gets a white document inside a dark page, which is the
    one thing that made the embedded report look broken.
    """
    report = build_report_model(payload())
    assert 'data-theme="dark"' in _root_tag(render_html(report, theme="dark"))
    assert 'data-theme="light"' in _root_tag(render_html(report, theme="light"))


def test_an_unrecognised_theme_leaves_the_document_following_the_system() -> None:
    """A file opened on its own should follow the reader, not a stale query parameter."""
    report = build_report_model(payload())
    for value in ("", "sepia", "DARKISH", "none"):
        assert "data-theme" not in _root_tag(render_html(report, theme=value))


def test_an_explicit_light_choice_survives_a_dark_system() -> None:
    """The media query has to be guarded, or it overrides the choice it is meant to defer to."""
    html = render_html(build_report_model(payload()), theme="light")
    assert ':root:not([data-theme="light"])' in html
    assert ':root[data-theme="dark"]' in html


def test_embedding_drops_only_the_furniture_the_host_already_provides() -> None:
    """The document keeps its structure and gives up its page-ness.

    Its own background, page margins and reading-width limit are what made it read as a
    second page inside the first. Everything that carries meaning stays.
    """
    report = build_report_model(payload())
    embedded = render_html(report, embedded=True)
    assert "data-embedded" in _root_tag(embedded)
    # Painted, not transparent: a transparent body leaves the iframe on the user agent's
    # white canvas, which showed as a band in every gap between two panels.
    assert "background: var(--bg)" in embedded
    assert "color-scheme: dark" in embedded
    # Still the whole report: the masthead, the contents and every section.
    assert '<nav class="toc">' in embedded
    for section in report.sections:
        assert f'id="{section.id}"' in embedded


def test_a_standalone_render_is_unchanged() -> None:
    """The default is still one self-contained document that stands on its own."""
    html = render_html(build_report_model(payload()))
    assert _root_tag(html) == '<html lang="en">'
    assert html.startswith("<!DOCTYPE html>")


def test_presentation_does_not_change_a_word_of_the_report() -> None:
    """Theme and embedding are styling. If either ever edits content, this fails."""
    report = build_report_model(payload())
    plain = re.sub(r"<[^>]+>", " ", render_html(report))
    dressed = re.sub(r"<[^>]+>", " ", render_html(report, theme="dark", embedded=True))
    assert " ".join(plain.split()) == " ".join(dressed.split())
