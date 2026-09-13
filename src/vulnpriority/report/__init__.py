"""The assessment report: what a security team or a client receives after a scan.

``vulnpriority.eval.report`` produces the *research* report - metrics, baselines, ablation,
calibration - which argues that the method works. This package produces the other document:
what was found, how bad it is in money and in practice, what to fix first, why, and what to
do about each one.

    from vulnpriority.report import build_report

    markdown = build_report(dashboard_data)                  # Markdown
    page     = build_report(dashboard_data, "html")          # one self-contained HTML file
    payload  = build_report(dashboard_data, "json")          # the AssessmentReport, dumped

Its input is the dashboard payload (:class:`~vulnpriority.web.schema.DashboardData`), so the
report and the website always describe the same run. A payload carrying only findings still
produces a complete document: sections with no data are omitted rather than left as empty
headings, and a figure that was not measured is reported as not measured rather than as
zero.

The default document is written for the person who has to fix the application, not for
someone evaluating the method. The research material - the traceability back to the
research gaps, the knowledge-base coverage listing, the numbered trust tiers, the
learned-ranker description and the per-finding feature attributions - is opt-in::

    build_report(data, options=ReportOptions(include_research=True))

The methodology section stays either way, because an engineer does need to know why one
finding outranks another; without the flag it is written in plain words.

There is no language model in this package. Every sentence is assembled deterministically
from the payload's numbers, because the same input has to produce the same words: this is a
document someone may sign.
"""

from __future__ import annotations

from typing import Any

from vulnpriority.report.model import (
    RESEARCH_SECTIONS,
    AssessmentReport,
    ReportMeta,
    ReportOptions,
    ReportSection,
    ReportTable,
    build_report_model,
)
from vulnpriority.report.remediation import (
    GENERIC_REMEDIATION,
    REMEDIATIONS,
    Remediation,
    guidance_for,
    known_cwes,
)
from vulnpriority.report.render_html import render_html
from vulnpriority.report.render_md import render_markdown

__all__ = [
    "build_report",
    "build_report_model",
    "AssessmentReport",
    "ReportSection",
    "ReportTable",
    "ReportMeta",
    "ReportOptions",
    "RESEARCH_SECTIONS",
    "Remediation",
    "REMEDIATIONS",
    "GENERIC_REMEDIATION",
    "guidance_for",
    "known_cwes",
    "render_markdown",
    "render_html",
    "REPORT_FORMATS",
]

#: The formats :func:`build_report` accepts.
REPORT_FORMATS: tuple[str, ...] = ("md", "html", "json")


def build_report(
    data: Any,
    fmt: str = "md",
    *,
    title: str | None = None,
    options: ReportOptions | None = None,
    theme: str = "",
    embedded: bool = False,
) -> str:
    """Build the assessment report and render it.

    Parameters
    ----------
    data:
        A :class:`~vulnpriority.web.schema.DashboardData` or its dictionary form.
    fmt:
        ``"md"`` for GitHub-flavoured Markdown, ``"html"`` for one self-contained document,
        ``"json"`` for the :class:`~vulnpriority.report.model.AssessmentReport` dumped, so the
        report is machine readable too.
    title:
        Overrides the generated title.
    options:
        Controls how much detail is included, and whether the research material appears at
        all (``include_research``, off by default); see
        :class:`~vulnpriority.report.model.ReportOptions`.
    theme:
        ``"light"`` or ``"dark"`` to pin the HTML document's palette, for a host page that
        has its own control and would otherwise disagree with it. Ignored by the other
        formats.
    embedded:
        Render the HTML without the page furniture a host page already provides. Ignored by
        the other formats.

    Returns
    -------
    str
        The rendered report.
    """
    normalised = str(fmt or "md").strip().lower()
    if normalised in ("markdown", "text"):
        normalised = "md"
    if normalised not in REPORT_FORMATS:
        raise ValueError(
            f"unknown report format {fmt!r}; expected one of {', '.join(REPORT_FORMATS)}"
        )

    report = build_report_model(data, title=title, options=options)
    if normalised == "md":
        return render_markdown(report)
    if normalised == "html":
        return render_html(report, theme=theme, embedded=embedded)
    return report.model_dump_json(indent=2)
