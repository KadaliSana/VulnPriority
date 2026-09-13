"""Markdown rendering of an :class:`~vulnprio.report.model.AssessmentReport`.

GitHub-flavoured Markdown and nothing else: no embedded HTML, no raw angle brackets, no
reliance on a renderer's extensions. A pipe inside a value is escaped so it cannot split a
cell, and a newline inside a value becomes a space so it cannot split a row. That is what
makes the tables survive being pasted into an issue tracker, a wiki or a pull request.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from vulnprio.report.model import AssessmentReport, ReportSection, ReportTable

__all__ = ["render_markdown", "markdown_table"]

#: A ``<`` that begins something a Markdown renderer would treat as raw HTML. Scanner output
#: routinely contains one - an XSS payload is quoted verbatim in the finding's own name - and
#: a report that renders it as markup rather than as text is a report that carries the
#: payload. Escaped, not stripped: the reader still sees exactly what the scanner saw.
_TAG_OPEN = re.compile(r"<(?=[A-Za-z/!?])")
#: An ``&`` that begins an HTML entity, for the same reason.
_ENTITY = re.compile(r"&(?=[A-Za-z#][A-Za-z0-9]*;)")


def _safe(value: object) -> str:
    """Neutralise raw HTML in a value without changing what the value says."""
    text = str(value)
    text = _TAG_OPEN.sub("\\\\<", text)
    return _ENTITY.sub("\\\\&", text)


def _cell(value: object) -> str:
    """One table cell: pipes escaped, newlines flattened, so the row cannot break."""
    text = _safe(value)
    text = text.replace("\\|", "|")          # avoid double-escaping an already-escaped pipe
    text = text.replace("|", "\\|")
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return " ".join(text.split()) or " "


def markdown_table(columns: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    """A GitHub-flavoured table with a consistent cell count on every row."""
    headers = [_cell(column) for column in columns]
    width = len(headers)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * width) + "|"]
    for row in rows:
        cells = [_cell(value) for value in row]
        if len(cells) < width:
            cells += [" "] * (width - len(cells))
        lines.append("| " + " | ".join(cells[:width]) + " |")
    return lines


def _render_table(table: ReportTable) -> list[str]:
    if table.is_empty:
        return []
    lines: list[str] = []
    if table.caption:
        lines += [f"**{_safe(table.caption)}**", ""]
    lines += markdown_table(table.columns, table.rows)
    if table.note:
        lines += ["", f"_{_safe(table.note)}_"]
    lines.append("")
    return lines


def _paragraph(text: str) -> str:
    """A prose block: HTML neutralised, hard line breaks folded into the paragraph."""
    return " ".join(_safe(text).split())


def _render_section(section: ReportSection) -> list[str]:
    if section.is_empty:
        return []
    level = max(2, min(6, section.level))
    lines: list[str] = [f"{'#' * level} {_paragraph(section.title)}", ""]
    if section.lead:
        lines += [_paragraph(section.lead), ""]
    for paragraph in section.paragraphs:
        lines += [_paragraph(paragraph), ""]
    for bullet in section.bullets:
        lines.append(f"- {_paragraph(bullet)}")
    if section.bullets:
        lines.append("")
    for table in section.tables:
        lines += _render_table(table)
    for sub in section.subsections:
        lines += _render_section(sub)
    return lines


def render_markdown(report: AssessmentReport) -> str:
    """Render the whole report as Markdown."""
    lines: list[str] = [f"# {_paragraph(report.title)}", ""]
    if report.subtitle:
        lines += [f"_{_paragraph(report.subtitle)}_", ""]

    top_level = [section for section in report.sections if not section.is_empty]
    if len(top_level) >= 3:
        lines += ["**Contents**", ""]
        for section in top_level:
            lines.append(f"- {_paragraph(section.title)}")
        lines.append("")

    for section in top_level:
        lines += _render_section(section)

    # Collapse the runs of blank lines that recursion leaves behind.
    out: list[str] = []
    for line in lines:
        if not line and out and not out[-1]:
            continue
        out.append(line)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"
