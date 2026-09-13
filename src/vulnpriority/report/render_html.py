"""HTML rendering of an :class:`~vulnpriority.report.model.AssessmentReport`.

One file, no dependencies at read time: the stylesheet is inline, there is no script, no
font is fetched, no image is linked, and nothing in the document causes the browser to open
a network connection. It opens from ``file://`` on a machine with no network - the same
property the framework claims for itself - and the print stylesheet makes the browser's own
"print to PDF" produce a sensible document without any extra tooling.

The palette is the dashboard's, so the two artefacts look like they came from the same
system, but nothing here depends on the dashboard's assets.
"""

from __future__ import annotations

import html
import re

from vulnpriority.report.model import AssessmentReport, ReportSection, ReportTable

__all__ = ["render_html", "STYLESHEET"]


STYLESHEET = """
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --panel-2: #fbfbfd;
  --ink: #14171c;
  --ink-2: #4a5260;
  --ink-3: #79828f;
  --line: #e2e5ea;
  --line-2: #eceff3;
  --accent: #2f6fd0;
  --accent-soft: #e8f0fc;
  --crit: #c0392b;
  --ok: #1d7a52;
  --radius: 10px;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
/* Three states, the same shape the dashboard uses. The media query is the default; an
   explicit ``data-theme`` on the root wins in both directions, which is what lets a host
   page that has its own light/dark control hand its choice to this document. Without the
   ``:not`` guard, a reader on a dark system who asked for light would still get dark. */
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e1116; --panel: #161b22; --panel-2: #1b2129; --ink: #e6edf3; --ink-2: #a9b4c0;
    --ink-3: #7d8895; --line: #2a313a; --line-2: #222831; --accent: #6ba3f5;
    --accent-soft: #17243a; --crit: #f27a6e; --ok: #56c596;
  }
}
:root[data-theme="dark"] {
  --bg: #0e1116; --panel: #161b22; --panel-2: #1b2129; --ink: #e6edf3; --ink-2: #a9b4c0;
  --ink-3: #7d8895; --line: #2a313a; --line-2: #222831; --accent: #6ba3f5;
  --accent-soft: #17243a; --crit: #f27a6e; --ok: #56c596;
}

/* Embedded in a host page that is already a page. The document keeps its structure - the
   masthead, the contents, the panelled sections - and gives up its page margins and its
   own reading-width limit, which the host already imposes.

   What it must NOT give up is a painted background. A transparent body does not make an
   iframe transparent: with nothing painted, the frame falls back to the user agent's
   default canvas, which is white. Against a dark host that shows as a white band in every
   gap between two panels - the document looked shot through with holes. So the embedded
   ground is painted, and painted in the *host's* values rather than this document's, so
   there is no seam where one seven-point-different grey meets another.

   ``color-scheme`` is declared for the same reason one step lower down: it is what makes
   the canvas, the scrollbars and any form control agree with the palette instead of
   defaulting to light. */
:root[data-embedded] { --bg: #ffffff; color-scheme: light; }
:root[data-embedded][data-theme="dark"] { --bg: #101317; color-scheme: dark; }
@media (prefers-color-scheme: dark) {
  :root[data-embedded]:not([data-theme="light"]) { --bg: #101317; color-scheme: dark; }
}
:root[data-embedded],
:root[data-embedded] body { background: var(--bg); }
:root[data-embedded] body { padding: 0 0 1.5rem; }
:root[data-embedded] .sheet { max-width: none; }
:root[data-embedded] header.masthead { margin-top: 0; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 1.25rem 4rem; background: var(--bg); color: var(--ink);
  font-family: var(--sans); font-size: 15px; line-height: 1.62;
  -webkit-text-size-adjust: 100%;
}
.sheet { max-width: 62rem; margin: 0 auto; }
header.masthead {
  background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1.75rem 1.75rem 1.4rem; margin: 2rem 0 1.5rem;
  border-top: 3px solid var(--accent);
}
header.masthead h1 { margin: 0 0 .35rem; font-size: 1.7rem; line-height: 1.25; letter-spacing: -.01em; }
header.masthead p.subtitle { margin: 0; color: var(--ink-2); font-size: .98rem; }
header.masthead p.stamp { margin: .75rem 0 0; color: var(--ink-3); font-size: .82rem; font-family: var(--mono); }
nav.toc {
  background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1rem 1.25rem; margin-bottom: 1.75rem;
}
nav.toc h2 { margin: 0 0 .5rem; font-size: .78rem; text-transform: uppercase;
  letter-spacing: .09em; color: var(--ink-3); font-weight: 600; }
nav.toc ol { margin: 0; padding-left: 1.2rem; columns: 2; column-gap: 2rem; }
nav.toc li { margin: .15rem 0; break-inside: avoid; }
nav.toc a { color: var(--accent); text-decoration: none; }
section.block {
  background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1.4rem 1.75rem 1.6rem; margin-bottom: 1.5rem;
}
h2 { font-size: 1.28rem; margin: 0 0 .35rem; letter-spacing: -.01em; }
h3 { font-size: 1.04rem; margin: 1.6rem 0 .3rem; color: var(--ink); }
h4 { font-size: .9rem; margin: 1.1rem 0 .25rem; color: var(--ink-2);
  text-transform: uppercase; letter-spacing: .06em; }
h2 + p.lead, h3 + p.lead { margin-top: .2rem; }
p { margin: .55rem 0; }
p.lead { color: var(--ink-2); }
ul { margin: .5rem 0; padding-left: 1.25rem; }
li { margin: .3rem 0; }
code { font-family: var(--mono); font-size: .88em; background: var(--accent-soft);
  color: var(--ink); padding: .1em .35em; border-radius: 4px; }
strong { font-weight: 650; }
.tablewrap { overflow-x: auto; margin: .85rem 0; }
table { border-collapse: collapse; width: 100%; font-size: .88rem; }
caption { caption-side: top; text-align: left; font-weight: 650; font-size: .88rem;
  padding-bottom: .4rem; color: var(--ink); }
th, td { text-align: left; padding: .45rem .7rem; border-bottom: 1px solid var(--line-2);
  vertical-align: top; }
thead th { border-bottom: 1px solid var(--line); color: var(--ink-2); font-weight: 600;
  white-space: nowrap; font-size: .82rem; text-transform: uppercase; letter-spacing: .05em; }
tbody tr:nth-child(even) { background: var(--panel-2); }
td.num { font-family: var(--mono); white-space: nowrap; }
p.note { color: var(--ink-3); font-size: .84rem; margin-top: .4rem; }
footer.colophon { color: var(--ink-3); font-size: .82rem; text-align: center;
  padding: 1.5rem 0 0; }
@media print {
  :root {
    --bg: #ffffff; --panel: #ffffff; --panel-2: #ffffff; --ink: #000000; --ink-2: #333333;
    --ink-3: #555555; --line: #bbbbbb; --line-2: #dddddd; --accent: #14417a;
    --accent-soft: #eeeeee;
  }
  body { font-size: 10.5pt; padding: 0; background: #ffffff; }
  .sheet { max-width: none; }
  nav.toc { break-after: page; }
  section.block { break-inside: auto; border: none; border-radius: 0; padding: 0 0 .5rem;
    margin-bottom: 1.2rem; border-top: 1px solid var(--line); padding-top: .8rem; }
  header.masthead { border: none; border-top: none; padding: 0 0 1rem; margin: 0 0 1rem; }
  h2, h3, h4 { break-after: avoid; }
  table, tr, .tablewrap { break-inside: avoid; }
  thead { display: table-header-group; }
}
"""


_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_CODE = re.compile(r"`([^`]+)`")


def _inline(text: str) -> str:
    """Escape, then apply the two inline markers the narrative uses. Escaping comes first."""
    escaped = html.escape(str(text), quote=False)
    escaped = _CODE.sub(lambda m: f"<code>{m.group(1)}</code>", escaped)
    escaped = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", escaped)
    return escaped


_NUMERIC = re.compile(r"^[\s$+\-]*[\d][\d,.\s%hoursx/]*$", re.IGNORECASE)


def _is_numeric(value: str) -> bool:
    return bool(_NUMERIC.match(value.strip())) and any(ch.isdigit() for ch in value)


def _render_table(table: ReportTable) -> list[str]:
    if table.is_empty:
        return []
    out = ['<div class="tablewrap">', "<table>"]
    if table.caption:
        out.append(f"<caption>{_inline(table.caption)}</caption>")
    out.append("<thead><tr>")
    out.extend(f"<th>{_inline(column)}</th>" for column in table.columns)
    out.append("</tr></thead>")
    out.append("<tbody>")
    width = len(table.columns)
    for row in table.rows:
        cells = [str(value) for value in row]
        cells += [""] * (width - len(cells))
        out.append("<tr>")
        for value in cells[:width]:
            css = ' class="num"' if _is_numeric(value) else ""
            out.append(f"<td{css}>{_inline(value)}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    if table.note:
        out.append(f'<p class="note">{_inline(table.note)}</p>')
    return out


def _render_section(section: ReportSection, *, top: bool) -> list[str]:
    if section.is_empty:
        return []
    level = max(2, min(6, section.level))
    out: list[str] = []
    if top:
        out.append(f'<section class="block" id="{html.escape(section.id, quote=True)}">')
    else:
        out.append(f'<div id="{html.escape(section.id, quote=True)}">')
    out.append(f"<h{level}>{_inline(section.title)}</h{level}>")
    if section.lead:
        out.append(f'<p class="lead">{_inline(section.lead)}</p>')
    for paragraph in section.paragraphs:
        out.append(f"<p>{_inline(paragraph)}</p>")
    if section.bullets:
        out.append("<ul>")
        out.extend(f"<li>{_inline(bullet)}</li>" for bullet in section.bullets)
        out.append("</ul>")
    for table in section.tables:
        out += _render_table(table)
    for sub in section.subsections:
        out += _render_section(sub, top=False)
    out.append("</section>" if top else "</div>")
    return out


def render_html(
    report: AssessmentReport, *, theme: str = "", embedded: bool = False
) -> str:
    """Render the whole report as one self-contained HTML document.

    ``theme`` is ``"light"`` or ``"dark"`` when a host page has its own light/dark control
    and wants this document to agree with it; anything else leaves the document following
    the reader's system preference, which is the right default for a file opened on its
    own. ``embedded`` drops the page furniture that would duplicate a host page's own.

    Both are presentation, and neither changes a word of the report.
    """
    sections = [section for section in report.sections if not section.is_empty]

    root_attrs = ' lang="en"'
    if str(theme).lower() in {"light", "dark"}:
        root_attrs += f' data-theme="{str(theme).lower()}"'
    if embedded:
        root_attrs += " data-embedded"

    parts: list[str] = [
        "<!DOCTYPE html>",
        f"<html{root_attrs}>",
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="generator" content="vulnpriority assessment report">',
        f"<title>{html.escape(report.title, quote=False)}</title>",
        f"<style>{STYLESHEET}</style>",
        "</head>",
        "<body>",
        '<div class="sheet">',
        '<header class="masthead">',
        f"<h1>{_inline(report.title)}</h1>",
    ]
    if report.subtitle:
        parts.append(f'<p class="subtitle">{_inline(report.subtitle)}</p>')
    stamp_bits = [
        f"run {report.meta.run_id}",
        f"config {report.meta.config_hash}",
        f"vulnpriority {report.meta.framework_version}",
    ]
    if report.meta.generated_at is not None:
        stamp_bits.append(f"generated {report.meta.generated_at.isoformat()}")
    stamp = " &middot; ".join(_inline(bit) for bit in stamp_bits)
    parts.append(f'<p class="stamp">{stamp}</p>')
    parts.append("</header>")

    if len(sections) >= 3:
        parts.append('<nav class="toc"><h2>Contents</h2><ol>')
        for section in sections:
            anchor = html.escape(section.id, quote=True)
            parts.append(f'<li><a href="#{anchor}">{_inline(section.title)}</a></li>')
        parts.append("</ol></nav>")

    for section in sections:
        parts += _render_section(section, top=True)

    parts += [
        '<footer class="colophon">',
        _inline(
            "Produced by vulnpriority. Every figure in this document is either read from the run "
            "or derived from it by the arithmetic stated on the page."
        ),
        "</footer>",
        "</div>",
        "</body>",
        "</html>",
        "",
    ]
    return "\n".join(parts)
