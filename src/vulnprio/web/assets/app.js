/* vulnprio dashboard.
 *
 * Plain ES2020, no build step and no external script: the page must open from file:// on a
 * machine with no network, because that is how the framework itself is designed to run.
 * Data arrives as window.VULNPRIO_DATA from data.js, written by vulnprio.web.exporter.
 *
 * The page has two lives. Opened from file:// it is exactly what it always was: a static
 * report over the bundled run. Served by vulnprio.web.app it gains an Analyze tab that can
 * produce a new run, and a Report and Novelty tab to go with it. Which life it is in is
 * decided by one probe of /api/health at startup: if that fails - and under file:// it
 * always fails - the extra tabs stay hidden and nothing else on the page behaves
 * differently. The interactive code lives in analyze.js and is only fetched when the probe
 * succeeds, so the static export never asks for a file it does not ship.
 */

(() => {
  "use strict";

  let D = window.VULNPRIO_DATA || {};
  let findings = D.findings || [];
  let graphs = D.graphs || [];
  let metrics = D.metrics || [];

  // ------------------------------------------------------------ formatting

  /* Precision is fixed per quantity and never varies, so a column of figures reads as a
     column: money to two significant figures above ten thousand and exact below it,
     probabilities to one decimal, hours to one decimal. */
  const sig2 = (n) => {
    if (!n) return 0;
    const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(n))) - 1);
    return Math.round(n / mag) * mag;
  };

  /* ---------------------------------------------------------------- money
   *
   * Two formatters, and the split between them is a design decision rather than an
   * accident. A headline figure is read out loud, so it is written the way it is said:
   * "₹1.23 crore", not eight digits a reader has to count. A table column is read
   * downwards and compared, and a column that mixes "1.2 crore" with "45 lakh" cannot be
   * scanned at all - so a table gets plain grouped digits at a consistent width, right
   * aligned, in tabular figures.
   *
   * Indian grouping is three digits and then twos (₹1,23,45,678, never ₹12,345,678).
   * Intl.NumberFormat with the en-IN locale does that correctly; hand-rolled grouping does
   * not, so it is not hand-rolled.
   *
   * The currency comes from the payload rather than from a constant, so a run configured
   * for another currency formats as that currency instead of printing rupee groupings
   * behind the wrong symbol. It is read defensively: this page has to keep working against
   * a payload written before the field existed.
   */
  const currencyCode = () => ((D.meta && D.meta.currency) || "INR").toUpperCase();
  const currencyLocale = () => (currencyCode() === "INR" ? "en-IN" : "en-US");

  const fmtCache = new Map();
  function currencyFormat() {
    const key = `${currencyLocale()}|${currencyCode()}`;
    if (!fmtCache.has(key)) {
      let f;
      try {
        f = new Intl.NumberFormat(currencyLocale(), {
          style: "currency", currency: currencyCode(),
          minimumFractionDigits: 0, maximumFractionDigits: 0,
        });
      } catch (_) {
        f = new Intl.NumberFormat(currencyLocale(), { maximumFractionDigits: 0 });
      }
      fmtCache.set(key, f);
    }
    return fmtCache.get(key);
  }
  /** The currency symbol on its own, for the one place a figure is split from its unit. */
  function currencySymbol() {
    try {
      return currencyFormat().formatToParts(1).find((p) => p.type === "currency").value;
    } catch (_) {
      return "";
    }
  }

  /** Table money: grouped digits, no magnitude word, so a column stays comparable. */
  const usd = (v) => {
    if (v == null || !isFinite(v)) return "–";
    const n = Math.abs(v) < 10000 ? Math.round(v) : sig2(v);
    return currencyFormat().format(n);
  };

  /* Lakh and crore for anything a person reads as a sentence. The western magnitudes are
     kept for a run configured in another currency, where "crore" would be nonsense. There is
     deliberately no word for a thousand in either list: nobody says "ten point eight
     thousand rupees", they say the number, so below the first magnitude a figure is written
     out in full with its grouping. */
  const INDIAN_UNITS = [[1e7, "crore"], [1e5, "lakh"]];
  const WESTERN_UNITS = [[1e9, "billion"], [1e6, "million"]];

  /** ``{ figure, unit }`` - the number said out loud, split so the unit can be set smaller. */
  function moneyParts(v) {
    if (v == null || !isFinite(v)) return { figure: "–", unit: "" };
    const units = currencyCode() === "INR" ? INDIAN_UNITS : WESTERN_UNITS;
    const abs = Math.abs(v);
    const step = units.find(([size]) => abs >= size);
    if (!step) return { figure: usd(v), unit: "" };
    const scaled = v / step[0];
    const dp = Math.abs(scaled) >= 100 ? 0 : Math.abs(scaled) >= 10 ? 1 : 2;
    return { figure: `${currencySymbol()}${scaled.toFixed(dp)}`, unit: step[1] };
  }
  /** Money in running prose: "₹1.23 crore". */
  const moneyText = (v) => {
    const { figure, unit } = moneyParts(v);
    return unit ? `${figure} ${unit}` : figure;
  };
  /** Money as the answer: the same thing, with the unit set subordinate to the figure. */
  const moneyLead = (v) => {
    const { figure, unit } = moneyParts(v);
    return unit ? `${esc(figure)}<span class="unit"> ${esc(unit)}</span>` : esc(figure);
  };

  /* The payload is dropping the ``_usd`` suffix from every money field, because a field
     named for dollars that holds rupees is a lie. Every money read goes through here, so
     the rename is one edit rather than forty and the page renders correctly on either side
     of it. Once the payload has settled this collapses to a plain property read. */
  const cash = (obj, name) => {
    if (!obj) return undefined;
    const v = obj[name];
    return v === undefined ? obj[`${name}_usd`] : v;
  };

  /** What a finding carries: its expected loss, raised by what fixing it also closes. */
  const riskOf = (f) => cash(f, "chain_adjusted") || cash(f, "expected_loss") || 0;
  /** Probabilities: always one decimal place. */
  const pct = (v, dp = 1) => (v == null || !isFinite(v) ? "–" : `${(v * 100).toFixed(dp)}%`);
  /** Shares of a whole read better without the decimal, and are never probabilities. */
  const share = (v) => (v == null || !isFinite(v) ? "–" : `${Math.round(v * 100)}%`);
  /** Effort: always one decimal place, always with its unit. */
  const hrs = (v) => (v ? `${Number(v).toFixed(1)} h` : "–");
  const num = (v, dp = 3) => (v == null || !isFinite(v) ? "–" : v.toFixed(dp));
  const esc = (s) =>
    String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const el = (html) => {
    const t = document.createElement("template");
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
  };
  const PRIV = ["anyone", "an authenticated user", "an administrator", "the host itself"];

  /* How a finding's evidence is described to someone who has to decide whether to trust it.
     The tier number is a real thing in the framework, but nobody reading a queue of bugs
     needs to know that; they need to know whether a claim came from a feed or from a blog
     post, and whether that matters. */
  const PROVENANCE = [
    "Derived entirely from the supplied configuration.",
    "Derived entirely from curated vulnerability feeds: the national database, the known-exploited catalogue, published exploit indexes.",
    "Derived entirely from what the scanner observed directly on the application.",
    "Partly derived from an advisory or write-up fetched from the internet. Such pages can be incorrect, out of date or deliberately misleading, so how far they are permitted to move a finding is capped.",
    "Partly derived from text the application itself returned. That text is written by whoever controls the application, so its influence is capped tightly and it can never override a known-exploited listing.",
  ];

  /* The likelihood model records a named term per piece of evidence. These are those terms
     said out loud: [what a positive weight means, what a negative one means]. An empty
     string means that direction is not worth a line. */
  const LIKELIHOOD_LABELS = {
    kev: ["it is on the government's known-exploited catalogue", ""],
    kev_ransomware: ["it has been used in ransomware campaigns", ""],
    epss_logit: ["public scoring indicates exploitation is likely in the near term", "public scoring indicates exploitation is unlikely in the near term"],
    exploit_maturity: ["working exploit code is publicly available", ""],
    feasibility: ["it is straightforward to exploit as deployed", "it is awkward to exploit as deployed"],
    applicability: ["the deployed version is affected", "the deployed version does not appear to be affected"],
    exposure: ["it is reachable without authentication", "it is not reachable anonymously"],
    asset_criticality: ["it sits on a high-value part of the application", "it sits on a low-value part of the application"],
    complexity_high: ["", "it requires unusual conditions to coincide"],
    user_interaction: ["", "it requires a user to be deceived into acting"],
    privileges_required: ["", "an attacker requires an account first"],
    skill: ["it is well within this attacker's skill", "it is beyond this attacker's usual skill"],
    resources: ["it is well within this attacker's resources", "it is beyond this attacker's resources"],
  };

  /** The strongest reasons this finding is judged likely (or unlikely) to be exploited. */
  function likelihoodReasons(terms, limit = 3) {
    return Object.entries(terms || {})
      .filter(([name]) => name !== "intercept" && name !== "w_intercept")
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .map(([name, value]) => {
        const pair = LIKELIHOOD_LABELS[name.replace(/^w_/, "")];
        if (!pair) return null;
        const text = value >= 0 ? pair[0] : pair[1];
        return text ? { text, value } : null;
      })
      .filter(Boolean)
      .slice(0, limit);
  }

  /** True when the ranking leaned hard on text we could not corroborate. */
  const needsSecondLook = (f) =>
    (f.alerts || []).length > 0 || (f.untrusted_influence_share || 0) >= 0.35;

  function severityChip(sev) {
    const s = String(sev || "").toLowerCase();
    const cls = s === "critical" ? "crit" : s === "high" ? "high" : s === "medium" ? "med" : "low";
    return `<span class="chip ${cls}">${esc(s || "unrated")}</span>`;
  }

  /** Severity as a letter in a coloured box: legible without colour, and compact in a table. */
  function severityMark(sev) {
    const s = String(sev || "unrated").toLowerCase();
    const letter = { critical: "C", high: "H", medium: "M", low: "L", info: "I" }[s] || "?";
    return `<span class="sev ${esc(s)}" aria-hidden="true">${letter}</span>` +
           `<span class="sr-only">severity ${esc(s)}. </span>`;
  }

  /** Every chart gets a line saying what to take from it. */
  function caption(id, text) {
    const host = document.getElementById(id);
    if (host && text) host.appendChild(el(`<p class="caption">${esc(text)}</p>`));
  }

  // ------------------------------------------------------------ tiny chart helpers

  const SVG = "http://www.w3.org/2000/svg";
  function svg(w, h) {
    const s = document.createElementNS(SVG, "svg");
    s.setAttribute("viewBox", `0 0 ${w} ${h}`);
    s.setAttribute("width", w);
    s.setAttribute("class", "chart");
    return s;
  }
  function tag(name, attrs, text) {
    const n = document.createElementNS(SVG, name);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  }
  /* Listeners must survive a data reload without stacking up: the render functions are
     called again for every new run, and an addEventListener in a render function is how a
     page ends up firing the same handler nine times. */
  const wired = new WeakMap();
  function wireOnce(node, event, handler) {
    if (!node) return;
    let seen = wired.get(node);
    if (!seen) wired.set(node, (seen = new Set()));
    if (seen.has(event)) return;
    seen.add(event);
    node.addEventListener(event, handler);
  }

  /* Charts are drawn at the width of the box they are going into, not at a width chosen
     when this file was written. An SVG with a fixed viewBox scaled down to fit a narrow
     column takes its type down with it, and 4px axis labels are worse than no chart. */
  function hostWidth(id, preferred) {
    const host = document.getElementById(id);
    const available = host ? Math.floor(host.clientWidth) : 0;
    if (!available) return preferred;
    return Math.max(260, Math.min(preferred, available));
  }

  function mount(id, node, emptyMsg) {
    const host = document.getElementById(id);
    if (!host) return;
    host.innerHTML = "";
    if (!node) {
      // An empty state is a sentence someone can act on, never a shrug. The default says
      // where the data would have come from, because "no data" tells nobody anything.
      host.appendChild(el(`<p class="empty">${esc(
        emptyMsg || "No data yet. Run an assessment from the Analyze tab."
      )}</p>`));
      markScrollable(host);
      return;
    }
    host.appendChild(node);
    markScrollable(host);
  }

  /** Horizontal bars with optional confidence whiskers. */
  function barChart(rows, opts = {}) {
    if (!rows.length) return null;
    const rowH = opts.rowH || 26;
    const w = opts.width || 640;
    /* The label gutter is a share of the chart, never a fixed number of pixels: at 300px a
       230px gutter leaves no chart, and at 900px it wastes half of one. */
    const labelW = Math.max(70, Math.min(opts.labelW || 150, Math.round(w * 0.42)));
    const fit = Math.max(6, Math.floor((labelW - 10) / 5.6));
    rows = rows.map((r) => (String(r.label).length > fit
      ? { ...r, label: `${String(r.label).slice(0, fit - 1)}…` } : r));
    const h = rows.length * rowH + 34;
    const plotW = Math.max(40, w - labelW - 70);
    const maxAbs = Math.max(...rows.map((r) => Math.abs(r.value)), ...rows.map((r) => Math.abs(r.ciHigh ?? 0)), 1e-9);
    const hasNeg = rows.some((r) => r.value < 0);
    const zeroX = hasNeg ? labelW + plotW / 2 : labelW;
    const scale = (v) => (v / maxAbs) * (hasNeg ? plotW / 2 : plotW);
    const s = svg(w, h);

    s.appendChild(tag("line", { class: "gridline", x1: zeroX, y1: 14, x2: zeroX, y2: h - 20 }));
    rows.forEach((r, i) => {
      const y = 18 + i * rowH;
      const len = scale(r.value);
      const x = len >= 0 ? zeroX : zeroX + len;
      s.appendChild(tag("text", { x: labelW - 8, y: y + 12, "text-anchor": "end" }, r.label));
      s.appendChild(tag("rect", {
        x, y: y + 2, width: Math.max(Math.abs(len), 1), height: rowH - 10,
        fill: r.color || "var(--accent)", rx: 2, opacity: r.dim ? 0.45 : 0.92,
      }));
      if (r.ciLow != null && r.ciHigh != null) {
        const x1 = zeroX + scale(r.ciLow), x2 = zeroX + scale(r.ciHigh), my = y + (rowH - 10) / 2 + 2;
        s.appendChild(tag("line", { class: "whisker", x1, y1: my, x2, y2: my }));
        s.appendChild(tag("line", { class: "whisker", x1, y1: my - 4, x2: x1, y2: my + 4 }));
        s.appendChild(tag("line", { class: "whisker", x1: x2, y1: my - 4, x2, y2: my + 4 }));
      }
      s.appendChild(tag("text", {
        class: "bar-label", x: (len >= 0 ? zeroX + Math.abs(len) + 6 : x - 6),
        y: y + 13, "text-anchor": len >= 0 ? "start" : "end",
      }, r.display != null ? r.display : num(r.value)));
    });
    return s;
  }

  /** Multi-series line chart on a shared index axis. */
  function lineChart(series, opts = {}) {
    if (!series.length || !series.some((s) => s.values.length)) return null;
    const w = opts.width || 620, h = opts.height || 260;
    const m = { top: 14, right: 14, bottom: 30, left: 56 };
    const n = Math.max(...series.map((s) => s.values.length));
    const maxY = opts.maxY != null ? opts.maxY : Math.max(...series.flatMap((s) => s.values), 1e-9);
    const minY = opts.minY != null ? opts.minY : 0;
    const px = (i) => m.left + (n <= 1 ? 0 : (i / (n - 1)) * (w - m.left - m.right));
    const py = (v) => h - m.bottom - ((v - minY) / (maxY - minY || 1)) * (h - m.top - m.bottom);
    const s = svg(w, h);

    for (let g = 0; g <= 4; g++) {
      const v = minY + ((maxY - minY) * g) / 4, y = py(v);
      s.appendChild(tag("line", { class: "gridline", x1: m.left, y1: y, x2: w - m.right, y2: y }));
      s.appendChild(tag("text", { x: m.left - 7, y: y + 4, "text-anchor": "end" }, opts.fmtY ? opts.fmtY(v) : num(v, 1)));
    }
    s.appendChild(tag("text", { x: m.left, y: h - 8 }, opts.xLabelStart || "0"));
    s.appendChild(tag("text", { x: w - m.right, y: h - 8, "text-anchor": "end" }, opts.xLabelEnd || String(n)));

    series.forEach((ser) => {
      if (!ser.values.length) return;
      const d = ser.values.map((v, i) => `${i ? "L" : "M"}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
      s.appendChild(tag("path", {
        d, fill: "none", stroke: ser.color, "stroke-width": ser.emphasis ? 2.6 : 1.6,
        "stroke-dasharray": ser.dashed ? "5 4" : "", opacity: ser.emphasis ? 1 : 0.85,
      }));
    });
    return s;
  }

  function legend(items) {
    return el(`<div class="legend">${items
      .map((i) => `<span><i style="background:${i.color}"></i>${esc(i.label)}</span>`)
      .join("")}</div>`);
  }

  /* Tables carry a priority per column. One is never dropped; two, three and four appear as
     the table's own container gets wide enough to hold them honestly. The stylesheet does
     the dropping - this only has to say which columns matter, which is a question about the
     data and belongs here. Columns with no stated priority are priority one, so a matrix
     (metrics by ranker, ablation cells) keeps every column and scrolls instead: dropping one
     observation from a matrix would change what the table says. */
  function table(headers, rows, opts = {}) {
    const pri = (h) => (h.pri && h.pri > 1 ? ` data-pri="${h.pri}"` : "");
    const head = headers
      .map((h) => `<th class="${h.num ? "num" : ""}"${pri(h)}>${esc(h.label)}</th>`).join("");
    const body = rows
      .map((r) => `<tr${r._cls ? ` class="${r._cls}"` : ""}>${headers
        .map((h) => `<td class="${h.num ? "num" : ""}"${pri(h)}>${r[h.key] == null ? "–" : r[h.key]}</td>`)
        .join("")}</tr>`)
      .join("");
    return el(`<table${opts.cls ? ` class="${opts.cls}"` : ""}><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`);
  }

  /* When a column is dropped its content does not disappear - it folds into the row's first
     cell as a second line, and the stylesheet shows that line only while those columns are
     away. Nothing is lost at a narrow width, only moved. */
  const fold = (...parts) => {
    const kept = parts.filter(Boolean).map((p) => breakable(p));
    return kept.length ? `<span class="rowfold">${kept.join('<span class="sep">·</span>')}</span>` : "";
  };

  /* A URL path has to be allowed to break, or it pushes its column wider than the table. Left
     to itself the browser breaks it mid-word - "/api/v1/files/upl oad" - so it is given break
     opportunities at the separators instead, and breaks where a reader would. */
  const breakable = (text) => esc(text).replace(/([/._-])/g, "$1<wbr>");

  /* A scroll container has to say that it scrolls, and it may only say so when it is true.
     CSS cannot ask whether something overflows, so this is measured and flagged, here and
     again whenever the page changes width. */
  const SCROLLERS = ".tablewrap, .graphbox";
  function markScrollable(node) {
    const boxes = node
      ? [node.closest(SCROLLERS)].filter(Boolean)
      : [...document.querySelectorAll(SCROLLERS)];
    boxes.forEach((box) => {
      box.classList.toggle("is-scrollable", box.scrollWidth - box.clientWidth > 2);
    });
  }

  // ------------------------------------------------------------ header

  /* What was assessed, and nothing else.
   *
   * The header used to carry five lines: target, scan date, scanner and version, attacker
   * model, and the day intelligence was gathered. All of it is true and all of it matters,
   * but none of it is what the reader came for, and stacking it above the answer put five
   * lines of metadata between a person and the thing they opened the page to see. It is
   * provenance, and the generated report has a Provenance section that carries every one of
   * those facts in sentences. Run ids, hashes, seeds and backends stay in the footer, where
   * someone reproducing a number goes looking.
   *
   * One thing earns a place in the chrome: which system these numbers are about, because
   * without it every figure on the page is unattributed. It is set quietly. */
  function targetName() {
    const source = (D.notes || {}).source || {};
    const scan = (D.scans || [])[0];
    if (source.target_url) return String(source.target_url);
    if (scan && (scan.app_name || scan.app_id)) return String(scan.app_name || scan.app_id);
    if (source.report_filename) return String(source.report_filename);
    return "";
  }

  /* Caveats are not provenance: "these figures are illustrative" and "nobody checked for
     live exploit intelligence" change how far a reader should trust the numbers, so they
     travel with the numbers rather than living in the chrome. */
  function dataCaveats() {
    const notes = D.notes || {};
    const intel = notes.intel || {};
    const out = [];
    if (notes.demonstration) {
      out.push("These are demonstration figures from a synthetic run, not an assessment of a real system.");
    }
    if (intel.requested === false) {
      out.push(intel.disabled_reason ||
        "No live exploit intelligence was gathered, so nothing here reflects what is being exploited today.");
    }
    /* How the queue was ordered. "Ranked by a learned model" is a claim, and it was being
       made on every single-scan run while the ordering was actually expected loss. Only the
       case where no model ran is worth a caveat; the other two are the normal ones. */
    const ranking = notes.ranking || {};
    if (ranking.fallback_reason) {
      out.push(
        "These findings are ordered by expected loss, not by the learned model: "
        + `${ranking.fallback_reason}, and no trained model was available to score them.`
      );
    }
    /* What the scan could not reach. This is the most consequential caveat on the page and
       the easiest to omit: a short queue from an application whose routes live in
       JavaScript is indistinguishable from a short queue from a well-configured one, and a
       reader who is not told which they have will assume the flattering reading. */
    const coverage = ((notes.scanner || {}).coverage_notes) || [];
    for (const note of coverage) {
      const text = String(note || "").trim();
      if (text) out.push(text.charAt(0).toUpperCase() + text.slice(1));
    }
    return out;
  }

  function renderMeta() {
    const m = D.meta || {};
    const target = targetName();
    const slot = document.getElementById("runmeta");
    slot.textContent = target;
    slot.title = target ? `Assessed: ${target}` : "";
    slot.hidden = !target;

    const comps = Object.entries(m.components || {})
      .filter(([, on]) => on).map(([k]) => k.toUpperCase()).join("") || "none";
    const scanner = (D.notes || {}).scanner || {};
    const scan = (D.scans || [])[0];
    const intel = (D.notes || {}).intel || {};
    /* The facts the header used to hold. They are kept, in full, where a reader checking a
       number will look for them - and the report says all of it again in prose. */
    const provenance = [
      scan && scan.scanned_at ? `scanned ${esc(String(scan.scanned_at).slice(0, 10))}` : "",
      scanner.tool
        ? `by ${esc(scanner.tool)}${scanner.tool_version ? ` ${esc(scanner.tool_version)}` : ""}`
        : (scan && scan.scanner ? `by ${esc(scan.scanner)}` : ""),
      m.attacker ? `assuming ${esc(String(m.attacker).replace(/_/g, " "))} attackers` : "",
      intel.ran && intel.gathered_on
        ? `intelligence gathered ${esc(intel.gathered_on)}`
        : (intel.requested === false ? "no live intelligence" : ""),
    ].filter(Boolean).join(" · ");

    const technical = [
      m.run_id ? `run <span class="mono">${esc(m.run_id)}</span>` : "",
      m.config_hash ? `config <span class="mono">${esc(m.config_hash)}</span>` : "",
      m.dataset_hash ? `data <span class="mono">${esc(m.dataset_hash)}</span>` : "",
      `components ${esc(comps)}`,
      m.llm_backend ? `assessor ${esc(m.llm_backend)}` : "",
      m.feed_mode ? `feeds ${esc(m.feed_mode)}` : "",
      m.seeds && m.seeds.length ? `seeds ${esc(m.seeds.join(", "))}` : "",
    ].filter(Boolean).join(" · ");

    const foot = document.getElementById("foot");
    if (!hasData()) {
      // Nothing has been assessed, so there is nothing to caveat and no run to identify.
      foot.innerHTML = `vulnprio ${esc(m.package_version || "")} - all processing is local to this machine.`;
      return;
    }
    foot.innerHTML =
      `Monetary figures are modelled estimates, not measurements: the selected impact model applied
       to what the scan found. Use them to compare findings with one another, not as a
       forecast.<br>` +
      (provenance ? `${provenance}<br>` : "") +
      `vulnprio ${esc(m.package_version || "")} · generated
       ${esc((m.generated_at || "").toString().slice(0, 19).replace("T", " "))}` +
      (technical ? `<br><span class="foottech">${technical}</span>` : "");
  }

  // ------------------------------------------------------------ overview

  function renderOverview() {
    const s = D.summary || {};
    const kev = findings.filter((f) => f.kev).length;
    const chokes = findings.filter((f) => f.is_chokepoint).length;
    const secondLook = findings.filter(needsSecondLook).length;
    const plan = (D.selections || [])[0];
    const inPlan = findings.filter((f) => f.selected_in_budget);
    const planHours = inPlan.reduce((t, f) => t + (f.remediation_hours || 0), 0);

    /* The supporting figures. The total at risk is deliberately absent: it is the lead, and
       repeating it here as one tile among six was the clearest sign that nothing on this
       page was more important than anything else.

       Nothing here borrows a money figure a colour. A magnitude is neither good nor bad, it
       is large or small, and the size of the type already says which. Colour is kept for the
       two counts where it carries information the number does not. */
    const tiles = [
      {
        k: "of it in the top tenth", v: share(s.top_decile_loss_share),
        n: "a short list captures most of the value",
      },
      {
        k: "to fix first", v: inPlan.length || Math.min(10, findings.length),
        n: inPlan.length
          ? `${hrs(planHours)} of work - see Plan`
          : "begin at the top of the list",
      },
      {
        k: "exploited in the wild", v: kev,
        n: kev ? "in active exploitation" : "none on the known-exploited catalogue",
        cls: kev ? "bad" : "good",
      },
      { k: "chokepoints", v: chokes, n: "fixing one closes several routes" },
      {
        k: "findings", v: s.n_findings || findings.length,
        n: `${s.n_clusters || 0} distinct causes across ${s.n_endpoints || 0} endpoints`,
      },
    ];
    if (secondLook) {
      tiles.push({
        k: "unverified evidence", v: secondLook,
        n: "positions resting partly on uncorroborated sources",
      });
    }
    document.getElementById("overviewTiles").innerHTML = findings.length ? tiles
      .map((t) => `<div class="tile ${t.cls || ""}"><div class="k">${esc(t.k)}</div>
        <div class="v">${t.v}</div><div class="n">${esc(t.n)}</div></div>`).join("") : "";

    /* The answer, and its explanation beside it rather than in a matching box. Someone who
       reads only the top of this page should still come away knowing the size of the problem
       and where to start. Money reads as a sentence here - "₹1.23 crore", the way a person
       would say it - because this figure is read out loud, not compared down a column. */
    const lead = findings.filter((f) => f.rank === 1)[0];
    const caveats = dataCaveats();
    document.getElementById("overviewHero").innerHTML = findings.length ? `
      <div class="lead-inner">
        <div>
          <div class="leadnum">${moneyLead(cash(s, "total_expected_loss"))}</div>
          <div class="leadlabel">estimated risk carried across all findings${
            targetName() ? ` on <b>${esc(targetName())}</b>` : ""}</div>
        </div>
        <div>
          <p class="leadtext">
            <b>${share(s.top_decile_loss_share)}</b> of it sits in the top tenth of
            ${s.n_findings || findings.length} findings.
            ${lead ? `Begin with <b>${esc(lead.name)}</b> on
              <span class="mono path">${breakable(lead.endpoint_path)}</span>${lead.remediation_hours
                ? ` - approximately ${lead.remediation_hours.toFixed(1)} hours of work` : ""}.` : ""}
          </p>
          <p class="leadnote">Modelled estimates, for comparing findings with one another
            rather than as a forecast.${caveats.length
              ? ` ${esc(caveats.join(" "))}` : ""}</p>
        </div>
      </div>` : "";

    // The headline: what a severity list would have cost you here.
    const banner = document.getElementById("overviewBanner");
    const promoted = findings.filter(
      (f) => f.rank && f.rank <= 10 && (f.ranks_by_policy || {}).cvss_only > 10);
    if (promoted.length) {
      const worth = promoted.reduce((t, f) => t + (riskOf(f) || 0), 0);
      banner.innerHTML = `<div class="banner"><b>${promoted.length} of the top ten</b> would not appear
        in a severity-ordered top ten - an estimated <b>${esc(moneyText(worth))}</b> of risk that
        would have been reached last.</div>`;
    } else {
      banner.innerHTML = "";
    }

    // Lorenz-style concentration curve of expected loss
    const loss = (f) => cash(f, "expected_loss") || 0;
    const sorted = [...findings].sort((a, b) => loss(b) - loss(a));
    const total = sorted.reduce((t, f) => t + loss(f), 0) || 1;
    let acc = 0;
    const curve = sorted.map((f) => (acc += loss(f)) / total);
    const diag = curve.map((_, i) => (i + 1) / curve.length);
    mount("lorenzChart", lineChart(
      [
        { values: curve, color: "var(--accent)", emphasis: true },
        { values: diag, color: "var(--ink-3)", dashed: true },
      ],
      { maxY: 1, fmtY: (v) => share(v), width: hostWidth("lorenzChart", 620),
        xLabelStart: "highest-loss finding", xLabelEnd: "all findings" }
    ), "No findings in this run.");
    const lh = document.getElementById("lorenzChart");
    if (lh && lh.firstChild) lh.appendChild(legend([
      { color: "var(--accent)", label: "cumulative estimated risk" },
      { color: "var(--ink-3)", label: "if every finding carried equal risk" },
    ]));
    caption("lorenzChart",
      "The further the solid line sits above the dashed one, the more a short list captures. " +
      "If they overlap, ordering makes little difference here and everything should simply be fixed.");

    // biggest movers against CVSS ordering
    const movers = findings
      .filter((f) => f.rank && f.ranks_by_policy && f.ranks_by_policy.cvss_only)
      .map((f) => ({ f, delta: f.ranks_by_policy.cvss_only - f.rank }))
      .filter((m) => m.delta !== 0)
      .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
      .slice(0, 12);
    mount("disagreeChart", barChart(movers.map((m) => ({
      label: shortName(m.f),
      value: m.delta,
      display: `${m.delta > 0 ? "+" : ""}${m.delta}`,
      color: m.delta > 0 ? "var(--ok)" : "var(--crit)",
    })), { labelW: 210, width: hostWidth("disagreeChart", 640) }),
      "No second ordering was computed for this run, so there is nothing to compare against.");
    caption("disagreeChart",
      "Bars are positions moved. A bar at +6 means a severity-ordered list would reach that " +
      "finding six places later than this ordering does.");

    const top = findings.filter((f) => f.rank > 0).sort((a, b) => a.rank - b.rank).slice(0, 10);
    mount("topQueue", top.length ? table(
      [
        { key: "rank", label: "#", num: true, pri: 1 },
        { key: "name", label: "Finding", pri: 1 },
        { key: "where", label: "Location", pri: 2 },
        { key: "sig", label: "", pri: 2 },
        { key: "why", label: "Rationale", pri: 3 },
        { key: "hours", label: "Effort", num: true, pri: 2 },
        { key: "loss", label: "Risk (est.)", num: true, pri: 1 },
      ],
      top.map((f) => ({
        rank: f.rank,
        name: `<b>${esc(f.name)}</b>${f.remediation_summary
          ? `<div class="note clamp2">${esc(f.remediation_summary)}</div>` : ""}${
          fold(`${f.endpoint_method} ${f.endpoint_path}`,
               f.remediation_hours ? hrs(f.remediation_hours) : "")}`,
        where: `<span class="mono path">${esc(f.endpoint_method)} ${breakable(f.endpoint_path)}</span>`,
        sig: signalChips(f, 2),
        why: esc((f.reason_codes || [])[0] || plainWhy(f)),
        hours: f.remediation_hours ? hrs(f.remediation_hours) : "–",
        loss: usd(riskOf(f)),
      })),
      { cls: "roomy" }
    ) : null, "Nothing has been analysed yet.");
  }

  /** A one-line reason when the ranker did not supply a written one. */
  function plainWhy(f) {
    const reasons = likelihoodReasons(f.likelihood_terms, 2).map((r) => r.text);
    if (f.is_chokepoint) reasons.unshift("it closes a route several findings depend on");
    if (!reasons.length) return `${f.endpoint_function || "an"} endpoint; estimated ${moneyText(cash(f, "impact"))} at stake.`;
    return reasons.join("; ") + ".";
  }

  function shortName(f) {
    const n = f.name.length > 26 ? f.name.slice(0, 25) + "…" : f.name;
    return `${n}  ${f.endpoint_path.length > 20 ? f.endpoint_path.slice(0, 19) + "…" : f.endpoint_path}`;
  }

  /** The marks worth carrying on a finding.
   *
   * ``limit`` puts it in table mode: at most that many marks, in their short spelling, with
   * a counter for the rest. A table row that becomes a stack of badges is unreadable, and
   * the drawer is one click away and has room for the full wording. Every short label keeps
   * its long form in the title attribute, so nothing is lost, only folded.
   */
  function signalChips(f, limit) {
    const brief = Boolean(limit);
    const out = [];
    if (f.kev) {
      out.push(`<span class="chip kev" title="On the government's known-exploited catalogue: in active exploitation">${
        brief ? "exploited" : "exploited in the wild"}</span>`);
    }
    if (f.epss != null && f.epss >= 0.05) {
      out.push(`<span class="chip high" title="Public scoring of how likely exploitation is in the next 30 days">${
        brief ? `${share(f.epss)} soon` : `${share(f.epss)} likely soon`}</span>`);
    }
    if (f.is_chokepoint) {
      out.push('<span class="chip choke" title="Several attack routes run through this one finding">chokepoint</span>');
    }
    if (f.applicability === "not_applicable") {
      out.push(`<span class="chip na" title="The deployed version does not appear to be affected">${
        brief ? "not affected" : "version not affected"}</span>`);
    }
    if (needsSecondLook(f)) {
      out.push(`<span class="chip alert" title="Its position rests on a source that could not be corroborated">${
        brief ? "unverified" : "unverified source"}</span>`);
    }
    if (!out.length) out.push(severityChip(f.scanner_severity));
    if (limit && out.length > limit) {
      const hidden = out.length - limit;
      return out.slice(0, limit).join(" ") +
        ` <span class="chip ghost" title="Open the finding for the remaining signals">+${hidden}</span>`;
    }
    return out.join(" ");
  }

  /** One metric per ranker, for the method views. Kept here because they share formatting. */
  function bestMetric(name, k) {
    const pick = (ranker) => {
      const m = metrics.find((x) => x.ranker === ranker && x.metric.startsWith(name) && (k == null || x.k === k));
      return m ? m.value : null;
    };
    return { learned: pick("lambdamart"), cvss: pick("cvss_only"), epss: pick("epss_only") };
  }

  // ------------------------------------------------------------ queue

  const queueState = { sort: "rank", dir: 1, policy: "rank", scan: "", q: "", kev: false, choke: false, alerts: false, budget: false };

  /* Identity first, then the number that drives the ordering, then the effort it costs,
     then the evidence behind both. A reader scanning left to right gets "what, where, how
     much, how long" before anything asks them to interpret a probability.

     ``pri`` is what survives a narrow column. One is the irreducible table - which position,
     which finding, what it is worth - and it is never dropped at any width. Two is what you
     need to act: where it is, how long it takes, what the evidence looks like. Three and
     four are the working it out, and they wait for room. When "Where" goes, the path does
     not vanish: it folds into the finding cell. */
  const QUEUE_COLS = [
    { key: "rank", label: "#", num: true, pri: 1, get: (f) => f._rank },
    { key: "name", label: "Finding", pri: 1, get: (f) => f.name },
    { key: "path", label: "Location", pri: 2, get: (f) => f.endpoint_path },
    { key: "loss", label: "Risk (est.)", num: true, pri: 1, get: (f) => cash(f, "expected_loss") },
    { key: "hours", label: "Effort", num: true, pri: 2, get: (f) => f.remediation_hours },
    { key: "sig", label: "Signals", pri: 2, get: (f) => (f.kev ? 1 : 0) },
    { key: "p", label: "Likelihood", num: true, pri: 3, get: (f) => f.p_exploit },
    { key: "vs", label: "vs severity", num: true, pri: 3, get: (f) => (f.ranks_by_policy?.cvss_only ?? 0) - f._rank },
    { key: "impact", label: "Impact", num: true, pri: 4, get: (f) => cash(f, "impact") },
    { key: "chain", label: "Chain risk", num: true, pri: 4, get: (f) => cash(f, "chain_delta") },
    { key: "cvss", label: "CVSS", num: true, pri: 4, get: (f) => f.cvss_base ?? -1 },
  ];

  function currentQueue() {
    let rows = findings.slice();
    if (queueState.scan) rows = rows.filter((f) => f.scan_id === queueState.scan);
    if (queueState.kev) rows = rows.filter((f) => f.kev);
    if (queueState.choke) rows = rows.filter((f) => f.is_chokepoint);
    if (queueState.alerts) rows = rows.filter((f) => (f.alerts || []).length);
    if (queueState.budget) rows = rows.filter((f) => f.selected_in_budget);
    if (queueState.q) {
      const q = queueState.q.toLowerCase();
      rows = rows.filter((f) =>
        `${f.name} ${f.endpoint_path} ${f.endpoint_method} ${(f.cve_ids || []).join(" ")} cwe-${f.cwe_id} ${f.endpoint_function}`
          .toLowerCase().includes(q));
    }
    const policy = queueState.policy;
    rows.forEach((f) => {
      f._rank = policy === "rank" ? f.rank : (f.ranks_by_policy || {})[policy] || 0;
    });
    const col = QUEUE_COLS.find((c) => c.key === queueState.sort) || QUEUE_COLS[0];
    rows.sort((a, b) => {
      const av = col.get(a), bv = col.get(b);
      if (av === bv) return (a._rank || 0) - (b._rank || 0);
      return (av > bv ? 1 : -1) * queueState.dir;
    });
    return rows;
  }

  function renderQueue() {
    const rows = currentQueue();
    const thead = document.querySelector("#queueTable thead");
    const tbody = document.querySelector("#queueTable tbody");
    thead.innerHTML = `<tr>${QUEUE_COLS.map((c) => {
      const active = queueState.sort === c.key;
      const arrow = active ? (queueState.dir === 1 ? "▲" : "▼") : "";
      const pri = c.pri > 1 ? ` data-pri="${c.pri}"` : "";
      return `<th class="sortable ${c.num ? "num" : ""}" data-col="${c.key}"${pri}>${
        esc(c.label)} <span class="arrow">${arrow}</span></th>`;
    }).join("")}</tr>`;
    thead.querySelectorAll("th").forEach((th) => th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (queueState.sort === col) queueState.dir *= -1;
      else { queueState.sort = col; queueState.dir = col === "rank" ? 1 : -1; }
      renderQueue();
    }));

    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="${QUEUE_COLS.length}" class="emptyrow">${
        findings.length
          ? "No finding matches the current filters. Clear the search box or disable a filter."
          : "No findings yet. Run an assessment from the Analyze tab."
      }</td></tr>`;
      document.getElementById("queueCount").textContent = "";
      return;
    }

    /* The top of the list is where the value is, so it is marked rather than left to the
       reader to infer from a rank column they have to find first. */
    const band = (f) => (f._rank <= 3 ? "p1" : f._rank <= 10 ? "p2" : "");

    tbody.innerHTML = rows.map((f) => {
      const d = (f.ranks_by_policy?.cvss_only ?? 0) - f._rank;
      const dcls = d > 0 ? "up" : d < 0 ? "down" : "same";
      const where = `${f.endpoint_method} ${f.endpoint_path}`;
      return `<tr class="clickable ${band(f)}" data-id="${esc(f.finding_id)}"
                  tabindex="0" title="Open the detail for this finding">
        <td class="num rankcell">${severityMark(f.scanner_severity)}${f._rank || "–"}</td>
        <td>${esc(f.name)}${f.cve_ids && f.cve_ids.length
          ? `<div class="mono cveline">${esc(f.cve_ids[0])}</div>` : ""}
          <span class="rowfold">${esc(where)}${f.remediation_hours
            ? `<span class="sep">·</span>${esc(hrs(f.remediation_hours))}` : ""}</span></td>
        <td data-pri="2"><span class="mono path">${breakable(where)}</span></td>
        <td class="num"><b>${usd(cash(f, "expected_loss"))}</b></td>
        <td class="num" data-pri="2">${hrs(f.remediation_hours)}</td>
        <td class="sigcell" data-pri="2">${signalChips(f, 1)}</td>
        <td class="num" data-pri="3">${pct(f.p_exploit)}</td>
        <td class="num" data-pri="3"><span class="delta ${dcls}" title="${d > 0 ? "a severity-ordered list would reach this later"
          : d < 0 ? "a severity-ordered list would reach this sooner" : "same position either way"}">${d > 0 ? "+" : ""}${d || "0"}</span></td>
        <td class="num" data-pri="4">${usd(cash(f, "impact"))}</td>
        <td class="num" data-pri="4">${cash(f, "chain_delta") ? usd(cash(f, "chain_delta")) : "–"}</td>
        <td class="num" data-pri="4">${f.cvss_base == null ? "–" : f.cvss_base.toFixed(1)}</td>
      </tr>`;
    }).join("");
    tbody.querySelectorAll("tr").forEach((tr) => {
      tr.addEventListener("click", () => openDrawer(tr.dataset.id));
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openDrawer(tr.dataset.id); }
      });
    });
    document.getElementById("queueCount").textContent =
      `${rows.length} of ${findings.length} findings` +
      (queueState.budget ? " within the plan" : "");
    markScrollable();
  }

  // ------------------------------------------------------------ drawer

  function impactSplit(f) {
    const parts = [
      { label: "data exposed", v: cash(f, "impact_confidentiality"), color: "var(--gA)" },
      { label: "data altered", v: cash(f, "impact_integrity"), color: "var(--gB)" },
      { label: "downtime", v: cash(f, "impact_availability"), color: "var(--gC)" },
      { label: "reputation", v: cash(f, "impact_reputational"), color: "var(--ink-3)" },
    ].filter((p) => p.v > 0);
    const total = parts.reduce((t, p) => t + p.v, 0);
    if (!total) return "";
    return `
      <div class="stack" style="margin:6px 0 8px">${parts
        .map((p) => `<i style="width:${(p.v / total) * 100}%;background:${p.color}" title="${esc(p.label)} ${esc(usd(p.v))}"></i>`)
        .join("")}</div>
      <div class="legend" style="margin-top:0">${parts
        .map((p) => `<span><i style="background:${p.color}"></i>${esc(p.label)} ${usd(p.v)}</span>`)
        .join("")}</div>`;
  }

  /* The drawer is a dialog: focus goes into it, Tab stays inside it, Escape closes it, and
     focus returns to the row that opened it so a keyboard user does not lose their place. */
  let drawerOpener = null;

  function drawerFocusables() {
    return [...document.getElementById("drawer").querySelectorAll(
      'a[href], button:not(:disabled), input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )].filter((node) => node.offsetParent !== null);
  }

  function trapDrawerFocus(event) {
    if (event.key !== "Tab") return;
    const nodes = drawerFocusables();
    if (!nodes.length) return;
    const first = nodes[0];
    const last = nodes[nodes.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function closeDrawer() {
    const drawer = document.getElementById("drawer");
    if (!drawer.classList.contains("open")) return;
    drawer.classList.remove("open");
    drawer.setAttribute("aria-hidden", "true");
    drawer.removeEventListener("keydown", trapDrawerFocus);
    if (drawerOpener && document.contains(drawerOpener)) drawerOpener.focus();
    drawerOpener = null;
  }

  function openDrawer(id) {
    const f = findings.find((x) => x.finding_id === id);
    if (!f) return;
    drawerOpener = document.activeElement;
    const body = document.getElementById("drawerBody");
    const reasons = likelihoodReasons(f.likelihood_terms, 3);
    const movedBy = (f.ranks_by_policy || {}).cvss_only
      ? f.ranks_by_policy.cvss_only - (f.rank || 0) : 0;

    body.innerHTML = `
      <h3>${esc(f.name)}</h3>
      <div class="drawerpath">${esc(f.endpoint_method)} ${breakable(f.endpoint_path)}</div>
      <div class="pill-row">${signalChips(f)}
        ${f.cwe_id ? `<span class="chip ghost">CWE-${f.cwe_id}</span>` : ""}
        ${(f.cve_ids || []).map((c) => `<span class="chip ghost">${esc(c)}</span>`).join("")}</div>

      ${f.description ? `<p class="drawerdesc">${esc(f.description)}</p>` : ""}

      ${f.remediation_summary ? `<div class="fixbox">
        <div class="fixlabel">Remediation</div>
        <div>${esc(f.remediation_summary)}</div>
        <div class="note">The full write-up, with the affected endpoints, is in the Report tab.</div>
      </div>` : ""}

      <h4 class="drawerh">Cost and likelihood <span class="estimate">estimated</span></h4>
      <dl class="kv">
        <dt>Impact if exploited</dt><dd><b>${usd(cash(f, "impact"))}</b></dd>
        <dt>Probability of exploitation</dt><dd>${pct(f.p_exploit)} over the attacker's horizon</dd>
        <dt>Expected loss</dt><dd><b>${usd(cash(f, "expected_loss"))}</b></dd>
        ${cash(f, "chain_delta") ? `<dt>Chain risk</dt>
          <dd>${usd(cash(f, "chain_delta"))}${f.is_chokepoint ? " - several attack routes run through it" : ""}</dd>` : ""}
        <dt>Remediation effort</dt><dd>${f.remediation_hours ? `approximately ${f.remediation_hours.toFixed(1)} hours` : "unknown"}${
          f.cluster_size > 1 ? ` · the same cause appears on ${f.cluster_size} endpoints and is fixed once` : ""}</dd>
      </dl>
      ${impactSplit(f)}

      ${reasons.length ? `<h4 class="drawerh">What drives the likelihood</h4>
        <ul class="reasons">${reasons.map((r) =>
          `<li>${esc(r.text.charAt(0).toUpperCase() + r.text.slice(1))}.</li>`).join("")}</ul>` : ""}

      ${(f.reason_codes || []).length ? `<h4 class="drawerh">Reasons for this position</h4>
        <ol class="reasons">${f.reason_codes.map((r) => `<li>${esc(r)}</li>`).join("")}</ol>` : ""}

      ${movedBy ? `<p class="note">${movedBy > 0
        ? `A severity-ordered list would have put this at position ${f.ranks_by_policy.cvss_only} - ${movedBy} places later than here.`
        : `A severity-ordered list would have put this at position ${f.ranks_by_policy.cvss_only}, ahead of its position here.`}</p>` : ""}

      <h4 class="drawerh">Evidence</h4>
      <p class="note">${esc(PROVENANCE[f.max_tier_used] || PROVENANCE[2])}</p>
      <dl class="kv">
        <dt>Version applicability</dt>
        <dd>${esc(String(f.applicability || "unknown").replace(/_/g, " "))}${
          f.version_match && f.version_match !== "unknown" ? ` - version evidence ${esc(f.version_match)}` : ""}</dd>
        <dt>Reachable by</dt><dd>${esc(PRIV[f.auth_required] || "unknown")}</dd>
        ${f.exploit_count ? `<dt>Public exploits</dt><dd>${f.exploit_count}</dd>` : ""}
      </dl>
      ${needsSecondLook(f) ? `<div class="banner"><b>Unverified evidence.</b>
        ${(f.alerts || []).length
          ? f.alerts.map(esc).join(" ")
          : `Approximately ${share(f.untrusted_influence_share)} of this finding's position rests on
             text read from a page rather than on a curated feed - typically an advisory that could
             not be corroborated. Verify the finding before acting on its position.`}</div>` : ""}
    `;
    const drawer = document.getElementById("drawer");
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    drawer.addEventListener("keydown", trapDrawerFocus);
    document.getElementById("drawerClose").focus();
  }

  // ------------------------------------------------------------ chains

  function renderChains() {
    const sel = document.getElementById("chainScan");
    if (!graphs.length) {
      mount("graphBox", null,
        "Attack chains were disabled for this run, so no routes were computed.");
      mount("pathStories", null, "");
      mount("pathTable", null, "");
      mount("chainChart", null, "");
      document.getElementById("chainTiles").innerHTML = "";
      return;
    }
    if (!sel.options.length) {
      graphs.forEach((g) => sel.appendChild(el(`<option value="${esc(g.scan_id)}">${esc(scanName(g.scan_id))}</option>`)));
    }
    wireOnce(sel, "change", renderChains);
    wireOnce(document.getElementById("chainPath"), "change", () => drawGraph(currentGraph(), pathIndex()));
    const g = currentGraph();
    const ps = document.getElementById("chainPath");
    ps.innerHTML = `<option value="-1">none</option>` + (g.top_paths || [])
      .map((p, i) => `<option value="${i}">path ${i + 1} · ${moneyText(cash(p, "expected_value"))} expected</option>`).join("");

    const chokes = findings.filter((f) => f.scan_id === g.scan_id && f.is_chokepoint).length;
    document.getElementById("chainTiles").innerHTML = [
      { k: "reachable by chaining", v: esc(moneyText(cash(g, "total_risk"))), n: "estimated value an attacker could reach" },
      { k: "routes worth closing", v: (g.top_paths || []).length, n: "described below" },
      { k: "chokepoints", v: chokes, n: "one fix closes several routes" },
      {
        k: "claims not admitted", v: g.rejected_untrusted_edges || 0,
        n: "steps claimed only by an unverifiable page",
      },
    ].map((t) => `<div class="tile ${t.cls || ""}"><div class="k">${esc(t.k)}</div><div class="v">${t.v}</div><div class="n">${esc(t.n)}</div></div>`).join("");

    drawGraph(g, pathIndex());

    const paths = (g.top_paths || []).slice(0, 8);
    const nameOf = (id) => {
      const f = findings.find((x) => x.finding_id === id);
      return f ? f.name : id;
    };

    // Each route as a sentence, because "state:host:SYSTEM" is not a thing anyone reasons
    // about and "they end up running code on the web server" is.
    mount("pathStories", paths.length ? el(`<div>${paths.map((p, i) => {
      const start = plainState(p.nodes[0]);
      const end = plainState(p.nodes[p.nodes.length - 1]);
      const uses = (p.finding_ids || []).map(nameOf);
      const steps = uses.length
        ? `using <b>${uses.map(esc).join("</b>, then <b>")}</b>`
        : "using steps the scan observed";
      return `<div class="story">
        <div class="storynum">${i + 1}</div>
        <div>
          <p>Starting as <b>${esc(start)}</b>, ${steps}, an attacker reaches
             <b>${esc(end)}</b> - an estimated <b>${esc(moneyText(cash(p, "target_value")))}</b> at stake.</p>
          <p class="note">Approximately ${share(p.probability)} probability of completing the route,
             giving ${esc(moneyText(cash(p, "expected_value")))} of risk along it.
             ${uses.length === 1 ? "Fixing that finding closes it."
               : uses.length > 1 ? `Fixing any one of those ${uses.length} closes it.` : ""}</p>
        </div></div>`;
    }).join("")}</div>`) : null, "No route in this scan reaches an asset of value.");

    mount("pathTable", paths.length ? table(
      [
        { key: "i", label: "#", num: true, pri: 1 },
        { key: "route", label: "Route through the graph", pri: 1 },
        { key: "p", label: "Likelihood", num: true, pri: 2 },
        { key: "e", label: "Risk (est.)", num: true, pri: 1 },
      ],
      paths.map((p, i) => ({
        i: i + 1,
        route: `<span class="mono path">${p.nodes.map((n) => breakable(nodeLabel(n))).join(" → ")}</span>`,
        p: pct(p.probability), e: `<b>${usd(cash(p, "expected_value"))}</b>`,
      }))
    ) : null, "");

    const contrib = findings
      .filter((f) => f.scan_id === g.scan_id && cash(f, "chain_delta") > 0)
      .sort((a, b) => cash(b, "chain_delta") - cash(a, "chain_delta")).slice(0, 14);
    mount("chainChart", barChart(contrib.map((f) => ({
      label: shortName(f), value: cash(f, "chain_delta"), display: usd(cash(f, "chain_delta")),
      color: f.is_chokepoint ? "var(--gC)" : "var(--accent)",
    })), { labelW: 230, width: hostWidth("chainChart", 700) }), "No finding in this scan enables further access.");
    caption("chainChart",
      "Each bar is the reachable risk removed by fixing that finding alone. Teal bars are " +
      "chokepoints, where several routes run through the same fix.");
  }

  /** "state:shop.example.com:SYSTEM" as something a person would say out loud. */
  function plainState(id) {
    const parts = String(id || "").replace(/^state:/, "").split(":");
    const asset = parts[0] || "the application";
    const level = (parts[1] || "").toUpperCase();
    if (asset === "internet") return "an anonymous visitor from the internet";
    return {
      NONE: `an unauthenticated visitor to ${asset}`,
      USER: `an authenticated user of ${asset}`,
      ADMIN: `an administrator of ${asset}`,
      SYSTEM: `control of the ${asset} host itself`,
    }[level] || `${asset} (${level.toLowerCase()})`;
  }

  const pathIndex = () => parseInt(document.getElementById("chainPath").value || "-1", 10);
  const currentGraph = () => {
    const id = document.getElementById("chainScan").value;
    return graphs.find((g) => g.scan_id === id) || graphs[0];
  };
  const scanName = (id) => {
    const s = (D.scans || []).find((x) => x.scan_id === id);
    return s ? `${s.app_name || s.app_id} · ${String(s.scanned_at).slice(0, 10)}` : id;
  };
  const nodeLabel = (id) => String(id).replace(/^state:/, "").replace(/:/g, " ");

  function drawGraph(g, highlight) {
    if (!g) return mount("graphBox", null, "");
    const nodes = g.nodes || [], edges = g.edges || [];
    const cols = [[], [], [], []];
    nodes.forEach((n) => cols[Math.min(3, n.privilege || 0)].push(n));
    const colW = 250, rowH = 54, pad = 26;
    const w = pad * 2 + colW * 4;
    const h = pad * 2 + Math.max(1, ...cols.map((c) => c.length)) * rowH;
    const pos = {};
    cols.forEach((col, ci) => col.forEach((n, ri) => {
      pos[n.id] = { x: pad + ci * colW, y: pad + ri * rowH, w: colW - 58, h: 34, node: n };
    }));

    const hotEdges = new Set();
    if (highlight >= 0 && (g.top_paths || [])[highlight]) {
      const p = g.top_paths[highlight];
      for (let i = 0; i + 1 < p.nodes.length; i++) hotEdges.add(`${p.nodes[i]}->${p.nodes[i + 1]}`);
    }

    /* A node-link diagram cannot be reflowed into a narrow column without becoming a
       different diagram, so it keeps its natural size and its box scrolls. That box is
       marked as scrollable the same way a wide table is. */
    const s = svg(w, h);
    s.setAttribute("class", "chart graph");
    ["none", "user", "admin", "system"].forEach((label, i) =>
      s.appendChild(tag("text", { x: pad + i * colW, y: 14, "font-weight": "600" }, label)));

    edges.forEach((e) => {
      const a = pos[e.src], b = pos[e.dst];
      if (!a || !b) return;
      const hot = hotEdges.has(`${e.src}->${e.dst}`);
      const x1 = a.x + a.w, y1 = a.y + a.h / 2, x2 = b.x, y2 = b.y + b.h / 2;
      const mx = (x1 + x2) / 2;
      s.appendChild(tag("path", {
        class: `edge ${e.kind} ${hot ? "hot" : ""}`,
        d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`,
        "stroke-width": hot ? 2.6 : Math.max(0.6, (e.probability || 0) * 4),
        opacity: hot ? 1 : 0.55,
      }));
    });

    nodes.forEach((n) => {
      const p = pos[n.id];
      if (!p) return;
      const grp = tag("g", { class: "node" });
      const fill = n.is_entry ? "var(--accent-soft)" : n.is_target ? "var(--panel)" : "var(--panel)";
      const stroke = n.is_target ? "var(--crit)" : n.is_entry ? "var(--accent)" : "var(--line)";
      grp.appendChild(tag("rect", { x: p.x, y: p.y, width: p.w, height: p.h, rx: 6, fill, stroke }));
      grp.appendChild(tag("text", { x: p.x + 9, y: p.y + 14 }, nodeLabel(n.id).slice(0, 30)));
      grp.appendChild(tag("text", { x: p.x + 9, y: p.y + 27, "font-size": "10", fill: "var(--ink-3)" },
        cash(n, "value") ? `${usd(cash(n, "value"))} at stake` : n.is_entry ? "attacker entry" : ""));
      s.appendChild(grp);
    });

    mount("graphBox", s, "");
  }

  // ------------------------------------------------------------ evaluation

  function metricKeys() {
    const keys = new Map();
    metrics.forEach((m) => keys.set(`${m.metric}${m.k ? `@${m.k}` : ""}`, m));
    return [...keys.keys()].sort();
  }

  function renderEvaluation() {
    if (!metrics.length) {
      mount("metricChart", null, "This assessment covered a single scan, so the ordering could not be measured against anything. Evaluation requires several scans with confirmed-exploitation history behind them.");
      mount("calChart", null, "");
      mount("minorityTable", null, "");
      mount("metricTable", null, "");
      return;
    }
    const sel = document.getElementById("evalMetric");
    if (!sel.options.length) {
      metricKeys().forEach((k) => sel.appendChild(el(`<option value="${esc(k)}">${esc(k)}</option>`)));
      sel.value = metricKeys().find((k) => k.startsWith("ndcg")) || metricKeys()[0];
    }
    wireOnce(sel, "change", renderEvaluation);
    const key = sel.value;
    const rows = metrics
      .filter((m) => `${m.metric}${m.k ? `@${m.k}` : ""}` === key)
      .sort((a, b) => b.value - a.value);
    mount("metricChart", barChart(rows.map((m) => ({
      label: m.ranker, value: m.value, ciLow: m.ci_low, ciHigh: m.ci_high,
      display: num(m.value), color: m.ranker === "lambdamart" ? "var(--accent)" : "var(--ink-3)",
      dim: m.ranker !== "lambdamart",
    })), { labelW: 160, width: hostWidth("metricChart", 660) }));

    const cal = (D.calibration || [])[0];
    if (cal && cal.bin_confidence.length) {
      const w = hostWidth("calChart", 380), h = 300, m = { t: 14, r: 14, b: 34, l: 46 };
      const s = svg(w, h);
      const px = (v) => m.l + v * (w - m.l - m.r);
      const py = (v) => h - m.b - v * (h - m.t - m.b);
      s.appendChild(tag("line", { class: "gridline", x1: px(0), y1: py(0), x2: px(1), y2: py(1) }));
      for (let g = 0; g <= 4; g++) {
        s.appendChild(tag("text", { x: m.l - 7, y: py(g / 4) + 4, "text-anchor": "end" }, `${g * 25}%`));
        s.appendChild(tag("text", { x: px(g / 4), y: h - 12, "text-anchor": "middle" }, `${g * 25}%`));
      }
      const pts = cal.bin_confidence.map((c, i) => `${i ? "L" : "M"}${px(c).toFixed(1)},${py(cal.bin_accuracy[i]).toFixed(1)}`).join(" ");
      s.appendChild(tag("path", { d: pts, fill: "none", stroke: "var(--accent)", "stroke-width": 2.2 }));
      cal.bin_confidence.forEach((c, i) =>
        s.appendChild(tag("circle", { cx: px(c), cy: py(cal.bin_accuracy[i]), r: 3.5, fill: "var(--accent)" })));
      mount("calChart", s);
      document.getElementById("calChart").appendChild(
        el(`<p class="note">Brier ${num(cal.brier)} · expected calibration error ${num(cal.ece)}. The dashed
            diagonal is perfect calibration.</p>`));
      const per = Object.entries(cal.per_class || {});
      mount("minorityTable", table(
        [{ key: "c", label: "Class", pri: 1 }, { key: "f1", label: "F1", num: true, pri: 1 },
         { key: "sup", label: "Support", num: true, pri: 2 }],
        [
          { c: "<b>overall MCC</b>", f1: num(cal.mcc), sup: "" },
          { c: "<b>balanced accuracy</b>", f1: num(cal.balanced_accuracy), sup: "" },
          { c: "<b>positive rate</b>", f1: share(cal.positive_rate), sup: "" },
          ...per.map(([k, v]) => ({ c: esc(k), f1: num(v.f1), sup: v.support ?? "" })),
        ]
      ));
    } else {
      mount("calChart", null, "No calibrated probability model was fitted, so there is no reliability curve to draw.");
      mount("minorityTable", null, "No per-class breakdown was produced for this run.");
    }

    const byRanker = {};
    metrics.forEach((m) => {
      const k = `${m.metric}${m.k ? `@${m.k}` : ""}`;
      (byRanker[m.ranker] ||= {})[k] = m.value;
    });
    const cols = metricKeys();
    mount("metricTable", table(
      [{ key: "r", label: "Ranker" }, ...cols.map((c) => ({ key: c, label: c, num: true }))],
      Object.entries(byRanker).map(([r, vals]) => {
        const row = { r: r === "lambdamart" ? `<b>${esc(r)}</b>` : esc(r) };
        cols.forEach((c) => (row[c] = vals[c] == null ? "–" : num(vals[c])));
        return row;
      })
    ));
  }

  // ------------------------------------------------------------ ablation

  function renderAblation() {
    const abl = D.ablation || {};
    const effects = abl.main_effects || {};
    const keys = Object.keys(effects);
    if (!keys.length && !(abl.cells || []).length) {
      mount("mainEffectChart", null, "No component ablation ran. The ablation re-runs the whole protocol with each component disabled, which requires labelled history rather than a single scan.");
      mount("interactionTable", null, "");
      mount("ablationTable", null, "");
      return;
    }
    const sel = document.getElementById("ablMetric");
    if (!sel.options.length && keys.length) {
      keys.forEach((k) => sel.appendChild(el(`<option value="${esc(k)}">${esc(k)}</option>`)));
      sel.value = keys.find((k) => k.startsWith("ndcg")) || keys[0];
    }
    wireOnce(sel, "change", renderAblation);
    const key = sel.value || keys[0];
    const eff = effects[key] || {};
    const NAMES = { A: "A · semantic assessment", B: "B · threat intelligence", C: "C · chain awareness" };
    mount("mainEffectChart", barChart(Object.entries(eff).map(([k, v]) => ({
      label: NAMES[k] || k, value: v, display: (v >= 0 ? "+" : "") + num(v),
      color: v >= 0 ? { A: "var(--gA)", B: "var(--gB)", C: "var(--gC)" }[k] || "var(--accent)" : "var(--crit)",
    })), { labelW: 190, width: hostWidth("mainEffectChart", 560) }));

    const inter = (abl.interactions || {})[key] || {};
    mount("interactionTable", Object.keys(inter).length ? table(
      [{ key: "k", label: "Combination", pri: 1 }, { key: "v", label: "Interaction", num: true, pri: 1 }],
      Object.entries(inter).map(([k, v]) => ({ k: esc(k), v: (v >= 0 ? "+" : "") + num(v) }))
    ) : null, "No interaction terms were computed for this metric.");

    const cells = abl.cells || [];
    const allKeys = [...new Set(cells.flatMap((c) => Object.keys(c.mean || {})))].sort();
    mount("ablationTable", cells.length ? table(
      [{ key: "c", label: "Components" }, ...allKeys.map((k) => ({ key: k, label: k, num: true }))],
      cells.map((c) => {
        const row = { c: `<b>${esc(c.label)}</b>`, _cls: c.label === "ABC" ? "selected" : "" };
        allKeys.forEach((k) => {
          const m = c.mean?.[k], sd = c.std?.[k];
          row[k] = m == null ? "–" : `${num(m)}${sd ? ` <span style="color:var(--ink-3)">±${num(sd, 3)}</span>` : ""}`;
        });
        return row;
      })
    ) : null, "No ablation cells were recorded.");
  }

  // ------------------------------------------------------------ budget

  const POLICY_NAMES = {
    lambdamart: "the vulnprio ordering",
    expected_loss: "by estimated risk",
    cvss_only: "by severity score",
    epss_only: "by exploitation likelihood",
    kev_first: "known-exploited first",
    scanner_severity: "by the scanner's own rating",
    vmc_chain: "a published chain-aware method",
    random: "in random order",
  };
  const policyName = (p) => POLICY_NAMES[p] || String(p || "").replace(/_/g, " ");

  function renderPlan() {
    const sels = D.selections || [];
    const ours = sels.find((s) => s.ranker === "lambdamart") || sels[0];
    const inPlan = findings.filter((f) => f.selected_in_budget)
      .sort((a, b) => (a.rank || 1e6) - (b.rank || 1e6));
    const deferred = findings.filter((f) => !f.selected_in_budget && f.rank)
      .sort((a, b) => riskOf(b) - riskOf(a));
    const hours = inPlan.reduce((t, f) => t + (f.remediation_hours || 0), 0);
    const retired = inPlan.reduce((t, f) => t + riskOf(f), 0);
    const carried = deferred.reduce((t, f) => t + riskOf(f), 0);

    document.getElementById("planTiles").innerHTML = (ours || inPlan.length ? [
      { k: "budget", v: ours ? hrs(ours.budget_hours) : "–", n: "engineer-hours this cycle" },
      { k: "fixes that fit", v: inPlan.length, n: `${hrs(hours)} of work` },
      { k: "risk retired", v: esc(moneyText(retired)), n: "estimated, if all are completed" },
      {
        k: "of the total risk", v: ours ? share(ours.risk_capture_fraction) : "–",
        n: "of the estimated risk this budget clears",
      },
      { k: "still carried", v: esc(moneyText(carried)), n: `across ${deferred.length} deferred findings` },
    ] : []).map((t) => `<div class="tile ${t.cls || ""}"><div class="k">${esc(t.k)}</div>
      <div class="v">${t.v}</div><div class="n">${esc(t.n)}</div></div>`).join("");

    let running = 0;
    mount("planTable", inPlan.length ? table(
      [
        { key: "i", label: "#", num: true, pri: 1 },
        { key: "name", label: "Fix", pri: 1 },
        { key: "where", label: "Location", pri: 2 },
        { key: "sig", label: "", pri: 3 },
        { key: "hours", label: "Effort", num: true, pri: 2 },
        { key: "cum", label: "Cumulative", num: true, pri: 3 },
        { key: "loss", label: "Risk retired (est.)", num: true, pri: 1 },
      ],
      inPlan.map((f, i) => {
        running += f.remediation_hours || 0;
        return {
          i: i + 1,
          name: `<b>${esc(f.name)}</b>${f.remediation_summary
            ? `<div class="note clamp2">${esc(f.remediation_summary)}</div>` : ""}${
            fold(`${f.endpoint_method} ${f.endpoint_path}`, hrs(f.remediation_hours))}`,
          where: `<span class="mono path">${esc(f.endpoint_method)} ${breakable(f.endpoint_path)}</span>` +
            (f.cluster_size > 1 ? `<div class="note">and ${f.cluster_size - 1} further endpoint(s), one fix</div>` : ""),
          sig: signalChips(f, 2),
          hours: hrs(f.remediation_hours),
          cum: hrs(running),
          loss: usd(riskOf(f)),
        };
      }),
      { cls: "roomy" }
    ) : null, "No plan was computed for this run. Run an assessment with a budget.");

    mount("deferredTable", deferred.length ? table(
      [
        { key: "name", label: "Deferred finding", pri: 1 },
        { key: "where", label: "Location", pri: 2 },
        { key: "sig", label: "", pri: 3 },
        { key: "hours", label: "Effort", num: true, pri: 2 },
        { key: "loss", label: "Risk carried (est.)", num: true, pri: 1 },
      ],
      deferred.slice(0, 25).map((f) => ({
        name: `${esc(f.name)}${fold(`${f.endpoint_method} ${f.endpoint_path}`,
          f.remediation_hours ? hrs(f.remediation_hours) : "")}`,
        where: `<span class="mono path">${esc(f.endpoint_method)} ${breakable(f.endpoint_path)}</span>`,
        sig: signalChips(f, 2),
        hours: hrs(f.remediation_hours),
        loss: usd(riskOf(f)),
      }))
    ) : null, "Nothing was deferred: the budget covers everything found.");

    mount("selectionChart", sels.length ? barChart(
      sels.slice().sort((a, b) => b.risk_capture_fraction - a.risk_capture_fraction).map((s) => ({
        label: policyName(s.ranker),
        value: s.risk_capture_fraction, display: share(s.risk_capture_fraction),
        color: s.ranker === "lambdamart" ? "var(--accent)" : "var(--ink-3)",
        dim: s.ranker !== "lambdamart",
      })), { labelW: 190, width: hostWidth("selectionChart", 620) }
    ) : null, "Only one selection rule was computed, so there is nothing to compare against.");
    caption("selectionChart",
      "The same hours spent under different selection rules. The bar is the share of estimated " +
      "risk each rule retires: taller is more risk cleared for the same hours.");

    mount("selectionTable", sels.length ? table(
      [
        { key: "r", label: "Selection rule", pri: 1 }, { key: "b", label: "Budget (h)", num: true, pri: 3 },
        { key: "n", label: "Fixes", num: true, pri: 2 }, { key: "h", label: "Hours used", num: true, pri: 3 },
        { key: "cap", label: "Risk retired (est.)", num: true, pri: 1 },
        { key: "frac", label: "Share", num: true, pri: 1 },
      ],
      sels.map((s) => ({
        r: s.ranker === "lambdamart" ? `<b>${esc(policyName(s.ranker))}</b>` : esc(policyName(s.ranker)),
        b: s.budget_hours, n: s.n_selected, h: s.total_hours.toFixed(1),
        cap: usd(cash(s, "risk_captured")), frac: share(s.risk_capture_fraction),
      }))
    ) : null, "");

    const sims = D.simulations || [];
    const colors = ["var(--accent)", "var(--gC)", "var(--high)", "var(--ink-3)", "var(--gA)", "var(--med)"];
    mount("exposureChart", sims.length ? lineChart(
      sims.map((s, i) => ({
        values: s.weekly_cumulative_exposure, color: colors[i % colors.length],
        emphasis: s.policy === "lambdamart", label: policyName(s.policy),
      })),
      {
        fmtY: (v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0)),
        xLabelStart: "week 1", xLabelEnd: `week ${sims[0].weeks}`,
        width: hostWidth("exposureChart", 900), height: 300,
      }
    ) : null, "No projection was run for this assessment.");
    if (sims.length) {
      document.getElementById("exposureChart").appendChild(
        legend(sims.map((s, i) => ({ color: colors[i % colors.length], label: policyName(s.policy) }))));
      caption("exposureChart",
        "Lower is better: the line counts days on which a fixable problem remained open. " +
        "A line that flattens sooner means the high-risk work was completed sooner.");
    }

    mount("simulationTable", sims.length ? table(
      [
        { key: "p", label: "Order of work", pri: 1 },
        { key: "e", label: "Days of exposure", num: true, pri: 1 },
        { key: "l", label: "Risk carried over time (est.)", num: true, pri: 3 },
        { key: "r", label: "vs a severity list", num: true, pri: 2 },
      ],
      sims.map((s) => ({
        p: s.policy === "lambdamart" ? `<b>${esc(policyName(s.policy))}</b>` : esc(policyName(s.policy)),
        e: s.exposure_days_total.toFixed(0),
        l: usd(cash(s, "expected_loss_days")),
        r: s.reduction_vs_reference == null ? "–" :
          `<span class="delta ${s.reduction_vs_reference > 0 ? "up" : "down"}">${
            s.reduction_vs_reference > 0 ? "−" : "+"}${share(Math.abs(s.reduction_vs_reference))} exposure</span>`,
      }))
    ) : null, "");
  }

  // ------------------------------------------------------------ robustness

  const DEFENCES = [
    "Normalise: Unicode folding, zero-width and bidirectional control removal, homoglyph folding, hidden markup dropped, oversized blobs elided.",
    "Redact instructions: a versioned multilingual pattern library replaces each imperative addressed to the model, and records the attempt.",
    "Envelope with a per-call nonce, so an attempt to close the envelope and speak as the operator is detectable.",
    "Plant a canary in the operator text: if it appears in output, the injection reached the model and that call is discarded.",
    "Force structured output: bounded fields make an out-of-range value unrepresentable rather than merely unlikely.",
    "Verify evidence spans: every claim must quote text that exists in the sanitised input, or it is dropped.",
    "Cap influence and enforce floors: a reference page may move a feature by at most 0.35, target-authored text by 0.15, and neither can argue a KEV listing away.",
  ];

  function renderRobustness() {
    document.getElementById("defenceList").innerHTML = DEFENCES.map((d) => `<li>${esc(d)}</li>`).join("");
    const a = D.adversarial;
    if (!a) {
      document.getElementById("advTiles").innerHTML =
        `<div class="tile"><div class="k">adversarial evaluation</div><div class="v">–</div>
         <div class="n">not run in this configuration</div></div>`;
      mount("advTable", null, "");
    } else {
      document.getElementById("advTiles").innerHTML = [
        { k: "attack success rate", v: pct(a.attack_success_rate), n: `${a.n_cases} cases`, cls: a.attack_success_rate > 0 ? "bad" : "good" },
        { k: "canary leak rate", v: pct(a.canary_leak_rate), n: "injection reached the model", cls: a.canary_leak_rate > 0 ? "bad" : "good" },
        { k: "detection rate", v: share(a.detection_rate), n: "attacks flagged", cls: a.detection_rate >= 0.8 ? "good" : "warn" },
        { k: "false positives", v: share(a.false_positive_rate), n: "on benign controls", cls: a.false_positive_rate <= 0.1 ? "good" : "warn" },
        { k: "mean rank shift", v: num(a.mean_abs_rank_shift, 2), n: `worst ${a.max_abs_rank_shift} positions` },
      ].map((t) => `<div class="tile ${t.cls || ""}"><div class="k">${esc(t.k)}</div><div class="v">${t.v}</div><div class="n">${esc(t.n)}</div></div>`).join("");

      const cats = Object.entries(a.per_category || {});
      mount("advTable", cats.length ? table(
        [
          { key: "c", label: "Category", pri: 1 }, { key: "n", label: "Cases", num: true, pri: 2 },
          { key: "s", label: "Success", num: true, pri: 1 }, { key: "d", label: "Detected", num: true, pri: 1 },
        ],
        cats.map(([k, v]) => ({
          c: esc(k.replace(/_/g, " ")), n: v.n ?? v.cases ?? "–",
          s: v.attack_success_rate == null ? "–" : share(v.attack_success_rate),
          d: v.detection_rate == null ? "–" : share(v.detection_rate),
        }))
      ) : null, "No per-category breakdown was recorded for this corpus run.");
    }

    const inf = findings.filter((f) => f.untrusted_influence_share > 0)
      .sort((a, b) => b.untrusted_influence_share - a.untrusted_influence_share).slice(0, 14);
    mount("influenceChart", inf.length ? barChart(inf.map((f) => ({
      label: shortName(f), value: f.untrusted_influence_share, display: share(f.untrusted_influence_share),
      color: (f.alerts || []).length ? "var(--crit)" : "var(--gA)",
    })), { labelW: 230, width: hostWidth("influenceChart", 700) }) : null, "No model-derived attribution recorded in this run.");
  }

  // ------------------------------------------------------------ gaps

  function renderGaps() {
    const gaps = D.gaps || [];
    mount("gapTable", gaps.length ? table(
      [
        { key: "id", label: "Gap", pri: 1 }, { key: "t", label: "What prior work lacked", pri: 1 },
        { key: "m", label: "What this framework does", pri: 2 },
        { key: "mod", label: "Where", pri: 4 },
        { key: "ev", label: "Evidence in this run", pri: 3 },
      ],
      gaps.map((g) => ({
        id: `<b>${esc(g.gap_id)}</b>`, t: esc(g.title), m: esc(g.mitigation),
        mod: (g.modules || []).map((m) => `<span class="chip ghost mono">${esc(m)}</span>`).join(" "),
        ev: esc(g.evidence || ""),
      }))
    ) : null, "No traceability rows were exported with this payload.");
  }

  // ------------------------------------------------------------ wiring

  const RENDERERS = {
    overview: renderOverview,
    findings: renderQueue,
    chains: renderChains,
    plan: renderPlan,
    evaluation: renderEvaluation,
    ablation: renderAblation,
    robustness: renderRobustness,
    gaps: renderGaps,
  };

  /* The five views behind "Method & evidence". They justify the ordering rather than help
     anyone act on it, so they are off the main road - but they are not deleted, they are
     not second-class, and a link straight to one of them still works. */
  const METHOD_VIEWS = new Set(["evaluation", "ablation", "robustness", "novelty", "gaps"]);

  /* Old hashes from links people already have. Renaming a tab should not break a bookmark. */
  const VIEW_ALIASES = { queue: "findings", budget: "plan" };

  /* Views contributed by analyze.js. It registers itself here rather than app.js knowing
     about it, so this file stays complete on its own under file://. */
  const extraViews = {};
  function registerView(name, render) { extraViews[name] = render; }

  function setMethodVisible(visible, persist = true) {
    document.getElementById("methodGroup").hidden = !visible;
    document.getElementById("methodToggle").setAttribute("aria-pressed", String(visible));
    if (persist) {
      try { localStorage.setItem("vulnprio-method", visible ? "1" : "0"); } catch (_) {}
    }
  }

  /* Which view is on screen. Kept because a change of width has to redraw whatever is
     showing: charts are generated at the size of the box they go into rather than at a
     width chosen when this file was written, so a resized window means a redraw. */
  let currentView = "overview";

  function switchView(name) {
    name = VIEW_ALIASES[name] || name;
    closeDrawer();
    // A deep link into the method section reveals it rather than bouncing to Overview.
    if (METHOD_VIEWS.has(name) && document.getElementById("methodGroup").hidden) {
      setMethodVisible(true);
    }
    const tab = document.querySelector(`[data-view="${name}"]`);
    if (!tab || tab.hidden || tab.disabled) {
      name = hasData() ? "overview" : "analyze";
    }
    document.querySelectorAll("#mainnav button[data-view]").forEach((b) => {
      const selected = b.dataset.view === name;
      b.setAttribute("aria-selected", String(selected));
      /* In a narrow column the navigation rail scrolls sideways, and a selected tab you
         cannot see is no better than no selected state at all. */
      if (selected && b.scrollIntoView) b.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
    document.querySelectorAll("section.view").forEach((s) => (s.hidden = s.id !== `view-${name}`));
    currentView = name;
    const render = RENDERERS[name] || extraViews[name];
    if (render) render();
    markScrollable();
    if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
  }

  /* One resize listener for the whole page, throttled to an animation frame and gated on the
     width actually having changed: a mobile browser fires resize every time the address bar
     slides, and redrawing every chart for that would be absurd. */
  let lastWidth = 0;
  let resizePending = false;
  function onViewportResize() {
    if (resizePending) return;
    resizePending = true;
    requestAnimationFrame(() => {
      resizePending = false;
      const width = document.documentElement.clientWidth;
      if (width === lastWidth) return;
      lastWidth = width;
      const render = RENDERERS[currentView] || extraViews[currentView];
      if (render) render();
      markScrollable();
    });
  }

  // ------------------------------------------------------------ data reload

  const hasData = () => (D.findings || []).length > 0 || (D.scans || []).length > 0;

  /* Every result item stays in the navigation from the start, so the operator can see what
     the product will give them. Until a result exists they are disabled rather than absent:
     a menu that grows items as you use it hides the shape of the product. An exported site
     opened from file:// already carries findings, so everything is live on load, exactly as
     it always has been. Once a run lands the items stay enabled, so a second run does not
     make the bar flicker. */
  function updateTabAvailability() {
    const ready = hasData();
    document.querySelectorAll("#mainnav button[data-view]").forEach((button) => {
      if (button.dataset.view === "analyze" || button.hidden) return;
      button.disabled = !ready;
      button.setAttribute("aria-disabled", String(!ready));
      button.title = ready ? "" : "Available once an assessment has run";
    });
    document.getElementById("methodToggle").disabled = !ready;
    if (!ready) setMethodVisible(false, false);
  }

  function populateQueueControls() {
    const scanSel = document.getElementById("queueScan");
    scanSel.innerHTML = "";
    scanSel.appendChild(el(`<option value="">all scans</option>`));
    (D.scans || []).forEach((s) =>
      scanSel.appendChild(el(`<option value="${esc(s.scan_id)}">${esc(scanName(s.scan_id))}</option>`)));
    scanSel.value = "";

    const polSel = document.getElementById("queuePolicy");
    polSel.innerHTML = "";
    const policies = new Set(["rank"]);
    findings.forEach((f) => Object.keys(f.ranks_by_policy || {}).forEach((p) => policies.add(p)));
    [...policies].forEach((p) =>
      polSel.appendChild(el(`<option value="${esc(p)}">${esc(p === "rank" ? "vulnprio ranking" : p)}</option>`)));
    polSel.value = "rank";
  }

  /** Swap in a fresh payload and re-render everything, without reloading the page. */
  function setData(payload) {
    D = payload || {};
    window.VULNPRIO_DATA = D;
    findings = D.findings || [];
    graphs = D.graphs || [];
    metrics = D.metrics || [];

    Object.assign(queueState, {
      sort: "rank", dir: 1, policy: "rank", scan: "", q: "",
      kev: false, choke: false, alerts: false, budget: false,
    });
    const search = document.getElementById("queueSearch");
    if (search) search.value = "";
    ["filterKev", "filterChoke", "filterAlerts", "filterBudget"].forEach((id) => {
      const b = document.getElementById(id);
      if (b) b.setAttribute("aria-pressed", "false");
    });
    // Selects cache their options against the previous run; clear them so they repopulate.
    ["chainScan", "chainPath", "evalMetric", "ablMetric"].forEach((id) => {
      const node = document.getElementById(id);
      if (node) node.innerHTML = "";
    });
    populateQueueControls();
    updateTabAvailability();
    renderMeta();
    renderOverview();
  }

  // ------------------------------------------------------------ server probe

  /* One probe, at startup. Under file:// the fetch rejects immediately and the page stays
     exactly the static report it has always been. */
  async function probeServer() {
    if (!window.fetch || location.protocol === "file:") return null;
    try {
      const response = await fetch("api/health", { cache: "no-store" });
      if (!response.ok) return null;
      return await response.json();
    } catch (_) {
      return null;
    }
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const node = document.createElement("script");
      node.src = src;
      node.onload = resolve;
      node.onerror = () => reject(new Error(`could not load ${src}`));
      document.head.appendChild(node);
    });
  }

  /* Captured before the first switchView writes one, so "did the operator arrive with a
     deep link?" stays answerable after routing has started. */
  let initialHash = "";

  async function enableInteractive() {
    const health = await probeServer();
    if (!health || !health.ok) return;
    try {
      await loadScript("analyze.js");
    } catch (error) {
      console.warn("vulnprio: interactive features unavailable - ", error.message);
      return;
    }
    if (!window.VULNPRIO_ANALYZE || typeof window.VULNPRIO_ANALYZE.init !== "function") return;

    document.querySelectorAll("[data-needs-server]").forEach((node) => (node.hidden = false));
    document.body.classList.add("interactive");
    updateTabAvailability();
    window.VULNPRIO_ANALYZE.init({
      health,
      session: window.VULNPRIO_SESSION || {},
      api: {
        setData, switchView, registerView, hasData,
        table, mount, legend, barChart, el, esc, usd, pct, num, signalChips,
      },
    });
    // Nothing has been analysed yet, so the Analyze tab is the only useful place to be.
    if (!hasData() && !initialHash) switchView("analyze");
  }

  // ------------------------------------------------------------ wiring

  function init() {
    initialHash = (location.hash || "").slice(1);
    renderMeta();
    renderOverview();

    /* A tab strip is a single tab stop with arrow keys inside it, which is what a screen
       reader and a keyboard user expect of role="tablist". */
    document.querySelectorAll("#mainnav").forEach((strip) => {
      const tabs = () => [...strip.querySelectorAll("button")].filter((b) => !b.hidden && !b.disabled);
      strip.querySelectorAll("button").forEach((b) => {
        b.addEventListener("click", () => switchView(b.dataset.view));
        b.addEventListener("keydown", (event) => {
          const list = tabs();
          const index = list.indexOf(b);
          const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
          let target = null;
          if (step !== undefined && index >= 0) {
            target = list[(index + step + list.length) % list.length];
          } else if (event.key === "Home") {
            target = list[0];
          } else if (event.key === "End") {
            target = list[list.length - 1];
          }
          if (target) {
            event.preventDefault();
            target.focus();
            switchView(target.dataset.view);
          }
        });
      });
    });

    const methodToggle = document.getElementById("methodToggle");
    methodToggle.addEventListener("click", () => {
      const showing = document.getElementById("methodGroup").hidden;
      setMethodVisible(showing);
      if (!showing && METHOD_VIEWS.has(location.hash.slice(1))) switchView("overview");
    });
    let remembered = null;
    try { remembered = localStorage.getItem("vulnprio-method"); } catch (_) {}
    if (remembered === "1" || METHOD_VIEWS.has(VIEW_ALIASES[initialHash] || initialHash)) {
      setMethodVisible(true, false);
    }

    populateQueueControls();
    document.getElementById("queueScan").addEventListener("change", (e) => {
      queueState.scan = e.target.value; renderQueue();
    });
    document.getElementById("queuePolicy").addEventListener("change", (e) => {
      queueState.policy = e.target.value; renderQueue();
    });
    document.getElementById("queueSearch").addEventListener("input", (e) => {
      queueState.q = e.target.value; renderQueue();
    });

    [["filterKev", "kev"], ["filterChoke", "choke"], ["filterAlerts", "alerts"], ["filterBudget", "budget"]]
      .forEach(([id, key]) => {
        const b = document.getElementById(id);
        b.addEventListener("click", () => {
          queueState[key] = !queueState[key];
          b.setAttribute("aria-pressed", String(queueState[key]));
          renderQueue();
        });
      });

    document.getElementById("drawerClose").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (e) => e.key === "Escape" && closeDrawer());

    document.getElementById("themeToggle").addEventListener("click", () => {
      const root = document.documentElement;
      const now = root.getAttribute("data-theme");
      const next = now === "dark" ? "light" : now === "light" ? "" : "dark";
      if (next) root.setAttribute("data-theme", next);
      else root.removeAttribute("data-theme");
      try { localStorage.setItem("vulnprio-theme", next); } catch (_) {}
    });
    try {
      const saved = localStorage.getItem("vulnprio-theme");
      if (saved) document.documentElement.setAttribute("data-theme", saved);
    } catch (_) {}

    updateTabAvailability();
    lastWidth = document.documentElement.clientWidth;
    switchView(initialHash || (hasData() ? "overview" : "analyze"));
    window.addEventListener("resize", onViewportResize, { passive: true });
    enableInteractive();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
