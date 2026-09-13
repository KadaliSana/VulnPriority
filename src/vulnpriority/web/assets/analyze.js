/* vulnpriority - the interactive half of the page.
 *
 * Loaded only when app.js has confirmed a server is answering /api/health, which is why
 * the static export never asks for this file: under file:// the probe fails and this code
 * is never fetched. Everything here talks to vulnpriority.web.app over fetch; nothing here
 * knows how to render a finding, because app.js already does that and the whole point is
 * that a run produced here fills the same views as a run exported from disk.
 *
 * Plain ES2020, no build step, no external script, same as the rest of the page.
 */

(() => {
  "use strict";

  let API = null;            // the helpers app.js lends us
  let SESSION = {};          // { token, token_header }
  let HEALTH = {};           // /api/health
  let OPTIONS = null;        // /api/config
  let jobId = null;          // the job whose result is on screen
  let stream = null;         // live EventSource, if any
  let poller = null;         // setTimeout handle, if we fell back to polling
  let novelty = null;        // cached /api/novelty
  let chosenFile = null;

  const state = { mode: "upload", running: false, logLines: 0, useIntel: true };
  let inspection = null;     // what the server made of the chosen report
  let scannerEnv = null;     // which scanners are installed, for the chosen profile

  const $ = (id) => document.getElementById(id);
  const esc = (s) => (API ? API.esc(s) : String(s == null ? "" : s));

  // ------------------------------------------------------------ transport

  function tokenHeader() {
    const name = SESSION.token_header || "X-VulnPriority-Token";
    return SESSION.token ? { [name]: SESSION.token } : {};
  }

  /** The server restarted under us: say so plainly and offer the one useful action. */
  let staleShown = false;
  function staleSession() {
    if (staleShown) return;
    staleShown = true;
    const bar = document.createElement("div");
    bar.className = "banner stale";
    bar.setAttribute("role", "alert");
    bar.innerHTML =
      '<b>The server restarted.</b> This page was issued by an earlier process, so its ' +
      'session is no longer valid and nothing will run until the page is reloaded. ' +
      '<button type="button" class="primary" id="reloadNow">Reload</button>';
    const host = document.getElementById("view-analyze") || document.body;
    host.insertBefore(bar, host.firstChild);
    const button = document.getElementById("reloadNow");
    if (button) button.addEventListener("click", () => location.reload());
  }

  /** fetch + JSON, turning the server's {error, detail} into a thrown Error. */
  async function call(path, options = {}) {
    const response = await fetch(path, options);
    const text = await response.text();
    let payload = null;
    try { payload = text ? JSON.parse(text) : null; } catch (_) { payload = null; }
    if (!response.ok) {
      const detail = (payload && (payload.detail || payload.error)) || text || response.statusText;
      const error = new Error(detail);
      error.status = response.status;
      error.slug = payload && payload.error;
      // A restarted server issues a new token, so an open page silently holds a dead one and
      // every action fails with a message the reader has to decode. The page knows what to
      // do about that, so it should do it rather than print advice.
      if (response.status === 403 && /token/i.test(String(detail))) staleSession();
      throw error;
    }
    return payload;
  }

  const postJSON = (path, body) =>
    call(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...tokenHeader() },
      body: JSON.stringify(body),
    });

  // ------------------------------------------------------------ the form

  function setMode(mode) {
    state.mode = mode;
    document.querySelectorAll("#analyzeMode button").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    $("pane-upload").hidden = mode !== "upload";
    $("pane-scan").hidden = mode !== "scan";
    refreshRunButton();
  }

  function describeProfile(name) {
    const profile = (OPTIONS.profiles || []).find((p) => p.name === name);
    if (!profile) return "";
    /* One list: what the profile does. The payload also carries `does_not_send` - the
       behaviours this profile is incapable of - and it is deliberately not rendered here.
       Two headed lists side by side made the shorter, negative one read as the important
       half, and a reader choosing a profile needs to know what it does, not to work out
       what it does by subtracting. The guarantees themselves are unchanged and are stated
       where they are enforced: the authorisation block below, and the generated report. */
    const items = profile.sends || [];
    return `
      <p class="profiledesc">${esc(profile.description)}</p>
      ${items.length
        ? `<ul class="plainlist sends">${items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>`
        : ""}
      <p class="note">${profile.authoritative
        ? "Read from the installed scanner, not from documentation written elsewhere."
        : "The scanner package is not installed, so this description is the interface's own "
          + "and cannot be verified against what would actually be sent."}</p>`;
  }

  function renderProfiles() {
    const host = $("profileChoices");
    const profiles = OPTIONS.profiles || [];
    host.innerHTML = profiles
      .map((p, i) => `
        <label class="profileopt">
          <input type="radio" name="scanProfile" value="${esc(p.name)}"${i === 0 ? " checked" : ""}>
          <span class="profilename">${esc(p.label || p.name)}</span>
        </label>`)
      .join("") + `<div class="profilebody" id="profileBody"></div>`;
    const update = () => {
      const picked = host.querySelector("input[name=scanProfile]:checked");
      $("profileBody").innerHTML = describeProfile(picked ? picked.value : "");
      $("authBox").classList.toggle("loud", picked && picked.value === "active");
      renderScanners();
      refreshRunButton();
    };
    host.querySelectorAll("input[name=scanProfile]").forEach((r) =>
      r.addEventListener("change", update));
    update();
  }

  function currentProfile() {
    const picked = document.querySelector("#profileChoices input[name=scanProfile]:checked");
    return picked ? picked.value : "passive";
  }

  // ------------------------------------------------------------ scanner choice

  /* Which scanner actually runs is not an implementation detail: ZAP and the built-in
     crawler find different things, and the answer depends on the profile, because Nuclei
     and Nikto request paths nobody linked to and so cannot serve a passive request. The
     environment is therefore re-read whenever the profile changes, and the consequence is
     shown here at selection time rather than discovered in the results. */
  function currentScanner() {
    const picked = document.querySelector("#scannerChoices input[name=scannerPick]:checked");
    return picked ? picked.value : "";
  }

  async function renderScanners() {
    const host = $("scannerChoices");
    const profile = currentProfile();
    host.innerHTML = `<p class="note">Checking which scanners are installed…</p>`;
    let env;
    try {
      env = await call(`api/scanners?profile=${encodeURIComponent(profile)}`);
    } catch (error) {
      host.innerHTML = `<p class="note">Could not check which scanners are installed:
        ${esc(error.message)}</p>`;
      return;
    }
    if (currentProfile() !== profile) return;        // the operator changed it meanwhile
    scannerEnv = env;

    if (!env.available) {
      host.innerHTML = `<p class="note">${esc(env.notice)}</p>`;
      return;
    }

    const usable = (env.tools || []).filter((t) => t.installed && t.supports_requested_profile);
    const wrongProfile = (env.tools || []).filter((t) => t.installed && !t.supports_requested_profile);
    const missing = (env.tools || []).filter((t) => !t.installed);

    const option = (value, title, detail, checked, extra) => `
      <label class="scanopt${extra ? " muted" : ""}">
        <input type="radio" name="scannerPick" value="${esc(value)}"${checked ? " checked" : ""}>
        <span>
          <span class="scanname">${esc(title)}</span>
          ${detail ? `<span class="scandetail">${esc(detail)}</span>` : ""}
        </span>
      </label>`;

    let html = "";
    if (usable.length) {
      const best = env.preferred || usable[0].name;
      html += option("", `Best available: ${best}`, "vulnpriority selects the strongest installed scanner.", true);
      usable.forEach((tool) => {
        html += option(
          tool.name,
          `${tool.name}${tool.version ? ` ${tool.version}` : ""}`,
          tool.summary, false);
      });
    }
    html += option(
      "builtin", "Built-in scanner",
      usable.length
        ? "vulnpriority's own crawler. Conservative by design; it finds less than a dedicated scanner."
        : "vulnpriority's own crawler. Conservative by design; it finds less than a dedicated scanner.",
      !usable.length);

    if (!usable.length) {
      // The honest version: say nothing better is installed and how to change that, rather
      // than quietly running the weaker scanner. The scan package's notice usually already
      // carries the install hint, so only add one when it does not.
      const hint = missing[0] || {};
      const notice = env.notice || "";
      const needsHint = hint.install_hint && !notice.includes(hint.install_hint);
      html += `<div class="scanwarn">
        <p><b>No external scanner is installed for the ${esc(profile)} profile.</b>
           ${esc(notice)}</p>
        ${needsHint ? `<p class="note"><b>${esc(hint.name)}:</b> ${esc(hint.install_hint)}</p>` : ""}
      </div>`;
    }

    if (wrongProfile.length) {
      // The awkward interaction, stated plainly while it can still be acted on.
      html += `<div class="scanwarn">
        <p><b>${esc(wrongProfile.map((t) => t.name).join(", "))}</b>
           ${wrongProfile.length === 1 ? "is installed but cannot" : "are installed but cannot"}
           serve a <b>${esc(profile)}</b> scan: ${wrongProfile.length === 1 ? "it requests" : "they request"}
           paths that are not linked from the application, which is exactly what the passive
           profile excludes. Switch to <b>Active</b> to use ${wrongProfile.length === 1 ? "it" : "them"}.</p>
      </div>`;
    }

    host.innerHTML = html;
    host.querySelectorAll("input[name=scannerPick]").forEach((r) =>
      r.addEventListener("change", refreshRunButton));
    refreshRunButton();
  }

  function fillPresets() {
    const fill = (select, presets, chosen) => {
      select.innerHTML = presets
        .map((p) => `<option value="${esc(p.name)}">${esc(p.label || p.name)}</option>`)
        .join("");
      if (presets.some((p) => p.name === chosen)) select.value = chosen;
    };
    const defaults = OPTIONS.defaults || {};
    fill($("optAttacker"), OPTIONS.attackers || [], defaults.attacker);
    fill($("optImpact"), OPTIONS.impact_models || [], defaults.impact_model);

    const note = (select, presets, target) => {
      const preset = presets.find((p) => p.name === select.value);
      const detail = Object.entries((preset && preset.detail) || {})
        .map(([k, v]) => `${k.replace(/_/g, " ")} ${typeof v === "number" ? v : esc(v)}`)
        .join(" · ");
      $(target).innerHTML = preset
        ? `${esc(preset.description)}${detail ? `<br><span class="mono">${esc(detail)}</span>` : ""}`
        : "";
    };
    const attackerNote = () => note($("optAttacker"), OPTIONS.attackers || [], "attackerNote");
    const impactNote = () => note($("optImpact"), OPTIONS.impact_models || [], "impactNote");
    $("optAttacker").addEventListener("change", attackerNote);
    $("optImpact").addEventListener("change", impactNote);
    attackerNote();
    impactNote();

    const budget = $("optBudget");
    budget.value = String(Math.min(200, Math.max(1, Math.round(defaults.budget_hours || 40))));
    const showBudget = () => ($("budgetOut").textContent = `${budget.value} h`);
    budget.addEventListener("input", showBudget);
    showBudget();

    const components = defaults.components || {};
    $("compA").checked = components.a !== false;
    $("compB").checked = components.b !== false;
    $("compC").checked = components.c !== false;

    if (!HEALTH.capabilities || !HEALTH.capabilities.scan) {
      const button = document.querySelector('#analyzeMode button[data-mode="scan"]');
      if (button) {
        button.disabled = true;
        button.title = "The vulnpriority.scan package is not installed in this build.";
      }
    }
  }

  // ------------------------------------------------------------ file choice

  const fileSize = (bytes) =>
    bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} kB`;

  /** Name the scanner from the first few kilobytes, so the user sees we understood the file.
      Only a label - the server sniffs the content properly before parsing it. */
  async function detectScanner(file) {
    let head = "";
    try {
      head = await file.slice(0, 4096).text();
    } catch (_) {
      return "";
    }
    const lower = head.toLowerCase();
    if (lower.includes("owasp zap") || lower.includes('"@programname"')) return "OWASP ZAP";
    if (lower.includes("<issues") || lower.includes("burpversion")) return "Burp Suite";
    if (file.name.toLowerCase().endsWith(".jsonl") || lower.includes('"template-id"')) return "Nuclei";
    if (lower.includes('"scan_id"') && lower.includes('"findings"')) return "vulnpriority scan";
    if (lower.trimStart().startsWith("<")) return "an XML report";
    return "";
  }

  async function acceptFile(file) {
    if (!file) return;
    const suffixes = OPTIONS.accepted_report_suffixes || [".json", ".xml", ".jsonl"];
    const lower = file.name.toLowerCase();
    const box = $("fileChosen");
    box.hidden = false;

    if (!suffixes.some((s) => lower.endsWith(s))) {
      clearFileState();
      box.hidden = false;
      box.className = "filechosen bad";
      box.innerHTML = `<div><b>${esc(file.name)}</b> is not a report format this build reads.</div>
        <div class="note">Export the scan as ${suffixes.map((s) => `<code>${esc(s)}</code>`).join(", ")}
        and retry.</div>`;
    } else if (file.size > (OPTIONS.max_upload_bytes || 8388608)) {
      clearFileState();
      box.hidden = false;
      box.className = "filechosen bad";
      box.innerHTML = `<div><b>${esc(file.name)}</b> is ${fileSize(file.size)}, over this
        server's ${fileSize(OPTIONS.max_upload_bytes)} limit.</div>
        <div class="note">Narrow the scan's scope, or split the report.</div>`;
    } else {
      chosenFile = file;
      box.className = "filechosen ok";
      box.innerHTML = `<div class="filerow">
          <span class="filetick" aria-hidden="true">&#10003;</span>
          <span><b>${esc(file.name)}</b><span class="filemeta" id="fileMeta"> · ${esc(fileSize(file.size))}</span></span>
          <button type="button" class="linkish" id="clearFile">remove</button>
        </div>`;
      $("clearFile").addEventListener("click", () => {
        clearFileState();
        $("fileInput").value = "";
        box.hidden = true;
        refreshRunButton();
      });
      detectScanner(file).then((scanner) => {
        const meta = $("fileMeta");
        if (meta && chosenFile === file) {
          meta.textContent = ` · ${fileSize(file.size)}${scanner ? ` · identified as ${scanner}` : ""}`;
        }
      });
      inspectChosenFile(file);
    }
    refreshRunButton();
  }

  function clearFileState() {
    chosenFile = null;
    inspection = null;
    state.useIntel = true;
    $("stalenotice").hidden = true;
    $("stalenotice").innerHTML = "";
  }

  /* Ask the server what the report actually is before anything is run on it. The scan's
     own date decides whether searching the web for exploit intelligence would describe the
     same moment the scan does; if it would not, the operator gets a choice rather than an
     assessment that quietly mixes two different months. */
  async function inspectChosenFile(file) {
    inspection = null;
    $("stalenotice").hidden = true;
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      form.append("use_intel", String(state.useIntel));
      const found = await call("api/inspect", { method: "POST", headers: tokenHeader(), body: form });
      if (chosenFile !== file) return;          // the operator moved on while we asked
      inspection = found;
      const meta = $("fileMeta");
      if (meta && found.scanned_at) {
        meta.textContent += ` · scanned ${found.scanned_at.slice(0, 10)}`
          + (found.n_findings ? ` · ${found.n_findings} findings` : "");
      }
      renderStaleNotice();
    } catch (error) {
      // Inspection is an improvement on the flow, not a gate on it: if it fails the
      // operator can still run the report, and the server will report any real problem.
      console.warn("vulnpriority: could not inspect the report - ", error.message);
    }
  }

  function renderStaleNotice() {
    const box = $("stalenotice");
    const asOf = inspection && inspection.as_of;
    const stale = asOf && ["stale", "refused"].includes(asOf.status);
    if (!stale || !state.useIntel) {
      box.hidden = true;
      box.innerHTML = "";
      return;
    }
    const months = Math.round((asOf.age_days || 0) / 30);
    const age = asOf.age_days > 60 ? `approximately ${months} months old` : `${asOf.age_days} days old`;
    box.hidden = false;
    box.innerHTML = `
      <div class="stalehead">The report and current exploit intelligence describe different moments</div>
      <p>The scan ran on <b>${esc((asOf.as_of || "").slice(0, 10))}</b>, which makes it ${esc(age)}.
         Live exploit intelligence describes <b>today</b>: what is being exploited now, not what
         was being exploited then. Combining the two produces an assessment that appears current
         and is not.</p>
      <div class="staleactions">
        <button type="button" class="toggle" id="staleProceed">Assess without live intelligence</button>
        ${inspection.suggested_target_url
          ? `<button type="button" class="toggle" id="staleRescan">Assess ${esc(inspection.host)} now instead</button>`
          : ""}
      </div>
      <p class="note">Assessing the target now yields a scan and intelligence from the same
        moment. It requires its own authorisation, because it is a fresh test of a live system.</p>`;

    $("staleProceed").addEventListener("click", () => {
      state.useIntel = false;
      renderStaleNotice();
      $("fileChosen").insertAdjacentHTML("beforeend",
        `<div class="note">Live exploit intelligence is disabled for this run, so all evidence
         is dated with the scan itself.</div>`);
      refreshRunButton();
    });
    const rescan = $("staleRescan");
    if (rescan) {
      rescan.addEventListener("click", () => {
        setMode("scan");
        $("targetUrl").value = inspection.suggested_target_url;
        $("targetUrl").dispatchEvent(new Event("input"));
        $("authorized").focus();
      });
    }
  }

  function wireDropzone() {
    const zone = $("dropzone");
    const input = $("fileInput");
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });
    input.addEventListener("change", () => acceptFile(input.files && input.files[0]));
    ["dragenter", "dragover"].forEach((name) =>
      zone.addEventListener(name, (e) => { e.preventDefault(); zone.classList.add("over"); }));
    ["dragleave", "drop"].forEach((name) =>
      zone.addEventListener(name, (e) => { e.preventDefault(); zone.classList.remove("over"); }));
    zone.addEventListener("drop", (e) => {
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length) acceptFile(files[0]);
    });
  }

  // ------------------------------------------------------------ run gating

  /** Hosts the scanner refuses without an explicit second confirmation. */
  const PRIVATE_HOST = /^(localhost|127\.|0\.0\.0\.0|::1|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.)/i;

  function targetHost(raw) {
    const bare = String(raw || "").trim().replace(/^[a-z][a-z0-9+.-]*:\/\//i, "");
    const host = bare.split("/")[0].split("@").pop() || "";
    return host.replace(/:\d+$/, "");
  }

  function isPrivateTarget(raw) {
    const host = targetHost(raw);
    return PRIVATE_HOST.test(host) || /\.(local|internal|localhost)$/i.test(host);
  }

  /** A bare host is a host, not a scheme.
   *
   *  "localhost:3000" is the obvious thing to type for a container and is also a valid URL
   *  whose scheme is "localhost", so prepending nothing sends a malformed request. The
   *  scheme follows the host rather than defaulting: a loopback address is a local service
   *  on plain HTTP, and forcing TLS there fails the handshake instead of scanning. The
   *  server normalises identically through vulnpriority.scan.safety.normalise_target; this
   *  mirror exists so the page can show the result before anything is sent. */
  function targetUrl() {
    const raw = $("targetUrl").value.trim();
    if (!raw) return "";
    if (/^[a-z][a-z0-9+.-]*:\/\//i.test(raw)) return raw;
    return `${isPrivateTarget(raw) ? "http" : "https"}://${raw}`;
  }

  /** Why the run button is disabled, or "" when it is not. The scan rule is the strict one:
      no authorisation tick and no written note means the request is never even formed. */
  function blockingReason() {
    if (state.running) return "An assessment is already running.";
    if (state.mode === "upload" && !chosenFile) return "Choose a scanner report first.";
    if (state.mode !== "scan") return "";
    const url = targetUrl();
    if (!url) return "Enter the target URL.";
    if (!/^https?:\/\/[^/\s]+/i.test(url)) return "The target must be an http or https address.";
    if (!$("authorized").checked) return "Confirm authorisation. Nothing is sent without it.";
    if ($("authNote").value.trim().length < 8) return "Record who authorised the test and what is in scope.";
    if (isPrivateTarget($("targetUrl").value) && !$("allowPrivate").checked) {
      return "Confirm that the internal address is intended.";
    }
    return "";
  }

  /** The one-line version of the Advanced panel, shown while it is collapsed. */
  function refreshAdvancedSummary() {
    const label = (select, presets) => {
      const preset = (presets || []).find((p) => p.name === select.value);
      return (preset && (preset.label || preset.name)) || select.value || "default";
    };
    const bits = [
      `${label($("optAttacker"), OPTIONS.attackers)} attacker`,
      `${label($("optImpact"), OPTIONS.impact_models)} impact`,
      `${$("optBudget").value} hour budget`,
    ];
    const off = [
      !$("compA").checked && "advisories",
      !$("compB").checked && "threat intelligence",
      !$("compC").checked && "attack chains",
    ].filter(Boolean);
    if (off.length) bits.push(`${off.join(" and ")} off`);
    $("advSummary").textContent = bits.join(" · ");
  }

  /** What will actually run, in one clause, for the run hint. */
  function scannerLine() {
    const chosen = currentScanner();
    if (chosen === "builtin") return "the built-in crawler";
    if (chosen) return chosen;
    const preferred = scannerEnv && scannerEnv.preferred;
    return preferred || "the built-in crawler";
  }

  function profileSendsLine() {
    const profile = (OPTIONS.profiles || []).find((p) => p.name === currentProfile());
    if (!profile) return "";
    return profile.name === "active"
      ? "It will crawl the site and send bounded probe values to the parameters it finds."
      : "It will only fetch pages and read the responses; nothing that changes behaviour is sent.";
  }

  function refreshRunButton() {
    const reason = blockingReason();
    const button = $("runButton");
    button.disabled = Boolean(reason);
    button.textContent = state.running ? "Running…" : "Run the assessment";

    const hint = $("runHint");
    hint.classList.toggle("blocked", Boolean(reason) && !state.running);
    hint.textContent = reason || (state.mode === "scan"
      ? `Ready - ${scannerLine()} will run. ${profileSendsLine()}`
      : "Ready. All processing is local to this machine; nothing is uploaded.");

    const privateRow = $("privateCheckRow");
    if (privateRow) {
      const raw = $("targetUrl").value.trim();
      const isPrivate = Boolean(raw) && isPrivateTarget(raw);
      privateRow.hidden = !isPrivate;
      if (isPrivate) $("privateHost").textContent = targetHost(raw);
      else $("allowPrivate").checked = false;
    }

    if ($("authHint")) {
      $("authHint").textContent = $("authorized").checked
        ? `Recorded with the run. ${profileSendsLine()}`
        : "Required. Without it the server refuses the request and sends nothing to the target.";
    }
    refreshAdvancedSummary();
  }

  // ------------------------------------------------------------ phase stepper

  /* The server reports a phase name; a person wants to know which of a handful of things is
     happening. These are those things, in order, with the server phases that map onto each. */
  const STEPS = [
    { key: "ingest", label: "Read the report", phases: ["starting", "input", "ingest", "scan"] },
    { key: "assess", label: "Assess each finding", phases: ["assess"] },
    { key: "enrich", label: "Estimate the risk", phases: ["enrich"] },
    { key: "chains", label: "Trace attack chains", phases: ["chain"] },
    { key: "rank", label: "Rank and select", phases: ["rank", "select"] },
    { key: "report", label: "Generate the report", phases: ["payload", "complete", "demo"] },
  ];

  function stepIndexFor(phase) {
    const head = String(phase || "").split(":")[0];
    const index = STEPS.findIndex((step) => step.phases.includes(head));
    return index < 0 ? -1 : index;
  }

  function renderStepper(phase, finished) {
    const active = finished ? STEPS.length : stepIndexFor(phase);
    $("progressSteps").innerHTML = STEPS.map((step, i) => {
      const cls = i < active ? "done" : i === active ? "active" : "todo";
      const mark = i < active ? "&#10003;" : String(i + 1);
      return `<li class="step ${cls}"><span class="stepmark">${mark}</span>
        <span class="steplabel">${esc(step.label)}</span></li>`;
    }).join("");
  }

  // ------------------------------------------------------------ starting

  function requestBody() {
    return {
      mode: state.mode,
      report: null,
      target_url: state.mode === "scan" ? targetUrl() : "",
      authorized: state.mode === "scan" ? $("authorized").checked : false,
      authorization_note: state.mode === "scan" ? $("authNote").value.trim() : "",
      allow_private_target: state.mode === "scan" && $("allowPrivate").checked,
      profile: currentProfile(),
      attacker: $("optAttacker").value,
      impact_model: $("optImpact").value,
      budget_hours: Number($("optBudget").value) || 40,
      components: { a: $("compA").checked, b: $("compB").checked, c: $("compC").checked },
      use_intel: state.useIntel,
      scanner: currentScanner(),
    };
  }

  async function run() {
    if (blockingReason()) return;
    stopWatching();
    state.running = true;
    state.logLines = 0;
    jobId = null;
    $("progressPanel").hidden = false;
    $("progressLog").textContent = "";
    $("progressError").innerHTML = "";
    $("cancelButton").hidden = false;
    setProgress(0, "starting", "Submitting the request.");
    refreshRunButton();

    try {
      let created;
      if (state.mode === "upload") {
        // A real multipart upload: the file streams to the server rather than being
        // base64-inflated through a JSON body.
        const form = new FormData();
        form.append("file", chosenFile, chosenFile.name);
        form.append("attacker", $("optAttacker").value);
        form.append("impact_model", $("optImpact").value);
        form.append("budget_hours", String(Number($("optBudget").value) || 40));
        form.append("component_a", String($("compA").checked));
        form.append("component_b", String($("compB").checked));
        form.append("component_c", String($("compC").checked));
        form.append("use_intel", String(state.useIntel));
        created = await call("api/analyze/upload", {
          method: "POST", headers: tokenHeader(), body: form,
        });
      } else {
        created = await postJSON("api/analyze", requestBody());
      }
      jobId = created.job_id;
      appendLog(`job ${jobId} accepted`);
      watch(jobId);
    } catch (error) {
      state.running = false;
      $("cancelButton").hidden = true;
      showError(error);
      refreshRunButton();
    }
  }

  async function cancel() {
    if (!jobId) return;
    $("cancelButton").disabled = true;
    try {
      await call(`api/jobs/${encodeURIComponent(jobId)}/cancel`, {
        method: "POST", headers: tokenHeader(),
      });
      appendLog("cancellation requested");
    } catch (error) {
      showError(error);
    } finally {
      $("cancelButton").disabled = false;
    }
  }

  // ------------------------------------------------------------ watching

  /* Server-Sent Events when the browser and the connection allow it, polling when they do
     not. The two carry the same job snapshot, so the rest of this file cannot tell which
     one delivered it. */
  function watch(id) {
    if (!window.EventSource) return poll(id);
    let sawAnything = false;
    try {
      stream = new EventSource(`api/jobs/${encodeURIComponent(id)}/events?after=0`);
    } catch (_) {
      return poll(id);
    }
    const fallback = setTimeout(() => {
      if (!sawAnything) { closeStream(); poll(id); }
    }, 4000);

    const onState = (event) => {
      sawAnything = true;
      clearTimeout(fallback);
      let payload = null;
      try { payload = JSON.parse(event.data); } catch (_) { return; }
      apply(payload, true);
    };
    stream.addEventListener("state", onState);
    stream.addEventListener("done", (event) => {
      onState(event);
      closeStream();
      finish(id);
    });
    stream.addEventListener("error", (event) => {
      // Either the server said the job is gone, or the connection dropped. Either way,
      // polling settles it: EventSource would otherwise reconnect from the beginning.
      clearTimeout(fallback);
      closeStream();
      if (state.running) poll(id);
    });
  }

  function poll(id) {
    const tick = async () => {
      try {
        const snapshot = await call(`api/jobs/${encodeURIComponent(id)}`);
        apply(snapshot, false);
        if (["done", "failed", "cancelled"].includes(snapshot.status)) return finish(id);
      } catch (error) {
        state.running = false;
        $("cancelButton").hidden = true;
        showError(error);
        refreshRunButton();
        return;
      }
      poller = setTimeout(tick, 600);
    };
    tick();
  }

  function closeStream() {
    if (stream) { try { stream.close(); } catch (_) {} stream = null; }
  }

  function stopWatching() {
    closeStream();
    if (poller) { clearTimeout(poller); poller = null; }
  }

  /** Apply a job snapshot. Stream payloads carry only new log lines; polled ones carry all. */
  function apply(snapshot, incremental) {
    setProgress(snapshot.progress, snapshot.phase, snapshot.message);
    const log = snapshot.log || [];
    if (incremental) {
      log.forEach(appendLog);
      state.logLines = snapshot.log_offset != null ? snapshot.log_offset : state.logLines + log.length;
    } else if (log.length > state.logLines) {
      log.slice(state.logLines).forEach(appendLog);
      state.logLines = log.length;
    }
  }

  async function finish(id) {
    stopWatching();
    let snapshot;
    try {
      snapshot = await call(`api/jobs/${encodeURIComponent(id)}`);
    } catch (error) {
      state.running = false;
      $("cancelButton").hidden = true;
      showError(error);
      refreshRunButton();
      return;
    }
    state.running = false;
    $("cancelButton").hidden = true;
    refreshRunButton();

    if (snapshot.status !== "done") {
      setProgress(snapshot.progress, snapshot.phase, snapshot.message);
      showError(new Error(snapshot.error || `The run ended as ${snapshot.status}.`),
        snapshot.status === "cancelled" ? "cancelled" : "failed");
      return;
    }

    try {
      const payload = await call(`api/jobs/${encodeURIComponent(id)}/result`);
      API.setData(payload);
      setProgress(1, "complete", `Ranked ${(payload.findings || []).length} findings.`, true);
      markReportAvailable();
      API.switchView("overview");
    } catch (error) {
      showError(error);
    }
  }

  // ------------------------------------------------------------ progress UI

  function setProgress(fraction, phase, message, finished) {
    const pct = Math.round(Math.min(1, Math.max(0, Number(fraction) || 0)) * 100);
    $("progressFill").style.width = `${pct}%`;
    $("progressPct").textContent = `${pct}%`;
    $("progressPhase").textContent = message || "";
    renderStepper(phase, finished);
  }

  function appendLog(line) {
    if (!line) return;
    const pre = $("progressLog");
    pre.textContent += (pre.textContent ? "\n" : "") + line;
    pre.scrollTop = pre.scrollHeight;
  }

  function showError(error, kind) {
    const what = kind === "cancelled" ? "Cancelled." : "The assessment did not complete.";
    $("progressError").innerHTML =
      `<div class="banner ${kind === "cancelled" ? "" : "bad"}"><b>${esc(what)}</b>
       ${esc(error.message || String(error))}</div>`;
    appendLog(error.message || String(error));
  }

  // ------------------------------------------------------------ report view

  function markReportAvailable() {
    const tab = document.querySelector('#mainnav button[data-view="report"]');
    if (!tab) return;
    // It was hidden while there was no run, so the availability pass in app.js skipped it.
    tab.hidden = false;
    tab.disabled = false;
    tab.removeAttribute("aria-disabled");
    tab.title = "";
  }

  function renderReport() {
    const note = $("reportNote");
    const frame = $("reportFrame");
    const empty = $("reportEmpty");
    const controls = $("reportControls");

    const loading = $("reportLoading");

    if (!jobId) {
      controls.querySelectorAll("button").forEach((b) => (b.disabled = true));
      frame.hidden = true;
      loading.hidden = true;
      frame.removeAttribute("src");
      empty.hidden = false;
      note.textContent = "No run in this session yet.";
      return;
    }
    controls.querySelectorAll("button").forEach((b) => (b.disabled = false));
    empty.hidden = true;
    const intel = ((window.VULNPRIORITY_DATA || {}).notes || {}).intel || {};
    const provenance = intel.ran && intel.gathered_on
      ? ` Exploit intelligence was gathered on ${intel.gathered_on}.`
      : intel.disabled_reason
        ? ` ${intel.disabled_reason}`
        : "";
    note.textContent = (HEALTH.capabilities && HEALTH.capabilities.report
      ? "Generated from this run's own figures."
      : "The report generator is not installed, so this is the framework's own minimal summary.")
      + provenance;

    // Pointed at the endpoint rather than inlined as srcdoc: the document is large, and a
    // real navigation streams it. Until it paints, the panel shows the shape of what is
    // coming rather than a spinner.
    //
    // ``embed`` drops the document's own page background and reading-width limit, and
    // ``theme`` hands it whichever palette this page is currently in. Without the second
    // one a reader who has chosen light on a dark system gets a white sheet inside a dark
    // page, which is the single thing that made this view look broken.
    const wanted = `api/jobs/${encodeURIComponent(jobId)}/report.html`
      + `?embed=1&theme=${encodeURIComponent(currentTheme())}`;
    if (frame.getAttribute("src") !== wanted) {
      loading.hidden = false;
      frame.hidden = true;
      frame.onload = () => {
        loading.hidden = true;
        frame.hidden = false;
        fitReportFrame();
      };
      frame.onerror = () => {
        loading.hidden = true;
        empty.hidden = false;
        empty.textContent = "The report could not be generated for this run. "
          + "The server log on this machine says why.";
      };
      frame.setAttribute("src", wanted);
    } else {
      loading.hidden = true;
      frame.hidden = false;
      fitReportFrame();
    }
  }

  /** Whichever palette the page is in right now, resolved to a word the server accepts. */
  function currentTheme() {
    const chosen = document.documentElement.getAttribute("data-theme");
    if (chosen === "dark" || chosen === "light") return chosen;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark" : "light";
  }

  /* Size the frame to its document so the page scrolls once instead of twice.
     `contentDocument` is readable because the frame is sandboxed with `allow-same-origin`
     and *without* `allow-scripts`: the report cannot run code, so reading its height costs
     nothing. If a browser ever refuses the read, the CSS minimum keeps a usable box. */
  function fitReportFrame() {
    const frame = $("reportFrame");
    if (!frame || frame.hidden) return;
    let doc = null;
    try { doc = frame.contentDocument; } catch (_) { doc = null; }
    if (!doc || !doc.documentElement) return;
    frame.style.blockSize = "0px";                 // let the content decide, not the last value
    const height = Math.max(
      doc.documentElement.scrollHeight, doc.body ? doc.body.scrollHeight : 0
    );
    frame.style.blockSize = height ? `${height}px` : "";
    frame.classList.toggle("fitted", Boolean(height));
  }

  /* One rAF-throttled listener. A narrower frame reflows the report's tables and changes
     its height, so the two have to be re-measured together. */
  let fitPending = false;
  function watchReportFrame() {
    const remeasure = () => {
      if (fitPending) return;
      fitPending = true;
      requestAnimationFrame(() => { fitPending = false; fitReportFrame(); });
    };
    window.addEventListener("resize", remeasure);
    // The theme is a property of the host page, and the report is rendered server-side, so
    // a change means re-fetching it rather than restyling it.
    const root = document.documentElement;
    new MutationObserver(() => renderReport())
      .observe(root, { attributes: true, attributeFilter: ["data-theme"] });
    if (window.matchMedia) {
      const media = window.matchMedia("(prefers-color-scheme: dark)");
      const onSystemChange = () => {
        if (!root.getAttribute("data-theme")) renderReport();
      };
      if (media.addEventListener) media.addEventListener("change", onSystemChange);
      else if (media.addListener) media.addListener(onSystemChange);
    }
  }

  function wireReportDownloads() {
    $("reportControls").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-fmt]");
      if (!button || !jobId) return;
      const anchor = document.createElement("a");
      anchor.href = `api/jobs/${encodeURIComponent(jobId)}/report.${button.dataset.fmt}?download=1`;
      anchor.download = `vulnpriority-report.${button.dataset.fmt}`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
    });
  }

  // ------------------------------------------------------------ novelty view

  const NATURE = { scientific: "research claim", engineering: "engineering control" };

  async function renderNovelty() {
    if (novelty === null) {
      $("noveltyBanner").innerHTML = `<div class="banner">Loading the prior-work analysis…</div>`;
      try {
        novelty = await call("api/novelty");
      } catch (error) {
        novelty = { available: false, detail: error.message };
      }
    }
    const data = novelty || {};
    if (data.available === false) {
      $("noveltyBanner").innerHTML =
        `<div class="banner"><b>Not available in this build.</b> ${esc(data.detail ||
          "The vulnpriority.novelty package is not installed, so there is nothing to show. " +
          "This section is deliberately empty rather than filled with a plausible guess.")}</div>`;
      $("noveltyTiles").innerHTML = "";
      ["noveltyNarrative", "noveltyMatrix", "noveltyNeighbours", "noveltyCaveats"]
        .forEach((id) => ($(id).innerHTML = ""));
      return;
    }
    $("noveltyBanner").innerHTML = "";

    const verdict = data.verdict || {};
    const tiles = [
      { k: "studies reviewed", v: data.corpus_size ?? verdict.corpus_size ?? "–", n: "the review's own corpus" },
      { k: "capabilities compared", v: verdict.dimensions ?? (data.capabilities || []).length, n: "each with a stated test" },
      { k: "fully satisfied here", v: verdict.framework_has ?? "–", n: `${verdict.framework_partial ?? 0} partially`, cls: "good" },
      { k: "unprecedented in the corpus", v: (data.unique_capabilities || []).length, n: "in no reviewed study" },
      { k: "rare in the corpus", v: (data.rare_capabilities || []).length, n: "in at most one study" },
      { k: "already established", v: (data.shared_capabilities || []).length, n: "in two or more studies" },
    ];
    $("noveltyTiles").innerHTML = tiles.map((t) =>
      `<div class="tile ${t.cls || ""}"><div class="k">${esc(t.k)}</div>
       <div class="v">${esc(t.v)}</div><div class="n">${esc(t.n)}</div></div>`).join("");

    const narrative = Array.isArray(data.narrative) ? data.narrative : [data.narrative].filter(Boolean);
    $("noveltyNarrative").innerHTML =
      narrative.map((p) => `<p class="leadtext">${esc(p)}</p>`).join("") +
      (verdict.strongest_prior_work
        ? `<p class="note">Strongest prior work in the corpus: <b>${esc(verdict.strongest_prior_work)}</b>.</p>`
        : "");

    const classOf = {};
    (data.unique_capabilities || []).forEach((c) => (classOf[c.dimension] = "novel"));
    (data.rare_capabilities || []).forEach((c) => (classOf[c.dimension] ||= "rare"));
    (data.shared_capabilities || []).forEach((c) => (classOf[c.dimension] ||= "shared"));

    const titles = {};
    (data.capabilities || []).forEach((c) => (titles[c.key] = c));

    const rows = Object.values(data.matrix || {});
    const draw = () => {
      const want = $("noveltyFilter").value;
      const shown = rows.filter((r) => want === "all" || classOf[r.dimension] === want);
      $("noveltyMatrix").innerHTML = shown.length
        ? `<table class="matrix"><thead><tr>
             <th>Capability</th><th data-pri="4">Kind</th><th data-pri="2">This framework</th>
             <th data-pri="3">The 45 reviewed studies</th><th class="num" data-pri="3">Present</th><th>Verdict</th>
           </tr></thead><tbody>${shown.map((row) => {
          const total = row.total || 1;
          const seg = (n, cls) => (n ? `<i class="${cls}" style="width:${(n / total) * 100}%"></i>` : "");
          const meta = titles[row.dimension] || {};
          const verdictClass = classOf[row.dimension] || "shared";
          const label = { novel: "unprecedented", rare: "rare", shared: "established" }[verdictClass];
          return `<tr>
            <td><b>${esc(row.title || row.dimension)}</b>
                ${meta.definition ? `<div class="note">${esc(meta.definition)}</div>` : ""}</td>
            <td data-pri="4">${esc(NATURE[row.nature] || row.nature || "")}</td>
            <td data-pri="2"><span class="chip ${row.framework_position === "has" ? "ok" : "ghost"}">${esc(row.framework_position || "")}</span></td>
            <td data-pri="3"><div class="stack">${seg(row.has_count, "has")}${seg(row.partial_count, "part")}${seg(row.lacks_count + (row.unknown_count || 0), "lacks")}</div></td>
            <td class="num" data-pri="3">${row.has_count} / ${total}</td>
            <td><span class="chip ${verdictClass === "novel" ? "choke" : verdictClass === "rare" ? "med" : "ghost"}">${esc(label)}</span></td>
          </tr>`;
        }).join("")}</tbody></table>`
        : `<p class="note">No capability falls in that category.</p>`;
    };
    $("noveltyFilter").onchange = draw;
    draw();

    const neighbours = data.nearest_neighbours || [];
    $("noveltyNeighbours").innerHTML = neighbours.length
      ? `<table><thead><tr><th>Study</th><th data-pri="3">Paradigm</th><th class="num">Shared</th>
         <th data-pri="2">What it lacks</th></tr></thead><tbody>${neighbours.map((n) => `
         <tr><td>${esc(n.label || n.key)}</td><td data-pri="3">${esc(n.paradigm || "")}</td>
             <td class="num">${n.shared_count ?? (n.shared || []).length}</td>
             <td data-pri="2">${(n.framework_only || []).slice(0, 4)
                  .map((d) => `<span class="chip ghost">${esc((titles[d] || {}).title || d)}</span>`).join(" ")}
                 ${(n.framework_only || []).length > 4 ? `<span class="note">+${n.framework_only.length - 4} more</span>` : ""}</td>
         </tr>`).join("")}</tbody></table>`
      : `<p class="note">No neighbour comparison was included in this payload.</p>`;

    $("noveltyCaveats").innerHTML = (data.caveats || [])
      .map((c) => `<li>${esc(typeof c === "string" ? c : c.text || "")}</li>`).join("");
  }

  // ------------------------------------------------------------ init

  function init({ health, session, api }) {
    API = api;
    HEALTH = health || {};
    SESSION = session || {};

    API.registerView("report", renderReport);
    API.registerView("novelty", renderNovelty);

    document.querySelectorAll("#analyzeMode button").forEach((b) =>
      b.addEventListener("click", () => !b.disabled && setMode(b.dataset.mode)));
    wireDropzone();
    wireReportDownloads();
    watchReportFrame();
    ["targetUrl", "authNote"].forEach((id) => $(id).addEventListener("input", refreshRunButton));
    $("authorized").addEventListener("change", refreshRunButton);
    $("allowPrivate").addEventListener("change", refreshRunButton);
    ["optAttacker", "optImpact", "optBudget", "compA", "compB", "compC"].forEach((id) =>
      $(id).addEventListener("change", refreshRunButton));
    $("optBudget").addEventListener("input", refreshAdvancedSummary);
    $("runButton").addEventListener("click", run);
    $("cancelButton").addEventListener("click", cancel);

    // The Report tab has nothing to show until a run exists in this session.
    const reportTab = document.querySelector('#mainnav button[data-view="report"]');
    if (reportTab) reportTab.hidden = true;

    call("api/config")
      .then((options) => {
        OPTIONS = options;
        fillPresets();
        renderProfiles();
        setMode("upload");
      })
      .catch((error) => {
        $("runHint").textContent = `Could not read the server's configuration: ${error.message}`;
        $("runButton").disabled = true;
      });
  }

  window.VULNPRIORITY_ANALYZE = { init };
})();
