# vulnprio - framework design

This is the implementation specification for the framework proposed in the literature
review *AI-Driven Automated Web Application Vulnerability Prioritization*. It is binding:
the shared contracts in `src/vulnprio/core/` are frozen, and every module below codes
against them.

The framework takes web application scanner output, enriches each finding with
vulnerability intelligence and agentic semantic assessment, scores each finding by its
contribution to reachable compromise, learns a ranking, and evaluates that ranking under a
protocol designed so the result survives peer review.

---

## 1. Architecture

```
scanner output ─► ingest ─► feeds/intel ─┐
                                          ├─► Component A (semantic assessment, sandboxed LLM)
                                          ├─► Component B (attacker model, monetary impact, expected loss)
                                          ├─► Component C (attack graph, chain contribution)
                                          └─► features ─► LambdaMART ranker ─► explanation ─► ranked queue
                                                              │
                                                              ├─► evaluation (splits, labels, metrics, ablation)
                                                              ├─► selection (knapsack under remediation budget)
                                                              ├─► simulation (longitudinal exposure)
                                                              └─► adversarial (injection + rank manipulation)
```

Three rules hold everywhere:

1. **Untrusted by default.** Any text from the target application or the internet is an
   `UntrustedText` carrying a `TrustTier`. It reaches a model only through the sandbox, and
   it can move a feature only as far as its influence budget allows.
2. **As-of dated.** Every feed access takes an `as_of` date and must not return anything
   later. This is what makes time-ordered evaluation honest.
3. **Live first, offline as the fallback, and never silently.** The default configuration
   resolves at run time: a real model backend and live feeds when credentials and
   connectivity allow, the deterministic heuristic and dated fixtures when they do not. The
   resolved choice and the reason for it are recorded in the run manifest, printed in the
   run summary, and stated in the report, so nobody has to guess whether a number came from
   the live internet or a fixture. `configs/offline.yaml` pins the offline path for
   reproduction, and the research protocol refuses live retrieval regardless of what keys
   happen to be present, because a reproducible evaluation cannot quietly acquire today's
   internet.

---

## 2. Priority, defined (Gap 1)

Priority is expected loss over the attacker's horizon:

```
expected_loss = P(exploit | evidence, attacker) × impact
```

* `P(exploit | evidence, attacker)` comes from `AttackerModel`, a logistic model over named
  evidence terms. Every term is recorded in `ExploitLikelihood.log_odds_terms`.
* `impact` comes from `ImpactModel`: records exposed × cost per record, downtime hours ×
  hourly cost, integrity loss, a regulatory multiplier and a reputational fraction.
* Chain-adjusted loss adds Component C:
  `chain_adjusted_loss = expected_loss + chain_weight × reach_delta`.

The learned ranker does not replace this definition: it learns the ordering over the same
evidence, and `expected_loss` is retained as a first-class baseline so that the learned
ordering can always be compared against the decision-theoretic one.

---

## 3. Modules

### 3.1 `ingest/` - scanner output to `Scan`

| File | Contents |
|---|---|
| `zap.py` | `ZapParser` for OWASP ZAP JSON and XML traditional reports |
| `burp.py` | `BurpParser` for Burp Suite XML issue exports |
| `nuclei.py` | `NucleiParser` for Nuclei JSONL |
| `generic.py` | `GenericJsonParser` for the framework's canonical `Scan` JSON; `detect_parser(path)` |
| `normalize.py` | URL canonicalisation, path templating, identifier construction, CWE/CVE extraction |
| `tech_fingerprint.py` | `TechComponent` inference from headers, cookies, body signatures, URL paths |
| `correlate.py` | `FindingCorrelator`: groups the same root cause across endpoints |

Details that matter:

* **Path templating.** `/users/123/orders/9` becomes `/users/{id}/orders/{id}`. Numeric
  segments, UUIDs, hashes of 32+ hex characters and base64-looking segments are replaced.
  This is what stops one vulnerability appearing as forty findings.
* **Identifiers.** `endpoint_id = stable_id("ep", app_id, host, method, templated_path)`,
  `finding_id = stable_id("f", scan_id, endpoint_id, plugin_id or name, param or "")`,
  `scan_id = stable_id("scan", app_id, scanned_at.isoformat(), scanner_name)`.
* **Auth inference.** `auth_required` is `NONE` when the endpoint answered 200 without
  credentials, `USER` when an unauthenticated probe returned 401/403 or the path matches an
  authenticated area, `ADMIN` when the path matches admin lexicon entries.
* **Correlation.** `dedup_key = stable_id("dk", app_id, str(cwe_id), sorted_cve_ids, plugin_id or name)`.
  `cluster_size` is the number of findings sharing that key within the scan. The framework
  ranks root causes, not alerts, so downstream code uses `cluster_size` as a feature and the
  selection layer charges remediation cost once per cluster.
* All text taken from the scan (descriptions, evidence, response bodies) is wrapped as
  `UntrustedText` with provenance `SCANNER_OUTPUT` for scanner text and `TARGET_RESPONSE`
  for anything echoed from the application.

### 3.2 `feeds/` - external intelligence, as-of dated

| File | Contents |
|---|---|
| `cache.py` | `FileCacheStore`: JSON on disk keyed by `(feed, key, as_of)`, with TTL and content hash |
| `base.py` | `FixtureFeed` base (loads from `data/fixtures/feeds`), `CachingFeed` wrapper, `as_of` filters |
| `nvd.py` | `NvdFixtureFeed`, `NvdLiveFeed` (NVD 2.0 API) → `VulnIntel` skeleton |
| `epss.py` | `EpssFixtureFeed`, `EpssLiveFeed` (FIRST EPSS, per-date snapshots) |
| `kev.py` | `KevFixtureFeed`, `KevLiveFeed` (CISA KEV; `date_added` respected for as-of) |
| `exploitdb.py` | `ExploitFixtureFeed`, `ExploitLiveFeed` → `tuple[ExploitEvidence, ...]` |
| `references.py` | `ReferenceFixtureFetcher`, `ReferenceLiveFetcher` (host allowlist, byte cap, HTML to text) |
| `bundle.py` | `build_feed_bundle(config)`, `DefaultIntelAssembler` |

* Offline feeds must raise `OfflineViolationError` if any network call is attempted, and
  the test suite asserts this by installing an `httpx` transport that raises.
* Live feeds are wrapped by `CachingFeed` in `live_with_cache` mode.
* `DefaultIntelAssembler.assemble(cve_id, as_of)` merges NVD, EPSS, KEV, exploit evidence
  and up to `max_references_per_cve` reference documents into one `VulnIntel`, and relies on
  `VulnIntel`'s own validator to reject temporal leakage.
* **CVSS selection policy** lives here: `select_cvss(records)` prefers the newest version
  present, prefers `NVD` over `CNA` on ties, and returns a `source_agreement` in `[0, 1]`
  equal to 1 minus the normalised spread of base scores across sources. Both the chosen
  record and the agreement become features (Gap 3: the instrument's own inconsistency is
  measured rather than ignored).

### 3.3 `sandbox/` - untrusted content handling (Architecture item 4)

| File | Contents |
|---|---|
| `normalize.py` | Unicode NFKC, zero-width and control-character strip, homoglyph fold, HTML to text, hidden-element removal, base64 blob elision, length cap |
| `instruction_filter.py` | Pattern-based instruction detection and redaction; `configs/sandbox/instruction_patterns.yaml` |
| `delimit.py` | Nonce-delimited envelopes `<untrusted id=... nonce=...> ... </untrusted nonce=...>`; envelope integrity check |
| `canary.py` | Canary generation and leak detection |
| `output_guard.py` | Schema validation, numeric bounds, evidence-span substring verification, cross-finding reference rejection |
| `pipeline.py` | `Sandbox` implementing `Sanitizer`; `build_sandboxed_prompt(...) -> SandboxedPrompt` |

Defences, in order of application:

1. Normalise (defeats homoglyph, zero-width and encoding tricks).
2. Strip instruction patterns, replacing each with `[REDACTED-INSTRUCTION]` and recording an
   `InjectionSignal`. Patterns cover English and non-English imperatives, role hijacks,
   system-prompt impersonation, delimiter escapes and schema smuggling.
3. Envelope each segment with a per-call random nonce; a model reply that reproduces the
   closing tag with the wrong nonce sets `envelope_broken`.
4. Plant a canary in the system text; a canary appearing in output raises `CanaryLeakError`.
5. Force structured output; bounded pydantic fields make out-of-range values unrepresentable.
6. Require every claim to quote an `evidence_span` that is a literal substring of the
   sanitized input; otherwise the field is discarded and the heuristic value used.
7. Cap influence: an untrusted tier may move any normalised feature by at most
   `influence_budget[tier]`, and may never push `p_exploit` below the floor implied by
   tier ≤ 1 evidence (KEV membership cannot be argued away by a blog post).

### 3.4 `llm/` - backends

| File | Contents |
|---|---|
| `schemas.py` | Bounded output schemas: `AssetCriticalityOut`, `ExploitabilityOut`, `ApplicabilityOut` |
| `prompts.py` | Frozen system prompts per task, hash-pinned |
| `heuristic.py` | `HeuristicBackend`: deterministic lexicon and rule scoring, the test default |
| `anthropic_backend.py` | `AnthropicBackend`: Anthropic SDK, structured output, retries |
| `guarded.py` | `GuardedBackend`: wraps any backend with sandbox, canary, output guard, fallback |
| `cache.py` | Response cache keyed by `prompt_hash` so reruns are free and deterministic |
| `consistency.py` | `ConsistencyGuard`: shrink model output toward the heuristic when they diverge |
| `factory.py` | `build_llm_backend(config)` |

`HeuristicBackend` is not a stub: it is a real, deterministic assessor over the sanitized
text and structured evidence, and it is what makes the entire pipeline runnable, testable
and reproducible offline. `AnthropicBackend` uses model id `claude-sonnet-5` by default,
configurable, and is exercised in tests only through a stubbed client.

### 3.5 `semantic/` - Component A (Goals 1–3, Architecture items 1–2)

| File | Contents |
|---|---|
| `lexicon.py` | Multilingual URL and body token lexicon per `EndpointFunction` |
| `criticality.py` | `assess_asset_criticality(endpoint, scan, backend)` → `AssetCriticality` |
| `exploitability.py` | `assess_exploitability(finding, intel, endpoint, backend)` → `ExploitabilityAssessment` |
| `applicability.py` | `assess_applicability(finding, intel, tech, backend)` → `ApplicabilityAssessment` |
| `cpe_match.py` | Deterministic version-range matching against observed components |
| `assessor.py` | `AgenticSemanticAssessor` implementing `SemanticAssessor` |

* **Criticality (Goal 1)** is inferred from structure only: path tokens, HTTP method,
  `auth_required`, response content type, `sets_cookie`, response size, parameter names, and
  personally-identifiable-information markers in the response sample. No manual asset tags.
  Structural features are computed first; the model may adjust `criticality` within its
  influence budget, and `evidence_features` records the structural inputs.
* **Applicability (Goal 3)** is decided by `cpe_match` first. A `MISMATCH` from tier ≤ 1
  version evidence is authoritative and the model cannot overturn it; the model only rules
  on preconditions that version data cannot settle.

### 3.6 `attacker/`, `decision/`, `enrich/` - Component B

| File | Contents |
|---|---|
| `attacker/model.py` | `p_exploit(attacker, evidence) -> ExploitLikelihood` |
| `attacker/presets.py` | `load_preset(name)` over `configs/attacker_models/` |
| `attacker/likelihood.py` | Evidence assembly: EPSS logit, KEV, maturity, feasibility, applicability, exposure, criticality, complexity, interaction, privileges |
| `decision/impact.py` | `estimate_impact(...) -> BusinessImpact` |
| `decision/remediation_cost.py` | `estimate_cost(...) -> RemediationCost`, unequal by CWE and cluster size |
| `decision/expected_loss.py` | `expected_loss(likelihood, impact)`, `chain_adjusted_loss(...)` |
| `enrich/trust.py` | Influence budgets, floors, corroboration, `TrustSummary` |
| `enrich/enricher.py` | `ContextualEnricher` implementing `Enricher` |

The likelihood model, explicitly:

```
z = w_intercept
  + w_epss_logit          × logit(clip(epss, 1e-6, 1-1e-6)) / 10
  + w_kev                 × 1[in KEV as of date]
  + w_kev_ransomware      × 1[known ransomware use]
  + w_exploit_maturity[m]
  + w_feasibility         × exploit_feasibility
  + w_applicability       × (p_applicable - 0.5) × 2
  + w_exposure            × exposure
  + w_asset_criticality   × criticality × target_preference.get(function, 1.0)
  + w_complexity_high     × 1[attack_complexity == HIGH]
  + w_user_interaction    × 1[user_interaction == REQUIRED]
  + w_privileges_required × max(0, privileges_required - entry_privilege)
  + w_skill               × skill
  + w_resources           × resources
p_exploit = clip(sigmoid(z), min_p, max_p) × horizon_factor
```

`horizon_factor = 1 - exp(-horizon_days / 365)` normalised so that the default 90-day
horizon leaves the weights interpretable. Every term is written to `log_odds_terms`.

Impact:

```
confidentiality = records_by_function[f] × cost_per_record × impact_c × data_sensitivity
integrity       = integrity_loss_by_function[f] × impact_i
availability    = downtime_hours_by_privilege[privilege_gained] × downtime_cost_per_hour × impact_a
subtotal        = (confidentiality + integrity + availability) × regulatory_multiplier
reputational    = subtotal × reputational_fraction
total           = min(subtotal + reputational, max_impact)
```

`asset_overrides[endpoint_id]` (operator tier) replaces the computed total when present.

### 3.7 `graph/` and `select/` - Component C (Gap 9)

| File | Contents |
|---|---|
| `graph/privilege_map.py` | CWE and impact to `(privileges_required, privilege_gained)` |
| `graph/attack_graph.py` | `AttackGraphBuilder` over a `networkx.DiGraph` |
| `graph/reachability.py` | Maximum-probability path risk, path enumeration, betweenness |
| `graph/chain_scorer.py` | `ReachabilityChainScorer` implementing `ChainScorer` |
| `graph/monotone.py` | `assert_monotone_under_patching(...)` |
| `select/knapsack.py` | `select_under_budget(...)` → `SelectionResult` |

Graph construction:

* **Nodes** are `(asset, privilege)` states, `state:<asset>:<PRIVILEGE>`. The asset is the
  host for exploit transitions and `internet` for the entry node. Entry is
  `state:internet:NONE` (or the attacker's `entry_privilege`).
* **Exploit edge** per finding `f` on endpoint `e`:
  from `state:<host(e)>:<privileges_required(f)>` to `state:<host(e)>:<privilege_gained(f)>`
  with probability `p_exploit(f) × p_applicable(f)`. Only tiers ≤ `SCANNER` may create an
  edge; an edge that exists only because untrusted text asserted it is rejected and counted
  in `rejected_untrusted_edges`.
* **Privilege implication edges** with probability 1: a higher privilege on an asset implies
  every lower privilege on it.
* **Lateral edges** from observed `links_to` between endpoints on different hosts, tier
  `SCANNER`, probability `min_edge_probability`-floored constant.
* **Target nodes** are the states that hold value: `value` is the impact of the assets
  reachable at that privilege, weighted by `criticality ** criticality_weight_exponent`.

Risk and contribution:

```
R(G) = Σ over target nodes t:  value(t) × maxpathprob(entry → t)
reach_delta(f) = R(G) − R(G without the edges of f)
```

Maximum-probability path is computed with Dijkstra over `-log p`, which makes
`reach_delta ≥ 0` provable: removing edges can only lengthen every path. `monotone.py`
verifies this at runtime when `verify_monotonicity` is set and raises
`MonotonicityViolationError` otherwise; the property test patches random subsets and asserts
risk never rises.

Selection is a 0/1 knapsack over remediation hours maximising captured
`chain_adjusted_loss`, exact by dynamic programming on a half-hour grid, with a greedy
value-per-hour fallback above `max_items_for_exact` and a `rank_prefix` control that simply
takes the ranking in order until the budget is spent.

### 3.8 `rank/` - learning to rank

| File | Contents |
|---|---|
| `features.py` | `FeatureBuilder.build(enriched, chain, flags)` → `FeatureFrame` |
| `lambdamart.py` | `LambdaMartRanker` (`xgboost.XGBRanker`, `rank:ndcg`, groups by scan) |
| `likelihood_head.py` | `CostSensitiveExploitHead` (`XGBClassifier` + isotonic calibration) |
| `baselines.py` | CVSS-only, EPSS-only, KEV-first, scanner severity, expected loss, VMC chain, random |
| `explain.py` | `ShapExplainer` → `Explanation` with templated reason codes |
| `compose.py` | `rank_scan(...)` → `RankingResult` |
| `rank_guard.py` | `RankManipulationDetector` |

* Features are exactly `FEATURE_SPECS` in `core/models.py`. In an ablation cell, disabled
  components' columns are **dropped**, not zeroed, so a disabled component cannot leak.
* Query groups are scans: a scan's findings must be contiguous in the frame.
* Relevance is the graded label from `LabelSet` (0–4). Sample weights are
  `1 + log1p(impact / 1000)` when `impact_weighted_pairs` is set, which is the
  cost-sensitive objective Gap 7 asks for.
* Monotone constraints force KEV, EPSS, expected loss and chain contribution to be
  non-decreasing in score, which also bounds what an injection can achieve.
* **VMC chain baseline** reproduces Shimizu and Hashimoto: keep findings with KEV membership
  or EPSS ≥ 0.088, then order by CVSS ≥ 7.0, and report efficiency and coverage against it.
* `RankManipulationDetector` raises a `ManipulationAlert` when the untrusted share of
  absolute SHAP exceeds `max_untrusted_shap_share`, when re-ranking with untrusted tiers
  neutralised moves a finding by more than the allowed displacement, when model output
  contradicts tier ≤ 1 evidence, or when a canary leaked.

### 3.9 `eval/` - evaluation (Gaps 3, 4, 5, 7, 10)

| File | Contents |
|---|---|
| `labels.py` | `LabelBuilder`: KEV, exploit evidence, incident and synthetic-oracle labels only |
| `splits.py` | `TimeOrderedSplitter`, `LeaveOneAppOutSplitter`, `RandomSplitter` (control) |
| `metrics.py` | NDCG@K, Precision@K, Recall@K, RiskCapture@K, MAP, MRR, Kendall tau, mean rank of exploited, ROC-AUC, PR-AUC, MCC, minority F1, balanced accuracy, efficiency, coverage, workload reduction |
| `calibration.py` | Brier, expected calibration error, reliability bins |
| `bootstrap.py` | Paired bootstrap confidence intervals, Wilcoxon signed-rank |
| `benchmark.py` | `BenchmarkRunner`: every ranker on identical data, preprocessing and metrics |
| `ablation.py` | `FullFactorialAblation` over the eight cells × seeds, main effects and interactions |
| `minority.py` | `MinorityClassReport` per rare class |
| `simulation.py` | `LongitudinalSimulator`: weekly capacity, exposure days, expected-loss days |
| `report.py` | `ReportBuilder`: `report.md`, `report.json`, figures |

* `LabelBuilder` raises `LabelPolicyError` if asked to use CVSS, and `GroundTruthLabel`
  carries `cvss_used_as_label: Literal[False]` so the prohibition is type-level.
* Labels are version-aware: a finding whose version evidence is a `MISMATCH` is dropped
  rather than labelled positive because its CVE was exploited somewhere else.
* Time-ordered splits place a `gap_days` buffer between train and test, and the splitter
  asserts no test scan precedes any train scan. `RandomSplitter` exists only so the report
  can show what random splitting overstates.
* Efficiency and coverage follow Shimizu and Hashimoto: efficiency is confirmed-exploited
  findings as a share of the selected set; coverage is the share of all confirmed-exploited
  findings the selection retains.

### 3.10 `adversarial/` - robustness evaluation

| File | Contents |
|---|---|
| `corpus.py` | `load_corpus`, `AdversarialCase` loading and integrity checks |
| `injector.py` | `inject(case, scan, intel)` → mutated copies at the declared injection point |
| `detectors.py` | Pre-model pattern detector, post-model anomaly detector |
| `evaluator.py` | `AdversarialEvaluator`: clean run vs injected run |

The corpus holds at least sixty attack cases across every `InjectionCategory` and at least
twenty benign controls that must **not** trigger detection, so a detector that flags
everything scores badly on false positives. Attack success is defined as any of: canary
leak, feature movement beyond the case's expectation, or rank displacement beyond it.

### 3.11 `synth/`, `pipeline/`, `cli.py`

| File | Contents |
|---|---|
| `synth/world.py` | Latent world: hidden exploitability, attacker behaviour, chain structure |
| `synth/topology.py` | Endpoint and privilege topology per application |
| `synth/feeds.py` | Synthetic NVD, EPSS time series, KEV, exploit fixtures |
| `synth/pages.py` | Synthetic reference pages, optionally carrying corpus payloads |
| `synth/oracle.py` | Exploitation events, the ground truth no real corpus provides |
| `synth/generator.py` | `SyntheticDataset.generate(config)` writing scans, fixtures and the oracle |
| `pipeline/stages.py` | Pure stage functions |
| `pipeline/runner.py` | `PipelineRunner.run(config)` → `RunArtifacts`, writes `RunManifest` |
| `pipeline/artifacts.py` | Artifact layout and typed read/write under `runs/<run_id>/` |
| `cli.py` | Typer application |

The synthetic world exists because the gaps demand things no public dataset provides:
confirmed exploitation ground truth, counterfactual remediation outcomes, and known-correct
chain structure. Its oracle decides exploitation from the latent attacker model, not from
the features the framework computes, so the evaluation is not circular.

CLI commands: `synth`, `ingest`, `assess`, `enrich`, `chain`, `rank`, `train-ranker`,
`explain`, `evaluate`, `ablate`, `select`, `simulate`, `adversarial`, `report`, `run-all`,
`fetch-feeds`, `manifest`.

`train-ranker` exists because ranking one application cannot fit a model: one scan is one
query group, and a pairwise ranking objective has no pair to learn from inside a single
group however many findings it holds. The model is therefore fitted once, on a corpus with
several scans and confirmed-exploitation labels, and persisted at
`RankingConfig.model_path`; a run over a single scan loads that booster and *scores* with
it, which needs neither labels nor groups. `RankingResult` records which of the three
things happened — fitted here, scored by a model trained elsewhere, or neither — so a queue
ordered by expected loss is never presented as a learned ranking.
The application layer of section 6 adds four more: `scan` (assess an authorized target),
`serve` (the interactive site), `web` (export the static site) and `novelty` (compare against
the reviewed corpus).

---

## 4. Evaluation protocol

| Element | Choice |
|---|---|
| Splits | Time-ordered with a 30-day gap, three folds; leave-one-application-out for transfer; random only as a control |
| Labels | KEV, exploit evidence at maturity ≥ FUNCTIONAL, incidents, synthetic oracle. Never CVSS |
| Ranking metrics | NDCG@{5,10,20}, Precision@K, RiskCapture@K, MAP, MRR, mean rank of exploited, Kendall tau against CVSS |
| Classification metrics | ROC-AUC, PR-AUC, MCC, minority-class F1, balanced accuracy |
| Calibration | Brier, expected calibration error, reliability bins |
| Decision metrics | Efficiency, coverage, workload reduction, knapsack risk capture under budget |
| Longitudinal | 26-week simulation, exposure days and expected-loss days per policy |
| Baselines | CVSS-only, EPSS-only, KEV-first, scanner severity, expected loss, VMC chain, random |
| Ablation | Full 2³ factorial over A, B, C × seeds, with main effects, interactions and paired bootstrap intervals |
| Uncertainty | Paired bootstrap intervals and Wilcoxon signed-rank over per-scan metrics |
| Reproducibility | `RunManifest` records config hash, seeds, dataset hash and library versions |

---

## 5. What each gap gets

Full traceability is in `GAP_TRACEABILITY.md`. In short: Gap 1 is the expected-loss
construct, Gap 2 the `AttackerModel`, Gap 3 the label policy and CVSS-agreement features,
Gap 4 the frozen protocol and manifest, Gap 5 the factorial ablation, Gap 6 the web
application evidence and scanner parsers, Gap 7 cost-sensitive weights with MCC and
minority reporting, Gap 8 sector and application transfer with multilingual evidence, Gap 9
the directed multi-hop attack graph with monetary impact, Gap 10 the knapsack selection and
longitudinal exposure simulation.

---

## 6. The application layer

Sections 1 to 5 describe the framework as a library and a research instrument. This section
describes it as a tool someone opens and uses. The addition does not change any contract
above: the assessment layer's product is an ordinary `Scan`, so everything downstream runs
unchanged whether the findings came from OWASP ZAP, from Burp, or from the framework itself.

### 6.1 `scan/` - assessing a target directly

Not every team has a scanner report to hand. When none exists, the framework produces one.

| File | Contents |
|---|---|
| `safety.py` | Authorization gate, scope lock, private-host refusal, rate limiter, robots policy, budgets |
| `http.py` | Rate-limited client with a byte cap and scope re-checked on every redirect |
| `crawler.py` | Same-origin, depth-limited crawl; records forms, never submits them |
| `checks.py` | The check registry, each entry carrying a CWE and a rationale |
| `passive.py` / `active.py` | The two profiles' check sets |
| `adapters.py` | Detects and drives an installed Nuclei, ZAP, Wapiti or Nikto, then parses its output |
| `runner.py` | `assess_target(request)` - the single entry point |

The safety posture is the design, not a disclaimer around it:

* **Authorization is a required field, not a flag.** `ScanRequest.authorized` must be `True`
  and carry a note, or the scanner raises before it opens a socket. The note is recorded in
  the generated report, so the assessment carries its own justification.
* **Passive is the default** and sends no payloads at all: it fetches pages and reads what
  comes back. The active profile adds only benign probes from a fixed allowlist - a random
  marker to detect output reflection, an `OPTIONS` request, a well-known path - and a test
  asserts nothing else can be sent. No exploitation, no authentication attacks, no fuzzing,
  no writes.
* **Scope is locked** to the authorized host, re-checked on every redirect, with private and
  cloud-metadata addresses refused unless explicitly permitted.
* **Every limit is enforced in one place**: requests per second, pages, depth, response
  bytes, wall clock. The request rate is the property that makes a scan a nuisance to a
  third party, so it is the one held low; a loopback target is not a third party and gets a
  higher default rate, which any explicit value overrides.
* **`robots.txt` is read, but advisory by default.** It is a crawler convention, not an
  access control - it keeps search engines out of a directory and keeps nobody else out of
  anything, and an attacker treats a `Disallow` list as an index of what the operator
  thought worth hiding. Honouring it in an authorized assessment therefore reports those
  paths as clean having never requested them, which is the most misleading result this
  scanner can produce; ZAP, Burp and Nuclei all ignore it for the same reason. The file is
  fetched and parsed in both modes and the paths it names are reported in both
  (`ScanOutcome.robots_named` counts them, `robots_blocked` counts only ones actually
  refused). `respect_robots=True` makes the rules bind. `Crawl-delay` is honoured
  regardless: unlike `Disallow`, it is a statement about what the server can take.
  What makes the scan safe is the attestation, the locked scope, the rate limit, the
  dangerous-link refusal and the probe allowlist - controls on what the scanner may *do*.
  `robots.txt` only ever constrained where it looked.
* **A single-page application is not mistaken for a small one.** `scan/spa.py` recognises an
  application shell - a framework mount element, no anchors, script bundles - and recovers
  the request paths the client code names in its own JavaScript string literals. Those are
  queued *behind* everything link-reachable, so mining spends only budget that link
  following left over, and each one is then an ordinary candidate subject to scope, robots,
  the dangerous-link refusal and the budgets. Client-side view routes are read but never
  fetched: under hash routing the server never sees them and under history routing they all
  return the same shell, so requesting them would inflate the page count without producing
  one distinct response. A response that is byte-identical to the shell is recognised as the
  framework's catch-all and neither recorded as an endpoint nor followed for links.
* **Interpolated identifiers are truncated, never invented.** `` `/rest/basket/${id}` ``
  yields `/rest/basket/`. Fabricating an object id is how a read-only scan starts reading
  other people's rows.
* **Coverage is reported, not implied.** `ScanOutcome.coverage_notes` states what the scan
  could not reach and why: the shell, the mined paths a budget cut short, the paths
  `robots.txt` disallowed. A short finding list is otherwise ambiguous between a clean
  application and a scan that never started, and the reader will assume the flattering one.
* **An origin-wide misconfiguration is reported once.** A missing security header is one
  server's defect, not one per page crawled; `Check.site_wide` marks the checks whose subject
  is the origin and `collapse_site_wide` merges their findings by evidence, naming every
  affected path on the survivor. Without it a queue's length is a function of crawl budget
  rather than of risk.

Preferring an installed scanner over the built-in one is deliberate. ZAP and Nuclei are
better scanners than anything worth writing here; the framework's contribution is what
happens to the findings afterwards. A containerised scanner is pointed at
`host.docker.internal` when the target is loopback, because `localhost` inside a container
is the container - the most common way a containerised scan silently returns nothing and is
read as a clean result.

### 6.2 `report/` - the assessment report

`eval/report.py` writes the research report: metrics, baselines, ablation. This package
writes the document a security team receives.

| Section | What it answers |
|---|---|
| Executive summary | How much is at risk, how concentrated, what to do first |
| What we would fix first | The top of the queue, each with its own reason codes |
| The remediation plan | What fits the budget, what it retires, what deferring costs |
| Attack chains | The reachable paths, as sentences, with the findings they depend on |
| Per-finding detail | Evidence, applicability, impact decomposition, remediation guidance |
| Methodology | The priority formula with this run's parameters, so it can be argued with |
| Assurance and caveats | What was not done, and which numbers are estimates |

Two rules make the report defensible. No language model writes any of it: the prose is
generated deterministically from the data, so the same run always produces the same
document. And no number is invented: a quantity that was not measured is reported as not
measured rather than estimated silently.

### 6.3 `web/` - the interactive site

The static export of section 3 remains, and the same page gains an application in front of
it. A FastAPI server accepts either an uploaded scanner report or an authorized target,
runs the pipeline, streams progress, and renders the result into the dashboard that already
existed.

* `POST /api/analyze` starts a job; `GET /api/jobs/{id}/events` streams its phase and log.
* `GET /api/jobs/{id}/result` returns the same `DashboardData` payload the static site
  consumes, so one front end serves both modes.
* `GET /api/jobs/{id}/report.{md,html,json}` returns the generated assessment report.
* The page probes the API once on load; with no server it falls back to the bundled run and
  hides the controls that would need one.

The server binds to localhost, issues a per-process token for mutating endpoints, caps
request bodies, and returns errors as JSON without a stack trace. It is a local tool and
says so.

### 6.4 `novelty/` - comparison against the reviewed corpus

The review surveys 43 studies. This package encodes what each of them does as a capability
vector, and computes where this framework actually stands.

* `capability_matrix()` - per capability, which studies have it, partially have it, or lack it.
* `unique_capabilities()` - capabilities no reviewed study has.
* `nearest_neighbours()` - the closest prior work by capability overlap, so the comparison is
  against the strongest alternatives rather than the weakest.
* `novelty_verdict()` - a claim level per capability (`novel`, `novel_combination`,
  `incremental`, `not_novel`), plus `threats_to_novelty`: the grounds on which a reviewer
  could push back.

The output is computed from the corpus rather than asserted, and it is required to name what
is *not* novel. Learning-to-rank over vulnerability features, exploitation-evidence
grounding, asset context and attack-graph position all exist in prior work. A novelty
analysis that failed to say so would not survive the first reviewer who read it.

### 6.5 `intel/` - retrieved exploit intelligence, as ranking features

The review's proposal is an agent that "automates the retrieval and interpretation of
references and exploit information published on the internet". Section 3.2's feed layer
reads the references a CVE record already lists, which is retrieval of what NVD knew. This
package is the other half: it searches for exploit material NVD does not link, reads what it
finds, and turns it into numbers.

| File | Contents |
|---|---|
| `queries.py` | Search formulation from CVE, product, version and weakness class; deduplicated against the references the feeds already supplied |
| `provider.py` | `SearchProvider` protocol; `anthropic_search.py` uses the web search and web fetch server tools, `offline.py` serves recorded fixtures |
| `agent.py` | `ExploitIntelAgent.gather(...)` - the two-phase loop, the sandbox, the cache, the as-of guard |
| `summarize.py` | The cited, model-written summary that reaches the report |
| `cache.py` | Content-hash cache, so a rerun is free and byte-identical |

**Two phases, for a reason.** Citations and structured output cannot coexist in one request:
the API rejects the combination. Phase one is an agentic call with the search and fetch tools
and citations enabled, which gathers and reads. Phase two is a separate toolless call over
the sanitized phase-one material, using a bounded output schema, which is where a claim
becomes a number. The split is forced by the API and happens to be the right architecture
anyway: gathering and judging are different jobs.

**Everything retrieved is untrusted.** The framework is now actively reading whatever the
internet returns for a query it composed itself, which is precisely the threat ADR-002 was
written for. Retrieved text is `REFERENCE_PAGE` tier, passes the same sandbox, and is subject
to the same influence budget as any other page. There is no second path into the score.

**The output grounds the ranking, not just the report.** This is the architectural point and
the easiest one to get wrong. The review is explicit that the model is a feature-extraction
layer and the ranker decides. Seven Component A features carry what was retrieved into the
feature vector alongside CVSS, EPSS and KEV:

| Feature | What it carries |
|---|---|
| `a_intel_documents` | How much material was found |
| `a_intel_public_exploit_urls` | How many distinct public exploits or proofs of concept |
| `a_intel_active_exploitation` | Whether retrieved material claims exploitation is happening |
| `a_intel_confidence` | How confident the extraction is in its own reading |
| `a_intel_corroborates_feeds` | The retrieved material agrees with the curated feeds |
| `a_intel_contradicts_feeds` | It contradicts them |
| `a_intel_injection_signals` | How much attempted manipulation arrived with the evidence |

Corroboration and contradiction are separate features rather than one signed axis, because a
web claim that agrees with CISA and one that contradicts it are different evidence, not
opposite ends of a scale. The injection-signal count is deliberately a feature and not only a
guard: it lets the ranker learn to discount evidence that arrived with a manipulation attempt
attached, rather than the framework silently trusting it or silently dropping it.

All seven take a neutral zero when gathering is disabled, unavailable, or found nothing.
Those three are the same state, and a finding with no intelligence must not be scored
differently from one whose search came back empty.

**Offline by default.** Gathering is off unless enabled, and the offline provider serves
recorded fixtures so the suite needs no key and no network.

**Time honesty, and the remedy for it.** Retrieved intelligence describes the internet as of
now. Whether that is correct depends on which run it is:

* **An operational run** scans a target now, so the scan and the search describe the same
  moment. Live search proceeds silently; there is no warning and no flag to remember.
* **An uploaded report from months ago** does not. Rather than refusing, the framework says
  what the mismatch is and offers the fix the situation actually has: re-scan the target, so
  the scan and the intelligence agree. The decision is returned as a structured outcome, not
  only a log line, so the interactive site can turn it into an action.
* **The research protocol** replays historical scans for time-ordered evaluation. There a
  live search imports knowledge that did not exist at the time and would inflate every
  metric, so it stays refused outright and fixtures are the correct answer.

Either way every retrieved document is stamped with `retrieved_at` and the result records the
scan date it was gathered for, so a reader months later can tell what the assessment knew and
when it knew it.
