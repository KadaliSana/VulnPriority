# VulnPriority

**AI-driven automated web application vulnerability prioritization.**

A single web application scan returns hundreds of findings. Sorting them by CVSS does not tell
you which ones an attacker will actually use: published severity correlates with real-world
weaponisation at about rho = 0.1, and three quarters of all CVEs cluster into ten distinct
scores. This framework orders a scan's findings by **expected loss** instead, combining what
the scanner saw, what public vulnerability intelligence says, what the referenced advisories
and exploit reports mean, and where the finding sits on the path to something worth stealing.

It is the implementation of the framework proposed in the accompanying literature review, and
it is built so that every claim in that review's gap analysis is answered by a module and a
test rather than by prose. See [GAP_TRACEABILITY.md](GAP_TRACEABILITY.md).

---

## What it does

```
scanner output ─► ingest ─► intelligence ─┬─► A  semantic assessment   (sandboxed model)
                                          ├─► B  attacker model + monetary impact
                                          ├─► C  attack graph contribution
                                          └─► features ─► LambdaMART ─► explained ranking
                                                              │
                                          evaluation ◄────────┴───────► selection, simulation,
                                          (splits, labels, metrics,      adversarial robustness
                                           ablation, calibration)
```

**Priority is defined, not asserted.** `expected_loss = P(exploit | evidence, attacker) × impact`,
in the configured currency, with every term in the probability recorded for audit.

**The attacker is explicit.** Five presets ship (opportunistic, targeted criminal, insider,
advanced persistent, content-controlling). Changing the attacker re-prioritises the queue
without retraining, because the adversary is a parameter rather than an assumption.

**The model extracts features; it does not decide.** Advisories and application responses are
read inside a sandbox and turned into bounded, schema-validated assessments. A gradient-boosted
ranker learns the ordering from confirmed-exploitation labels.

**Attacker-controlled text is contained.** Evidence carries a trust tier, and a tier's
influence on any feature is capped. A reference page can move a normalised feature by at most
0.35; text authored by the target itself by at most 0.15; neither can argue away a KEV listing.
Robustness is measured against a corpus of injection attacks, in the test suite.

**Chains are first class.** Findings are scored by how much reachable compromise they enable,
over a directed multi-hop attack graph, so a low-severity chokepoint outranks a high-severity
leaf. Patching can never increase computed risk, and a property test proves it.

**It runs offline.** The default configuration needs no API key and no network. A synthetic
world with a latent exploitation oracle supplies the confirmed ground truth and the
counterfactual remediation outcomes that no public dataset provides.

---

## Install

Python 3.12 or newer.

```bash
python -m pip install -e .
```

Optional extras: `pip install -e ".[gemini]"` for the free-tier backend, `".[llm]"` for
Anthropic, `".[web]"` for the interactive site, `".[dev]"` for pytest.

Every command below also works as `python -m vulnpriority ...`. On Windows the console script
lands in Python's user scripts directory, which is often not on `PATH`, so `vulnpriority serve`
can fail with "not recognized as a name of a cmdlet" even though the install succeeded. The
module form needs nothing but the interpreter that already imported the package.

## Quick start

Generate a synthetic world, run the whole pipeline on it, and write the report:

```bash
vulnpriority run-all --config configs/default.yaml --synthetic
```

Rank a real OWASP ZAP report:

```bash
vulnpriority ingest --input scan.json --scanner zap --output runs/myapp
vulnpriority run-all --config configs/default.yaml --scan runs/myapp/scan.json
```

Use live feeds and the Anthropic backend:

```bash
export ANTHROPIC_API_KEY=...        # Windows: setx ANTHROPIC_API_KEY ...
vulnpriority run-all --config configs/live.yaml --scan runs/myapp/scan.json
```

Re-prioritise the same scan for a different adversary:

```bash
vulnpriority rank --config configs/default.yaml --attacker insider
```

## The results website

The framework ships with a website that presents a run: the remediation queue with the
reasoning behind each position, the attack graph and its highest-value paths, the evaluation
against baselines, the component ablation, budget and exposure outcomes, the adversarial
robustness measurements, and a live traceability table tying each research gap to the numbers
this run produced.

It runs in two modes.

**As an application.** Start the server, open it, and give it either a scanner report to
upload or a target to assess. It runs the pipeline and shows the prioritized queue, the attack
chains, the generated report and the novelty analysis.

```bash
pip install -e ".[web]"
vulnpriority serve
```

Assessing a target requires an explicit authorization attestation: you state that you are
authorized to test it, and that statement is recorded in the report. The scanner stays on the
host you name, rate-limits itself, refuses to follow destructive-looking links, and sends no
exploitation payloads. If OWASP ZAP, Nuclei, Wapiti or Nikto is installed, it drives those
instead of its own crawler.

`robots.txt` is read but **advisory by default**. It is a crawler convention, not an access
control: it keeps search engines out of a directory and keeps nobody else out of anything,
and an attacker reads it as a list of what the operator thought worth hiding. An authorized
assessment that honours it reports those paths as clean when it never looked at them. Every
dedicated scanner ignores it for the same reason. The paths it names are reported either
way, and `--respect-robots` restores the crawler-courtesy behaviour; `Crawl-delay` is
honoured regardless, because that one is a statement about what the server can take.

**Single-page applications are visible to it.** An application whose HTML is one mount
element and three bundles gives a link-following crawler nothing to follow, so a naive
crawl ends after the shell and reports a short queue - which reads as a clean application
rather than as a scan that never started. Instead, the request paths the client code names
in its own JavaScript are recovered and crawled, subject to every existing control; routes
that come back as the shell are recognised and not counted twice. On OWASP Juice Shop that
is the difference between 6 endpoints and 90.

**What the scan could not see is stated, not implied.** Every assessment ends with coverage
notes: that the target was a single-page application, how many mined paths a budget cut
short, which paths `robots.txt` disallowed and therefore went unassessed. A short list of
findings is ambiguous on its own, and these sentences are what disambiguate it.

**As a static page.** Export a finished run to a self-contained page: one HTML file, one
stylesheet, one script and one data document. It opens from disk with no server and no
network, which matters because offline reproducibility is the framework's own claim.

```bash
vulnpriority web --run runs/<run_id> --output site
python -m vulnpriority.web --demo --serve      # see the page before running anything
```

## Commands

| Command | What it does |
|---|---|
| `synth` | Generate the synthetic world: scans, feed fixtures, reference pages, exploitation oracle |
| `ingest` | Parse ZAP, Burp, Nuclei or canonical JSON into a `Scan`, template paths, correlate root causes |
| `fetch-feeds` | Populate the feed cache for an as-of date |
| `assess` | Component A: asset criticality, exploitability, applicability |
| `enrich` | Component B: attacker likelihood, monetary impact, remediation cost, expected loss |
| `chain` | Component C: build the attack graph and score each finding's reachability contribution |
| `rank` | Build features, run the ranker, produce the explained remediation queue |
| `train-ranker` | Fit the LambdaMART ranker on a labelled corpus and save it, so single-scan runs score with a learned model rather than falling back |
| `explain` | Show the factors behind one finding's position |
| `evaluate` | Time-ordered evaluation against confirmed-exploitation labels, with baselines |
| `ablate` | Full 2³ factorial ablation over components A, B and C |
| `select` | Knapsack selection under a remediation budget |
| `simulate` | Longitudinal exposure simulation per policy |
| `adversarial` | Run the prompt-injection and rank-manipulation corpus |
| `report` | Write `report.md`, `report.json` and figures |
| `run-all` | The whole chain, one command |
| `scan` | Assess an authorized target directly, with no scanner report |
| `serve` | Run the interactive site: upload a report or assess a target |
| `web` | Export the static results website from a finished run |
| `novelty` | Compare this framework's capabilities against the reviewed prior work |
| `manifest` | Print the run manifest: config hash, seeds, dataset hash, library versions |

## Configuration

`configs/default.yaml` is offline and deterministic. `configs/live.yaml` uses live feeds and
the Anthropic backend. Attacker presets live in `configs/attacker_models/`, impact models in
`configs/impact_models/`, and the injection pattern library in `configs/sandbox/`.

Any field can be overridden on the command line, for example `--set ranking.n_estimators=500`.

## Evaluation

The protocol is fixed in advance and recorded in the run manifest, because the review's
clearest finding is that published results are not comparable to each other.

| Element | Choice |
|---|---|
| Splits | Time-ordered with a 30-day gap; leave-one-application-out for transfer; random only as a control |
| Labels | KEV, exploit evidence, incidents, synthetic oracle. Never CVSS |
| Ranking | NDCG@{5,10,20}, Precision@K, RiskCapture@K, MAP, MRR, mean rank of exploited |
| Classification | ROC-AUC, PR-AUC, MCC, minority-class F1, balanced accuracy |
| Calibration | Brier, expected calibration error, reliability bins |
| Decision | Efficiency, coverage, workload reduction, risk capture under budget |
| Longitudinal | 26-week exposure-day simulation per policy |
| Baselines | CVSS-only, EPSS-only, KEV-first, scanner severity, expected loss, threat-chained, random |
| Ablation | Full factorial over A, B, C with main effects, interactions and paired bootstrap intervals |

## Tests

```bash
python -m pytest -q
```

Everything runs offline. The suite includes property tests (patching never increases graph
risk), adversarial tests (attack success rate and canary leak rate are assertions), temporal
leakage tests, and an end-to-end run on synthetic data.

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) - module-by-module specification
- [docs/adr/ADR-001](docs/adr/ADR-001-decision-theoretic-priority-with-learned-ranking.md) - why expected loss with a learned ranker
- [docs/adr/ADR-002](docs/adr/ADR-002-trust-tiered-untrusted-content-sandbox.md) - why trust tiers and influence budgets
- [docs/adr/ADR-003](docs/adr/ADR-003-monotone-attack-graph-contribution.md) - why maximum-probability paths
- [GAP_TRACEABILITY.md](GAP_TRACEABILITY.md) - every research gap mapped to code and tests

## Limitations

The exploitation oracle used for offline evaluation is synthetic. It is deliberately driven by
latent variables the framework never sees, so the evaluation is not circular, but a synthetic
world is not a real one and the absolute numbers should be read as protocol validation rather
than as field performance. Attacker and impact parameters are configured, not fitted; the
weights are defensible priors awaiting deployment data. Influence budgets are judgement calls
measured by the adversarial corpus rather than derived. Maximum-probability paths understate
risk when many independent paths reach the same target.
