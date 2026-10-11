# VulnPriority

**Decides which web application vulnerabilities to fix first.**

A scan of one web application returns hundreds of findings, and sorting them by CVSS severity
does not tell you which ones an attacker will actually use. VulnPriority ranks them by
**expected loss**: how likely the chosen attacker is to exploit each finding, multiplied by what
it would cost the business. It combines the scanner output, public vulnerability intelligence
(CVSS, EPSS, CISA KEV), advisories and exploit write-ups, and the finding's place on attack
paths through the application.

<p align="center">
  <img src="assets/readme/architecture-overview.png" width="560"
       alt="Scanner reports, vulnerability databases and public write-ups are prepared, assessed by three components (AI evidence, attacker and cost, attack paths), combined by a learned ranker and turned into a prioritised fix list">
</p>

## How it works

- **A · AI evidence.** A sandboxed language model reads advisories and exploit write-ups and
  fills in a fixed set of facts. It never decides the order itself, and text an attacker could
  control has a capped influence on any score.
- **B · Attacker and cost.** Five attacker profiles ship (opportunistic, targeted criminal,
  insider, advanced persistent, content-controlling). Switching profile re-ranks the queue
  without retraining.
- **C · Attack paths.** Findings are scored by the valuable assets they open up, alone or
  chained with others, so a quiet chokepoint can outrank a loud dead end.
- **Ranking and plan.** A gradient-boosted ranker (XGBoost LambdaMART) combines the signals,
  explains each position, and picks the fixes worth the most within the engineering hours you
  have.

<p align="center">
  <img src="assets/readme/attack-graph-example.png" width="680"
       alt="Example attack graph: SQL injection on /search is the most likely route to the customer database">
  <br><sub>Percentages are the attacker's chance of succeeding at each step; the bold route is
  the most likely path to the customer database.</sub>
</p>

It runs **offline by default**: no API key and no network. A synthetic world supplies the
confirmed-exploitation ground truth that no public dataset provides.

## Install

Python 3.12 or newer.

```bash
python -m pip install -e ".[web]"
```

Other extras: `.[gemini]` or `.[llm]` for a live language model, `.[dev]` for the tests. If
the `vulnpriority` command is not found (common on Windows), use `python -m vulnpriority`.

## Quick start

```bash
# Run the whole pipeline on a generated synthetic world
vulnpriority run-all --synthetic

# Rank your own scanner report (OWASP ZAP, Burp, Nuclei, Wapiti, Nikto)
vulnpriority run-all --scan report.json

# The same, for a different attacker
vulnpriority run-all --scan report.json --set component_b.attacker_preset=insider

# Open the dashboard: upload a report, or assess a target you are authorized to test
vulnpriority serve
```

`vulnpriority --help` lists every command. Settings live in `configs/`; any of them can be
overridden with `--set key=value`. `configs/live.yaml` switches on live feeds and a language
model.

## Dashboard

Each position in the queue comes with the reasons behind it. These screenshots are from a
run on the synthetic world.

<p align="center">
  <img src="assets/readme/dashboard-explanation.png" width="820"
       alt="Findings in VulnPriority order, with the explanation panel for the top finding: impact, probability of exploitation, expected loss and the reasons for its position">
  <br><sub>One scan's findings in fix order, with the panel explaining the top-ranked one.</sub>
</p>

<p align="center">
  <img src="assets/readme/dashboard-attack-chains.png" width="820"
       alt="Attack chains view: routes an attacker could take through the application and the finding that holds each one open">
  <br><sub>Attack routes through the application, and the finding that holds each one open.</sub>
</p>

`vulnpriority web --run runs/<run_id>` exports a finished run as a static page that opens
without a server.

## Results

Every ranker is tested the same way: time-ordered splits with a 30-day gap, labels from
confirmed exploitation (never CVSS), and the same metrics. VulnPriority is compared with six
practitioner baselines and nine methods re-implemented from the research literature
(`src/vulnpriority/rank/literature.py`). On a synthetic world of 24 scans and 901 findings:

| Ranker | Finds exploited flaws early (NDCG@10) | Monetary risk in top 10 |
|---|---|---|
| **VulnPriority** | 0.883 | **0.741** |
| VPSS random forest (Hore et al.) | **0.900** | 0.508 |
| LICALITY (Zeng et al.) | 0.785 | 0.665 |
| VMC chain (Shimizu & Hashimoto) | 0.702 | 0.683 |
| EPSS only | 0.693 | 0.673 |
| CVSS only | 0.503 | 0.667 |

VulnPriority is level with the best competitor at finding exploited flaws (the confidence
intervals overlap) and captures the most monetary risk. Under a stricter test that gives no
credit for answers a ranker can read straight from KEV, it ranks first (0.610 against 0.573).

<p align="center">
  <img src="assets/readme/literature-comparison.png" width="760"
       alt="All sixteen rankers: NDCG@10 and RiskCapture@10 with 95% bootstrap intervals">
</p>

Over a simulated 26 weeks with 20 engineering hours a week, it fixed 12 of 145 exploited flaws
before they were first exploited (1 of 145 for CVSS, EPSS or KEV ordering), and cut the time
exploited flaws stayed open by 32.5% compared with CVSS.

To reproduce the comparison on a finished run:

```bash
python scripts/compare_literature.py runs/<run_id> inclusive   # or: strict
```

These numbers come from synthetic data. They show that the method and the evaluation work,
not how it will perform on a particular real application.

## The shipped model

`src/vulnpriority/rank/models/lambdamart.ubj` ranks a single real scan, which on its own has
too few labelled findings to learn from. It is trained on 10,000 synthetic scans that are not
committed but are fixed by a seed, so the model can be rebuilt:

```bash
python scripts/build_training_corpus.py --scans 10000
```

## Tests

```bash
python -m pytest -q
```

Everything runs offline, including property tests (fixing a vulnerability never increases
attack-path risk) and adversarial tests against a corpus of prompt-injection attacks.

## Limitations

The ground truth is synthetic, so absolute numbers validate the method rather than predict
field performance. Attacker and cost parameters are configured priors, not fitted to real
incidents, and attack-path risk can be understated when many independent routes reach the
same asset.
