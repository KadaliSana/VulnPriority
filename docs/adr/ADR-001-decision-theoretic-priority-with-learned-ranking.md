# ADR-001: Decision-theoretic priority with a learned ranker over component features

**Status:** Accepted
**Date:** 2026-09-12
**Deciders:** M. Srikar Bharadwaj, K. Navneet Sai, K. Shasanth Reddy

## Context

A single web application scan returns hundreds of findings, more than any team can remediate
in a patch cycle. The standard practice is to sort by CVSS base score. The literature review
that precedes this work establishes that this does not work and, more importantly, that the
replacements proposed so far each fail in a specific, documented way.

The forces at play:

- **The severity metric is not a risk metric.** CVSS base scores correlate with weaponisation
  at rho = 0.099, the top ten scores account for roughly three quarters of all CVEs, and the
  two main scoring sources agree on severity category only 65.9% of the time.
- **Labels derived from that metric are unreliable.** Most learning-based studies train and
  test against CVSS-derived labels, which puts an empirical ceiling on what any reported
  accuracy figure can mean.
- **Priority is not defined.** The corpus contains at least five incompatible
  operationalisations of the target construct: technical severity, probability of
  exploitation, expected financial loss, structural exposure reduction, and remediation
  decision quality under resource constraints.
- **The adversary is absent.** Most frameworks model the vulnerability, not the attacker.
- **Context beats formula refinement.** The strongest ablation in the corpus shows CVSS alone
  reaching ROC-AUC 0.656, confidentiality/integrity/availability sub-metrics adding 0.034, and
  asset context adding 0.231.
- **Position in a graph expresses what per-vulnerability scoring cannot.** Chokepoints and
  asset-free nodes are invisible to any per-CVE score by construction.
- **The evidence itself is attacker-controlled.** The framework reads the target's responses
  and pages fetched from the internet. A model in that loop is a manipulation surface.

The framework must also be buildable and defensible: a security team has to be able to say
why a finding is ranked where it is, and a reviewer has to be able to reproduce the numbers.

## Decision

Priority is **expected loss over an explicit attacker's horizon**:

```
expected_loss = P(exploit | evidence, attacker) × impact
```

with a chain adjustment that adds the finding's contribution to reachable compromise:

```
chain_adjusted_loss = expected_loss + chain_weight × reach_delta
```

The language model is a **feature extractor, not the decision maker**. It reads references,
advisories and application responses inside a sandbox and emits bounded, schema-validated
assessments of exploitability, applicability and asset criticality. Those assessments join
CVSS, EPSS, KEV, exploit availability and attack-graph position in a feature vector, and an
XGBoost LambdaMART model learns the ordering from confirmed-exploitation labels. The
decision-theoretic expected-loss ordering is retained as a first-class ranker, so the learned
model always has a principled ordering to beat rather than replacing one.

Untrusted evidence is tiered and budgeted: a reference page may move a normalised feature by
at most 0.35, target-authored content by at most 0.15, and neither may push the exploitation
probability below the floor implied by curated-feed evidence.

## Options Considered

### Option A: Refine the severity formula (VRSS/VIEWSS lineage)

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | Very low; no training data, no model serving |
| Scalability | High; arithmetic over existing fields |
| Team familiarity | High |

**Pros:** Fully auditable, defensible under compliance, runs where no labelled data exists,
reproducible by anyone with the formula.
**Cons:** Cannot represent contextual, temporal or relational risk. Inherits the score
clustering that motivated the work. The corpus shows these formulae are poorly calibrated
against actual exploitation, which is the failure we are trying to fix.

### Option B: End-to-end language model ranking

| Dimension | Assessment |
|---|---|
| Complexity | Low to build, high to trust |
| Cost | High; one or more model calls per finding, per rerun |
| Scalability | Poor at hundreds of findings per scan across many scans |
| Team familiarity | High |

**Pros:** Minimal engineering, handles heterogeneous evidence natively, produces fluent
explanations.
**Cons:** The ranking becomes a text-generation artefact. It is non-deterministic across
models and parameters, it cannot be calibrated or ablated, and the surveyed evidence shows
model assessments collapsing on exactly the rare high-severity classes that matter most
(zero F1 on high attack complexity in one study). Decisively, it hands the ranking to an
adversary who controls the text: the page being read can ask for a rank.

### Option C: Graph-only prioritization (attack-path lineage)

| Dimension | Assessment |
|---|---|
| Complexity | High |
| Cost | Moderate compute, high data acquisition |
| Scalability | Moderate; graph size is the binding constraint |
| Team familiarity | Moderate |

**Pros:** The only paradigm that expresses chaining and chokepoints. Effective resistance and
similar formulations give monotonicity guarantees. Identifies critical assets carrying no
CVEs at all.
**Cons:** Needs an accurate topology the scanner cannot always supply. Says nothing about
whether a given vulnerability is exploitable in practice, since edge probabilities still have
to come from somewhere. Evaluated in simulation rather than against observed exploitation.

### Option D: Decision-theoretic target, model as feature extractor, learned ranking over A/B/C features (chosen)

| Dimension | Assessment |
|---|---|
| Complexity | High; four subsystems plus an evaluation stack |
| Cost | Moderate; one bounded model call per finding per assessment, cached, with a free offline backend |
| Scalability | Good; training is per scan group, inference is a tree ensemble |
| Team familiarity | Moderate; XGBoost ranking and SHAP are standard, the sandbox is not |

**Pros:** Names the construct, so results are interpretable and comparable. Keeps the
adversary explicit and configurable. Uses the model where it is strong (reading unstructured
advisories) and not where it is weak (arithmetic, calibration, consistency). Supports full
factorial ablation because the components are separable. Bounds what a prompt injection can
achieve, structurally rather than by hoping the model resists. Runs completely offline for
tests and reproduction.
**Cons:** Substantially more code and more moving parts. Needs confirmed-exploitation labels,
which are scarce, hence the synthetic world. Attacker and impact parameters must be chosen,
and a badly chosen attacker model produces confidently wrong numbers. The learned layer needs
enough scans to train on.

## Trade-off Analysis

**Interpretability against expressiveness.** Option A is the most auditable and the least
expressive; Option B is the reverse and is additionally non-reproducible. The chosen design
keeps an auditable arithmetic core, the expected-loss computation with every log-odds term
recorded, and puts learning on top of it rather than underneath it. A reviewer can read the
ranking two ways: the learned score with SHAP attributions, and the expected loss with its
term breakdown.

**Where the model sits.** Between Option B and Option D, the deciding argument is adversarial,
not statistical. The framework's input includes text authored by the system under test. If the
model decides the rank, controlling the text controls the rank. As a feature extractor with
bounded outputs, influence budgets and monotone constraints on trusted features, the worst a
successful injection achieves is a bounded perturbation of a few features, and the rank-guard
detects the attempt. That property is testable, and the adversarial corpus tests it.

**Chaining against practicality.** Option C alone cannot say whether an edge exists in
practice. Component C therefore consumes Components A and B rather than competing with them:
edge probabilities are exploitation probabilities, so the graph inherits grounded numbers, and
maximum-probability paths keep the contribution provably non-negative under patching.

**Cost of complexity.** The honest cost of Option D is the evaluation stack, which is roughly
as large as the pipeline itself. That is deliberate. The review's clearest finding is that
the field's results are not comparable because protocols differ; a framework that adds another
incomparable number is not worth building.

## Consequences

**What becomes easier**

- Priority means one thing, in money, and every number is traceable to its inputs.
- Components can be turned off independently, so a full factorial ablation quantifies what
  each contributes rather than asserting it.
- Swapping the attacker or the impact model re-prioritises the whole queue without retraining.
- The framework runs end to end with no API key and no network, so results reproduce.
- Prompt-injection resistance is measurable, and regressions in it fail the test suite.

**What becomes harder**

- More surface to maintain: four subsystems, two model backends, a feed layer with offline and
  live modes.
- Impact and attacker parameters are now explicit choices that must be justified, where a
  severity formula let them stay implicit.
- The learned ranker needs labelled scans; on a first deployment it falls back to the
  expected-loss ordering until enough history exists.
- The synthetic world must be maintained alongside the real pipeline, and its oracle must stay
  independent of the framework's own features or the evaluation becomes circular.

**What we will need to revisit**

- Influence budgets (0.35 and 0.15) are judgement calls. The adversarial corpus measures the
  attack success rate they buy; if it is non-zero they must come down.
- Maximum-probability paths under-count risk when many independent paths exist. If real
  topologies show this mattering, move to an exact noisy-OR over enumerated paths and prove
  monotonicity over the same path set.
- The logistic attacker model is deliberately simple. If deployment data becomes available,
  its weights should be fitted rather than assumed, with the presets kept as priors.
- CVSS v4.0 handling is currently the same version-aware selection as v3.x; it deserves its
  own treatment once adoption is wide enough to matter.

## Action Items

1. [x] Freeze shared contracts in `src/vulnpriority/core/` before any parallel implementation.
2. [x] Implement Components A, B and C behind the interfaces, with an offline deterministic
       backend so the pipeline runs with no API key.
3. [x] Implement the evaluation stack: time-ordered splits, confirmed-exploitation labels,
       ranking and calibration metrics, full factorial ablation, knapsack selection,
       longitudinal simulation.
4. [x] Build the adversarial corpus and wire attack success rate into the test suite.
5. [ ] Fit the attacker model weights on deployment data once a real scan history exists.
6. [ ] Re-examine the influence budgets after the first adversarial evaluation on live pages.
7. [ ] Add CVSS v4.0 specific handling when the corpus of v4.0-scored CVEs is large enough.
8. [ ] Run the longitudinal protocol against a real organisation's scan history to replace the
       simulated exposure reduction with an observed one.
