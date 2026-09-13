# Gap traceability

Every research gap, research goal and proposed-architecture item from the literature review,
mapped to the module that mitigates it and the test that proves it. Rows are verified against
the code: if a row names a function or a test, it exists.

Run everything referenced here with `python -m pytest -q`.

---

## Research gaps (review section 2.7)

### Gap 1 - No validated construct definition of priority

*Five incompatible operationalisations of "priority" exist in the corpus, and CVSS treats
ordinal measurements as ratio quantities.*

| | |
|---|---|
| **Mitigation** | Priority is one thing: expected loss, in money, over an explicit attacker's horizon. `expected_loss = P(exploit \| evidence, attacker) × impact`, with a chain adjustment that adds reachable-compromise contribution. Nothing else in the framework is called priority. |
| **Modules** | `decision/expected_loss.py` (`expected_loss`, `chain_adjusted_loss`), `decision/impact.py` (`estimate_impact`), `attacker/model.py` (`p_exploit`) |
| **Contract** | `EnrichedFinding.expected_loss`, `RankedFinding.chain_adjusted_loss`, `BusinessImpact` |
| **Tests** | `tests/test_decision_expected_loss.py`, `tests/test_decision_impact.py` |
| **Evidence it holds** | Impact is monetary and decomposed into confidentiality, integrity, availability and reputational terms; the probability is a logistic model whose every term is recorded in `ExploitLikelihood.log_odds_terms`, so the construct is inspectable rather than asserted. |

### Gap 2 - Absence of an explicit attacker model

*Most frameworks model the vulnerability, not the adversary; two studies explicitly defer
attacker modelling to future work.*

| | |
|---|---|
| **Mitigation** | `AttackerModel` is a first-class configuration object with skill, resources, entry privilege, horizon, maximum chain length, target preferences and named log-odds weights. Five presets ship. Changing the attacker re-prioritises the queue without retraining. |
| **Modules** | `attacker/model.py`, `attacker/likelihood.py`, `attacker/presets.py`, `configs/attacker_models/*.yaml` |
| **Contract** | `AttackerModel`, `ExploitLikelihood` |
| **Tests** | `tests/test_attacker_model.py` |
| **Evidence it holds** | The preset-difference test asserts the presets behave differently in the specified directions: the insider is insensitive to internet exposure, the advanced persistent adversary is less sensitive to public exploit maturity, the opportunistic adversary is penalised hardest by required privileges. |

### Gap 3 - Circular validation against unreliable labels

*Most learning studies train and test on CVSS-derived labels, which agree across sources only
65.9% of the time and correlate with weaponisation at rho = 0.099.*

| | |
|---|---|
| **Mitigation** | CVSS is never a label. Ground truth comes only from KEV membership, exploit evidence at maturity ≥ FUNCTIONAL, recorded incidents, or the synthetic oracle. Labels are version-aware (a version mismatch is dropped rather than labelled positive) and source-aware (inter-source agreement is recorded and becomes a feature). |
| **Modules** | `eval/labels.py` (`LabelBuilder`), `feeds/cvss_policy.py` (`select_cvss`) |
| **Contract** | `GroundTruthLabel.cvss_used_as_label: Literal[False]`, `LabelPolicy`, `LabelSource` (no CVSS member), `LabelPolicyError` |
| **Tests** | `tests/test_eval_labels.py`, `tests/test_feeds_cvss_policy.py` |
| **Evidence it holds** | The prohibition is enforced at three levels: the enum has no CVSS member, the label model pins a `Literal[False]` field, and `LabelBuilder` raises `LabelPolicyError`. The CVSS policy test covers version-2-only records, mixed-version records and NVD-versus-CNA disagreement. |

### Gap 4 - No standardised benchmark or evaluation protocol

*Studies report incomparable metrics on incomparable data; only 3 of 84 surveyed studies tested
adversarial robustness, and random rather than time-ordered splits predominate.*

| | |
|---|---|
| **Mitigation** | One frozen protocol: time-ordered splits with a gap, confirmed-exploitation labels, a fixed metric set, seven baselines on identical data and preprocessing, paired bootstrap intervals, and a run manifest recording the config hash, seeds, dataset hash and library versions. Random splitting exists only as a labelled control so the report can quantify what it overstates. |
| **Modules** | `eval/splits.py`, `eval/metrics.py`, `eval/benchmark.py`, `eval/bootstrap.py`, `eval/report.py`, `pipeline/runner.py` (manifest) |
| **Contract** | `Split`, `MetricBundle`, `RunManifest`, `EvaluationConfig` |
| **Tests** | `tests/test_eval_splits.py`, `tests/test_eval_metrics.py`, `tests/test_eval_report.py`, `tests/test_pipeline_artifacts.py` |
| **Evidence it holds** | `Split` validates at construction that train and test do not overlap and that the test window starts after the training window ends. Metrics are checked against hand-computed values and against scikit-learn. Adversarial robustness is a test, not a discussion: see Architecture item 4. |

### Gap 5 - Insufficient component-level ablation in hybrid systems

*Hybrid frameworks report end-to-end gains without isolating each component's contribution.*

| | |
|---|---|
| **Mitigation** | Full 2³ factorial ablation over Components A, B and C across seeds, reporting main effects, two-way and three-way interactions with paired bootstrap intervals. Disabled components' features are dropped from the matrix, not zeroed, so a disabled component cannot leak through a constant. |
| **Modules** | `eval/ablation.py` (`FullFactorialAblation`), `rank/features.py` (`FeatureBuilder.build` honouring `feature_names_for`) |
| **Contract** | `ComponentFlags.all_cells()`, `AblationTable`, `feature_names_for` |
| **Tests** | `tests/test_eval_ablation.py`, `tests/test_rank_features.py` |
| **Evidence it holds** | `FeatureFrame` validates that its columns exactly match `feature_names_for(flags)`, so an ablation cell physically cannot carry a disabled component's signal. The ablation arithmetic is checked against a synthetic table with a known answer. |

### Gap 6 - Web application vulnerabilities are under-represented as a population

*Only three studies in the corpus address web applications directly, on a single deliberately
vulnerable application, one attack class, and an unbenchmarked prototype respectively.*

| | |
|---|---|
| **Mitigation** | The framework is web-application-native throughout: it ingests OWASP ZAP, Burp Suite and Nuclei output, its unit of analysis is the endpoint, its criticality inference reads URL structure, HTTP method, authentication level and response characteristics, and its attack graph is built from observed endpoints and links. |
| **Modules** | `ingest/zap.py`, `ingest/burp.py`, `ingest/nuclei.py`, `ingest/normalize.py`, `ingest/correlate.py`, `semantic/criticality.py`, `semantic/lexicon.py` |
| **Contract** | `Endpoint`, `Finding`, `Scan`, `EndpointFunction` |
| **Tests** | `tests/test_ingest_parsers.py`, `tests/test_ingest_normalize.py`, `tests/test_ingest_correlate.py`, `tests/test_semantic_criticality.py` |
| **Evidence it holds** | Three real scanner formats parse end to end from committed fixtures. Path templating collapses `/users/123` and `/users/456` into one root cause, which is the specific overhead problem web scanning creates. |

### Gap 7 - Systematic failure on minority and high-severity classes

*The most consequential vulnerabilities are the rarest, and models degrade precisely there: an
LLM scored F1 = 0% on high attack complexity, and compression cost 38% of MCC.*

| | |
|---|---|
| **Mitigation** | Cost-sensitive training and minority-aware reporting. Ranking sample weights scale with monetary impact; the probability head computes `scale_pos_weight` from the label balance; evaluation reports MCC, per-class F1 with support and balanced accuracy alongside aggregate figures. |
| **Modules** | `rank/features.py` (impact-weighted pairs), `rank/likelihood_head.py`, `eval/minority.py`, `eval/metrics.py` (`mcc`, `f1_minority`, `balanced_accuracy`) |
| **Contract** | `MinorityClassReport`, `RankingConfig.impact_weighted_pairs`, `RankingConfig.head_scale_pos_weight` |
| **Tests** | `tests/test_rank_lambdamart.py`, `tests/test_eval_metrics.py` |
| **Evidence it holds** | MCC is verified against scikit-learn, and the report carries per-class support so a class with three examples cannot hide inside an aggregate accuracy figure. |

### Gap 8 - Evidence is organisationally narrow and non-English threat sources unexplored

*Almost every deployment evaluation in the corpus rests on one or two organisations, and
Russian and Middle Eastern sources are identified as unexamined.*

| | |
|---|---|
| **Mitigation** | Transfer is measured, not assumed: applications carry a sector, the synthetic world spans four sectors, and a leave-one-application-out splitter reports cross-application generalisation. The token lexicon and the injection pattern library are multilingual (Russian, Chinese, Spanish, German, French, Arabic), and reference fixtures include non-English pages. |
| **Modules** | `eval/splits.py` (`LeaveOneAppOutSplitter`), `semantic/lexicon.py`, `configs/sandbox/instruction_patterns.yaml`, `synth/generator.py` (sectors) |
| **Contract** | `Scan.sector`, `Split.held_out_app_id`, `SyntheticConfig.sectors`, `SyntheticConfig.non_english_fraction` |
| **Tests** | `tests/test_eval_splits.py`, `tests/test_semantic_criticality.py`, `tests/test_sandbox_filter.py` |
| **Evidence it holds** | Criticality classification is tested on non-English paths, and the injection corpus carries multilingual payloads that must be detected. |

### Gap 9 - Multi-hop chaining and business impact remain outside the models

*Chaining is a first-order failure of the incumbent standard; prior work models single-hop
exploitation or discards directionality, and business impact is almost never quantified.*

| | |
|---|---|
| **Mitigation** | A directed, multi-hop attack graph over (asset, privilege) states. Each finding is scored by `reach_delta`, the reduction in value-weighted reachability when it is patched, computed over maximum-probability paths so the contribution is provably non-negative. Impact is monetary throughout. |
| **Modules** | `graph/attack_graph.py`, `graph/reachability.py`, `graph/chain_scorer.py`, `graph/monotone.py`, `graph/privilege_map.py`, `decision/impact.py` |
| **Contract** | `ChainScore`, `AttackPath`, `AttackGraphSummary`, `GraphEdge`, `MonotonicityViolationError` |
| **Tests** | `tests/test_graph_build.py`, `tests/test_graph_reachability.py`, `tests/test_graph_monotone.py` |
| **Evidence it holds** | The chokepoint test reproduces the result the review cites: a low-severity finding that is a prerequisite for everything else outranks a high-severity leaf. A property test patches random subsets and asserts risk never rises. Directionality is tested by reversing an edge and asserting the contribution goes to zero. |

### Gap 10 - Evaluation stops at prediction rather than remediation outcome

*No study validates against post-remediation counterfactual outcomes; one explicitly
approximates remediation benefit by predicted probability and assumes equal effort.*

| | |
|---|---|
| **Mitigation** | Two decision-level evaluations. A 0/1 knapsack selects under a remediation-hour budget with unequal per-finding cost, charged once per root-cause cluster. A 26-week longitudinal simulation spends a weekly capacity following each policy's order and measures exposure days, expected-loss days and how many exploited findings were fixed before their first evidence date. |
| **Modules** | `select/knapsack.py`, `decision/remediation_cost.py`, `eval/simulation.py`, `synth/oracle.py` (counterfactuals) |
| **Contract** | `SelectionResult`, `SimulationResult`, `RemediationCost` |
| **Tests** | `tests/test_select_knapsack.py`, `tests/test_eval_simulation.py` |
| **Evidence it holds** | Remediation cost is genuinely unequal: it varies by CWE remediation class and grows with cluster size, which is the assumption the cited work had to make and could not test. The oracle supplies the counterfactual "would have been exploited if not remediated by date D" that real corpora do not contain. |

---

## Research goals (review section 5)

| Goal | Mitigation | Modules | Tests |
|---|---|---|---|
| **1. Semantic asset criticality** without manual asset tagging | Criticality inferred from URL tokens, HTTP method, authentication level, content type, cookie behaviour, response size, parameter names and personally-identifiable-information markers. No asset tags anywhere in the contract. | `semantic/criticality.py`, `semantic/lexicon.py` | `tests/test_semantic_criticality.py` |
| **2. Automated exploitability analysis** | Feasibility, maturity, attack complexity, privileges, user interaction and CIA impact extracted from references and exploit records through a sandboxed backend, fused with a deterministic baseline. Exploit material the feeds do not link is found by search and read by the model, and what it finds enters the ranking as features rather than only the report. | `semantic/exploitability.py`, `intel/agent.py`, `intel/anthropic_search.py`, `llm/guarded.py` | `tests/test_semantic_exploitability.py`, `tests/test_intel_agent.py` |
| **3. Vulnerability applicability** | Version-range matching against observed components decides first; the model rules only on preconditions version data cannot settle. | `semantic/applicability.py`, `semantic/cpe_match.py` | `tests/test_semantic_applicability.py`, `tests/test_semantic_cpe_match.py` |
| **4. Vulnerability ranking** combining CVSS, EPSS, KEV and model-derived features with XGBoost LambdaMART | 53 named features across BASE, A, B and C groups; `rank:ndcg` objective, groups by scan, monotone constraints on trusted features, impact-weighted pairs. | `rank/features.py`, `rank/lambdamart.py`, `rank/compose.py` | `tests/test_rank_features.py`, `tests/test_rank_lambdamart.py` |
| **5. Explainable prioritization** evaluated against CVSS and benchmark rankings | SHAP attributions tagged by component and trust tier, templated reason codes, and seven baselines evaluated on identical data. | `rank/explain.py`, `rank/baselines.py`, `eval/benchmark.py` | `tests/test_rank_explain.py`, `tests/test_rank_baselines.py` |

---

## Proposed architecture (review section 4)

| Item | Mitigation | Modules | Tests |
|---|---|---|---|
| **1. AI-based exploitability analysis** | Two kinds of retrieval: the references a CVE record lists, and a search for exploit material it does not. Both are analysed for feasibility, maturity, preconditions and impact, and both are untrusted input to the same sandbox. | `feeds/references.py`, `feeds/bundle.py`, `intel/queries.py`, `intel/agent.py`, `semantic/exploitability.py` | `tests/test_feeds_offline.py`, `tests/test_intel_agent.py`, `tests/test_semantic_exploitability.py` |
| **2. Applicability assessment** | Observed technology and versions compared against vulnerability requirements; result joins CVSS, EPSS and KEV in the feature set. | `semantic/applicability.py`, `semantic/cpe_match.py`, `ingest/tech_fingerprint.py` | `tests/test_semantic_applicability.py`, `tests/test_ingest_parsers.py` |
| **3. Learning-based ranking** | XGBoost LambdaMART produces a per-scan remediation queue and SHAP identifies the factors behind each position. The model never sets the priority: what it reads becomes features, and the ranker learns their weight, which is the separation the review's architecture specifies. | `rank/features.py`, `rank/lambdamart.py`, `rank/explain.py`, `rank/compose.py` | `tests/test_rank_features.py`, `tests/test_rank_lambdamart.py`, `tests/test_rank_explain.py` |
| **4. Adversarial robustness** | Target-derived content is untrusted by construction; a seven-layer sandbox plus influence budgets plus a rank guard, measured against a versioned corpus of injection and rank-manipulation attacks with benign controls. | `sandbox/*`, `llm/guarded.py`, `enrich/trust.py`, `rank/rank_guard.py`, `adversarial/*` | `tests/test_sandbox_*.py`, `tests/test_llm_guarded.py`, `tests/test_enrich_trust.py`, `tests/test_rank_guard.py`, `tests/test_adversarial_*.py` |

---

## Components (review section 2.8)

| Component | Scope | Modules | Ablation flag |
|---|---|---|---|
| **A** - agentic semantic assessment on web application evidence | Asset criticality, exploitability, applicability, from scanner findings, CVE descriptions and request/response artefacts | `semantic/`, `llm/`, `sandbox/` | `ComponentFlags.a` |
| **B** - contextual and threat-intelligence enrichment | EPSS, KEV, exploit availability fused with asset criticality, exposure and monetary business impact under an explicit attacker | `attacker/`, `decision/`, `enrich/`, `feeds/` | `ComponentFlags.b` |
| **C** - topology- and chain-aware ranking with resource-constrained selection | Contribution to reachable compromise paths, plus knapsack selection under a remediation budget | `graph/`, `select/` | `ComponentFlags.c` |

Each flag drops its component's feature columns from the matrix, so the 2³ factorial ablation
measures each component's contribution rather than assuming it.
