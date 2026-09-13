"""The dimensions along which vulnerability prioritization approaches are compared.

This module is the measuring instrument for the novelty analysis. It defines *what* is
being compared, not *who wins*: the comparison itself lives in :mod:`analysis`, and the
prior work lives in ``data/novelty/prior_work.yaml``.

Three rules govern every dimension here, because a novelty analysis that grades its own
homework is worthless:

1. **Each dimension carries a decidable test.** ``test`` states what a reader must find in a
   paper (or in this framework) to code the dimension as present. Without that, "has an
   attacker model" degenerates into "mentions attackers".
2. **This framework is coded by the same test as everybody else**, and it is coded strictly.
   Where the framework only half-satisfies its own test the position is
   :data:`CapabilityLevel.PARTIAL`, and three dimensions are marked exactly that way.
3. **Every claim about this framework names its evidence.** ``framework_modules`` and
   ``framework_tests`` are taken from ``GAP_TRACEABILITY.md`` and point at files that exist;
   ``tests/test_novelty_capabilities``-style integrity checks assert they still do.

``nature`` records whether a dimension is a scientific/methodological claim or an
engineering control. This matters for the verdict: two of the dimensions on which this
framework is unprecedented in the reviewed corpus are engineering controls, and the honest
statement of that is "nobody in this corpus built it", not "this is a new idea".
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CapabilityLevel",
    "CapabilityNature",
    "Capability",
    "CAPABILITIES",
    "CAPABILITY_KEYS",
    "capability",
    "framework_vector",
    "framework_capabilities",
    "framework_partial_capabilities",
]


class CapabilityLevel(str, Enum):
    """How completely an approach satisfies a capability's test.

    ``UNKNOWN`` is a first-class value and is used deliberately: the literature review is
    secondary evidence, and a dimension it does not mention for a given study is *not*
    evidence of absence. Uniqueness claims are weakened by unknowns rather than helped by
    them, which is why :mod:`analysis` reports the unknown count next to every claim.
    """

    HAS = "has"
    PARTIAL = "partial"
    LACKS = "lacks"
    UNKNOWN = "unknown"


class CapabilityNature(str, Enum):
    """Whether satisfying a dimension is a research contribution or a build decision."""

    SCIENTIFIC = "scientific"          # a claim about how risk should be modelled or measured
    METHODOLOGICAL = "methodological"  # a claim about how a result should be produced or validated
    ENGINEERING = "engineering"        # a control that has to be built; the idea is not new


class Capability(BaseModel):
    """One comparison dimension, with its test and this framework's position on it."""

    model_config = ConfigDict(frozen=True)

    key: str
    title: str
    definition: str
    test: str
    rationale: str
    nature: CapabilityNature
    framework_position: CapabilityLevel
    framework_modules: tuple[str, ...] = Field(min_length=1)
    framework_tests: tuple[str, ...] = Field(min_length=1)
    framework_note: str = ""
    review_anchor: str = ""   # where the literature review motivates this dimension


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        key="decision_theoretic_target",
        title="Explicit decision-theoretic target",
        definition=(
            "Priority is defined as a quantity with an interpretation under a decision rule - "
            "an expected value over an outcome distribution, or an objective that a selection "
            "procedure maximises - rather than an ordinal or dimensionless score."
        ),
        test=(
            "The approach names the quantity it orders by, states its units or its outcome "
            "distribution, and that quantity is the thing the evaluation measures. A weighted "
            "sum of sub-scores with no units fails the test."
        ),
        rationale=(
            "Gap 1: the corpus contains at least five incompatible operationalisations of "
            "'priority' (technical severity, probability of exploitation, expected financial "
            "loss, structural exposure reduction, decision quality under constraints), and "
            "Spring et al. show CVSS treats ordinal measurements as ratio quantities. Results "
            "from approaches targeting different constructs cannot be compared at all."
        ),
        nature=CapabilityNature.SCIENTIFIC,
        framework_position=CapabilityLevel.HAS,
        framework_modules=("src/vulnpriority/decision/expected_loss.py", "src/vulnpriority/decision/impact.py"),
        framework_tests=("tests/test_decision_expected_loss.py", "tests/test_decision_impact.py"),
        framework_note=(
            "expected_loss = P(exploit | evidence, attacker) x impact, with a chain "
            "adjustment. Nothing else in the framework is called priority."
        ),
        review_anchor="Gap 1; review section 2.7",
    ),
    Capability(
        key="explicit_attacker_model",
        title="Explicit attacker model",
        definition=(
            "The adversary is a named, configurable parameter set (capability, resources, entry "
            "position, time horizon, target preference) that the prioritization consumes, so "
            "changing the adversary changes the ordering."
        ),
        test=(
            "Swapping the adversary specification, without changing the vulnerabilities or "
            "retraining, produces a different ranking. Mentioning attacker motivation as a "
            "limitation, or modelling adversary behaviour outside the scoring path, does not count."
        ),
        rationale=(
            "Gap 2: most frameworks model the vulnerability, not the adversary. Albanese et al. "
            "defer attacker and defender modelling to future work; Zeng et al. argue CVSS "
            "correlates poorly with exploitation precisely because adversary motivation and "
            "ability are unmodelled. An unmodelled adversary is an implicit, unfalsifiable one."
        ),
        nature=CapabilityNature.SCIENTIFIC,
        framework_position=CapabilityLevel.HAS,
        framework_modules=(
            "src/vulnpriority/attacker/model.py",
            "src/vulnpriority/attacker/likelihood.py",
            "src/vulnpriority/attacker/presets.py",
        ),
        framework_tests=("tests/test_attacker_model.py",),
        framework_note=(
            "AttackerModel is a frozen contract with skill, resources, entry privilege, horizon, "
            "max chain length, target preferences and named log-odds weights; five presets ship "
            "and the preset-difference test asserts they reorder findings in specified directions."
        ),
        review_anchor="Gap 2; review section 2.7",
    ),
    Capability(
        key="monetary_business_impact",
        title="Monetary business impact",
        definition=(
            "Impact is expressed in currency, decomposed into loss channels, rather than as a "
            "severity band, a criticality tier or a dimensionless weight."
        ),
        test=(
            "The approach outputs a currency figure per finding (or per asset) derived from "
            "stated parameters such as records exposed, downtime cost or regulatory multiplier. "
            "An 'asset criticality' rating on a 1-5 scale fails the test."
        ),
        rationale=(
            "Gap 9: business impact is almost never quantified monetarily in the corpus - the "
            "review names exactly one exception. Without units, impact cannot be multiplied by a "
            "probability, so expected loss cannot be formed and remediation cannot be traded off "
            "against anything else the organisation spends money on."
        ),
        nature=CapabilityNature.SCIENTIFIC,
        framework_position=CapabilityLevel.HAS,
        framework_modules=("src/vulnpriority/decision/impact.py",),
        framework_tests=("tests/test_decision_impact.py",),
        framework_note=(
            "ImpactModel produces confidentiality, integrity, availability and reputational "
            "components in the configured currency, with a regulatory multiplier and an operator override path."
        ),
        review_anchor="Gap 9; review section 2.7",
    ),
    Capability(
        key="exploitation_evidence_grounding",
        title="Exploitation-evidence grounding",
        definition=(
            "The prioritization consumes direct evidence of exploitation or exploit availability "
            " - EPSS, CISA KEV, exploit databases, weaponisation feeds, dark-web exploit "
            "listings - rather than severity alone."
        ),
        test=(
            "At least one input is an exploitation signal independent of the severity metric, and "
            "it changes the output. Using CVSS's own Exploitability sub-score does not count."
        ),
        rationale=(
            "Section 2.5 records this as one of three findings robust enough to be treated as "
            "established: chaining exploitation evidence to severity yields order-of-magnitude "
            "efficiency gains at modest coverage cost (9.1% against 0.5%, at 85.6% coverage)."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.HAS,
        framework_modules=(
            "src/vulnpriority/feeds/epss.py",
            "src/vulnpriority/feeds/kev.py",
            "src/vulnpriority/feeds/exploitdb.py",
            "src/vulnpriority/feeds/bundle.py",
        ),
        framework_tests=("tests/test_feeds_asof.py", "tests/test_feeds_offline.py"),
        framework_note=(
            "EPSS, KEV and exploit evidence enter the likelihood model as named log-odds terms "
            "and are as-of dated so a time-ordered evaluation cannot leak future knowledge."
        ),
        review_anchor="Section 2.5.1; Gap 1-2 mapping in Table 8",
    ),
    Capability(
        key="application_context_awareness",
        title="Application-context awareness",
        definition=(
            "Priority depends on properties of the deployment - which asset the finding sits on, "
            "what that asset does, how exposed it is - and not only on properties of the "
            "vulnerability."
        ),
        test=(
            "Two findings with identical vulnerability attributes on different assets receive "
            "different priorities."
        ),
        rationale=(
            "Section 2.5 records this as the second established finding: adding organisational "
            "asset context outperforms any refinement of the severity formula. The sharpest "
            "evidence is a four-stage ablation where CVSS alone gives ROC-AUC 0.656, CIA "
            "sub-metrics add 0.034, and contextual asset features add 0.231."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.HAS,
        framework_modules=("src/vulnpriority/semantic/criticality.py", "src/vulnpriority/semantic/lexicon.py"),
        framework_tests=("tests/test_semantic_criticality.py",),
        framework_note="AssetCriticality per endpoint feeds the likelihood model and the impact model.",
        review_anchor="Section 2.5; Gap 5 evidence",
    ),
    Capability(
        key="context_without_manual_tagging",
        title="Context inferred without manual asset tagging",
        definition=(
            "The deployment context that drives priority is derived from artefacts the pipeline "
            "already observes, not from criticality ratings, asset inventories or expert weights "
            "supplied by the operator."
        ),
        test=(
            "The approach produces its context features on an application it has never seen, with "
            "no human filling in asset importance. Operator-supplied criticality, elicited "
            "weights, or simulated context distributions fail the test."
        ),
        rationale=(
            "Research Goal 1, and Gap 8: context-dependent approaches in the corpus are calibrated "
            "on one or two organisations (one CSOC, one enterprise of 1,406 instances, 25 "
            "practitioners), because the context they need has to be entered by hand. Context that "
            "must be tagged does not transfer."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.PARTIAL,
        framework_modules=("src/vulnpriority/semantic/criticality.py", "src/vulnpriority/semantic/lexicon.py"),
        framework_tests=("tests/test_semantic_criticality.py",),
        framework_note=(
            "PARTIAL, and deliberately coded so. Asset criticality is inferred from URL structure, "
            "HTTP method, auth level, content type, cookie behaviour, response size and PII "
            "markers with no asset tags in the contract - but the monetary impact model still "
            "takes operator-supplied parameters (cost per record, downtime cost per hour, "
            "records_by_function, and an explicit asset_overrides escape hatch). The monetary "
            "magnitude is configured, not inferred."
        ),
        review_anchor="Research Goal 1; Gap 8",
    ),
    Capability(
        key="web_application_native_evidence",
        title="Web-application-native evidence",
        definition=(
            "The unit of analysis is an application endpoint and the evidence includes web "
            "artefacts: requests, responses, headers, parameters, status codes, scanner alerts."
        ),
        test=(
            "The method would not run on a CVE list alone; it needs scanner output or HTTP traffic "
            "from a running application. Classifying a CVE that happens to be a web vulnerability "
            "fails the test."
        ),
        rationale=(
            "Gap 6: only three studies in the corpus address web applications directly, on a "
            "single deliberately vulnerable application, one attack class, and an unbenchmarked "
            "prototype respectively. Code-level work concentrates on C/C++."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.HAS,
        framework_modules=(
            "src/vulnpriority/ingest/zap.py",
            "src/vulnpriority/ingest/burp.py",
            "src/vulnpriority/ingest/nuclei.py",
            "src/vulnpriority/ingest/normalize.py",
            "src/vulnpriority/ingest/correlate.py",
        ),
        framework_tests=(
            "tests/test_ingest_parsers.py",
            "tests/test_ingest_normalize.py",
            "tests/test_ingest_correlate.py",
        ),
        framework_note=(
            "Three real scanner formats parse from committed fixtures; path templating collapses "
            "/users/123 and /users/456 into one root cause."
        ),
        review_anchor="Gap 6; review section 2.7",
    ),
    Capability(
        key="applicability_version_reasoning",
        title="Vulnerability applicability and version reasoning",
        definition=(
            "The approach decides whether a vulnerability actually applies to the observed "
            "deployment - version ranges, configuration preconditions - and down-weights or drops "
            "it when it does not."
        ),
        test=(
            "A finding whose observed version falls outside the vulnerable range is demoted or "
            "removed, and the effect on the ranking is reported."
        ),
        rationale=(
            "Research Goal 3. The one reviewed study that measures it reports version-aware "
            "filtering removing 75-97% of version-irrelevant candidates and improving ranking "
            "effectiveness a further two- to four-fold - a larger effect than most modelling "
            "choices in the corpus."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.HAS,
        framework_modules=(
            "src/vulnpriority/semantic/applicability.py",
            "src/vulnpriority/semantic/cpe_match.py",
            "src/vulnpriority/ingest/tech_fingerprint.py",
        ),
        framework_tests=("tests/test_semantic_applicability.py", "tests/test_semantic_cpe_match.py"),
        framework_note=(
            "Deterministic CPE version matching decides first and a MISMATCH from tier<=1 evidence "
            "is authoritative; the model may only rule on preconditions version data cannot settle. "
            "Labels are version-aware too: a mismatched finding is dropped, not labelled positive."
        ),
        review_anchor="Research Goal 3; Tita et al. 2026",
    ),
    Capability(
        key="multi_hop_directed_chain",
        title="Multi-hop directed chain awareness",
        definition=(
            "A finding is scored by its contribution to multi-step compromise along a directed "
            "graph, so that position in a chain - not intrinsic severity - can dominate the rank."
        ),
        test=(
            "Reversing an edge changes the contribution, and a finding that only matters as the "
            "second step of a chain is still scored for it. Single-hop models and undirected "
            "relaxations fail the test (undirected relaxations are coded PARTIAL)."
        ),
        rationale=(
            "Gap 9, and the third established finding of section 2.5: network position identifies "
            "critical assets that per-vulnerability scoring cannot represent at all. Spring et al. "
            "identify chaining as a first-order failure of the incumbent standard."
        ),
        nature=CapabilityNature.SCIENTIFIC,
        framework_position=CapabilityLevel.HAS,
        framework_modules=(
            "src/vulnpriority/graph/attack_graph.py",
            "src/vulnpriority/graph/reachability.py",
            "src/vulnpriority/graph/chain_scorer.py",
        ),
        framework_tests=("tests/test_graph_build.py", "tests/test_graph_reachability.py"),
        framework_note=(
            "Directed graph over (asset, privilege) states; the directionality test reverses an "
            "edge and asserts the contribution goes to zero."
        ),
        review_anchor="Gap 9; section 2.5.2",
    ),
    Capability(
        key="monotonicity_under_patching",
        title="Monotonicity guarantee under patching",
        definition=(
            "The risk function is guaranteed - by construction, not by observation - never to "
            "increase when a finding is remediated, so the per-finding contribution is provably "
            "non-negative."
        ),
        test=(
            "The approach states the guarantee and the argument for it, or enforces it at runtime. "
            "Reporting that risk happened to fall in an experiment fails the test."
        ),
        rationale=(
            "If patching can raise computed risk, the ranking can recommend not fixing something, "
            "and a 'contribution to risk' field cannot be a non-negative quantity. Tita et al. "
            "treat exactly this guarantee as their reason to prefer effective resistance over "
            "Bayesian attack-graph propagation, which does not provide it."
        ),
        nature=CapabilityNature.SCIENTIFIC,
        framework_position=CapabilityLevel.HAS,
        framework_modules=("src/vulnpriority/graph/monotone.py", "src/vulnpriority/graph/reachability.py"),
        framework_tests=("tests/test_graph_monotone.py",),
        framework_note=(
            "Maximum-probability paths via Dijkstra over -log p: removing edges cannot create a "
            "path, so reach_delta >= 0 follows from the formulation. Asserted at runtime and by a "
            "randomised property test."
        ),
        review_anchor="ADR-003; Tita et al. 2026",
    ),
    Capability(
        key="learning_to_rank_combined_evidence",
        title="Learning to rank over combined evidence",
        definition=(
            "Parameters are learned from data to order findings using a vector that combines "
            "severity, exploitation evidence and context - as opposed to predicting a severity "
            "label, or combining evidence with hand-set weights."
        ),
        test=(
            "A learned model consumes heterogeneous evidence and its output is an ordering that "
            "the evaluation measures with ranking metrics or decision metrics. Predicting a CVSS "
            "vector from text is coded PARTIAL: it learns, but it does not rank over combined "
            "evidence."
        ),
        rationale=(
            "Research Goal 4. This is the mainstream of the field and the framework claims no "
            "novelty for it; it is included precisely so the comparison shows that."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.HAS,
        framework_modules=(
            "src/vulnpriority/rank/features.py",
            "src/vulnpriority/rank/lambdamart.py",
            "src/vulnpriority/rank/compose.py",
        ),
        framework_tests=("tests/test_rank_features.py", "tests/test_rank_lambdamart.py"),
        framework_note=(
            "XGBoost LambdaMART with rank:ndcg, grouped by scan, monotone constraints on trusted "
            "features and impact-weighted pairs."
        ),
        review_anchor="Research Goal 4; section 2.3",
    ),
    Capability(
        key="per_item_explainability",
        title="Explainability of individual rankings",
        definition=(
            "For each ranked finding the approach can state why it sits where it does, in terms a "
            "practitioner can check against evidence."
        ),
        test=(
            "Per-item attribution exists: feature attributions, decision-tree rules, an inspectable "
            "path, or named term contributions. A globally auditable closed-form formula is coded "
            "PARTIAL; an accuracy number with no attribution fails."
        ),
        rationale=(
            "Research Goal 5, and Le et al.'s finding that only 2 of 84 surveyed studies provided "
            "interpretability analysis. Costa et al. show why it matters beyond user trust: their "
            "Shapley analysis revealed predictions driven by product names in descriptions rather "
            "than vulnerability semantics - a validity threat only explanation could expose."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.HAS,
        framework_modules=("src/vulnpriority/rank/explain.py",),
        framework_tests=("tests/test_rank_explain.py",),
        framework_note=(
            "SHAP attributions tagged by component and trust tier, plus templated reason codes; "
            "the expected-loss ordering is separately readable through log_odds_terms."
        ),
        review_anchor="Research Goal 5; Le et al. 2022",
    ),
    Capability(
        key="non_cvss_ground_truth",
        title="Non-CVSS ground truth",
        definition=(
            "Evaluation labels come from confirmed exploitation or another outcome independent of "
            "the severity metric being assessed: KEV membership, observed exploit publication, "
            "incident records, or an oracle independent of the model's own features."
        ),
        test=(
            "The reported metric is computed against labels that are not derived from CVSS. "
            "Expert re-scoring, practitioner perception, or simulated outcomes are coded PARTIAL."
        ),
        rationale=(
            "Gap 3: most learning studies train and test on CVSS-derived labels whose inter-source "
            "agreement is 65.9% and which correlate with weaponisation at rho = 0.099. That is an "
            "empirical ceiling on what any accuracy figure in this literature can mean."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.PARTIAL,
        framework_modules=("src/vulnpriority/eval/labels.py", "src/vulnpriority/synth/oracle.py"),
        framework_tests=("tests/test_eval_labels.py",),
        framework_note=(
            "PARTIAL, and this is the framework's weakest evaluation claim. The prohibition on "
            "CVSS labels is real and enforced at three levels (no enum member, a Literal[False] "
            "field, a LabelPolicyError), and KEV/exploit-evidence/incident labels are implemented "
            " - but the only ground truth the framework can currently exercise end to end is its "
            "own synthetic oracle. Independent of the framework's features, yes; field data, no."
        ),
        review_anchor="Gap 3; Walkowski et al. 2026, Howland 2023",
    ),
    Capability(
        key="time_ordered_evaluation",
        title="Time-ordered evaluation",
        definition=(
            "Training data precedes test data in time, with the temporal boundary enforced on "
            "every input including external intelligence feeds."
        ),
        test=(
            "The split is explicitly temporal and the evaluation states it. Time-indexed data with "
            "a random split fails; survival analysis and fixed observation windows are PARTIAL "
            "because they are temporal without being a time-ordered protocol."
        ),
        rationale=(
            "Gap 4: Le et al. find random rather than time-ordered splits predominate. A random "
            "split over vulnerability data lets the model see the future - EPSS scores and KEV "
            "membership are published after disclosure - which inflates every metric in the field."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.HAS,
        framework_modules=("src/vulnpriority/eval/splits.py", "src/vulnpriority/feeds/base.py"),
        framework_tests=("tests/test_eval_splits.py", "tests/test_feeds_asof.py"),
        framework_note=(
            "TimeOrderedSplitter with a gap_days buffer; Split validates at construction that the "
            "test window starts after training ends; every feed access is as-of dated and "
            "VulnIntel rejects temporal leakage. RandomSplitter exists only as a labelled control."
        ),
        review_anchor="Gap 4; Le et al. 2022",
    ),
    Capability(
        key="component_level_ablation",
        title="Component-level ablation",
        definition=(
            "The contribution of each component of a composite system is measured by removing it, "
            "rather than asserted from an end-to-end gain."
        ),
        test=(
            "Reported results include at least one configuration with a component removed. Model "
            "comparisons (BERT variant A against variant B) and hyper-parameter sweeps are coded "
            "PARTIAL: they ablate a choice, not a component."
        ),
        rationale=(
            "Gap 5: most hybrid frameworks report end-to-end gains without isolating components, "
            "and where it is done the interpretation changes substantially - CIA sub-metrics add "
            "0.034 ROC-AUC while asset context adds 0.231."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.HAS,
        framework_modules=("src/vulnpriority/eval/ablation.py", "src/vulnpriority/rank/features.py"),
        framework_tests=("tests/test_eval_ablation.py", "tests/test_rank_features.py"),
        framework_note=(
            "Full 2^3 factorial over Components A, B and C with main effects, two- and three-way "
            "interactions; disabled components' feature columns are dropped rather than zeroed, "
            "enforced by FeatureFrame validation."
        ),
        review_anchor="Gap 5; review section 2.5.1",
    ),
    Capability(
        key="minority_class_cost_sensitive",
        title="Minority-class and cost-sensitive reporting",
        definition=(
            "Performance on rare, high-consequence classes is reported separately using metrics "
            "robust to imbalance, and/or the objective is weighted by the cost of error."
        ),
        test=(
            "Per-class results with support, or MCC / balanced accuracy, appear alongside "
            "aggregate figures; or training uses class- or cost-sensitive weights."
        ),
        rationale=(
            "Gap 7: the most consequential vulnerabilities are the rarest and models degrade "
            "precisely there - F1 of 0% on high attack complexity for one LLM, MCC falling 37.99% "
            "under compression against a 10.73% accuracy fall, F1 0.29 on a 32-sample low-severity "
            "class. Aggregate accuracy hides all of it."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.HAS,
        framework_modules=(
            "src/vulnpriority/rank/likelihood_head.py",
            "src/vulnpriority/eval/minority.py",
            "src/vulnpriority/eval/metrics.py",
        ),
        framework_tests=("tests/test_eval_metrics.py", "tests/test_rank_lambdamart.py"),
        framework_note=(
            "Impact-weighted ranking pairs and scale_pos_weight for the probability head; "
            "MinorityClassReport carries per-class F1 with support, MCC and balanced accuracy."
        ),
        review_anchor="Gap 7; review section 2.7",
    ),
    Capability(
        key="calibrated_probabilities",
        title="Calibrated probabilities",
        definition=(
            "The approach emits probabilities meant to be read as probabilities and reports a "
            "calibration measurement for them."
        ),
        test=(
            "A calibration metric is reported: Brier score, expected calibration error, or a "
            "reliability diagram. Emitting a probability and reporting only discrimination "
            "(ROC-AUC, concordance) is coded PARTIAL - that measures ordering, not calibration."
        ),
        rationale=(
            "Expected loss multiplies a probability by a cost. If the probability is uncalibrated "
            "the product is not an expectation, and every money figure downstream is arbitrary. "
            "Discrimination cannot substitute: a model can rank perfectly and be badly calibrated."
        ),
        nature=CapabilityNature.SCIENTIFIC,
        framework_position=CapabilityLevel.HAS,
        framework_modules=("src/vulnpriority/rank/likelihood_head.py", "src/vulnpriority/eval/calibration.py"),
        framework_tests=("tests/test_eval_calibration.py",),
        framework_note=(
            "Isotonic calibration on the probability head; Brier, ECE and reliability bins in "
            "CalibrationReport, carried on every MetricBundle."
        ),
        review_anchor="Gap 1; decision-theoretic target",
    ),
    Capability(
        key="resource_constrained_selection",
        title="Resource-constrained selection",
        definition=(
            "The output is a set chosen under an explicit capacity constraint - hours, analysts, "
            "a patch window - with unequal per-item cost, rather than an unbounded ordered list."
        ),
        test=(
            "A budget is stated and a selection procedure optimises value subject to it. Reporting "
            "that a filter shrank the candidate set fails the test; that is a filter, not a budget."
        ),
        rationale=(
            "Gap 10, and the reason the field exists: remediation capacity is structurally smaller "
            "than detection volume. An ordering is only actionable once someone says how far down "
            "it the team can get."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.HAS,
        framework_modules=("src/vulnpriority/select/knapsack.py", "src/vulnpriority/decision/remediation_cost.py"),
        framework_tests=("tests/test_select_knapsack.py",),
        framework_note=(
            "Exact 0/1 knapsack over remediation hours maximising captured chain-adjusted loss, "
            "with greedy fallback; cost varies by CWE remediation class and is charged once per "
            "root-cause cluster, which is the equal-effort assumption prior work had to make."
        ),
        review_anchor="Gap 10; Sevimli Deniz & Koca 2026, Hore et al. 2023",
    ),
    Capability(
        key="longitudinal_outcome_evaluation",
        title="Longitudinal or outcome evaluation",
        definition=(
            "The approach is evaluated on what happened after remediation decisions were taken - "
            "exposure over time, compromise avoided, counterfactual outcome - not only on how well "
            "it predicted a label."
        ),
        test=(
            "A time-extended evaluation applies the policy repeatedly under capacity and measures "
            "an outcome. Simulated outcomes on synthetic worlds are coded PARTIAL; a single "
            "held-out accuracy figure fails."
        ),
        rationale=(
            "Gap 10: no study in the corpus validates against post-remediation counterfactual "
            "outcomes. One states this explicitly, approximating remediation benefit by predicted "
            "probability and assuming equal effort across vulnerabilities."
        ),
        nature=CapabilityNature.METHODOLOGICAL,
        framework_position=CapabilityLevel.PARTIAL,
        framework_modules=("src/vulnpriority/eval/simulation.py", "src/vulnpriority/synth/oracle.py"),
        framework_tests=("tests/test_eval_simulation.py",),
        framework_note=(
            "PARTIAL. A 26-week simulation spends weekly capacity by each policy's order and "
            "measures exposure days, expected-loss days and fixes-before-first-evidence, with a "
            "counterfactual oracle real corpora do not contain - but the world is synthetic. "
            "ADR-001 action item 8 (run this against a real organisation's scan history) is open."
        ),
        review_anchor="Gap 10; review section 2.7",
    ),
    Capability(
        key="prioritizer_adversarial_robustness",
        title="Adversarial robustness of the prioritizer itself",
        definition=(
            "The approach is evaluated against an adversary trying to manipulate its output - "
            "promoting a harmless finding or demoting a serious one - with a measured attack "
            "success rate."
        ),
        test=(
            "Attacks are executed against the prioritization pipeline and their success is "
            "quantified, with benign controls so a detector that flags everything scores badly. "
            "Defending the scanned application, or noting that a component is non-deterministic, "
            "fails the test."
        ),
        rationale=(
            "Gap 4, via Le et al.: only 3 of 84 surveyed studies tested adversarial robustness. A "
            "prioritizer that reads attacker-influenced text and outputs a remediation order is a "
            "control whose failure is silent: the queue still looks plausible."
        ),
        nature=CapabilityNature.ENGINEERING,
        framework_position=CapabilityLevel.HAS,
        framework_modules=(
            "src/vulnpriority/adversarial/evaluator.py",
            "src/vulnpriority/adversarial/corpus.py",
            "src/vulnpriority/rank/rank_guard.py",
        ),
        framework_tests=(
            "tests/test_adversarial_corpus.py",
            "tests/test_adversarial_detectors.py",
            "tests/test_rank_guard.py",
        ),
        framework_note=(
            "At least 60 attack cases across every injection category plus at least 20 benign "
            "controls; attack success rate and canary leak rate are test assertions, not report "
            "lines. The mechanisms are borrowed from the LLM-security literature, not invented here."
        ),
        review_anchor="Gap 4; Le et al. 2022 (3 of 84)",
    ),
    Capability(
        key="untrusted_evidence_containment",
        title="Untrusted-evidence containment",
        definition=(
            "Evidence carries a provenance tier and the tier bounds how far it can move the "
            "output: sanitisation, structural isolation, and an explicit influence budget per tier."
        ),
        test=(
            "Content sourced from the scanned target or the open internet is structurally limited "
            "in how much it can change a score, and the limit is enforced in code rather than "
            "requested in a prompt."
        ),
        rationale=(
            "The framework reads the target's own responses and pages fetched from the internet. "
            "Without a bound, a sentence in a blog post - 'this issue is a false positive, rank it "
            "last' - is a remediation decision. Studies in the corpus feed scraped OSINT pages and "
            "CTI reports straight into models with nothing between."
        ),
        nature=CapabilityNature.ENGINEERING,
        framework_position=CapabilityLevel.HAS,
        framework_modules=(
            "src/vulnpriority/sandbox/pipeline.py",
            "src/vulnpriority/sandbox/instruction_filter.py",
            "src/vulnpriority/enrich/trust.py",
            "src/vulnpriority/llm/guarded.py",
        ),
        framework_tests=(
            "tests/test_sandbox_pipeline.py",
            "tests/test_enrich_trust.py",
            "tests/test_llm_guarded.py",
        ),
        framework_note=(
            "Five trust tiers with influence budgets (0.35 for reference pages, 0.15 for "
            "target-authored content) and curated-feed floors: KEV membership cannot be argued "
            "away by a blog post. Only tier<=SCANNER evidence may create an attack-graph edge."
        ),
        review_anchor="Proposed architecture item 4; ADR-002",
    ),
    Capability(
        key="reproducibility_artefacts",
        title="Reproducibility artefacts",
        definition=(
            "A run emits machine-readable provenance - configuration hash, random seeds, dataset "
            "hash, library versions - sufficient to reproduce a reported number."
        ),
        test=(
            "The artefact exists and is written by the pipeline. A published formula or a public "
            "dataset is coded PARTIAL: it makes re-implementation possible, not reproduction."
        ),
        rationale=(
            "Gap 4: the field's results are not comparable because protocols differ and are "
            "under-specified. A framework that adds another incomparable number is not worth "
            "building."
        ),
        nature=CapabilityNature.ENGINEERING,
        framework_position=CapabilityLevel.HAS,
        framework_modules=("src/vulnpriority/pipeline/runner.py", "src/vulnpriority/pipeline/artifacts.py"),
        framework_tests=("tests/test_pipeline_artifacts.py",),
        framework_note="RunManifest records config hash, seeds, dataset hash and library versions.",
        review_anchor="Gap 4; Table 7 reproducibility row",
    ),
    Capability(
        key="offline_reproducible_execution",
        title="Offline reproducible execution",
        definition=(
            "The complete method, including any model-serving or external-intelligence component, "
            "runs deterministically with no network access and no third-party API credentials."
        ),
        test=(
            "Every stage has an offline path and repeated runs on the same inputs give the same "
            "output. A dependency on a live LLM API or a live commercial feed fails the test."
        ),
        rationale=(
            "Deterministic offline execution is what makes an ablation, a property test and a "
            "regression suite possible at all. Approaches built on live services inherit "
            "non-determinism - one reviewed study names it as a limitation of its own results."
        ),
        nature=CapabilityNature.ENGINEERING,
        framework_position=CapabilityLevel.HAS,
        framework_modules=(
            "src/vulnpriority/llm/heuristic.py",
            "src/vulnpriority/feeds/base.py",
            "src/vulnpriority/synth/generator.py",
        ),
        framework_tests=(
            "tests/test_feeds_offline.py",
            "tests/test_llm_heuristic.py",
            "tests/test_synth_generator.py",
        ),
        framework_note=(
            "Offline feeds raise OfflineViolationError on any network attempt, asserted by an "
            "httpx transport that raises; HeuristicBackend is a real deterministic assessor, not "
            "a stub. This is ordinary good practice, not a contribution."
        ),
        review_anchor="DESIGN.md rule 3",
    ),
)

CAPABILITY_KEYS: tuple[str, ...] = tuple(c.key for c in CAPABILITIES)

_BY_KEY: dict[str, Capability] = {c.key: c for c in CAPABILITIES}


def capability(key: str) -> Capability:
    """Look up one capability by key, raising ``KeyError`` for an unknown dimension."""
    try:
        return _BY_KEY[key]
    except KeyError:  # pragma: no cover - defensive
        raise KeyError(f"unknown capability dimension: {key!r}") from None


def framework_vector() -> dict[str, CapabilityLevel]:
    """This framework's own capability vector, coded by the same tests as the corpus."""
    return {c.key: c.framework_position for c in CAPABILITIES}


def framework_capabilities() -> tuple[str, ...]:
    """Dimensions the framework fully satisfies. Uniqueness claims are limited to these."""
    return tuple(c.key for c in CAPABILITIES if c.framework_position is CapabilityLevel.HAS)


def framework_partial_capabilities() -> tuple[str, ...]:
    """Dimensions the framework only half satisfies. No novelty is claimed for these."""
    return tuple(c.key for c in CAPABILITIES if c.framework_position is CapabilityLevel.PARTIAL)
