"""Refusing to grade a ranker against labels it can read off its own features (Gap 3).

``KEV`` membership and exploit evidence are accepted ground truth *and* feature columns
(``b_kev``, ``b_exploit_count``). A positive justified by nothing else is not a prediction
the model made, it is a join it performed, and grading against it measures the join. On the
synthetic corpus that was 51% of the positives, and the reported NDCG@10 fell from 0.817 to
0.604 the moment they stopped counting.

Three layers are pinned here: :class:`LabelSet` knows which of its own positives are
circular, :func:`scoring_labels` applies the policy and refuses to invent a metric when
nothing independent is left, and :func:`evaluate_stage` grades on what comes back.

The end-to-end case uses the most damning ranker available, on purpose. ``kev_first`` sorts
by exactly the column in question, so the distance between its two scores is the whole of
its apparent skill.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from vulnpriority.core.config import PipelineConfig
from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    AttackComplexity,
    CvssVersion,
    EndpointFunction,
    ExploitMaturity,
    HttpMethod,
    LabelSource,
    MetricName,
    PrivilegeLevel,
    Provenance,
    RankerName,
    ScannerSeverity,
    ScoreSource,
    SplitKind,
    UserInteraction,
    VersionMatch,
)
from vulnpriority.core.errors import ConfigError
from vulnpriority.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    BusinessImpact,
    CvssRecord,
    Endpoint,
    EnrichedFinding,
    ExploitLikelihood,
    ExploitabilityAssessment,
    Finding,
    GroundTruthLabel,
    KevRecord,
    LabelSet,
    RemediationCost,
    Split,
    UntrustedText,
    VulnIntel,
)
from vulnpriority.pipeline.stages import evaluate_stage, scoring_labels

CUTOFF = date(2024, 6, 1)
AS_OF = date(2024, 6, 1)
OBSERVED = datetime(2024, 3, 1, 9, 0, 0)


def label(finding_id: str, *sources: LabelSource, grade: int = 4) -> GroundTruthLabel:
    return GroundTruthLabel(
        finding_id=finding_id,
        exploited=grade > 0,
        relevance_grade=grade,
        sources=tuple(sources),
    )


def label_set(*labels: GroundTruthLabel) -> LabelSet:
    return LabelSet(observation_cutoff=CUTOFF, labels=labels)


def config_with(exclude: bool = True) -> PipelineConfig:
    base = PipelineConfig()
    evaluation = base.evaluation.model_copy(update={"exclude_circular_labels": exclude})
    return base.model_copy(update={"evaluation": evaluation})


# ---------------------------------------------------------------------------
# Which positives are a lookup
# ---------------------------------------------------------------------------


def test_a_positive_resting_only_on_kev_is_circular() -> None:
    labels = label_set(label("f1", LabelSource.KEV))
    assert [item.finding_id for item in labels.circular_labels()] == ["f1"]


def test_a_positive_resting_only_on_exploit_evidence_is_circular() -> None:
    labels = label_set(label("f1", LabelSource.EXPLOIT_EVIDENCE))
    assert [item.finding_id for item in labels.circular_labels()] == ["f1"]


def test_both_feature_visible_sources_together_are_still_circular() -> None:
    """Two views of the same two columns do not corroborate each other into evidence."""
    labels = label_set(label("f1", LabelSource.KEV, LabelSource.EXPLOIT_EVIDENCE))
    assert len(labels.circular_labels()) == 1


def test_one_source_outside_the_feature_set_makes_the_label_independent() -> None:
    """The question is whether *every* source is visible, not whether any is.

    A finding that is in KEV and that the oracle also recorded being exploited is evidence:
    the oracle's vote sits in no column, so the label is not a restatement of the input.
    Discarding it because KEV also voted would throw away the corroborated cases, which are
    the most trustworthy labels in the set.
    """
    mixed = label_set(label("f1", LabelSource.KEV, LabelSource.SYNTHETIC_ORACLE))
    assert mixed.circular_labels() == ()
    assert mixed.for_evaluation() is mixed

    incident = label_set(label("f2", LabelSource.EXPLOIT_EVIDENCE, LabelSource.INCIDENT))
    assert incident.circular_labels() == ()


def test_a_negative_is_never_circular() -> None:
    """Only positives can leak. A negative sourced from KEV says the CVE was *not* listed."""
    labels = label_set(label("f1", LabelSource.KEV, grade=0))
    assert labels.circular_labels() == ()


def test_a_positive_with_no_recorded_source_is_left_alone() -> None:
    """An unsourced label is a data problem for the label builder, not leakage.

    Demoting it here would silently delete ground truth whose provenance simply was not
    written down, which is a different failure and wants a different fix.
    """
    labels = label_set(label("f1"))
    assert labels.circular_labels() == ()


def test_circular_positives_are_demoted_and_not_dropped() -> None:
    """The row stays in the query group; it just stops earning credit.

    Dropping it would shrink the group the ranker is scored over, which makes the ranking
    task easier by removing a document the model still has to place somewhere. Demotion
    keeps the task identical and changes only the payout.
    """
    labels = label_set(
        label("f1", LabelSource.KEV),
        label("f2", LabelSource.SYNTHETIC_ORACLE),
        label("f3", LabelSource.KEV, grade=0),
    )
    scoring = labels.for_evaluation()

    assert [item.finding_id for item in scoring.labels] == ["f1", "f2", "f3"]

    demoted = scoring.by_id("f1")
    assert demoted is not None
    assert demoted.relevance_grade == 0
    assert demoted.exploited is False
    # Provenance survives the demotion, so a reader can still see why it was not scored.
    assert demoted.sources == (LabelSource.KEV,)

    kept = scoring.by_id("f2")
    assert kept is not None and kept.relevance_grade == 4 and kept.exploited is True


def test_a_clean_label_set_is_returned_unchanged() -> None:
    labels = label_set(label("f1", LabelSource.SYNTHETIC_ORACLE))
    assert labels.for_evaluation() is labels
    assert labels.independent_positive_count() == 1


def test_independent_positive_count_ignores_circular_and_negative_labels() -> None:
    labels = label_set(
        label("f1", LabelSource.KEV),
        label("f2", LabelSource.EXPLOIT_EVIDENCE),
        label("f3", LabelSource.INCIDENT),
        label("f4", LabelSource.SYNTHETIC_ORACLE, grade=0),
    )
    assert labels.independent_positive_count() == 1


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def test_the_policy_is_on_by_default() -> None:
    assert PipelineConfig().evaluation.exclude_circular_labels is True


def test_turning_the_policy_off_hands_back_the_labels_untouched() -> None:
    """The escape hatch exists to reproduce a number that was computed the old way."""
    labels = label_set(label("f1", LabelSource.KEV), label("f2", LabelSource.SYNTHETIC_ORACLE))
    assert scoring_labels(config_with(exclude=False), labels) is labels


def test_the_policy_demotes_when_something_independent_survives() -> None:
    labels = label_set(label("f1", LabelSource.KEV), label("f2", LabelSource.SYNTHETIC_ORACLE))
    assert scoring_labels(config_with(), labels).positives() == {"f2"}


def test_a_label_set_made_entirely_of_lookups_is_refused() -> None:
    """No metric at all beats a metric that means something other than it appears to.

    A deployment whose only ground truth is KEV cannot honestly grade a ranker that reads
    KEV, and the useful output in that situation is the sentence saying so.
    """
    labels = label_set(
        label("f1", LabelSource.KEV),
        label("f2", LabelSource.EXPLOIT_EVIDENCE),
        label("f3", LabelSource.KEV, grade=0),
    )
    with pytest.raises(ConfigError) as excinfo:
        scoring_labels(config_with(), labels)

    message = str(excinfo.value)
    assert "2 positive label(s)" in message
    assert "exclude_circular_labels" in message


def test_a_label_set_with_no_positives_at_all_is_not_refused() -> None:
    """Zero positives is an ordinary empty-fold condition, and the splitter reports it.

    Raising here would replace that specific complaint with a leakage complaint about a
    corpus that has no leakage.
    """
    labels = label_set(label("f1", LabelSource.KEV, grade=0))
    assert scoring_labels(config_with(), labels).positives() == set()


def test_the_demotion_is_logged_with_counts(caplog: pytest.LogCaptureFixture) -> None:
    """Silence would make a metric drop look like a regression in the model."""
    labels = label_set(
        label("f1", LabelSource.KEV),
        label("f2", LabelSource.KEV),
        label("f3", LabelSource.SYNTHETIC_ORACLE),
    )
    with caplog.at_level("INFO", logger="vulnpriority.pipeline.stages"):
        scoring_labels(config_with(), labels)
    assert "2 of 3 positive label(s)" in caplog.text


# ---------------------------------------------------------------------------
# End to end through evaluate_stage
# ---------------------------------------------------------------------------


def make_endpoint(endpoint_id: str) -> Endpoint:
    return Endpoint(
        endpoint_id=endpoint_id,
        app_id="app1",
        host="shop.example.com",
        url=f"https://shop.example.com/{endpoint_id}",
        path=f"/{endpoint_id}",
        method=HttpMethod.POST,
        auth_required=PrivilegeLevel.USER,
        parameters=("q",),
    )


def make_intel(cve_id: str, *, in_kev: bool, base_score: float) -> VulnIntel:
    return VulnIntel(
        cve_id=cve_id,
        as_of=AS_OF,
        cvss=(
            CvssRecord(
                version=CvssVersion.V31,
                source=ScoreSource.NVD,
                base_score=base_score,
                vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            ),
        ),
        kev=KevRecord(
            cve_id=cve_id,
            in_kev=in_kev,
            date_added=date(2024, 1, 15) if in_kev else None,
            as_of=AS_OF,
        ),
    )


def make_enriched(
    finding_id: str, scan_id: str, index: int, *, in_kev: bool
) -> EnrichedFinding:
    # A fixed id per row: ``hash`` is salted per interpreter, and a corpus that differs
    # between runs would make any ordering assertion below flaky rather than wrong.
    cve_id = f"CVE-2024-{1000 + index:04d}"
    finding = Finding(
        finding_id=finding_id,
        scan_id=scan_id,
        app_id="app1",
        endpoint_id=f"ep_{finding_id}",
        name="SQL Injection",
        cwe_id=89,
        cve_ids=(cve_id,),
        scanner="zap",
        scanner_severity=ScannerSeverity.HIGH,
        scanner_confidence=0.9,
        description=UntrustedText(text="injection", provenance=Provenance.SCANNER_OUTPUT),
        observed_at=OBSERVED,
        dedup_key=finding_id,
    )
    endpoint = make_endpoint(f"ep_{finding_id}")
    return EnrichedFinding(
        finding=finding,
        endpoint=endpoint,
        # CVSS descends across the row set so that ``kev_first``, which breaks its KEV
        # bands by CVSS, has a total order rather than a pile of ties.
        intel=(make_intel(cve_id, in_kev=in_kev, base_score=9.8 - index * 0.5),),
        asset=AssetCriticality(
            endpoint_id=endpoint.endpoint_id,
            function=EndpointFunction.AUTH,
            criticality=0.8,
            data_sensitivity=0.7,
            exposure=1.0,
            confidence=0.6,
        ),
        exploitability=ExploitabilityAssessment(
            finding_id=finding_id,
            exploit_feasibility=0.7,
            exploit_maturity=ExploitMaturity.POC,
            attack_complexity=AttackComplexity.LOW,
            privileges_required=PrivilegeLevel.NONE,
            user_interaction=UserInteraction.NONE,
            impact_c=0.9,
            impact_i=0.6,
            impact_a=0.3,
            privilege_gained=PrivilegeLevel.ADMIN,
            confidence=0.8,
        ),
        applicability=ApplicabilityAssessment(
            finding_id=finding_id,
            verdict=ApplicabilityVerdict.APPLICABLE,
            p_applicable=0.9,
            version_match=VersionMatch.MATCH,
            confidence=0.4,
        ),
        likelihood=ExploitLikelihood(
            finding_id=finding_id,
            attacker="opportunistic",
            p_exploit=0.55,
            p_exploit_uncapped=0.55,
            horizon_days=90,
        ),
        impact=BusinessImpact(finding_id=finding_id, total=250_000.0),
        remediation=RemediationCost(finding_id=finding_id, hours=8.0, cost=960.0),
        expected_loss=137_500.0,
        as_of=AS_OF,
    )


#: The graded scan. The first two rows are in KEV and the rest are not, so a ranker that
#: sorts by KEV puts f_t1 and f_t2 on top and nothing else moves.
TEST_LAYOUT = (
    ("f_t1", True), ("f_t2", True), ("f_t3", False),
    ("f_t4", False), ("f_t5", False), ("f_t6", False),
)

#: Two positives the ranker can look up and one it cannot. ``kev_first`` ranks f_t1 and
#: f_t2 first by construction and leaves f_t5 far down, where its CVSS puts it.
GRADED = LabelSet(
    observation_cutoff=CUTOFF,
    labels=(
        label("f_t1", LabelSource.KEV),
        label("f_t2", LabelSource.KEV),
        label("f_t5", LabelSource.SYNTHETIC_ORACLE),
    ),
)


@pytest.fixture
def corpus() -> tuple[list[EnrichedFinding], list[Split]]:
    rows = [
        make_enriched(f"f_r{i}", "scan_train", i, in_kev=i < 2)
        for i in range(6)
    ]
    rows += [
        make_enriched(name, "scan_test", i, in_kev=kev)
        for i, (name, kev) in enumerate(TEST_LAYOUT)
    ]
    split = Split(
        kind=SplitKind.TIME_ORDERED,
        fold=0,
        train_scan_ids=("scan_train",),
        test_scan_ids=("scan_test",),
        train_end=date(2024, 4, 1),
        test_start=date(2024, 5, 1),
    )
    return rows, [split]


def ndcg_of(bundles, ranker: RankerName = RankerName.KEV_FIRST) -> float:
    for bundle in bundles:
        if bundle.ranker == ranker:
            value = bundle.get(MetricName.NDCG_AT_K, k=10)
            assert value is not None, "the bundle carries no ndcg@10"
            return value
    raise AssertionError(f"{ranker} produced no metrics")


def test_evaluate_stage_stops_paying_kev_first_for_looking_kev_up(corpus) -> None:
    """The same ranker, the same rows, the same fold: only the payout changes.

    ``kev_first`` is a lookup on ``b_kev``, and two of the three positives were labelled
    positive *because* of ``b_kev``. With those counted it looks like a competent ranker.
    With them excluded it is graded on the one outcome its column does not contain, which is
    the only honest question to ask of it.
    """
    rows, splits = corpus
    honest = evaluate_stage(
        config_with(exclude=True), (), rows, GRADED,
        splits=splits, rankers=[RankerName.KEV_FIRST],
    )
    optimistic = evaluate_stage(
        config_with(exclude=False), (), rows, GRADED,
        splits=splits, rankers=[RankerName.KEV_FIRST],
    )
    assert ndcg_of(optimistic) > ndcg_of(honest)


def test_evaluate_stage_refuses_a_fold_whose_every_positive_is_a_lookup(corpus) -> None:
    rows, splits = corpus
    lookups = LabelSet(
        observation_cutoff=CUTOFF,
        labels=(label("f_t1", LabelSource.KEV), label("f_t2", LabelSource.KEV)),
    )
    with pytest.raises(ConfigError, match="exclude_circular_labels"):
        evaluate_stage(
            config_with(exclude=True), (), rows, lookups,
            splits=splits, rankers=[RankerName.KEV_FIRST],
        )
