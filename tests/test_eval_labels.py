"""``LabelBuilder``: exploitation ground truth, and what it refuses (Gap 3).

The headline test here is :func:`test_label_builder_refuses_a_cvss_derived_source`. The
rest establish that the labels the builder *does* produce are version-aware, source-aware
and strictly as-of dated.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from vulnprio.core.enums import (
    CvssVersion,
    ExploitMaturity,
    ExploitSource,
    LabelSource,
    Provenance,
    ScannerSeverity,
    ScoreSource,
    VersionMatch,
)
from vulnprio.core.errors import LabelPolicyError
from vulnprio.core.models import (
    CvssRecord,
    EpssRecord,
    ExploitEvidence,
    Finding,
    GroundTruthLabel,
    KevRecord,
    LabelPolicy,
    LabelSet,
    UntrustedText,
    VulnIntel,
)
from vulnprio.eval.labels import LabelBuilder, assert_not_cvss_derived, resolve_label_source

CUTOFF = date(2024, 6, 1)
OBSERVED = datetime(2024, 3, 1, 9, 0, 0)


def finding(finding_id: str, cve: str | None = None, dedup: str | None = None) -> Finding:
    return Finding(
        finding_id=finding_id,
        scan_id="scan_1",
        app_id="app1",
        endpoint_id="ep_1",
        name="Injection",
        cwe_id=89,
        cve_ids=(cve,) if cve else (),
        scanner="zap",
        scanner_severity=ScannerSeverity.HIGH,
        description=UntrustedText(text="finding text", provenance=Provenance.SCANNER_OUTPUT),
        observed_at=OBSERVED,
        dedup_key=dedup or finding_id,
    )


def intel(
    cve: str,
    *,
    as_of: date = CUTOFF,
    kev: bool | None = None,
    kev_added: date | None = date(2024, 2, 1),
    maturity: ExploitMaturity | None = None,
    exploit_published: date | None = date(2024, 2, 10),
    cvss: float | None = 9.8,
) -> VulnIntel:
    return VulnIntel(
        cve_id=cve,
        as_of=as_of,
        published=date(2024, 1, 1),
        cvss=(
            (CvssRecord(version=CvssVersion.V31, source=ScoreSource.NVD, base_score=cvss),)
            if cvss is not None
            else ()
        ),
        epss=EpssRecord(cve_id=cve, score=0.5, percentile=0.9, as_of=as_of),
        kev=None if kev is None else KevRecord(cve_id=cve, in_kev=kev, date_added=kev_added if kev else None, as_of=as_of),
        exploits=(
            ()
            if maturity is None
            else (
                ExploitEvidence(
                    source=ExploitSource.EXPLOIT_DB,
                    published=exploit_published,
                    maturity=maturity,
                    verified=True,
                ),
            )
        ),
    )


# ---------------------------------------------------------------------------
# The prohibition (Gap 3)
# ---------------------------------------------------------------------------


def test_label_builder_refuses_a_cvss_derived_source() -> None:
    """A caller asking to label from CVSS gets an error, not a label set."""
    with pytest.raises(LabelPolicyError, match="circular"):
        LabelBuilder(LabelPolicy(), sources=["kev", "cvss"])

    for name in ("cvss", "CVSS", "cvss_v3_base", "CVSS v3.1 base score", "base_score", "severity"):
        with pytest.raises(LabelPolicyError):
            LabelBuilder(sources=[name])
        with pytest.raises(LabelPolicyError):
            resolve_label_source(name)
        with pytest.raises(LabelPolicyError):
            assert_not_cvss_derived(name)


def test_epss_is_refused_for_the_same_reason_as_cvss() -> None:
    """EPSS is a prediction of exploitation; training on it predicts another model."""
    with pytest.raises(LabelPolicyError):
        LabelBuilder(sources=["epss"])


def test_an_unknown_source_is_refused_rather_than_ignored() -> None:
    with pytest.raises(LabelPolicyError):
        LabelBuilder(sources=["analyst_hunch"])


def test_the_prohibition_is_enforced_at_the_type_level_too() -> None:
    """Three independent locks: the enum, the model field and the builder."""
    assert not any("cvss" in source.value for source in LabelSource)
    assert GroundTruthLabel(finding_id="f").cvss_used_as_label is False
    with pytest.raises(ValidationError):
        GroundTruthLabel(finding_id="f", cvss_used_as_label=True)


def test_accepted_sources_are_exactly_the_four_evidence_sources() -> None:
    assert set(LabelSource) == {
        LabelSource.KEV,
        LabelSource.EXPLOIT_EVIDENCE,
        LabelSource.INCIDENT,
        LabelSource.SYNTHETIC_ORACLE,
    }
    assert LabelBuilder(sources=list(LabelSource)).accepted == set(LabelSource)


# ---------------------------------------------------------------------------
# Positives come from evidence, and only from evidence
# ---------------------------------------------------------------------------


def test_kev_membership_produces_a_positive_at_the_kev_grade() -> None:
    builder = LabelBuilder(LabelPolicy(weight_by_impact=False))
    labels = builder.build(
        [finding("f_kev", "CVE-2024-0001")],
        {"CVE-2024-0001": intel("CVE-2024-0001", kev=True)},
        observation_cutoff=CUTOFF,
    )
    label = labels.by_id("f_kev")
    assert label is not None
    assert label.exploited is True
    assert label.relevance_grade == LabelPolicy().kev_grade == 4
    assert LabelSource.KEV in label.sources
    assert label.first_evidence_date == date(2024, 2, 1)
    assert label.cve_id == "CVE-2024-0001"
    assert label.cvss_used_as_label is False


def test_exploit_evidence_at_the_maturity_floor_produces_a_positive() -> None:
    builder = LabelBuilder(LabelPolicy(weight_by_impact=False))
    labels = builder.build(
        [finding("f_exp", "CVE-2024-0002")],
        {"CVE-2024-0002": intel("CVE-2024-0002", maturity=ExploitMaturity.FUNCTIONAL)},
        observation_cutoff=CUTOFF,
    )
    label = labels.by_id("f_exp")
    assert label is not None and label.exploited is True
    assert label.relevance_grade == LabelPolicy().exploit_evidence_grade == 3
    assert label.sources == (LabelSource.EXPLOIT_EVIDENCE,)


def test_a_proof_of_concept_is_not_proof_of_exploitation() -> None:
    """Maturity below the policy floor is evidence of nothing but research interest."""
    builder = LabelBuilder()
    labels = builder.build(
        [finding("f_poc", "CVE-2024-0003")],
        {"CVE-2024-0003": intel("CVE-2024-0003", maturity=ExploitMaturity.POC)},
        observation_cutoff=CUTOFF,
    )
    label = labels.by_id("f_poc")
    assert label is not None and label.exploited is False and label.relevance_grade == 0


def test_a_high_cvss_alone_never_produces_a_positive() -> None:
    """The finding that would be labelled positive by most of the literature."""
    builder = LabelBuilder()
    labels = builder.build(
        [finding("f_scary", "CVE-2024-0004")],
        {"CVE-2024-0004": intel("CVE-2024-0004", kev=False, maturity=None, cvss=10.0)},
        observation_cutoff=CUTOFF,
    )
    label = labels.by_id("f_scary")
    assert label is not None and label.exploited is False
    assert builder.audit.n_positive == 0


def test_a_finding_with_no_cve_is_labelled_negative_not_dropped() -> None:
    builder = LabelBuilder()
    labels = builder.build([finding("f_xss")], {}, observation_cutoff=CUTOFF)
    label = labels.by_id("f_xss")
    assert label is not None and label.exploited is False and label.cve_id is None
    assert label.source_agreement == 1.0


def test_the_synthetic_oracle_is_an_accepted_source() -> None:
    """The only counterfactually complete ground truth that exists anywhere."""
    builder = LabelBuilder(LabelPolicy(weight_by_impact=False))
    labels = builder.build(
        [finding("f_oracle"), finding("f_quiet")],
        {},
        {"f_oracle": date(2024, 4, 5)},
        observation_cutoff=CUTOFF,
    )
    hit = labels.by_id("f_oracle")
    miss = labels.by_id("f_quiet")
    assert hit is not None and hit.exploited is True
    assert hit.sources == (LabelSource.SYNTHETIC_ORACLE,)
    assert hit.first_evidence_date == date(2024, 4, 5)
    assert miss is not None and miss.exploited is False


def test_oracle_events_are_accepted_as_objects_or_mappings() -> None:
    """Whatever shape ``synth/oracle.py`` emits, the builder reads it."""
    builder = LabelBuilder(LabelPolicy(weight_by_impact=False))
    as_mappings = builder.build(
        [finding("f_a")],
        {},
        [{"finding_id": "f_a", "exploited_at": "2024-04-05"}],
        observation_cutoff=CUTOFF,
    )
    assert as_mappings.positives() == {"f_a"}

    class Event:
        finding_id = "f_a"
        event_date = date(2024, 4, 5)

    as_objects = builder.build([finding("f_a")], {}, [Event()], observation_cutoff=CUTOFF)
    assert as_objects.positives() == {"f_a"}


def test_incidents_are_only_used_when_the_policy_accepts_them() -> None:
    policy = LabelPolicy(weight_by_impact=False)
    assert LabelSource.INCIDENT not in policy.accepted_sources

    ignored = LabelBuilder(policy).build(
        [finding("f_inc", "CVE-2024-0005")],
        {},
        incidents={"CVE-2024-0005": date(2024, 5, 1)},
        observation_cutoff=CUTOFF,
    )
    assert ignored.positives() == set()

    accepted = LabelBuilder(
        policy.model_copy(update={"accepted_sources": (LabelSource.INCIDENT,)})
    ).build(
        [finding("f_inc", "CVE-2024-0005")],
        {},
        incidents={"CVE-2024-0005": date(2024, 5, 1)},
        observation_cutoff=CUTOFF,
    )
    label = accepted.by_id("f_inc")
    assert label is not None and label.exploited is True
    assert label.relevance_grade == policy.incident_grade


# ---------------------------------------------------------------------------
# As-of dating
# ---------------------------------------------------------------------------


def test_evidence_dated_after_the_cutoff_does_not_create_a_positive() -> None:
    """What makes a time-ordered split honest: the label cannot see the future."""
    builder = LabelBuilder()
    labels = builder.build(
        [finding("f_late", "CVE-2024-0006")],
        {"CVE-2024-0006": intel("CVE-2024-0006", kev=True, kev_added=date(2024, 5, 20))},
        observation_cutoff=date(2024, 4, 1),
    )
    label = labels.by_id("f_late")
    assert label is not None and label.exploited is False
    assert builder.audit.evidence_after_cutoff == 1

    later = LabelBuilder().build(
        [finding("f_late", "CVE-2024-0006")],
        {"CVE-2024-0006": intel("CVE-2024-0006", kev=True, kev_added=date(2024, 5, 20))},
        observation_cutoff=CUTOFF,
    )
    assert later.positives() == {"f_late"}


def test_an_oracle_event_after_the_cutoff_is_not_yet_a_positive() -> None:
    labels = LabelBuilder().build(
        [finding("f_future")],
        {},
        {"f_future": date(2024, 9, 1)},
        observation_cutoff=CUTOFF,
    )
    assert labels.positives() == set()
    assert labels.observation_cutoff == CUTOFF


# ---------------------------------------------------------------------------
# Version awareness
# ---------------------------------------------------------------------------


def test_a_version_mismatch_is_dropped_rather_than_labelled_positive() -> None:
    """The CVE was exploited somewhere; this installation is not running that version."""
    builder = LabelBuilder()
    labels = builder.build(
        [finding("f_match", "CVE-2024-0007"), finding("f_mismatch", "CVE-2024-0007")],
        {"CVE-2024-0007": intel("CVE-2024-0007", kev=True)},
        observation_cutoff=CUTOFF,
        version_match_by_finding={
            "f_match": VersionMatch.MATCH,
            "f_mismatch": VersionMatch.MISMATCH,
        },
    )
    assert labels.by_id("f_mismatch") is None              # dropped, not labelled either way
    assert labels.by_id("f_match") is not None
    assert labels.positives() == {"f_match"}
    assert builder.audit.dropped_version_mismatch == ("f_mismatch",)
    assert builder.audit.n_labelled == 1


def test_a_version_mismatch_can_be_demoted_instead_of_dropped() -> None:
    policy = LabelPolicy(drop_version_mismatch=False)
    labels = LabelBuilder(policy).build(
        [finding("f_mismatch", "CVE-2024-0007")],
        {"CVE-2024-0007": intel("CVE-2024-0007", kev=True)},
        observation_cutoff=CUTOFF,
        version_match_by_finding={"f_mismatch": VersionMatch.MISMATCH},
    )
    label = labels.by_id("f_mismatch")
    assert label is not None
    assert label.exploited is False
    assert label.version_confirmed == VersionMatch.MISMATCH


def test_version_evidence_can_be_switched_off_entirely() -> None:
    policy = LabelPolicy(require_version_match=False, weight_by_impact=False)
    labels = LabelBuilder(policy).build(
        [finding("f_mismatch", "CVE-2024-0007")],
        {"CVE-2024-0007": intel("CVE-2024-0007", kev=True)},
        observation_cutoff=CUTOFF,
        version_match_by_finding={"f_mismatch": VersionMatch.MISMATCH},
    )
    assert labels.positives() == {"f_mismatch"}


def test_unknown_version_evidence_does_not_block_a_positive() -> None:
    """Most web findings carry no version data at all; requiring MATCH would erase them."""
    labels = LabelBuilder().build(
        [finding("f_unknown", "CVE-2024-0008")],
        {"CVE-2024-0008": intel("CVE-2024-0008", kev=True)},
        observation_cutoff=CUTOFF,
    )
    label = labels.by_id("f_unknown")
    assert label is not None and label.exploited is True
    assert label.version_confirmed == VersionMatch.UNKNOWN


# ---------------------------------------------------------------------------
# Source awareness
# ---------------------------------------------------------------------------


def test_source_agreement_is_recorded() -> None:
    """Unanimous evidence scores 1.0; a source that disagrees pulls it down."""
    unanimous = LabelBuilder(LabelPolicy(weight_by_impact=False)).build(
        [finding("f_both", "CVE-2024-0009")],
        {"CVE-2024-0009": intel("CVE-2024-0009", kev=True, maturity=ExploitMaturity.WEAPONIZED)},
        observation_cutoff=CUTOFF,
    )
    label = unanimous.by_id("f_both")
    assert label is not None
    assert set(label.sources) == {LabelSource.KEV, LabelSource.EXPLOIT_EVIDENCE}
    assert label.source_agreement == pytest.approx(1.0)

    split = LabelBuilder(LabelPolicy(weight_by_impact=False)).build(
        [finding("f_kev_only", "CVE-2024-0010")],
        {"CVE-2024-0010": intel("CVE-2024-0010", kev=True, maturity=ExploitMaturity.POC)},
        observation_cutoff=CUTOFF,
    )
    partial = split.by_id("f_kev_only")
    assert partial is not None and partial.exploited is True
    assert partial.sources == (LabelSource.KEV,)
    assert partial.source_agreement == pytest.approx(0.5)


def test_min_source_agreement_can_require_corroboration() -> None:
    """A stricter policy drops the uncorroborated positive rather than guessing."""
    policy = LabelPolicy(min_source_agreement=0.9)
    builder = LabelBuilder(policy)
    labels = builder.build(
        [finding("f_kev_only", "CVE-2024-0010")],
        {"CVE-2024-0010": intel("CVE-2024-0010", kev=True, maturity=ExploitMaturity.POC)},
        observation_cutoff=CUTOFF,
    )
    assert labels.by_id("f_kev_only") is None
    assert builder.audit.dropped_low_agreement == ("f_kev_only",)


# ---------------------------------------------------------------------------
# Graded relevance
# ---------------------------------------------------------------------------


def test_impact_quartile_weighting_moves_grades_within_the_positive_class() -> None:
    """Impact refines the grade of a positive; it can never create or destroy one."""
    findings = [finding(f"f{index}", "CVE-2024-0011") for index in range(8)]
    impacts = {f"f{index}": float(index + 1) * 1000.0 for index in range(8)}
    intel_map = {"CVE-2024-0011": intel("CVE-2024-0011", maturity=ExploitMaturity.FUNCTIONAL)}

    weighted = LabelBuilder(LabelPolicy(weight_by_impact=True)).build(
        findings, intel_map, observation_cutoff=CUTOFF, impact_by_finding=impacts
    )
    flat = LabelBuilder(LabelPolicy(weight_by_impact=False)).build(
        findings, intel_map, observation_cutoff=CUTOFF, impact_by_finding=impacts
    )

    assert weighted.positives() == flat.positives()
    assert set(flat.relevance().values()) == {3}
    graded = weighted.relevance()
    assert graded["f0"] < graded["f3"] <= graded["f7"]
    assert graded["f0"] == 2 and graded["f7"] == 4
    assert all(1 <= grade <= 4 for grade in graded.values())


def test_grades_stay_inside_the_contract_range() -> None:
    findings = [finding(f"g{index}", "CVE-2024-0012") for index in range(4)]
    labels = LabelBuilder().build(
        findings,
        {"CVE-2024-0012": intel("CVE-2024-0012", kev=True)},
        observation_cutoff=CUTOFF,
        impact_by_finding={f"g{index}": float(index) for index in range(4)},
    )
    for label in labels.labels:
        assert 0 <= label.relevance_grade <= 4


def test_negative_labels_are_never_impact_weighted() -> None:
    labels = LabelBuilder().build(
        [finding("f_neg")], {}, observation_cutoff=CUTOFF, impact_by_finding={"f_neg": 1e9}
    )
    label = labels.by_id("f_neg")
    assert label is not None and label.relevance_grade == 0


# ---------------------------------------------------------------------------
# The label set as a whole
# ---------------------------------------------------------------------------


def test_label_set_carries_the_policy_and_the_cutoff() -> None:
    policy = LabelPolicy(kev_grade=4, weight_by_impact=False)
    labels = LabelBuilder(policy).build([finding("f1")], {}, observation_cutoff=CUTOFF)
    assert isinstance(labels, LabelSet)
    assert labels.observation_cutoff == CUTOFF
    assert labels.policy.accepted_sources == policy.accepted_sources


def test_the_audit_explains_every_decision() -> None:
    builder = LabelBuilder()
    builder.build(
        [
            finding("f_kev", "CVE-2024-0013"),
            finding("f_mismatch", "CVE-2024-0013"),
            finding("f_plain"),
        ],
        {"CVE-2024-0013": intel("CVE-2024-0013", kev=True)},
        observation_cutoff=CUTOFF,
        version_match_by_finding={"f_mismatch": VersionMatch.MISMATCH},
    )
    audit = builder.audit.as_dict()
    assert audit["n_findings"] == 3
    assert audit["n_labelled"] == 2
    assert audit["n_positive"] == 1
    assert audit["n_dropped_version_mismatch"] == 1
    assert audit["positives_by_source"] == {"kev": 1}


def test_building_twice_is_deterministic() -> None:
    builder = LabelBuilder()
    args = (
        [finding("f_kev", "CVE-2024-0014"), finding("f_none")],
        {"CVE-2024-0014": intel("CVE-2024-0014", kev=True, maturity=ExploitMaturity.FUNCTIONAL)},
    )
    first = builder.build(*args, observation_cutoff=CUTOFF)
    second = builder.build(*args, observation_cutoff=CUTOFF)
    assert first.labels == second.labels
