"""CVSS selection policy and submetric flags (DESIGN.md 3.2, Gap 3).

Two things are pinned here. First, *which* record wins when a CVE carries several: newest
version, then NVD over CNA, then deterministic tie-breaks - so the feature a model trains on
does not depend on JSON ordering. Second, that disagreement between scoring organisations is
reported as a number rather than hidden by picking a favourite.
"""

from __future__ import annotations

from datetime import date

import pytest

from vulnpriority.core.config import PROJECT_ROOT
from vulnpriority.core.enums import CvssVersion, ScoreSource
from vulnpriority.core.models import CvssRecord
from vulnpriority.feeds import (
    NvdFixtureFeed,
    cvss_features,
    cvss_version_ordinal,
    parse_cvss_vector,
    select_cvss,
    source_agreement,
    submetric_flags,
)

AS_OF = date(2025, 6, 1)  # late enough that every fixture CVE is published
FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "feeds"


def record(
    version: CvssVersion,
    source: ScoreSource,
    score: float,
    vector: str | None = None,
) -> CvssRecord:
    return CvssRecord(
        version=version,
        source=source,
        base_score=score,
        vector=vector,
        submetrics=parse_cvss_vector(vector),
    )


def fixture_cvss(cve_id: str) -> tuple[CvssRecord, ...]:
    intel = NvdFixtureFeed(FIXTURE_DIR).get(cve_id, AS_OF)
    assert intel is not None
    return intel.cvss


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_no_records_means_no_score_and_full_agreement() -> None:
    chosen, agreement = select_cvss(())
    assert chosen is None
    assert agreement == 1.0
    assert select_cvss(None) == (None, 1.0)


def test_version_ordinals_rank_newest_highest() -> None:
    ordinals = [cvss_version_ordinal(version) for version in
                (CvssVersion.V2, CvssVersion.V30, CvssVersion.V31, CvssVersion.V40)]
    assert ordinals == sorted(ordinals) == [0, 1, 2, 3]


def test_newest_version_wins_even_when_it_scores_lower() -> None:
    records = (
        record(CvssVersion.V2, ScoreSource.NVD, 10.0, "AV:N/AC:L/Au:N/C:C/I:C/A:C"),
        record(CvssVersion.V31, ScoreSource.NVD, 7.5, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H"),
    )
    chosen, _ = select_cvss(records)
    assert chosen is not None and chosen.version == CvssVersion.V31
    assert chosen.base_score == 7.5


def test_nvd_beats_cna_at_the_same_version() -> None:
    nvd = record(CvssVersion.V31, ScoreSource.NVD, 7.5, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N")
    cna = record(CvssVersion.V31, ScoreSource.CNA, 9.8, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
    for ordering in ((nvd, cna), (cna, nvd)):
        chosen, _ = select_cvss(ordering)
        assert chosen is not None and chosen.source == ScoreSource.NVD


def test_source_priority_orders_scanner_and_other_last() -> None:
    records = (
        record(CvssVersion.V31, ScoreSource.OTHER, 9.9),
        record(CvssVersion.V31, ScoreSource.SCANNER, 9.8),
        record(CvssVersion.V31, ScoreSource.CNA, 5.0),
    )
    chosen, _ = select_cvss(records)
    assert chosen is not None and chosen.source == ScoreSource.CNA


def test_ties_within_a_source_break_on_the_higher_score_then_the_vector() -> None:
    low = record(CvssVersion.V31, ScoreSource.NVD, 5.0, "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:L/I:N/A:N")
    high = record(CvssVersion.V31, ScoreSource.NVD, 9.8, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
    assert select_cvss((low, high))[0] is high
    assert select_cvss((high, low))[0] is high
    # identical scores: the vector string decides, so the result is order-independent
    a = record(CvssVersion.V31, ScoreSource.NVD, 7.5, "CVSS:3.1/AV:A/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N")
    b = record(CvssVersion.V31, ScoreSource.NVD, 7.5, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N")
    assert select_cvss((a, b))[0].vector == select_cvss((b, a))[0].vector


# ---------------------------------------------------------------------------
# Source agreement
# ---------------------------------------------------------------------------


def test_single_source_always_fully_agrees() -> None:
    assert source_agreement((record(CvssVersion.V31, ScoreSource.NVD, 9.8),)) == 1.0


def test_two_versions_from_one_source_are_not_a_disagreement() -> None:
    """A v2/v3.1 pair from NVD is a scale change, not two organisations disputing severity."""
    records = (
        record(CvssVersion.V31, ScoreSource.NVD, 9.8, "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"),
        record(CvssVersion.V2, ScoreSource.NVD, 7.5, "AV:N/AC:L/Au:N/C:P/I:P/A:P"),
    )
    assert source_agreement(records) == 1.0


@pytest.mark.parametrize(
    "nvd_score,cna_score,expected",
    [(9.8, 9.8, 1.0), (9.8, 7.5, 0.77), (10.0, 0.0, 0.0), (9.0, 4.0, 0.5)],
)
def test_agreement_is_one_minus_the_normalised_spread(
    nvd_score: float, cna_score: float, expected: float
) -> None:
    records = (
        record(CvssVersion.V31, ScoreSource.NVD, nvd_score),
        record(CvssVersion.V31, ScoreSource.CNA, cna_score),
    )
    assert source_agreement(records) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Real fixture records
# ---------------------------------------------------------------------------


def test_v2_only_cve_selects_the_v2_record() -> None:
    """CVE-2014-0160 (Heartbleed) predates CVSS v3 entirely."""
    records = fixture_cvss("CVE-2014-0160")
    assert len(records) == 1
    chosen, agreement = select_cvss(records)
    assert chosen is not None
    assert chosen.version == CvssVersion.V2
    assert chosen.base_score == pytest.approx(5.0)
    assert agreement == 1.0
    flags = submetric_flags(chosen)
    assert flags["cvss_ac_low"] == 1.0        # AC:L
    assert flags["cvss_pr_none"] == 1.0       # Au:N
    assert flags["cvss_ui_none"] == 1.0       # v2 has no UI metric
    assert flags["cvss_c_high"] == 0.0        # C:P is partial, not complete
    assert flags["cvss_i_high"] == 0.0
    assert flags["cvss_a_high"] == 0.0


def test_v31_and_v2_from_the_same_source_prefers_v31() -> None:
    """CVE-2019-11043 carries NVD scores under both v3.1 (9.8) and v2 (7.5)."""
    records = fixture_cvss("CVE-2019-11043")
    assert {r.version for r in records} == {CvssVersion.V31, CvssVersion.V2}
    chosen, agreement = select_cvss(records)
    assert chosen is not None
    assert chosen.version == CvssVersion.V31
    assert chosen.base_score == pytest.approx(9.8)
    assert agreement == 1.0
    assert submetric_flags(chosen) == {
        "cvss_ac_low": 1.0,
        "cvss_pr_none": 1.0,
        "cvss_ui_none": 1.0,
        "cvss_c_high": 1.0,
        "cvss_i_high": 1.0,
        "cvss_a_high": 1.0,
    }


def test_nvd_versus_cna_disagreement_is_measured_not_hidden() -> None:
    """CVE-2020-1938: NVD scores Ghostcat 9.8, the Apache CNA scores it 7.5."""
    records = fixture_cvss("CVE-2020-1938")
    chosen, agreement = select_cvss(records)
    assert chosen is not None
    assert chosen.source == ScoreSource.NVD
    assert chosen.base_score == pytest.approx(9.8)
    assert agreement == pytest.approx(0.77)
    # the CNA's lower impact assessment is still present in the record set
    cna = [r for r in records if r.source == ScoreSource.CNA]
    assert cna and cna[0].base_score == pytest.approx(7.5)
    assert submetric_flags(cna[0])["cvss_i_high"] == 0.0


def test_cna_only_cve_with_a_v4_metric_selects_v4() -> None:
    """CVE-2024-27198 has no NVD analysis; JetBrains published v4.0 and v3.1."""
    records = fixture_cvss("CVE-2024-27198")
    chosen, agreement = select_cvss(records)
    assert chosen is not None
    assert chosen.version == CvssVersion.V40
    assert chosen.source == ScoreSource.CNA
    assert agreement == 1.0
    flags = submetric_flags(chosen)
    assert flags["cvss_c_high"] == 1.0   # VC:H
    assert flags["cvss_ui_none"] == 1.0  # UI:N


def test_disagreeing_sources_in_the_synthetic_cve() -> None:
    """CVE-2024-0001: NVD 9.8, a secondary CNA 8.3, and an older NVD v2 score."""
    records = fixture_cvss("CVE-2024-0001")
    chosen, agreement = select_cvss(records)
    assert chosen is not None
    assert chosen.source == ScoreSource.NVD and chosen.version == CvssVersion.V31
    assert agreement == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Submetric flags
# ---------------------------------------------------------------------------


def test_submetric_flags_of_a_hard_to_exploit_v3_record() -> None:
    hard = record(
        CvssVersion.V30, ScoreSource.NVD, 6.3, "CVSS:3.0/AV:N/AC:H/PR:L/UI:R/S:U/C:L/I:L/A:L"
    )
    assert submetric_flags(hard) == {
        "cvss_ac_low": 0.0,
        "cvss_pr_none": 0.0,
        "cvss_ui_none": 0.0,
        "cvss_c_high": 0.0,
        "cvss_i_high": 0.0,
        "cvss_a_high": 0.0,
    }


def test_submetric_flags_of_a_complete_impact_v2_record() -> None:
    complete = record(CvssVersion.V2, ScoreSource.NVD, 10.0, "AV:N/AC:L/Au:N/C:C/I:C/A:C")
    flags = submetric_flags(complete)
    assert flags["cvss_c_high"] == flags["cvss_i_high"] == flags["cvss_a_high"] == 1.0
    assert flags["cvss_pr_none"] == 1.0


def test_v2_authentication_maps_onto_privileges_required() -> None:
    single = record(CvssVersion.V2, ScoreSource.NVD, 6.5, "AV:N/AC:L/Au:S/C:P/I:P/A:P")
    assert submetric_flags(single)["cvss_pr_none"] == 0.0


def test_submetric_flags_fall_back_to_the_vector_when_submetrics_are_empty() -> None:
    bare = CvssRecord(
        version=CvssVersion.V31,
        source=ScoreSource.NVD,
        base_score=9.8,
        vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
    )
    assert bare.submetrics == {}
    assert submetric_flags(bare)["cvss_c_high"] == 1.0


def test_submetric_flags_of_unknown_or_empty_records_are_all_zero() -> None:
    assert set(submetric_flags(None).values()) == {0.0}
    naked = CvssRecord(version=CvssVersion.V31, source=ScoreSource.NVD, base_score=5.0)
    assert set(submetric_flags(naked).values()) == {0.0}


def test_parse_cvss_vector_handles_both_prefixed_and_bare_forms() -> None:
    assert parse_cvss_vector("CVSS:3.1/AV:N/AC:L")["AV"] == "N"
    assert parse_cvss_vector("AV:N/AC:L/Au:N")["AU"] == "N"
    assert parse_cvss_vector(None) == {}
    assert parse_cvss_vector("garbage") == {}


# ---------------------------------------------------------------------------
# Feature bundle
# ---------------------------------------------------------------------------


def test_cvss_features_emit_the_base_group_columns() -> None:
    features = cvss_features(fixture_cvss("CVE-2020-1938"))
    assert features["cvss_base_max"] == pytest.approx(9.8)
    assert features["cvss_version_ord"] == 2.0           # v3.1
    assert features["cvss_source_agreement"] == pytest.approx(0.77)
    assert features["cvss_ac_low"] == 1.0
    assert set(features) == {
        "cvss_base_max",
        "cvss_version_ord",
        "cvss_source_agreement",
        "cvss_ac_low",
        "cvss_pr_none",
        "cvss_ui_none",
        "cvss_c_high",
        "cvss_i_high",
        "cvss_a_high",
    }


def test_cvss_features_of_an_unscored_cve_are_neutral() -> None:
    features = cvss_features(())
    assert features["cvss_base_max"] == 0.0
    assert features["cvss_version_ord"] == 0.0
    assert features["cvss_source_agreement"] == 1.0
