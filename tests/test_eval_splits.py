"""Splitting protocol (DESIGN.md 3.9 and 4, Gap 4 and Gap 8).

The temporal-leakage test is the one that matters: a time-ordered split that quietly
admits a later scan into training invalidates every number downstream of it, and it is
the single most common defect in the evaluations the review surveyed.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from vulnprio.core.config import EvaluationConfig
from vulnprio.core.enums import SplitKind
from vulnprio.core.errors import ConfigError, TemporalLeakageError
from vulnprio.core.interfaces import Splitter
from vulnprio.core.models import LabelPolicy, LabelSet, Scan, Split
from vulnprio.eval.splits import (
    LeaveOneAppOutSplitter,
    RandomSplitter,
    TimeOrderedSplitter,
    assert_no_temporal_leakage,
    build_splitter,
    scan_dates,
)

START = datetime(2024, 1, 15, 9, 0, 0)


def make_scans(n_apps: int = 3, per_app: int = 4, interval_days: int = 30) -> list[Scan]:
    """``n_apps * per_app`` scans on a regular cadence, interleaved across applications."""
    scans: list[Scan] = []
    for round_index in range(per_app):
        for app_index in range(n_apps):
            when = START + timedelta(days=round_index * interval_days + app_index)
            scans.append(
                Scan(
                    scan_id=f"scan_a{app_index}_r{round_index}",
                    app_id=f"app{app_index}",
                    app_name=f"App {app_index}",
                    sector="ecommerce",
                    scanned_at=when,
                    scanner_name="zap",
                    hosts=(f"app{app_index}.example.com",),
                )
            )
    return scans


@pytest.fixture
def scans() -> list[Scan]:
    return make_scans()


@pytest.fixture
def labels() -> LabelSet:
    return LabelSet(policy=LabelPolicy(), observation_cutoff=date(2025, 1, 1), labels=())


# ---------------------------------------------------------------------------
# Temporal leakage: the load-bearing assertion
# ---------------------------------------------------------------------------


def test_time_ordered_split_never_puts_a_later_scan_in_train(scans, labels) -> None:
    """No training scan may be dated after any test scan, on any fold."""
    dates = scan_dates(scans)
    splits = TimeOrderedSplitter(n_folds=3, gap_days=30, min_train_scans=3).split(scans, labels)
    assert splits

    for split in splits:
        assert split.kind == SplitKind.TIME_ORDERED
        assert split.train_scan_ids and split.test_scan_ids
        latest_train = max(dates[scan_id] for scan_id in split.train_scan_ids)
        earliest_test = min(dates[scan_id] for scan_id in split.test_scan_ids)
        assert earliest_test >= latest_train
        for train_id in split.train_scan_ids:
            for test_id in split.test_scan_ids:
                assert dates[train_id] <= dates[test_id]
        assert not set(split.train_scan_ids) & set(split.test_scan_ids)


def test_the_gap_buffer_is_actually_enforced(scans, labels) -> None:
    """Labels arrive with a lag; the buffer discards the window they are missing from."""
    dates = scan_dates(scans)
    gap = 45
    splits = TimeOrderedSplitter(n_folds=2, gap_days=gap, min_train_scans=3).split(scans, labels)
    for split in splits:
        assert split.gap_days == gap
        latest_train = max(dates[scan_id] for scan_id in split.train_scan_ids)
        earliest_test = min(dates[scan_id] for scan_id in split.test_scan_ids)
        assert (earliest_test - latest_train).days >= gap


def test_the_leakage_assertion_rejects_a_hand_built_leaky_fold(scans) -> None:
    """The guard is real: hand a leaky fold to it and it raises."""
    dates = scan_dates(scans)
    ordered = sorted(dates, key=lambda scan_id: dates[scan_id])
    leaky = Split(
        kind=SplitKind.TIME_ORDERED,
        fold=0,
        train_scan_ids=(ordered[0], ordered[-1]),      # a scan from the future in training
        test_scan_ids=(ordered[4],),
        train_end=dates[ordered[0]],
        test_start=dates[ordered[4]],
        gap_days=0,
    )
    with pytest.raises(TemporalLeakageError, match="precedes"):
        assert_no_temporal_leakage(leaky, dates)


def test_the_leakage_assertion_rejects_an_undersized_gap(scans) -> None:
    dates = scan_dates(scans)
    ordered = sorted(dates, key=lambda scan_id: dates[scan_id])
    tight = Split(
        kind=SplitKind.TIME_ORDERED,
        fold=0,
        train_scan_ids=tuple(ordered[:3]),
        test_scan_ids=(ordered[3],),
        train_end=dates[ordered[2]],
        test_start=dates[ordered[3]],
        gap_days=90,
    )
    with pytest.raises(TemporalLeakageError, match="require"):
        assert_no_temporal_leakage(tight, dates)


def test_the_frozen_split_contract_refuses_an_inverted_window() -> None:
    with pytest.raises(Exception):
        Split(
            kind=SplitKind.TIME_ORDERED,
            train_scan_ids=("a",),
            test_scan_ids=("b",),
            train_end=date(2024, 6, 1),
            test_start=date(2024, 5, 1),
        )


# ---------------------------------------------------------------------------
# Rolling origin behaviour
# ---------------------------------------------------------------------------


def test_folds_roll_forward_and_training_grows(scans, labels) -> None:
    splits = TimeOrderedSplitter(n_folds=3, gap_days=0, min_train_scans=3).split(scans, labels)
    assert [split.fold for split in splits] == sorted(split.fold for split in splits)
    sizes = [len(split.train_scan_ids) for split in splits]
    assert sizes == sorted(sizes)
    assert sizes[0] >= 3
    starts = [split.test_start for split in splits]
    assert starts == sorted(starts)


def test_min_train_scans_is_respected(scans, labels) -> None:
    splits = TimeOrderedSplitter(n_folds=2, gap_days=0, min_train_scans=6).split(scans, labels)
    assert all(len(split.train_scan_ids) >= 6 for split in splits)


def test_too_few_scans_raises_rather_than_returning_nothing(labels) -> None:
    with pytest.raises(ConfigError):
        TimeOrderedSplitter(min_train_scans=4).split(make_scans(n_apps=1, per_app=3), labels)


def test_an_impossible_gap_raises_rather_than_silently_evaluating_nothing(scans, labels) -> None:
    with pytest.raises(ConfigError, match="no usable"):
        TimeOrderedSplitter(n_folds=3, gap_days=3650, min_train_scans=3).split(scans, labels)


def test_validation_scans_come_out_of_training_and_stay_before_the_gap(scans, labels) -> None:
    dates = scan_dates(scans)
    splits = TimeOrderedSplitter(n_folds=2, gap_days=0, min_train_scans=4, n_valid_scans=2).split(
        scans, labels
    )
    for split in splits:
        assert len(split.valid_scan_ids) == 2
        assert not set(split.valid_scan_ids) & set(split.train_scan_ids)
        assert not set(split.valid_scan_ids) & set(split.test_scan_ids)
        latest_valid = max(dates[scan_id] for scan_id in split.valid_scan_ids)
        assert latest_valid <= min(dates[scan_id] for scan_id in split.test_scan_ids)


def test_time_ordered_splitting_is_deterministic(scans, labels) -> None:
    splitter = TimeOrderedSplitter(n_folds=3, gap_days=30, min_train_scans=3)
    assert splitter.split(scans, labels) == splitter.split(list(reversed(scans)), labels)


# ---------------------------------------------------------------------------
# Leave one application out (Gap 8)
# ---------------------------------------------------------------------------


def test_leave_one_app_out_holds_out_exactly_one_application(scans, labels) -> None:
    splits = LeaveOneAppOutSplitter().split(scans, labels)
    app_ids = sorted({scan.app_id for scan in scans})
    assert len(splits) == len(app_ids)
    by_id = {scan.scan_id: scan.app_id for scan in scans}
    for split in splits:
        assert split.kind == SplitKind.LEAVE_ONE_APP_OUT
        assert split.held_out_app_id is not None
        assert {by_id[scan_id] for scan_id in split.test_scan_ids} == {split.held_out_app_id}
        assert split.held_out_app_id not in {by_id[scan_id] for scan_id in split.train_scan_ids}
        assert not set(split.train_scan_ids) & set(split.test_scan_ids)


def test_leave_one_app_out_needs_at_least_two_applications(labels) -> None:
    with pytest.raises(ConfigError):
        LeaveOneAppOutSplitter().split(make_scans(n_apps=1, per_app=4), labels)


def test_leave_one_app_out_can_additionally_demand_time_ordering(scans, labels) -> None:
    dates = scan_dates(scans)
    splits = LeaveOneAppOutSplitter(time_ordered=True).split(scans, labels)
    for split in splits:
        latest_train = max(dates[scan_id] for scan_id in split.train_scan_ids)
        assert min(dates[scan_id] for scan_id in split.test_scan_ids) >= latest_train


# ---------------------------------------------------------------------------
# The random control (never a result)
# ---------------------------------------------------------------------------


def test_random_splitter_is_labelled_as_the_control_condition(scans, labels) -> None:
    splits = RandomSplitter(n_folds=3, seed=7).split(scans, labels)
    assert splits
    assert all(split.kind == SplitKind.RANDOM for split in splits)
    covered = {scan_id for split in splits for scan_id in split.test_scan_ids}
    assert covered == {scan.scan_id for scan in scans}
    for split in splits:
        assert not set(split.train_scan_ids) & set(split.test_scan_ids)


def test_the_random_control_leaks_time_which_is_the_point(scans, labels) -> None:
    """Demonstrating the leak is the control's whole purpose (Gap 4)."""
    dates = scan_dates(scans)
    splits = RandomSplitter(n_folds=3, seed=7).split(scans, labels)
    leaked = any(
        max(dates[scan_id] for scan_id in split.train_scan_ids)
        > min(dates[scan_id] for scan_id in split.test_scan_ids)
        for split in splits
    )
    assert leaked
    with pytest.raises(TemporalLeakageError):
        for split in splits:
            assert_no_temporal_leakage(split, dates)


def test_random_splitting_is_seeded_and_reproducible(scans, labels) -> None:
    first = RandomSplitter(n_folds=3, seed=11).split(scans, labels)
    again = RandomSplitter(n_folds=3, seed=11).split(scans, labels)
    other = RandomSplitter(n_folds=3, seed=12).split(scans, labels)
    assert [split.test_scan_ids for split in first] == [split.test_scan_ids for split in again]
    assert [split.test_scan_ids for split in first] != [split.test_scan_ids for split in other]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_every_splitter_implements_the_frozen_interface() -> None:
    for splitter in (TimeOrderedSplitter(), LeaveOneAppOutSplitter(), RandomSplitter()):
        assert isinstance(splitter, Splitter)


def test_build_splitter_honours_the_configured_kind() -> None:
    assert isinstance(build_splitter(EvaluationConfig()), TimeOrderedSplitter)
    assert isinstance(
        build_splitter(EvaluationConfig(split_kind=SplitKind.LEAVE_ONE_APP_OUT)),
        LeaveOneAppOutSplitter,
    )
    assert isinstance(
        build_splitter(EvaluationConfig(split_kind=SplitKind.RANDOM)), RandomSplitter
    )


def test_build_splitter_passes_the_protocol_defaults_through() -> None:
    config = EvaluationConfig()
    splitter = build_splitter(config)
    assert isinstance(splitter, TimeOrderedSplitter)
    assert splitter.gap_days == config.gap_days == 30
    assert splitter.n_folds == config.n_folds == 3
    assert splitter.min_train_scans == config.min_train_scans
