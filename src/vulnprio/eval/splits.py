"""Train/test splitting (DESIGN.md 3.9 and 4, Gap 4 and Gap 8).

Three splitters, each answering a different question:

``TimeOrderedSplitter``
    *Would this have worked?* Rolling-origin folds with a ``gap_days`` buffer between
    the end of training and the start of testing. This is the protocol's primary
    splitter, and the only one whose numbers may be reported as performance.

``LeaveOneAppOutSplitter``
    *Does it transfer?* One fold per application: train on every other application, test
    on the held-out one. Gap 8's answer to evaluations that rest on a single organisation.

``RandomSplitter``
    *How much does random splitting overstate?* Provided **only** as a labelled control.
    It is a leaky design for this problem - findings from the same application and the
    same week land on both sides - and the report prints the gap between it and the
    time-ordered result as a measurement of that leak, never as a result.

The gap buffer matters more than it looks. Exploitation evidence arrives with a lag
(``SyntheticConfig.label_lag_days`` models 3 to 45 days): a model trained on scans right
up to the test window would be trained on labels that did not exist when those scans were
taken. The buffer discards exactly that window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

import numpy as np

from vulnprio.core.enums import SplitKind
from vulnprio.core.errors import ConfigError, TemporalLeakageError
from vulnprio.core.models import LabelSet, Scan, Split
from vulnprio.core.config import EvaluationConfig
from vulnprio.core.interfaces import Splitter

__all__ = [
    "TimeOrderedSplitter",
    "LeaveOneAppOutSplitter",
    "RandomSplitter",
    "build_splitter",
    "assert_no_temporal_leakage",
    "scan_dates",
]


def scan_dates(scans: Sequence[Scan]) -> dict[str, date]:
    """``{scan_id: scan date}``; the whole module works in dates, not datetimes."""
    return {scan.scan_id: scan.scanned_at.date() for scan in scans}


def assert_no_temporal_leakage(split: Split, dates: dict[str, date]) -> None:
    """Raise unless every test scan is dated on or after every training scan.

    This is the assertion the protocol rests on, so it runs on every fold every time
    rather than in a test only. ``Split``'s own validator checks the declared window
    (``test_start >= train_end``); this checks the scans actually placed in the fold.
    """
    train = [dates[scan_id] for scan_id in split.train_scan_ids if scan_id in dates]
    test = [dates[scan_id] for scan_id in split.test_scan_ids if scan_id in dates]
    if not train or not test:
        return
    latest_train, earliest_test = max(train), min(test)
    if earliest_test < latest_train:
        raise TemporalLeakageError(
            f"fold {split.fold}: test scan dated {earliest_test} precedes training scan "
            f"dated {latest_train}; a time-ordered split may never look forward"
        )
    observed_gap = (earliest_test - latest_train).days
    if observed_gap < split.gap_days:
        raise TemporalLeakageError(
            f"fold {split.fold}: only {observed_gap} days separate training from testing, "
            f"but the policy requires {split.gap_days}"
        )


@dataclass
class TimeOrderedSplitter(Splitter):
    """Rolling-origin folds with a buffer (the protocol's primary splitter).

    With scans sorted by date, fold ``i`` trains on the first ``min_train_scans + i*block``
    scans and tests on the next ``block``, where ``block`` divides the remaining scans
    over ``n_folds``. Every scan dated within ``gap_days`` after the training window is
    excluded from that fold entirely: it is too recent to be tested on honestly and too
    close to the boundary to be trained on.

    ``n_valid_scans`` carves that many scans off the end of the training window as a
    validation set (for early stopping), which keeps validation strictly before the gap
    and therefore strictly before the test window.

    Folds that end up with no training or no test scan are dropped rather than emitted
    degenerate; a configuration that produces no usable fold raises ``ConfigError``,
    since silently returning an empty list would make the whole evaluation vanish.
    """

    n_folds: int = 3
    gap_days: int = 30
    min_train_scans: int = 4
    n_valid_scans: int = 0

    def split(self, scans: list[Scan], labels: LabelSet | None = None) -> list[Split]:
        if self.n_folds < 1:
            raise ConfigError("n_folds must be at least 1")
        dates = scan_dates(scans)
        ordered = sorted(scans, key=lambda scan: (scan.scanned_at, scan.scan_id))
        ids = [scan.scan_id for scan in ordered]
        n = len(ids)
        if n <= self.min_train_scans:
            raise ConfigError(
                f"{n} scans cannot support a time-ordered split needing more than "
                f"{self.min_train_scans} training scans"
            )

        remaining = n - self.min_train_scans
        block = max(1, remaining // self.n_folds)
        splits: list[Split] = []
        for fold in range(self.n_folds):
            train_end_index = self.min_train_scans + fold * block
            if train_end_index >= n:
                break
            is_last = fold == self.n_folds - 1
            test_stop = n if is_last else min(n, train_end_index + block)
            train_ids = ids[:train_end_index]
            candidate_test = ids[train_end_index:test_stop]
            if not train_ids or not candidate_test:
                continue

            train_end = max(dates[scan_id] for scan_id in train_ids)
            earliest_allowed = train_end + timedelta(days=self.gap_days)
            test_ids = [scan_id for scan_id in candidate_test if dates[scan_id] >= earliest_allowed]
            if not test_ids:
                continue

            valid_ids: tuple[str, ...] = ()
            if self.n_valid_scans > 0 and len(train_ids) > self.n_valid_scans:
                valid_ids = tuple(train_ids[-self.n_valid_scans :])
                train_ids = train_ids[: -self.n_valid_scans]
                train_end = max(dates[scan_id] for scan_id in train_ids)

            split = Split(
                kind=SplitKind.TIME_ORDERED,
                fold=fold,
                train_scan_ids=tuple(train_ids),
                valid_scan_ids=valid_ids,
                test_scan_ids=tuple(test_ids),
                train_end=train_end,
                test_start=min(dates[scan_id] for scan_id in test_ids),
                gap_days=self.gap_days,
            )
            assert_no_temporal_leakage(split, dates)
            splits.append(split)

        if not splits:
            raise ConfigError(
                f"no usable time-ordered fold over {n} scans with gap_days={self.gap_days} "
                f"and min_train_scans={self.min_train_scans}"
            )
        return splits


@dataclass
class LeaveOneAppOutSplitter(Splitter):
    """One fold per application: transfer, not time (Gap 8).

    Separation here is by application, so the date fields are bookkeeping rather than a
    claim: ``train_end`` is the last training scan's date and ``test_start`` is clamped
    to it, because the frozen ``Split`` contract requires ``test_start >= train_end`` and
    a held-out application's scans naturally overlap the training period in time. Set
    ``time_ordered=True`` to additionally require that the held-out application's test
    scans come after the training window, which is the strict (and much more demanding)
    reading of transfer.
    """

    min_train_scans: int = 1
    time_ordered: bool = False

    def split(self, scans: list[Scan], labels: LabelSet | None = None) -> list[Split]:
        dates = scan_dates(scans)
        app_ids = sorted({scan.app_id for scan in scans})
        if len(app_ids) < 2:
            raise ConfigError("leave-one-application-out needs at least two applications")

        splits: list[Split] = []
        for fold, held_out in enumerate(app_ids):
            train_ids = [scan.scan_id for scan in scans if scan.app_id != held_out]
            test_ids = [scan.scan_id for scan in scans if scan.app_id == held_out]
            if len(train_ids) < self.min_train_scans or not test_ids:
                continue
            train_end = max(dates[scan_id] for scan_id in train_ids)
            if self.time_ordered:
                test_ids = [scan_id for scan_id in test_ids if dates[scan_id] >= train_end]
                if not test_ids:
                    continue
            test_start = max(train_end, min(dates[scan_id] for scan_id in test_ids))
            splits.append(
                Split(
                    kind=SplitKind.LEAVE_ONE_APP_OUT,
                    fold=fold,
                    train_scan_ids=tuple(sorted(train_ids)),
                    test_scan_ids=tuple(sorted(test_ids)),
                    train_end=train_end,
                    test_start=test_start,
                    gap_days=0,
                    held_out_app_id=held_out,
                )
            )
        if not splits:
            raise ConfigError("no usable leave-one-application-out fold")
        return splits


@dataclass
class RandomSplitter(Splitter):
    """K-fold over shuffled scans. **The unrealistic control condition, never a result.**

    Random splitting is the design most of the surveyed literature uses, and it is wrong
    for this problem in two compounding ways: a scan from June can train a model tested
    on a scan from January (the model sees exploitation evidence that did not yet exist),
    and consecutive scans of the *same* application differ by a handful of findings, so
    near-duplicates land on both sides of the split.

    It exists here so the report can print one number: how much performance the naive
    protocol overstates. Every ``Split`` it produces is stamped ``SplitKind.RANDOM`` so
    a downstream consumer cannot mistake it for the real thing, and ``assert_no_temporal_leakage``
    is deliberately *not* called - leakage is the property being demonstrated.
    """

    n_folds: int = 3
    seed: int = 42

    def split(self, scans: list[Scan], labels: LabelSet | None = None) -> list[Split]:
        if self.n_folds < 2:
            raise ConfigError("random control needs at least two folds")
        dates = scan_dates(scans)
        ids = np.array(sorted(scan.scan_id for scan in scans), dtype=object)
        if len(ids) < self.n_folds:
            raise ConfigError(f"{len(ids)} scans cannot support {self.n_folds} random folds")
        rng = np.random.default_rng(self.seed)
        shuffled = ids[rng.permutation(len(ids))]
        folds = np.array_split(shuffled, self.n_folds)

        splits: list[Split] = []
        for fold, test_block in enumerate(folds):
            test_ids = [str(scan_id) for scan_id in test_block]
            train_ids = [str(scan_id) for scan_id in shuffled if str(scan_id) not in set(test_ids)]
            if not train_ids or not test_ids:
                continue
            # The date fields are meaningless for a random split; they are set to the
            # last training date so the frozen contract validates, and the SplitKind is
            # what tells a reader the fold carries no temporal guarantee.
            train_end = max(dates[scan_id] for scan_id in train_ids)
            splits.append(
                Split(
                    kind=SplitKind.RANDOM,
                    fold=fold,
                    train_scan_ids=tuple(sorted(train_ids)),
                    test_scan_ids=tuple(sorted(test_ids)),
                    train_end=train_end,
                    test_start=train_end,
                    gap_days=0,
                )
            )
        return splits


def build_splitter(config: EvaluationConfig, seed: int = 42) -> Splitter:
    """The splitter named by ``EvaluationConfig.split_kind``."""
    if config.split_kind == SplitKind.TIME_ORDERED:
        return TimeOrderedSplitter(
            n_folds=config.n_folds,
            gap_days=config.gap_days,
            min_train_scans=config.min_train_scans,
        )
    if config.split_kind == SplitKind.LEAVE_ONE_APP_OUT:
        return LeaveOneAppOutSplitter()
    return RandomSplitter(n_folds=config.n_folds, seed=seed)
