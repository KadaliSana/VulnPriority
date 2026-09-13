"""As-of correctness: the property the whole evaluation protocol rests on.

Every feed takes a cut-off date and must not return anything later. The cases that matter
are the ones a naive join gets wrong: a KEV entry added *after* the cut-off, an EPSS score
published later than the scan, a CVE that did not exist yet, and an exploit released after
the attacker's horizon. Each has a fixture record chosen to fail loudly if the guard breaks.
"""

from __future__ import annotations

from datetime import date

import pytest

from vulnpriority.core.config import PROJECT_ROOT, FeedsConfig
from vulnpriority.core.enums import FeedMode
from vulnpriority.core.errors import TemporalLeakageError
from vulnpriority.core.interfaces import EpssFeed, FeedBundle, KevFeed
from vulnpriority.core.models import EpssRecord, KevRecord, VulnIntel
from vulnpriority.feeds import (
    DefaultIntelAssembler,
    EpssFixtureFeed,
    ExploitFixtureFeed,
    KevFixtureFeed,
    NvdFixtureFeed,
    ReferenceFixtureFetcher,
    as_of_guard,
    build_fixture_bundle,
    parse_feed_date,
    parse_feed_datetime,
    require_not_future,
)

AS_OF = date(2024, 6, 1)
FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "feeds"
CONFIG = FeedsConfig(fixture_dir=FIXTURE_DIR)

#: EPSS fixture snapshot dates, in order.
SNAPSHOTS = (date(2024, 1, 15), date(2024, 3, 1), date(2024, 5, 15), date(2024, 7, 1))


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        ("", True),
        (date(2024, 5, 31), True),
        (date(2024, 6, 1), True),
        (date(2024, 6, 2), False),
        ("2024-05-31", True),
        ("2024-06-02", False),
        ("2024-06-02T09:00:00.000", False),
        ("2024-05-31T09:00:00.0000Z", True),
    ],
)
def test_as_of_guard(value: object, expected: bool) -> None:
    assert as_of_guard(value, AS_OF) is expected  # type: ignore[arg-type]


def test_require_not_future_raises() -> None:
    require_not_future(date(2024, 1, 1), AS_OF, "thing")
    with pytest.raises(TemporalLeakageError):
        require_not_future(date(2024, 12, 1), AS_OF, "thing")


def test_date_parsing_tolerates_upstream_shapes() -> None:
    assert parse_feed_date("2021-12-10T10:15:09.143") == date(2021, 12, 10)
    assert parse_feed_date("2024-09-10T14:00:00.0000Z") == date(2024, 9, 10)
    assert parse_feed_date("2024-09-10") == date(2024, 9, 10)
    assert parse_feed_date("nonsense") is None
    assert parse_feed_datetime(None) is None
    assert parse_feed_datetime("2024-09-10T14:00:00.0000Z") is not None


# ---------------------------------------------------------------------------
# EPSS: latest snapshot at or before the cut-off, never a later one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "as_of,expected",
    [
        (date(2024, 1, 1), None),
        (date(2024, 1, 15), SNAPSHOTS[0]),
        (date(2024, 2, 28), SNAPSHOTS[0]),
        (date(2024, 3, 1), SNAPSHOTS[1]),
        (date(2024, 5, 14), SNAPSHOTS[1]),
        (date(2024, 6, 1), SNAPSHOTS[2]),
        (date(2024, 6, 30), SNAPSHOTS[2]),
        (date(2024, 7, 1), SNAPSHOTS[3]),
        (date(2025, 1, 1), SNAPSHOTS[3]),
    ],
)
def test_epss_picks_the_latest_snapshot_not_after_as_of(as_of: date, expected: date | None) -> None:
    record = EpssFixtureFeed(FIXTURE_DIR).get("CVE-2024-0001", as_of)
    if expected is None:
        assert record is None
    else:
        assert record is not None and record.as_of == expected


def test_epss_never_returns_a_future_snapshot_for_any_fixture_cve() -> None:
    feed = EpssFixtureFeed(FIXTURE_DIR)
    for cve_id in feed.keys():
        for as_of in (date(2024, 2, 1), AS_OF, date(2024, 8, 1)):
            record = feed.get(cve_id, as_of)
            assert record is None or record.as_of <= as_of


def test_epss_score_actually_changes_across_snapshots() -> None:
    """A feed that ignored as_of would return the same value at every cut-off."""
    feed = EpssFixtureFeed(FIXTURE_DIR)
    scores = [feed.get("CVE-2024-3400", as_of).score for as_of in SNAPSHOTS]
    assert scores[0] < scores[1] < scores[2] < scores[3]
    assert scores[0] < 0.01 < 0.9 < scores[2]


# ---------------------------------------------------------------------------
# KEV: membership is dated, and the future is invisible
# ---------------------------------------------------------------------------


def test_kev_entry_added_after_as_of_is_not_in_kev() -> None:
    """CVE-2024-3400 is in the fixture catalogue with dateAdded 2024-09-10."""
    feed = KevFixtureFeed(FIXTURE_DIR)
    before = feed.get("CVE-2024-3400", AS_OF)
    assert before.in_kev is False
    assert before.date_added is None
    assert before.due_date is None
    assert before.known_ransomware_use is False
    assert before.as_of == AS_OF


def test_kev_entry_becomes_visible_once_the_cut_off_passes() -> None:
    after = KevFixtureFeed(FIXTURE_DIR).get("CVE-2024-3400", date(2024, 10, 1))
    assert after.in_kev is True
    assert after.date_added == date(2024, 9, 10)
    assert after.known_ransomware_use is True


@pytest.mark.parametrize(
    "as_of,expected",
    [
        (date(2024, 1, 31), False),
        (date(2024, 2, 1), True),
        (date(2024, 6, 1), True),
    ],
)
def test_kev_membership_turns_on_exactly_at_date_added(as_of: date, expected: bool) -> None:
    assert KevFixtureFeed(FIXTURE_DIR).get("CVE-2024-0001", as_of).in_kev is expected


def test_kev_never_reports_a_date_added_after_as_of() -> None:
    feed = KevFixtureFeed(FIXTURE_DIR)
    for cve_id in feed.keys():
        for as_of in (date(2021, 1, 1), date(2022, 6, 1), AS_OF, date(2025, 1, 1)):
            record = feed.get(cve_id, as_of)
            assert record.date_added is None or record.date_added <= as_of
            assert record.in_kev == (record.date_added is not None)


# ---------------------------------------------------------------------------
# NVD: a CVE published after the cut-off did not exist yet
# ---------------------------------------------------------------------------


def test_nvd_hides_a_cve_published_after_as_of() -> None:
    feed = NvdFixtureFeed(FIXTURE_DIR)
    assert feed.get("CVE-2025-1234", AS_OF) is None
    later = feed.get("CVE-2025-1234", date(2025, 3, 1))
    assert later is not None and later.published == date(2025, 1, 20)


def test_nvd_drops_a_last_modified_that_post_dates_as_of() -> None:
    intel = NvdFixtureFeed(FIXTURE_DIR).get("CVE-2021-44228", date(2022, 1, 1))
    assert intel is not None
    assert intel.published == date(2021, 12, 10)
    assert intel.last_modified is None  # fixture says 2023-11-07


# ---------------------------------------------------------------------------
# Exploit evidence and reference pages
# ---------------------------------------------------------------------------


def test_exploit_published_after_as_of_is_excluded() -> None:
    feed = ExploitFixtureFeed(FIXTURE_DIR)
    assert feed.get("CVE-2023-44487", AS_OF) == ()
    later = feed.get("CVE-2023-44487", date(2024, 9, 1))
    assert len(later) == 1 and later[0].published == date(2024, 8, 1)


def test_no_exploit_evidence_ever_post_dates_as_of() -> None:
    feed = ExploitFixtureFeed(FIXTURE_DIR)
    for cve_id in feed.keys():
        for as_of in (date(2018, 1, 1), AS_OF, date(2025, 1, 1)):
            for evidence in feed.get(cve_id, as_of):
                assert evidence.published is None or evidence.published <= as_of


def test_reference_fetched_after_as_of_is_excluded() -> None:
    fetcher = ReferenceFixtureFetcher(FIXTURE_DIR)
    url = "https://portswigger.net/daily-swig/spring4shell"
    assert fetcher.get(url, AS_OF) is None
    assert fetcher.get(url, date(2024, 9, 1)) is not None


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _assembler() -> DefaultIntelAssembler:
    return DefaultIntelAssembler(build_fixture_bundle(CONFIG), CONFIG)


def test_assembly_is_internally_consistent_across_cut_offs() -> None:
    assembler = _assembler()
    for as_of in (date(2022, 1, 1), date(2023, 6, 1), AS_OF, date(2025, 3, 1)):
        for cve_id in NvdFixtureFeed(FIXTURE_DIR).keys():
            intel = assembler.assemble(cve_id, as_of)
            assert intel.as_of == as_of
            assert intel.published is None or intel.published <= as_of
            assert intel.last_modified is None or intel.last_modified <= as_of
            assert intel.epss is None or intel.epss.as_of <= as_of
            assert intel.kev is None or not intel.kev.in_kev or intel.kev.date_added <= as_of
            assert all(item.published is None or item.published <= as_of for item in intel.exploits)


def test_assembled_pan_os_intel_hides_its_later_kev_membership() -> None:
    intel = _assembler().assemble("CVE-2024-3400", AS_OF)
    assert intel.kev is not None and intel.kev.in_kev is False
    assert len(intel.exploits) == 1  # the 2024-04-16 entry is admissible
    assert intel.epss is not None and intel.epss.score > 0.9


def test_assembly_of_an_unknown_cve_is_an_empty_but_valid_intel() -> None:
    intel = _assembler().assemble("CVE-1999-9999", AS_OF)
    assert isinstance(intel, VulnIntel)
    assert intel.cvss == () and intel.epss is None and intel.exploits == ()
    assert intel.kev is not None and intel.kev.in_kev is False


# --- leaking collaborators --------------------------------------------------


class _FutureEpssFeed(EpssFeed):
    """A deliberately broken feed: answers with a snapshot from after the cut-off."""

    mode = FeedMode.OFFLINE

    def get(self, key: str, as_of: date) -> EpssRecord:
        return EpssRecord(cve_id=key, score=0.9, percentile=0.99, as_of=date(2025, 1, 1))


class _FutureKevFeed(KevFeed):
    """A deliberately broken feed: claims KEV membership dated after the cut-off."""

    mode = FeedMode.OFFLINE

    def get(self, key: str, as_of: date) -> KevRecord:
        return KevRecord(cve_id=key, in_kev=True, date_added=date(2025, 1, 1), as_of=as_of)


def _bundle_with(**overrides: object) -> FeedBundle:
    bundle = build_fixture_bundle(CONFIG)
    return FeedBundle(
        nvd=overrides.get("nvd", bundle.nvd),
        epss=overrides.get("epss", bundle.epss),
        kev=overrides.get("kev", bundle.kev),
        exploits=overrides.get("exploits", bundle.exploits),
        references=overrides.get("references", bundle.references),
        mode=FeedMode.OFFLINE,
    )


def test_assembly_rejects_a_feed_that_leaks_a_future_epss_score() -> None:
    assembler = DefaultIntelAssembler(_bundle_with(epss=_FutureEpssFeed()), CONFIG)
    with pytest.raises(TemporalLeakageError):
        assembler.assemble("CVE-2024-0001", AS_OF)


def test_assembly_rejects_a_feed_that_leaks_future_kev_membership() -> None:
    assembler = DefaultIntelAssembler(_bundle_with(kev=_FutureKevFeed()), CONFIG)
    with pytest.raises(TemporalLeakageError):
        assembler.assemble("CVE-2024-0001", AS_OF)


def test_leaking_feeds_are_fine_once_the_cut_off_moves_past_them() -> None:
    assembler = DefaultIntelAssembler(_bundle_with(epss=_FutureEpssFeed(), kev=_FutureKevFeed()), CONFIG)
    intel = assembler.assemble("CVE-2024-0001", date(2025, 6, 1))
    assert intel.epss is not None and intel.epss.as_of == date(2025, 1, 1)
    assert intel.kev is not None and intel.kev.in_kev is True
