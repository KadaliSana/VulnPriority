"""Offline feed behaviour: fixtures, the cache store, and the no-network guarantee.

The central assertion of this file is negative: with every httpx entry point replaced by a
function that raises, the whole fixture path still works. That is what "offline by default"
means operationally, and it is the only way to prove a feed did not quietly open a socket.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from vulnpriority.core.config import PROJECT_ROOT, FeedsConfig, PipelineConfig
from vulnpriority.core.enums import (
    ExploitMaturity,
    ExploitSource,
    FeedMode,
    Provenance,
    TrustTier,
)
from vulnpriority.core.errors import FeedUnavailableError, OfflineViolationError
from vulnpriority.feeds import (
    DefaultIntelAssembler,
    EpssFixtureFeed,
    ExploitFixtureFeed,
    FileCacheStore,
    KevFixtureFeed,
    NvdFixtureFeed,
    ReferenceFixtureFetcher,
    build_feed_bundle,
    build_fixture_bundle,
    clear_fixture_cache,
    detect_language,
    extract_title,
    html_to_text,
    is_allowed_host,
    load_fixture_records,
)

AS_OF = date(2024, 6, 1)
FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "feeds"

#: Every CVE the fixture corpus knows about.
FIXTURE_CVES = (
    "CVE-2024-0001",
    "CVE-2021-44228",
    "CVE-2017-5638",
    "CVE-2014-0160",
    "CVE-2019-11043",
    "CVE-2022-22965",
    "CVE-2020-1938",
    "CVE-2023-44487",
    "CVE-2018-11776",
    "CVE-2024-3400",
    "CVE-2024-27198",
    "CVE-2025-1234",
)


class _NetworkUsed(AssertionError):
    """Raised in place of any httpx call so an accidental fetch fails the test."""


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every httpx entry point with a landmine."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise _NetworkUsed("offline code path attempted to use httpx")

    for attribute in ("Client", "AsyncClient", "get", "post", "request", "stream", "HTTPTransport"):
        monkeypatch.setattr(httpx, attribute, _boom, raising=False)


# ---------------------------------------------------------------------------
# Fixture feeds under a disabled network
# ---------------------------------------------------------------------------


def test_fixture_corpus_has_twelve_cves(no_network: None) -> None:
    feed = NvdFixtureFeed(FIXTURE_DIR)
    assert set(feed.keys()) == set(FIXTURE_CVES)
    assert len(FIXTURE_CVES) == 12


def test_nvd_fixture_parses_real_nvd_shape(no_network: None) -> None:
    intel = NvdFixtureFeed(FIXTURE_DIR).get("CVE-2024-0001", AS_OF)
    assert intel is not None
    assert intel.published == date(2024, 1, 10)
    assert intel.description is not None
    assert intel.description.provenance == Provenance.NVD
    # three CVSS records: NVD 3.1, CNA 3.1, NVD 2.0
    assert len(intel.cvss) == 3
    assert {record.version.value for record in intel.cvss} == {"3.1", "2.0"}
    assert intel.cvss[0].submetrics["AC"] == "L"
    # configurations -> affected products with version bounds
    assert intel.affected[0].cpe.startswith("cpe:2.3:a:apache:struts")
    assert intel.affected[0].version_end_excluding == "2.5.22"
    assert len(intel.references) == 3


def test_nvd_fixture_is_case_insensitive_and_misses_cleanly(no_network: None) -> None:
    feed = NvdFixtureFeed(FIXTURE_DIR)
    assert feed.get("cve-2024-0001", AS_OF) is not None
    assert feed.get("CVE-1999-9999", AS_OF) is None


def test_epss_fixture_returns_latest_snapshot_at_or_before_as_of(no_network: None) -> None:
    record = EpssFixtureFeed(FIXTURE_DIR).get("CVE-2024-0001", AS_OF)
    assert record is not None
    assert record.as_of == date(2024, 5, 15)
    assert record.score == pytest.approx(0.4215)
    assert record.percentile == pytest.approx(0.9705)


def test_kev_fixture_returns_a_record_even_for_unlisted_cves(no_network: None) -> None:
    record = KevFixtureFeed(FIXTURE_DIR).get("CVE-2014-0160", AS_OF)
    assert record is not None
    assert record.in_kev is False
    assert record.date_added is None


def test_kev_fixture_reports_ransomware_use(no_network: None) -> None:
    record = KevFixtureFeed(FIXTURE_DIR).get("CVE-2021-44228", AS_OF)
    assert record.in_kev is True
    assert record.known_ransomware_use is True
    assert record.due_date == date(2021, 12, 24)


def test_exploit_fixture_classifies_maturity(no_network: None) -> None:
    feed = ExploitFixtureFeed(FIXTURE_DIR)
    log4shell = feed.get("CVE-2021-44228", AS_OF)
    assert len(log4shell) == 2
    by_source = {item.source: item for item in log4shell}
    assert by_source[ExploitSource.EXPLOIT_DB].maturity == ExploitMaturity.WEAPONIZED
    assert by_source[ExploitSource.GITHUB_POC].maturity == ExploitMaturity.POC
    assert by_source[ExploitSource.GITHUB_POC].verified is False
    heartbleed = feed.get("CVE-2014-0160", AS_OF)
    assert heartbleed[0].maturity == ExploitMaturity.POC
    php = feed.get("CVE-2019-11043", AS_OF)
    assert php[0].maturity == ExploitMaturity.FUNCTIONAL


def test_exploit_fixture_titles_are_untrusted(no_network: None) -> None:
    """An exploit title is free text written by whoever submitted the exploit.

    The record's structured fields are curated, but the title is not, so it must not carry a
    curated feed's unrestricted influence budget. Tagging it as a reference page caps what a
    hostile title can move, which is the property the adversarial corpus tests.
    """
    evidence = ExploitFixtureFeed(FIXTURE_DIR).get("CVE-2024-0001", AS_OF)
    assert evidence[0].title is not None
    assert evidence[0].title.provenance == Provenance.REFERENCE_PAGE
    assert evidence[0].title.tier == TrustTier.REFERENCE_PAGE
    assert evidence[0].title.tier > TrustTier.CURATED_FEED


def test_exploit_fixture_returns_empty_tuple_for_unknown_cve(no_network: None) -> None:
    assert ExploitFixtureFeed(FIXTURE_DIR).get("CVE-1999-9999", AS_OF) == ()


# ---------------------------------------------------------------------------
# Reference pages: untrusted, allowlisted, script-free
# ---------------------------------------------------------------------------


def test_reference_fixture_returns_untrusted_text(no_network: None) -> None:
    doc = ReferenceFixtureFetcher(FIXTURE_DIR).get(
        "https://nvd.nist.gov/vuln/detail/CVE-2024-0001", AS_OF
    )
    assert doc is not None
    assert doc.content.provenance == Provenance.REFERENCE_PAGE
    assert doc.content.tier == TrustTier.REFERENCE_PAGE
    assert doc.content.source_url == doc.url
    assert "Example Struts Commerce" in doc.content.text


def test_reference_fixture_strips_scripts_but_preserves_injection_text(no_network: None) -> None:
    """The fetcher removes executable markup; neutralising prose is the sandbox's job.

    If the fetcher silently deleted the injection, the adversarial corpus would have nothing
    to detect and the defence would look perfect for the wrong reason.
    """
    doc = ReferenceFixtureFetcher(FIXTURE_DIR).get(
        "https://github.com/example/struts-advisories/blob/main/CVE-2024-0001.md", AS_OF
    )
    assert doc is not None
    assert "window.analytics" not in doc.content.text
    assert "<script" not in doc.content.text
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in doc.content.text
    assert doc.content.tier == TrustTier.REFERENCE_PAGE


def test_reference_fixture_enforces_the_host_allowlist(no_network: None) -> None:
    fetcher = ReferenceFixtureFetcher(FIXTURE_DIR)
    # present in the fixture index, but the host is not allowlisted
    assert "https://malicious.example.net/cve-2024-0001" in fetcher.keys()
    assert fetcher.get("https://malicious.example.net/cve-2024-0001", AS_OF) is None


def test_reference_fixture_carries_non_english_pages(no_network: None) -> None:
    fetcher = ReferenceFixtureFetcher(FIXTURE_DIR)
    spanish = fetcher.get("https://owasp.org/es/log4shell-resumen", AS_OF)
    japanese = fetcher.get("https://github.com/ejemplo/log4shell-notas", AS_OF)
    assert spanish is not None and spanish.language == "es"
    assert japanese is not None and japanese.language == "ja"
    assert "ejecucion remota" in spanish.content.text
    assert "遠隔から任意のコード" in japanese.content.text


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://nvd.nist.gov/vuln/detail/CVE-2024-0001", True),
        ("http://github.com/x", True),
        ("https://GITHUB.COM/x", True),
        ("https://evil-github.com/x", False),
        ("https://github.com.attacker.net/x", False),
        ("ftp://github.com/x", False),
        ("file:///etc/passwd", False),
        ("not a url", False),
    ],
)
def test_is_allowed_host(url: str, expected: bool) -> None:
    assert is_allowed_host(url) is expected


# ---------------------------------------------------------------------------
# HTML to text
# ---------------------------------------------------------------------------


def test_html_to_text_removes_code_bearing_elements() -> None:
    markup = (
        "<html><head><title>Advisory &amp; notes</title>"
        "<script>alert('x')</script><style>p{color:red}</style></head>"
        "<body><p>First line</p><p>Second &lt;line&gt;</p>"
        "<iframe src='http://x'>frame</iframe></body></html>"
    )
    text = html_to_text(markup)
    assert "alert(" not in text
    assert "color:red" not in text
    assert "frame" not in text
    assert "First line" in text
    assert "Second <line>" in text
    assert extract_title(markup) == "Advisory & notes"


def test_html_to_text_handles_unclosed_script_and_comments() -> None:
    text = html_to_text("<p>keep</p><!-- hide me --><script>var a = 1;")
    assert "keep" in text
    assert "hide me" not in text
    assert "var a" not in text


def test_html_to_text_respects_the_character_budget() -> None:
    assert len(html_to_text("<p>" + "a" * 500 + "</p>", 100)) <= 100
    assert html_to_text(None) == ""
    assert extract_title("<p>no title</p>") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The remote attacker can exploit this vulnerability", "en"),
        ("La vulnerabilidad permite una ejecucion remota con los datos", "es"),
        ("この脆弱性により遠隔から任意のコードが実行されます", "ja"),
        ("", "und"),
        ("hi", "und"),
    ],
)
def test_detect_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


# ---------------------------------------------------------------------------
# Offline violations
# ---------------------------------------------------------------------------


def test_fixture_feeds_refuse_to_fetch(no_network: None) -> None:
    for feed in (
        NvdFixtureFeed(FIXTURE_DIR),
        EpssFixtureFeed(FIXTURE_DIR),
        KevFixtureFeed(FIXTURE_DIR),
        ExploitFixtureFeed(FIXTURE_DIR),
        ReferenceFixtureFetcher(FIXTURE_DIR),
    ):
        with pytest.raises(OfflineViolationError):
            feed.fetch("https://example.com/anything")
        with pytest.raises(OfflineViolationError):
            _ = feed.client
        assert feed.mode == FeedMode.OFFLINE
        feed.close()


def test_missing_fixture_file_is_an_explicit_error(tmp_path: Path, no_network: None) -> None:
    with pytest.raises(FeedUnavailableError):
        NvdFixtureFeed(tmp_path).get("CVE-2024-0001", AS_OF)


# ---------------------------------------------------------------------------
# Bundle assembly, still offline
# ---------------------------------------------------------------------------


def test_offline_bundle_assembles_every_fixture_cve(no_network: None) -> None:
    bundle = build_fixture_bundle(FeedsConfig(fixture_dir=FIXTURE_DIR))
    assembler = DefaultIntelAssembler(bundle, FeedsConfig(fixture_dir=FIXTURE_DIR))
    assert bundle.mode == FeedMode.OFFLINE
    for cve_id in FIXTURE_CVES:
        intel = assembler.assemble(cve_id, AS_OF)
        assert intel.cve_id == cve_id
        assert intel.as_of == AS_OF


def test_offline_bundle_from_pipeline_config(offline_config: PipelineConfig, no_network: None) -> None:
    bundle = build_feed_bundle(offline_config)
    assert bundle.mode == FeedMode.OFFLINE
    intel = DefaultIntelAssembler(bundle, offline_config).assemble("CVE-2021-44228", AS_OF)
    assert intel.kev is not None and intel.kev.in_kev is True
    assert intel.epss is not None and intel.epss.as_of <= AS_OF
    assert len(intel.exploits) == 2
    assert all(doc.content.tier == TrustTier.REFERENCE_PAGE for doc in intel.references)


def test_assembler_respects_the_reference_budget(no_network: None) -> None:
    config = FeedsConfig(fixture_dir=FIXTURE_DIR, max_references_per_cve=2)
    assembler = DefaultIntelAssembler(build_fixture_bundle(config), config)
    intel = assembler.assemble("CVE-2021-44228", AS_OF, max_references=8)
    assert len(intel.references) == 2
    assert assembler.assemble("CVE-2021-44228", AS_OF, max_references=0).references == ()


def test_assemble_many_preserves_order_and_dedupes(no_network: None) -> None:
    config = FeedsConfig(fixture_dir=FIXTURE_DIR)
    assembler = DefaultIntelAssembler(build_fixture_bundle(config), config)
    result = assembler.assemble_many(("CVE-2021-44228", "cve-2024-0001", "CVE-2021-44228"), AS_OF)
    assert [item.cve_id for item in result] == ["CVE-2021-44228", "CVE-2024-0001"]


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def test_fixture_file_is_parsed_once_per_path(no_network: None) -> None:
    clear_fixture_cache()
    first = load_fixture_records(FIXTURE_DIR / "kev" / "kev.json")
    second = load_fixture_records(FIXTURE_DIR / "kev" / "kev.json")
    assert first is second


def test_fixture_cache_notices_a_rewritten_file(tmp_path: Path, no_network: None) -> None:
    path = tmp_path / "kev.json"
    path.write_text(json.dumps({"vulnerabilities": []}), encoding="utf-8")
    assert load_fixture_records(path)[0]["vulnerabilities"] == []
    path.write_text(json.dumps({"vulnerabilities": [{"cveID": "CVE-2024-0001"}]}), encoding="utf-8")
    assert load_fixture_records(path)[0]["vulnerabilities"][0]["cveID"] == "CVE-2024-0001"


# ---------------------------------------------------------------------------
# FileCacheStore
# ---------------------------------------------------------------------------


def _frozen(moment: datetime):
    return lambda: moment


def test_cache_store_round_trip(tmp_path: Path) -> None:
    store = FileCacheStore(tmp_path, ttl_days=7, now=_frozen(datetime(2024, 6, 1, tzinfo=timezone.utc)))
    assert store.get("epss", "CVE-2024-0001", AS_OF) is None
    path = store.put("epss", "CVE-2024-0001", AS_OF, {"score": 0.42})
    assert path.is_file()
    assert store.get("epss", "CVE-2024-0001", AS_OF) == {"score": 0.42}
    assert len(store) == 1


def test_cache_keys_separate_feeds_dates_and_keys(tmp_path: Path) -> None:
    store = FileCacheStore(tmp_path, ttl_days=7)
    a = store.path_for("epss", "CVE-2024-0001", AS_OF)
    assert a != store.path_for("kev", "CVE-2024-0001", AS_OF)
    assert a != store.path_for("epss", "CVE-2024-0002", AS_OF)
    assert a != store.path_for("epss", "CVE-2024-0001", date(2024, 7, 1))
    # URLs are legal keys even though they are not legal Windows filenames
    url_path = store.path_for("references", "https://nvd.nist.gov/vuln/detail/CVE-2024-0001", AS_OF)
    assert url_path.suffix == ".json"


def test_cache_entries_expire(tmp_path: Path) -> None:
    written = datetime(2024, 6, 1, tzinfo=timezone.utc)
    store = FileCacheStore(tmp_path, ttl_days=2, now=_frozen(written))
    store.put("nvd", "CVE-2024-0001", AS_OF, {"x": 1})

    fresh = FileCacheStore(tmp_path, ttl_days=2, now=_frozen(written + timedelta(days=1)))
    assert fresh.get("nvd", "CVE-2024-0001", AS_OF) == {"x": 1}

    stale = FileCacheStore(tmp_path, ttl_days=2, now=_frozen(written + timedelta(days=3)))
    assert stale.get("nvd", "CVE-2024-0001", AS_OF) is None
    # the file survives; only the freshness verdict changed
    assert stale.read_entry("nvd", "CVE-2024-0001", AS_OF) is not None


def test_zero_ttl_disables_reuse(tmp_path: Path) -> None:
    store = FileCacheStore(tmp_path, ttl_days=0)
    store.put("nvd", "CVE-2024-0001", AS_OF, {"x": 1})
    assert store.get("nvd", "CVE-2024-0001", AS_OF) is None


def test_tampered_cache_entry_is_a_miss(tmp_path: Path) -> None:
    store = FileCacheStore(tmp_path, ttl_days=7)
    path = store.put("nvd", "CVE-2024-0001", AS_OF, {"score": 1.0})
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload"] = {"score": 10.0}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert store.get("nvd", "CVE-2024-0001", AS_OF) is None


def test_corrupt_cache_file_is_a_miss(tmp_path: Path) -> None:
    store = FileCacheStore(tmp_path, ttl_days=7)
    path = store.put("nvd", "CVE-2024-0001", AS_OF, {"score": 1.0})
    path.write_text("{not json", encoding="utf-8")
    assert store.get("nvd", "CVE-2024-0001", AS_OF) is None


def test_cache_invalidate_and_clear(tmp_path: Path) -> None:
    store = FileCacheStore(tmp_path, ttl_days=7)
    store.put("nvd", "CVE-2024-0001", AS_OF, {"a": 1})
    store.put("kev", "CVE-2024-0001", AS_OF, {"a": 2})
    assert store.invalidate("nvd", "CVE-2024-0001", AS_OF) is True
    assert store.invalidate("nvd", "CVE-2024-0001", AS_OF) is False
    assert store.clear() == 1
    assert len(store) == 0
