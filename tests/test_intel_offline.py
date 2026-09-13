"""The fixture corpus, the offline provider and the recorder.

These tests are the offline-by-default claim made checkable: the shipped corpus must
cover the four cases the design calls for, every document it serves must be untrusted
reference-page text, and nothing here may need a key or a socket.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vulnpriority.core.config import PROJECT_ROOT
from vulnpriority.core.enums import Provenance, TrustTier
from vulnpriority.intel.models import (
    IntelCitation,
    IntelConfig,
    IntelGather,
    IntelQuery,
    IntelQueryKind,
    IntelSourceKind,
    IntelUsage,
)
from vulnpriority.intel.offline import (
    FIXTURE_VERSION,
    FixtureSearchProvider,
    RecordingProvider,
    fixture_key_for,
    load_fixture,
)
from vulnpriority.intel.provider import BaseSearchProvider, NullSearchProvider, make_document

FIXTURE = PROJECT_ROOT / "data" / "fixtures" / "intel" / "searches.json"


def query_for(cve: str) -> IntelQuery:
    return IntelQuery(
        text=f"{cve} exploit", kind=IntelQueryKind.POC, cve_id=cve, finding_id="f1"
    )


@pytest.fixture
def provider() -> FixtureSearchProvider:
    return FixtureSearchProvider(FIXTURE)


# ---------------------------------------------------------------------------
# The shipped corpus
# ---------------------------------------------------------------------------


def test_corpus_exists_and_declares_its_version() -> None:
    data = load_fixture(FIXTURE)
    assert data.get("version") == FIXTURE_VERSION
    assert isinstance(data.get("entries"), dict)


def test_corpus_covers_the_four_designed_cases(provider: FixtureSearchProvider) -> None:
    """A public PoC, active exploitation, nothing found, and an injected page."""
    config = IntelConfig()
    poc = provider.gather([query_for("CVE-2024-0001")], config)
    exploited = provider.gather([query_for("CVE-2021-44228")], config)
    nothing = provider.gather([query_for("CVE-2025-1234")], config)
    injected = provider.gather([query_for("CVE-2017-5638")], config)

    assert any(document.source_kind is IntelSourceKind.POC for document in poc.documents)
    assert "proof of concept" in poc.narrative.lower()

    assert "exploited" in exploited.narrative.lower()
    assert any("cisa.gov" in document.url for document in exploited.documents)

    assert nothing.documents == ()
    assert nothing.narrative == ""

    injected_text = "\n".join(document.snippet.text for document in injected.documents)
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in injected_text


def test_every_served_document_is_untrusted_reference_text(
    provider: FixtureSearchProvider,
) -> None:
    config = IntelConfig()
    for key in provider.keys():
        query = IntelQuery(text="x", cve_id=key, finding_id="f1")
        for document in provider.gather([query], config).documents:
            assert document.snippet.provenance is Provenance.REFERENCE_PAGE
            assert document.tier is TrustTier.REFERENCE_PAGE
            assert document.retrieved_at is not None


def test_documents_carry_a_retrieval_timestamp(provider: FixtureSearchProvider) -> None:
    """Provenance in time is the whole basis of the as-of guard."""
    gathered = provider.gather([query_for("CVE-2021-44228")], IntelConfig())
    for document in gathered.documents:
        assert document.retrieved_at.year >= 2024
        assert document.snippet.fetched_at == document.retrieved_at


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------


def test_unknown_cve_is_an_empty_result_not_an_error(provider: FixtureSearchProvider) -> None:
    gathered = provider.gather([query_for("CVE-1999-9999")], IntelConfig())
    assert gathered.documents == ()
    assert any("no recorded intel" in message for message in gathered.errors)


def test_search_returns_the_documents_of_gather(provider: FixtureSearchProvider) -> None:
    config = IntelConfig()
    queries = [query_for("CVE-2021-44228")]
    assert provider.search(queries, config) == list(provider.gather(queries, config).documents)


def test_snippet_char_budget_is_enforced(provider: FixtureSearchProvider) -> None:
    gathered = provider.gather([query_for("CVE-2021-44228")], IntelConfig(snippet_char_budget=200))
    assert gathered.documents
    assert all(len(document.snippet.text) <= 200 for document in gathered.documents)


def test_max_documents_is_enforced(provider: FixtureSearchProvider) -> None:
    gathered = provider.gather([query_for("CVE-2021-44228")], IntelConfig(max_documents=1))
    assert len(gathered.documents) == 1


def test_a_missing_corpus_is_an_empty_result(tmp_path: Path) -> None:
    empty = FixtureSearchProvider(tmp_path / "absent.json")
    assert empty.available() is False
    gathered = empty.gather([query_for("CVE-2024-0001")], IntelConfig())
    assert gathered.documents == ()
    assert gathered.errors


def test_strict_mode_raises_on_a_missing_corpus(tmp_path: Path) -> None:
    strict = FixtureSearchProvider(tmp_path / "absent.json", strict=True)
    with pytest.raises(FileNotFoundError):
        strict.gather([query_for("CVE-2024-0001")], IntelConfig())


def test_a_corpus_from_a_future_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "searches.json"
    path.write_text(json.dumps({"version": "99", "entries": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_fixture(path)


def test_fixture_key_falls_back_to_the_finding(tmp_path: Path) -> None:
    assert fixture_key_for(IntelQuery(text="x", cve_id="cve-2024-0001")) == "CVE-2024-0001"
    assert fixture_key_for(IntelQuery(text="x", finding_id="f9")) == "finding:f9"


def test_null_provider_reports_its_reason() -> None:
    null = NullSearchProvider("no key configured")
    assert null.available() is False
    gathered = null.gather([query_for("CVE-2024-0001")], IntelConfig())
    assert gathered.errors == ("no key configured",)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class _StubProvider(BaseSearchProvider):
    """A provider that returns a fixed session, standing in for a live one."""

    name = "stub"

    def __init__(self, gathered: IntelGather) -> None:
        self.gathered = gathered

    def gather(self, queries, config, *, instruction: str = "") -> IntelGather:
        return self.gathered


def _session() -> IntelGather:
    stamp = datetime(2024, 5, 24, 12, 0, tzinfo=timezone.utc)
    return IntelGather(
        documents=(
            make_document(
                "https://github.com/example/new-poc",
                "A working exploit for the issue.",
                title="example/new-poc",
                retrieved_at=stamp,
                source_kind=IntelSourceKind.POC,
            ),
        ),
        narrative="A public exploit exists.",
        citations=(
            IntelCitation(
                url="https://github.com/example/new-poc", cited_text="A working exploit"
            ),
        ),
        usage=IntelUsage(input_tokens=100, output_tokens=20, calls=1),
        model="claude-opus-5",
        provider="stub",
    )


def test_recording_writes_a_replayable_entry(tmp_path: Path) -> None:
    path = tmp_path / "searches.json"
    recorder = RecordingProvider(_StubProvider(_session()), path)
    recorder.gather([query_for("CVE-2030-0001")], IntelConfig())

    replayed = FixtureSearchProvider(path).gather([query_for("CVE-2030-0001")], IntelConfig())
    assert [document.url for document in replayed.documents] == [
        "https://github.com/example/new-poc"
    ]
    assert replayed.narrative == "A public exploit exists."
    assert replayed.citations[0].url == "https://github.com/example/new-poc"
    assert replayed.model == "claude-opus-5"


def test_recording_merges_rather_than_replaces(tmp_path: Path) -> None:
    """Recording the same CVE twice must grow the entry, not overwrite it."""
    path = tmp_path / "searches.json"
    RecordingProvider(_StubProvider(_session()), path).gather(
        [query_for("CVE-2030-0001")], IntelConfig()
    )

    second = _session().model_copy(
        update={
            "documents": (
                make_document("https://www.exploit-db.com/exploits/1", "Another entry."),
            )
        }
    )
    RecordingProvider(_StubProvider(second), path).gather(
        [query_for("CVE-2030-0001")], IntelConfig()
    )

    urls = {
        document.url
        for document in FixtureSearchProvider(path)
        .gather([query_for("CVE-2030-0001")], IntelConfig())
        .documents
    }
    assert urls == {
        "https://github.com/example/new-poc",
        "https://www.exploit-db.com/exploits/1",
    }


def test_recording_an_empty_session_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "searches.json"
    RecordingProvider(_StubProvider(IntelGather()), path).gather(
        [query_for("CVE-2030-0002")], IntelConfig()
    )
    assert not path.exists()
