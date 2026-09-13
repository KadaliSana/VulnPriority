"""Query construction, URL handling and deduplication against the feeds.

Offline and deterministic: nothing here opens a socket or needs a key.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from vulnprio.core.enums import (
    ExploitMaturity,
    ExploitSource,
    Provenance,
    ScannerSeverity,
)
from vulnprio.core.models import (
    AffectedProduct,
    ExploitEvidence,
    Finding,
    ReferenceDoc,
    TechComponent,
    UntrustedText,
    VulnIntel,
)
from vulnprio.intel.models import IntelConfig, IntelSourceKind
from vulnprio.intel.provider import make_document
from vulnprio.intel.queries import (
    build_queries,
    classify_source,
    dedupe_documents,
    known_urls,
    normalize_url,
    phase1_instruction,
    phase2_context,
    product_terms,
)

AS_OF = date(2024, 6, 1)


def make_finding(
    *,
    cve_ids: tuple[str, ...] = ("CVE-2024-0001",),
    cwe_id: int | None = 89,
    component: TechComponent | None = None,
) -> Finding:
    return Finding(
        finding_id="f1",
        scan_id="scan_1",
        app_id="app1",
        endpoint_id="ep1",
        name="SQL Injection",
        cwe_id=cwe_id,
        cve_ids=cve_ids,
        scanner="zap",
        scanner_severity=ScannerSeverity.HIGH,
        scanner_confidence=0.9,
        description=UntrustedText(
            text="SQL injection in the username parameter",
            provenance=Provenance.SCANNER_OUTPUT,
        ),
        affected_component=component,
        observed_at=datetime(2024, 5, 1, 9, 0, 0),
    )


def make_intel(
    *,
    references: tuple[str, ...] = (),
    exploit_urls: tuple[str, ...] = (),
) -> VulnIntel:
    return VulnIntel(
        cve_id="CVE-2024-0001",
        as_of=AS_OF,
        references=tuple(
            ReferenceDoc(
                url=url,
                content=UntrustedText(text="advisory body", provenance=Provenance.REFERENCE_PAGE),
            )
            for url in references
        ),
        exploits=tuple(
            ExploitEvidence(
                source=ExploitSource.EXPLOIT_DB,
                url=url,
                published=date(2024, 1, 20),
                maturity=ExploitMaturity.POC,
            )
            for url in exploit_urls
        ),
    )


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def test_cve_finding_gets_targeted_exploit_queries() -> None:
    queries = build_queries(make_finding(), (), IntelConfig())
    texts = [query.text for query in queries]
    assert "CVE-2024-0001 exploit" in texts
    assert "CVE-2024-0001 proof of concept github" in texts
    assert "CVE-2024-0001 exploited in the wild" in texts
    assert all(query.cve_id == "CVE-2024-0001" for query in queries if query.cve_id)
    assert all(query.finding_id == "f1" for query in queries)


def test_product_and_version_reach_the_query() -> None:
    component = TechComponent(vendor="apache", product="struts", version="2.5.12")
    finding = make_finding(cve_ids=(), component=component)
    texts = " | ".join(query.text for query in build_queries(finding, (), IntelConfig()))
    assert "apache struts 2.5.12" in texts
    assert product_terms(finding) == "apache struts 2.5.12"


def test_cve_less_finding_still_gets_a_plan() -> None:
    """Most web application findings carry no CVE; the layer must still search."""
    finding = make_finding(cve_ids=(), cwe_id=79, component=None)
    queries = build_queries(finding, (), IntelConfig())
    assert queries, "a CVE-less finding must still produce searches"
    assert any("cross site scripting" in query.text for query in queries)


def test_unknown_cwe_falls_back_to_the_identifier() -> None:
    finding = make_finding(cve_ids=(), cwe_id=9999)
    texts = " ".join(query.text for query in build_queries(finding, (), IntelConfig()))
    assert "CWE-9999" in texts


def test_queries_are_deduplicated_and_capped() -> None:
    finding = make_finding(cve_ids=("CVE-2024-0001", "CVE-2024-0001"))
    queries = build_queries(finding, (), IntelConfig(max_queries=3))
    assert len(queries) == 3
    assert len({query.key for query in queries}) == 3


def test_cve_comes_from_intel_when_the_finding_has_none() -> None:
    queries = build_queries(make_finding(cve_ids=()), (make_intel(),), IntelConfig())
    assert any(query.cve_id == "CVE-2024-0001" for query in queries)


# ---------------------------------------------------------------------------
# URLs and deduplication against what the feeds already had
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,right",
    [
        ("https://WWW.Example.com/a/", "http://example.com/a"),
        ("https://example.com/a?utm_source=x", "https://example.com/a"),
        ("https://example.com/a#frag", "https://example.com/a"),
    ],
)
def test_normalize_url_collapses_equivalent_forms(left: str, right: str) -> None:
    assert normalize_url(left) == normalize_url(right)


def test_known_urls_covers_references_and_exploit_records() -> None:
    intel = make_intel(
        references=("https://nvd.nist.gov/vuln/detail/CVE-2024-0001",),
        exploit_urls=("https://www.exploit-db.com/exploits/51999",),
    )
    known = known_urls((intel,))
    assert normalize_url("https://nvd.nist.gov/vuln/detail/CVE-2024-0001") in known
    assert normalize_url("https://www.exploit-db.com/exploits/51999") in known


def test_search_results_the_feed_already_supplied_are_dropped() -> None:
    """The search must add reach, not re-read pages the reference fetcher already has."""
    intel = make_intel(references=("https://nvd.nist.gov/vuln/detail/CVE-2024-0001",))
    documents = [
        make_document("https://nvd.nist.gov/vuln/detail/CVE-2024-0001", "already fetched"),
        make_document("https://github.com/example/poc", "new material"),
    ]
    kept = dedupe_documents(documents, known_urls((intel,)))
    assert [document.url for document in kept] == ["https://github.com/example/poc"]


def test_duplicate_results_within_one_search_are_dropped() -> None:
    documents = [
        make_document("https://github.com/example/poc", "one"),
        make_document("https://github.com/example/poc/", "same page, trailing slash"),
    ]
    assert len(dedupe_documents(documents)) == 1


def test_dedupe_respects_the_limit() -> None:
    documents = [make_document(f"https://example.com/{index}", "x") for index in range(10)]
    assert len(dedupe_documents(documents, (), limit=4)) == 4


# ---------------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/example/poc", IntelSourceKind.POC),
        ("https://www.exploit-db.com/exploits/51999", IntelSourceKind.POC),
        ("https://nvd.nist.gov/vuln/detail/CVE-2024-0001", IntelSourceKind.ADVISORY),
        ("https://www.cisa.gov/alert", IntelSourceKind.ADVISORY),
        ("https://msrc.microsoft.com/update-guide", IntelSourceKind.VENDOR),
        ("https://news.ycombinator.com/item?id=1", IntelSourceKind.SOCIAL),
        ("https://some-blog.example/post", IntelSourceKind.WRITEUP),
    ],
)
def test_classify_source(url: str, expected: IntelSourceKind) -> None:
    assert classify_source(url) == expected


def test_a_title_cannot_promote_a_page_to_advisory() -> None:
    """Claiming to be an advisory is exactly what a hostile page would do."""
    kind = classify_source("https://attacker.example/page", "Official CISA Advisory")
    assert kind is not IntelSourceKind.ADVISORY


# ---------------------------------------------------------------------------
# Operator context blocks
# ---------------------------------------------------------------------------


def test_phase1_instruction_names_what_the_feeds_already_read() -> None:
    intel = make_intel(references=("https://nvd.nist.gov/vuln/detail/CVE-2024-0001",))
    finding = make_finding()
    text = phase1_instruction(
        finding, (intel,), build_queries(finding, (intel,), IntelConfig()), known_urls((intel,))
    )
    assert "nvd.nist.gov/vuln/detail/CVE-2024-0001" in text
    assert "do not spend a search" in text


def test_phase2_context_states_the_curated_facts_as_facts() -> None:
    intel = VulnIntel(
        cve_id="CVE-2024-0001",
        as_of=AS_OF,
        affected=(AffectedProduct(cpe="cpe:2.3:a:example:struts_commerce:*"),),
    )
    documents = [make_document("https://github.com/example/poc", "body")]
    context = phase2_context(make_finding(), (intel,), documents)
    assert "feed_kev:" in context
    assert "feed_epss:" in context
    assert "retrieved_documents: 1" in context
    # Untrusted prose never appears in the operator block.
    assert "body" not in context
