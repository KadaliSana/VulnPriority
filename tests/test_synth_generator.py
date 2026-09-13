"""The synthetic world: determinism, shape, and the non-circularity it exists to provide.

The load-bearing assertions here are the last two groups. A generated dataset is only worth
anything to the evaluation protocol if (a) the same seed reproduces it byte for byte, so a
published number can be regenerated, and (b) the exploitation oracle is *not* a function of
the evidence the framework reads, so a ranker cannot score well by echoing its own features
back. Everything else in this file is shape checking in service of those two.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import date, timedelta
from pathlib import Path

import pytest

from vulnpriority.core.config import FeedsConfig, SyntheticConfig
from vulnpriority.core.enums import ExploitMaturity, PrivilegeLevel
from vulnpriority.core.models import Scan
from vulnpriority.feeds.base import clear_fixture_cache
from vulnpriority.feeds.bundle import DefaultIntelAssembler, build_fixture_bundle
from vulnpriority.ingest.generic import parse_scan
from vulnpriority.synth.generator import SyntheticDataset
from vulnpriority.synth.oracle import HAZARD_WEIGHTS
from vulnpriority.synth.pages import (
    generate_reference_pages,
    injected_page_urls,
    load_injection_payloads,
)
from vulnpriority.synth.topology import SECTOR_TECH, generate_app_specs, generate_endpoints
from vulnpriority.synth.world import build_world

SMALL = SyntheticConfig(
    seed=1234,
    n_apps=4,
    scans_per_app=3,
    endpoints_per_app=(12, 16),
    findings_per_app=(14, 22),
    non_english_fraction=0.3,
)


def _tree_digest(root: Path) -> dict[str, str]:
    """SHA-256 of every file under ``root``, keyed by its relative POSIX path."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


#: A wider world for the tests that are statistical rather than structural: with only four
#: applications the exploited-and-CVE-bearing subset is too small to say anything about.
WIDE = SyntheticConfig(seed=2024, n_apps=8, scans_per_app=3)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> SyntheticDataset:
    """One generated world, written to disk, shared by every test in this module."""
    target = tmp_path_factory.mktemp("synthetic")
    return SyntheticDataset.generate(SMALL, target)


@pytest.fixture(scope="module")
def wide_dataset() -> SyntheticDataset:
    """A larger world, in memory only, for the distributional assertions."""
    return SyntheticDataset.generate(WIDE, write=False)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_files(tmp_path: Path) -> None:
    """The same configuration writes byte-identical datasets in two separate directories."""
    first = SyntheticDataset.generate(SMALL, tmp_path / "a")
    second = SyntheticDataset.generate(SMALL, tmp_path / "b")

    assert first.dataset_hash == second.dataset_hash
    left, right = _tree_digest(first.root), _tree_digest(second.root)
    assert set(left) == set(right)
    differing = [name for name in left if left[name] != right[name]]
    assert differing == [], f"non-deterministic artifacts: {differing}"


def test_same_seed_produces_identical_objects() -> None:
    """Determinism is a property of the generator, not only of the JSON writer."""
    first = SyntheticDataset.generate(SMALL, write=False)
    second = SyntheticDataset.generate(SMALL, write=False)

    assert [scan.model_dump(mode="json") for scan in first.ordered_scans()] == [
        scan.model_dump(mode="json") for scan in second.ordered_scans()
    ]
    assert first.world.model_dump(mode="json") == second.world.model_dump(mode="json")
    assert first.oracle.model_dump(mode="json") == second.oracle.model_dump(mode="json")


def test_a_different_seed_produces_a_different_world() -> None:
    """A seed change must actually change the data, or 'seeded' means nothing."""
    other = SyntheticDataset.generate(SMALL.model_copy(update={"seed": SMALL.seed + 1}), write=False)
    baseline = SyntheticDataset.generate(SMALL, write=False)
    assert other.dataset_hash != baseline.dataset_hash
    assert other.oracle.positives() != baseline.oracle.positives()


def test_endpoint_generation_is_a_pure_function_of_the_generator() -> None:
    """``generate_endpoints`` draws only from the rng it is handed."""
    from random import Random

    specs = generate_app_specs(2, ("ecommerce", "fintech"), Random(7))
    first = generate_endpoints(specs[0], Random(11), SMALL)
    second = generate_endpoints(specs[0], Random(11), SMALL)
    assert [item.model_dump(mode="json") for item in first] == [
        item.model_dump(mode="json") for item in second
    ]


# ---------------------------------------------------------------------------
# Shape: applications, sectors, time
# ---------------------------------------------------------------------------


def test_applications_span_sectors_and_carry_versioned_stacks(dataset: SyntheticDataset) -> None:
    app_ids = {scan.app_id for scan in dataset.scans}
    assert len(app_ids) == SMALL.n_apps
    sectors = {scan.sector for scan in dataset.scans}
    assert sectors <= set(SMALL.sectors)
    assert len(sectors) >= 2, "leave-one-application-out transfer needs more than one sector"

    for scan in dataset.scans:
        assert scan.tech_stack, "applicability needs an observed stack to match against"
        assert all(component.version for component in scan.tech_stack)
        assert len(set(scan.hosts)) >= 2, "lateral movement needs more than one host"


def test_each_application_is_scanned_repeatedly_over_time(dataset: SyntheticDataset) -> None:
    """Repeated, staggered scans are what a time-ordered split with a gap cuts through."""
    by_app: dict[str, list[Scan]] = {}
    for scan in dataset.scans:
        by_app.setdefault(scan.app_id, []).append(scan)

    for app_id, scans in by_app.items():
        dates = sorted(scan.scanned_at for scan in scans)
        assert len(scans) == SMALL.scans_per_app, f"{app_id} was not scanned {SMALL.scans_per_app} times"
        assert len(set(dates)) == len(dates), f"{app_id} has two scans on the same timestamp"
        gaps = {(later - earlier).days for earlier, later in zip(dates, dates[1:])}
        assert gaps == {SMALL.scan_interval_days}

    all_dates = sorted({scan.scanned_at.date() for scan in dataset.scans})
    assert len(all_dates) >= SMALL.scans_per_app + 1, "application start dates are not staggered"
    span = (all_dates[-1] - all_dates[0]).days
    assert span > 30, "the scan window is too short for a 30-day train/test gap"


def test_the_surface_covers_every_structural_case(dataset: SyntheticDataset) -> None:
    """Public pages, an auth boundary, identifier segments, admin, API, upload, static."""
    scan = dataset.ordered_scans()[0]
    paths = {endpoint.path for endpoint in scan.endpoints}

    assert "/login" in paths
    assert any(path.startswith("/admin") for path in paths)
    assert any(path.startswith("/api/") for path in paths)
    assert "/upload" in paths
    assert "/search" in paths
    assert any(path.startswith("/static/") for path in paths)
    assert any("{id}" in path for path in paths), "no templated identifier segments"

    assert any(endpoint.sets_cookie for endpoint in scan.endpoints)
    assert {endpoint.auth_required for endpoint in scan.endpoints} >= {
        PrivilegeLevel.NONE,
        PrivilegeLevel.USER,
        PrivilegeLevel.ADMIN,
    }
    assert any(endpoint.links_to for endpoint in scan.endpoints), "no links_to edges"
    cross_host = {
        endpoint.endpoint_id: endpoint.host for endpoint in scan.endpoints
    }
    assert any(
        cross_host.get(target) != endpoint.host
        for endpoint in scan.endpoints
        for target in endpoint.links_to
    ), "no cross-host link, so the graph has no lateral edge to admit"


def test_findings_are_well_formed_and_uniquely_identified(dataset: SyntheticDataset) -> None:
    for scan in dataset.scans:
        ids = [finding.finding_id for finding in scan.findings]
        assert len(ids) == len(set(ids)), f"{scan.scan_id} has colliding finding ids"
        for finding in scan.findings:
            assert scan.endpoint_by_id(finding.endpoint_id) is not None
            assert finding.dedup_key, "ingest.correlate did not run"
            assert finding.cluster_size >= 1
            assert finding.description.text
    assert any(finding.cve_ids for scan in dataset.scans for finding in scan.findings)
    assert any(not finding.cve_ids for scan in dataset.scans for finding in scan.findings)


# ---------------------------------------------------------------------------
# Shape: the oracle
# ---------------------------------------------------------------------------


def test_oracle_positives_exist_but_are_a_minority(dataset: SyntheticDataset) -> None:
    """Gap 7 is about the rare class; a world where half the findings are exploited is not it."""
    oracle = dataset.oracle
    total = len(oracle.events)
    positives = len(oracle.positives())

    assert total == sum(len(scan.findings) for scan in dataset.scans)
    assert positives > 0, "the oracle produced no positives at all"
    rate = positives / total
    assert 0.0 < rate < 0.25, f"positive rate {rate:.3f} is not a minority class"


def test_every_exploitation_has_a_date_inside_the_horizon(dataset: SyntheticDataset) -> None:
    """Dates are measured from the defect's *first* observation, not from each re-observation.

    A defect found in January and exploited in March is still reported by the April scan;
    that finding's exploitation date is legitimately earlier than its own ``observed_at``.
    """
    oracle = dataset.oracle
    first_seen: dict[str, date] = {}
    for event in oracle.events:
        current = first_seen.get(event.defect_key)
        if current is None or event.observed_at < current:
            first_seen[event.defect_key] = event.observed_at

    for event in oracle.events:
        if not event.exploited:
            assert event.exploit_date is None
            continue
        origin = first_seen[event.defect_key]
        assert event.exploit_date is not None
        assert event.exploit_date >= origin
        assert (event.exploit_date - origin).days <= oracle.horizon_days


def test_one_defect_gets_one_exploitation_across_repeated_scans(dataset: SyntheticDataset) -> None:
    """A defect re-observed in three scans is one event in the world, not three draws."""
    by_defect: dict[str, set[tuple[bool, date | None]]] = {}
    for event in dataset.oracle.events:
        by_defect.setdefault(event.defect_key, set()).add((event.exploited, event.exploit_date))
    inconsistent = {key: values for key, values in by_defect.items() if len(values) > 1}
    assert inconsistent == {}
    assert any(
        sum(1 for event in dataset.oracle.events if event.defect_key == key) > 1
        for key in by_defect
    ), "no defect persisted across scans, so the longitudinal simulation has nothing to measure"


def test_counterfactual_answers_the_gap_10_question(dataset: SyntheticDataset) -> None:
    """'Would this have been exploited had we not fixed it by D' is answerable per finding."""
    oracle = dataset.oracle
    exploited = sorted(oracle.positives())
    assert exploited, "need at least one positive to exercise the counterfactual"

    finding_id = exploited[0]
    when = oracle.exploit_date(finding_id)
    assert when is not None

    # Remediated the day before: the exploitation was averted.
    assert oracle.would_be_exploited_if_not_remediated_by(finding_id, when - timedelta(days=1))
    # Remediated on the day it happened, or later: too late to have prevented anything.
    assert not oracle.would_be_exploited_if_not_remediated_by(finding_id, when)
    assert not oracle.would_be_exploited_if_not_remediated_by(finding_id, when + timedelta(days=30))
    # Never remediated: the world's own outcome stands.
    assert oracle.would_be_exploited_if_not_remediated_by(finding_id, None)
    # The DESIGN.md spelling is the same predicate.
    assert oracle.prevented_by_remediation(finding_id, when - timedelta(days=1))

    negative = next(event.finding_id for event in oracle.events if not event.exploited)
    assert not oracle.would_be_exploited_if_not_remediated_by(negative, None)
    assert oracle.exposure_days(negative, None, date(2030, 1, 1)) > 0


def test_oracle_label_map_covers_every_finding(dataset: SyntheticDataset) -> None:
    labels = dataset.oracle.label_map()
    findings = {finding.finding_id for scan in dataset.scans for finding in scan.findings}
    assert set(labels) == findings
    dates = dataset.oracle.first_evidence_dates()
    assert set(dates) == dataset.oracle.positives()


# ---------------------------------------------------------------------------
# Non-circularity: the oracle reads latents, the framework reads evidence
# ---------------------------------------------------------------------------


def test_the_hazard_model_names_only_latent_inputs() -> None:
    """A guard against someone quietly wiring an observable into the oracle."""
    forbidden = {
        "cvss",
        "epss",
        "kev",
        "exploit_maturity",
        "scanner_severity",
        "reach_delta",
        "chain_score",
        "expected_loss",
        "p_exploit",
    }
    for name in HAZARD_WEIGHTS:
        assert not any(marker in name for marker in forbidden), f"{name} is an observable"


def test_exploitation_is_not_a_function_of_the_published_evidence(
    wide_dataset: SyntheticDataset,
) -> None:
    """Positives are not simply 'the high-CVSS ones' or 'the KEV ones'.

    If they were, a CVSS-only baseline would be optimal by construction and the whole
    evaluation would be measuring the generator rather than the framework.
    """
    index = wide_dataset.world.index()
    rows = [
        (index[event.cve_id].cvss_nvd, event.exploited, event.finding_id, index[event.cve_id].in_kev)
        for event in wide_dataset.oracle.events
        if event.cve_id and event.cve_id in index
    ]
    assert rows and any(row[1] for row in rows) and any(not row[1] for row in rows)

    # 1. The latent hazard is only weakly correlated with the published severity.
    paired = [
        (event.hazard, index[event.cve_id].cvss_nvd)
        for event in wide_dataset.oracle.events
        if event.cve_id and event.cve_id in index
    ]
    mean_hazard = statistics.mean(value for value, _ in paired)
    mean_score = statistics.mean(value for _, value in paired)
    covariance = sum((h - mean_hazard) * (s - mean_score) for h, s in paired)
    spread = (
        sum((h - mean_hazard) ** 2 for h, _ in paired) * sum((s - mean_score) ** 2 for _, s in paired)
    ) ** 0.5
    correlation = covariance / spread
    assert abs(correlation) < 0.6, f"the oracle tracks CVSS too closely (r={correlation:.2f})"

    # 2. Ordering by CVSS alone puts the exploited findings nowhere near the top.
    rows.sort(key=lambda row: (-row[0], row[2]))
    normalised = [position / len(rows) for position, row in enumerate(rows) if row[1]]
    assert statistics.mean(normalised) > 0.05, "a CVSS-only ordering is near-optimal by construction"

    # 3. No CVSS threshold separates the classes: plenty of negatives outscore the positives.
    lowest_positive = min(row[0] for row in rows if row[1])
    negatives_above = sum(1 for row in rows if not row[1] and row[0] >= lowest_positive)
    assert negatives_above > 0.2 * sum(1 for row in rows if not row[1])

    # 4. KEV membership does not enumerate the positives either, and some exploited findings
    #    carry no CVE at all, so no feed-derived feature can reach them.
    kev_positive = sum(1 for row in rows if row[1] and row[3])
    assert kev_positive < sum(1 for row in rows if row[1])
    assert any(
        event.exploited and not event.cve_id for event in wide_dataset.oracle.events
    ), "every positive carries a CVE, so feed features would trivially cover them"


def test_observed_evidence_carries_realistic_disagreement_and_rarity(
    dataset: SyntheticDataset,
) -> None:
    world = dataset.world
    spreads = [abs(vuln.cvss_nvd - vuln.cvss_cna) for vuln in world.vulns]
    assert max(spreads) > 0.5, "NVD and the CNA never disagree, so source agreement is constant"

    kev_fraction = world.kev_fraction()
    assert 0.0 < kev_fraction < 0.35, f"KEV membership is not a minority ({kev_fraction:.2f})"

    with_exploits = [vuln for vuln in world.vulns if vuln.exploit_published is not None]
    assert with_exploits, "no exploit records at all"
    assert all(vuln.exploit_published > vuln.published for vuln in with_exploits), (
        "exploit records must appear after publication, not with it"
    )
    assert {vuln.exploit_maturity for vuln in with_exploits} & {
        ExploitMaturity.FUNCTIONAL,
        ExploitMaturity.WEAPONIZED,
    }

    assert any(vuln.version_applies for vuln in world.vulns)
    assert any(not vuln.version_applies for vuln in world.vulns), (
        "every CVE applies to the deployed version, so applicability has nothing to decide"
    )


def test_epss_is_correlated_with_latent_exploitability_but_not_equal_to_it() -> None:
    stacks = [entry for group in SECTOR_TECH.values() for entry in group]
    start = date(2024, 1, 15)
    snapshots = [start + timedelta(days=14 * step) for step in range(6)]
    world = build_world(96, stacks, 99, start_date=start, epss_dates=snapshots)

    latent = [vuln.true_exploitability for vuln in world.vulns]
    observed = [vuln.epss_anchor for vuln in world.vulns]
    mean_latent, mean_observed = statistics.mean(latent), statistics.mean(observed)
    covariance = sum((x - mean_latent) * (y - mean_observed) for x, y in zip(latent, observed))
    denominator = (
        sum((x - mean_latent) ** 2 for x in latent) * sum((y - mean_observed) ** 2 for y in observed)
    ) ** 0.5
    correlation = covariance / denominator

    assert 0.25 < correlation < 0.95, f"EPSS correlation with the truth is {correlation:.2f}"
    assert statistics.median(observed) < 0.2, "EPSS must stay right-skewed, as the real feed is"


# ---------------------------------------------------------------------------
# Feeds, pages and persistence
# ---------------------------------------------------------------------------


def test_fixtures_are_read_by_the_real_feed_readers(dataset: SyntheticDataset) -> None:
    """The generated feeds go through ``vulnpriority.feeds``, not through a synthetic shortcut."""
    clear_fixture_cache()
    config = FeedsConfig(fixture_dir=dataset.fixture_dir)
    assembler = DefaultIntelAssembler(build_fixture_bundle(config), config)

    kev_vuln = next(vuln for vuln in dataset.world.vulns if vuln.in_kev)
    late = max(scan.scanned_at.date() for scan in dataset.scans)
    intel = assembler.assemble(kev_vuln.cve_id, max(late, kev_vuln.kev_date_added or late))

    assert intel.cve_id == kev_vuln.cve_id
    assert len(intel.cvss) >= 2, "the CVSS selection policy needs competing records"
    assert {record.source.value for record in intel.cvss} >= {"nvd", "cna"}
    assert intel.epss is not None and intel.epss.as_of <= intel.as_of
    assert intel.kev is not None and intel.kev.in_kev
    assert intel.references, "reference pages were not retrievable through the fetcher"


def test_feeds_respect_the_as_of_cut_off(dataset: SyntheticDataset) -> None:
    """A KEV listing added later must be invisible at an earlier cut-off (no leakage)."""
    clear_fixture_cache()
    config = FeedsConfig(fixture_dir=dataset.fixture_dir)
    assembler = DefaultIntelAssembler(build_fixture_bundle(config), config)

    candidate = next(
        vuln
        for vuln in dataset.world.vulns
        if vuln.in_kev and vuln.kev_date_added is not None
    )
    before = candidate.kev_date_added - timedelta(days=1)
    early = assembler.assemble(candidate.cve_id, before)
    assert early.kev is not None and not early.kev.in_kev
    assert early.kev.date_added is None

    after = assembler.assemble(candidate.cve_id, candidate.kev_date_added)
    assert after.kev is not None and after.kev.in_kev


def test_reference_pages_are_multilingual_and_injectable(dataset: SyntheticDataset) -> None:
    """Gap 8 needs non-English evidence; the adversarial evaluation needs injected evidence."""
    languages = {str(page["language"]) for page in dataset.pages}
    assert "en" in languages
    assert len(languages) > 1, "non_english_fraction produced no non-English page"
    assert dataset.config.injection_fraction == 0.0
    assert injected_page_urls(dataset.pages) == (), "the default world must be clean"

    payloads = load_injection_payloads()
    assert payloads, "no injection payloads are available at all"

    clean = generate_reference_pages(dataset.world, seed=5, injection_fraction=0.0)
    assert injected_page_urls(clean, payloads) == ()

    injected = generate_reference_pages(dataset.world, seed=5, injection_fraction=1.0)
    assert len(injected_page_urls(injected, payloads)) == len(injected)

    half = generate_reference_pages(dataset.world, seed=5, injection_fraction=0.5)
    carried = len(injected_page_urls(half, payloads))
    assert 0 < carried < len(half), f"injection_fraction was not honoured ({carried}/{len(half)})"


def test_written_scans_parse_through_the_canonical_parser(dataset: SyntheticDataset) -> None:
    paths = dataset.scan_paths()
    assert paths and all(path.is_file() for path in paths)
    reparsed = parse_scan(paths[0])
    expected = dataset.ordered_scans()[0]
    assert reparsed.model_dump(mode="json") == expected.model_dump(mode="json")


def test_dataset_round_trips_through_disk(dataset: SyntheticDataset) -> None:
    reloaded = SyntheticDataset.load(dataset.root)
    assert reloaded.dataset_hash == dataset.dataset_hash
    assert reloaded.config == dataset.config
    assert [scan.scan_id for scan in reloaded.ordered_scans()] == [
        scan.scan_id for scan in dataset.ordered_scans()
    ]
    assert reloaded.oracle.model_dump(mode="json") == dataset.oracle.model_dump(mode="json")
    assert reloaded.world.model_dump(mode="json") == dataset.world.model_dump(mode="json")
    assert len(reloaded.latent_findings) == len(dataset.latent_findings)


def test_the_dataset_manifest_describes_what_was_written(dataset: SyntheticDataset) -> None:
    manifest = json.loads((dataset.root / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["dataset_hash"] == dataset.dataset_hash
    assert manifest["summary"]["scans"] == len(dataset.scans)
    assert len(manifest["scans"]) == len(dataset.scans)
    for entry in manifest["scans"]:
        assert (dataset.root / entry["path"]).is_file()
