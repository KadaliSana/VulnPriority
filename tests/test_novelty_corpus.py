"""Integrity of the prior-work corpus and of the capability dimensions.

The corpus is evidence, so it is tested like evidence. Three classes of failure matter:

* **Structural** - a missing study, a duplicate key, a capability vector that does not cover
  every dimension. These make the matrix arithmetic silently wrong.
* **Factual** - a result attributed to a study that the literature review does not contain.
  A handful of the review's most-cited figures are spot-checked verbatim, because a novelty
  analysis built on invented numbers is worse than no analysis at all.
* **Self-serving** - a capability with no test, no rationale, or an evidence pointer that does
  not exist on disk; or a framework vector with no ``partial`` in it, which would mean the
  coding was never applied strictly to the framework itself.
"""

from __future__ import annotations

import re

import pytest

from vulnpriority.core.config import PROJECT_ROOT
from vulnpriority.core.errors import ConfigError
from vulnpriority.novelty.capabilities import (
    CAPABILITIES,
    CAPABILITY_KEYS,
    CapabilityLevel,
    CapabilityNature,
    capability,
    framework_capabilities,
    framework_partial_capabilities,
    framework_vector,
)
from vulnpriority.novelty.corpus import (
    MIN_STUDIES,
    Paradigm,
    corpus_caveats,
    default_corpus_path,
    load_corpus,
    studies_by_paradigm,
    study,
)

#: Figures taken verbatim from the literature review. If a coding drifts away from the source,
#: this is where it is caught.
REVIEW_SPOT_CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Howland: CVSS v3 correlates with weaponisation at rho = 0.099.
    ("howland2023", ("0.099", "0.180", "28,779")),
    # Walkowski: NVD and CNA agree on severity category only 65.9% of the time.
    ("walkowski2026", ("65.9%", "2.32%", "1.57", "297,780")),
    # Shimizu & Hashimoto: 9.1% efficiency against 0.5% for CVSS alone, at 85.6% coverage.
    ("shimizu2026", ("9.1%", "0.5%", "85.6%", "96.9%", "48")),
    # Sevimli Deniz & Koca: ROC-AUC 0.941 against 0.589 for a CVSS-only baseline.
    ("sevimlideniz2026", ("0.941", "0.589", "0.90", "0.656", "0.921")),
    # Tita et al.: 4.4-fold structural risk reduction over EPSS-only ranking.
    ("tita2026", ("4.4-fold", "3.4%", "0.0%", "75-97%")),
    # Zeng et al. LICALITY: AUC 0.926 and a 2.89-fold workload reduction.
    ("zeng2022_licality", ("0.926", "2.89-fold", "4.97%")),
    # Farghaly et al.: F1 0% for ChatGPT on HIGH attack complexity.
    ("farghaly2025", ("F1 0%", "15.38%", "93.3%", "46.7%")),
    # Hore et al.: 52-week simulation, 92% skill matching.
    ("hore2023", ("52-week", "92%")),
    # Le et al.: 2 of 84 interpretability, 3 of 84 adversarial robustness.
    ("le2022", ("2 of 84", "3 of 84")),
)


def test_corpus_loads_and_is_large_enough() -> None:
    corpus = load_corpus()
    assert len(corpus) >= MIN_STUDIES
    # The review's Table 1 lists 45 distinct studies once Sheng et al., counted twice there,
    # is counted once.
    assert len(corpus) == 45


def test_study_keys_are_unique_and_well_formed() -> None:
    keys = load_corpus().keys()
    assert len(set(keys)) == len(keys)
    for key in keys:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", key), key


def test_every_capability_vector_covers_every_known_dimension() -> None:
    known = set(CAPABILITY_KEYS)
    for entry in load_corpus().studies:
        assert set(entry.capabilities) == known, entry.key


def test_capability_values_are_from_the_allowed_set() -> None:
    allowed = set(CapabilityLevel)
    for entry in load_corpus().studies:
        for dimension, level in entry.capabilities.items():
            assert level in allowed, (entry.key, dimension, level)


def test_every_study_carries_the_fields_a_reader_needs_to_check_the_coding() -> None:
    for entry in load_corpus().studies:
        assert entry.description.strip()
        assert entry.dataset.strip()
        assert entry.headline_result.strip()
        assert entry.limitation.strip()
        assert entry.venue.strip()
        assert entry.authors.strip()


@pytest.mark.parametrize("key,fragments", REVIEW_SPOT_CHECKS)
def test_headline_results_match_the_review(key: str, fragments: tuple[str, ...]) -> None:
    """No study may cite a result the review does not contain."""
    entry = study(key)
    recorded = f"{entry.dataset} || {entry.headline_result}"
    for fragment in fragments:
        assert fragment in recorded, (
            f"{key}: expected {fragment!r} in the recorded dataset or headline result, got "
            f"{recorded!r}"
        )


def test_references_stay_inside_the_reviews_bibliography() -> None:
    """Guards against inventing a study: every reference is one the review actually cites.

    The reviewed corpus is [25]-[67] plus Spring et al. [6] and Hore et al. [15], which the
    review cites earlier but lists in Table 1 as reviewed studies.
    """
    allowed = set(range(25, 68)) | {6, 15}
    seen: set[int] = set()
    for entry in load_corpus().studies:
        match = re.fullmatch(r"\[(\d+)\]", entry.reference)
        assert match, f"{entry.key}: malformed reference {entry.reference!r}"
        number = int(match.group(1))
        assert number in allowed, f"{entry.key}: reference [{number}] is not a reviewed study"
        assert number not in seen, f"reference [{number}] used twice"
        seen.add(number)
    assert seen == allowed, f"missing reviewed studies: {sorted(allowed - seen)}"


def test_paradigm_distribution_matches_the_reviews_table_1() -> None:
    """Table 1's counts, with Sheng et al. classified once as a survey rather than twice."""
    groups = studies_by_paradigm()
    assert len(groups[Paradigm.STATISTICAL.value]) == 11
    assert len(groups[Paradigm.MACHINE_LEARNING.value]) == 6
    assert len(groups[Paradigm.HYBRID.value]) == 13
    assert len(groups[Paradigm.GRAPH.value]) == 5
    # Deep learning is 9 in Table 1 including Sheng et al.; here Sheng et al. is a survey.
    assert len(groups[Paradigm.DEEP_LEARNING.value]) + len(groups[Paradigm.SURVEY.value]) == 10


def test_surveys_are_flagged_and_coded_unknown_throughout() -> None:
    """A secondary study has no method, so it cannot satisfy or fail a capability test."""
    surveys = [s for s in load_corpus().studies if s.survey]
    assert len(surveys) == 2
    for entry in surveys:
        assert set(entry.capabilities.values()) == {CapabilityLevel.UNKNOWN}, entry.key


def test_corpus_declares_its_own_limitations() -> None:
    caveats = corpus_caveats()
    assert len(caveats) >= 5
    joined = " ".join(caveats).lower()
    # The three that a reviewer will reach for first.
    assert "not about the literature" in joined            # corpus boundary
    assert "secondary" in joined or "primary papers" in joined
    assert "proprietary" in joined or "tenable" in joined  # commercial systems unassessed


def test_study_caveats_attach_to_real_dimensions() -> None:
    known = set(CAPABILITY_KEYS)
    total = 0
    for entry in load_corpus().studies:
        for caveat in entry.caveats:
            assert caveat.dimension in known
            assert caveat.note.strip()
            total += 1
    assert total >= 20, "too few coding caveats to make the judgement calls auditable"


def test_missing_corpus_file_is_a_config_error(tmp_path) -> None:
    with pytest.raises(ConfigError):
        load_corpus(tmp_path / "nope.yaml")


def test_malformed_corpus_is_a_config_error(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\nstudies: []\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_corpus(bad)


def test_corpus_rejects_an_unknown_capability_dimension(tmp_path) -> None:
    """A typo in a dimension name must fail loudly, not vanish into an unread key."""
    import yaml

    raw = yaml.safe_load(default_corpus_path().read_text(encoding="utf-8"))
    raw["studies"][0]["capabilities"]["quantum_resistance"] = "has"
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown dimension"):
        load_corpus(path)


def test_corpus_rejects_a_missing_capability_dimension(tmp_path) -> None:
    import yaml

    raw = yaml.safe_load(default_corpus_path().read_text(encoding="utf-8"))
    raw["studies"][0]["capabilities"].pop(CAPABILITY_KEYS[0])
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing codings"):
        load_corpus(path)


# ------------------------------------------------------------------ capability definitions


def test_dimension_keys_are_unique() -> None:
    assert len(set(CAPABILITY_KEYS)) == len(CAPABILITY_KEYS)
    assert len(CAPABILITY_KEYS) >= 23


@pytest.mark.parametrize("cap", CAPABILITIES, ids=[c.key for c in CAPABILITIES])
def test_every_capability_has_a_definition_a_test_and_a_rationale(cap) -> None:
    assert len(cap.definition) > 60, cap.key
    assert len(cap.test) > 40, cap.key
    assert len(cap.rationale) > 60, cap.key
    assert cap.title.strip()
    assert cap.review_anchor.strip(), f"{cap.key}: no anchor back into the review"


@pytest.mark.parametrize("cap", CAPABILITIES, ids=[c.key for c in CAPABILITIES])
def test_every_capability_names_evidence_that_exists(cap) -> None:
    """A claim about this framework points at a module and a test, and both are real files."""
    assert cap.framework_modules, cap.key
    assert cap.framework_tests, cap.key
    for relative in cap.framework_modules + cap.framework_tests:
        assert (PROJECT_ROOT / relative).is_file(), f"{cap.key}: missing evidence {relative}"


def test_partial_positions_explain_themselves() -> None:
    """A 'partial' with no explanation is an unfalsifiable hedge."""
    for cap in CAPABILITIES:
        if cap.framework_position is CapabilityLevel.PARTIAL:
            assert "PARTIAL" in cap.framework_note, cap.key
            assert len(cap.framework_note) > 120, cap.key


def test_the_framework_is_coded_strictly_against_its_own_tests() -> None:
    """If nothing came out partial, the coding was not applied to the framework at all."""
    vector = framework_vector()
    assert set(vector) == set(CAPABILITY_KEYS)
    assert len(framework_partial_capabilities()) >= 2
    assert len(framework_capabilities()) < len(CAPABILITY_KEYS)
    assert CapabilityLevel.UNKNOWN not in vector.values(), (
        "the framework's own capabilities cannot be unknown to its authors"
    )


def test_engineering_and_scientific_claims_are_distinguished() -> None:
    """The verdict must be able to say where the advantage is build work, not research."""
    natures = {cap.nature for cap in CAPABILITIES}
    assert CapabilityNature.ENGINEERING in natures
    assert CapabilityNature.SCIENTIFIC in natures
    assert capability("untrusted_evidence_containment").nature is CapabilityNature.ENGINEERING
