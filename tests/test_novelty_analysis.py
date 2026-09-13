"""The comparison, its arithmetic, and the honesty properties the verdict must hold.

Three of these tests exist specifically to catch a flattering analysis rather than a broken
one:

* ``test_not_every_capability_is_unique`` - a novelty suite in which everything comes out novel
  is a suite that is not checking anything.
* ``test_verdict_names_what_is_not_novel`` - the corpus contains learning to rank, EPSS/KEV
  grounding, asset context, attack-graph position and SHAP explanation, so the verdict has to
  say so.
* ``test_nearest_neighbours_are_strong_prior_work_not_surveys`` - comparing against the weakest
  prior work is the oldest trick in the related-work section.
"""

from __future__ import annotations

import json

import pytest

from vulnpriority.novelty import (
    CAPABILITY_KEYS,
    CapabilityLevel,
    CapabilityNature,
    ClaimLevel,
    EvidenceStrength,
    capability,
    capability_matrix,
    framework_capabilities,
    framework_partial_capabilities,
    load_corpus,
    nearest_neighbours,
    novelty_payload,
    novelty_verdict,
    rare_capabilities,
    shared_capabilities,
    unique_capabilities,
)
from vulnpriority.novelty.analysis import _narrative  # noqa: PLC2701 - narrative is derived output


def _clear_caches() -> None:
    """Force recomputation so 'stable across runs' means more than 'cached'."""
    for fn in (capability_matrix, unique_capabilities, shared_capabilities, novelty_verdict):
        fn.cache_clear()


# --------------------------------------------------------------------------------- matrix


def test_matrix_covers_every_dimension() -> None:
    matrix = capability_matrix()
    assert list(matrix) == list(CAPABILITY_KEYS)


def test_matrix_counts_sum_to_the_corpus_size() -> None:
    corpus_size = len(load_corpus())
    for key, row in capability_matrix().items():
        assert row.total == corpus_size, key
        assert row.counts_sum() == corpus_size, key


def test_matrix_study_lists_match_their_counts_and_partition_the_corpus() -> None:
    corpus_keys = set(load_corpus().keys())
    for key, row in capability_matrix().items():
        assert len(row.has_studies) == row.has_count, key
        assert len(row.partial_studies) == row.partial_count, key
        assert len(row.lacks_studies) == row.lacks_count, key
        assert len(row.unknown_studies) == row.unknown_count, key
        listed = (
            set(row.has_studies)
            | set(row.partial_studies)
            | set(row.lacks_studies)
            | set(row.unknown_studies)
        )
        assert listed == corpus_keys, key
        assert len(row.has_studies) == len(set(row.has_studies))
        assert list(row.has_studies) == sorted(row.has_studies), key


def test_matrix_agrees_with_the_raw_corpus() -> None:
    """Spot-check the counting against the source data rather than against itself."""
    corpus = load_corpus()
    for key in CAPABILITY_KEYS:
        expected = sorted(s.key for s in corpus.studies if s.has(key))
        assert list(capability_matrix()[key].has_studies) == expected, key


# ------------------------------------------------------------------------------ uniqueness


def test_unique_capabilities_are_consistent_with_the_matrix() -> None:
    matrix = capability_matrix()
    unique = {u.dimension for u in unique_capabilities()}
    for dimension in unique:
        row = matrix[dimension]
        assert row.has_count == 0, dimension
        assert row.partial_count == 0, dimension
        assert row.framework_position is CapabilityLevel.HAS, dimension
    # And nothing qualifying was left out.
    expected = {
        key
        for key in framework_capabilities()
        if matrix[key].has_count == 0 and matrix[key].partial_count == 0
    }
    assert unique == expected


def test_not_every_capability_is_unique() -> None:
    """A suite where everything is novel is a suite that is not checking."""
    unique = unique_capabilities()
    assert 0 < len(unique) < len(framework_capabilities())
    # A concrete non-unique example, so the assertion cannot be satisfied by an empty corpus.
    assert capability_matrix()["learning_to_rank_combined_evidence"].has_count >= 2


def test_unique_claims_report_what_they_were_checked_against() -> None:
    corpus_size = len(load_corpus())
    for claim in unique_capabilities():
        assert claim.checked_against, claim.dimension
        assert len(claim.checked_against) + len(claim.unknown_studies) == corpus_size
        assert claim.framework_evidence, claim.dimension


def test_rare_capabilities_are_monotone_in_the_threshold() -> None:
    tight = {r.dimension for r in rare_capabilities(0)}
    loose = {r.dimension for r in rare_capabilities(2)}
    widest = {r.dimension for r in rare_capabilities(len(load_corpus()))}
    assert tight <= loose <= widest
    assert widest == set(CAPABILITY_KEYS)
    for row in rare_capabilities(2):
        assert row.has_count <= 2


def test_rare_capabilities_rejects_a_negative_threshold() -> None:
    with pytest.raises(ValueError):
        rare_capabilities(-1)


# ---------------------------------------------------------------------- what is not novel


def test_shared_capabilities_name_the_prior_work_explicitly() -> None:
    shared = {s.dimension: s for s in shared_capabilities()}
    assert shared, "the framework cannot share nothing with 45 studies"
    # The capabilities the review shows are established, which the framework also has.
    for dimension in (
        "learning_to_rank_combined_evidence",
        "exploitation_evidence_grounding",
        "application_context_awareness",
        "multi_hop_directed_chain",
        "per_item_explainability",
    ):
        assert dimension in shared, dimension
        assert shared[dimension].has_count >= 2
        assert shared[dimension].has_studies
        assert "no novelty" in shared[dimension].note.lower()


def test_shared_and_unique_are_disjoint() -> None:
    assert not ({s.dimension for s in shared_capabilities()} & {u.dimension for u in unique_capabilities()})


# --------------------------------------------------------------------- nearest neighbours


def test_nearest_neighbours_are_strong_prior_work_not_surveys() -> None:
    corpus = load_corpus()
    top = nearest_neighbours(5)
    assert len(top) == 5
    for neighbour in top:
        assert not corpus.get(neighbour.key).survey, neighbour.key
    # Sorted by descending similarity.
    assert [n.jaccard for n in top] == sorted((n.jaccard for n in top), reverse=True)
    assert top[0].jaccard > 0.0
    # The strongest prior work in the corpus, by the review's own account of it.
    plausible = {
        "tita2026",
        "sevimlideniz2026",
        "hore2023",
        "shimizu2026",
        "zeng2024_illation",
        "parente2025",
        "jyoti2025",
        "zhao2011",
        "tchimwabouom2024",
    }
    assert top[0].key in plausible
    assert len({n.key for n in top} & plausible) >= 4


def test_nearest_neighbour_sets_are_coherent() -> None:
    mine = set(framework_capabilities())
    corpus = load_corpus()
    for neighbour in nearest_neighbours(10):
        theirs = set(corpus.get(neighbour.key).capability_set())
        assert set(neighbour.shared) == mine & theirs
        assert set(neighbour.framework_only) == mine - theirs
        assert set(neighbour.study_only) == theirs - mine
        assert neighbour.shared_count == len(mine & theirs)


def test_nearest_neighbours_rejects_a_non_positive_k() -> None:
    with pytest.raises(ValueError):
        nearest_neighbours(0)


# ------------------------------------------------------------------------------- verdict


def test_verdict_claim_levels_follow_the_stated_rules() -> None:
    matrix = capability_matrix()
    for claim in novelty_verdict().claims:
        row = matrix[claim.dimension]
        if row.has_count >= 2:
            expected = ClaimLevel.NOT_NOVEL
        elif row.has_count == 1:
            expected = ClaimLevel.NOVEL_COMBINATION
        elif row.partial_count >= 1:
            expected = ClaimLevel.INCREMENTAL
        else:
            expected = ClaimLevel.NOVEL
        assert claim.level is expected, claim.dimension
        assert claim.has_count == row.has_count
        assert claim.partial_count == row.partial_count


def test_verdict_claims_cover_every_dimension_exactly_once() -> None:
    claims = novelty_verdict().claims
    assert [c.dimension for c in claims] == list(CAPABILITY_KEYS)


def test_verdict_buckets_only_contain_capabilities_the_framework_actually_has() -> None:
    verdict = novelty_verdict()
    mine = set(framework_capabilities())
    buckets = (
        set(verdict.novel)
        | set(verdict.novel_combination)
        | set(verdict.incremental)
        | set(verdict.not_novel)
    )
    assert buckets == mine
    assert set(verdict.not_claimed) == set(framework_partial_capabilities())
    assert not buckets & set(verdict.not_claimed)


def test_verdict_names_what_is_not_novel() -> None:
    """The point of the exercise: prior work already does several of these things."""
    verdict = novelty_verdict()
    assert verdict.not_novel, "a verdict with nothing not-novel has not been checked"
    assert len(verdict.not_novel) >= 1
    assert "learning_to_rank_combined_evidence" in verdict.not_novel
    for dimension in verdict.not_novel:
        claim = next(c for c in verdict.claims if c.dimension == dimension)
        assert claim.prior_work_with, dimension
        assert "not novel" in claim.statement.lower()


def test_every_claim_names_the_studies_it_was_checked_against() -> None:
    corpus_keys = set(load_corpus().keys())
    for claim in novelty_verdict().claims:
        assert set(claim.prior_work_with) <= corpus_keys
        assert set(claim.prior_work_partial) <= corpus_keys
        assert claim.statement.strip()
        if claim.level is ClaimLevel.NOT_NOVEL:
            assert claim.prior_work_with
        if claim.level is ClaimLevel.NOVEL:
            assert not claim.prior_work_with and not claim.prior_work_partial


def test_verdict_has_substantive_threats_to_novelty() -> None:
    threats = novelty_verdict().threats_to_novelty
    assert len(threats) >= 3
    for threat in threats:
        assert len(threat) > 60, threat
    joined = " ".join(threats).lower()
    # The three a reviewer will raise first, all required to be present.
    assert "not about the literature" in joined              # corpus boundary, not the field
    assert "synthetic" in joined or "real deployment" in joined
    assert "proprietary" in joined or "tenable" in joined    # unassessed commercial systems


def test_novel_claims_say_where_the_advantage_is_engineering() -> None:
    verdict = novelty_verdict()
    for dimension in verdict.novel:
        claim = next(c for c in verdict.claims if c.dimension == dimension)
        if capability(dimension).nature is CapabilityNature.ENGINEERING:
            assert "engineering, not science" in claim.statement


def test_weakly_evidenced_novel_claims_generate_their_own_threat() -> None:
    verdict = novelty_verdict()
    joined = " ".join(verdict.threats_to_novelty)
    for claim in verdict.claims:
        if claim.level is ClaimLevel.NOVEL and claim.evidence_strength is EvidenceStrength.WEAK:
            assert claim.title in joined, claim.dimension


def test_single_precedent_claims_are_flagged_as_fragile() -> None:
    verdict = novelty_verdict()
    joined = " ".join(verdict.threats_to_novelty)
    for dimension in verdict.novel_combination:
        assert capability(dimension).title in joined, dimension


def test_overall_statement_is_quantified_not_promotional() -> None:
    verdict = novelty_verdict()
    statement = verdict.overall_statement
    assert str(verdict.corpus_size) in statement
    assert str(verdict.max_shared_capabilities) in statement
    assert verdict.strongest_prior_work in statement
    for word in ("state-of-the-art", "breakthrough", "revolutionary", "first-ever"):
        assert word not in statement.lower()


def test_max_shared_capabilities_matches_the_neighbour_table() -> None:
    verdict = novelty_verdict()
    neighbours = nearest_neighbours(len(load_corpus()))
    assert verdict.max_shared_capabilities == max(n.shared_count for n in neighbours)
    assert verdict.strongest_prior_work == neighbours[0].label
    assert verdict.max_shared_capabilities < verdict.framework_has


# ------------------------------------------------------------------------------- payload


def test_payload_is_json_serialisable() -> None:
    payload = novelty_payload()
    text = json.dumps(payload)
    assert len(text) > 10_000
    assert json.loads(text) == payload


def test_payload_is_stable_across_runs() -> None:
    first = json.dumps(novelty_payload(), sort_keys=True)
    _clear_caches()
    second = json.dumps(novelty_payload(), sort_keys=True)
    assert first == second


def test_payload_carries_what_the_site_needs() -> None:
    payload = novelty_payload()
    for key in (
        "matrix",
        "verdict",
        "nearest_neighbours",
        "narrative",
        "caveats",
        "studies",
        "capabilities",
        "unique_capabilities",
        "rare_capabilities",
        "shared_capabilities",
        "claim_rules",
    ):
        assert key in payload, key
    assert len(payload["studies"]) == len(load_corpus())
    assert len(payload["capabilities"]) == len(CAPABILITY_KEYS)
    assert len(payload["matrix"]) == len(CAPABILITY_KEYS)
    assert len(payload["nearest_neighbours"]) == 5
    assert payload["verdict"]["threats_to_novelty"]
    assert payload["caveats"]


def test_payload_contains_no_python_objects() -> None:
    """Enums and tuples must already be plain JSON types, not rely on a custom encoder."""

    def walk(node: object, path: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                assert isinstance(key, str), path
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        else:
            assert isinstance(node, (str, int, float, bool, type(None))), f"{path}: {type(node)}"

    walk(novelty_payload())


def test_narrative_states_both_halves_of_the_answer() -> None:
    lines = _narrative(novelty_verdict(), capability_matrix())
    assert len(lines) >= 3
    joined = " ".join(lines).lower()
    assert "unprecedented" in joined
    assert "established prior work" in joined
    assert "only partly satisfies" in joined
