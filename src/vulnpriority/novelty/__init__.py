"""Novelty analysis: what in this framework is new, and what is not.

The question this package answers is "does the proposed solution have novelty over the
existing solutions", and it is built to answer it in a way that survives a reviewer rather
than in a way that flatters the framework. Three design choices follow from that:

* **The prior work is data.** ``data/novelty/prior_work.yaml`` holds all 45 studies from the
  literature review's Table 1, each with the review's own dataset, headline result and stated
  limitation, and a coding against 23 comparison dimensions. Conclusions are computed from it.
* **The dimensions carry tests.** :mod:`capabilities` states, for each dimension, what a paper
  must do to be coded as satisfying it - and codes this framework by the same test, strictly
  enough that three dimensions come out ``partial``.
* **The verdict reports what is not new.** :func:`shared_capabilities` names the capabilities
  prior work already had, and ``novelty_verdict().threats_to_novelty`` lists the concrete ways
  a reviewer could push back on every claim that remains.

Typical use::

    from vulnpriority.novelty import novelty_payload, novelty_verdict

    verdict = novelty_verdict()
    print(verdict.overall_statement)
    for threat in verdict.threats_to_novelty:
        print("-", threat)
"""

from __future__ import annotations

from vulnpriority.novelty.analysis import (
    DEFAULT_RARE_THRESHOLD,
    WEAK_EVIDENCE_UNKNOWN_FRACTION,
    CapabilityClaim,
    ClaimLevel,
    EvidenceStrength,
    MatrixRow,
    Neighbour,
    NoveltyVerdict,
    RareCapability,
    SharedCapability,
    UniqueCapability,
    capability_matrix,
    nearest_neighbours,
    novelty_payload,
    novelty_verdict,
    rare_capabilities,
    shared_capabilities,
    unique_capabilities,
)
from vulnpriority.novelty.capabilities import (
    CAPABILITIES,
    CAPABILITY_KEYS,
    Capability,
    CapabilityLevel,
    CapabilityNature,
    capability,
    framework_capabilities,
    framework_partial_capabilities,
    framework_vector,
)
from vulnpriority.novelty.corpus import (
    MIN_STUDIES,
    Corpus,
    CorpusCaveat,
    Paradigm,
    PriorStudy,
    StudyCaveat,
    corpus_caveats,
    default_corpus_path,
    load_corpus,
    studies_by_paradigm,
    study,
)

__all__ = [
    # capabilities
    "CAPABILITIES",
    "CAPABILITY_KEYS",
    "Capability",
    "CapabilityLevel",
    "CapabilityNature",
    "capability",
    "framework_capabilities",
    "framework_partial_capabilities",
    "framework_vector",
    # corpus
    "MIN_STUDIES",
    "Corpus",
    "CorpusCaveat",
    "Paradigm",
    "PriorStudy",
    "StudyCaveat",
    "corpus_caveats",
    "default_corpus_path",
    "load_corpus",
    "studies_by_paradigm",
    "study",
    # analysis
    "DEFAULT_RARE_THRESHOLD",
    "WEAK_EVIDENCE_UNKNOWN_FRACTION",
    "CapabilityClaim",
    "ClaimLevel",
    "EvidenceStrength",
    "MatrixRow",
    "Neighbour",
    "NoveltyVerdict",
    "RareCapability",
    "SharedCapability",
    "UniqueCapability",
    "capability_matrix",
    "nearest_neighbours",
    "novelty_payload",
    "novelty_verdict",
    "rare_capabilities",
    "shared_capabilities",
    "unique_capabilities",
]
