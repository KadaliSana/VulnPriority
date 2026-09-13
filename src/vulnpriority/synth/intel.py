"""Synthetic retrieved intelligence, so the ranker can learn to use the real thing.

Component A's ``a_intel_*`` features describe what the intelligence agent read on the
internet about a finding: how many documents it retrieved, whether any named a public
exploit, whether the sources claim active exploitation, how confident the extraction was,
whether it agreed with the curated feeds, and how much attempted prompt injection arrived
attached to the evidence.

In an offline run none of that happens, so all seven columns sit at their neutral and are
*constant* across the corpus. A tree model cannot split on a constant, so a ranker trained
that way never learns to use them - and then a live run, which does gather intelligence,
hands the model seven features it has no opinion about. The framework's whole
search-and-summarise layer would move the queue not at all. That is a train/serve mismatch
rather than a missing nicety, which is why the synthetic world has to produce this layer
alongside the CVSS, EPSS and KEV records it already produces.

**Derived from the latents, never from the label.** Exactly like the feed records, an
intelligence result here is a noisy observation of ``true_exploitability`` and
``true_attacker_interest``. It is therefore informative - the ranker can learn to weigh it -
without being the answer, and the oracle remains the only thing that knows the outcome. A
fraction of findings get no intelligence at all, because real gathering often finds nothing,
and that fraction must stay distinguishable from "gathering never ran" only by whether it
happened, not by the feature values it produces.
"""

from __future__ import annotations

from datetime import date, datetime, time
from random import Random
from typing import Iterable, Mapping

from vulnpriority.core.enums import (
    ExploitMaturity,
    AgreementAxis,
    IntelSourceKind,
    Provenance,
    TrustTier,
)
from vulnpriority.core.models import (
    ExploitIntelOut,
    FeedAgreement,
    IntelDocument,
    IntelResult,
    UntrustedText,
)
from vulnpriority.synth.world import clamp01

__all__ = ["INTEL_COVERAGE", "synthesise_intel", "attach_intel"]

#: Fraction of CVE-bearing findings that intelligence gathering returns anything for. The
#: rest get an empty result, which is what a real search that found nothing produces.
INTEL_COVERAGE = 0.62

#: Fraction of results carrying at least one injection attempt in the retrieved text. Real
#: reference pages do contain them, and the feature exists so the ranker can learn to
#: discount evidence that arrived with manipulation attached.
INJECTION_RATE = 0.08


def _documents(rng: Random, count: int, cve_id: str, when: date) -> tuple[IntelDocument, ...]:
    """Placeholder documents. Only their number and kind reach the feature layer."""
    kinds = [
        IntelSourceKind.ADVISORY,
        IntelSourceKind.POC,
        IntelSourceKind.VENDOR,
        IntelSourceKind.WRITEUP,
    ]
    stamp = datetime.combine(when, time(9, 0))
    return tuple(
        IntelDocument(
            url=f"https://example.invalid/{cve_id.lower()}/{index}",
            title=f"{cve_id} analysis {index}",
            # Retrieved text is untrusted by construction, and the synthetic corpus has
            # to obey that too: a snippet that arrived in a trusted wrapper would train
            # the sandbox layer on a world it never sees.
            snippet=UntrustedText(
                text=f"Synthetic reference material for {cve_id}.",
                provenance=Provenance.REFERENCE_PAGE,
                source_url=f"https://example.invalid/{cve_id.lower()}/{index}",
            ),
            retrieved_at=stamp,
            source_kind=kinds[index % len(kinds)],
            relevance=round(clamp01(rng.betavariate(5, 2)), 3),
        )
        for index in range(count)
    )


def synthesise_intel(
    finding_id: str,
    cve_id: str,
    as_of: date,
    *,
    true_exploitability: float,
    true_attacker_interest: float,
    in_kev: bool,
    seed: int,
) -> IntelResult | None:
    """One finding's worth of retrieved intelligence, or ``None`` when nothing was found.

    The latents drive *how much* there is to find and *what it says*, with noise at every
    step, so the mapping from evidence back to the hazard is learnable but not invertible.
    """
    rng = Random(seed)
    if rng.random() > INTEL_COVERAGE:
        return None

    # A widely exploited, widely discussed defect has more written about it. The Poisson-ish
    # draw keeps the count over-dispersed rather than a clean function of the latent.
    appetite = clamp01(0.55 * true_exploitability + 0.45 * true_attacker_interest)
    n_documents = max(1, int(rng.gauss(1.0 + 7.0 * appetite, 1.6)))
    n_documents = min(n_documents, 14)

    # Public proof-of-concept code tracks exploitability, with a floor: plenty of easy bugs
    # have nothing written for them, and the occasional obscure one has three.
    p_exploit_code = clamp01(0.85 * true_exploitability - 0.1 + rng.gauss(0.0, 0.12))
    n_exploit_urls = sum(1 for _ in range(3) if rng.random() < p_exploit_code)

    # An "actively exploited" claim is the loudest signal a page can carry, so it is the one
    # most worth making noisy: it agrees with KEV most of the time and not always.
    p_active = clamp01(0.72 * true_attacker_interest + (0.25 if in_kev else 0.0))
    active = rng.random() < p_active

    confidence = clamp01(rng.betavariate(2 + 4 * appetite, 3))

    # Agreement with the curated feeds. Contradiction is rare and is what the trust ledger
    # exists to arbitrate; both being true at once is impossible by construction.
    corroborates = active == in_kev and rng.random() < 0.8
    contradicts = (not corroborates) and rng.random() < 0.15

    injection_signals = 0
    if rng.random() < INJECTION_RATE:
        injection_signals = rng.randint(1, 3)

    extraction = ExploitIntelOut(
        exploit_maturity=(
            ExploitMaturity.WEAPONIZED if active and n_exploit_urls
            else ExploitMaturity.FUNCTIONAL if n_exploit_urls
            else ExploitMaturity.POC if n_documents > 3
            else ExploitMaturity.UNPROVEN
        ),
        exploit_feasibility=round(clamp01(true_exploitability + rng.gauss(0.0, 0.15)), 3),
        active_exploitation_claimed=active,
        confidence=round(confidence, 3),
        rationale="Synthetic extraction derived from the latent world.",
    )

    return IntelResult(
        finding_id=finding_id,
        cve_id=cve_id,
        as_of=as_of,
        documents=_documents(rng, n_documents, cve_id, as_of),
        extraction=extraction,
        agreement=FeedAgreement(
            corroborates=corroborates,
            contradicts=contradicts,
            kev=(
                AgreementAxis.AGREE if active and in_kev
                else AgreementAxis.CONTRADICT if active != in_kev
                else AgreementAxis.SILENT
            ),
            reason="synthetic",
        ),
        public_exploit_urls=tuple(
            f"https://example.invalid/{cve_id.lower()}/poc/{i}" for i in range(n_exploit_urls)
        ),
        active_exploitation_claimed=active,
        injection_signals=injection_signals,
        max_tier_used=TrustTier.REFERENCE_PAGE,
        provider="synthetic",
        model="synthetic",
    )


def attach_intel(enriched: Iterable, latents: Mapping[str, tuple[float, float]], seed: int = 42):
    """Fold synthetic intelligence onto enriched findings, returning new objects.

    ``latents`` maps ``finding_id`` to ``(true_exploitability, true_attacker_interest)``.
    A finding with no CVE gets nothing, which is correct: there is no CVE to search for, and
    it is also the common case on a real web application scan.
    """
    out = []
    for index, item in enumerate(enriched):
        cve_ids = getattr(item.finding, "cve_ids", ()) or ()
        pair = latents.get(item.finding_id)
        if not cve_ids or pair is None:
            out.append(item)
            continue
        exploitability, interest = pair
        in_kev = any(
            record.kev is not None and record.kev.in_kev for record in (item.intel or ())
        )
        result = synthesise_intel(
            item.finding_id,
            str(cve_ids[0]),
            item.as_of,
            true_exploitability=exploitability,
            true_attacker_interest=interest,
            in_kev=in_kev,
            seed=seed + index,
        )
        out.append(item if result is None else item.model_copy(update={"intel_result": result}))
    return out
