"""Placing a corpus payload on the surface its case declares (DESIGN.md 3.10).

The injector answers one question precisely: *if an adversary could write this text on that
surface, what would the pipeline have read?* Four surfaces are attacker-writable, and each
maps to exactly one place in the frozen data model:

============================  =====================================================  ====
``AdversarialCase``           where the payload lands                                tier
============================  =====================================================  ====
``reference_page``            :attr:`ReferenceDoc.content` of a :class:`VulnIntel`      3
``target_response``           :attr:`Endpoint.response_sample`                         4
``scanner_output``            :attr:`Finding.description`                              2
``exploit_db``                :attr:`ExploitEvidence.title`                            1
============================  =====================================================  ====

Two properties matter more than anything else here:

**Nothing is mutated.** Every model in the contract is frozen, but frozen is not the same as
unshared: a ``model_copy(update=...)`` leaves untouched branches pointing at the original
objects. The evaluator compares a clean run against an injected run, so a shared branch would
make a real difference indistinguishable from an aliasing bug. Both inputs are therefore
deep-copied before anything is rewritten, and the originals are returned unchanged.

**Target selection is deterministic.** A case that hits a different finding on every run
produces a rank displacement that cannot be reproduced. When a case names
``target_finding_id`` that finding is used; otherwise the target is chosen by hashing the
case id against the scan's own sorted finding ids, which is stable across processes,
platforms and Python's hash randomisation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from vulnprio.core.enums import ExploitMaturity, ExploitSource, Provenance
from vulnprio.core.errors import ConfigError
from vulnprio.core.hashing import sha256_text
from vulnprio.core.models import (
    AdversarialCase,
    Endpoint,
    ExploitEvidence,
    Finding,
    ReferenceDoc,
    Scan,
    UntrustedText,
    VulnIntel,
)

__all__ = [
    "PAYLOAD_SEPARATOR",
    "SYNTHETIC_REFERENCE_URL",
    "InjectionSite",
    "stable_index",
    "select_finding",
    "select_endpoint",
    "select_intel_key",
    "target_finding_id",
    "plan_injection",
    "inject",
    "inject_many",
    "site_text",
]

#: Payloads are appended to the surface's existing text rather than replacing it, because a
#: real injection arrives *inside* legitimate content; replacing the text would also test a
#: detector against a document with nothing benign in it, which flatters the detector.
PAYLOAD_SEPARATOR = "\n\n"

#: Used when the chosen intel record carries no reference document to write on. The host is
#: in the ``.invalid`` TLD so no fetch of it can ever succeed.
SYNTHETIC_REFERENCE_URL = "https://advisory.invalid/adversarial-corpus"


@dataclass(frozen=True)
class InjectionSite:
    """Where a case's payload goes, resolved against a concrete scan and intel set.

    Produced by :func:`plan_injection` and reused by :func:`inject` so the evaluator can ask
    "what did this case actually touch?" without re-deriving the answer.
    """

    case_id: str
    provenance: Provenance
    field: str
    finding_id: str | None = None
    endpoint_id: str | None = None
    cve_id: str | None = None
    reference_url: str | None = None
    original_text: str = ""
    injected_text: str = ""
    created_container: bool = False

    @property
    def description(self) -> str:
        """Human-readable location, for alert messages and report rows."""
        owner = self.cve_id or self.finding_id or self.endpoint_id or "?"
        return f"{self.provenance.value}:{owner}.{self.field}"


def stable_index(key: str, modulus: int) -> int:
    """Deterministic index in ``[0, modulus)`` derived from ``key``.

    Uses SHA-256 rather than :func:`hash` because Python salts string hashing per process:
    a corpus run must select the same targets today, tomorrow and on another machine.
    """
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return int(sha256_text(key)[:16], 16) % modulus


def _combine(original: str, payload: str) -> str:
    """Append ``payload`` to ``original``, keeping both readable."""
    left = (original or "").rstrip()
    right = payload.strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left}{PAYLOAD_SEPARATOR}{right}"


def select_finding(case: AdversarialCase, scan: Scan) -> Finding:
    """The finding a case targets.

    ``case.target_finding_id`` wins when it names a finding in the scan; otherwise the scan's
    findings are sorted by id and one is chosen by :func:`stable_index` over the case id.
    """
    if not scan.findings:
        raise ConfigError(f"cannot inject case {case.case_id!r}: the scan has no findings")
    if case.target_finding_id:
        for finding in scan.findings:
            if finding.finding_id == case.target_finding_id:
                return finding
        raise ConfigError(
            f"adversarial case {case.case_id!r} targets finding {case.target_finding_id!r}, "
            f"which is not in scan {scan.scan_id!r}"
        )
    ordered = sorted(scan.findings, key=lambda item: item.finding_id)
    return ordered[stable_index(f"{case.case_id}|finding", len(ordered))]


def select_endpoint(case: AdversarialCase, scan: Scan) -> Endpoint:
    """The endpoint a case targets: the target finding's endpoint when it exists."""
    if not scan.endpoints:
        raise ConfigError(f"cannot inject case {case.case_id!r}: the scan has no endpoints")
    finding = select_finding(case, scan)
    endpoint = scan.endpoint_by_id(finding.endpoint_id)
    if endpoint is not None:
        return endpoint
    ordered = sorted(scan.endpoints, key=lambda item: item.endpoint_id)
    return ordered[stable_index(f"{case.case_id}|endpoint", len(ordered))]


def select_intel_key(
    case: AdversarialCase, scan: Scan, intel: Mapping[str, VulnIntel]
) -> str | None:
    """The intel record a case targets, preferring one cited by the target finding.

    Returns ``None`` when there is no intel at all, which is a legitimate state: a scan of an
    application with no CVE-bearing findings simply has no reference page to poison.
    """
    if not intel:
        return None
    if scan.findings:
        cited = [cve for cve in select_finding(case, scan).cve_ids if cve in intel]
        if cited:
            return sorted(cited)[stable_index(f"{case.case_id}|cve", len(cited))]
    keys = sorted(intel)
    return keys[stable_index(f"{case.case_id}|cve", len(keys))]


def target_finding_id(case: AdversarialCase, scan: Scan) -> str:
    """Finding id whose rank displacement this case is measured against."""
    return select_finding(case, scan).finding_id


def _reference_injection(
    case: AdversarialCase, intel: dict[str, VulnIntel], key: str | None
) -> InjectionSite:
    """Write the payload into a reference document body (tier 3)."""
    if key is None:
        raise ConfigError(
            f"case {case.case_id!r} injects at a reference page but no VulnIntel was supplied"
        )
    record = intel[key]
    references = list(record.references)
    created = not references
    if created:
        index = 0
        original_text = ""
        content = UntrustedText(
            text=case.payload, provenance=Provenance.REFERENCE_PAGE, language=case.language
        )
        references = [
            ReferenceDoc(
                url=SYNTHETIC_REFERENCE_URL,
                title=f"Advisory for {record.cve_id}",
                tags=("adversarial-corpus",),
                content=content,
                language=case.language,
            )
        ]
    else:
        index = stable_index(f"{case.case_id}|ref", len(references))
        document = references[index]
        original_text = document.content.text
        content = UntrustedText(
            text=_combine(original_text, case.payload),
            provenance=Provenance.REFERENCE_PAGE,
            source_url=document.content.source_url or document.url,
            fetched_at=document.content.fetched_at,
            language=document.content.language,
        )
        references[index] = document.model_copy(update={"content": content})
    intel[key] = record.model_copy(update={"references": tuple(references)})
    return InjectionSite(
        case_id=case.case_id,
        provenance=Provenance.REFERENCE_PAGE,
        field="references[].content",
        cve_id=record.cve_id,
        reference_url=references[index].url,
        original_text=original_text,
        injected_text=content.text,
        created_container=created,
    )


def _target_response_injection(case: AdversarialCase, scan: Scan) -> tuple[Scan, InjectionSite]:
    """Write the payload into an endpoint response sample (tier 4)."""
    endpoint = select_endpoint(case, scan)
    endpoints = list(scan.endpoints)
    index = next(
        position
        for position, item in enumerate(endpoints)
        if item.endpoint_id == endpoint.endpoint_id
    )
    existing = endpoint.response_sample
    original_text = existing.text if existing is not None else ""
    sample = UntrustedText(
        text=_combine(original_text, case.payload),
        provenance=Provenance.TARGET_RESPONSE,
        source_url=existing.source_url if existing is not None else endpoint.url,
        fetched_at=existing.fetched_at if existing is not None else None,
        language=existing.language if existing is not None else case.language,
    )
    endpoints[index] = endpoint.model_copy(update={"response_sample": sample})
    site = InjectionSite(
        case_id=case.case_id,
        provenance=Provenance.TARGET_RESPONSE,
        field="response_sample",
        endpoint_id=endpoint.endpoint_id,
        finding_id=target_finding_id(case, scan),
        original_text=original_text,
        injected_text=sample.text,
        created_container=existing is None,
    )
    return scan.model_copy(update={"endpoints": tuple(endpoints)}), site


def _scanner_injection(case: AdversarialCase, scan: Scan) -> tuple[Scan, InjectionSite]:
    """Write the payload into a scanner finding description (tier 2)."""
    finding = select_finding(case, scan)
    findings = list(scan.findings)
    index = next(
        position
        for position, item in enumerate(findings)
        if item.finding_id == finding.finding_id
    )
    original_text = finding.description.text
    description = UntrustedText(
        text=_combine(original_text, case.payload),
        provenance=Provenance.SCANNER_OUTPUT,
        source_url=finding.description.source_url,
        fetched_at=finding.description.fetched_at,
        language=finding.description.language,
    )
    findings[index] = finding.model_copy(update={"description": description})
    site = InjectionSite(
        case_id=case.case_id,
        provenance=Provenance.SCANNER_OUTPUT,
        field="description",
        finding_id=finding.finding_id,
        endpoint_id=finding.endpoint_id,
        original_text=original_text,
        injected_text=description.text,
    )
    return scan.model_copy(update={"findings": tuple(findings)}), site


def _exploit_injection(
    case: AdversarialCase, intel: dict[str, VulnIntel], key: str | None
) -> InjectionSite:
    """Write the payload into an exploit record title.

    ``Provenance.EXPLOIT_DB`` maps to ``TrustTier.CURATED_FEED``, whose influence budget is
    unrestricted - an exploit *title* is third-party prose sitting behind a tier-1 label.
    That mismatch is the point of these cases and is reported, not worked around here.
    """
    if key is None:
        raise ConfigError(
            f"case {case.case_id!r} injects at an exploit record but no VulnIntel was supplied"
        )
    record = intel[key]
    exploits = list(record.exploits)
    created = not exploits
    if created:
        title = UntrustedText(
            text=case.payload, provenance=Provenance.EXPLOIT_DB, language=case.language
        )
        exploits = [
            ExploitEvidence(
                source=ExploitSource.EXPLOIT_DB,
                url=None,
                published=None,
                maturity=ExploitMaturity.UNKNOWN,
                verified=False,
                language=case.language,
                title=title,
            )
        ]
        original_text = ""
        index = 0
    else:
        index = stable_index(f"{case.case_id}|exploit", len(exploits))
        evidence = exploits[index]
        original_text = evidence.title.text if evidence.title is not None else ""
        title = UntrustedText(
            text=_combine(original_text, case.payload),
            provenance=Provenance.EXPLOIT_DB,
            source_url=evidence.title.source_url if evidence.title is not None else evidence.url,
            language=evidence.title.language if evidence.title is not None else case.language,
        )
        exploits[index] = evidence.model_copy(update={"title": title})
    intel[key] = record.model_copy(update={"exploits": tuple(exploits)})
    return InjectionSite(
        case_id=case.case_id,
        provenance=Provenance.EXPLOIT_DB,
        field="exploits[].title",
        cve_id=record.cve_id,
        original_text=original_text,
        injected_text=title.text,
        created_container=created,
    )


def plan_injection(
    case: AdversarialCase, scan: Scan, intel: Mapping[str, VulnIntel] | None = None
) -> InjectionSite:
    """Resolve where a case would land without producing the mutated copies.

    Useful for reporting and for the pre-model detector, which wants the text that *would*
    be read rather than a whole mutated scan.
    """
    _, _, site = _apply(case, scan, intel or {})
    return site


def inject(
    case: AdversarialCase,
    scan: Scan,
    intel: Mapping[str, VulnIntel] | None = None,
) -> tuple[Scan, dict[str, VulnIntel]]:
    """Place ``case.payload`` at its declared injection point.

    Returns deep copies of the scan and the intel mapping with the payload written in. The
    arguments are never modified: the returned graph shares no object with them, so an
    evaluator can compare clean and injected runs without worrying about aliasing.

    Raises :class:`~vulnprio.core.errors.ConfigError` when the declared surface does not
    exist in the supplied data - for example a ``reference_page`` case with no intel, which
    would otherwise silently evaluate as a perfectly robust pipeline.
    """
    mutated_scan, mutated_intel, _ = _apply(case, scan, intel or {})
    return mutated_scan, mutated_intel


def inject_many(
    cases: Sequence[AdversarialCase],
    scan: Scan,
    intel: Mapping[str, VulnIntel] | None = None,
) -> list[tuple[AdversarialCase, Scan, dict[str, VulnIntel]]]:
    """Inject each case independently into the *same* clean baseline.

    One case per row, never cumulative: two payloads in one scan would make the attribution
    of a feature delta ambiguous, and the report is per case.
    """
    baseline = intel or {}
    return [(case, *inject(case, scan, baseline)) for case in cases]


def site_text(case: AdversarialCase, scan: Scan, intel: Mapping[str, VulnIntel] | None = None) -> str:
    """The full text the pipeline would read at this case's injection point."""
    return plan_injection(case, scan, intel).injected_text


def _apply(
    case: AdversarialCase, scan: Scan, intel: Mapping[str, VulnIntel]
) -> tuple[Scan, dict[str, VulnIntel], InjectionSite]:
    """Deep-copy both inputs, write the payload in, and report the site."""
    scan_copy = scan.model_copy(deep=True)
    intel_copy: dict[str, VulnIntel] = {
        key: value.model_copy(deep=True) for key, value in intel.items()
    }
    point = case.injection_point

    if point in (Provenance.REFERENCE_PAGE, Provenance.EXPLOIT_DB):
        key = select_intel_key(case, scan_copy, intel_copy)
        if point == Provenance.REFERENCE_PAGE:
            site = _reference_injection(case, intel_copy, key)
        else:
            site = _exploit_injection(case, intel_copy, key)
        if scan_copy.findings:
            finding = select_finding(case, scan_copy)
            site = replace(site, finding_id=finding.finding_id, endpoint_id=finding.endpoint_id)
        return scan_copy, intel_copy, site

    if point == Provenance.TARGET_RESPONSE:
        mutated, site = _target_response_injection(case, scan_copy)
        return mutated, intel_copy, site

    if point == Provenance.SCANNER_OUTPUT:
        mutated, site = _scanner_injection(case, scan_copy)
        return mutated, intel_copy, site

    raise ConfigError(
        f"adversarial case {case.case_id!r} declares injection point {point.value!r}, "
        "which is not an attacker-writable surface"
    )
