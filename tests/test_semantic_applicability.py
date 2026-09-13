"""Applicability: is this finding actually exploitable in the deployment in front of us?

Research goal 3, and architecture item 2. The security property tested here is the important
one: a verdict reached from curated feed version data is authoritative, and a model reading an
attacker-controlled page cannot overturn it. Without that, the cheapest attack on the whole
framework is a blog post asserting that a patched product is still vulnerable, or that a
vulnerable one is not.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    LLMBackendKind,
    Provenance,
    ScannerSeverity,
    VersionMatch,
)
from vulnpriority.core.interfaces import LLMBackend, LLMResult, SandboxedPrompt
from vulnpriority.core.models import (
    AffectedProduct,
    Finding,
    LLMAudit,
    TechComponent,
    UntrustedText,
    VulnIntel,
)
from vulnpriority.core.config import load_config
from vulnpriority.llm.schemas import ApplicabilityOut
from vulnpriority.sandbox.pipeline import Sandbox
from vulnpriority.semantic.applicability import assess_applicability, baseline_applicability

AS_OF = date(2024, 6, 1)
WHEN = datetime(2024, 5, 1, 9, 0, 0)

AFFECTED = AffectedProduct(
    cpe="cpe:2.3:a:apache:struts:*:*:*:*:*:*:*:*",
    version_start_including="2.5.0",
    version_end_excluding="2.5.14",
)


class ShoutingBackend(LLMBackend):
    """A model that always insists the finding is applicable with total confidence.

    It stands in for a model that has been successfully prompt-injected. The point of the
    test is that believing it must not be possible when version evidence says otherwise.
    """

    kind = LLMBackendKind.HEURISTIC
    model_id = "shouting"

    def __init__(self, verdict: ApplicabilityVerdict = ApplicabilityVerdict.APPLICABLE, p: float = 0.99) -> None:
        self.verdict = verdict
        self.p = p
        self.calls = 0

    def complete_structured(self, prompt: SandboxedPrompt, schema: type) -> LLMResult:
        self.calls += 1
        parsed = ApplicabilityOut(
            confidence=1.0,
            rationale="The vendor advisory confirms every deployment is affected.",
            evidence_spans=[],
            verdict=self.verdict,
            p_applicable=self.p,
        )
        return LLMResult(parsed=parsed, raw_text="", audit=LLMAudit(backend=self.kind, model=self.model_id))


def _finding(cve: str | None = "CVE-2024-0001") -> Finding:
    return Finding(
        finding_id="f_app",
        scan_id="s",
        app_id="a",
        endpoint_id="e",
        name="Remote code execution in Struts",
        cwe_id=502,
        cve_ids=(cve,) if cve else (),
        scanner="zap",
        scanner_severity=ScannerSeverity.CRITICAL,
        description=UntrustedText(text="Struts deserialization flaw", provenance=Provenance.SCANNER_OUTPUT),
        observed_at=WHEN,
    )


def _intel(cve: str = "CVE-2024-0001", with_text: bool = False) -> VulnIntel:
    """Intel for the CVE. ``with_text`` adds the untrusted prose a model would read."""
    description = (
        UntrustedText(
            text="Exploitation requires the vulnerable plugin to be enabled and reachable.",
            provenance=Provenance.NVD,
        )
        if with_text
        else None
    )
    return VulnIntel(
        cve_id=cve, as_of=AS_OF, published=date(2024, 1, 10), affected=(AFFECTED,),
        description=description,
    )


CONFIG = load_config("configs/default.yaml")
SANDBOX = Sandbox(CONFIG.sandbox)

VULNERABLE = (TechComponent(vendor="apache", product="struts", version="2.5.12"),)
PATCHED = (TechComponent(vendor="apache", product="struts", version="2.5.31"),)
UNKNOWN_STACK = (TechComponent(vendor="apache", product="struts"),)


def test_version_inside_the_range_is_applicable() -> None:
    result = assess_applicability(_finding(), (_intel(),), VULNERABLE)
    assert result.version_match == VersionMatch.MATCH
    assert result.verdict == ApplicabilityVerdict.APPLICABLE
    assert result.p_applicable > 0.5


def test_patched_version_is_not_applicable() -> None:
    result = assess_applicability(_finding(), (_intel(),), PATCHED)
    assert result.version_match == VersionMatch.MISMATCH
    assert result.verdict == ApplicabilityVerdict.NOT_APPLICABLE
    assert result.p_applicable < 0.5


def test_unknown_version_stays_uncertain_rather_than_guessing() -> None:
    result = assess_applicability(_finding(), (_intel(),), UNKNOWN_STACK)
    assert result.version_match == VersionMatch.UNKNOWN
    assert result.verdict in (ApplicabilityVerdict.UNCERTAIN, ApplicabilityVerdict.APPLICABLE)


def test_a_model_cannot_overturn_a_version_mismatch() -> None:
    """The security property: curated version evidence beats anything a page says.

    A patched deployment must stay demoted even when the model returns maximum confidence
    that the finding applies, because that model may be reading attacker-authored text.
    """
    backend = ShoutingBackend(ApplicabilityVerdict.APPLICABLE, p=0.99)
    result = assess_applicability(_finding(), (_intel(with_text=True),), PATCHED, backend=backend, sandbox=SANDBOX, config=CONFIG)
    assert result.version_match == VersionMatch.MISMATCH
    assert result.verdict == ApplicabilityVerdict.NOT_APPLICABLE
    assert result.p_applicable < 0.5, "a model overturned authoritative version evidence"


def test_a_model_cannot_argue_a_matching_version_away_either() -> None:
    """Deflation is the mirror attack, and is capped the same way."""
    backend = ShoutingBackend(ApplicabilityVerdict.NOT_APPLICABLE, p=0.01)
    result = assess_applicability(_finding(), (_intel(with_text=True),), VULNERABLE, backend=backend, sandbox=SANDBOX, config=CONFIG)
    assert result.version_match == VersionMatch.MATCH
    assert result.p_applicable > 0.5, "a model argued away a confirmed version match"


def test_the_model_is_consulted_where_version_data_cannot_decide() -> None:
    """With prose to read and no decisive version evidence, the model gets a say."""
    backend = ShoutingBackend(ApplicabilityVerdict.APPLICABLE, p=0.9)
    result = assess_applicability(_finding(), (_intel(with_text=True),), UNKNOWN_STACK, backend=backend, sandbox=SANDBOX, config=CONFIG)
    assert backend.calls >= 1, "the model should be asked when version evidence is silent"
    assert 0.0 <= result.p_applicable <= 1.0


def test_the_finding_description_alone_is_enough_to_warrant_a_call() -> None:
    """The scanner's own description is untrusted text, so there is always something to read."""
    backend = ShoutingBackend()
    assess_applicability(_finding(), (_intel(),), UNKNOWN_STACK, backend=backend, sandbox=SANDBOX, config=CONFIG)
    assert backend.calls == 1


def test_no_intel_at_all_is_uncertain_not_applicable() -> None:
    result = assess_applicability(_finding(cve=None), (), VULNERABLE)
    assert result.version_match == VersionMatch.UNKNOWN
    assert 0.0 <= result.p_applicable <= 1.0


def test_baseline_is_deterministic_and_needs_no_model() -> None:
    first, first_authoritative = baseline_applicability(_finding(), (_intel(),), PATCHED)
    second, second_authoritative = baseline_applicability(_finding(), (_intel(),), PATCHED)
    assert first.model_dump() == second.model_dump()
    assert first.verdict == ApplicabilityVerdict.NOT_APPLICABLE
    assert first_authoritative is second_authoritative is True


def test_result_is_always_a_valid_bounded_assessment() -> None:
    for tech in (VULNERABLE, PATCHED, UNKNOWN_STACK, ()):
        result = assess_applicability(_finding(), (_intel(),), tech)
        assert result.finding_id == "f_app"
        assert 0.0 <= result.p_applicable <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.rationale) <= 600


def test_untrusted_text_never_reaches_a_model_without_a_sandbox() -> None:
    """Refusing the call is the safe failure: no sandbox, no untrusted text in a prompt."""
    backend = ShoutingBackend()
    result = assess_applicability(_finding(), (_intel(with_text=True),), UNKNOWN_STACK, backend=backend)
    assert backend.calls == 0
    assert result.audit is not None
