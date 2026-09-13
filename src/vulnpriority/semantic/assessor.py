"""Component A wired together: one object, three assessments, one cache, honest audits.

``AgenticSemanticAssessor`` is what the pipeline actually holds. Its job beyond
delegation is twofold.

*Caching.* A scan repeats the same endpoint across dozens of findings and the same CVE
across several endpoints. Assessments that depend only on an endpoint, or only on a
(CWE, CVE-set, endpoint) tuple, are computed once per run. This matters for cost with a
real backend and for determinism with any backend: the same inputs must produce the same
assessment inside one run, or the evaluation protocol is measuring noise.

*Truthful audits.* Every returned assessment carries an :class:`LLMAudit` describing what
really happened -- which backend ran, whether it was a cache hit, whether the result fell
back to the heuristic, which trust tier the prompt reached, and which injection signals
the sandbox raised. Downstream, ``a_injection_signals`` is a ranking feature and the
adversarial evaluation reads these flags, so an audit that flattered itself would corrupt
the evaluation rather than merely mislead a reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from vulnpriority.core.config import PipelineConfig
from vulnpriority.core.enums import LLMBackendKind
from vulnpriority.core.hashing import stable_id
from vulnpriority.core.interfaces import LLMBackend, Sanitizer, SemanticAssessor
from vulnpriority.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    Endpoint,
    ExploitabilityAssessment,
    Finding,
    LLMAudit,
    Scan,
    TechComponent,
    VulnIntel,
)
from vulnpriority.semantic.applicability import assess_applicability
from vulnpriority.semantic.criticality import assess_asset_criticality
from vulnpriority.semantic.exploitability import assess_exploitability

__all__ = ["AssessorStats", "AgenticSemanticAssessor"]


@dataclass
class AssessorStats:
    """Per-run counters, reported in the run manifest and useful in tests."""

    asset_calls: int = 0
    asset_cache_hits: int = 0
    exploitability_calls: int = 0
    exploitability_cache_hits: int = 0
    applicability_calls: int = 0
    applicability_cache_hits: int = 0
    model_backed: int = 0
    heuristic_fallbacks: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


def _tech_key(tech: Sequence[TechComponent]) -> str:
    parts = sorted(
        f"{component.vendor or ''}:{component.product}:{component.version or ''}" for component in tech
    )
    return "|".join(parts)


@dataclass
class AgenticSemanticAssessor(SemanticAssessor):
    """Component A: sandboxed, cached, auditable semantic assessment.

    ``backend`` and ``sandbox`` are optional. With neither, every assessment degrades to
    its deterministic structural form, which is exactly what the offline default
    configuration and the ablation cell ``A=off`` need.
    """

    backend: LLMBackend | None = None
    sandbox: Sanitizer | None = None
    config: PipelineConfig = field(default_factory=PipelineConfig)
    stats: AssessorStats = field(default_factory=AssessorStats)
    _assets: dict[str, AssetCriticality] = field(default_factory=dict, repr=False)
    _exploitability: dict[str, ExploitabilityAssessment] = field(default_factory=dict, repr=False)
    _applicability: dict[str, ApplicabilityAssessment] = field(default_factory=dict, repr=False)

    # -- caching helpers ----------------------------------------------------

    def reset_cache(self) -> None:
        """Drop every cached assessment. Call between runs, never inside one."""
        self._assets.clear()
        self._exploitability.clear()
        self._applicability.clear()

    @staticmethod
    def _as_cached(audit: LLMAudit | None, task: str) -> LLMAudit:
        if audit is None:
            return LLMAudit(
                backend=LLMBackendKind.HEURISTIC,
                model="structural",
                task=task,
                cached=True,
                fell_back_to_heuristic=True,
            )
        return audit.model_copy(update={"cached": True})

    def _record(self, audit: LLMAudit | None) -> None:
        if audit is None or audit.fell_back_to_heuristic:
            self.stats.heuristic_fallbacks += 1
        else:
            self.stats.model_backed += 1

    # -- SemanticAssessor ---------------------------------------------------

    def assess_asset(self, endpoint: Endpoint, scan: Scan) -> AssetCriticality:
        """Asset criticality for one endpoint, cached by ``endpoint_id`` within the run."""
        key = endpoint.endpoint_id
        cached = self._assets.get(key)
        if cached is not None:
            self.stats.asset_cache_hits += 1
            return cached.model_copy(update={"audit": self._as_cached(cached.audit, "asset_criticality")})

        self.stats.asset_calls += 1
        assessment = assess_asset_criticality(endpoint, scan, self.backend, self.sandbox, self.config)
        self._record(assessment.audit)
        self._assets[key] = assessment
        return assessment

    def assess_exploitability(
        self,
        finding: Finding,
        intel: tuple[VulnIntel, ...] = (),
        endpoint: Endpoint | None = None,
    ) -> ExploitabilityAssessment:
        """Exploitability, cached per (CWE, CVE set, endpoint, scanner verdict).

        Those are the only inputs the assessment reads, so two findings sharing them --
        the common case after ``ingest.correlate`` groups a root cause across endpoints
        -- genuinely have the same answer and are computed once.
        """
        key = stable_id(
            "xa",
            str(finding.cwe_id),
            ",".join(sorted(finding.cve_ids)),
            endpoint.endpoint_id if endpoint is not None else "-",
            finding.scanner_severity.value,
            f"{finding.scanner_confidence:.3f}",
            ",".join(sorted(item.cve_id for item in intel)),
        )
        cached = self._exploitability.get(key)
        if cached is not None:
            self.stats.exploitability_cache_hits += 1
            return cached.model_copy(
                update={
                    "finding_id": finding.finding_id,
                    "audit": self._as_cached(cached.audit, "exploitability"),
                }
            )

        self.stats.exploitability_calls += 1
        assessment = assess_exploitability(
            finding, intel, endpoint, self.backend, self.sandbox, self.config
        )
        self._record(assessment.audit)
        self._exploitability[key] = assessment
        return assessment

    def assess_applicability(
        self,
        finding: Finding,
        intel: tuple[VulnIntel, ...] = (),
        tech: tuple[TechComponent, ...] = (),
    ) -> ApplicabilityAssessment:
        """Applicability, cached per (CVE set, CWE, observed stack, scanner confidence)."""
        key = stable_id(
            "ap",
            ",".join(sorted(finding.cve_ids)),
            str(finding.cwe_id),
            _tech_key(tech),
            _tech_key((finding.affected_component,) if finding.affected_component else ()),
            f"{finding.scanner_confidence:.3f}",
        )
        cached = self._applicability.get(key)
        if cached is not None:
            self.stats.applicability_cache_hits += 1
            return cached.model_copy(
                update={
                    "finding_id": finding.finding_id,
                    "audit": self._as_cached(cached.audit, "applicability"),
                }
            )

        self.stats.applicability_calls += 1
        assessment = assess_applicability(
            finding, intel, tech, self.backend, self.sandbox, self.config
        )
        self._record(assessment.audit)
        self._applicability[key] = assessment
        return assessment

    # -- convenience --------------------------------------------------------

    def assess_scan(self, scan: Scan, intel_by_cve: dict[str, VulnIntel] | None = None) -> dict[str, dict]:
        """Assess every finding in a scan. Returns ``finding_id -> the three assessments``.

        Provided so the pipeline stage is a one-liner and so the caching behaviour is
        exercised end to end rather than only per call.
        """
        intel_by_cve = intel_by_cve or {}
        out: dict[str, dict] = {}
        for finding in scan.findings:
            endpoint = scan.endpoint_by_id(finding.endpoint_id)
            intel = tuple(
                intel_by_cve[cve_id] for cve_id in finding.cve_ids if cve_id in intel_by_cve
            )
            asset = self.assess_asset(endpoint, scan) if endpoint is not None else None
            out[finding.finding_id] = {
                "asset": asset,
                "exploitability": self.assess_exploitability(finding, intel, endpoint),
                "applicability": self.assess_applicability(finding, intel, scan.tech_stack),
            }
        return out
