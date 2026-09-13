"""The latent world: hidden truths, and the noisy evidence they cast (DESIGN.md 3.11).

**This module is the reason the synthetic evaluation is not circular.** Every vulnerability
in the generated universe is created by first drawing four *latent* variables that the
framework never observes:

``true_exploitability``
    How readily a competent attacker can turn this defect into working access.
``true_applicability``
    The probability that the defect is genuinely present in a given deployment, rather than
    merely reported against a product family the deployment happens to run.
``true_attacker_interest``
    How attractive the defect is to the adversary being modelled, independently of how
    technically easy it is.
``true_chain_position``
    How much of the compromise graph the defect unlocks: a chokepoint scores high even when
    its direct impact is trivial.

The *observed* evidence is then generated **from** those latents, with the noise real
intelligence sources have: CVSS records that disagree between NVD and the CNA, EPSS that is
correlated with real exploitability but far from a bijection, a KEV listing for a minority,
and exploit records that appear only after a lag. The exploitation oracle in
:mod:`vulnpriority.synth.oracle` reads the latents; the framework reads the evidence. Neither
reads the other's inputs.

That separation is what Gap 10 needs. If the oracle were driven by the features the
framework computes - CVSS, EPSS, KEV, the chain score - then a ranker that reproduces those
features would score perfectly by construction and the entire evaluation would measure
nothing. Here, a ranker can only do well by inferring the latent structure through the noise,
which is the actual research question.

Nothing in this module reaches the network, reads the clock, or draws from any random source
other than the seeded generators handed to it.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from random import Random
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from vulnpriority.core.enums import (
    AttackComplexity,
    EndpointFunction,
    ExploitMaturity,
    PrivilegeLevel,
    ScannerSeverity,
    UserInteraction,
)
from vulnpriority.synth.topology import derive_seed

__all__ = [
    "VulnTemplate",
    "VULN_TEMPLATES",
    "TEMPLATE_BY_CWE",
    "LatentVuln",
    "LatentWorld",
    "EpssPoint",
    "build_world",
    "sigmoid",
    "logit",
    "clamp01",
]


def clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


def sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def logit(p: float) -> float:
    bounded = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(bounded / (1.0 - bounded))


# ---------------------------------------------------------------------------
# Vulnerability class templates
# ---------------------------------------------------------------------------


class VulnTemplate(BaseModel):
    """A CWE class with the prior structure that class really has.

    The priors are *not* derived from CVSS: they are statements about the world (a
    deserialisation bug tends to give system privileges; an information disclosure tends not
    to) from which both the latents and, separately, the CVSS records are generated.
    """

    model_config = ConfigDict(frozen=True)

    cwe_id: int
    name: str
    title: str
    exploitability_prior: float = Field(ge=0.0, le=1.0)
    impact_prior: float = Field(ge=0.0, le=1.0)
    chain_prior: float = Field(ge=0.0, le=1.0)
    interest_prior: float = Field(ge=0.0, le=1.0)
    privileges_required: PrivilegeLevel = PrivilegeLevel.NONE
    privilege_gained: PrivilegeLevel = PrivilegeLevel.USER
    user_interaction: UserInteraction = UserInteraction.NONE
    impact_c: float = Field(0.5, ge=0.0, le=1.0)
    impact_i: float = Field(0.5, ge=0.0, le=1.0)
    impact_a: float = Field(0.0, ge=0.0, le=1.0)
    severity: ScannerSeverity = ScannerSeverity.MEDIUM
    plugin_id: str = "90000"
    functions: tuple[EndpointFunction, ...] = ()
    owasp_topic: str = "injection"
    carries_cve: float = Field(0.6, ge=0.0, le=1.0)


#: The defect classes the generated world contains. Sixteen classes is enough to give the
#: ranker a non-degenerate CWE feature and the graph a mix of privilege transitions.
VULN_TEMPLATES: tuple[VulnTemplate, ...] = (
    VulnTemplate(
        cwe_id=89, name="SQL Injection", title="SQL injection in the {param} parameter",
        exploitability_prior=0.80, impact_prior=0.85, chain_prior=0.65, interest_prior=0.80,
        privilege_gained=PrivilegeLevel.ADMIN, impact_c=1.0, impact_i=0.8, impact_a=0.3,
        severity=ScannerSeverity.HIGH, plugin_id="40018",
        functions=(EndpointFunction.API_DATA, EndpointFunction.SEARCH, EndpointFunction.AUTH),
        owasp_topic="sql-injection", carries_cve=0.45,
    ),
    VulnTemplate(
        cwe_id=79, name="Reflected Cross Site Scripting",
        title="Reflected cross-site scripting in the {param} parameter",
        exploitability_prior=0.55, impact_prior=0.35, chain_prior=0.35, interest_prior=0.40,
        privilege_gained=PrivilegeLevel.USER, user_interaction=UserInteraction.REQUIRED,
        impact_c=0.5, impact_i=0.5, impact_a=0.0,
        severity=ScannerSeverity.MEDIUM, plugin_id="40012",
        functions=(EndpointFunction.SEARCH, EndpointFunction.STATIC_CONTENT),
        owasp_topic="cross-site-scripting", carries_cve=0.30,
    ),
    VulnTemplate(
        cwe_id=78, name="OS Command Injection", title="OS command injection via {param}",
        exploitability_prior=0.72, impact_prior=0.95, chain_prior=0.85, interest_prior=0.85,
        privilege_gained=PrivilegeLevel.SYSTEM, impact_c=1.0, impact_i=1.0, impact_a=1.0,
        severity=ScannerSeverity.CRITICAL, plugin_id="90020",
        functions=(EndpointFunction.FILE_IO, EndpointFunction.ADMIN),
        owasp_topic="os-command-injection", carries_cve=0.70,
    ),
    VulnTemplate(
        cwe_id=22, name="Path Traversal", title="Path traversal in the {param} parameter",
        exploitability_prior=0.62, impact_prior=0.70, chain_prior=0.60, interest_prior=0.55,
        privilege_gained=PrivilegeLevel.USER, impact_c=1.0, impact_i=0.2, impact_a=0.2,
        severity=ScannerSeverity.HIGH, plugin_id="6",
        functions=(EndpointFunction.FILE_IO,),
        owasp_topic="file-path-traversal", carries_cve=0.55,
    ),
    VulnTemplate(
        cwe_id=502, name="Deserialization of Untrusted Data",
        title="Unsafe deserialisation of the {param} value",
        exploitability_prior=0.58, impact_prior=0.95, chain_prior=0.88, interest_prior=0.78,
        privilege_gained=PrivilegeLevel.SYSTEM, impact_c=1.0, impact_i=1.0, impact_a=1.0,
        severity=ScannerSeverity.CRITICAL, plugin_id="90030",
        functions=(EndpointFunction.API_DATA,),
        owasp_topic="deserialization", carries_cve=0.85,
    ),
    VulnTemplate(
        cwe_id=918, name="Server Side Request Forgery",
        title="Server-side request forgery via {param}",
        exploitability_prior=0.50, impact_prior=0.70, chain_prior=0.80, interest_prior=0.60,
        privilege_gained=PrivilegeLevel.USER, impact_c=0.8, impact_i=0.3, impact_a=0.2,
        severity=ScannerSeverity.HIGH, plugin_id="90021",
        functions=(EndpointFunction.API_DATA,),
        owasp_topic="ssrf", carries_cve=0.50,
    ),
    VulnTemplate(
        cwe_id=287, name="Improper Authentication",
        title="Authentication can be bypassed on this endpoint",
        exploitability_prior=0.66, impact_prior=0.80, chain_prior=0.82, interest_prior=0.72,
        privilege_gained=PrivilegeLevel.USER, impact_c=0.9, impact_i=0.6, impact_a=0.0,
        severity=ScannerSeverity.HIGH, plugin_id="10105",
        functions=(EndpointFunction.AUTH,),
        owasp_topic="authentication", carries_cve=0.60,
    ),
    VulnTemplate(
        cwe_id=862, name="Missing Authorization",
        title="Missing function-level authorisation check",
        exploitability_prior=0.70, impact_prior=0.72, chain_prior=0.70, interest_prior=0.62,
        privilege_gained=PrivilegeLevel.ADMIN, impact_c=0.9, impact_i=0.7, impact_a=0.0,
        severity=ScannerSeverity.HIGH, plugin_id="90022",
        functions=(EndpointFunction.ADMIN, EndpointFunction.API_DATA),
        owasp_topic="access-control", carries_cve=0.25,
    ),
    VulnTemplate(
        cwe_id=639, name="Insecure Direct Object Reference",
        title="Direct object reference in the {param} parameter",
        exploitability_prior=0.74, impact_prior=0.60, chain_prior=0.45, interest_prior=0.55,
        privilege_gained=PrivilegeLevel.USER, impact_c=0.9, impact_i=0.3, impact_a=0.0,
        severity=ScannerSeverity.MEDIUM, plugin_id="90023",
        functions=(EndpointFunction.PII_DATA, EndpointFunction.API_DATA),
        owasp_topic="access-control", carries_cve=0.20,
    ),
    VulnTemplate(
        cwe_id=434, name="Unrestricted File Upload",
        title="Unrestricted upload of a file with a dangerous type",
        exploitability_prior=0.60, impact_prior=0.90, chain_prior=0.80, interest_prior=0.70,
        privilege_gained=PrivilegeLevel.SYSTEM, impact_c=0.8, impact_i=1.0, impact_a=0.8,
        severity=ScannerSeverity.CRITICAL, plugin_id="90024",
        functions=(EndpointFunction.FILE_IO,),
        owasp_topic="file-upload", carries_cve=0.55,
    ),
    VulnTemplate(
        cwe_id=352, name="Cross Site Request Forgery",
        title="Cross-site request forgery on a state-changing endpoint",
        exploitability_prior=0.40, impact_prior=0.45, chain_prior=0.30, interest_prior=0.30,
        privilege_gained=PrivilegeLevel.USER, user_interaction=UserInteraction.REQUIRED,
        impact_c=0.0, impact_i=0.8, impact_a=0.0,
        severity=ScannerSeverity.MEDIUM, plugin_id="20012",
        functions=(EndpointFunction.PAYMENT, EndpointFunction.ADMIN),
        owasp_topic="csrf", carries_cve=0.15,
    ),
    VulnTemplate(
        cwe_id=611, name="XML External Entity",
        title="XML external entity expansion in the request body",
        exploitability_prior=0.48, impact_prior=0.75, chain_prior=0.65, interest_prior=0.50,
        privilege_gained=PrivilegeLevel.USER, impact_c=1.0, impact_i=0.2, impact_a=0.5,
        severity=ScannerSeverity.HIGH, plugin_id="90025",
        functions=(EndpointFunction.API_DATA,),
        owasp_topic="xxe", carries_cve=0.65,
    ),
    VulnTemplate(
        cwe_id=601, name="Open Redirect", title="Open redirect via the {param} parameter",
        exploitability_prior=0.66, impact_prior=0.20, chain_prior=0.22, interest_prior=0.25,
        privilege_gained=PrivilegeLevel.NONE, user_interaction=UserInteraction.REQUIRED,
        impact_c=0.2, impact_i=0.2, impact_a=0.0,
        severity=ScannerSeverity.LOW, plugin_id="10028",
        functions=(EndpointFunction.AUTH,),
        owasp_topic="open-redirect", carries_cve=0.10,
    ),
    VulnTemplate(
        cwe_id=798, name="Use of Hard-coded Credentials",
        title="Hard-coded credentials disclosed in the response",
        exploitability_prior=0.55, impact_prior=0.85, chain_prior=0.75, interest_prior=0.65,
        privilege_gained=PrivilegeLevel.ADMIN, impact_c=1.0, impact_i=0.8, impact_a=0.3,
        severity=ScannerSeverity.HIGH, plugin_id="90026",
        functions=(EndpointFunction.STATIC_CONTENT, EndpointFunction.API_DATA),
        owasp_topic="secrets", carries_cve=0.35,
    ),
    VulnTemplate(
        cwe_id=200, name="Information Exposure",
        title="Sensitive information disclosed in the response",
        exploitability_prior=0.70, impact_prior=0.25, chain_prior=0.30, interest_prior=0.28,
        privilege_gained=PrivilegeLevel.NONE, impact_c=0.4, impact_i=0.0, impact_a=0.0,
        severity=ScannerSeverity.LOW, plugin_id="10036",
        functions=(EndpointFunction.STATIC_CONTENT,),
        owasp_topic="information-disclosure", carries_cve=0.20,
    ),
    VulnTemplate(
        cwe_id=94, name="Code Injection", title="Server-side template / code injection via {param}",
        exploitability_prior=0.52, impact_prior=0.95, chain_prior=0.85, interest_prior=0.80,
        privilege_gained=PrivilegeLevel.SYSTEM, impact_c=1.0, impact_i=1.0, impact_a=1.0,
        severity=ScannerSeverity.CRITICAL, plugin_id="90027",
        functions=(EndpointFunction.SEARCH, EndpointFunction.API_DATA),
        owasp_topic="server-side-template-injection", carries_cve=0.75,
    ),
)

TEMPLATE_BY_CWE: dict[int, VulnTemplate] = {item.cwe_id: item for item in VULN_TEMPLATES}

#: Impact submetric letters for a CVSS v3.1 vector, by impact magnitude.
def _impact_letter(value: float) -> str:
    if value >= 0.66:
        return "H"
    if value >= 0.25:
        return "L"
    return "N"


_PR_LETTER: dict[PrivilegeLevel, str] = {
    PrivilegeLevel.NONE: "N",
    PrivilegeLevel.USER: "L",
    PrivilegeLevel.ADMIN: "H",
    PrivilegeLevel.SYSTEM: "H",
}


# ---------------------------------------------------------------------------
# Latent vulnerability
# ---------------------------------------------------------------------------


class EpssPoint(BaseModel):
    """One EPSS snapshot for one CVE."""

    model_config = ConfigDict(frozen=True)

    cve_id: str
    as_of: date
    score: float = Field(ge=0.0, le=1.0)
    percentile: float = Field(ge=0.0, le=1.0)


class LatentVuln(BaseModel):
    """One vulnerability, latent truths first and observable evidence second.

    The field ordering is the point: everything above ``--- observed ---`` is invisible to
    the framework and is what the oracle reads; everything below is what the feeds publish.
    """

    model_config = ConfigDict(frozen=True)

    # --- identity -------------------------------------------------------
    cve_id: str
    cwe_id: int
    name: str
    description: str
    vendor: str
    product: str
    observed_version: str
    version_end_excluding: str
    version_applies: bool = True
    published: date

    # --- latent truths (never observable) --------------------------------
    true_exploitability: float = Field(ge=0.0, le=1.0)
    true_applicability: float = Field(ge=0.0, le=1.0)
    true_attacker_interest: float = Field(ge=0.0, le=1.0)
    true_chain_position: float = Field(ge=0.0, le=1.0)
    true_impact: float = Field(ge=0.0, le=1.0)

    # --- observed evidence (generated from the latents, with noise) ------
    cvss_nvd: float = Field(ge=0.0, le=10.0)
    cvss_cna: float = Field(ge=0.0, le=10.0)
    cvss_v2: float | None = None
    cvss_vector: str = ""
    attack_complexity: AttackComplexity = AttackComplexity.LOW
    user_interaction: UserInteraction = UserInteraction.NONE
    privileges_required: PrivilegeLevel = PrivilegeLevel.NONE
    privilege_gained: PrivilegeLevel = PrivilegeLevel.USER
    impact_c: float = Field(0.0, ge=0.0, le=1.0)
    impact_i: float = Field(0.0, ge=0.0, le=1.0)
    impact_a: float = Field(0.0, ge=0.0, le=1.0)

    epss_anchor: float = Field(ge=0.0, le=1.0)
    epss_drift: float = 0.0

    in_kev: bool = False
    kev_date_added: date | None = None
    kev_due_date: date | None = None
    kev_ransomware: bool = False

    exploit_published: date | None = None
    exploit_verified: bool = False
    exploit_maturity: ExploitMaturity = ExploitMaturity.UNKNOWN
    exploit_id: str = ""
    exploit_title: str = ""

    reference_urls: tuple[str, ...] = ()

    @property
    def template(self) -> VulnTemplate:
        return TEMPLATE_BY_CWE[self.cwe_id]

    @property
    def cpe(self) -> str:
        return f"cpe:2.3:a:{self.vendor}:{self.product}:*:*:*:*:*:*:*:*"

    def epss_at(self, as_of: date) -> float:
        """EPSS level on ``as_of``: the anchor, drifting, with jumps at the public events.

        A published exploit and a KEV listing both move EPSS in the real feed, and the jump
        is what makes the time-ordered split informative: the same CVE carries different
        evidence in March and in September.
        """
        base = logit(self.epss_anchor)
        age_days = (as_of - self.published).days
        base += self.epss_drift * (age_days / 365.0)
        if self.exploit_published is not None and as_of >= self.exploit_published:
            base += 1.25
        if self.kev_date_added is not None and as_of >= self.kev_date_added:
            base += 1.60
        return clamp01(sigmoid(base))


class LatentWorld(BaseModel):
    """The generated vulnerability universe and its evidence time series."""

    model_config = ConfigDict(frozen=True)

    seed: int
    vulns: tuple[LatentVuln, ...] = ()
    epss_dates: tuple[date, ...] = ()

    def by_id(self, cve_id: str) -> LatentVuln | None:
        wanted = str(cve_id).strip().upper()
        for vuln in self.vulns:
            if vuln.cve_id == wanted:
                return vuln
        return None

    def index(self) -> dict[str, LatentVuln]:
        return {vuln.cve_id: vuln for vuln in self.vulns}

    def for_template(self, cwe_id: int) -> tuple[LatentVuln, ...]:
        return tuple(vuln for vuln in self.vulns if vuln.cwe_id == int(cwe_id))

    def kev_fraction(self) -> float:
        return (sum(1 for vuln in self.vulns if vuln.in_kev) / len(self.vulns)) if self.vulns else 0.0

    def epss_series(self) -> tuple[EpssPoint, ...]:
        """Every snapshot of every CVE, with percentiles computed within each snapshot date.

        Percentile is a *rank* in the real feed, so it is computed here as a rank too:
        deriving it from the score with a fixed formula would make the two features
        collinear and would hand the ranker information the real world does not supply.
        """
        points: list[EpssPoint] = []
        for snapshot in self.epss_dates:
            live = [vuln for vuln in self.vulns if vuln.published <= snapshot]
            if not live:
                continue
            scored = sorted(
                ((vuln.cve_id, vuln.epss_at(snapshot)) for vuln in live),
                key=lambda item: (item[1], item[0]),
            )
            total = len(scored)
            for rank, (cve_id, score) in enumerate(scored):
                points.append(
                    EpssPoint(
                        cve_id=cve_id,
                        as_of=snapshot,
                        score=round(score, 6),
                        percentile=round((rank + 1) / total, 6),
                    )
                )
        points.sort(key=lambda point: (point.as_of, point.cve_id))
        return tuple(points)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _draw_latent(rng: Random, prior: float, spread: float = 0.22) -> float:
    """A latent truth around ``prior``, bounded away from the degenerate endpoints."""
    return clamp01(rng.normalvariate(prior, spread))


def _cvss_vector(template: VulnTemplate, complexity: AttackComplexity,
                 impact_c: float, impact_i: float, impact_a: float) -> tuple[str, dict[str, str]]:
    submetrics = {
        "AV": "N",
        "AC": "H" if complexity == AttackComplexity.HIGH else "L",
        "PR": _PR_LETTER[template.privileges_required],
        "UI": "R" if template.user_interaction == UserInteraction.REQUIRED else "N",
        "S": "C" if template.privilege_gained == PrivilegeLevel.SYSTEM else "U",
        "C": _impact_letter(impact_c),
        "I": _impact_letter(impact_i),
        "A": _impact_letter(impact_a),
    }
    vector = "CVSS:3.1/" + "/".join(f"{key}:{value}" for key, value in submetrics.items())
    return vector, submetrics


def _reference_urls(cve_id: str, vuln_topic: str, rng: Random, *, kev: bool,
                    exploit_id: str | None) -> tuple[str, ...]:
    """Advisory links for a CVE, all on the configured reference host allowlist."""
    urls = [f"https://nvd.nist.gov/vuln/detail/{cve_id}"]
    if kev:
        urls.append("https://www.cisa.gov/known-exploited-vulnerabilities-catalog")
    if exploit_id:
        urls.append(f"https://www.exploit-db.com/exploits/{exploit_id}")
    if rng.random() < 0.65:
        urls.append(f"https://github.com/vulnpriority-synthetic/advisories/blob/main/{cve_id}.md")
    if rng.random() < 0.45:
        urls.append(f"https://owasp.org/www-community/attacks/{vuln_topic}")
    if rng.random() < 0.35:
        urls.append(f"https://portswigger.net/web-security/{vuln_topic}")
    return tuple(dict.fromkeys(urls))


def _make_vuln(
    index: int,
    template: VulnTemplate,
    stack_entry: tuple[str, str, str],
    rng: Random,
    *,
    year: int,
    published: date,
) -> tuple[LatentVuln, float, float]:
    """One vulnerability plus its KEV and exploit-record propensities.

    The propensities are returned rather than thresholded here so that
    :func:`build_world` can give KEV and Exploit-DB their configured population shares
    exactly, instead of whatever a fixed cut-off happens to produce.
    """
    vendor, product, observed_version = stack_entry

    true_exploitability = _draw_latent(rng, template.exploitability_prior)
    true_impact = _draw_latent(rng, template.impact_prior, 0.16)
    true_chain_position = _draw_latent(rng, template.chain_prior, 0.20)
    true_attacker_interest = clamp01(
        0.55 * _draw_latent(rng, template.interest_prior, 0.20) + 0.45 * true_impact
    )
    true_applicability = clamp01(rng.betavariate(5.0, 2.2))

    # --- observed CVSS: correlated with (exploitability, impact) but noisy, and the two
    # sources disagree, which is exactly the instrument unreliability of Gap 3.
    core = 0.42 * true_exploitability + 0.48 * true_impact + 0.10 * true_attacker_interest
    raw = 1.2 + 8.8 * clamp01(core + rng.normalvariate(0.0, 0.07))
    cvss_nvd = round(min(10.0, max(0.1, raw + rng.normalvariate(0.0, 0.45))), 1)
    cvss_cna = round(min(10.0, max(0.1, raw + rng.normalvariate(0.0, 1.25))), 1)
    cvss_v2 = round(min(10.0, max(0.1, raw * 0.86 + rng.normalvariate(0.0, 0.8))), 1) if rng.random() < 0.45 else None

    complexity = AttackComplexity.HIGH if true_exploitability < 0.42 else AttackComplexity.LOW
    impact_c = clamp01(template.impact_c * (0.75 + 0.5 * true_impact))
    impact_i = clamp01(template.impact_i * (0.75 + 0.5 * true_impact))
    impact_a = clamp01(template.impact_a * (0.75 + 0.5 * true_impact))
    vector, _submetrics = _cvss_vector(template, complexity, impact_c, impact_i, impact_a)

    # --- observed EPSS anchor: correlated with true exploitability, correlation well
    # short of 1 so that an EPSS-only baseline is good but beatable.
    # The intercept is deliberately far negative: real EPSS is heavily right-skewed, with
    # the median CVE well under 1%, and a synthetic feed centred near 0.5 would make an
    # EPSS-only baseline trivially strong.
    epss_anchor = clamp01(
        sigmoid(-9.8 + 6.6 * true_exploitability + 1.4 * true_attacker_interest
                + rng.normalvariate(0.0, 1.15))
    )
    epss_drift = rng.normalvariate(0.05, 0.20)

    # --- KEV and exploit-record propensities. Both are skewed towards genuinely
    # exploitable and interesting defects; build_world turns them into memberships.
    kev_propensity = (
        3.4 * true_exploitability + 2.6 * true_attacker_interest + rng.normalvariate(0.0, 0.8)
    )
    exploit_propensity = 3.6 * true_exploitability + rng.normalvariate(0.0, 0.7)

    # Draw the dependent details unconditionally so that the random stream does not depend
    # on the outcome; unused values are simply discarded when the membership is not granted.
    kev_lag = int(20 + rng.expovariate(1.0 / 150.0))
    kev_ransomware = rng.random() < 0.35 + 0.3 * true_attacker_interest
    exploit_lag = int(3 + rng.expovariate(1.0 / 95.0))
    exploit_verified = rng.random() < 0.35 + 0.45 * true_exploitability
    weaponized_roll = rng.random()

    cve_id = f"CVE-{year}-{20000 + index:05d}"
    fixed_version, version_applies = _fixed_version(observed_version, rng)
    description = (
        f"{template.name} in {vendor} {product} before {fixed_version} allows a remote "
        f"attacker to {_effect_phrase(template)}."
    )
    exploit_id = f"{50000 + index}"
    vuln = LatentVuln(
        cve_id=cve_id,
        cwe_id=template.cwe_id,
        name=template.name,
        description=description,
        vendor=vendor,
        product=product,
        observed_version=observed_version,
        version_end_excluding=fixed_version,
        version_applies=version_applies,
        published=published,
        true_exploitability=round(true_exploitability, 6),
        true_applicability=round(true_applicability, 6),
        true_attacker_interest=round(true_attacker_interest, 6),
        true_chain_position=round(true_chain_position, 6),
        true_impact=round(true_impact, 6),
        cvss_nvd=cvss_nvd,
        cvss_cna=cvss_cna,
        cvss_v2=cvss_v2,
        cvss_vector=vector,
        attack_complexity=complexity,
        user_interaction=template.user_interaction,
        privileges_required=template.privileges_required,
        privilege_gained=template.privilege_gained,
        impact_c=round(impact_c, 6),
        impact_i=round(impact_i, 6),
        impact_a=round(impact_a, 6),
        epss_anchor=round(epss_anchor, 6),
        epss_drift=round(epss_drift, 6),
        in_kev=False,
        kev_date_added=published + timedelta(days=kev_lag),
        kev_due_date=published + timedelta(days=kev_lag + 21),
        kev_ransomware=kev_ransomware,
        exploit_published=published + timedelta(days=exploit_lag),
        exploit_verified=exploit_verified,
        exploit_maturity=(
            ExploitMaturity.WEAPONIZED
            if (true_exploitability > 0.75 and weaponized_roll < 0.5)
            else (ExploitMaturity.FUNCTIONAL if exploit_verified else ExploitMaturity.POC)
        ),
        exploit_id=exploit_id,
        exploit_title=f"{product} {observed_version} - {template.name}",
        reference_urls=(),
    )
    return vuln, kev_propensity, exploit_propensity


_EFFECT_PHRASES: dict[int, str] = {
    89: "read and modify arbitrary database rows",
    79: "execute script in the browser of a signed-in user",
    78: "execute arbitrary operating system commands",
    22: "read files outside the intended directory",
    502: "execute arbitrary code during object reconstruction",
    918: "make the server issue requests to internal hosts",
    287: "authenticate as another user without valid credentials",
    862: "invoke privileged functions without authorisation",
    639: "read records belonging to other users",
    434: "upload and execute a file of an arbitrary type",
    352: "cause a signed-in user to submit an unintended request",
    611: "read local files through external entity expansion",
    601: "redirect a signed-in user to an attacker-controlled host",
    798: "authenticate using credentials embedded in the product",
    200: "obtain sensitive configuration and version information",
    94: "evaluate attacker-supplied expressions on the server",
}


def _effect_phrase(template: VulnTemplate) -> str:
    return _EFFECT_PHRASES.get(template.cwe_id, "compromise the application")


def _parse_numbers(version: str) -> list[int]:
    numbers = [int(part) if part.isdigit() else 0 for part in version.split(".")]
    while len(numbers) < 3:
        numbers.append(0)
    return numbers


def _fixed_version(observed: str, rng: Random, *, vulnerable_share: float = 0.55) -> tuple[str, bool]:
    """The ``versionEndExcluding`` NVD publishes, and whether it covers ``observed``.

    A little over half the universe is fixed *after* the deployed version, so the deployment
    is genuinely vulnerable; the rest is fixed *before* it, so the CVE is real but does not
    apply here. Without both cases, Goal 3 (applicability) would have nothing to decide and
    the label policy's version-mismatch rule would never fire.
    """
    numbers = _parse_numbers(observed)
    width = max(2, len(observed.split(".")))
    vulnerable = rng.random() < vulnerable_share
    if vulnerable:
        numbers[-1] += rng.randint(1, 9)
        if rng.random() < 0.5:
            numbers[1] += rng.randint(1, 3)
    else:
        if numbers[-1] > 0:
            numbers[-1] = max(0, numbers[-1] - rng.randint(1, max(1, numbers[-1])))
        elif numbers[1] > 0:
            numbers[1] -= 1
            numbers[-1] = rng.randint(0, 9)
        else:
            numbers[0] = max(0, numbers[0] - 1)
        vulnerable = _parse_numbers(".".join(str(n) for n in numbers)) > _parse_numbers(observed)
    return ".".join(str(number) for number in numbers[:width]), vulnerable


def build_world(
    n_vulns: int,
    stacks: Sequence[tuple[str, str, str]],
    seed: int,
    *,
    start_date: date,
    epss_dates: Iterable[date],
    kev_share: float = 0.14,
    exploit_share: float = 0.32,
) -> LatentWorld:
    """Draw ``n_vulns`` latent vulnerabilities and their observable evidence.

    ``stacks`` is the pool of ``(vendor, product, version)`` triples actually deployed in
    the generated world, so every CVE is *about* something the scanner can observe; that is
    what gives ``semantic.cpe_match`` a real decision to make.

    KEV and Exploit-DB membership are assigned by ranking the per-CVE propensities and
    taking the configured share, so both remain the minorities they are in the real feeds
    (Gap 7) regardless of how the latent draws happen to fall.
    """
    rng = Random(derive_seed(seed, "world", n_vulns))
    pool = list(stacks) or [("acme", "widget", "1.0.0")]
    drawn: list[tuple[LatentVuln, float, float]] = []
    for index in range(int(n_vulns)):
        template = VULN_TEMPLATES[index % len(VULN_TEMPLATES)]
        stack_entry = pool[rng.randrange(len(pool))]
        published = start_date - timedelta(days=rng.randint(30, 900))
        drawn.append(
            _make_vuln(index, template, stack_entry, rng, year=published.year, published=published)
        )

    total = len(drawn)
    n_kev = int(round(total * clamp01(kev_share)))
    n_exploit = int(round(total * clamp01(exploit_share)))
    kev_ids = {
        item[0].cve_id
        for item in sorted(drawn, key=lambda item: (-item[1], item[0].cve_id))[:n_kev]
    }
    exploit_ids = {
        item[0].cve_id
        for item in sorted(drawn, key=lambda item: (-item[2], item[0].cve_id))[:n_exploit]
    }

    vulns: list[LatentVuln] = []
    for vuln, _kev_propensity, _exploit_propensity in drawn:
        in_kev = vuln.cve_id in kev_ids
        has_exploit = vuln.cve_id in exploit_ids or in_kev
        updates: dict[str, object] = {
            "in_kev": in_kev,
            "kev_date_added": vuln.kev_date_added if in_kev else None,
            "kev_due_date": vuln.kev_due_date if in_kev else None,
            "kev_ransomware": bool(in_kev and vuln.kev_ransomware),
            "exploit_published": vuln.exploit_published if has_exploit else None,
            "exploit_verified": bool(has_exploit and vuln.exploit_verified),
            "exploit_maturity": vuln.exploit_maturity if has_exploit else ExploitMaturity.UNKNOWN,
            "exploit_id": vuln.exploit_id if has_exploit else "",
            "exploit_title": vuln.exploit_title if has_exploit else "",
        }
        candidate = vuln.model_copy(update=updates)
        vulns.append(
            candidate.model_copy(
                update={
                    "reference_urls": _reference_urls(
                        candidate.cve_id,
                        candidate.template.owasp_topic,
                        Random(derive_seed(seed, "refs", candidate.cve_id)),
                        kev=in_kev,
                        exploit_id=candidate.exploit_id or None,
                    )
                }
            )
        )
    return LatentWorld(
        seed=int(seed),
        vulns=tuple(vulns),
        epss_dates=tuple(sorted(set(epss_dates))),
    )
