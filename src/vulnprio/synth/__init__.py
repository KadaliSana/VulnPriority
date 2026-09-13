"""The synthetic world (DESIGN.md 3.11).

The literature review's Gap 10 is that no study validates a prioritisation against
post-remediation counterfactual outcomes, and Gap 3 is that CVSS - the label most studies
fall back on - is an unreliable instrument rather than an outcome. Both gaps have the same
root cause: **there is no public corpus that records which web application findings were
actually exploited, when, and what would have happened had they been patched sooner.**

This package manufactures one. Its defining property, stated once here and honoured
throughout, is that the exploitation oracle is driven by **latent** variables -
``true_exploitability``, ``true_applicability``, ``true_attacker_interest``,
``true_chain_position`` - and never by the features the framework computes. CVSS, EPSS, KEV
membership, exploit records and the attack-graph score are all generated *downstream* of the
same latents, with realistic noise, disagreement and publication lag. A ranker therefore has
to infer the latent structure through the evidence; it cannot simply reproduce the label,
because the label was never a function of the evidence. Without that separation the
evaluation would be circular and would measure nothing at all.

Modules:

* :mod:`~vulnprio.synth.topology` - application surfaces: hosts, routes, links, stacks.
* :mod:`~vulnprio.synth.world` - the latent vulnerability universe and its noisy evidence.
* :mod:`~vulnprio.synth.feeds` - fixtures in the exact shapes ``vulnprio.feeds`` reads.
* :mod:`~vulnprio.synth.pages` - multilingual reference pages, optionally injected.
* :mod:`~vulnprio.synth.oracle` - exploitation events and remediation counterfactuals.
* :mod:`~vulnprio.synth.generator` - :class:`SyntheticDataset`, which writes it all out.
"""

from __future__ import annotations

from vulnprio.synth.feeds import FEED_FILES, write_feed_fixtures
from vulnprio.synth.generator import (
    DEFAULT_SYNTHETIC_ROOT,
    LATENT_FUNCTION_VALUE,
    SyntheticDataset,
    synthetic_config_of,
)
from vulnprio.synth.oracle import (
    HAZARD_WEIGHTS,
    ExploitationEvent,
    ExploitationOracle,
    LatentFinding,
    simulate_exploitation,
)
from vulnprio.synth.pages import (
    FALLBACK_PAYLOADS,
    PAGE_LANGUAGES,
    generate_reference_pages,
    page_for_url,
)
from vulnprio.synth.topology import (
    SECTOR_TECH,
    SECTORS,
    AppSpec,
    RouteTemplate,
    derive_seed,
    generate_app_specs,
    generate_endpoints,
    intended_function,
)
from vulnprio.synth.world import (
    VULN_TEMPLATES,
    EpssPoint,
    LatentVuln,
    LatentWorld,
    VulnTemplate,
    build_world,
)

__all__ = [
    "AppSpec",
    "DEFAULT_SYNTHETIC_ROOT",
    "EpssPoint",
    "ExploitationEvent",
    "ExploitationOracle",
    "FALLBACK_PAYLOADS",
    "FEED_FILES",
    "HAZARD_WEIGHTS",
    "LATENT_FUNCTION_VALUE",
    "LatentFinding",
    "LatentVuln",
    "LatentWorld",
    "PAGE_LANGUAGES",
    "RouteTemplate",
    "SECTORS",
    "SECTOR_TECH",
    "SyntheticDataset",
    "VULN_TEMPLATES",
    "VulnTemplate",
    "build_world",
    "derive_seed",
    "generate_app_specs",
    "generate_endpoints",
    "generate_reference_pages",
    "intended_function",
    "page_for_url",
    "simulate_exploitation",
    "synthetic_config_of",
    "write_feed_fixtures",
]
