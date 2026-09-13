"""Component B, attacker half: the explicit adversary and P(exploit | evidence) (Gap 2).

``build_evidence`` turns a finding plus its intelligence and Component A assessments
into the named evidence vector of DESIGN.md 3.6; ``p_exploit`` weights that vector with
an ``AttackerModel`` and returns a fully audited ``ExploitLikelihood``.
"""

from __future__ import annotations

from vulnprio.attacker.likelihood import (
    EVIDENCE_FUNCTIONS,
    EVIDENCE_TERMS,
    build_evidence,
    epss_logit_term,
    function_from_ordinal,
    function_ordinal,
    logit,
)
from vulnprio.attacker.model import (
    HORIZON_REFERENCE_DAYS,
    LOG_ODDS_TERMS,
    TERM_LABELS,
    explain_terms,
    horizon_factor,
    log_odds_terms,
    p_exploit,
    sigmoid,
)
from vulnprio.attacker.presets import (
    DEFAULT_PRESET,
    PRESET_DIR,
    list_presets,
    load_all_presets,
    load_preset,
)

__all__ = [
    "EVIDENCE_FUNCTIONS",
    "EVIDENCE_TERMS",
    "build_evidence",
    "epss_logit_term",
    "function_from_ordinal",
    "function_ordinal",
    "logit",
    "HORIZON_REFERENCE_DAYS",
    "LOG_ODDS_TERMS",
    "TERM_LABELS",
    "explain_terms",
    "horizon_factor",
    "log_odds_terms",
    "p_exploit",
    "sigmoid",
    "DEFAULT_PRESET",
    "PRESET_DIR",
    "list_presets",
    "load_all_presets",
    "load_preset",
]
