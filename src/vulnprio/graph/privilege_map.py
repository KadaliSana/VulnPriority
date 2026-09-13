"""CWE and impact to attack-graph privilege transitions (DESIGN.md 3.7, Gap 9).

An attack graph is only as honest as its edges, and the edge a finding contributes is
entirely determined by two privileges: what the attacker must already hold to run the
exploit, and what they hold afterwards. Getting those from a language model would put the
shape of the graph at the mercy of whatever text the target application chose to serve,
so the mapping is an **explicit operator-tier table** keyed by CWE, and the model's own
``privilege_gained`` is used only as a fallback for CWEs the table does not cover.

That split is what makes the attack graph's trust rule expressible at all
(``vulnprio.graph.attack_graph.edge_evidence_tier``): when the CWE is in this table the
transition rests on curated knowledge and a scanner-observed CWE identifier, both at tier
<= ``TrustTier.SCANNER``; when it is not, the transition rests on whatever tier produced
the exploitability assessment, and an edge asserted only by a reference page or a target
response is rejected.

The table groups CWEs by the *class of control* an exploit of that weakness hands over:

* **Host / OS control (SYSTEM).** Command and code injection, deserialisation of
  untrusted data, unrestricted file upload: the attacker runs code on the host.
* **Application administration (ADMIN).** Authentication and authorisation weaknesses:
  the attacker becomes, or acts as, a privileged application principal.
* **Application data access (USER or higher, decided by CIA impact).** Injection into a
  data store or a file path: how far it goes depends on what the assessment says the
  exploit actually reads, writes or breaks, so the level is derived from the assessed
  confidentiality / integrity / availability impact rather than fixed.
* **Session-level access requiring a victim (USER, user interaction).** Cross-site
  scripting and request forgery ride an authenticated victim's session. The cost of
  needing that victim is already priced into ``p_exploit`` by the attacker model's
  ``w_user_interaction`` term, so it is recorded here but not charged a second time.
* **No privilege gained (NONE).** Information disclosure and configuration weaknesses
  are real findings with real expected loss, but they move nobody up the privilege
  lattice, so they contribute no edge and no chain contribution. Component B, not
  Component C, is where their priority comes from.
"""

from __future__ import annotations

from dataclasses import dataclass

from vulnprio.core.enums import PrivilegeLevel
from vulnprio.core.models import ExploitabilityAssessment

__all__ = [
    "PrivilegeRule",
    "CWE_PRIVILEGE_RULES",
    "SYSTEM_CONTROL_CWES",
    "ADMIN_CONTROL_CWES",
    "DATA_ACCESS_CWES",
    "USER_INTERACTION_CWES",
    "NO_PRIVILEGE_CWES",
    "CIA_ADMIN_THRESHOLD",
    "CIA_SYSTEM_THRESHOLD",
    "cia_derived_privilege",
    "is_known_cwe",
    "rule_for",
    "privileges_for",
]


@dataclass(frozen=True, slots=True)
class PrivilegeRule:
    """One row of the CWE privilege table.

    ``gained`` of ``None`` means "derive the level from the assessed CIA impact" via
    :func:`cia_derived_privilege`; every other value is fixed operator knowledge.
    """

    required: PrivilegeLevel
    gained: PrivilegeLevel | None
    user_interaction: bool = False
    note: str = ""


#: Exploitation yields code execution on the host.
SYSTEM_CONTROL_CWES: tuple[int, ...] = (
    77,    # command injection
    78,    # OS command injection
    94,    # code injection
    434,   # unrestricted upload of file with dangerous type
    502,   # deserialization of untrusted data
)

#: Exploitation yields a privileged application principal.
ADMIN_CONTROL_CWES: tuple[int, ...] = (
    269,   # improper privilege management
    287,   # improper authentication
    306,   # missing authentication for critical function
    798,   # use of hard-coded credentials
    862,   # missing authorization
    863,   # incorrect authorization
)

#: Exploitation yields data-store or file-system access; how far depends on CIA impact.
DATA_ACCESS_CWES: tuple[int, ...] = (
    22,    # path traversal
    89,    # SQL injection
    98,    # PHP remote file inclusion
    611,   # XML external entity
    918,   # server-side request forgery
)

#: Exploitation rides an authenticated victim's session.
USER_INTERACTION_CWES: tuple[int, ...] = (
    79,    # cross-site scripting
    352,   # cross-site request forgery
)

#: Real findings that move nobody up the privilege lattice.
NO_PRIVILEGE_CWES: tuple[int, ...] = (
    16,    # configuration
    200,   # exposure of sensitive information
    209,   # generation of error message containing sensitive information
    548,   # exposure of information through directory listing
)

#: Mean CIA impact at or above which a data-access weakness is treated as administrative.
CIA_ADMIN_THRESHOLD: float = 0.6

#: Mean CIA impact (with matching integrity impact) at or above which a data-access
#: weakness is treated as host control: writing arbitrary content through a file or query
#: path is how a database or traversal bug turns into code execution.
CIA_SYSTEM_THRESHOLD: float = 0.85


def _rules() -> dict[int, PrivilegeRule]:
    """Assemble the table once, so the groups above stay the single source of truth."""
    table: dict[int, PrivilegeRule] = {}
    for cwe in SYSTEM_CONTROL_CWES:
        table[cwe] = PrivilegeRule(
            required=PrivilegeLevel.NONE,
            gained=PrivilegeLevel.SYSTEM,
            note="execution on the host",
        )
    for cwe in ADMIN_CONTROL_CWES:
        # 269 and 863 are *escalation* weaknesses: they presuppose an authenticated
        # principal whose privileges are then mismanaged. 287, 306, 798 and 862 are
        # *bypass* weaknesses reachable without one.
        required = PrivilegeLevel.USER if cwe in (269, 863) else PrivilegeLevel.NONE
        table[cwe] = PrivilegeRule(
            required=required,
            gained=PrivilegeLevel.ADMIN,
            note="privileged application principal",
        )
    for cwe in DATA_ACCESS_CWES:
        table[cwe] = PrivilegeRule(
            required=PrivilegeLevel.NONE,
            gained=None,
            note="data or file access; level derived from assessed CIA impact",
        )
    for cwe in USER_INTERACTION_CWES:
        table[cwe] = PrivilegeRule(
            required=PrivilegeLevel.NONE,
            gained=PrivilegeLevel.USER,
            user_interaction=True,
            note="rides an authenticated victim's session",
        )
    for cwe in NO_PRIVILEGE_CWES:
        table[cwe] = PrivilegeRule(
            required=PrivilegeLevel.NONE,
            gained=PrivilegeLevel.NONE,
            note="no privilege gained; priority comes from Component B",
        )
    return table


#: The explicit table. Operator tier: it is code, not model output, and not configuration
#: that untrusted content can reach.
CWE_PRIVILEGE_RULES: dict[int, PrivilegeRule] = _rules()


def cia_derived_privilege(exploitability: ExploitabilityAssessment) -> PrivilegeLevel:
    """Privilege implied by an assessed confidentiality/integrity/availability impact.

    A monotone ladder: a data-access weakness always yields at least ``USER``, yields
    ``ADMIN`` once the mean CIA impact says the attacker reaches material amounts of the
    application's data, and yields ``SYSTEM`` only when the impact is near-total *and*
    includes integrity, since reading everything is not the same as being able to write
    to the host.
    """
    mean = exploitability.impact_cia_mean
    if mean >= CIA_SYSTEM_THRESHOLD and exploitability.impact_i >= CIA_SYSTEM_THRESHOLD:
        return PrivilegeLevel.SYSTEM
    if mean >= CIA_ADMIN_THRESHOLD:
        return PrivilegeLevel.ADMIN
    return PrivilegeLevel.USER


def is_known_cwe(cwe_id: int | None) -> bool:
    """True when the privilege transition for this CWE is operator knowledge.

    The attack-graph trust rule turns on this: a transition taken from the table does not
    depend on any untrusted text, while a transition taken from the fallback does.
    """
    return cwe_id is not None and int(cwe_id) in CWE_PRIVILEGE_RULES


def rule_for(cwe_id: int | None) -> PrivilegeRule | None:
    """The table row for a CWE, or ``None`` when the CWE is not covered."""
    if cwe_id is None:
        return None
    return CWE_PRIVILEGE_RULES.get(int(cwe_id))


def privileges_for(
    cwe_id: int | None,
    exploitability: ExploitabilityAssessment,
) -> tuple[PrivilegeLevel, PrivilegeLevel]:
    """``(privileges_required, privilege_gained)`` for one finding.

    Resolution order:

    1. The explicit CWE table, with the gained level derived from CIA impact for the
       data-access group.
    2. For an unknown or absent CWE, the assessment's own
       ``privileges_required`` / ``privilege_gained``. This is the only path on which the
       shape of the graph can depend on a model, and
       :func:`vulnprio.graph.attack_graph.edge_evidence_tier` treats an edge built this
       way as carrying the assessment's trust tier.

    The returned pair is normalised so that ``gained >= required``: a rule can never
    describe a transition that *loses* privilege, because the downward direction is
    already covered at probability 1 by the graph's privilege-implication edges. When the
    two are equal the finding is non-escalating and the builder gives it no edge.
    """
    rule = rule_for(cwe_id)
    if rule is not None:
        required = rule.required
        gained = rule.gained if rule.gained is not None else cia_derived_privilege(exploitability)
    else:
        required = exploitability.privileges_required
        gained = exploitability.privilege_gained
    return required, PrivilegeLevel(max(int(required), int(gained)))
