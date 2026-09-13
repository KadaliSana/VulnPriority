"""Influence budgets, evidence floors and corroboration (DESIGN.md 3.3 item 7).

Architecture rule 1 says untrusted text may move a feature only as far as its influence
budget allows. This module is where that accounting actually happens, and where the
sentence "KEV membership cannot be argued away by a blog post" becomes an executable
constraint.

Three mechanisms, all recorded in :class:`TrustSummary`:

1. **Budgets.** A tier may move any normalised feature by at most
   ``SandboxConfig.influence_budget[tier]``. A larger requested delta is clamped, and the
   binding cap is written to ``caps_applied`` so an ablation can show what was withheld.
2. **Floors.** Tier <= 1 evidence (CISA KEV membership, verified functional exploit code)
   implies a lower bound on ``p_exploit``. Untrusted text may not push the probability
   below it. The floor is applied *relative to the trusted baseline*: it can undo an
   untrusted argument, never invent confidence the curated feeds did not supply.
3. **Corroboration.** An untrusted claim that agrees with tier <= 1 evidence is not the
   same risk as one that contradicts it, so it is granted
   ``SandboxConfig.corroborated_budget`` instead of its tier budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from vulnprio.core.config import SandboxConfig
from vulnprio.core.enums import ExploitMaturity, TrustTier
from vulnprio.core.errors import InfluenceBudgetExceeded
from vulnprio.core.models import TrustSummary, VulnIntel

__all__ = [
    "KEV_FLOOR_P",
    "KEV_RANSOMWARE_FLOOR_P",
    "VERIFIED_FUNCTIONAL_FLOOR_P",
    "VERIFIED_WEAPONIZED_FLOOR_P",
    "InfluenceRecord",
    "TrustLedger",
    "compute_floor",
]

#: A CVE on the CISA Known Exploited Vulnerabilities catalogue is, by definition, being
#: exploited. No amount of untrusted prose may argue ``p_exploit`` below this.
KEV_FLOOR_P: float = 0.50

#: KEV entries flagged for ransomware campaign use carry a higher floor still.
KEV_RANSOMWARE_FLOOR_P: float = 0.70

#: Verified, functional exploit code exists in a curated index (tier 1).
VERIFIED_FUNCTIONAL_FLOOR_P: float = 0.35

#: Verified, weaponised exploit code: packaged and point-and-click.
VERIFIED_WEAPONIZED_FLOOR_P: float = 0.50


@dataclass(frozen=True)
class InfluenceRecord:
    """One attempt by a tier to move one feature, and what was actually allowed."""

    feature: str
    tier: TrustTier
    requested: float
    applied: float
    budget: float
    corroborated: bool = False

    @property
    def was_capped(self) -> bool:
        return abs(self.requested) > abs(self.applied) + 1e-12


def compute_floor(
    intel: tuple[VulnIntel, ...],
    as_of: date,
) -> tuple[float, str]:
    """Lower bound on ``p_exploit`` implied by tier <= 1 evidence, with its reason.

    Only curated feeds count. Component A's own maturity judgement is deliberately not a
    floor source: it is downstream of untrusted text, so letting it set a floor would let
    an injection manufacture one.
    """
    floor = 0.0
    reason = ""

    def raise_to(value: float, why: str) -> None:
        nonlocal floor, reason
        if value > floor:
            floor, reason = value, why

    for record in intel:
        if record.as_of > as_of:
            continue
        kev = record.kev
        if kev is not None and kev.in_kev and not (kev.date_added is not None and kev.date_added > as_of):
            raise_to(KEV_FLOOR_P, f"{record.cve_id} is in CISA KEV")
            if kev.known_ransomware_use:
                raise_to(KEV_RANSOMWARE_FLOOR_P, f"{record.cve_id} is in CISA KEV with known ransomware use")
        for exploit in record.exploits:
            if not exploit.verified:
                continue
            if exploit.published is not None and exploit.published > as_of:
                continue
            if exploit.maturity >= ExploitMaturity.WEAPONIZED:
                raise_to(VERIFIED_WEAPONIZED_FLOOR_P, f"{record.cve_id} has verified weaponised exploit code")
            elif exploit.maturity >= ExploitMaturity.FUNCTIONAL:
                raise_to(VERIFIED_FUNCTIONAL_FLOOR_P, f"{record.cve_id} has verified functional exploit code")

    return floor, reason


class TrustLedger:
    """Per-finding accounting of who moved what, by how much, and what stopped them.

    Mutable on purpose: it is the one side-effecting object in Component B, and it exists
    so that the enricher's arithmetic can stay pure while the security decisions stay in
    one auditable place.
    """

    def __init__(
        self,
        finding_id: str,
        sandbox: SandboxConfig | None = None,
        *,
        strict: bool = False,
    ) -> None:
        """``strict`` turns a budget breach into :class:`InfluenceBudgetExceeded`.

        The pipeline runs non-strict (clamp and record) because a single over-eager
        reference page should degrade one feature, not abort a scan; the adversarial
        evaluation runs strict, where an unclamped attempt is the finding itself.
        """
        self.finding_id = finding_id
        self.sandbox = sandbox if sandbox is not None else SandboxConfig()
        self.strict = strict
        self._records: list[InfluenceRecord] = []
        self._conflicts: list[str] = []
        self._max_tier: TrustTier = TrustTier.OPERATOR
        self._floor: float = 0.0
        self._floor_reason: str = ""
        #: The floor that was actually enforceable once the trusted reference was known.
        #: ``None`` until :meth:`apply_floor` runs; the nominal floor is ``_floor``.
        self._effective_floor: float | None = None
        self._signals: int = 0
        self._canary_leaked: bool = False

    # -- observations -------------------------------------------------------

    @property
    def records(self) -> tuple[InfluenceRecord, ...]:
        """Every influence attempt, in the order it was made."""
        return tuple(self._records)

    @property
    def floor_p_exploit(self) -> float:
        """Current tier <= 1 floor on ``p_exploit``."""
        return self._floor

    @property
    def nominal_floor(self) -> float:
        """The floor the curated evidence implies, before the trusted reference bounds it."""
        return self._floor

    @property
    def effective_floor(self) -> float:
        """The floor that was actually enforceable, once the trusted reference was known.

        This is what a downstream consumer must compare ``p_exploit`` against. The floor
        mechanism exists to undo an untrusted argument, never to raise a finding above what
        the curated feeds and the attacker model already justified, so a finding the
        attacker model scores below the nominal floor is not a breach.
        """
        return self._floor if self._effective_floor is None else self._effective_floor

    @property
    def floor_reason(self) -> str:
        """Why the floor is where it is; empty when there is no floor."""
        return self._floor_reason

    @property
    def max_tier_used(self) -> TrustTier:
        """Least trusted tier that contributed anything to this finding."""
        return self._max_tier

    def note_tier(self, tier: TrustTier) -> None:
        """Record that content of this tier was consulted."""
        if tier > self._max_tier:
            self._max_tier = tier

    def note_signals(self, count: int) -> None:
        """Add injection signals raised by the sandbox while preparing this finding."""
        self._signals += max(0, int(count))

    def note_canary(self, leaked: bool) -> None:
        """Record a canary leak; sticky, because one leak is one too many."""
        self._canary_leaked = self._canary_leaked or bool(leaked)

    def note_conflict(self, message: str) -> None:
        """Record a disagreement worth surfacing in the explanation."""
        if message and message not in self._conflicts:
            self._conflicts.append(message)

    def set_floor(self, floor: float, reason: str = "") -> None:
        """Raise the floor. Floors never fall: a later, weaker source cannot lower one."""
        value = min(max(float(floor), 0.0), 1.0)
        if value > self._floor:
            self._floor = value
            self._floor_reason = reason

    # -- budget enforcement -------------------------------------------------

    def budget_for(self, tier: TrustTier, corroborated: bool = False) -> float:
        """Maximum absolute movement this tier may apply to a normalised feature."""
        base = float(self.sandbox.influence_budget.get(tier, 0.0))
        if corroborated and tier in (TrustTier.REFERENCE_PAGE, TrustTier.TARGET_CONTENT):
            return max(base, float(self.sandbox.corroborated_budget))
        return base

    def record(
        self,
        feature: str,
        tier: TrustTier,
        delta: float,
        *,
        corroborated: bool = False,
    ) -> float:
        """Clamp ``delta`` to the tier's budget, book it, and return what may be applied."""
        self.note_tier(tier)
        budget = self.budget_for(tier, corroborated)
        requested = float(delta)
        applied = min(max(requested, -budget), budget)

        entry = InfluenceRecord(
            feature=feature,
            tier=tier,
            requested=requested,
            applied=applied,
            budget=budget,
            corroborated=corroborated,
        )
        self._records.append(entry)

        if entry.was_capped:
            message = (
                f"tier {tier.name} tried to move {feature} by {requested:+.3f}; "
                f"budget {budget:.3f} applied {applied:+.3f}"
            )
            self.note_conflict(message)
            if self.strict:
                raise InfluenceBudgetExceeded(f"{self.finding_id}: {message}")
        return applied

    # -- floor enforcement --------------------------------------------------

    def apply_floor(self, p_value: float, trusted_reference: float | None = None) -> float:
        """Hold ``p_value`` at or above the tier <= 1 floor.

        ``trusted_reference`` is the probability computed *without* untrusted movement.
        The effective floor is ``min(floor, trusted_reference)``, so the mechanism can
        only undo an untrusted argument - it never raises a finding above what the
        curated feeds and the attacker model already justified. With no untrusted
        movement the two are equal and this is a no-op.
        """
        if self.sandbox.allow_downgrade_below_floor or self._floor <= 0.0:
            return float(p_value)

        effective = self._floor if trusted_reference is None else min(self._floor, float(trusted_reference))
        # Record what the floor could actually enforce, not what the evidence nominally
        # implies. A consumer that compares p_exploit against the nominal floor concludes
        # the floor was breached on every finding where the attacker model legitimately
        # scored below it, which is a false alarm rather than a finding.
        self._effective_floor = effective
        if float(p_value) >= effective:
            return float(p_value)

        self.note_conflict(
            f"p_exploit held at the tier<=1 floor {effective:.3f} "
            f"(untrusted evidence argued {float(p_value):.3f}; {self._floor_reason or 'curated feed evidence'})"
        )
        return effective

    # -- output -------------------------------------------------------------

    def influence_used(self) -> dict[str, float]:
        """Total absolute movement actually applied, per feature."""
        used: dict[str, float] = {}
        for entry in self._records:
            if entry.applied:
                used[entry.feature] = used.get(entry.feature, 0.0) + abs(entry.applied)
        return used

    def caps_applied(self) -> dict[str, float]:
        """Binding cap per feature, for features where the budget actually bound."""
        caps: dict[str, float] = {}
        for entry in self._records:
            if entry.was_capped:
                caps[entry.feature] = min(caps.get(entry.feature, entry.budget), entry.budget)
        return caps

    def corroborated(self) -> bool:
        """True when at least one untrusted claim agreed with tier <= 1 evidence."""
        return any(entry.corroborated and entry.applied for entry in self._records)

    def summary(self) -> TrustSummary:
        """Immutable view of the ledger for attachment to an :class:`EnrichedFinding`."""
        return TrustSummary(
            max_tier_used=self._max_tier,
            injection_signal_count=self._signals,
            conflicts=tuple(self._conflicts),
            influence_used=self.influence_used(),
            caps_applied=self.caps_applied(),
            # The NOMINAL floor: what the curated evidence implies. Consumers that want to
            # know whether the floor was actually breached must read ``conflicts``, which the
            # ledger writes only when it held a value up; re-deriving the comparison from this
            # number without the trusted reference reports a breach on every finding the
            # attacker model legitimately scores below it.
            floor_p_exploit=self._floor,
            corroborated=self.corroborated(),
            canary_leaked=self._canary_leaked,
        )
