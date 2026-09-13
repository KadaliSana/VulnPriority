"""The exploitation oracle: ground truth no public corpus provides (DESIGN.md 3.11, Gap 10).

Two things the literature review says nobody has, and that this module supplies:

1. **Confirmed exploitation**, per finding, with a date. Real evaluations proxy it with
   CVSS or with KEV membership, and both are instruments rather than outcomes.
2. **Counterfactual remediation outcomes.** "Would this finding have been exploited had we
   not fixed it by date *D*?" is unanswerable in the field - you only ever observe one arm -
   and it is exactly what the longitudinal simulation needs in order to say that a policy
   *prevented* something rather than merely *ordered things differently*.

**The oracle reads only latent variables.** Its hazard is a function of
``true_exploitability``, ``true_attacker_interest``, ``true_chain_position``, whether the
defect genuinely applies to the deployment, the latent exposure and value of the endpoint,
and the configured :class:`~vulnprio.core.models.AttackerModel`. It never reads CVSS, EPSS,
KEV, the attack-graph score, the assessments Component A produced or any feature the ranker
sees. Those are all *downstream* of the same latents, which is why a good ranker can
approach the oracle without ever being handed it - and why a circular evaluation is
impossible here by construction rather than by care.

The population exploitation rate is calibrated to ``SyntheticConfig.base_exploit_rate`` by
solving for the intercept that makes the mean hazard equal it, so positives stay the rare
class the whole of Gap 7 is about.
"""

from __future__ import annotations

from datetime import date, timedelta
from random import Random
from typing import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from vulnprio.core.enums import EndpointFunction, PrivilegeLevel
from vulnprio.core.models import AttackerModel
from vulnprio.synth.topology import derive_seed
from vulnprio.synth.world import LatentWorld, clamp01, sigmoid

__all__ = [
    "HAZARD_WEIGHTS",
    "LatentFinding",
    "ExploitationEvent",
    "ExploitationOracle",
    "simulate_exploitation",
]

#: Weights of the latent hazard. Named so that the oracle's model of the world is as
#: auditable as the framework's model of the attacker, and so a reader can verify at a
#: glance that no observable feature appears in it.
HAZARD_WEIGHTS: dict[str, float] = {
    "true_exploitability": 2.60,
    "true_attacker_interest": 1.90,
    "true_chain_position": 1.40,
    "latent_exposure": 1.20,
    "latent_asset_value": 1.05,
    "chain_depth_penalty": -0.55,
    "not_applicable_penalty": -3.20,
    "attacker_skill": 1.30,
    "attacker_resources": 0.80,
}


class LatentFinding(BaseModel):
    """One generated finding, described only by variables the framework cannot see."""

    model_config = ConfigDict(frozen=True)

    finding_id: str
    scan_id: str
    app_id: str
    endpoint_id: str
    defect_key: str                       # identity of the underlying defect across scans
    cve_id: str | None = None
    cwe_id: int | None = None
    function: EndpointFunction = EndpointFunction.UNKNOWN
    observed_at: date

    # --- latent context (never observable) -------------------------------
    latent_exposure: float = Field(0.5, ge=0.0, le=1.0)
    latent_asset_value: float = Field(0.5, ge=0.0, le=1.0)
    latent_chain_depth: int = Field(0, ge=0)
    latent_applies: bool = True
    latent_privilege_gained: PrivilegeLevel = PrivilegeLevel.USER


class ExploitationEvent(BaseModel):
    """What the world did to one finding, and what it would have done.

    ``exploit_date`` is the day the attacker reached this defect in the counterfactual
    world where nobody remediated it. When it is ``None`` the defect was never reached
    within the horizon, and no remediation schedule could have changed that.
    """

    model_config = ConfigDict(frozen=True)

    finding_id: str
    scan_id: str
    app_id: str
    defect_key: str
    cve_id: str | None = None
    exploited: bool = False
    exploit_date: date | None = None
    hazard: float = Field(0.0, ge=0.0, le=1.0)
    observed_at: date


class ExploitationOracle(BaseModel):
    """Latent ground truth for a generated dataset.

    Consumers: ``eval/labels.py`` (``LabelSource.SYNTHETIC_ORACLE``) for the exploited flag
    and its first-evidence date, and ``eval/simulation.py`` for the counterfactual that
    turns a remediation schedule into prevented exploitations and exposure days.
    """

    model_config = ConfigDict(frozen=True)

    seed: int = 42
    horizon_days: int = Field(180, ge=1)
    base_rate: float = Field(0.06, ge=0.0, le=1.0)
    attacker: str = "opportunistic"
    intercept: float = 0.0
    events: tuple[ExploitationEvent, ...] = ()

    # -- lookups ------------------------------------------------------------

    def index(self) -> dict[str, ExploitationEvent]:
        return {event.finding_id: event for event in self.events}

    def event(self, finding_id: str) -> ExploitationEvent | None:
        for item in self.events:
            if item.finding_id == finding_id:
                return item
        return None

    def exploited(self, finding_id: str) -> bool:
        """Was this finding exploited within the horizon (nothing having been remediated)?"""
        found = self.event(finding_id)
        return bool(found and found.exploited)

    def exploit_date(self, finding_id: str) -> date | None:
        found = self.event(finding_id)
        return found.exploit_date if found else None

    def positives(self) -> set[str]:
        return {event.finding_id for event in self.events if event.exploited}

    def positive_rate(self) -> float:
        return (len(self.positives()) / len(self.events)) if self.events else 0.0

    def label_map(self) -> dict[str, bool]:
        """``finding_id -> exploited``, the shape ``eval/labels.py`` consumes."""
        return {event.finding_id: event.exploited for event in self.events}

    def first_evidence_dates(self) -> dict[str, date]:
        return {
            event.finding_id: event.exploit_date
            for event in self.events
            if event.exploited and event.exploit_date is not None
        }

    # -- counterfactuals ----------------------------------------------------

    def would_be_exploited_if_not_remediated_by(
        self, finding_id: str, remediated_on: date | None
    ) -> bool:
        """The Gap 10 counterfactual: did remediating on ``remediated_on`` avert an exploit?

        True when the world had an exploitation scheduled for this defect *after* the
        remediation date - so fixing it when we did prevented a real event. False when the
        defect was never going to be reached, or was already reached before the fix landed
        (in which case the remediation was too late to matter).
        """
        found = self.event(finding_id)
        if found is None or not found.exploited or found.exploit_date is None:
            return False
        if remediated_on is None:
            return True
        return found.exploit_date > remediated_on

    #: DESIGN.md names the counterfactual this way; the long name above states the semantics.
    prevented_by_remediation = would_be_exploited_if_not_remediated_by

    def exploited_by(self, finding_id: str, cutoff: date) -> bool:
        """Was the finding exploited at or before ``cutoff``, absent remediation?"""
        found = self.event(finding_id)
        return bool(
            found and found.exploited and found.exploit_date is not None and found.exploit_date <= cutoff
        )

    def exposure_days(
        self, finding_id: str, remediated_on: date | None, horizon_end: date
    ) -> float:
        """Days this finding stayed open, from observation to remediation or the horizon."""
        found = self.event(finding_id)
        if found is None:
            return 0.0
        end = min(remediated_on, horizon_end) if remediated_on is not None else horizon_end
        return float(max(0, (end - found.observed_at).days))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _hazard_score(
    latent: LatentFinding,
    world_index: Mapping[str, object],
    attacker: AttackerModel,
) -> float:
    """Latent log-odds of this defect being reached within the horizon.

    Reads only latent variables and operator-tier attacker parameters. Findings with no CVE
    still have latent exploitability: they are application-logic defects, and the generator
    stores their latents on the finding rather than on a vulnerability record.
    """
    weights = HAZARD_WEIGHTS
    vuln = world_index.get(latent.cve_id) if latent.cve_id else None
    if vuln is not None:
        exploitability = float(getattr(vuln, "true_exploitability", 0.5))
        interest = float(getattr(vuln, "true_attacker_interest", 0.5))
        chain_position = float(getattr(vuln, "true_chain_position", 0.5))
    else:
        # No CVE: the latent structure lives in the endpoint context alone.
        exploitability = latent.latent_exposure * 0.5 + latent.latent_asset_value * 0.2 + 0.25
        interest = latent.latent_asset_value
        chain_position = clamp01(float(int(latent.latent_privilege_gained)) / 3.0)

    preference = float(attacker.target_preference.get(latent.function, 1.0))
    score = (
        weights["true_exploitability"] * exploitability
        + weights["true_attacker_interest"] * interest * preference
        + weights["true_chain_position"] * chain_position
        + weights["latent_exposure"] * latent.latent_exposure
        + weights["latent_asset_value"] * latent.latent_asset_value
        + weights["chain_depth_penalty"] * float(latent.latent_chain_depth)
        + weights["attacker_skill"] * float(attacker.skill)
        + weights["attacker_resources"] * float(attacker.resources)
    )
    if not latent.latent_applies:
        score += weights["not_applicable_penalty"]
    return score


def _solve_intercept(scores: Sequence[float], target_rate: float) -> float:
    """Intercept ``c`` with ``mean(sigmoid(z - c)) == target_rate``.

    Bisection on a monotone function, so the answer is unique and the same on every
    platform. Calibrating rather than hand-tuning a constant is what keeps the positive
    rate equal to the configured ``base_exploit_rate`` no matter how the latents fall.
    """
    if not scores:
        return 0.0
    target = min(max(float(target_rate), 1e-6), 1.0 - 1e-6)

    def mean_rate(offset: float) -> float:
        return sum(sigmoid(value - offset) for value in scores) / len(scores)

    low, high = -40.0, 40.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if mean_rate(middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def simulate_exploitation(
    latents: Iterable[LatentFinding],
    world: LatentWorld,
    attacker: AttackerModel,
    *,
    seed: int = 42,
    base_rate: float = 0.06,
    horizon_days: int = 180,
    lag_bounds: tuple[int, int] = (3, 45),
) -> ExploitationOracle:
    """Run the world forward and record who got exploited, when, and counterfactually.

    One draw per *defect*, not per finding: the same defect observed in three consecutive
    scans is one event in the world, and all three findings inherit it. Without that, a
    longitudinal simulation would be able to "prevent" an exploitation that a later scan
    then re-rolled.
    """
    items = list(latents)
    index = world.index()

    scores = [_hazard_score(item, index, attacker) for item in items]

    # Collapse to defects, keeping the earliest observation as the clock's origin. The
    # intercept is calibrated over *defects*, because that is the unit the Bernoulli draw
    # happens on; calibrating over findings would leave the realised positive rate at the
    # mercy of how many scans happened to re-observe each defect.
    by_defect: dict[str, list[int]] = {}
    for position, item in enumerate(items):
        by_defect.setdefault(item.defect_key, []).append(position)
    defect_keys = sorted(by_defect)
    defect_scores = [
        max(scores[position] for position in by_defect[key]) for key in defect_keys
    ]
    intercept = _solve_intercept(defect_scores, base_rate)

    minimum_lag, maximum_lag = int(lag_bounds[0]), int(max(lag_bounds))
    maximum_lag = max(minimum_lag + 1, min(maximum_lag, int(horizon_days)))

    outcomes: dict[str, tuple[bool, date | None, float]] = {}
    for defect_key, defect_score in zip(defect_keys, defect_scores):
        positions = by_defect[defect_key]
        hazard = clamp01(sigmoid(defect_score - intercept))
        rng = Random(derive_seed(seed, "oracle", defect_key))
        exploited = rng.random() < hazard
        exploit_date: date | None = None
        if exploited:
            origin = min(items[position].observed_at for position in positions)
            # A more attractive defect is reached sooner: the lag's mean shrinks with the
            # hazard, bounded by the configured label lag and the horizon.
            mean_lag = minimum_lag + (maximum_lag - minimum_lag) * (1.0 - hazard)
            lag = int(round(rng.expovariate(1.0 / max(1.0, mean_lag))))
            lag = max(minimum_lag, min(int(horizon_days), lag))
            exploit_date = origin + timedelta(days=lag)
        outcomes[defect_key] = (exploited, exploit_date, hazard)

    events: list[ExploitationEvent] = []
    for position, item in enumerate(items):
        exploited, exploit_date, hazard = outcomes[item.defect_key]
        events.append(
            ExploitationEvent(
                finding_id=item.finding_id,
                scan_id=item.scan_id,
                app_id=item.app_id,
                defect_key=item.defect_key,
                cve_id=item.cve_id,
                exploited=exploited,
                exploit_date=exploit_date,
                hazard=round(hazard, 6),
                observed_at=item.observed_at,
            )
        )
    return ExploitationOracle(
        seed=int(seed),
        horizon_days=int(horizon_days),
        base_rate=float(base_rate),
        attacker=attacker.name,
        intercept=round(intercept, 6),
        events=tuple(events),
    )
