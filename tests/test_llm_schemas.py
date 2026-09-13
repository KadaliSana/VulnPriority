"""The bounded output schemas and the baseline -> core-model converters.

These tests are the type-level half of the injection defence: if a schema accepts an
out-of-range number, an unknown category or an extra field, every later guard is
guarding something that already got through.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vulnpriority.core.enums import (
    ApplicabilityVerdict,
    AttackComplexity,
    EndpointFunction,
    ExploitMaturity,
    LLMBackendKind,
    PrivilegeLevel,
    UserInteraction,
    VersionMatch,
)
from vulnpriority.core.models import (
    ApplicabilityAssessment,
    AssetCriticality,
    ExploitabilityAssessment,
    LLMAudit,
)
from vulnpriority.llm.schemas import (
    MAX_EVIDENCE_SPAN_CHARS,
    MAX_EVIDENCE_SPANS,
    MAX_RATIONALE_CHARS,
    ApplicabilityOut,
    AssetCriticalityOut,
    ExploitabilityOut,
    SCHEMA_FOR_TASK,
    schema_for_task,
    task_for_schema,
    to_applicability,
    to_asset_criticality,
    to_exploitability,
)

ALL_SCHEMAS = (AssetCriticalityOut, ExploitabilityOut, ApplicabilityOut)


@pytest.mark.parametrize("schema", ALL_SCHEMAS)
def test_extra_fields_are_forbidden(schema) -> None:
    with pytest.raises(ValidationError):
        schema(priority_override=10)


@pytest.mark.parametrize("schema", ALL_SCHEMAS)
def test_rationale_and_spans_are_capped(schema) -> None:
    with pytest.raises(ValidationError):
        schema(rationale="x" * (MAX_RATIONALE_CHARS + 1))
    with pytest.raises(ValidationError):
        schema(evidence_spans=("x",) * (MAX_EVIDENCE_SPANS + 1))
    with pytest.raises(ValidationError):
        schema(evidence_spans=("x" * (MAX_EVIDENCE_SPAN_CHARS + 1),))
    ok = schema(
        rationale="x" * MAX_RATIONALE_CHARS,
        evidence_spans=("x" * MAX_EVIDENCE_SPAN_CHARS,) * MAX_EVIDENCE_SPANS,
    )
    assert len(ok.evidence_spans) == MAX_EVIDENCE_SPANS


@pytest.mark.parametrize(
    "schema,field",
    [
        (AssetCriticalityOut, "criticality"),
        (AssetCriticalityOut, "data_sensitivity"),
        (AssetCriticalityOut, "exposure"),
        (AssetCriticalityOut, "confidence"),
        (ExploitabilityOut, "exploit_feasibility"),
        (ExploitabilityOut, "impact_c"),
        (ExploitabilityOut, "impact_i"),
        (ExploitabilityOut, "impact_a"),
        (ApplicabilityOut, "p_applicable"),
    ],
)
@pytest.mark.parametrize("bad", [-0.01, 1.01, 42.0, -1e9])
def test_numeric_fields_are_bounded_to_unit_interval(schema, field, bad) -> None:
    with pytest.raises(ValidationError):
        schema(**{field: bad})


@pytest.mark.parametrize(
    "schema,field,bad",
    [
        (AssetCriticalityOut, "function", "crown_jewel"),
        (ExploitabilityOut, "attack_complexity", "impossible"),
        (ExploitabilityOut, "exploit_maturity", 9),
        (ExploitabilityOut, "privileges_required", 7),
        (ExploitabilityOut, "user_interaction", "maybe"),
        (ApplicabilityOut, "verdict", "probably"),
    ],
)
def test_categorical_fields_are_closed_sets(schema, field, bad) -> None:
    with pytest.raises(ValidationError):
        schema(**{field: bad})


def test_applicability_schema_cannot_express_a_version_verdict() -> None:
    """Version evidence is settled deterministically; the model may not touch it."""
    assert "version_match" not in ApplicabilityOut.model_fields
    with pytest.raises(ValidationError):
        ApplicabilityOut(version_match=VersionMatch.MATCH)


def test_schema_task_registry_is_bidirectional() -> None:
    for task, schema in SCHEMA_FOR_TASK.items():
        assert schema_for_task(task) is schema
        assert task_for_schema(schema) == task


def test_schemas_are_frozen() -> None:
    out = AssetCriticalityOut(criticality=0.4)
    with pytest.raises(ValidationError):
        out.criticality = 0.9


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def _asset_baseline() -> AssetCriticality:
    return AssetCriticality(
        endpoint_id="ep_login",
        function=EndpointFunction.AUTH,
        criticality=0.6,
        data_sensitivity=0.5,
        exposure=1.0,
        evidence_features={"param_count": 2.0},
    )


def test_to_asset_criticality_keeps_identity_and_structural_features() -> None:
    baseline = _asset_baseline()
    audit = LLMAudit(backend=LLMBackendKind.HEURISTIC, model="heuristic-v1")
    out = AssetCriticalityOut(
        function=EndpointFunction.ADMIN,
        criticality=0.9,
        data_sensitivity=0.8,
        exposure=0.25,
        is_admin_surface=True,
        rationale="admin path",
        evidence_spans=("/admin",),
    )

    merged = to_asset_criticality(baseline, out, audit=audit)

    assert merged.endpoint_id == "ep_login"
    assert merged.evidence_features == {"param_count": 2.0}
    assert merged.function == EndpointFunction.ADMIN
    assert merged.criticality == 0.9
    assert merged.is_admin_surface is True
    assert merged.audit is audit


def test_converters_return_the_baseline_when_there_is_no_model_output() -> None:
    baseline = _asset_baseline()
    audit = LLMAudit(backend=LLMBackendKind.HEURISTIC, model="heuristic-v1", fell_back_to_heuristic=True)

    merged = to_asset_criticality(baseline, None, audit=audit)

    assert merged.criticality == baseline.criticality
    assert merged.function == baseline.function
    assert merged.audit is audit


def test_to_exploitability_falls_back_to_baseline_preconditions() -> None:
    baseline = ExploitabilityAssessment(
        finding_id="f_sqli",
        exploit_feasibility=0.4,
        preconditions=("attacker holds USER privileges",),
    )
    out = ExploitabilityOut(
        exploit_feasibility=0.8,
        exploit_maturity=ExploitMaturity.FUNCTIONAL,
        attack_complexity=AttackComplexity.LOW,
        privileges_required=PrivilegeLevel.NONE,
        user_interaction=UserInteraction.NONE,
        impact_c=0.9,
        privilege_gained=PrivilegeLevel.USER,
    )

    merged = to_exploitability(baseline, out)

    assert merged.finding_id == "f_sqli"
    assert merged.preconditions == ("attacker holds USER privileges",)
    assert merged.exploit_feasibility == 0.8
    assert merged.exploit_maturity == ExploitMaturity.FUNCTIONAL


def test_to_applicability_cannot_overturn_a_version_mismatch() -> None:
    """DESIGN 3.5: a tier <= 1 MISMATCH is authoritative."""
    baseline = ApplicabilityAssessment(
        finding_id="f_sqli",
        verdict=ApplicabilityVerdict.NOT_APPLICABLE,
        p_applicable=0.08,
        version_match=VersionMatch.MISMATCH,
        preconditions_met={"version_in_affected_range": False},
    )
    out = ApplicabilityOut(
        verdict=ApplicabilityVerdict.APPLICABLE,
        p_applicable=0.99,
        preconditions_met={"plugin_enabled": True},
    )

    merged = to_applicability(baseline, out)

    assert merged.version_match == VersionMatch.MISMATCH
    assert merged.verdict == ApplicabilityVerdict.NOT_APPLICABLE
    assert merged.p_applicable == 0.08
    assert merged.preconditions_met == {"version_in_affected_range": False, "plugin_enabled": True}


def test_to_applicability_accepts_a_model_verdict_when_versions_are_unknown() -> None:
    baseline = ApplicabilityAssessment(finding_id="f_xss", p_applicable=0.5, version_match=VersionMatch.UNKNOWN)
    out = ApplicabilityOut(verdict=ApplicabilityVerdict.APPLICABLE, p_applicable=0.82)

    merged = to_applicability(baseline, out)

    assert merged.verdict == ApplicabilityVerdict.APPLICABLE
    assert merged.p_applicable == 0.82
    assert merged.version_match == VersionMatch.UNKNOWN
