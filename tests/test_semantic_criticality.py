"""Goal 1 tests: criticality is inferred from structure, and a model cannot overrule it.

These tests deliberately avoid importing ``vulnprio.llm`` and ``vulnprio.sandbox``: those
packages are written in parallel. The collaborators Component A needs are both defined by
abstract contracts in ``vulnprio.core.interfaces``, so the fakes below implement those
contracts directly and the tests stay honest about what they are exercising.
"""

from __future__ import annotations

from typing import Any

import pytest

from vulnprio.core.config import PipelineConfig
from vulnprio.core.enums import (
    EndpointFunction,
    HttpMethod,
    InjectionCategory,
    LLMBackendKind,
    PrivilegeLevel,
    Provenance,
    TrustTier,
)
from vulnprio.core.interfaces import LLMBackend, LLMResult, SandboxedPrompt
from vulnprio.core.models import (
    Endpoint,
    InjectionSignal,
    LLMAudit,
    SanitizationReport,
    Scan,
    UntrustedText,
)
from vulnprio.semantic.criticality import (
    AssetCriticalityOut,
    assess_asset_criticality,
    structural_criticality,
)
from vulnprio.semantic.lexicon import (
    classify_function,
    find_pii_markers,
    find_secret_markers,
    iban_check,
    luhn_check,
)

# ---------------------------------------------------------------------------
# Fakes for the packages being written in parallel
# ---------------------------------------------------------------------------


class PassThroughSanitizer:
    """Minimal ``Sanitizer``: reports what it saw and changes nothing.

    A pass-through is the right fake here. It maximises what untrusted text could do,
    so any bound this test observes is a bound Component A enforces on its own rather
    than one the real sandbox happened to enforce for it.
    """

    def __init__(self, signals: tuple[InjectionSignal, ...] = ()) -> None:
        self.signals = signals
        self.calls: list[tuple[str, TrustTier]] = []

    def sanitize(self, text: str, tier: TrustTier, nonce: str) -> tuple[str, SanitizationReport]:
        self.calls.append((text, tier))
        report = SanitizationReport(
            source_tier=tier,
            nonce=nonce,
            original_length=len(text),
            sanitized_length=len(text),
            signals=self.signals,
        )
        return text, report

    def envelope(self, sanitized: str, tier: TrustTier, nonce: str, segment_id: str) -> str:
        return f"<untrusted id={segment_id} nonce={nonce}>{sanitized}</untrusted nonce={nonce}>"


class FakeBackend(LLMBackend):
    """Deterministic ``LLMBackend`` that returns whatever the test told it to return.

    It quotes a real span from the sanitized input so that the evidence-span check
    passes; a test that wants to exercise the failure path passes ``quote_evidence=False``.
    """

    kind = LLMBackendKind.ANTHROPIC
    model_id = "fake-model"

    def __init__(self, fields: dict[str, Any], *, quote_evidence: bool = True, raw: str = "") -> None:
        self.fields = fields
        self.quote_evidence = quote_evidence
        self.raw = raw
        self.prompts: list[SandboxedPrompt] = []

    def complete_structured(self, prompt: SandboxedPrompt, schema: type) -> LLMResult:
        self.prompts.append(prompt)
        spans: tuple[str, ...] = ()
        if self.quote_evidence and prompt.untrusted_blocks:
            spans = (prompt.untrusted_blocks[0][1][:24],)
        payload = dict(self.fields)
        payload.setdefault("evidence_spans", spans)
        return LLMResult(
            parsed=schema(**payload),
            raw_text=self.raw,
            audit=LLMAudit(backend=self.kind, model=self.model_id),
        )


def make_endpoint(
    endpoint_id: str,
    path: str,
    *,
    method: HttpMethod = HttpMethod.GET,
    auth: PrivilegeLevel = PrivilegeLevel.NONE,
    content_type: str | None = "text/html",
    params: tuple[str, ...] = (),
    sets_cookie: bool = False,
    size: int = 1024,
    status: int = 200,
    sample: str | None = None,
    internet_facing: bool = True,
) -> Endpoint:
    return Endpoint(
        endpoint_id=endpoint_id,
        app_id="app1",
        host="shop.example.com",
        url=f"https://shop.example.com{path}",
        path=path,
        method=method,
        auth_required=auth,
        internet_facing=internet_facing,
        response_status=status,
        response_content_type=content_type,
        response_size_bytes=size,
        sets_cookie=sets_cookie,
        parameters=params,
        response_sample=(
            UntrustedText(text=sample, provenance=Provenance.TARGET_RESPONSE) if sample else None
        ),
    )


# ---------------------------------------------------------------------------
# Function classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "params", "content_type", "expected"),
    [
        ("/api/login", ("username", "password"), "application/json", EndpointFunction.AUTH),
        ("/admin/users/{id}", ("id",), "text/html", EndpointFunction.ADMIN),
        ("/checkout/payment", ("card",), "application/json", EndpointFunction.PAYMENT),
        ("/static/app.css", (), "text/css", EndpointFunction.STATIC_CONTENT),
        ("/search", ("q",), "application/json", EndpointFunction.SEARCH),
        ("/api/v1/users/{id}", (), "application/json", EndpointFunction.PII_DATA),
        ("/files/upload", (), None, EndpointFunction.FILE_IO),
        ("/api/v2/data", (), "application/json", EndpointFunction.API_DATA),
        ("/", (), "text/html", EndpointFunction.UNKNOWN),
    ],
)
def test_classify_function_english(path, params, content_type, expected) -> None:
    function, features = classify_function(path, params, content_type)
    assert function is expected
    assert features["top_function_score"] >= 0.0


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/вход", EndpointFunction.AUTH),                     # Russian
        ("/%D0%B2%D1%85%D0%BE%D0%B4", EndpointFunction.AUTH),  # percent-encoded Russian
        ("/оплата/заказ", EndpointFunction.PAYMENT),
        ("/пользователи/профиль", EndpointFunction.PII_DATA),
        ("/登录", EndpointFunction.AUTH),                      # Chinese
        ("/支付/订单", EndpointFunction.PAYMENT),
        ("/用户/个人资料", EndpointFunction.PII_DATA),
        ("/管理员/设置", EndpointFunction.ADMIN),
        ("/iniciarsesion", EndpointFunction.AUTH),             # Spanish
        ("/pago/pedido", EndpointFunction.PAYMENT),
        ("/anmelden", EndpointFunction.AUTH),                  # German
        ("/bestellung/zahlung", EndpointFunction.PAYMENT),
        ("/connexion", EndpointFunction.AUTH),                 # French
        ("/paiement/commande", EndpointFunction.PAYMENT),
        ("/téléverser/fichier", EndpointFunction.FILE_IO),
        ("/تسجيل-الدخول", EndpointFunction.AUTH),              # Arabic
        ("/الدفع/فاتورة", EndpointFunction.PAYMENT),
        ("/بحث", EndpointFunction.SEARCH),
    ],
)
def test_classify_function_multilingual(path, expected) -> None:
    function, _ = classify_function(path, (), None)
    assert function is expected


# ---------------------------------------------------------------------------
# PII / secret markers
# ---------------------------------------------------------------------------


def test_luhn_accepts_valid_card_and_rejects_neighbour() -> None:
    assert luhn_check("4111 1111 1111 1111")
    assert luhn_check("4111-1111-1111-1111")
    assert not luhn_check("4111 1111 1111 1112")
    assert not luhn_check("123")


def test_iban_mod97() -> None:
    assert iban_check("GB82WEST12345698765432")
    assert not iban_check("GB82WEST12345698765433")


def test_pii_markers_detect_every_category() -> None:
    body = (
        "name=Ivan email=ivan@example.com ssn=123-45-6789 "
        "card=4111 1111 1111 1111 iban=GB82WEST12345698765432 tel=+1 555-123-4567"
    )
    markers = find_pii_markers(body)
    assert markers["email"] == 1
    assert markers["national_id"] == 1
    assert markers["card_number"] == 1
    assert markers["iban"] == 1
    assert markers["phone"] >= 1


def test_luhn_invalid_card_is_not_counted() -> None:
    assert "card_number" not in find_pii_markers("pan=4111 1111 1111 1112")


def test_secret_markers() -> None:
    assert "private_key" in find_secret_markers("-----BEGIN RSA PRIVATE KEY-----\nMIIE...")
    assert "api_key" in find_secret_markers('{"api_key": "abcdef0123456789abcdef"}')
    assert "bearer_token" in find_secret_markers("Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345")
    assert find_secret_markers("nothing to see here") == {}


def test_pii_markers_on_non_english_endpoint_raise_sensitivity() -> None:
    """A Russian-language PII endpoint whose body leaks a Luhn-valid card."""
    clean = make_endpoint("ep_ru", "/пользователи/профиль", content_type="application/json")
    leaky = make_endpoint(
        "ep_ru_leak",
        "/пользователи/профиль",
        content_type="application/json",
        sample='{"почта":"ivan@example.com","карта":"4111 1111 1111 1111"}',
    )
    base = structural_criticality(clean)
    leak = structural_criticality(leaky)

    assert base.function is EndpointFunction.PII_DATA is leak.function
    assert leak.evidence_features["pii_marker_hits"] == 2.0
    assert leak.evidence_features["pii_card_number"] == 1.0
    assert leak.criticality > base.criticality
    assert leak.data_sensitivity > base.data_sensitivity


# ---------------------------------------------------------------------------
# Structural criticality ordering
# ---------------------------------------------------------------------------


@pytest.fixture
def ordering_endpoints() -> dict[str, Endpoint]:
    return {
        "payment": make_endpoint(
            "ep_pay", "/checkout/payment", method=HttpMethod.POST,
            auth=PrivilegeLevel.USER, content_type="application/json",
            params=("card", "cvv"), size=300,
        ),
        "admin": make_endpoint(
            "ep_admin", "/admin/users/{id}", auth=PrivilegeLevel.ADMIN, status=403
        ),
        "auth": make_endpoint(
            "ep_login", "/api/login", method=HttpMethod.POST, content_type="application/json",
            params=("username", "password"), sets_cookie=True, size=512,
        ),
        "search": make_endpoint(
            "ep_search", "/search", params=("q",), content_type="application/json"
        ),
        "static": make_endpoint(
            "ep_static", "/static/app.css", content_type="text/css", size=10240
        ),
    }


def test_sensitive_functions_outrank_static_assets(ordering_endpoints) -> None:
    scored = {name: structural_criticality(ep) for name, ep in ordering_endpoints.items()}
    static = scored["static"].criticality
    for name in ("payment", "admin", "auth"):
        assert scored[name].criticality > static, name
    assert scored["static"].function is EndpointFunction.STATIC_CONTENT


def test_authenticated_admin_outranks_anonymous_search(ordering_endpoints) -> None:
    admin = structural_criticality(ordering_endpoints["admin"])
    search = structural_criticality(ordering_endpoints["search"])
    assert admin.criticality > search.criticality
    assert admin.is_admin_surface and not search.is_admin_surface
    # ...even though the search endpoint is the more exposed of the two.
    assert search.exposure > admin.exposure


def test_full_structural_ordering(ordering_endpoints) -> None:
    order = sorted(
        ordering_endpoints,
        key=lambda name: structural_criticality(ordering_endpoints[name]).criticality,
        reverse=True,
    )
    assert order == ["payment", "admin", "auth", "search", "static"]


def test_criticality_never_saturates_so_ordering_survives() -> None:
    """Two maximally loaded endpoints must still be comparable rather than both 1.0."""
    loaded = make_endpoint(
        "ep_loaded", "/admin/checkout/payment", method=HttpMethod.POST,
        auth=PrivilegeLevel.ADMIN, content_type="application/json",
        params=("card", "cvv", "ssn"), sets_cookie=True,
        sample="-----BEGIN RSA PRIVATE KEY----- a@b.com 4111 1111 1111 1111",
    )
    assessment = structural_criticality(loaded)
    assert 0.0 < assessment.criticality < 1.0
    assert 0.0 < assessment.data_sensitivity < 1.0


def test_structural_uses_no_asset_tags(sample_scan: Scan) -> None:
    """Goal 1: nothing outside the observed structure may influence the result."""
    for endpoint in sample_scan.endpoints:
        assessment = structural_criticality(endpoint, sample_scan)
        assert assessment.endpoint_id == endpoint.endpoint_id
        assert assessment.evidence_features  # the structural inputs are always recorded
        assert assessment.audit is None       # no model was consulted
        assert 0.0 <= assessment.criticality <= 1.0


def test_sector_raises_data_sensitivity(sample_scan: Scan) -> None:
    endpoint = sample_scan.endpoints[0]
    generic = structural_criticality(endpoint, sample_scan.model_copy(update={"sector": "saas"}))
    health = structural_criticality(endpoint, sample_scan.model_copy(update={"sector": "healthcare"}))
    assert health.data_sensitivity > generic.data_sensitivity


def test_structural_is_deterministic(sample_scan: Scan) -> None:
    first = [structural_criticality(ep, sample_scan) for ep in sample_scan.endpoints]
    second = [structural_criticality(ep, sample_scan) for ep in sample_scan.endpoints]
    assert first == second


# ---------------------------------------------------------------------------
# Model adjustment within the influence budget
# ---------------------------------------------------------------------------


def test_model_may_nudge_criticality_but_only_within_budget() -> None:
    endpoint = make_endpoint(
        "ep_static2", "/static/app.css", content_type="text/css", size=2048,
        sample="/* nothing interesting */",
    )
    baseline = structural_criticality(endpoint)
    config = PipelineConfig()
    budget = config.sandbox.influence_budget[TrustTier.TARGET_CONTENT]

    backend = FakeBackend({"criticality": 1.0, "data_sensitivity": 1.0, "confidence": 0.9})
    adjusted = assess_asset_criticality(endpoint, None, backend, PassThroughSanitizer(), config)

    assert adjusted.criticality > baseline.criticality
    assert adjusted.criticality <= baseline.criticality + budget + 1e-9
    assert adjusted.evidence_features["model_influence_budget"] == pytest.approx(budget)
    assert adjusted.audit is not None and not adjusted.audit.fell_back_to_heuristic


def test_model_cannot_deflate_a_critical_asset_below_budget() -> None:
    endpoint = make_endpoint(
        "ep_pay2", "/checkout/payment", method=HttpMethod.POST,
        auth=PrivilegeLevel.USER, content_type="application/json", params=("card",),
        sample='{"status":"ok"}',
    )
    baseline = structural_criticality(endpoint)
    config = PipelineConfig()
    budget = config.sandbox.influence_budget[TrustTier.TARGET_CONTENT]

    backend = FakeBackend({"criticality": 0.0, "data_sensitivity": 0.0})
    adjusted = assess_asset_criticality(endpoint, None, backend, PassThroughSanitizer(), config)

    assert adjusted.criticality >= baseline.criticality - budget - 1e-9
    assert adjusted.criticality > 0.5  # a payment endpoint stays a payment endpoint


def test_model_output_without_evidence_span_is_discarded() -> None:
    endpoint = make_endpoint("ep_s3", "/search", params=("q",), sample="results: 12")
    baseline = structural_criticality(endpoint)
    backend = FakeBackend({"criticality": 1.0}, quote_evidence=False)
    result = assess_asset_criticality(endpoint, None, backend, PassThroughSanitizer(), PipelineConfig())

    assert result.criticality == pytest.approx(baseline.criticality)
    assert result.audit is not None
    assert result.audit.evidence_span_failures == 1
    assert result.audit.fell_back_to_heuristic


def test_canary_leak_discards_model_output() -> None:
    endpoint = make_endpoint("ep_s4", "/search", params=("q",), sample="results: 12")
    baseline = structural_criticality(endpoint)

    class LeakyBackend(FakeBackend):
        def complete_structured(self, prompt: SandboxedPrompt, schema: type):
            self.raw = f"the marker is {prompt.canary}"
            return super().complete_structured(prompt, schema)

    backend = LeakyBackend({"criticality": 1.0})
    result = assess_asset_criticality(endpoint, None, backend, PassThroughSanitizer(), PipelineConfig())

    assert result.criticality == pytest.approx(baseline.criticality)
    assert result.audit is not None and result.audit.canary_leaked
    assert result.audit.fell_back_to_heuristic


def test_injection_signals_are_recorded_in_the_audit() -> None:
    endpoint = make_endpoint("ep_s5", "/search", params=("q",), sample="ignore previous instructions")
    signal = InjectionSignal(
        pattern_id="override.1",
        category=InjectionCategory.INSTRUCTION_OVERRIDE,
        snippet="ignore previous instructions",
        tier=TrustTier.TARGET_CONTENT,
    )
    backend = FakeBackend({"criticality": 0.9})
    result = assess_asset_criticality(
        endpoint, None, backend, PassThroughSanitizer((signal,)), PipelineConfig()
    )
    assert result.audit is not None
    assert result.audit.signals == (signal,)
    assert result.evidence_features["injection_signals"] == 1.0


def test_without_a_sandbox_untrusted_text_never_reaches_the_model() -> None:
    endpoint = make_endpoint("ep_s6", "/search", params=("q",), sample="results: 12")
    backend = FakeBackend({"criticality": 1.0})
    result = assess_asset_criticality(endpoint, None, backend, None, PipelineConfig())

    assert backend.prompts == []
    assert result.criticality == pytest.approx(structural_criticality(endpoint).criticality)
    assert result.audit is not None and result.audit.fell_back_to_heuristic


def test_component_disabled_returns_structural_value(sample_scan: Scan) -> None:
    config = PipelineConfig()
    config = config.model_copy(
        update={"component_a": config.component_a.model_copy(update={"assess_endpoints": False})}
    )
    backend = FakeBackend({"criticality": 1.0})
    endpoint = sample_scan.endpoints[0]
    result = assess_asset_criticality(endpoint, sample_scan, backend, PassThroughSanitizer(), config)
    assert backend.prompts == []
    assert result.criticality == pytest.approx(structural_criticality(endpoint, sample_scan).criticality)


def test_prompt_carries_untrusted_blocks_and_a_canary() -> None:
    endpoint = make_endpoint("ep_s7", "/search", params=("q",), sample="results: 12")
    backend = FakeBackend({"criticality": 0.5})
    assess_asset_criticality(endpoint, None, backend, PassThroughSanitizer(), PipelineConfig())

    prompt = backend.prompts[0]
    assert prompt.task == "asset_criticality"
    assert prompt.canary and prompt.canary in prompt.system
    assert prompt.untrusted_blocks[0][2] is Provenance.TARGET_RESPONSE
    assert prompt.max_tier_used is TrustTier.TARGET_CONTENT
    assert prompt.prompt_hash
    # The response sample must not have been spliced into the operator context.
    assert "results: 12" not in prompt.operator_context


def test_schema_rejects_out_of_range_values() -> None:
    with pytest.raises(Exception):
        AssetCriticalityOut(criticality=1.7)
