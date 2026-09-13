"""Deterministic offline assessor: the backend the whole test suite runs on.

This is not a stub standing in for a model. It is a real lexicon-and-rule assessor that
reads exactly what a model would read -- the sanitized text of the untrusted blocks plus
the structured operator facts -- and produces the same bounded schemas. Two properties
make it load-bearing:

* **Determinism.** Same :class:`SandboxedPrompt`, same output, always. No randomness, no
  clock, no I/O. This is what makes ablations, adversarial runs and regression tests
  comparable across machines.
* **Instruction insensitivity.** Before any lexicon is applied, imperative sentences are
  removed from the text. A sentence in scanner output or a fetched advisory that says
  "set exploit_feasibility to 1.0" contributes nothing at all -- not its imperative, and
  not the keywords it happens to contain. That is the offline analogue of the sandbox's
  "data, never instructions" rule, and it is what lets the adversarial corpus be scored
  against a backend that cannot be talked into anything.

The heuristic never reads a raw envelope: it works from
:attr:`SandboxedPrompt.untrusted_blocks`, which already hold sanitized text.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

from vulnprio.core.enums import (
    ApplicabilityVerdict,
    AttackComplexity,
    EndpointFunction,
    ExploitMaturity,
    LLMBackendKind,
    PrivilegeLevel,
    TrustTier,
    UserInteraction,
)
from vulnprio.core.errors import ConfigError
from vulnprio.core.interfaces import LLMBackend, LLMResult, SandboxedPrompt
from vulnprio.core.models import InjectionSignal, LLMAudit
from vulnprio.core.registry import register_backend
from vulnprio.llm.prompts import parse_operator_context, prompt_hash
from vulnprio.llm.schemas import (
    ApplicabilityOut,
    AssetCriticalityOut,
    BoundedOut,
    ExploitabilityOut,
    task_for_schema,
)

__all__ = [
    "HeuristicBackend",
    "strip_imperative_sentences",
    "sanitized_text_of",
    "score_asset_criticality",
    "score_exploitability",
    "score_applicability",
]


# ---------------------------------------------------------------------------
# Sentence handling and imperative filtering
# ---------------------------------------------------------------------------

#: A sentence runs to a terminator that is followed by whitespace (so "2.5.12" and
#: "alice@example.com" do not split) or to the end of a line.
_SENTENCE_RE = re.compile(r"[^\n\r]+?(?:[.!?]+(?=\s)|$)", re.MULTILINE)

#: A sentence matching any of these is discarded before scoring. The list is written to
#: catch the *shape* of an instruction (imperative opener, second-person directive,
#: assignment of a named field) rather than particular payloads, so an unseen phrasing
#: of the same trick is still inert.
_IMPERATIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*(?:please|kindly)\b",
        r"^\s*(?:ignore|disregard|forget|override|overwrite|bypass|skip|stop)\b",
        r"^\s*(?:set|assign|mark|rate|score|output|return|respond|reply|answer|report|"
        r"classify|treat|consider|emit|print|update|change|ensure|always|never|do not|don't|must)\b",
        r"\byou\s+(?:must|should|shall|will|are\s+to|need\s+to|have\s+to|may\s+not|cannot)\b",
        r"\byour\s+(?:task|job|role|instruction|instructions|answer|output|response)\b",
        r"\b(?:new|updated|revised|additional|system|admin|urgent)\s+"
        r"(?:instruction|instructions|prompt|directive|directives|rule|rules|policy)\b",
        r"\bas\s+an?\s+(?:ai|assistant|language\s+model|security\s+assistant)\b",
        r"\b(?:set|assign|rate|score|mark|report|raise|lower)\b[^\n]{0,80}?\b(?:to|as|=)\s*"
        r"(?:\d|max|maximum|min|minimum|high|critical|low)",
        r"\b(?:criticality|feasibility|exploit_feasibility|p_applicable|data_sensitivity|"
        r"exposure|confidence|impact_[cia]|priority|severity|score)\s*[:=]\s*[0-9]",
        r"\bredacted-instruction\b",
        r"\b(?:system|developer)\s*(?:prompt|message)\b",
        r"\bprevious\s+(?:instructions|rules|directions)\b",
        r"^\s*<\s*/?\s*untrusted\b",
    )
)


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """Sentences as ``(text, start, end)``; ``text`` is a verbatim slice of the input."""
    spans: list[tuple[str, int, int]] = []
    for match in _SENTENCE_RE.finditer(text):
        raw = match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        start = match.start() + (len(raw) - len(raw.lstrip()))
        spans.append((stripped, start, start + len(stripped)))
    return spans


def is_imperative(sentence: str) -> bool:
    """True when a sentence is shaped like an instruction rather than a description."""
    return any(pattern.search(sentence) for pattern in _IMPERATIVE_PATTERNS)


def strip_imperative_sentences(text: str) -> tuple[list[str], int]:
    """Split into sentences and drop every imperative one.

    Returns the surviving sentences (verbatim slices of ``text``, so they remain valid
    evidence spans) and the number dropped.
    """
    kept: list[str] = []
    dropped = 0
    for sentence, _start, _end in _sentence_spans(text):
        if is_imperative(sentence):
            dropped += 1
            continue
        kept.append(sentence)
    return kept, dropped


def sanitized_text_of(prompt: SandboxedPrompt) -> str:
    """The sanitized untrusted text the model would see, blocks joined in order."""
    return "\n".join(text for _segment_id, text, _provenance in prompt.untrusted_blocks)


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

#: URL/parameter tokens per endpoint function. Multilingual entries are present because
#: Gap 8 asks for evidence that the framework is not English-only.
_FUNCTION_TOKENS: dict[EndpointFunction, tuple[str, ...]] = {
    EndpointFunction.AUTH: (
        "login", "signin", "sign-in", "logout", "signout", "session", "sessions", "token",
        "oauth", "saml", "sso", "password", "passwd", "credential", "register", "signup",
        "mfa", "otp", "2fa", "auth", "authenticate", "authorize", "reset-password",
        "connexion", "anmelden", "iniciar", "sesion", "acceso", "entrar", "senha",
    ),
    EndpointFunction.PAYMENT: (
        "pay", "payment", "payments", "checkout", "billing", "invoice", "invoices",
        "card", "cards", "creditcard", "stripe", "paypal", "order", "orders", "cart",
        "refund", "subscription", "wallet", "transaction", "transactions", "price",
        "paiement", "zahlung", "pago", "pagamento", "facture",
    ),
    EndpointFunction.ADMIN: (
        "admin", "admins", "administrator", "administration", "manage", "manager",
        "console", "dashboard", "backend", "backoffice", "wp-admin", "sysadmin",
        "superuser", "root", "settings", "config", "configuration", "system",
        "verwaltung", "administracion", "gestion",
    ),
    EndpointFunction.PII_DATA: (
        "user", "users", "profile", "profiles", "account", "accounts", "customer",
        "customers", "member", "members", "patient", "patients", "employee", "employees",
        "person", "people", "contact", "contacts", "address", "addresses", "ssn",
        "record", "records", "medical", "health", "identity", "kyc",
        "benutzer", "utilisateur", "usuario", "cliente", "kunde",
    ),
    EndpointFunction.FILE_IO: (
        "upload", "uploads", "download", "downloads", "file", "files", "attachment",
        "attachments", "import", "export", "backup", "backups", "document", "documents",
        "media", "image", "images", "avatar", "report", "reports", "csv", "pdf",
        "fichier", "datei", "archivo",
    ),
    EndpointFunction.API_DATA: (
        "api", "apis", "v1", "v2", "v3", "graphql", "rest", "rpc", "jsonrpc", "soap",
        "service", "services", "endpoint", "data", "feed", "webhook", "webhooks",
    ),
    EndpointFunction.SEARCH: (
        "search", "query", "find", "lookup", "filter", "autocomplete", "suggest",
        "suchen", "buscar", "recherche", "pesquisa",
    ),
    EndpointFunction.STATIC_CONTENT: (
        "static", "assets", "asset", "public", "dist", "build", "css", "js", "fonts",
        "font", "favicon", "img", "vendor", "bundle", "robots.txt", "sitemap",
    ),
}

#: Tie-break order when two functions score equally: the more consequential wins, so a
#: path like /admin/users is ADMIN rather than PII_DATA.
_FUNCTION_PRIORITY: tuple[EndpointFunction, ...] = (
    EndpointFunction.ADMIN,
    EndpointFunction.PAYMENT,
    EndpointFunction.AUTH,
    EndpointFunction.PII_DATA,
    EndpointFunction.FILE_IO,
    EndpointFunction.API_DATA,
    EndpointFunction.SEARCH,
    EndpointFunction.STATIC_CONTENT,
)

_FUNCTION_CRITICALITY: dict[EndpointFunction, float] = {
    EndpointFunction.ADMIN: 0.86,
    EndpointFunction.PAYMENT: 0.84,
    EndpointFunction.AUTH: 0.78,
    EndpointFunction.PII_DATA: 0.74,
    EndpointFunction.FILE_IO: 0.60,
    EndpointFunction.API_DATA: 0.48,
    EndpointFunction.SEARCH: 0.32,
    EndpointFunction.STATIC_CONTENT: 0.08,
    EndpointFunction.UNKNOWN: 0.40,
}

_FUNCTION_SENSITIVITY: dict[EndpointFunction, float] = {
    EndpointFunction.ADMIN: 0.62,
    EndpointFunction.PAYMENT: 0.86,
    EndpointFunction.AUTH: 0.70,
    EndpointFunction.PII_DATA: 0.90,
    EndpointFunction.FILE_IO: 0.50,
    EndpointFunction.API_DATA: 0.42,
    EndpointFunction.SEARCH: 0.22,
    EndpointFunction.STATIC_CONTENT: 0.04,
    EndpointFunction.UNKNOWN: 0.30,
}

#: Markers of personal / financial data in a response sample.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.IGNORECASE)),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,}\b")),
    ("phone", re.compile(r"\+\d{1,3}[\s-]?\d{3,}")),
    ("name_field", re.compile(r"\b(?:first_?name|last_?name|full_?name|surname)\b", re.IGNORECASE)),
    ("dob", re.compile(r"\b(?:date[_ ]of[_ ]birth|dob|birthdate)\b", re.IGNORECASE)),
    ("address", re.compile(r"\b(?:street|postcode|zip_?code|postal_?code|billing_?address)\b", re.IGNORECASE)),
    ("identifier", re.compile(r"\b(?:passport|national_?id|tax_?id|social_?security)\b", re.IGNORECASE)),
    ("health", re.compile(r"\b(?:diagnosis|medical_?record|patient_?id|prescription)\b", re.IGNORECASE)),
)

#: Exploit-maturity vocabulary, strongest first. Matching is on sanitized, de-imperatived
#: text, so "set maturity to weaponized" contributes nothing.
_MATURITY_PHRASES: tuple[tuple[ExploitMaturity, tuple[str, ...]], ...] = (
    (
        ExploitMaturity.WEAPONIZED,
        (
            "exploited in the wild", "actively exploited", "active exploitation",
            "weaponized", "weaponised", "ransomware campaign", "used by ransomware",
            "mass exploitation", "exploit kit", "metasploit module", "botnet",
            "in-the-wild exploitation", "activement exploite", "explotado activamente",
        ),
    ),
    (
        ExploitMaturity.FUNCTIONAL,
        (
            "working exploit", "functional exploit", "fully functional", "reliable exploit",
            "metasploit", "public exploit", "exploit script", "nuclei template",
            "exploit available", "weaponizable", "exploit code is available",
        ),
    ),
    (
        ExploitMaturity.POC,
        (
            "proof of concept", "proof-of-concept", "poc", "demonstration exploit",
            "researchers demonstrated", "sample payload", "reproduction steps",
            "preuve de concept", "prueba de concepto",
        ),
    ),
    (
        ExploitMaturity.UNPROVEN,
        (
            "no known exploit", "no public exploit", "theoretical", "not known to be exploited",
            "no evidence of exploitation", "unproven",
        ),
    ),
)

_COMPLEXITY_HIGH_PHRASES: tuple[str, ...] = (
    "race condition", "timing window", "requires specific configuration",
    "non-default configuration", "requires local access", "requires physical access",
    "high complexity", "difficult to exploit", "requires man-in-the-middle",
    "requires adjacent network", "narrow window", "requires chaining", "must be chained",
    "requires knowledge of", "requires a valid session",
)

_COMPLEXITY_LOW_PHRASES: tuple[str, ...] = (
    "single request", "single http request", "trivially exploitable", "trivial to exploit",
    "low complexity", "no authentication required", "unauthenticated remote",
    "remotely exploitable without authentication", "simple crafted request",
    "one crafted request",
)

_PRIVILEGE_PHRASES: tuple[tuple[PrivilegeLevel, tuple[str, ...]], ...] = (
    (PrivilegeLevel.ADMIN, ("administrative privileges", "admin privileges", "administrator account",
                            "requires admin", "as an administrator")),
    (PrivilegeLevel.USER, ("authenticated user", "requires authentication", "logged-in user",
                           "valid credentials", "requires a user account", "authenticated attacker")),
    (PrivilegeLevel.NONE, ("unauthenticated", "without authentication", "anonymous attacker",
                           "pre-authentication", "no credentials")),
)

_INTERACTION_PHRASES: tuple[str, ...] = (
    "user interaction", "victim must", "user must click", "requires the victim",
    "social engineering", "phishing", "tricked into", "opens a crafted",
    "visits a malicious", "requires a user to",
)

_IMPACT_C_PHRASES: tuple[str, ...] = (
    "information disclosure", "sensitive information", "disclose", "disclosure",
    "read arbitrary", "leak", "exfiltrate", "dump the database", "data breach",
    "sql injection", "path traversal", "directory traversal", "credential theft",
    "session hijack", "read files", "obtain sensitive",
)

_IMPACT_I_PHRASES: tuple[str, ...] = (
    "modify", "tamper", "overwrite", "arbitrary write", "inject", "injection",
    "upload arbitrary", "arbitrary code", "code execution", "command execution",
    "cross-site scripting", "csrf", "cross-site request forgery", "alter data",
    "deserialization", "template injection", "insert arbitrary",
)

_IMPACT_A_PHRASES: tuple[str, ...] = (
    "denial of service", "crash", "resource exhaustion", "unavailable", "outage",
    "shut down", "shutdown", "infinite loop", "memory exhaustion", "service disruption",
)

_PRIVILEGE_GAIN_PHRASES: tuple[tuple[PrivilegeLevel, tuple[str, ...]], ...] = (
    (PrivilegeLevel.SYSTEM, ("remote code execution", "arbitrary code execution",
                             "command injection", "os command", "webshell", "web shell",
                             "deserialization of untrusted data", "arbitrary file upload",
                             "shell access", "container escape")),
    (PrivilegeLevel.ADMIN, ("privilege escalation", "escalate privileges", "admin account takeover",
                            "authentication bypass", "auth bypass", "become administrator",
                            "access the admin")),
    (PrivilegeLevel.USER, ("account takeover", "session hijacking", "session fixation",
                           "impersonate a user", "insecure direct object reference", "idor",
                           "sql injection", "read the database")),
)

#: CWE -> (privilege gained, baseline CIA emphasis). Structural, tier-0 knowledge.
_CWE_PRIVILEGE: dict[int, PrivilegeLevel] = {
    77: PrivilegeLevel.SYSTEM, 78: PrivilegeLevel.SYSTEM, 94: PrivilegeLevel.SYSTEM,
    502: PrivilegeLevel.SYSTEM, 434: PrivilegeLevel.SYSTEM, 1188: PrivilegeLevel.ADMIN,
    287: PrivilegeLevel.ADMIN, 306: PrivilegeLevel.ADMIN, 269: PrivilegeLevel.ADMIN,
    862: PrivilegeLevel.ADMIN, 863: PrivilegeLevel.ADMIN,
    89: PrivilegeLevel.USER, 79: PrivilegeLevel.USER, 639: PrivilegeLevel.USER,
    352: PrivilegeLevel.USER, 22: PrivilegeLevel.USER, 918: PrivilegeLevel.USER,
    200: PrivilegeLevel.NONE, 16: PrivilegeLevel.NONE, 693: PrivilegeLevel.NONE,
}

#: Ceiling on the share of the headroom above a base score that secondary signals may
#: claim. Structure decides the band; evidence decides the position inside it, and never
#: the very top, which is reserved for a judgement no heuristic can make.
_MAX_BONUS = 0.85

_SEVERITY_BUMP: dict[str, float] = {
    "critical": 0.15, "high": 0.10, "medium": 0.0, "low": -0.25, "info": -0.45,
}

_APPLICABLE_PHRASES: tuple[str, ...] = (
    "versions prior to", "versions before", "affected versions", "is affected",
    "are affected", "vulnerable versions", "all versions up to", "affects all",
    "confirmed vulnerable",
)

_NOT_APPLICABLE_PHRASES: tuple[str, ...] = (
    "not affected", "is not vulnerable", "does not affect", "fixed in", "patched in",
    "mitigated by default", "disabled by default", "no longer vulnerable",
    "only affects windows", "only affects the enterprise edition",
)

_PRECONDITION_PHRASES: tuple[str, ...] = (
    "only if", "only when", "only applies", "applies only", "requires the",
    "must be enabled", "if enabled", "is enabled", "are enabled", "when enabled",
    "non-default configuration", "when configured", "requires the module",
    "requires the plugin", "depends on the configuration", "unless configured",
)


# ---------------------------------------------------------------------------
# Small parsing helpers over the operator context
# ---------------------------------------------------------------------------


def _fact(facts: Mapping[str, str], key: str) -> str:
    return (facts.get(key) or "").strip()


def _fact_float(facts: Mapping[str, str], key: str, default: float | None = None) -> float | None:
    raw = _fact(facts, key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _fact_int(facts: Mapping[str, str], key: str, default: int | None = None) -> int | None:
    value = _fact_float(facts, key, None)
    return default if value is None else int(value)


def _fact_bool(facts: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = _fact(facts, key).lower()
    if raw in ("true", "yes", "1"):
        return True
    if raw in ("false", "no", "0"):
        return False
    return default


def _fact_privilege(facts: Mapping[str, str], key: str, default: PrivilegeLevel | None = None) -> PrivilegeLevel | None:
    raw = _fact(facts, key).upper().replace("PRIVILEGELEVEL.", "")
    if not raw:
        return default
    for level in PrivilegeLevel:
        if raw == level.name or raw == str(int(level)):
            return level
    return default


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return round(min(high, max(low, value)), 4)


def _tokens(path: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9\.]+", path.lower()) if token]


def _count_phrases(haystack: str, phrases: Iterable[str]) -> int:
    return sum(1 for phrase in phrases if phrase in haystack)


def _pick_spans(sentences: Sequence[str], needles: Iterable[str], limit: int = 3) -> tuple[str, ...]:
    """Up to ``limit`` surviving sentences that mention one of ``needles``.

    Sentences are verbatim slices of the sanitized text, so they always satisfy the
    output guard's substring requirement, and they are truncated to the schema's span cap.
    """
    wanted = [needle for needle in needles if needle]
    chosen: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(needle in lowered for needle in wanted):
            candidate = sentence[:200]
            if candidate not in chosen:
                chosen.append(candidate)
        if len(chosen) >= limit:
            break
    return tuple(chosen)


def _pick_spans_matching(
    sentences: Sequence[str], patterns: Iterable[re.Pattern[str]], limit: int = 3
) -> tuple[str, ...]:
    """Up to ``limit`` surviving sentences matched by one of ``patterns``."""
    compiled = list(patterns)
    chosen: list[str] = []
    for sentence in sentences:
        if any(pattern.search(sentence) for pattern in compiled):
            candidate = sentence[:200]
            if candidate not in chosen:
                chosen.append(candidate)
        if len(chosen) >= limit:
            break
    return tuple(chosen)


def _rationale(parts: Sequence[str]) -> str:
    text = "; ".join(part for part in parts if part)
    return text[:600]


# ---------------------------------------------------------------------------
# Task scorers (pure functions of facts + sanitized text)
# ---------------------------------------------------------------------------


def score_asset_criticality(facts: Mapping[str, str], text: str) -> AssetCriticalityOut:
    """Structural criticality of an endpoint. Pure; see module docstring for the rules."""
    sentences, _dropped = strip_imperative_sentences(text)
    # Case is preserved for the PII detectors (IBANs and card formats are case-sensitive)
    # and folded for the phrase lexicons.
    body_cased = "\n".join(sentences)

    path = _fact(facts, "path") or _fact(facts, "endpoint_id")
    method = _fact(facts, "method").upper() or "GET"
    content_type = _fact(facts, "response_content_type").lower()
    parameters = [item for item in re.split(r"[,\s]+", _fact(facts, "parameters")) if item]
    auth = _fact_privilege(facts, "auth_required", PrivilegeLevel.NONE) or PrivilegeLevel.NONE
    internet_facing = _fact_bool(facts, "internet_facing", True)
    sets_cookie = _fact_bool(facts, "sets_cookie", False)
    status = _fact_int(facts, "response_status", None)

    observed = set(_tokens(path)) | {token.lower() for token in parameters}
    lowered_path = path.lower()
    scores: dict[EndpointFunction, int] = {}
    for function, tokens in _FUNCTION_TOKENS.items():
        # Exact token matches only, except for tokens that carry their own separator
        # (``wp-admin``, ``robots.txt``), which are matched against the raw path. Substring
        # matching would make "/rapid" an API endpoint and "/business" a bus lookup.
        hits = sum(
            1
            for token in tokens
            if token in observed or (not token.isalnum() and token in lowered_path)
        )
        if hits:
            scores[function] = hits
    if content_type.startswith(("text/css", "application/javascript", "text/javascript", "image/", "font/")):
        scores[EndpointFunction.STATIC_CONTENT] = scores.get(EndpointFunction.STATIC_CONTENT, 0) + 2

    function = EndpointFunction.UNKNOWN
    if scores:
        best = max(scores.values())
        for candidate in _FUNCTION_PRIORITY:
            if scores.get(candidate, 0) == best:
                function = candidate
                break

    pii_hits = sorted({name for name, pattern in _PII_PATTERNS if pattern.search(body_cased)})

    # Bonuses are applied to the headroom above the function's base rather than added
    # outright, so a stack of modest signals cannot saturate the scale and destroy the
    # ordering between a merely important endpoint and a critical one.
    bonus = 0.0
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        bonus += 0.30
    if auth >= PrivilegeLevel.ADMIN:
        bonus += 0.24
    elif auth == PrivilegeLevel.USER:
        bonus += 0.12
    if sets_cookie:
        bonus += 0.15
    bonus += min(0.45, 0.15 * len(pii_hits))
    if content_type.startswith(("application/json", "application/xml", "text/xml")):
        bonus += 0.12
    if len(parameters) >= 3:
        bonus += 0.09
    base = _FUNCTION_CRITICALITY[function]
    criticality = base + (1.0 - base) * min(_MAX_BONUS, bonus)

    sensitivity_base = _FUNCTION_SENSITIVITY[function]
    sensitivity_bonus = min(0.6, 0.15 * len(pii_hits)) + (0.1 if sets_cookie else 0.0)
    sensitivity = sensitivity_base + (1.0 - sensitivity_base) * min(_MAX_BONUS, sensitivity_bonus)

    if auth == PrivilegeLevel.NONE:
        exposure = 0.6 if status in (401, 403) else 1.0
    elif auth == PrivilegeLevel.USER:
        exposure = 0.5
    elif auth == PrivilegeLevel.ADMIN:
        exposure = 0.25
    else:
        exposure = 0.10
    if not internet_facing:
        exposure *= 0.4

    is_admin_surface = function == EndpointFunction.ADMIN or auth >= PrivilegeLevel.ADMIN
    is_auth_boundary = function == EndpointFunction.AUTH or sets_cookie

    signals = sum(
        1
        for present in (
            bool(path),
            bool(content_type),
            bool(parameters),
            bool(pii_hits),
            status is not None,
            function != EndpointFunction.UNKNOWN,
        )
        if present
    )
    confidence = _clip(0.35 + 0.08 * signals, 0.2, 0.9)

    reasons = [f"path tokens indicate {function.value}", f"method {method}", f"auth {auth.name}"]
    if pii_hits:
        reasons.append("PII markers: " + ", ".join(pii_hits))
    if sets_cookie:
        reasons.append("issues a session cookie")

    return AssetCriticalityOut(
        function=function,
        criticality=_clip(criticality),
        data_sensitivity=_clip(sensitivity),
        exposure=_clip(exposure),
        is_auth_boundary=is_auth_boundary,
        is_admin_surface=is_admin_surface,
        confidence=confidence,
        rationale=_rationale(reasons),
        # Only cite untrusted text when untrusted text actually contributed: a purely
        # structural judgement has nothing to quote and should say so with no spans.
        evidence_spans=(
            _pick_spans_matching(sentences, [pattern for name, pattern in _PII_PATTERNS if name in pii_hits])
            if pii_hits
            else ()
        ),
    )


def _maturity_from_text(body: str) -> tuple[ExploitMaturity, str | None]:
    for maturity, phrases in _MATURITY_PHRASES:
        for phrase in phrases:
            if phrase in body:
                return maturity, phrase
    return ExploitMaturity.UNKNOWN, None


def score_exploitability(facts: Mapping[str, str], text: str) -> ExploitabilityOut:
    """Exploit feasibility, maturity, conditions and impact. Pure; deterministic."""
    sentences, _dropped = strip_imperative_sentences(text)
    body = "\n".join(sentences).lower()

    cwe_id = _fact_int(facts, "cwe_id", None)
    severity = _fact(facts, "scanner_severity").lower()
    scanner_confidence = _fact_float(facts, "scanner_confidence", 0.5) or 0.5
    epss = _fact_float(facts, "epss", 0.0) or 0.0
    kev = _fact_bool(facts, "kev", False)
    kev_ransomware = _fact_bool(facts, "kev_ransomware", False)
    feed_maturity = _fact_int(facts, "exploit_maturity_feed", None)
    exploit_count = _fact_int(facts, "exploit_count", 0) or 0
    cvss_base = _fact_float(facts, "cvss_base", None)
    cvss_ac = _fact(facts, "cvss_ac").upper()
    cvss_pr = _fact(facts, "cvss_pr").upper()
    cvss_ui = _fact(facts, "cvss_ui").upper()
    auth = _fact_privilege(facts, "auth_required", None)

    text_maturity, maturity_phrase = _maturity_from_text(body)
    maturity = text_maturity
    if feed_maturity is not None:
        try:
            maturity = max(maturity, ExploitMaturity(feed_maturity))
        except ValueError:
            pass
    if kev and maturity < ExploitMaturity.FUNCTIONAL:
        maturity = ExploitMaturity.FUNCTIONAL
    if exploit_count > 0 and maturity < ExploitMaturity.POC:
        maturity = ExploitMaturity.POC

    high_hits = _count_phrases(body, _COMPLEXITY_HIGH_PHRASES)
    low_hits = _count_phrases(body, _COMPLEXITY_LOW_PHRASES)
    if cvss_ac in ("L", "LOW"):
        low_hits += 1
    elif cvss_ac in ("H", "HIGH"):
        high_hits += 1
    if high_hits > low_hits:
        complexity = AttackComplexity.HIGH
    elif low_hits > high_hits:
        complexity = AttackComplexity.LOW
    else:
        complexity = AttackComplexity.UNKNOWN

    privileges = PrivilegeLevel.NONE
    if cvss_pr in ("N", "NONE"):
        privileges = PrivilegeLevel.NONE
    elif cvss_pr in ("L", "LOW"):
        privileges = PrivilegeLevel.USER
    elif cvss_pr in ("H", "HIGH"):
        privileges = PrivilegeLevel.ADMIN
    elif auth is not None:
        privileges = auth
    else:
        for level, phrases in _PRIVILEGE_PHRASES:
            if _count_phrases(body, phrases):
                privileges = level
                break

    if cvss_ui in ("N", "NONE"):
        interaction = UserInteraction.NONE
    elif cvss_ui in ("R", "REQUIRED"):
        interaction = UserInteraction.REQUIRED
    elif _count_phrases(body, _INTERACTION_PHRASES):
        interaction = UserInteraction.REQUIRED
    elif cwe_id in (79, 352):
        interaction = UserInteraction.REQUIRED
    else:
        interaction = UserInteraction.UNKNOWN

    impact_c = min(1.0, 0.18 * _count_phrases(body, _IMPACT_C_PHRASES))
    impact_i = min(1.0, 0.18 * _count_phrases(body, _IMPACT_I_PHRASES))
    impact_a = min(1.0, 0.18 * _count_phrases(body, _IMPACT_A_PHRASES))
    for key, boost in (("cvss_c", "c"), ("cvss_i", "i"), ("cvss_a", "a")):
        letter = _fact(facts, key).upper()
        value = {"H": 0.9, "HIGH": 0.9, "L": 0.4, "LOW": 0.4, "N": 0.0, "NONE": 0.0}.get(letter)
        if value is None:
            continue
        if boost == "c":
            impact_c = max(impact_c, value)
        elif boost == "i":
            impact_i = max(impact_i, value)
        else:
            impact_a = max(impact_a, value)

    privilege_gained = PrivilegeLevel.NONE
    for level, phrases in _PRIVILEGE_GAIN_PHRASES:
        if _count_phrases(body, phrases):
            privilege_gained = level
            break
    if cwe_id in _CWE_PRIVILEGE:
        privilege_gained = max(privilege_gained, _CWE_PRIVILEGE[cwe_id])
    privilege_gained = max(privilege_gained, privileges)

    # Maturity sets the band (what artefacts demonstrably exist); everything else moves
    # within the headroom of that band, so no stack of secondary signals can reach 1.0
    # and no single penalty can drive a weaponized exploit to zero.
    core = 0.10 + 0.16 * int(maturity)
    positives = (
        0.45 * epss
        + (0.25 if kev else 0.0)
        + (0.10 if kev_ransomware else 0.0)
        + (0.15 if complexity == AttackComplexity.LOW else 0.0)
        + (0.10 if privileges == PrivilegeLevel.NONE else 0.0)
        + 0.20 * ((cvss_base or 0.0) / 10.0)
        + max(0.0, _SEVERITY_BUMP.get(severity, 0.0))
        + 0.10 * scanner_confidence
    )
    penalties = (
        (0.35 if complexity == AttackComplexity.HIGH else 0.0)
        + {
            PrivilegeLevel.NONE: 0.0,
            PrivilegeLevel.USER: 0.10,
            PrivilegeLevel.ADMIN: 0.30,
            PrivilegeLevel.SYSTEM: 0.50,
        }[privileges]
        + (0.20 if interaction == UserInteraction.REQUIRED else 0.0)
        - min(0.0, _SEVERITY_BUMP.get(severity, 0.0))
    )
    feasibility = core + (1.0 - core) * min(_MAX_BONUS, positives) - core * min(0.9, penalties)

    preconditions: list[str] = []
    if privileges > PrivilegeLevel.NONE:
        preconditions.append(f"attacker holds {privileges.name} privileges")
    if interaction == UserInteraction.REQUIRED:
        preconditions.append("a user must interact with attacker-controlled content")
    if complexity == AttackComplexity.HIGH:
        preconditions.append("exploitation depends on conditions outside the attacker's control")

    reasons = [f"maturity {maturity.name}"]
    if maturity_phrase:
        reasons.append(f"text evidence: {maturity_phrase}")
    if kev:
        reasons.append("CVE is in CISA KEV")
    if epss:
        reasons.append(f"EPSS {epss:.3f}")
    reasons.append(f"complexity {complexity.value}, privileges {privileges.name}")

    needles = [phrase for _level, phrases in _MATURITY_PHRASES for phrase in phrases]
    needles += list(_IMPACT_C_PHRASES) + list(_IMPACT_I_PHRASES) + list(_IMPACT_A_PHRASES)

    confidence = _clip(
        0.35
        + (0.15 if maturity != ExploitMaturity.UNKNOWN else 0.0)
        + (0.10 if cvss_base is not None else 0.0)
        + (0.10 if kev or epss else 0.0),
        0.2,
        0.9,
    )

    return ExploitabilityOut(
        exploit_feasibility=_clip(feasibility),
        exploit_maturity=maturity,
        attack_complexity=complexity,
        privileges_required=privileges,
        user_interaction=interaction,
        preconditions=tuple(preconditions[:8]),
        impact_c=_clip(impact_c),
        impact_i=_clip(impact_i),
        impact_a=_clip(impact_a),
        privilege_gained=privilege_gained,
        confidence=confidence,
        rationale=_rationale(reasons),
        evidence_spans=_pick_spans(sentences, needles, limit=3),
    )


def score_applicability(facts: Mapping[str, str], text: str) -> ApplicabilityOut:
    """Whether the finding applies to the observed deployment. Pure; deterministic."""
    sentences, _dropped = strip_imperative_sentences(text)
    body = "\n".join(sentences).lower()

    version_match = _fact(facts, "version_match").lower()
    observed_version = _fact(facts, "observed_version")

    if version_match == "match":
        base = 0.85
    elif version_match == "mismatch":
        base = 0.08
    else:
        base = 0.5

    affected_hits = _count_phrases(body, _APPLICABLE_PHRASES)
    clear_hits = _count_phrases(body, _NOT_APPLICABLE_PHRASES)
    precondition_hits = _count_phrases(body, _PRECONDITION_PHRASES)

    if version_match == "mismatch":
        # A tier <= 1 version mismatch is authoritative: narrative text may not argue it
        # back up, only further down.
        probability = _clip(base - base * min(0.9, 0.35 * clear_hits))
    else:
        # Each kind of statement is capped separately: one clause often matches several
        # phrases ("only applies if the plugin is enabled") and must not be counted twice.
        positives = min(0.5, 0.25 * affected_hits)
        negatives = min(0.5, 0.35 * clear_hits) + min(0.3, 0.15 * precondition_hits)
        probability = _clip(base + (1.0 - base) * positives - base * negatives)
    if version_match == "mismatch":
        verdict = ApplicabilityVerdict.NOT_APPLICABLE
    elif probability >= 0.7:
        verdict = ApplicabilityVerdict.APPLICABLE
    elif probability <= 0.3:
        verdict = ApplicabilityVerdict.NOT_APPLICABLE
    else:
        verdict = ApplicabilityVerdict.UNCERTAIN

    preconditions_met = {
        "version_in_affected_range": version_match == "match",
        "no_vendor_fix_reported": clear_hits == 0,
        "default_configuration_sufficient": precondition_hits == 0,
    }

    reasons = [f"version evidence {version_match or 'unknown'}"]
    if observed_version:
        reasons.append(f"observed version {observed_version}")
    if affected_hits:
        reasons.append(f"{affected_hits} affected-range statement(s)")
    if clear_hits:
        reasons.append(f"{clear_hits} not-affected / fixed statement(s)")
    if precondition_hits:
        reasons.append(f"{precondition_hits} conditional precondition(s)")

    confidence = _clip(
        0.35 + (0.25 if version_match in ("match", "mismatch") else 0.0) + 0.05 * (affected_hits + clear_hits),
        0.2,
        0.9,
    )

    return ApplicabilityOut(
        verdict=verdict,
        p_applicable=probability,
        preconditions_met=preconditions_met,
        confidence=confidence,
        rationale=_rationale(reasons),
        evidence_spans=_pick_spans(
            sentences,
            list(_APPLICABLE_PHRASES) + list(_NOT_APPLICABLE_PHRASES) + list(_PRECONDITION_PHRASES),
            limit=3,
        ),
    )


_SCORERS = {
    "asset_criticality": score_asset_criticality,
    "exploitability": score_exploitability,
    "applicability": score_applicability,
}


@register_backend(LLMBackendKind.HEURISTIC)
class HeuristicBackend(LLMBackend):
    """Offline, deterministic :class:`LLMBackend`.

    It is the default backend, the fallback for every other backend, and the baseline
    the consistency guard shrinks model answers toward.
    """

    kind = LLMBackendKind.HEURISTIC
    model_id = "heuristic-v1"

    def __init__(self, model_id: str | None = None) -> None:
        """``model_id`` is overridable only so an experiment can label a variant lexicon."""
        if model_id:
            self.model_id = model_id

    def complete_structured(self, prompt: SandboxedPrompt, schema: type[BoundedOut]) -> LLMResult:
        """Score ``prompt`` into ``schema`` using lexicons and the operator facts."""
        try:
            task = task_for_schema(schema)
        except KeyError as exc:  # pragma: no cover - guarded by the schema registry
            raise ConfigError(f"heuristic backend cannot answer schema {schema.__name__}") from exc

        facts = parse_operator_context(prompt.operator_context)
        text = sanitized_text_of(prompt)
        parsed = _SCORERS[task](facts, text)
        return LLMResult(parsed=parsed, raw_text=parsed.model_dump_json(), audit=self.audit_for(prompt, task))

    def audit_for(self, prompt: SandboxedPrompt, task: str) -> LLMAudit:
        """Audit record for a heuristic call: no network, no retries, nothing leaked."""
        signals: list[InjectionSignal] = []
        for report in prompt.reports:
            signals.extend(report.signals)
        try:
            pinned = prompt.prompt_hash or prompt_hash(task)
        except ConfigError:
            pinned = prompt.prompt_hash
        return LLMAudit(
            backend=LLMBackendKind.HEURISTIC,
            model=self.model_id,
            task=task or prompt.task,
            prompt_hash=pinned,
            signals=tuple(signals),
            max_tier_used=prompt.max_tier_used if prompt.reports else TrustTier.OPERATOR,
        )

    def available(self) -> bool:
        """Always available: it needs no key, no network and no files."""
        return True
