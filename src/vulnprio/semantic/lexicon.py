"""Multilingual URL / body token lexicon and sensitive-data markers.

Goal 1 of the framework says endpoint criticality must be *inferred*, never taken from a
manually maintained asset inventory. That only works if the inference survives the fact
that real applications are not written in English: ``/оплата``, ``/支付``, ``/paiement``
and ``/checkout`` are the same business function. This module is therefore the lowest
layer of Component A: a weighted, multilingual token lexicon per
:class:`~vulnprio.core.enums.EndpointFunction`, plus regular-expression markers for
personally identifiable information and for secrets that leak into response bodies.

Everything here is a pure function over strings. No model, no network, no state.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Mapping, Sequence
from urllib.parse import unquote

from vulnprio.core.enums import EndpointFunction

__all__ = [
    "TOKEN_LEXICON",
    "FUNCTION_PRIORITY",
    "PII_MARKERS",
    "SECRET_MARKERS",
    "SENSITIVE_PARAM_TOKENS",
    "normalize_text",
    "tokenize",
    "luhn_check",
    "iban_check",
    "find_pii_markers",
    "find_secret_markers",
    "pii_marker_count",
    "secret_marker_count",
    "score_functions",
    "matched_tokens",
    "classify_function",
    "merge_feature_maps",
    "iter_lexicon_tokens",
]


# ---------------------------------------------------------------------------
# Token lexicon
# ---------------------------------------------------------------------------

#: Token -> weight per endpoint function. Weights are additive; a token that names the
#: function outright (``checkout``, ``admin``, ``登录``) outweighs a token that merely
#: hints at it (``v1``, ``data``). Non-ASCII tokens are matched as substrings because
#: Chinese and Arabic paths do not separate words the way ASCII paths do.
TOKEN_LEXICON: dict[EndpointFunction, dict[str, float]] = {
    EndpointFunction.AUTH: {
        # English
        "login": 1.2, "logon": 1.1, "signin": 1.2, "sign-in": 1.2, "signup": 1.0,
        "register": 1.0, "registration": 1.0, "logout": 0.9, "signout": 0.9,
        "auth": 1.0, "authenticate": 1.1, "authentication": 1.1, "oauth": 1.0,
        "sso": 1.0, "saml": 0.9, "session": 0.8, "sessions": 0.8, "token": 0.7,
        "password": 1.1, "passwd": 1.0, "credential": 1.0, "credentials": 1.0,
        "mfa": 1.0, "otp": 1.0, "2fa": 1.0, "totp": 1.0, "verify": 0.6,
        "reset-password": 1.2, "forgot": 0.8,
        # Russian
        "вход": 1.2, "войти": 1.2, "логин": 1.2, "авторизац": 1.1, "пароль": 1.1,
        "регистрац": 1.0, "сесси": 0.8, "выход": 0.7,
        # Chinese
        "登录": 1.2, "登陆": 1.2, "登入": 1.2, "认证": 1.1, "鉴权": 1.1, "密码": 1.1,
        "注册": 1.0, "会话": 0.8, "退出登录": 0.9,
        # Spanish
        "iniciarsesion": 1.2, "inicio-sesion": 1.2, "acceso": 0.9, "contrasena": 1.1,
        "contraseña": 1.1, "registro": 1.0, "autenticacion": 1.1, "autenticación": 1.1,
        # German
        "anmelden": 1.2, "anmeldung": 1.2, "abmelden": 0.9, "passwort": 1.1,
        "registrieren": 1.0, "authentifizierung": 1.1, "kennwort": 1.1,
        # French
        "connexion": 1.2, "seconnecter": 1.2, "motdepasse": 1.1, "authentification": 1.1,
        "inscription": 1.0, "deconnexion": 0.9, "identifiant": 1.0,
        # Arabic
        "تسجيل": 1.1, "دخول": 1.1, "كلمةالمرور": 1.1, "مصادقة": 1.1, "تسجيلالدخول": 1.2,
    },
    EndpointFunction.PAYMENT: {
        # English
        "payment": 1.3, "payments": 1.3, "pay": 1.0, "paying": 0.9, "checkout": 1.3,
        "billing": 1.1, "invoice": 1.1, "invoices": 1.1, "order": 1.1, "orders": 1.1,
        "cart": 1.0, "basket": 1.0, "card": 0.9, "cards": 0.9, "creditcard": 1.3,
        "refund": 1.0, "refunds": 1.0, "subscription": 0.9, "subscriptions": 0.9,
        "transaction": 1.0, "transactions": 1.0, "wallet": 1.0, "payout": 1.1,
        "stripe": 1.2, "paypal": 1.2, "braintree": 1.2, "adyen": 1.2, "iban": 1.1,
        "price": 0.5, "purchase": 1.0, "charge": 0.8,
        # Russian
        "оплат": 1.3, "платеж": 1.3, "платёж": 1.3, "счет": 0.9, "счёт": 0.9,
        "заказ": 1.1, "корзин": 1.0, "карт": 0.6, "покупк": 1.0, "возврат": 0.8,
        # Chinese
        "支付": 1.3, "付款": 1.3, "结算": 1.2, "结账": 1.2, "订单": 1.1, "购物车": 1.0,
        "发票": 1.1, "退款": 1.0, "钱包": 1.0, "交易": 1.0,
        # Spanish
        "pago": 1.3, "pagos": 1.3, "pagar": 1.2, "factura": 1.1, "facturacion": 1.1,
        "pedido": 1.1, "pedidos": 1.1, "carrito": 1.0, "compra": 1.0, "reembolso": 1.0,
        # German
        "zahlung": 1.3, "zahlungen": 1.3, "bezahlen": 1.2, "rechnung": 1.1,
        "bestellung": 1.1, "bestellungen": 1.1, "warenkorb": 1.0, "kasse": 1.0,
        "erstattung": 1.0,
        # French
        "paiement": 1.3, "paiements": 1.3, "payer": 1.2, "facture": 1.1,
        "commande": 1.1, "commandes": 1.1, "panier": 1.0, "caisse": 1.0,
        "remboursement": 1.0,
        # Arabic
        "دفع": 1.3, "الدفع": 1.3, "فاتورة": 1.1, "طلب": 0.9, "الطلبات": 1.1,
        "سلة": 1.0, "محفظة": 1.0,
    },
    EndpointFunction.ADMIN: {
        # English
        "admin": 1.4, "admins": 1.4, "administrator": 1.4, "administration": 1.4,
        "wp-admin": 1.5, "phpmyadmin": 1.5, "backoffice": 1.3, "back-office": 1.3,
        "console": 1.0, "dashboard": 0.9, "manage": 1.0, "manager": 1.0,
        "management": 1.0, "sysadmin": 1.4, "superuser": 1.3, "root": 1.0,
        "settings": 0.8, "config": 0.9, "configuration": 0.9, "acp": 0.9,
        "cpanel": 1.3, "actuator": 1.1, "debug": 0.8, "internal": 0.8,
        # Russian
        "админ": 1.4, "администратор": 1.4, "администрирован": 1.4, "управлен": 1.0,
        "панель": 1.0, "настройк": 0.8,
        # Chinese
        "管理": 1.2, "管理员": 1.4, "后台": 1.3, "控制台": 1.2, "仪表板": 0.9, "设置": 0.8,
        # Spanish
        "administrador": 1.4, "administracion": 1.4, "administración": 1.4,
        "gestion": 1.0, "gestión": 1.0, "ajustes": 0.8, "tablero": 0.9,
        # German
        "verwaltung": 1.3, "verwalten": 1.2, "einstellungen": 0.8, "administrieren": 1.3,
        # French
        "administrateur": 1.4, "administration": 1.4, "gestion": 1.0,
        "parametres": 0.8, "paramètres": 0.8, "tableaudebord": 0.9,
        # Arabic
        "إدارة": 1.3, "المدير": 1.3, "لوحة": 1.0, "إعدادات": 0.8,
    },
    EndpointFunction.PII_DATA: {
        # English
        "user": 0.9, "users": 1.1, "profile": 1.0, "profiles": 1.0, "account": 1.1,
        "accounts": 1.1, "customer": 1.1, "customers": 1.1, "patient": 1.3,
        "patients": 1.3, "employee": 1.0, "employees": 1.0, "member": 0.9,
        "members": 0.9, "contact": 0.8, "contacts": 0.8, "address": 0.9,
        "addresses": 0.9, "phone": 0.9, "email": 0.9, "ssn": 1.4, "dob": 1.1,
        "birthdate": 1.1, "passport": 1.3, "identity": 1.0, "medical": 1.3,
        "health": 1.1, "record": 0.6, "records": 0.6, "person": 0.9, "people": 0.8,
        "subscriber": 0.9, "directory": 0.6,
        # Russian
        "пользовател": 1.1, "профил": 1.0, "аккаунт": 1.1, "учетн": 1.0, "учётн": 1.0,
        "клиент": 1.0, "адрес": 0.9, "телефон": 0.9, "почта": 0.7, "паспорт": 1.3,
        # Chinese
        "用户": 1.1, "個人資料": 1.0, "个人资料": 1.0, "账户": 1.1, "帐户": 1.1,
        "客户": 1.1, "会员": 0.9, "身份证": 1.4, "地址": 0.9, "电话": 0.9, "病人": 1.3,
        # Spanish
        "usuario": 1.1, "usuarios": 1.1, "perfil": 1.0, "cuenta": 1.1, "cuentas": 1.1,
        "cliente": 1.1, "clientes": 1.1, "miembro": 0.9, "direccion": 0.9,
        "dirección": 0.9, "telefono": 0.9, "teléfono": 0.9, "paciente": 1.3,
        # German
        "benutzer": 1.1, "nutzer": 1.1, "profil": 1.0, "konto": 1.1, "konten": 1.1,
        "kunde": 1.1, "kunden": 1.1, "mitglied": 0.9, "adresse": 0.9, "telefon": 0.9,
        "patient": 1.3,
        # French
        "utilisateur": 1.1, "utilisateurs": 1.1, "profil": 1.0, "compte": 1.1,
        "comptes": 1.1, "client": 1.1, "membre": 0.9, "adresse": 0.9,
        "telephone": 0.9, "téléphone": 0.9,
        # Arabic
        "مستخدم": 1.1, "المستخدمين": 1.1, "الملفالشخصي": 1.0, "حساب": 1.1,
        "عميل": 1.1, "عنوان": 0.9, "هاتف": 0.9,
    },
    EndpointFunction.FILE_IO: {
        # English
        "upload": 1.3, "uploads": 1.3, "download": 1.1, "downloads": 1.1,
        "file": 1.0, "files": 1.0, "attachment": 1.1, "attachments": 1.1,
        "import": 0.9, "export": 0.9, "media": 0.9, "avatar": 1.0, "image": 0.7,
        "document": 0.9, "documents": 0.9, "backup": 1.1, "archive": 0.9,
        "blob": 0.9, "storage": 0.9, "attach": 1.0,
        # Russian
        "загрузк": 1.2, "загрузить": 1.3, "файл": 1.0, "вложени": 1.1, "документ": 0.9,
        # Chinese
        "上传": 1.3, "下载": 1.1, "文件": 1.0, "附件": 1.1, "导入": 0.9, "导出": 0.9,
        # Spanish
        "subir": 1.2, "cargar": 1.1, "descargar": 1.1, "archivo": 1.0,
        "adjunto": 1.1, "documento": 0.9,
        # German
        "hochladen": 1.3, "herunterladen": 1.1, "datei": 1.0, "dateien": 1.0,
        "anhang": 1.1, "dokument": 0.9,
        # French
        "televerser": 1.3, "téléverser": 1.3, "telecharger": 1.1, "télécharger": 1.1,
        "fichier": 1.0, "piecejointe": 1.1, "document": 0.9,
        # Arabic
        "رفع": 1.2, "تحميل": 1.1, "ملف": 1.0, "مرفق": 1.1, "مستند": 0.9,
    },
    EndpointFunction.API_DATA: {
        "api": 0.5, "apis": 0.5, "rest": 0.5, "graphql": 0.8, "rpc": 0.6,
        "jsonrpc": 0.8, "v1": 0.2, "v2": 0.2, "v3": 0.2, "v4": 0.2,
        "json": 0.4, "endpoint": 0.4, "service": 0.4, "services": 0.4,
        "data": 0.5, "feed": 0.5, "webhook": 0.7, "webhooks": 0.7, "graph": 0.3,
        "resource": 0.4, "swagger": 0.6, "openapi": 0.6, "интерфейс": 0.4,
        "接口": 0.6, "数据": 0.5, "servicio": 0.4, "dienst": 0.4, "donnees": 0.4,
    },
    EndpointFunction.SEARCH: {
        "search": 1.2, "searches": 1.2, "query": 0.9, "find": 0.8, "lookup": 0.9,
        "filter": 0.7, "autocomplete": 1.0, "suggest": 0.9, "suggestions": 0.9,
        "typeahead": 1.0, "q": 0.5,
        "поиск": 1.2, "найти": 0.9, "искать": 1.0,
        "搜索": 1.2, "查询": 1.0, "查找": 1.0,
        "buscar": 1.2, "busqueda": 1.2, "búsqueda": 1.2, "consulta": 0.8,
        "suche": 1.2, "suchen": 1.2, "abfrage": 0.8,
        "recherche": 1.2, "rechercher": 1.2, "chercher": 1.0,
        "بحث": 1.2, "البحث": 1.2,
    },
    EndpointFunction.STATIC_CONTENT: {
        "static": 1.3, "assets": 1.1, "asset": 0.9, "css": 1.1, "js": 0.9,
        "javascript": 0.9, "img": 0.9, "images": 0.8, "fonts": 0.9, "font": 0.7,
        "favicon": 1.2, "robots": 0.9, "sitemap": 0.9, "public": 0.6, "dist": 0.7,
        "build": 0.5, "bundle": 0.7, "vendor": 0.6, "styles": 0.8, "scripts": 0.6,
        "svg": 0.7, "png": 0.7, "jpg": 0.7, "jpeg": 0.7, "gif": 0.7, "woff": 0.9,
        "woff2": 0.9, "ttf": 0.8, "webp": 0.7, "map": 0.4,
    },
    EndpointFunction.UNKNOWN: {},
}

#: Deterministic tie-break order when two functions score identically. Higher-risk
#: surfaces win ties so that ambiguity errs on the side of over- rather than
#: under-prioritising.
FUNCTION_PRIORITY: tuple[EndpointFunction, ...] = (
    EndpointFunction.PAYMENT,
    EndpointFunction.ADMIN,
    EndpointFunction.AUTH,
    EndpointFunction.PII_DATA,
    EndpointFunction.FILE_IO,
    EndpointFunction.SEARCH,
    EndpointFunction.API_DATA,
    EndpointFunction.STATIC_CONTENT,
    EndpointFunction.UNKNOWN,
)

#: Parameter names that betray sensitive handling regardless of the path.
SENSITIVE_PARAM_TOKENS: frozenset[str] = frozenset(
    {
        "password", "passwd", "pwd", "pass", "secret", "token", "apikey", "api_key",
        "access_token", "refresh_token", "session", "sessionid", "jsessionid", "auth",
        "authorization", "card", "cardnumber", "cvv", "cvc", "pan", "iban", "ssn",
        "sin", "nino", "taxid", "dob", "birthdate", "passport", "licence", "license",
        "пароль", "токен", "密码", "令牌", "contrasena", "passwort", "motdepasse",
    }
)


# ---------------------------------------------------------------------------
# Normalisation and tokenisation
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """NFKC-fold, percent-decode and lowercase so that encoded paths match the lexicon.

    ``/%D0%B2%D1%85%D0%BE%D0%B4`` and ``/вход`` must classify identically; scanners
    report either form depending on the target's redirects.
    """
    if not text:
        return ""
    try:
        decoded = unquote(text)
    except Exception:  # pragma: no cover - unquote is total for str in practice
        decoded = text
    return unicodedata.normalize("NFKC", decoded).casefold()


def tokenize(text: str) -> list[str]:
    """Alphanumeric runs of the normalised text, in order of appearance."""
    return _TOKEN_RE.findall(normalize_text(text))


def _token_matches(token: str, haystack: str, token_set: frozenset[str]) -> bool:
    """Whether one lexicon token is present.

    ASCII tokens match a whole path segment token, or as a substring when they are long
    enough (>= 5 characters) that a substring hit is not coincidence -- this catches
    ``wp-admin`` inside ``wpadmin`` and ``administration`` inside ``administracion``.
    Non-ASCII tokens always match as substrings because Chinese and Arabic scripts do
    not delimit words.
    """
    if token in token_set:
        return True
    if not token.isascii():
        return token in haystack
    return len(token) >= 5 and token in haystack


def matched_tokens(text: str, function: EndpointFunction) -> tuple[str, ...]:
    """Lexicon tokens of ``function`` that occur in ``text`` (deterministic order)."""
    haystack = normalize_text(text)
    token_set = frozenset(_TOKEN_RE.findall(haystack))
    return tuple(
        token for token in TOKEN_LEXICON.get(function, {}) if _token_matches(token, haystack, token_set)
    )


def score_functions(text: str) -> dict[EndpointFunction, float]:
    """Additive lexicon score per function for one blob of path/parameter text."""
    haystack = normalize_text(text)
    token_set = frozenset(_TOKEN_RE.findall(haystack))
    scores: dict[EndpointFunction, float] = {}
    for function, tokens in TOKEN_LEXICON.items():
        total = 0.0
        for token, weight in tokens.items():
            if _token_matches(token, haystack, token_set):
                total += weight
        scores[function] = round(total, 6)
    return scores


# ---------------------------------------------------------------------------
# Sensitive-data markers
# ---------------------------------------------------------------------------

PII_MARKERS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    # US SSN, Spanish DNI/NIE, French INSEE-ish and generic dd-dd-dddd national ids.
    "national_id": re.compile(
        r"\b(?:\d{3}-\d{2}-\d{4}"
        r"|[XYZ]?\d{7,8}[\-\s]?[A-HJ-NP-TV-Z]"
        r"|\d{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3})\b"
    ),
    # 12-19 digits with optional single space/dash separators; Luhn-filtered below.
    "card_number": re.compile(r"(?<![\d\-])\d(?:[ \-]?\d){11,18}(?![\d\-])"),
    "phone": re.compile(
        r"(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.\-]\d{3}[\s.\-]\d{2,4}(?!\d)"
    ),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
}

SECRET_MARKERS: dict[str, re.Pattern[str]] = {
    "api_key": re.compile(
        r"(?i)(?:api[_\-]?key|apikey|secret[_\-]?key|client[_\-]?secret|access[_\-]?key)"
        r"[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,}"
    ),
    "provider_key": re.compile(r"\b(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,})\b"),
    "bearer_token": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{20,}={0,2}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
}


def luhn_check(candidate: str) -> bool:
    """Luhn (mod-10) checksum, used so that random long digit runs are not called cards.

    The framework counts payment-card markers as strong evidence of a high-sensitivity
    asset, so a false positive here inflates criticality; the checksum is what keeps the
    marker honest.
    """
    digits = [int(ch) for ch in candidate if ch.isdigit()]
    if len(digits) < 12 or len(digits) > 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def iban_check(candidate: str) -> bool:
    """ISO 13616 mod-97 check, for the same reason ``luhn_check`` exists."""
    compact = "".join(candidate.split()).upper()
    if len(compact) < 15 or len(compact) > 34 or not compact[:2].isalpha() or not compact[2:4].isdigit():
        return False
    rearranged = compact[4:] + compact[:4]
    digits = ""
    for char in rearranged:
        if char.isdigit():
            digits += char
        elif "A" <= char <= "Z":
            digits += str(ord(char) - 55)
        else:
            return False
    remainder = 0
    for char in digits:
        remainder = (remainder * 10 + int(char)) % 97
    return remainder == 1


def find_pii_markers(text: str | None) -> dict[str, int]:
    """Count validated PII markers per category. Empty dict means none were found."""
    if not text:
        return {}
    found: dict[str, int] = {}
    for name, pattern in PII_MARKERS.items():
        hits = pattern.findall(text)
        if name == "card_number":
            hits = [hit for hit in hits if luhn_check(hit)]
        elif name == "iban":
            hits = [hit for hit in hits if iban_check(hit)]
        if hits:
            found[name] = len(hits)
    return found


def find_secret_markers(text: str | None) -> dict[str, int]:
    """Count credential-shaped markers per category (API keys, bearer tokens, private keys)."""
    if not text:
        return {}
    found: dict[str, int] = {}
    for name, pattern in SECRET_MARKERS.items():
        hits = pattern.findall(text)
        if hits:
            found[name] = len(hits)
    return found


def pii_marker_count(text: str | None) -> int:
    """Total validated PII marker hits across all categories."""
    return sum(find_pii_markers(text).values())


def secret_marker_count(text: str | None) -> int:
    """Total secret-marker hits across all categories."""
    return sum(find_secret_markers(text).values())


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_STATIC_CONTENT_TYPES = ("text/css", "text/javascript", "application/javascript", "image/", "font/", "application/font")
_FILE_CONTENT_TYPES = ("application/pdf", "application/octet-stream", "application/zip", "multipart/")


def _content_type_bonus(content_type: str | None) -> tuple[dict[EndpointFunction, float], dict[str, float]]:
    """Content-type driven score bonuses and the flags recorded as evidence features."""
    bonus: dict[EndpointFunction, float] = {}
    flags = {"ct_json": 0.0, "ct_html": 0.0, "ct_static": 0.0, "ct_file": 0.0}
    if not content_type:
        return bonus, flags
    lowered = content_type.split(";")[0].strip().casefold()
    if lowered.startswith(_STATIC_CONTENT_TYPES):
        bonus[EndpointFunction.STATIC_CONTENT] = 1.0
        flags["ct_static"] = 1.0
    elif lowered == "application/json" or lowered.endswith("+json"):
        bonus[EndpointFunction.API_DATA] = 0.3
        flags["ct_json"] = 1.0
    elif lowered.startswith(_FILE_CONTENT_TYPES):
        bonus[EndpointFunction.FILE_IO] = 0.5
        flags["ct_file"] = 1.0
    elif lowered.startswith("text/html"):
        flags["ct_html"] = 1.0
    return bonus, flags


def classify_function(
    path: str,
    params: Sequence[str] | None = None,
    content_type: str | None = None,
) -> tuple[EndpointFunction, dict[str, float]]:
    """Infer the business function of an endpoint from its structure alone.

    Returns the winning :class:`EndpointFunction` and the structural evidence features
    that produced it. The features are returned rather than hidden so that
    ``AssetCriticality.evidence_features`` can carry an auditable record of *why* an
    endpoint was called critical -- which is the whole point of Goal 1.
    """
    params = tuple(params or ())
    blob = " ".join((path or "", *params))
    scores = score_functions(blob)

    bonus, ct_flags = _content_type_bonus(content_type)
    for function, extra in bonus.items():
        scores[function] = round(scores.get(function, 0.0) + extra, 6)

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], FUNCTION_PRIORITY.index(item[0])),
    )
    top_function, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score <= 0.0:
        top_function = EndpointFunction.UNKNOWN

    normalized_path = normalize_text(path or "")
    segments = [segment for segment in normalized_path.split("/") if segment]
    param_tokens = {token for param in params for token in tokenize(param)}

    features: dict[str, float] = {f"lex_{function.value}": score for function, score in scores.items()}
    features.update(ct_flags)
    features.update(
        {
            "path_depth": float(len(segments)),
            "path_token_count": float(len(tokenize(normalized_path))),
            "param_count": float(len(params)),
            "sensitive_param_count": float(len(param_tokens & SENSITIVE_PARAM_TOKENS)),
            "has_id_template": 1.0 if "{" in (path or "") else 0.0,
            "top_function_score": float(top_score),
            "function_margin": float(round(top_score - runner_up, 6)),
        }
    )
    return top_function, features


def merge_feature_maps(*maps: Mapping[str, float] | None) -> dict[str, float]:
    """Shallow-merge evidence feature maps, later maps winning. Used by criticality.py."""
    merged: dict[str, float] = {}
    for mapping in maps:
        if mapping:
            merged.update({key: float(value) for key, value in mapping.items()})
    return merged


def iter_lexicon_tokens() -> Iterable[tuple[EndpointFunction, str, float]]:
    """Flat iteration over the lexicon; used by tests and by documentation generation."""
    for function, tokens in TOKEN_LEXICON.items():
        for token, weight in tokens.items():
            yield function, token, weight
