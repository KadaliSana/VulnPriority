"""Endpoint and privilege topology for one synthetic application (DESIGN.md 3.11).

A synthetic application is only useful if its *surface* is realistic, because Component A
infers criticality from structure alone and Component C builds its graph from hosts and
observed links. A world of twelve identical ``/page/{id}`` routes would make both
components look better than they are.

So each application here gets the shape a real one has: anonymous marketing pages, a login
endpoint that sets a cookie, authenticated user routes carrying identifier segments, an
administrative surface on its own host, a JSON API on a second host, file upload and
download, a search endpoint, and static assets. Endpoints carry ``links_to`` edges, some of
them crossing hosts, which is what gives the attack graph its lateral movement; and a
technology stack that differs per sector, which is what gives applicability something real
to match versions against.

Nothing here decides anything about exploitation. The hidden truths live in
:mod:`vulnprio.synth.world`; this module only lays out the ground they sit on.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from random import Random
from typing import Sequence

from vulnprio.core.enums import EndpointFunction, HttpMethod, PrivilegeLevel, Provenance
from vulnprio.core.models import Endpoint, TechComponent, UntrustedText
from vulnprio.ingest.normalize import make_endpoint_id

__all__ = [
    "SECTORS",
    "SECTOR_TECH",
    "SECTOR_ROUTES",
    "AppSpec",
    "RouteTemplate",
    "CORE_ROUTES",
    "OPTIONAL_ROUTES",
    "derive_seed",
    "generate_app_specs",
    "generate_endpoints",
    "endpoint_functions",
    "INTENDED_FUNCTION",
    "intended_function",
]

#: Sectors the generator knows how to dress an application in (Gap 8 transfer axis).
SECTORS: tuple[str, ...] = ("ecommerce", "healthcare", "saas", "fintech")


def derive_seed(base: int, *parts: object) -> int:
    """Deterministic child seed for ``base`` and a tuple of labels.

    Derived through SHA-256 rather than arithmetic so that neighbouring applications do not
    get correlated streams, and so that the value does not depend on Python's per-process
    string hash randomisation.
    """
    payload = "|".join([str(int(base))] + [str(part) for part in parts])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Technology stacks
# ---------------------------------------------------------------------------

#: Plausible, version-bearing stacks per sector. Versions matter: ``semantic.cpe_match``
#: decides applicability from them, so a stack without versions would make every
#: applicability verdict UNKNOWN and quietly disable Goal 3 in the synthetic world.
SECTOR_TECH: dict[str, tuple[tuple[str, str, str], ...]] = {
    "ecommerce": (
        ("nginx", "nginx", "1.24.0"),
        ("php", "php", "8.1.2"),
        ("wordpress", "wordpress", "6.3.1"),
        ("oracle", "mysql", "8.0.33"),
        ("jquery", "jquery", "3.5.1"),
    ),
    "healthcare": (
        ("apache", "http_server", "2.4.54"),
        ("apache", "tomcat", "9.0.65"),
        ("vmware", "spring_framework", "5.3.20"),
        ("postgresql", "postgresql", "14.5"),
        ("oracle", "openjdk", "11.0.16"),
    ),
    "saas": (
        ("nodejs", "node.js", "18.16.0"),
        ("openjs", "express", "4.18.2"),
        ("facebook", "react", "18.2.0"),
        ("nginx", "nginx", "1.22.1"),
        ("redis", "redis", "7.0.5"),
    ),
    "fintech": (
        ("apache", "struts", "2.5.20"),
        ("oracle", "weblogic_server", "12.2.1.4"),
        ("oracle", "openjdk", "11.0.16"),
        ("apache", "log4j", "2.14.1"),
        ("postgresql", "postgresql", "13.8"),
    ),
    "generic": (
        ("apache", "http_server", "2.4.52"),
        ("python", "django", "4.1.7"),
        ("postgresql", "postgresql", "14.3"),
    ),
}

#: Application name fragments per sector, so app names read like products rather than ids.
_NAME_PARTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "ecommerce": (("Northwind", "Bluecart", "Marketly", "Shopfront", "Tradewinds"),
                  ("Commerce", "Store", "Market", "Retail", "Checkout")),
    "healthcare": (("Medisys", "Carelink", "Vitalis", "Healthpoint", "Clinova"),
                   ("Portal", "Records", "Clinic", "Health", "Care")),
    "saas": (("Fluxdesk", "Teamly", "Opsgrid", "Signalbox", "Workstream"),
             ("Cloud", "Suite", "Platform", "Workspace", "Hub")),
    "fintech": (("Ledgerly", "Paycrest", "Vaultbank", "Quantile", "Clearline"),
                ("Payments", "Bank", "Capital", "Treasury", "Finance")),
    "generic": (("Acme", "Globex", "Initech", "Umbrella", "Hooli"),
                ("App", "Portal", "Service", "System", "Web")),
}


@dataclass(frozen=True)
class AppSpec:
    """Identity and stack of one synthetic application."""

    app_id: str
    app_name: str
    sector: str
    slug: str
    web_host: str
    api_host: str
    admin_host: str
    tech_stack: tuple[TechComponent, ...]
    scanner_name: str = "zap"
    scanner_version: str = "2.14.0"

    @property
    def hosts(self) -> tuple[str, ...]:
        return (self.web_host, self.api_host, self.admin_host)


# ---------------------------------------------------------------------------
# Route templates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteTemplate:
    """One route of the synthetic surface, before it is bound to an application."""

    path: str
    method: HttpMethod
    auth: PrivilegeLevel
    function: EndpointFunction
    host: str = "web"                       # "web" | "api" | "admin"
    content_type: str = "text/html"
    status: int = 200
    sets_cookie: bool = False
    parameters: tuple[str, ...] = ()
    size_bytes: int = 4096
    links: tuple[str, ...] = ()             # paths this route links to
    body: str = ""                          # response sample (untrusted target content)


#: Routes every application has. The framework's structural inference is exercised by all
#: of them: an auth boundary, an admin surface, a PII carrier, an API, upload, search.
CORE_ROUTES: tuple[RouteTemplate, ...] = (
    RouteTemplate(
        path="/", method=HttpMethod.GET, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.STATIC_CONTENT, size_bytes=18_432,
        links=("/login", "/search", "/products"),
        body="Welcome. Sign in to manage your account.",
    ),
    RouteTemplate(
        path="/login", method=HttpMethod.POST, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.AUTH, content_type="application/json", sets_cookie=True,
        parameters=("username", "password", "remember_me"), size_bytes=512,
        links=("/account/profile", "/api/v1/session"),
        body='{"status":"ok","session":"set","mfa_required":false}',
    ),
    RouteTemplate(
        path="/register", method=HttpMethod.POST, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.AUTH, content_type="application/json",
        parameters=("email", "password", "full_name"), size_bytes=640,
        links=("/login",),
        body='{"status":"created","email":"new.user@example.com"}',
    ),
    RouteTemplate(
        path="/search", method=HttpMethod.GET, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.SEARCH, parameters=("q", "page", "sort"), size_bytes=9_216,
        links=("/products/{id}",),
        body="Search results for your query. 42 matches.",
    ),
    RouteTemplate(
        path="/products/{id}", method=HttpMethod.GET, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.STATIC_CONTENT, parameters=("id", "variant"), size_bytes=12_288,
        links=("/cart",),
        body="Product detail page with price and stock level.",
    ),
    RouteTemplate(
        path="/account/profile", method=HttpMethod.GET, auth=PrivilegeLevel.USER,
        function=EndpointFunction.PII_DATA, content_type="text/html", size_bytes=7_168,
        links=("/account/orders/{id}", "/api/v1/users/{id}"),
        body=(
            "Account profile. email: jane.doe@example.com phone: +1-555-0143 "
            "date of birth: 1984-02-11 address: 14 Mill Lane"
        ),
    ),
    RouteTemplate(
        path="/account/orders/{id}", method=HttpMethod.GET, auth=PrivilegeLevel.USER,
        function=EndpointFunction.PII_DATA, parameters=("id",), size_bytes=6_144,
        links=("/checkout/payment",),
        body="Order 10041 shipped to 14 Mill Lane. Billing name: Jane Doe.",
    ),
    RouteTemplate(
        path="/checkout/payment", method=HttpMethod.POST, auth=PrivilegeLevel.USER,
        function=EndpointFunction.PAYMENT, content_type="application/json",
        parameters=("card_number", "cvv", "amount", "currency"), size_bytes=1_024,
        links=("/api/v1/payments",),
        body='{"status":"authorised","card_last4":"4242","amount":"129.00","currency":"USD"}',
    ),
    RouteTemplate(
        path="/upload", method=HttpMethod.POST, auth=PrivilegeLevel.USER,
        function=EndpointFunction.FILE_IO, content_type="application/json",
        parameters=("file", "filename", "content_type"), size_bytes=384,
        links=("/files/{id}/download",),
        body='{"status":"stored","path":"/var/www/uploads/9f2.bin"}',
    ),
    RouteTemplate(
        path="/files/{id}/download", method=HttpMethod.GET, auth=PrivilegeLevel.USER,
        function=EndpointFunction.FILE_IO, content_type="application/octet-stream",
        parameters=("id", "disposition"), size_bytes=65_536,
        body="binary attachment stream",
    ),
    RouteTemplate(
        path="/api/v1/session", method=HttpMethod.POST, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.AUTH, host="api", content_type="application/json",
        sets_cookie=True, parameters=("username", "password"), size_bytes=320,
        links=("/api/v1/users/{id}",),
        body='{"token":"eyJhbGciOiJIUzI1NiJ9.payload.signature","expires_in":3600}',
    ),
    RouteTemplate(
        path="/api/v1/users/{id}", method=HttpMethod.GET, auth=PrivilegeLevel.USER,
        function=EndpointFunction.API_DATA, host="api", content_type="application/json",
        parameters=("id", "include"), size_bytes=2_048,
        links=("/api/v1/orders",),
        body='{"id":4711,"email":"jane.doe@example.com","ssn":"078-05-1120","role":"user"}',
    ),
    RouteTemplate(
        path="/api/v1/orders", method=HttpMethod.GET, auth=PrivilegeLevel.USER,
        function=EndpointFunction.API_DATA, host="api", content_type="application/json",
        parameters=("customer_id", "limit", "offset"), size_bytes=16_384,
        body='{"orders":[{"id":10041,"total":"129.00","customer":"jane.doe@example.com"}]}',
    ),
    RouteTemplate(
        path="/admin", method=HttpMethod.GET, auth=PrivilegeLevel.ADMIN,
        function=EndpointFunction.ADMIN, host="admin", status=302, size_bytes=1_024,
        links=("/admin/users/{id}", "/admin/settings"),
        body="Administration console. Redirecting to sign-in.",
    ),
    RouteTemplate(
        path="/admin/users/{id}", method=HttpMethod.GET, auth=PrivilegeLevel.ADMIN,
        function=EndpointFunction.ADMIN, host="admin", status=403,
        parameters=("id", "role"), size_bytes=5_120,
        links=("/admin/export",),
        body="Forbidden. Administrator role required to view user records.",
    ),
    RouteTemplate(
        path="/static/app.css", method=HttpMethod.GET, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.STATIC_CONTENT, content_type="text/css", size_bytes=40_960,
        body=".header{color:#333}",
    ),
)

#: Routes drawn on top of the core set until the endpoint budget is filled.
OPTIONAL_ROUTES: tuple[RouteTemplate, ...] = (
    RouteTemplate(
        path="/about", method=HttpMethod.GET, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.STATIC_CONTENT, size_bytes=8_192,
        body="About the company.",
    ),
    RouteTemplate(
        path="/contact", method=HttpMethod.POST, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.STATIC_CONTENT, parameters=("name", "email", "message"),
        size_bytes=2_048, body="Thank you, we will be in touch.",
    ),
    RouteTemplate(
        path="/password-reset", method=HttpMethod.POST, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.AUTH, content_type="application/json",
        parameters=("email", "token"), size_bytes=512,
        body='{"status":"sent","token_ttl_minutes":30}',
    ),
    RouteTemplate(
        path="/cart", method=HttpMethod.GET, auth=PrivilegeLevel.USER,
        function=EndpointFunction.API_DATA, content_type="application/json",
        parameters=("session",), size_bytes=3_072,
        links=("/checkout/payment",),
        body='{"items":3,"subtotal":"129.00"}',
    ),
    RouteTemplate(
        path="/users/{id}/orders/{id}", method=HttpMethod.GET, auth=PrivilegeLevel.USER,
        function=EndpointFunction.PII_DATA, parameters=("id",), size_bytes=6_656,
        body="Order detail for customer 4711. Shipping: 14 Mill Lane.",
    ),
    RouteTemplate(
        path="/account/settings", method=HttpMethod.PUT, auth=PrivilegeLevel.USER,
        function=EndpointFunction.PII_DATA, content_type="application/json",
        parameters=("email", "phone", "notify"), size_bytes=1_536,
        body='{"email":"jane.doe@example.com","phone":"+1-555-0143"}',
    ),
    RouteTemplate(
        path="/api/v1/payments", method=HttpMethod.POST, auth=PrivilegeLevel.USER,
        function=EndpointFunction.PAYMENT, host="api", content_type="application/json",
        parameters=("amount", "currency", "token"), size_bytes=1_024,
        body='{"status":"captured","amount":"129.00","card_last4":"4242"}',
    ),
    RouteTemplate(
        path="/api/v1/export", method=HttpMethod.GET, auth=PrivilegeLevel.ADMIN,
        function=EndpointFunction.API_DATA, host="api", content_type="application/json",
        parameters=("format", "since"), size_bytes=131_072,
        body='{"rows":12840,"format":"csv"}',
    ),
    RouteTemplate(
        path="/api/v1/webhooks", method=HttpMethod.POST, auth=PrivilegeLevel.USER,
        function=EndpointFunction.API_DATA, host="api", content_type="application/json",
        parameters=("url", "secret", "events"), size_bytes=768,
        body='{"status":"registered","url":"https://hooks.example.net/inbound"}',
    ),
    RouteTemplate(
        path="/admin/settings", method=HttpMethod.POST, auth=PrivilegeLevel.ADMIN,
        function=EndpointFunction.ADMIN, host="admin", content_type="application/json",
        parameters=("key", "value"), size_bytes=1_024,
        body='{"status":"saved"}',
    ),
    RouteTemplate(
        path="/admin/export", method=HttpMethod.GET, auth=PrivilegeLevel.ADMIN,
        function=EndpointFunction.ADMIN, host="admin", content_type="text/csv",
        parameters=("table", "format"), size_bytes=262_144,
        body="id,email,ssn,balance\\n4711,jane.doe@example.com,078-05-1120,1420.50",
    ),
    RouteTemplate(
        path="/admin/backup", method=HttpMethod.POST, auth=PrivilegeLevel.ADMIN,
        function=EndpointFunction.FILE_IO, host="admin", content_type="application/json",
        parameters=("target",), size_bytes=512,
        body='{"status":"queued","target":"s3://backups/nightly"}',
    ),
    RouteTemplate(
        path="/static/app.js", method=HttpMethod.GET, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.STATIC_CONTENT, content_type="application/javascript",
        size_bytes=184_320, body="window.app=window.app||{};",
    ),
    RouteTemplate(
        path="/assets/logo.png", method=HttpMethod.GET, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.STATIC_CONTENT, content_type="image/png",
        size_bytes=24_576, body="",
    ),
    RouteTemplate(
        path="/health", method=HttpMethod.GET, auth=PrivilegeLevel.NONE,
        function=EndpointFunction.API_DATA, host="api", content_type="application/json",
        size_bytes=128, body='{"status":"up","version":"3.4.1"}',
    ),
)

#: Sector-specific routes, appended to the optional pool for that sector. These are what
#: make leave-one-application-out transfer a real test rather than a relabelling exercise.
SECTOR_ROUTES: dict[str, tuple[RouteTemplate, ...]] = {
    "ecommerce": (
        RouteTemplate(
            path="/checkout/coupon", method=HttpMethod.POST, auth=PrivilegeLevel.USER,
            function=EndpointFunction.PAYMENT, content_type="application/json",
            parameters=("code",), size_bytes=384, body='{"discount":"10%"}',
        ),
        RouteTemplate(
            path="/api/v1/inventory/{id}", method=HttpMethod.GET, auth=PrivilegeLevel.USER,
            function=EndpointFunction.API_DATA, host="api", content_type="application/json",
            parameters=("id",), size_bytes=1_024, body='{"sku":"A-771","stock":12}',
        ),
    ),
    "healthcare": (
        RouteTemplate(
            path="/patients/{id}", method=HttpMethod.GET, auth=PrivilegeLevel.USER,
            function=EndpointFunction.PII_DATA, parameters=("id",), size_bytes=10_240,
            body=(
                "Patient record. name: John Roe, date of birth: 1971-09-02, "
                "medical record number: MRN-88421, diagnosis codes: E11.9"
            ),
        ),
        RouteTemplate(
            path="/api/v1/records/{id}", method=HttpMethod.GET, auth=PrivilegeLevel.USER,
            function=EndpointFunction.PII_DATA, host="api", content_type="application/json",
            parameters=("id", "section"), size_bytes=8_192,
            body='{"mrn":"MRN-88421","ssn":"078-05-1120","insurer":"Statewide Health"}',
        ),
        RouteTemplate(
            path="/admin/audit-log", method=HttpMethod.GET, auth=PrivilegeLevel.ADMIN,
            function=EndpointFunction.ADMIN, host="admin", content_type="application/json",
            parameters=("since",), size_bytes=98_304, body='{"entries":9142}',
        ),
    ),
    "saas": (
        RouteTemplate(
            path="/api/v1/tenants/{id}", method=HttpMethod.GET, auth=PrivilegeLevel.ADMIN,
            function=EndpointFunction.API_DATA, host="api", content_type="application/json",
            parameters=("id",), size_bytes=2_048, body='{"tenant":"acme","plan":"enterprise"}',
        ),
        RouteTemplate(
            path="/api/v1/graphql", method=HttpMethod.POST, auth=PrivilegeLevel.USER,
            function=EndpointFunction.API_DATA, host="api", content_type="application/json",
            parameters=("query", "variables"), size_bytes=4_096, body='{"data":{"me":{"id":4711}}}',
        ),
        RouteTemplate(
            path="/integrations/oauth/callback", method=HttpMethod.GET,
            auth=PrivilegeLevel.NONE, function=EndpointFunction.AUTH,
            parameters=("code", "state", "redirect_uri"), size_bytes=768,
            body="Linking your account...",
        ),
    ),
    "fintech": (
        RouteTemplate(
            path="/accounts/{id}/transactions", method=HttpMethod.GET,
            auth=PrivilegeLevel.USER, function=EndpointFunction.PAYMENT,
            parameters=("id", "from", "to"), size_bytes=32_768,
            body="Statement for account 4711. Balance 1420.50 USD. IBAN GB29 NWBK 6016 1331 9268 19",
        ),
        RouteTemplate(
            path="/transfers", method=HttpMethod.POST, auth=PrivilegeLevel.USER,
            function=EndpointFunction.PAYMENT, content_type="application/json",
            parameters=("from_account", "to_account", "amount", "currency"), size_bytes=640,
            body='{"status":"pending","amount":"5000.00","currency":"EUR"}',
        ),
        RouteTemplate(
            path="/admin/ledger", method=HttpMethod.GET, auth=PrivilegeLevel.ADMIN,
            function=EndpointFunction.ADMIN, host="admin", content_type="text/csv",
            parameters=("day",), size_bytes=524_288,
            body="account,balance\\n4711,1420.50",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Application specs
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in name).strip("-")


def generate_app_specs(
    n_apps: int,
    sectors: Sequence[str],
    rng: Random,
    *,
    domain: str = "example.com",
) -> tuple[AppSpec, ...]:
    """Identities and stacks for ``n_apps`` applications, cycling through ``sectors``.

    Sectors are cycled rather than sampled so that a run with four sectors and eight
    applications has exactly two of each: leave-one-application-out transfer is only
    interpretable when the sectors are balanced.
    """
    pool = tuple(sectors) or SECTORS
    specs: list[AppSpec] = []
    for index in range(int(n_apps)):
        sector = pool[index % len(pool)]
        heads, tails = _NAME_PARTS.get(sector, _NAME_PARTS["generic"])
        head = heads[index % len(heads)]
        tail = tails[(index // len(heads) + index) % len(tails)]
        app_name = f"{head} {tail}"
        slug = f"{_slugify(head)}{index + 1}"
        app_id = f"app_{slug}"
        stack_source = SECTOR_TECH.get(sector, SECTOR_TECH["generic"])
        # Drop at most one component so that two applications in a sector are not
        # byte-identical stacks; the dropped one is chosen deterministically from rng.
        keep = list(stack_source)
        if len(keep) > 3 and rng.random() < 0.5:
            keep.pop(rng.randrange(len(keep)))
        stack = tuple(
            TechComponent(
                vendor=vendor,
                product=product,
                version=version,
                cpe=f"cpe:2.3:a:{vendor}:{product}:{version}:*:*:*:*:*:*:*",
            )
            for vendor, product, version in keep
        )
        specs.append(
            AppSpec(
                app_id=app_id,
                app_name=app_name,
                sector=sector,
                slug=slug,
                web_host=f"www.{slug}.{domain}",
                api_host=f"api.{slug}.{domain}",
                admin_host=f"admin.{slug}.{domain}",
                tech_stack=stack,
            )
        )
    return tuple(specs)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _host_of(app: AppSpec, kind: str) -> str:
    if kind == "api":
        return app.api_host
    if kind == "admin":
        return app.admin_host
    return app.web_host


def _select_routes(count: int, sector: str, rng: Random) -> tuple[RouteTemplate, ...]:
    """Core routes plus enough optional ones to reach ``count``, deterministically."""
    chosen = list(CORE_ROUTES)
    pool = list(OPTIONAL_ROUTES) + list(SECTOR_ROUTES.get(sector, ()))
    rng.shuffle(pool)
    for route in pool:
        if len(chosen) >= count:
            break
        chosen.append(route)
    return tuple(chosen)


def generate_endpoints(
    app: AppSpec,
    rng: Random,
    config: object | None = None,
    *,
    n_endpoints: int | None = None,
) -> tuple[Endpoint, ...]:
    """Build the endpoint surface of one application.

    ``config`` is a :class:`~vulnprio.core.config.SyntheticConfig` (or anything exposing
    ``endpoints_per_app``); ``n_endpoints`` overrides it. The result is stable for a given
    ``(app, rng state, count)``: every value is drawn from ``rng`` and nothing consults the
    clock, the environment or the network.
    """
    if n_endpoints is None:
        bounds = getattr(config, "endpoints_per_app", (12, 30))
        low, high = int(bounds[0]), int(bounds[1])
        low = max(len(CORE_ROUTES), low)
        high = max(low, high)
        n_endpoints = rng.randint(low, high)

    routes = _select_routes(int(n_endpoints), app.sector, rng)

    # Pass one: identifiers, so links_to can reference endpoints that do not exist yet.
    bound: list[tuple[RouteTemplate, str, str]] = []
    by_path: dict[str, str] = {}
    for route in routes:
        host = _host_of(app, route.host)
        endpoint_id = make_endpoint_id(app.app_id, host, route.method, route.path)
        bound.append((route, host, endpoint_id))
        by_path.setdefault(route.path, endpoint_id)

    endpoints: list[Endpoint] = []
    for route, host, endpoint_id in bound:
        links = tuple(
            dict.fromkeys(
                by_path[target] for target in route.links if target in by_path and by_path[target] != endpoint_id
            )
        )
        # A small amount of observed stack per endpoint: the server banner is visible
        # everywhere, the application framework only where it renders.
        observed: list[TechComponent] = []
        if app.tech_stack:
            observed.append(app.tech_stack[0])
            if route.function not in (EndpointFunction.STATIC_CONTENT,) and len(app.tech_stack) > 1:
                observed.append(app.tech_stack[1 + rng.randrange(len(app.tech_stack) - 1)])
        scheme = "https"
        url = f"{scheme}://{host}{route.path}"
        sample = (
            UntrustedText(text=route.body, provenance=Provenance.TARGET_RESPONSE, language="en")
            if route.body
            else None
        )
        endpoints.append(
            Endpoint(
                endpoint_id=endpoint_id,
                app_id=app.app_id,
                host=host,
                url=url,
                path=route.path,
                method=route.method,
                auth_required=route.auth,
                internet_facing=route.host != "admin" or rng.random() < 0.6,
                response_status=route.status,
                response_content_type=route.content_type,
                response_size_bytes=route.size_bytes,
                sets_cookie=route.sets_cookie,
                parameters=route.parameters,
                links_to=links,
                response_sample=sample,
                observed_tech=tuple(dict.fromkeys(observed)),
            )
        )
    return tuple(endpoints)


def endpoint_functions(routes: Sequence[RouteTemplate] | None = None) -> dict[str, EndpointFunction]:
    """Path to intended :class:`EndpointFunction`, used by the oracle's latent criticality.

    This is the *generator's* intent, not the framework's inference. The two are never
    compared inside the pipeline; the mapping exists so the latent world can weight a
    payment route above a stylesheet without reading Component A's answer.
    """
    if routes is None:
        routes = (
            list(CORE_ROUTES)
            + list(OPTIONAL_ROUTES)
            + [route for group in SECTOR_ROUTES.values() for route in group]
        )
    return {route.path: route.function for route in routes}


#: Every known route's intended function, resolved once.
INTENDED_FUNCTION: dict[str, EndpointFunction] = endpoint_functions()


def intended_function(path: str) -> EndpointFunction:
    """Generator-intended function of a path (``UNKNOWN`` for anything unrecognised)."""
    return INTENDED_FUNCTION.get(path, EndpointFunction.UNKNOWN)
