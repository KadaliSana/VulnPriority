"""Technology fingerprinting from scanner-observed responses.

Why this matters: Goal 3 (applicability) is decided by matching an advisory's affected
version range against what the application actually runs. That comparison needs observed
components, and the only evidence a black-box scan leaves behind is response headers,
framework cookies, generator meta tags and telltale URL paths. Everything inferred here
is tier ``SCANNER`` structural evidence, never a claim the target wrote about itself.

Pure and deterministic: same inputs, same tuple, same order.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

from vulnprio.core.models import TechComponent

__all__ = [
    "fingerprint_response",
    "fingerprint_library",
    "merge_tech",
    "parse_product_tokens",
    "make_cpe",
]

#: Token (as written in a header, cookie or meta tag) to ``(vendor, product)``.
_PRODUCTS: dict[str, tuple[str | None, str]] = {
    "apache": ("apache", "http_server"),
    "httpd": ("apache", "http_server"),
    "apache-coyote": ("apache", "tomcat"),
    "tomcat": ("apache", "tomcat"),
    "struts": ("apache", "struts"),
    "struts2": ("apache", "struts"),
    "nginx": ("nginx", "nginx"),
    "openresty": ("openresty", "openresty"),
    "openssl": ("openssl", "openssl"),
    "php": ("php", "php"),
    "asp.net": ("microsoft", "asp.net"),
    "aspnet": ("microsoft", "asp.net"),
    "iis": ("microsoft", "iis"),
    "microsoft-iis": ("microsoft", "iis"),
    "kestrel": ("microsoft", "kestrel"),
    "jetty": ("eclipse", "jetty"),
    "servlet": ("oracle", "java_servlet"),
    "jsp": ("oracle", "jsp"),
    "express": ("openjs", "express"),
    "node.js": ("nodejs", "node.js"),
    "nodejs": ("nodejs", "node.js"),
    "gunicorn": ("gunicorn", "gunicorn"),
    "werkzeug": ("pallets", "werkzeug"),
    "flask": ("pallets", "flask"),
    "django": ("djangoproject", "django"),
    "laravel": ("laravel", "laravel"),
    "codeigniter": ("codeigniter", "codeigniter"),
    "symfony": ("sensiolabs", "symfony"),
    "wordpress": ("wordpress", "wordpress"),
    "woocommerce": ("woocommerce", "woocommerce"),
    "drupal": ("drupal", "drupal"),
    "joomla": ("joomla", "joomla"),
    "joomla!": ("joomla", "joomla"),
    "typo3": ("typo3", "typo3"),
    "magento": ("magento", "magento"),
    "phpmyadmin": ("phpmyadmin", "phpmyadmin"),
    "spring": ("vmware", "spring_framework"),
    "next.js": ("vercel", "next.js"),
    "nextjs": ("vercel", "next.js"),
    "cloudflare": ("cloudflare", "cloudflare"),
    "rails": ("rubyonrails", "ruby_on_rails"),
    "passenger": ("phusion", "passenger"),
    "jquery": ("jquery", "jquery"),
    "bootstrap": ("getbootstrap", "bootstrap"),
    "angular": ("angularjs", "angular.js"),
    "angularjs": ("angularjs", "angular.js"),
    "vue": ("vuejs", "vue.js"),
    "react": ("facebook", "react"),
    "lodash": ("lodash", "lodash"),
    "moment": ("momentjs", "moment.js"),
    "handlebars": ("handlebarsjs", "handlebars"),
}

#: Framework session / CSRF cookies. Presence of the cookie alone identifies the stack.
_COOKIE_PRODUCTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^jsessionid$", re.IGNORECASE), "servlet"),
    (re.compile(r"^phpsessid$", re.IGNORECASE), "php"),
    (re.compile(r"^asp\.?net_sessionid$", re.IGNORECASE), "asp.net"),
    (re.compile(r"^\.aspnet\.", re.IGNORECASE), "asp.net"),
    (re.compile(r"^asp\.net_sessionid$", re.IGNORECASE), "asp.net"),
    (re.compile(r"^csrftoken$", re.IGNORECASE), "django"),
    (re.compile(r"^django_language$", re.IGNORECASE), "django"),
    (re.compile(r"^laravel_session$", re.IGNORECASE), "laravel"),
    (re.compile(r"^ci_session$", re.IGNORECASE), "codeigniter"),
    (re.compile(r"^connect\.sid$", re.IGNORECASE), "express"),
    (re.compile(r"^wordpress_(logged_in|sec|test_cookie)", re.IGNORECASE), "wordpress"),
    (re.compile(r"^wp-settings", re.IGNORECASE), "wordpress"),
    (re.compile(r"^sess[0-9a-f]{20,}$", re.IGNORECASE), "drupal"),
    (re.compile(r"^joomla_", re.IGNORECASE), "joomla"),
    (re.compile(r"^_rails_session$", re.IGNORECASE), "rails"),
)

#: URL path signatures. Cheap, high-precision and available even without a response body.
_PATH_SIGNATURES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"/wp-(admin|content|includes|json)(/|$)", re.IGNORECASE), "wordpress"),
    (re.compile(r"/wp-login\.php", re.IGNORECASE), "wordpress"),
    (re.compile(r"/sites/(default|all)/(files|modules|themes)(/|$)", re.IGNORECASE), "drupal"),
    (re.compile(r"/core/misc/drupal\.js", re.IGNORECASE), "drupal"),
    (re.compile(r"/administrator/index\.php", re.IGNORECASE), "joomla"),
    (re.compile(r"[?&]option=com_", re.IGNORECASE), "joomla"),
    (re.compile(r"/_next/static/", re.IGNORECASE), "next.js"),
    (re.compile(r"\.(action|do)(\?|$)", re.IGNORECASE), "struts"),
    (re.compile(r"/phpmyadmin(/|$)", re.IGNORECASE), "phpmyadmin"),
    (re.compile(r"/typo3(/|conf/)", re.IGNORECASE), "typo3"),
    (re.compile(r"/static/version\d+/frontend/", re.IGNORECASE), "magento"),
)

#: Response-body signatures.
_BODY_SIGNATURES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"csrfmiddlewaretoken", re.IGNORECASE), "django"),
    (re.compile(r"__VIEWSTATE"), "asp.net"),
    (re.compile(r"wp-content/(themes|plugins)/", re.IGNORECASE), "wordpress"),
    (re.compile(r"(Drupal\.settings|data-drupal-)", re.IGNORECASE), "drupal"),
    (re.compile(r"/_next/static/", re.IGNORECASE), "next.js"),
    (re.compile(r"/media/(system|jui)/js/", re.IGNORECASE), "joomla"),
    (re.compile(r"laravel_session", re.IGNORECASE), "laravel"),
)

#: Versioned front-end library filenames, e.g. ``jquery-1.12.4.min.js``.
_LIBRARY_FILE = re.compile(
    r"\b(?P<name>jquery(?:-ui)?|angular|bootstrap|vue|react|lodash|moment|handlebars|dojo)"
    r"[-.](?P<version>\d+(?:\.\d+){1,3})(?:\.min)?\.js",
    re.IGNORECASE,
)

#: ``Server: Apache/2.4.41`` style tokens.
_PRODUCT_TOKEN = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9_.+-]*)(?:/(?P<version>[^\s;,()]+))?$")

#: ``<meta name="generator" content="WordPress 6.4.2">`` in either attribute order.
_META_GENERATOR = (
    re.compile(
        r"<meta[^>]*?name\s*=\s*[\"']generator[\"'][^>]*?content\s*=\s*[\"']([^\"']+)[\"']",
        re.IGNORECASE,
    ),
    re.compile(
        r"<meta[^>]*?content\s*=\s*[\"']([^\"']+)[\"'][^>]*?name\s*=\s*[\"']generator[\"']",
        re.IGNORECASE,
    ),
)

_VERSION_RE = re.compile(r"^v?\d+(?:\.\d+)*(?:[-_.]?[A-Za-z0-9]+)?$")

#: Headers whose value is a product token list.
_PRODUCT_HEADERS = ("server", "x-powered-by", "x-generator", "via", "x-server")

#: Headers whose value is a bare version for a known product.
_VERSION_HEADERS: dict[str, str] = {
    "x-aspnet-version": "asp.net",
    "x-aspnetmvc-version": "asp.net",
    "x-drupal-cache": "drupal",
    "x-drupal-dynamic-cache": "drupal",
    "x-nextjs-cache": "next.js",
    "x-shopify-stage": "shopify",
}


def make_cpe(vendor: str | None, product: str, version: str | None) -> str | None:
    """CPE 2.3 string for a component, so ``semantic.cpe_match`` has something to match on."""
    if not vendor or not product:
        return None
    return f"cpe:2.3:a:{vendor}:{product}:{version or '*'}:*:*:*:*:*:*:*"


def _clean_version(raw: str | None) -> str | None:
    if not raw:
        return None
    candidate = raw.strip().strip("()[],;").lstrip("vV")
    if not candidate or not _VERSION_RE.match(candidate) or not candidate[0].isdigit():
        return None
    return candidate


def _component(token: str, version: str | None) -> TechComponent | None:
    """Build a component for a recognised product token, or ``None`` if unknown."""
    entry = _PRODUCTS.get(token.strip().lower())
    if entry is None:
        return None
    vendor, product = entry
    cleaned = _clean_version(version)
    return TechComponent(
        vendor=vendor, product=product, version=cleaned, cpe=make_cpe(vendor, product, cleaned)
    )


def parse_product_tokens(value: str | None) -> tuple[TechComponent, ...]:
    """Parse a ``Name/version Name/version`` header value into known components.

    ``"Apache/2.4.41 (Ubuntu) OpenSSL/1.1.1f"`` yields the Apache and OpenSSL components
    and silently ignores the platform comment, which carries no CPE meaning.
    """
    if not value:
        return ()
    found: list[TechComponent] = []
    for raw_token in re.split(r"[\s,;]+", value.strip()):
        token = raw_token.strip()
        if not token or token.startswith("("):
            continue
        match = _PRODUCT_TOKEN.match(token)
        if not match:
            continue
        component = _component(match.group("name"), match.group("version"))
        if component is not None:
            found.append(component)
    return tuple(found)


def _cookie_names(
    cookies: Iterable[str] | Mapping[str, str] | None, headers: Mapping[str, str]
) -> tuple[str, ...]:
    names: list[str] = []
    if isinstance(cookies, Mapping):
        names.extend(str(key) for key in cookies)
    elif cookies is not None:
        for entry in cookies:
            names.append(str(entry).split("=", 1)[0].strip())
    for header in ("set-cookie", "cookie"):
        raw = headers.get(header, "")
        for chunk in re.split(r"[\n;,]", raw):
            name = chunk.split("=", 1)[0].strip()
            if name and "/" not in name and " " not in name:
                names.append(name)
    seen: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.append(name)
    return tuple(seen)


def _generator_components(body: str) -> list[TechComponent]:
    """Components declared by a ``generator`` meta tag (``WordPress 6.4.2``)."""
    out: list[TechComponent] = []
    for pattern in _META_GENERATOR:
        for content in pattern.findall(body):
            words = re.split(r"[\s/]+", str(content).strip())
            if not words:
                continue
            name = words[0].strip().rstrip("!,")
            version = words[1] if len(words) > 1 else None
            component = _component(name, version)
            if component is not None:
                out.append(component)
    return out


def _library_components(text: str) -> list[TechComponent]:
    out: list[TechComponent] = []
    for match in _LIBRARY_FILE.finditer(text):
        component = _component(match.group("name").lower(), match.group("version"))
        if component is not None:
            out.append(component)
    return out


def fingerprint_response(
    headers: Mapping[str, str] | None = None,
    cookies: Iterable[str] | Mapping[str, str] | None = None,
    body: str | None = None,
    url: str | None = None,
) -> tuple[TechComponent, ...]:
    """Infer the technology stack behind one observed response.

    Combines four independent structural signals - product headers, framework cookies,
    ``generator`` meta tags and path/body signatures - because any one of them is easy for
    an operator to suppress and the applicability decision needs whatever survives.
    """
    lowered = {str(key).lower(): str(value) for key, value in (headers or {}).items()}
    found: list[TechComponent] = []

    for header in _PRODUCT_HEADERS:
        found.extend(parse_product_tokens(lowered.get(header)))
    for header, token in _VERSION_HEADERS.items():
        if header in lowered:
            component = _component(token, lowered[header])
            if component is not None:
                found.append(component)

    for name in _cookie_names(cookies, lowered):
        for pattern, token in _COOKIE_PRODUCTS:
            if pattern.search(name):
                component = _component(token, None)
                if component is not None:
                    found.append(component)
                break

    body_text = body or ""
    if body_text:
        found.extend(_generator_components(body_text))
        for pattern, token in _BODY_SIGNATURES:
            if pattern.search(body_text):
                component = _component(token, None)
                if component is not None:
                    found.append(component)
        found.extend(_library_components(body_text))

    if url:
        for pattern, token in _PATH_SIGNATURES:
            if pattern.search(url):
                component = _component(token, None)
                if component is not None:
                    found.append(component)
        found.extend(_library_components(url))

    return merge_tech(found)


def fingerprint_library(*texts: str | None) -> TechComponent | None:
    """The single versioned component a finding is *about*, if any.

    Used for ``Finding.affected_component``: a "vulnerable JS library" alert names the
    library and version in its evidence, and that - not the web server - is the thing the
    CVE applies to.
    """
    for text in texts:
        if not text:
            continue
        libraries = _library_components(str(text))
        if libraries:
            return libraries[0]
    for text in texts:
        if not text:
            continue
        for match in re.finditer(
            r"\b([A-Za-z][A-Za-z0-9_.+-]*)[/\s]v?(\d+(?:\.\d+){1,3})\b", str(text)
        ):
            component = _component(match.group(1), match.group(2))
            if component is not None and component.version is not None:
                return component
    return None


def merge_tech(components: Iterable[TechComponent]) -> tuple[TechComponent, ...]:
    """De-duplicate components by ``(vendor, product)``, preferring a versioned observation.

    Sorted deterministically so that a scan's ``tech_stack`` is reproducible byte for byte.
    """
    best: dict[tuple[str | None, str], TechComponent] = {}
    for component in components:
        key = (component.vendor, component.product)
        current = best.get(key)
        if current is None or (current.version is None and component.version is not None):
            best[key] = component
    return tuple(
        sorted(best.values(), key=lambda item: (item.product, item.vendor or "", item.version or ""))
    )
