"""Per-weakness remediation guidance.

A curated knowledge base keyed by CWE. Every entry is written from a named public source
(an OWASP cheat sheet where one exists, otherwise the OWASP Testing Guide or the relevant
project) and says four things a scanner does not:

* what the fix actually is, as steps someone can follow;
* how you know it worked, because "we changed the code" is not evidence;
* roughly what it costs, so the remediation plan is not a wish list;
* the common wrong fix, because the wrong fix is what gets shipped under time pressure.

Nothing here is generated. The text is fixed, reviewed prose, so two runs over the same
findings produce the same document, and a reader can argue with a specific sentence.

``guidance_for`` never guesses: a CWE with no entry returns the generic playbook, which
says in its own first line that it is generic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = [
    "Remediation",
    "REMEDIATIONS",
    "GENERIC_REMEDIATION",
    "guidance_for",
    "known_cwes",
    "weakness_name",
]


class Remediation(BaseModel):
    """One weakness's remediation playbook."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cwe_id: int | None = None
    weakness: str = ""
    summary: str = ""
    steps: tuple[str, ...] = ()
    verification: str = ""
    typical_effort: str = ""
    do_not: str = ""
    source: str = ""
    generic: bool = False
    #: Filled by :func:`guidance_for` from the finding itself; never from the knowledge base.
    context: str = ""

    @property
    def label(self) -> str:
        return f"CWE-{self.cwe_id} {self.weakness}".strip() if self.cwe_id else self.weakness


def _r(
    cwe_id: int,
    weakness: str,
    summary: str,
    steps: tuple[str, ...],
    verification: str,
    typical_effort: str,
    do_not: str,
    source: str,
) -> Remediation:
    return Remediation(
        cwe_id=cwe_id,
        weakness=weakness,
        summary=summary,
        steps=steps,
        verification=verification,
        typical_effort=typical_effort,
        do_not=do_not,
        source=source,
    )


REMEDIATIONS: dict[int, Remediation] = {
    16: _r(
        16,
        "Configuration",
        "A deployed setting, not the code, is what exposes this; fix it in the configuration "
        "baseline so it cannot come back on the next deployment.",
        (
            "Write the hardened setting into the configuration that is actually deployed "
            "(infrastructure-as-code, container image or server config), not into a running host.",
            "Turn off debug and developer modes, sample applications, default accounts and "
            "modules the application does not use.",
            "Separate configuration per environment so a development default cannot reach "
            "production, and keep no production secret in the repository.",
            "Re-run the change through the normal deployment path so the fix is reproducible.",
            "Add a drift check that fails the build or alerts when the setting is not present.",
        ),
        "Re-run the configuration scan (or the hardening benchmark) against the deployed "
        "environment and confirm the setting is reported as compliant, then redeploy from a "
        "clean pipeline and confirm it is still compliant.",
        "1 to 4 hours for a single setting; a day or more if a baseline has to be written first.",
        "Do not fix it by hand on the production host. The next deployment overwrites the fix "
        "and the finding returns with nobody watching for it.",
        "OWASP Testing Guide, Configuration and Deployment Management Testing (OTG-CONFIG); "
        "OWASP Top 10 A05 Security Misconfiguration.",
    ),
    22: _r(
        22,
        "Path traversal",
        "Stop deriving file system paths from request data; where a path must be derived, "
        "resolve it first and then prove it is inside the permitted directory.",
        (
            "Replace the user-supplied path with an identifier that the server maps to a path "
            "it already knows (a database row or a fixed lookup table).",
            "Where a path really must be built, canonicalise the full resolved path first, then "
            "assert it is a descendant of the base directory, and reject it otherwise.",
            "Decode the input exactly once before validating, and reject anything that still "
            "contains separators, null bytes or drive letters after decoding.",
            "Run the process with an account that cannot read anything outside the content "
            "directory, and use a chroot, container mount or jail where the platform allows it.",
            "Return the same response for a rejected path and a missing file so the check is not "
            "itself an enumeration oracle.",
        ),
        "Request the affected parameter with `../`, a doubly-encoded `%252e%252e%252f`, a "
        "`....//` variant and an absolute path; each must return the same rejection, and the "
        "server log must show the resolved-path check refusing them.",
        "3 to 8 hours per affected handler, plus regression tests.",
        "Do not strip `../` from the string. A single non-recursive strip turns `....//` back "
        "into `../`, and the filter becomes the vulnerability.",
        "OWASP Cheat Sheet Series, Input Validation; OWASP Testing Guide, Testing for Local "
        "File Inclusion.",
    ),
    78: _r(
        78,
        "OS command injection",
        "Do not build a shell command string from request data; call the operating system "
        "through an argument array, or do not call it at all.",
        (
            "Replace the shell call with a library or language API that performs the same work "
            "without a shell (file operations, image handling, archive handling).",
            "Where a subprocess is genuinely required, pass the program and its arguments as a "
            "list with shell interpretation disabled, so metacharacters are never parsed.",
            "Allow only a fixed set of programs, and validate each argument against a strict "
            "pattern (an enumerated value, or a bounded character class) before it is used.",
            "Run the subprocess under a low-privilege account with a minimal environment and a "
            "timeout.",
            "Log the exact argument vector that was executed, so an injection attempt is "
            "visible after the fact.",
        ),
        "Submit `; id`, `| id`, `$(id)` and a newline-separated payload in the affected "
        "parameter; each must appear verbatim as a single argument in the process log and "
        "produce no additional process.",
        "4 to 12 hours per call site, more when a shell pipeline has to be reimplemented.",
        "Do not escape or quote the metacharacters and keep the shell. Quoting rules differ "
        "between shells and platforms, and one missed context restores the injection.",
        "OWASP Cheat Sheet Series, OS Command Injection Defense.",
    ),
    79: _r(
        79,
        "Cross-site scripting",
        "Encode on output for the context the value lands in, and let the template engine do it "
        "everywhere rather than trusting each developer to remember.",
        (
            "Turn on the template engine's contextual auto-escaping and remove every construct "
            "that bypasses it (raw, safe, unescaped, `|n` filters and their equivalents).",
            "Encode for the exact sink: HTML body, attribute, JavaScript string, URL parameter "
            "and CSS each need different encoding, and HTML-encoding into a script block is not "
            "a fix.",
            "Where user-authored HTML must be rendered, run it through a maintained allowlist "
            "sanitiser (DOMPurify or an equivalent) on the server, not a regular expression.",
            "Remove dangerous client sinks: assign with `textContent` rather than `innerHTML`, "
            "and never pass request data to `eval`, `setTimeout` with a string, or "
            "`document.write`.",
            "Add a Content-Security-Policy without `unsafe-inline` as defence in depth, and set "
            "`X-Content-Type-Options: nosniff`.",
        ),
        "Re-send the scanner's payload and read the raw response: the payload must appear "
        "encoded as text, the rendered page must show it as literal characters, and the browser "
        "console must report no script execution.",
        "2 to 6 hours for a single reflected sink; longer where stored content must be "
        "re-sanitised on the way out.",
        "Do not blacklist `<script>` or strip characters. Escaping is context-dependent and "
        "denylists have been bypassed with every payload list published since 2005.",
        "OWASP Cheat Sheet Series, Cross Site Scripting Prevention; DOM based XSS Prevention.",
    ),
    89: _r(
        89,
        "SQL injection",
        "Send the query and the data separately: parameterise every statement so the database "
        "never parses user input as SQL.",
        (
            "Rewrite the affected statement as a prepared statement with bound parameters; no "
            "concatenation, no string interpolation, no format strings.",
            "For the parts that cannot be parameterised (table names, column names, sort "
            "direction, LIMIT), map the input through a server-side allowlist of permitted "
            "values.",
            "Use the data-access library's safe API rather than its raw-SQL escape hatch, and "
            "grep the codebase for the escape hatch to find the rest.",
            "Give the application's database account only the rights it needs: no DDL, no access "
            "to tables it does not use, and a separate read-only account where reads dominate.",
            "Add a regression test that asserts the injected payload is stored or matched as a "
            "literal value.",
        ),
        "Re-run the scanner payload and check the database's statement log: the statement must "
        "show a bound placeholder with the payload as a parameter value, and time-based payloads "
        "must produce no delay.",
        "2 to 8 hours per affected query, plus a codebase sweep for the same pattern.",
        "Do not escape quotes or filter keywords. Escaping breaks on numeric contexts, "
        "multi-byte character sets and second-order injection, and a keyword blacklist has never "
        "held.",
        "OWASP Cheat Sheet Series, SQL Injection Prevention; Query Parameterization.",
    ),
    94: _r(
        94,
        "Code injection",
        "Remove the dynamic evaluation. If the application interprets request data as code, no "
        "amount of filtering makes that safe.",
        (
            "Find the evaluation sink (`eval`, `exec`, `Function`, dynamic `require`/`import`, a "
            "template compiled from request data) and delete it.",
            "Express the requirement as data instead of code: a configuration value, a lookup "
            "table, or a small expression grammar you parse yourself with a fixed operator set.",
            "Where a scripting feature is a product requirement, run it in an interpreter with "
            "no host bindings, no file or network access, a memory cap and a timeout.",
            "Never compile a template whose body comes from a request; pass request data as "
            "template variables into a template that is part of the source tree.",
            "Add a lint or CI rule that fails the build when an evaluation sink reappears.",
        ),
        "Search the code for evaluation sinks and show none is reachable from request data; then "
        "submit a payload that would execute (a sleep, an arithmetic expression) and confirm it "
        "is stored or rendered as a literal string.",
        "1 to 3 days, because this is usually a design change rather than a patch.",
        "Do not filter language keywords or characters. Interpreters have many equivalent "
        "spellings for the same operation, and the filter will be bypassed rather than the "
        "design fixed.",
        "OWASP Cheat Sheet Series, Injection Prevention; OWASP Top 10 A03 Injection.",
    ),
    200: _r(
        200,
        "Exposure of sensitive information",
        "Decide what each response is allowed to contain and serialise exactly that, rather than "
        "returning whatever the internal object happens to hold.",
        (
            "Enumerate what the affected response exposes today, including headers, and mark "
            "each field as required or not.",
            "Serialise responses from an explicit output schema per role, so adding a column to "
            "a table does not silently add it to the API.",
            "Remove product and version banners, internal host names, absolute file paths, "
            "stack frames and debug fields from responses and headers.",
            "Set `Cache-Control: no-store` on responses carrying personal or authenticated data, "
            "and confirm no intermediary is caching them.",
            "Re-check the same endpoint as an anonymous user, a normal user and an administrator; "
            "the field sets must differ.",
        ),
        "Diff the response bodies and headers for anonymous, user and administrator requests "
        "against the documented contract; no field outside the contract may appear in any of "
        "them.",
        "2 to 6 hours per endpoint; longer where a shared serialiser is used everywhere.",
        "Do not rename or obfuscate the field. It is still returned, still logged by every proxy "
        "in the path, and still readable by whoever asked for it.",
        "OWASP Testing Guide, Information Gathering; OWASP Cheat Sheet Series, Error Handling "
        "and Secure Headers Project.",
    ),
    209: _r(
        209,
        "Error message containing sensitive information",
        "Return one generic error to the client and keep the detail in the server log, keyed by "
        "a correlation identifier.",
        (
            "Install a catch-all error handler so no unhandled exception ever renders a "
            "framework debug page.",
            "Return a fixed error body with an HTTP status and a correlation id; put the "
            "exception type, stack trace, query and parameters in the server log only.",
            "Turn off debug, verbose error and development modes in the deployed configuration, "
            "including for staging environments that share an origin or a data set.",
            "Check the error paths that bypass the handler: template rendering errors, database "
            "driver errors, reverse-proxy error pages and static-file handlers.",
            "Make sure the log itself redacts credentials and tokens before writing them.",
        ),
        "Force a 500 (a malformed parameter, a database timeout) and read the response body: it "
        "must contain the generic message and a correlation id, no stack frame, no SQL, no file "
        "path. Then find that id in the server log.",
        "3 to 8 hours, most of it finding the error paths that bypass the handler.",
        "Do not leave verbose errors on in staging because it is convenient. Staging usually "
        "shares code, and frequently shares data, with production.",
        "OWASP Cheat Sheet Series, Error Handling; OWASP Testing Guide, Testing for Error "
        "Handling.",
    ),
    269: _r(
        269,
        "Improper privilege management",
        "Write down which role may perform which operation, enforce it server-side on every "
        "request, and make it impossible for a subject to grant itself a role.",
        (
            "Define the role-to-operation matrix explicitly, and make the default answer deny.",
            "Enforce the decision on the server for every request, from the session subject, "
            "never from a role or identifier supplied by the client.",
            "Separate administrative operations from user operations, and route privilege "
            "changes through a dedicated flow that a normal user cannot reach.",
            "Strip role, permission, owner and tenant fields from request payloads before "
            "binding them to a model, so mass assignment cannot escalate.",
            "Audit every privilege change with actor, subject, old value and new value.",
        ),
        "Run the escalation attempts directly: a user calling an administrator operation, a user "
        "editing their own role field, and a user replaying an administrator's request. All "
        "three must be denied and must appear in the audit log.",
        "1 to 3 days where a role model has to be introduced; hours if one already exists.",
        "Do not hide the administrative controls in the user interface. The endpoint is still "
        "there, and the finding is about the endpoint.",
        "OWASP Cheat Sheet Series, Authorization; OWASP Top 10 A01 Broken Access Control.",
    ),
    287: _r(
        287,
        "Improper authentication",
        "Verify the credential or token properly on the server, with a vetted library, before "
        "anything downstream trusts the identity.",
        (
            "Use the platform's authentication framework or an established identity provider "
            "rather than a hand-written check.",
            "Verify tokens completely: signature, algorithm against an expected value, issuer, "
            "audience, expiry and not-before. Reject unsigned and `none`-algorithm tokens.",
            "Compare secrets in constant time, hash passwords with an accepted "
            "memory-hard function, and rate-limit and lock out on repeated failure.",
            "Issue a new session identifier on login and on privilege change, and bind the "
            "session to a bounded lifetime and an idle timeout.",
            "Require a second factor for administrative and high-value operations.",
        ),
        "Tamper with a token's payload, drop its signature, and replay an expired one; each must "
        "return 401. Confirm the session identifier changes at login and that the old one stops "
        "working.",
        "1 to 5 days, depending on whether the authentication is replaced or repaired.",
        "Do not enforce the check in the client. A client-side gate is a user-interface "
        "convenience and carries no security weight.",
        "OWASP Cheat Sheet Series, Authentication; Session Management; JSON Web Token for Java.",
    ),
    306: _r(
        306,
        "Missing authentication for a critical function",
        "Make authentication the default at a single chokepoint, and prove every route is "
        "covered rather than assuming it.",
        (
            "Apply authentication in middleware or the router so a new route is protected unless "
            "it is explicitly and deliberately made public.",
            "Enumerate every route the application serves and record, per route, what it "
            "requires; treat an unlisted route as a defect.",
            "Cover the surfaces that are not the user interface: internal APIs, administrative "
            "consoles, health and metrics endpoints, management ports, debug handlers and "
            "queue consumers.",
            "Remove or bind to localhost any diagnostic interface that does not need to be "
            "reachable.",
            "Add a test that walks the route inventory anonymously and asserts 401 or 403 for "
            "everything not on the public list.",
        ),
        "Run the route-inventory test as an anonymous client: every non-public route returns 401 "
        "or 403, and the public list is short enough to read in one sitting.",
        "4 hours to 2 days, mostly spent discovering the routes nobody remembered.",
        "Do not rely on the endpoint being unlinked or having an unguessable name. Directory "
        "brute forcing, JavaScript bundles and archived responses all find it.",
        "OWASP Cheat Sheet Series, Authorization; OWASP Top 10 A01 Broken Access Control.",
    ),
    319: _r(
        319,
        "Cleartext transmission of sensitive information",
        "Carry everything over TLS, including the hops inside your own network, and make the "
        "plaintext option unavailable rather than merely discouraged.",
        (
            "Serve the application over TLS 1.2 or later with modern cipher suites and a valid "
            "certificate chain.",
            "Redirect plain HTTP to HTTPS and send `Strict-Transport-Security` with a long "
            "max-age; add `includeSubDomains` once every subdomain is ready.",
            "Encrypt service-to-service and database connections too, not only the browser hop.",
            "Set `Secure` on every cookie and remove mixed content so no sub-resource loads over "
            "plaintext.",
            "Disable or firewall the remaining plaintext listeners so the fallback cannot be "
            "negotiated.",
        ),
        "Scan the host: no plaintext listener answers with application content, the TLS "
        "configuration reports no deprecated protocol or cipher, `Strict-Transport-Security` is "
        "present, and the browser reports no mixed content.",
        "4 hours to 2 days, depending on certificate management and internal hops.",
        "Do not encrypt the payload at the application layer and leave the channel in plaintext. "
        "Headers, cookies, URLs and traffic patterns are still exposed.",
        "OWASP Cheat Sheet Series, Transport Layer Security; HTTP Strict Transport Security.",
    ),
    352: _r(
        352,
        "Cross-site request forgery",
        "Require a value on state-changing requests that a cross-origin page cannot read or "
        "guess, and verify it on the server against the session.",
        (
            "Enable the framework's synchroniser-token defence, with a per-session token "
            "verified server-side on every state-changing request.",
            "Reject state-changing requests that arrive by GET; use POST, PUT, PATCH or DELETE, "
            "and make read methods free of side effects.",
            "Set `SameSite=Lax` or `Strict` on the session cookie as defence in depth, and "
            "`Secure` with it.",
            "Check `Origin` (falling back to `Referer`) on state-changing requests and reject "
            "mismatches.",
            "Require re-authentication or a second factor for the highest-value operations, such "
            "as changing the password, the email address or payment details.",
        ),
        "Replay a state-changing request from a different origin with the session cookie present "
        "but no token: it must be rejected. Then replay it with another user's token: also "
        "rejected.",
        "2 to 8 hours where the framework supports it; longer for a single-page application that "
        "has to carry the token.",
        "Do not rely on a custom header or the `Referer` alone, and do not use a token that is "
        "not bound to the session. A static token is a second cookie.",
        "OWASP Cheat Sheet Series, Cross-Site Request Forgery Prevention.",
    ),
    434: _r(
        434,
        "Unrestricted file upload",
        "Decide what the application accepts, verify it by inspecting the file, and store it "
        "somewhere that cannot execute anything.",
        (
            "Allow an explicit list of extensions and content types, and verify the file's "
            "actual type by inspecting its contents rather than trusting the declared type.",
            "Store uploads outside the web root under a server-generated name, keeping the "
            "original name only as metadata.",
            "Serve them back through a handler that sets a fixed `Content-Type`, "
            "`Content-Disposition: attachment` and `X-Content-Type-Options: nosniff`.",
            "Disable script execution and interpreter handlers in the storage location, or use "
            "object storage that cannot execute at all.",
            "Enforce a size limit, a per-user rate limit and, where the file is shared with "
            "others, a malware scan before it is made available.",
        ),
        "Upload a file with a script body and an image header, and one with a double extension; "
        "neither may be executable when requested directly, and the response must arrive with "
        "the fixed content type as an attachment.",
        "1 to 3 days, because storage and serving usually both have to change.",
        "Do not trust the `Content-Type` header or check only the extension. Both are client "
        "supplied, and content sniffing will disagree with them.",
        "OWASP Cheat Sheet Series, File Upload; OWASP Testing Guide, Testing Upload of "
        "Unexpected File Types.",
    ),
    502: _r(
        502,
        "Deserialization of untrusted data",
        "Do not hand attacker-controlled bytes to a deserialiser that can construct arbitrary "
        "objects; move to a data-only format with a schema.",
        (
            "Replace native serialisation with a data-only format (JSON) parsed into an explicit "
            "schema or typed structure, with unknown fields rejected.",
            "Remove every call that deserialises request data with a native mechanism: pickle, "
            "Java `ObjectInputStream`, PHP `unserialize`, .NET `BinaryFormatter`, a full YAML "
            "loader.",
            "Where a native format cannot be removed, restrict it to an allowlist of permitted "
            "classes enforced before construction, not after.",
            "Sign serialised blobs that must round-trip through the client and verify the "
            "signature before parsing anything.",
            "Keep the libraries that supply gadget chains patched, and remove those the "
            "application does not use.",
        ),
        "Send a known gadget payload for the platform: it must be rejected at the parsing "
        "boundary with no class constructed and no side effect, and the rejection must be "
        "logged.",
        "1 to 4 days; replacing a serialisation format usually touches the client too.",
        "Do not blacklist known gadget classes. New chains are published against the same "
        "libraries regularly, and the blacklist is always behind them.",
        "OWASP Cheat Sheet Series, Deserialization.",
    ),
    548: _r(
        548,
        "Directory listing exposed",
        "Turn off directory browsing at the web server and stop keeping non-public files inside "
        "the document root.",
        (
            "Disable automatic index generation in the web server or framework configuration "
            "(for example `autoindex off`, `Options -Indexes`) for the whole site, not one path.",
            "Move backups, archives, configuration, source maps and version-control directories "
            "out of the document root entirely.",
            "Serve static content from an explicit root that contains only what is meant to be "
            "public.",
            "Return the same 404 for a directory and a missing file so the listing is not "
            "replaced by an enumeration oracle.",
        ),
        "Request the affected directory and several sibling directories: each must return 403 or "
        "404 with no file names in the body, including for paths that previously listed.",
        "1 to 3 hours.",
        "Do not drop an empty `index.html` into each directory. The files are still served if "
        "their names are known or guessed, and the next new directory has no index file.",
        "OWASP Testing Guide, Review Webserver Metafiles and Directory Browsing; web server "
        "hardening guidance.",
    ),
    598: _r(
        598,
        "Sensitive information in a GET query string",
        "Move the sensitive value out of the URL: URLs are logged, cached, referred and stored "
        "in history at every hop.",
        (
            "Send the value in the request body of a POST or PUT, or in a header where the "
            "protocol requires one.",
            "Treat anything already exposed in a URL (tokens, reset links, session identifiers, "
            "personal data) as compromised and expire or rotate it.",
            "Purge the parameter from access logs, proxy logs, analytics and error trackers, and "
            "add a redaction rule so it stops being written.",
            "Set `Referrer-Policy: strict-origin-when-cross-origin` or stricter so URLs do not "
            "leak to third parties.",
            "Set `Cache-Control: no-store` on the affected responses.",
        ),
        "Exercise the flow with a proxy: the sensitive value appears in no request line, no "
        "access-log entry and no `Referer` header, and the browser history shows only the "
        "parameterless URL.",
        "2 to 8 hours, plus log retention clean-up.",
        "Do not encrypt or encode the value and leave it in the URL. It is still logged, still "
        "shared in a copied link and usually still replayable.",
        "OWASP Cheat Sheet Series, Session Management; OWASP Testing Guide, Testing for Exposed "
        "Session Variables.",
    ),
    601: _r(
        601,
        "Open redirect",
        "Do not let the request choose an arbitrary destination; map a short identifier to a "
        "destination the server already knows.",
        (
            "Where possible, remove the destination parameter and redirect to a fixed location "
            "chosen by the server.",
            "Where a destination is genuinely needed, accept an identifier or token and look the "
            "URL up server-side.",
            "If a URL must be accepted, parse it and allowlist scheme and host; accept relative "
            "paths only when they begin with exactly one `/` and contain no backslash.",
            "Validate after decoding, and reject `//host`, `/\\host`, `https:/\\host`, embedded "
            "credentials and non-HTTP schemes such as `javascript:` and `data:`.",
            "Show an interstitial page for any destination outside the application.",
        ),
        "Try `//evil.example`, `https://evil.example`, `/\\evil.example`, `%2f%2fevil.example` "
        "and a `javascript:` URL: every one must be refused, and the response must not carry a "
        "`Location` header pointing off-site.",
        "2 to 6 hours per redirect handler.",
        "Do not check whether your domain appears in the string. `https://evil.example/?x=your"
        ".example` and `https://your.example.evil.example` both pass a substring check.",
        "OWASP Cheat Sheet Series, Unvalidated Redirects and Forwards.",
    ),
    611: _r(
        611,
        "XML external entity processing",
        "Turn off document type definitions in every XML parser the application uses; that one "
        "setting removes the class.",
        (
            "Disable DTD processing entirely in each parser, using that library's documented "
            "flag, and make it the default in a shared factory rather than per call site.",
            "Where a DTD is genuinely required, disable external general entities, external "
            "parameter entities and entity expansion, and set an expansion limit.",
            "Cover the parsers that are not obviously XML: SOAP, SVG, DOCX and XLSX handling, "
            "XML-RPC, SAML and configuration loaders.",
            "Prefer a format without entities (JSON) for new interfaces.",
            "Keep the XML libraries patched, because defaults have changed between versions in "
            "both directions.",
        ),
        "Submit an entity referencing a local file and a second referencing an external host: "
        "the first must fail to parse with no file content in the response, and the second must "
        "produce no outbound connection in the egress log.",
        "2 to 8 hours per parser factory.",
        "Do not filter `<!ENTITY` or `<!DOCTYPE` from the body. Encoding, parameter entities and "
        "nested schema references defeat the filter; the parser flag does not.",
        "OWASP Cheat Sheet Series, XML External Entity Prevention.",
    ),
    614: _r(
        614,
        "Sensitive cookie without the Secure attribute",
        "Mark every session and authentication cookie `Secure` so it is never sent over "
        "plaintext, and set it where the cookie is configured rather than at each call site.",
        (
            "Set `Secure` on the session, authentication and CSRF cookies in the framework's "
            "cookie configuration.",
            "Serve the site over HTTPS only, redirect plain HTTP and send "
            "`Strict-Transport-Security`.",
            "Audit every `Set-Cookie` the application emits, including from reverse proxies, "
            "load balancers and third-party components.",
            "Pair it with `HttpOnly` and an appropriate `SameSite` value in the same change.",
        ),
        "Request the site over plain HTTP with a session established: no session cookie is sent, "
        "and every `Set-Cookie` in the HTTPS responses carries `Secure`.",
        "1 to 2 hours.",
        "Do not rely on the HTTP-to-HTTPS redirect alone. The redirect response itself, and any "
        "request that reaches the plaintext listener, can still carry the cookie.",
        "OWASP Cheat Sheet Series, Session Management; Secure Headers Project.",
    ),
    639: _r(
        639,
        "Authorization bypass through a user-controlled key",
        "Authorise the object, not the route: every access must be checked against the session "
        "subject, using an identifier the server trusts.",
        (
            "Check ownership or membership on the server for every read, write and delete, using "
            "the subject from the session rather than any identifier in the request.",
            "Push the constraint into the data layer, so the query itself filters by owner or "
            "tenant and cannot return another subject's row.",
            "Centralise the decision in one authorisation component and call it from every "
            "entry point, including bulk endpoints, exports and background jobs.",
            "Return the same response for 'not yours' and 'does not exist' so the endpoint is "
            "not an enumeration oracle.",
            "Add an automated test with two accounts that asserts each cannot reach the other's "
            "objects.",
        ),
        "With two accounts, request every affected object identifier as the wrong user: each "
        "must return the same denial, and the database query log must show the ownership filter "
        "in the statement.",
        "1 to 3 days, because the check usually has to be added in many places at once.",
        "Do not replace sequential identifiers with UUIDs and call it fixed. That hides "
        "enumeration and leaves the authorisation defect exactly where it was.",
        "OWASP Cheat Sheet Series, Insecure Direct Object Reference Prevention; Authorization.",
    ),
    650: _r(
        650,
        "Trusting HTTP permission methods on the server side",
        "Authorise the operation, not the verb, and stop serving the methods the application "
        "does not implement.",
        (
            "List the methods each route actually implements and return 405 for everything else, "
            "at the application and at the reverse proxy.",
            "Disable WebDAV verbs, `TRACE` and `TRACK` at the web server unless they are a "
            "product requirement.",
            "Make the authorisation decision on the operation being performed, so a `PUT` and a "
            "`POST` that do the same thing are checked the same way.",
            "Reject `X-HTTP-Method-Override`, `X-Method-Override` and `_method` unless a "
            "documented client needs them; if one does, authorise the overridden method.",
            "Keep read methods free of side effects so a permitted `GET` cannot change state.",
        ),
        "Send each unimplemented method (`PUT`, `DELETE`, `TRACE`, `PROPFIND`) to the affected "
        "path and to a static path: all must return 405 or 501, and `OPTIONS` must advertise "
        "only what is implemented.",
        "2 to 6 hours.",
        "Do not block the methods only at the content delivery network or load balancer. Any "
        "request that reaches the origin directly bypasses that control.",
        "OWASP Testing Guide, Test HTTP Methods; web server hardening guidance.",
    ),
    693: _r(
        693,
        "Protection mechanism failure",
        "A control the application is supposed to rely on is missing or can be bypassed; apply "
        "it at one chokepoint so no route can be served without it.",
        (
            "Identify precisely which control is missing or bypassable, and on which responses.",
            "Apply it in one place - middleware, a base handler or the reverse proxy - so every "
            "response is covered, including errors, redirects and static files.",
            "For missing response headers, set `Content-Security-Policy` (with `frame-ancestors` "
            "rather than the obsolete `X-Frame-Options` alone), `X-Content-Type-Options: "
            "nosniff` and a `Referrer-Policy`.",
            "Remove the per-route ability to turn the control off, or require an explicit, "
            "reviewed annotation to do so.",
            "Add a regression test that asserts the control is present on a representative "
            "response of each kind.",
        ),
        "Fetch a normal page, an error page, a redirect and a static asset: the control is "
        "present on all four, and the bypass reported by the scanner no longer works.",
        "2 to 8 hours once the chokepoint is identified.",
        "Do not add the header on the home page and declare it done. Scanners and attackers both "
        "look at the routes that were forgotten.",
        "OWASP Secure Headers Project; OWASP Cheat Sheet Series, Content Security Policy.",
    ),
    798: _r(
        798,
        "Hard-coded credentials",
        "Treat the credential as already compromised: rotate first, then remove it from the code "
        "and from the history, then stop it recurring.",
        (
            "Rotate or revoke the credential now, before the code change, and check the logs for "
            "use of the old value.",
            "Move the secret into a secrets manager or an injected environment variable, read at "
            "start-up, never written to a log.",
            "Remove it from the repository history as well as the working tree, and from built "
            "artefacts such as container image layers and source maps.",
            "Scope the replacement credential to the least privilege the application needs, and "
            "give it an expiry.",
            "Add a secret scanner as a pre-commit hook and as a blocking CI check.",
        ),
        "Run a secret scanner over the full history and the built image: the value is absent. "
        "Then attempt to authenticate with the old credential and confirm it fails.",
        "4 to 8 hours, plus whatever the rotation costs in coordination.",
        "Do not simply delete the line in the latest commit. The value stays in the history, in "
        "forks, in build caches and in anyone's local clone.",
        "OWASP Cheat Sheet Series, Secrets Management; OWASP Top 10 A07 Identification and "
        "Authentication Failures.",
    ),
    863: _r(
        863,
        "Incorrect authorization",
        "The check exists but reaches the wrong answer; move the decision into one component and "
        "test it as a matrix rather than a route at a time.",
        (
            "Centralise authorisation in a single policy component that takes subject, action and "
            "resource, and call it from every entry point.",
            "Make the decision from server-side identity and attributes; never from a role, "
            "tenant or capability supplied in the request.",
            "Fail closed: an error while evaluating the policy denies the request.",
            "Cover every surface, including GraphQL resolvers, batch and bulk endpoints, exports, "
            "webhooks and background jobs.",
            "Write negative tests for the full role-by-resource matrix, not only for the route "
            "that was reported.",
        ),
        "Run the role-by-resource matrix: every cell that should be denied returns 403, every "
        "cell that should be allowed returns 200, and the policy component logs the decision "
        "with its inputs.",
        "1 to 4 days, because the matrix usually reveals more than the reported case.",
        "Do not patch only the endpoint in the report. Incorrect authorisation is nearly always a "
        "pattern, and the scanner found one instance of it.",
        "OWASP Cheat Sheet Series, Authorization; OWASP Top 10 A01 Broken Access Control.",
    ),
    918: _r(
        918,
        "Server-side request forgery",
        "Decide server-side where outbound requests may go, and validate the address that is "
        "actually connected to, not the string that was submitted.",
        (
            "Allowlist destination scheme, host and port; reject everything else rather than "
            "blocking known-bad values.",
            "Resolve the host, validate the resolved addresses against loopback, link-local, "
            "private and multicast ranges, and connect to the validated address so DNS cannot "
            "change between check and use.",
            "Disable redirect following, or re-run the whole validation on every hop.",
            "Block the cloud metadata address at the network level and remove any credential the "
            "instance can obtain without authentication.",
            "Send outbound fetches through a dedicated egress proxy with its own allowlist, in a "
            "network segment that cannot reach internal services.",
        ),
        "Request the cloud metadata address, a loopback address, a private range address and a "
        "host that resolves to a private address on its second lookup: all four must fail, and "
        "the egress log must show no connection attempt to an internal address.",
        "1 to 3 days, because the network control is usually part of the fix.",
        "Do not block the strings `localhost` and `127.0.0.1`. Decimal, octal, IPv6-mapped and "
        "rebinding forms all reach the same host without matching the filter.",
        "OWASP Cheat Sheet Series, Server-Side Request Forgery Prevention.",
    ),
    942: _r(
        942,
        "Permissive cross-domain policy",
        "Name the origins you trust; never reflect the requesting origin, and never combine a "
        "wildcard with credentials.",
        (
            "Replace a wildcard or reflected `Access-Control-Allow-Origin` with an explicit "
            "allowlist checked against the `Origin` header.",
            "Do not send `Access-Control-Allow-Credentials: true` unless the origin is on that "
            "allowlist, and never with a wildcard origin.",
            "Restrict the allowed methods and headers to what the documented clients use, and "
            "keep the preflight cache short.",
            "Delete legacy `crossdomain.xml` and `clientaccesspolicy.xml`, or restrict them to "
            "named domains.",
            "Add `Vary: Origin` so a permissive response for one origin is not cached and served "
            "to another.",
        ),
        "Send requests with an allowlisted origin, an unknown origin and a null origin: only the "
        "first receives an `Access-Control-Allow-Origin`, and no response pairs a wildcard with "
        "`Access-Control-Allow-Credentials`.",
        "2 to 6 hours.",
        "Do not reflect the `Origin` header back and call it an allowlist. Reflection accepts "
        "every origin, which is the wildcard with credentials enabled.",
        "OWASP Cheat Sheet Series, HTML5 Security and Cross-Origin Resource Sharing guidance.",
    ),
    1004: _r(
        1004,
        "Sensitive cookie without the HttpOnly flag",
        "Mark session and authentication cookies `HttpOnly` so a scripting flaw cannot read "
        "them, and keep the values scripts need in a separate, non-sensitive cookie.",
        (
            "Set `HttpOnly` on session, authentication and refresh cookies in the framework's "
            "cookie configuration.",
            "Where client script needs a value (a display name, a CSRF token pattern that "
            "requires it), put that value in its own cookie and leave the session cookie "
            "unreadable.",
            "Audit every `Set-Cookie` emitted by the application, its proxies and its third-party "
            "components.",
            "Set `Secure` and an appropriate `SameSite` in the same change.",
        ),
        "With a session established, read `document.cookie` in the browser console: the session "
        "cookie is absent, and the request still carries it.",
        "1 to 2 hours.",
        "Do not treat an input filter as the remedy. `HttpOnly` is what limits the damage when "
        "the filter is bypassed, which is the case this finding is about.",
        "OWASP Cheat Sheet Series, Session Management; Secure Headers Project.",
    ),
    1104: _r(
        1104,
        "Use of an unmaintained third-party component",
        "An unmaintained dependency has no upgrade path when the next issue lands; plan the "
        "replacement rather than pinning and waiting.",
        (
            "Produce a software bill of materials and confirm which components have no recent "
            "release and no security contact.",
            "Give each unmaintained component an owner and a decision: replace, fork and "
            "maintain, or vendor the small part actually used.",
            "Remove dependencies the application does not use, which is usually a meaningful "
            "share of them.",
            "Pin versions and subscribe the remaining components to automated advisory "
            "monitoring so a new issue arrives as a ticket.",
            "Record the decision and its date, so the next reviewer inherits a position rather "
            "than a surprise.",
        ),
        "Diff the bill of materials before and after: the component is replaced, removed, or "
        "recorded with an owner and a dated decision, and the dependency scanner is clean for it.",
        "Hours to remove an unused dependency; days to weeks to replace one that is load-bearing.",
        "Do not pin the old version and suppress the alert. Suppression removes the warning, not "
        "the dependency, and the next reviewer cannot tell the difference.",
        "OWASP Top 10 A06 Vulnerable and Outdated Components; OWASP Dependency-Check "
        "documentation.",
    ),
    1275: _r(
        1275,
        "Sensitive cookie with an improper SameSite attribute",
        "Set `SameSite` explicitly on session cookies, and keep the anti-forgery token: the "
        "attribute is defence in depth, not a replacement for it.",
        (
            "Set `SameSite=Lax` as the baseline for session and authentication cookies, and "
            "`Strict` where no cross-site entry flow needs them.",
            "Use `SameSite=None` only with a documented cross-site requirement, and always "
            "together with `Secure`.",
            "Keep synchroniser tokens on state-changing requests regardless of the attribute.",
            "Audit every `Set-Cookie` the application and its proxies emit, and set the value "
            "in one configuration rather than per handler.",
        ),
        "Trigger a cross-site state-changing request: the session cookie is not attached. Then "
        "confirm every `Set-Cookie` in a normal session carries an explicit `SameSite` value.",
        "1 to 3 hours, plus testing any cross-site sign-in flow.",
        "Do not set `SameSite=None` without `Secure` (browsers reject it), and do not remove the "
        "CSRF token because the attribute is set. Browser behaviour and defaults differ.",
        "OWASP Cheat Sheet Series, Cross-Site Request Forgery Prevention; Session Management.",
    ),
}


GENERIC_REMEDIATION = Remediation(
    cwe_id=None,
    weakness="No weakness-specific playbook",
    summary=(
        "This is generic guidance. The knowledge base holds no playbook for this weakness, so "
        "nothing below is specific to it and none of it should be treated as a verified fix."
    ),
    steps=(
        "Read the weakness description at its authoritative source (the CWE entry, and the "
        "vendor advisory for any CVE listed on the finding) before changing anything.",
        "Reproduce the finding manually and record the exact request and response, so the fix "
        "can be shown to change that behaviour.",
        "Identify the trust boundary the finding crosses, and fix the defect on the server side "
        "of it; a client-side change is not a fix.",
        "Apply the change at the place that covers every instance (a shared handler, middleware "
        "or configuration), then search for the same pattern elsewhere in the codebase.",
        "Add a regression test that fails against the old behaviour, so the finding cannot "
        "return silently.",
    ),
    verification=(
        "Replay the recorded request: the recorded exploitable response must no longer occur, "
        "the regression test must fail on the old code and pass on the new, and a rescan must no "
        "longer report the finding."
    ),
    typical_effort="Not estimated: no playbook is held for this weakness.",
    do_not=(
        "Do not close the finding on the strength of a rescan alone. A scanner that stops "
        "reporting a finding has stopped detecting it, which is not the same as the defect "
        "having been removed."
    ),
    source="Generic: no source-backed playbook is held for this weakness.",
    generic=True,
)


def known_cwes() -> tuple[int, ...]:
    """The CWE identifiers the knowledge base covers, ascending."""
    return tuple(sorted(REMEDIATIONS))


def weakness_name(cwe_id: int | None) -> str:
    """Human-readable weakness name for a CWE, or an empty string when unknown."""
    if cwe_id is None:
        return ""
    entry = REMEDIATIONS.get(int(cwe_id))
    return entry.weakness if entry is not None else ""


def guidance_for(cwe_id: int | None, finding: Any = None) -> Remediation:
    """Remediation guidance for a CWE.

    ``finding`` is optional and is used only to add a factual context line drawn from the
    finding itself (endpoint, cluster size). It never changes the guidance text, so the same
    CWE always produces the same playbook.

    An unknown or missing CWE returns :data:`GENERIC_REMEDIATION`, whose ``generic`` flag is
    ``True`` and whose first line says so.
    """
    entry: Remediation | None = None
    if cwe_id is not None:
        try:
            entry = REMEDIATIONS.get(int(cwe_id))
        except (TypeError, ValueError):
            entry = None
    if entry is None:
        entry = GENERIC_REMEDIATION
        if cwe_id is not None:
            entry = entry.model_copy(
                update={
                    "cwe_id": int(cwe_id),
                    "summary": (
                        f"This is generic guidance. The knowledge base holds no playbook for "
                        f"CWE-{int(cwe_id)}, so nothing below is specific to it and none of it "
                        f"should be treated as a verified fix."
                    ),
                }
            )
    if finding is None:
        return entry
    return entry.model_copy(update={"context": _context_line(finding)})


def _context_line(finding: Any) -> str:
    """A factual one-liner about where this weakness was seen. Data only, no judgement."""
    method = str(getattr(finding, "endpoint_method", "") or "").strip()
    path = str(getattr(finding, "endpoint_path", "") or "").strip()
    where = f"{method} {path}".strip()
    cluster = getattr(finding, "cluster_size", 1) or 1
    parts: list[str] = []
    if where:
        parts.append(f"Reported at {where}.")
    try:
        cluster = int(cluster)
    except (TypeError, ValueError):
        cluster = 1
    if cluster > 1:
        parts.append(
            f"The scan grouped {cluster} alerts under this root cause, so the fix is charged "
            f"once and is expected to close all {cluster}."
        )
    return " ".join(parts)
