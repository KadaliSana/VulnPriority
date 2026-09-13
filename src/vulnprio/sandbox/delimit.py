"""Stage 3 of the sandbox: nonce-delimited envelopes around untrusted segments.

A fixed delimiter is a published escape sequence: the attacker writes the closing tag into
the page and the rest of their text is read as operator context. A per-call random nonce
removes that option, because the attacker cannot know the nonce at authoring time. The
nonce is also a tripwire: if it, or any envelope tag, appears in the model's reply then the
model has been reading the envelope machinery instead of the content, and the call is
treated as compromised.
"""

from __future__ import annotations

import re
import secrets
from random import Random

from vulnprio.core.enums import TrustTier

__all__ = [
    "NONCE_ALPHABET",
    "make_nonce",
    "envelope",
    "closing_tag",
    "envelope_intact",
    "envelope_violations",
]

#: Unambiguous alphanumerics: a nonce that can be mistyped is a nonce that false-alarms.
NONCE_ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"

_OPEN_TAG_RE = re.compile(r"<\s*untrusted\b[^>]{0,200}>", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"<\s*/?\s*untrusted\s*[:\s][^>]{0,200}>|<\s*/\s*untrusted\b[^>]{0,200}>", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def make_nonce(n: int = 16, rng: Random | None = None) -> str:
    """Return an ``n``-character nonce.

    Without ``rng`` the nonce comes from :mod:`secrets`, because unpredictability is the
    entire security property. ``rng`` exists so tests and reproducible runs can pin it;
    a seeded nonce is reproducible and therefore *not* secret, which is fine offline.
    """
    if n <= 0:
        raise ValueError("nonce length must be positive")
    if rng is None:
        return "".join(secrets.choice(NONCE_ALPHABET) for _ in range(n))
    return "".join(rng.choice(NONCE_ALPHABET) for _ in range(n))


def closing_tag(nonce: str) -> str:
    """The only sequence that legitimately ends an envelope."""
    return f"</untrusted:{nonce}>"


def envelope(sanitized: str, tier: TrustTier, nonce: str, segment_id: str) -> str:
    """Wrap one sanitized segment in its nonce-bearing envelope.

    The opening tag carries the segment id and trust tier so the model can be told, in
    operator text, how much weight each tier is allowed to carry.
    """
    tier_value = int(TrustTier(tier))
    return (
        f'<untrusted id="{segment_id}" tier="{tier_value}" nonce="{nonce}">\n'
        f"{sanitized}\n"
        f"{closing_tag(nonce)}"
    )


def _normalized(text: str) -> str:
    """Lowercase alphanumerics only, so ``AB cd`` and ``a-b-c-d`` compare equal."""
    return _NON_ALNUM.sub("", text.lower())


def envelope_violations(model_output: str, nonce: str) -> list[str]:
    """Names of every envelope-integrity violation found in ``model_output``.

    Three things count as a violation: emitting an opening tag (forging an envelope),
    emitting a closing tag (whether or not the nonce is right), and reproducing the nonce
    itself anywhere, which means the envelope framing leaked into the answer.
    """
    violations: list[str] = []
    if not model_output:
        return violations
    if _OPEN_TAG_RE.search(model_output):
        violations.append("forged_open_tag")
    if _CLOSE_TAG_RE.search(model_output) or "</untrusted" in model_output.lower():
        violations.append("forged_close_tag")
    if nonce and _normalized(nonce) and _normalized(nonce) in _normalized(model_output):
        violations.append("nonce_echoed")
    return violations


def envelope_intact(model_output: str, nonce: str) -> bool:
    """True when the reply shows no attempt to close, forge or echo the envelope."""
    return not envelope_violations(model_output, nonce)
