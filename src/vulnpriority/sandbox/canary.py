"""Stage 4 of the sandbox: a canary planted in the system text.

A canary answers one question no other control can: did the untrusted content actually
succeed in steering the model? Bounds checks and schema validation constrain the *shape* of
the answer, but a canary appearing in the output is direct proof that the model followed an
instruction from the page instead of the operator. That is why a leak is an exception
(``CanaryLeakError``) rather than a score adjustment.

Detection normalises before comparing, because an attacker who suspects a canary will ask
for it spaced out, hyphenated or in lower case.
"""

from __future__ import annotations

import re
import secrets
from random import Random

__all__ = [
    "CANARY_ALPHABET",
    "CANARY_PREFIX",
    "make_canary",
    "normalize_for_canary",
    "canary_in_output",
    "find_canary_leaks",
]

#: Upper-case letters and digits only: after normalisation the comparison is case-folded,
#: so a mixed-case alphabet would add no entropy that survives the comparison.
CANARY_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

#: A human-recognisable prefix so a leaked canary is obvious in a transcript.
CANARY_PREFIX = "VPCANARY"

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def make_canary(length: int = 24, rng: Random | None = None) -> str:
    """Return a canary of ``length`` characters, prefixed with ``VPCANARY``.

    Without ``rng`` the value comes from :mod:`secrets`; ``rng`` pins it for reproducible
    offline runs. ``length`` counts the whole token, so it must leave room for the prefix.
    """
    if length < len(CANARY_PREFIX) + 8:
        raise ValueError(f"canary length must be at least {len(CANARY_PREFIX) + 8}")
    n_random = length - len(CANARY_PREFIX)
    if rng is None:
        body = "".join(secrets.choice(CANARY_ALPHABET) for _ in range(n_random))
    else:
        body = "".join(rng.choice(CANARY_ALPHABET) for _ in range(n_random))
    return CANARY_PREFIX + body


def normalize_for_canary(text: str) -> str:
    """Lower-case alphanumerics only.

    ``VP CANARY-ab 12`` and ``vpcanaryab12`` normalise to the same string, so whitespace
    insertion, hyphenation and case changes cannot smuggle the canary past the check.
    """
    return _NON_ALNUM.sub("", text.lower())


def canary_in_output(text: str, canary: str) -> bool:
    """True when ``canary`` appears in ``text`` under the normalised comparison."""
    if not canary or not text:
        return False
    needle = normalize_for_canary(canary)
    if not needle:
        return False
    return needle in normalize_for_canary(text)


def find_canary_leaks(text: str, canaries: object) -> list[str]:
    """Every canary from ``canaries`` that appears in ``text``.

    Accepts a single canary string or any iterable of them, so a caller batching several
    prompts can check one reply against all of their canaries at once.
    """
    if isinstance(canaries, str):
        candidates: list[str] = [canaries]
    else:
        candidates = [str(item) for item in canaries or []]
    return [candidate for candidate in candidates if canary_in_output(text, candidate)]
