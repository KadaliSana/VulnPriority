"""Untrusted-content sandbox (DESIGN.md section 3.3, Architecture item 4).

Seven defences, applied in order, each of which assumes the previous one failed:

1. :mod:`~vulnprio.sandbox.normalize` - NFKC, invisible-character strip, homoglyph fold,
   HTML to visible text, whitespace collapse, blob elision, hard length cap.
2. :mod:`~vulnprio.sandbox.instruction_filter` - pattern detection and redaction.
3. :mod:`~vulnprio.sandbox.delimit` - per-call nonce envelopes and integrity checks.
4. :mod:`~vulnprio.sandbox.canary` - canary planting and leak detection.
5. :mod:`~vulnprio.sandbox.output_guard` - schema validation and numeric clamping.
6. :mod:`~vulnprio.sandbox.output_guard` - evidence-span and cross-reference verification.
7. :func:`~vulnprio.sandbox.output_guard.clamp_influence` - the per-tier influence budget.

:class:`~vulnprio.sandbox.pipeline.Sandbox` composes 1-3 behind the ``Sanitizer`` protocol
and :func:`~vulnprio.sandbox.pipeline.build_sandboxed_prompt` is the only supported way to
render untrusted text into a prompt.
"""

from __future__ import annotations

from vulnprio.sandbox.canary import (
    CANARY_PREFIX,
    canary_in_output,
    find_canary_leaks,
    make_canary,
    normalize_for_canary,
)
from vulnprio.sandbox.delimit import (
    closing_tag,
    envelope,
    envelope_intact,
    envelope_violations,
    make_nonce,
)
from vulnprio.sandbox.instruction_filter import (
    REDACTION_MARKER,
    CompiledPattern,
    InstructionFilter,
    default_patterns_path,
    load_patterns,
)
from vulnprio.sandbox.normalize import (
    HOMOGLYPHS,
    collapse_whitespace,
    elide_blobs,
    fold_homoglyphs,
    html_to_text,
    normalize_untrusted,
    strip_invisible,
)
from vulnprio.sandbox.output_guard import (
    OutputGuard,
    assert_within_budget,
    clamp_influence,
    numeric_bounds,
    span_matches,
)
from vulnprio.sandbox.pipeline import (
    UNTRUSTED_HANDLING_RULES,
    Sandbox,
    build_sandboxed_prompt,
    render_user_message,
    worst_verdict,
)

__all__ = [
    # normalize
    "HOMOGLYPHS",
    "normalize_untrusted",
    "fold_homoglyphs",
    "strip_invisible",
    "html_to_text",
    "collapse_whitespace",
    "elide_blobs",
    # instruction filter
    "REDACTION_MARKER",
    "CompiledPattern",
    "InstructionFilter",
    "load_patterns",
    "default_patterns_path",
    # delimit
    "make_nonce",
    "envelope",
    "closing_tag",
    "envelope_intact",
    "envelope_violations",
    # canary
    "CANARY_PREFIX",
    "make_canary",
    "normalize_for_canary",
    "canary_in_output",
    "find_canary_leaks",
    # output guard
    "OutputGuard",
    "clamp_influence",
    "assert_within_budget",
    "numeric_bounds",
    "span_matches",
    # pipeline
    "Sandbox",
    "build_sandboxed_prompt",
    "render_user_message",
    "worst_verdict",
    "UNTRUSTED_HANDLING_RULES",
]
