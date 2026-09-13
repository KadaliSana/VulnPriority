"""Stage 2 of the sandbox: find imperatives hiding in untrusted text and remove them.

The filter is deliberately a *pattern* detector rather than a model: a model asked to judge
whether text contains an injection is itself a target, and the whole point of the sandbox
is that no untrusted byte reaches a model with its meaning intact. Pattern matching is
auditable, deterministic and free, which is what a security control needs to be.

Detection is separated from redaction so the adversarial evaluator can measure detection
rate (including on benign controls, where the rate must be zero) without mutating text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from vulnprio.core.config import PROJECT_ROOT, SandboxConfig
from vulnprio.core.enums import InjectionCategory, InjectionVerdict, TrustTier
from vulnprio.core.errors import ConfigError
from vulnprio.core.models import InjectionSignal

__all__ = [
    "REDACTION_MARKER",
    "CompiledPattern",
    "InstructionFilter",
    "load_patterns",
    "default_patterns_path",
]

REDACTION_MARKER = "[REDACTED-INSTRUCTION]"

#: InjectionSignal.snippet is bounded at 200 characters by the frozen contract.
_MAX_SNIPPET = 200


@dataclass(frozen=True)
class CompiledPattern:
    """One entry of the pattern library, ready to run."""

    pattern_id: str
    category: InjectionCategory
    regex: re.Pattern[str]
    description: str = ""


def default_patterns_path() -> Path:
    """Path to the shipped pattern library, resolved against the repository root."""
    return PROJECT_ROOT / "configs" / "sandbox" / "instruction_patterns.yaml"


def load_patterns(path: str | Path | None = None) -> tuple[str, tuple[CompiledPattern, ...]]:
    """Load and compile the versioned pattern library.

    Returns ``(version, patterns)``. Every structural problem is a :class:`ConfigError`
    rather than a silent skip: a pattern library that quietly lost half its entries would
    make the adversarial evaluation report a defence the framework does not have.
    """
    resolved = Path(path) if path is not None else default_patterns_path()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.exists():
        raise ConfigError(f"instruction pattern library not found: {resolved}")

    try:
        document = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - malformed file
        raise ConfigError(f"instruction pattern library is not valid YAML: {resolved}") from exc
    if not isinstance(document, dict):
        raise ConfigError(f"instruction pattern library must be a mapping: {resolved}")

    version = str(document.get("version", "0"))
    raw_patterns = document.get("patterns")
    if not isinstance(raw_patterns, list) or not raw_patterns:
        raise ConfigError(f"instruction pattern library has no patterns: {resolved}")

    seen: set[str] = set()
    compiled: list[CompiledPattern] = []
    for index, entry in enumerate(raw_patterns):
        if not isinstance(entry, dict):
            raise ConfigError(f"pattern #{index} is not a mapping in {resolved}")
        pattern_id = str(entry.get("id", "")).strip()
        if not pattern_id:
            raise ConfigError(f"pattern #{index} has no id in {resolved}")
        if pattern_id in seen:
            raise ConfigError(f"duplicate pattern id {pattern_id!r} in {resolved}")
        seen.add(pattern_id)
        try:
            category = InjectionCategory(str(entry.get("category", "")))
        except ValueError as exc:
            raise ConfigError(
                f"pattern {pattern_id!r} has an unknown category {entry.get('category')!r}"
            ) from exc
        if category == InjectionCategory.BENIGN_CONTROL:
            raise ConfigError(f"pattern {pattern_id!r} may not use the benign_control category")
        expression = entry.get("regex")
        if not isinstance(expression, str) or not expression:
            raise ConfigError(f"pattern {pattern_id!r} has no regex")
        try:
            regex = re.compile(expression, re.IGNORECASE | re.UNICODE)
        except re.error as exc:
            raise ConfigError(f"pattern {pattern_id!r} does not compile: {exc}") from exc
        compiled.append(
            CompiledPattern(
                pattern_id=pattern_id,
                category=category,
                regex=regex,
                description=str(entry.get("description", "")),
            )
        )
    return version, tuple(compiled)


def _merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching match spans so one payload redacts to one marker."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


class InstructionFilter:
    """Pattern-based detection and redaction of instructions in untrusted text."""

    def __init__(
        self,
        patterns_file: str | Path | None = None,
        *,
        config: SandboxConfig | None = None,
    ) -> None:
        """Load the library named by ``patterns_file`` (or by ``config.patterns_file``)."""
        self.config = config or SandboxConfig()
        source = patterns_file if patterns_file is not None else self.config.patterns_file
        self.version, self.patterns = load_patterns(source)

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.patterns)

    @property
    def categories(self) -> set[InjectionCategory]:
        """Categories the loaded library can actually produce."""
        return {pattern.category for pattern in self.patterns}

    def pattern(self, pattern_id: str) -> CompiledPattern | None:
        for candidate in self.patterns:
            if candidate.pattern_id == pattern_id:
                return candidate
        return None

    # -- detection ---------------------------------------------------------

    def detect(self, text: str, tier: TrustTier = TrustTier.TARGET_CONTENT) -> list[InjectionSignal]:
        """Every pattern hit in ``text``, in document order.

        One signal per match, so a payload that stacks three techniques is three signals
        and crosses the ``injected`` threshold on its own.
        """
        if not text:
            return []
        found: list[tuple[int, InjectionSignal]] = []
        for pattern in self.patterns:
            for match in pattern.regex.finditer(text):
                snippet = match.group(0).strip()
                if not snippet:
                    continue
                found.append(
                    (
                        match.start(),
                        InjectionSignal(
                            pattern_id=pattern.pattern_id,
                            category=pattern.category,
                            snippet=snippet[:_MAX_SNIPPET],
                            tier=tier,
                        ),
                    )
                )
        found.sort(key=lambda item: (item[0], item[1].pattern_id))
        return [signal for _, signal in found]

    def redact(
        self, text: str, tier: TrustTier = TrustTier.TARGET_CONTENT
    ) -> tuple[str, list[InjectionSignal], list[str]]:
        """Replace every hit with ``[REDACTED-INSTRUCTION]``.

        Returns ``(redacted_text, signals, stripped_pattern_ids)``. Surrounding prose is
        kept: the model still needs the advisory, it just no longer receives the order.
        """
        if not text:
            return "", [], []

        spans: list[tuple[int, int]] = []
        signals: list[tuple[int, InjectionSignal]] = []
        stripped: list[str] = []
        for pattern in self.patterns:
            for match in pattern.regex.finditer(text):
                snippet = match.group(0)
                if not snippet.strip():
                    continue
                spans.append((match.start(), match.end()))
                signals.append(
                    (
                        match.start(),
                        InjectionSignal(
                            pattern_id=pattern.pattern_id,
                            category=pattern.category,
                            snippet=snippet.strip()[:_MAX_SNIPPET],
                            tier=tier,
                        ),
                    )
                )
                if pattern.pattern_id not in stripped:
                    stripped.append(pattern.pattern_id)

        if not spans:
            return text, [], []

        out: list[str] = []
        cursor = 0
        for start, end in _merge_spans(spans):
            out.append(text[cursor:start])
            out.append(REDACTION_MARKER)
            cursor = end
        out.append(text[cursor:])

        signals.sort(key=lambda item: (item[0], item[1].pattern_id))
        redacted = re.sub(r"\s{2,}", " ", "".join(out)).strip()
        return redacted, [signal for _, signal in signals], stripped

    # -- verdict -----------------------------------------------------------

    def verdict(self, signals: Iterable[InjectionSignal]) -> InjectionVerdict:
        """Map a signal count onto the configured verdict thresholds."""
        count = len(list(signals))
        if count >= self.config.injected_signal_threshold:
            return InjectionVerdict.INJECTED
        if count >= self.config.suspicious_signal_threshold:
            return InjectionVerdict.SUSPICIOUS
        return InjectionVerdict.CLEAN
