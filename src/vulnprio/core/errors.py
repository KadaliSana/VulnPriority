"""Exception hierarchy shared by every vulnprio module."""

from __future__ import annotations

__all__ = [
    "VulnprioError",
    "ConfigError",
    "ParseError",
    "FeedUnavailableError",
    "OfflineViolationError",
    "TemporalLeakageError",
    "SandboxViolationError",
    "CanaryLeakError",
    "SchemaRejectedError",
    "EvidenceSpanError",
    "InfluenceBudgetExceeded",
    "GraphError",
    "MonotonicityViolationError",
    "RankerNotFittedError",
    "LabelPolicyError",
]


class VulnprioError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(VulnprioError):
    """Invalid or inconsistent configuration."""


class ParseError(VulnprioError):
    """Scanner output could not be parsed into a Scan."""


class FeedUnavailableError(VulnprioError):
    """A feed could not supply data for the requested key and as-of date."""


class OfflineViolationError(VulnprioError):
    """Offline mode attempted a network access. Always a bug, never a warning."""


class TemporalLeakageError(VulnprioError):
    """Data dated after the as-of cut-off entered the pipeline (Gap 3/4)."""


class SandboxViolationError(VulnprioError):
    """Untrusted content reached a prompt without passing through the sandbox."""


class CanaryLeakError(SandboxViolationError):
    """A canary token appeared in model output: the injection reached the model."""


class SchemaRejectedError(SandboxViolationError):
    """Model output failed schema validation after all retries."""


class EvidenceSpanError(SandboxViolationError):
    """A quoted evidence span is not a substring of the sanitized input."""


class InfluenceBudgetExceeded(VulnprioError):
    """An untrusted tier tried to move a feature further than its budget allows."""


class GraphError(VulnprioError):
    """Attack graph construction failed."""


class MonotonicityViolationError(GraphError):
    """Patching a finding increased computed risk, which must be impossible."""


class RankerNotFittedError(VulnprioError):
    """score() called before fit() on a ranker that requires fitting."""


class LabelPolicyError(VulnprioError):
    """An attempt was made to build labels from a disallowed source (for example CVSS)."""
