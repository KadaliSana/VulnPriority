"""Named attacker presets from ``configs/attacker_models/`` (Gap 2).

The presets are the operator-tier statement of *whose* risk is being ranked. Keeping
them as files rather than constants means a reviewer can diff the adversary, and means
the ablation and the adversarial evaluation can swap adversaries without touching code.
"""

from __future__ import annotations

from pathlib import Path

from vulnpriority.core.config import PROJECT_ROOT, load_attacker_preset
from vulnpriority.core.errors import ConfigError
from vulnpriority.core.models import AttackerModel

__all__ = [
    "PRESET_DIR",
    "DEFAULT_PRESET",
    "list_presets",
    "load_preset",
    "load_all_presets",
]

#: Where the shipped presets live. Overridable per call for tests and alternative decks.
PRESET_DIR: Path = PROJECT_ROOT / "configs" / "attacker_models"

#: Matches ``ComponentBConfig.attacker_preset``; the mass-scanning adversary.
DEFAULT_PRESET: str = "opportunistic"


def list_presets(root: Path | None = None) -> list[str]:
    """Preset names available on disk, sorted so listings are reproducible."""
    directory = Path(root) if root is not None else PRESET_DIR
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def load_preset(name: str, root: Path | None = None) -> AttackerModel:
    """Load one preset by name.

    Delegates the parse to :func:`vulnpriority.core.config.load_attacker_preset` so the
    frozen contract owns validation; this wrapper only improves the error message by
    listing what *is* available, which is what a user actually needs when they typo.
    """
    directory = Path(root) if root is not None else PRESET_DIR
    try:
        return load_attacker_preset(name, root=directory)
    except ConfigError as exc:
        available = ", ".join(list_presets(directory)) or "none found"
        raise ConfigError(f"{exc} - available presets: {available}") from exc


def load_all_presets(root: Path | None = None) -> dict[str, AttackerModel]:
    """Every preset on disk, keyed by name; used by comparison reports and tests."""
    directory = Path(root) if root is not None else PRESET_DIR
    return {name: load_preset(name, directory) for name in list_presets(directory)}
