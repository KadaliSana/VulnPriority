"""The framework's own canonical ``Scan`` JSON, plus scanner auto-detection.

Why a parser for our own format: every stage after ingest reads a ``Scan``, so the
pipeline, the synthetic generator and the fixtures all need a lossless on-disk form.
``Scan.model_dump(mode="json")`` in, an identical ``Scan`` out - which is exactly what the
round-trip test asserts, and what makes a run reproducible from artefacts alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from vulnprio.core.errors import ParseError
from vulnprio.core.interfaces import ScannerParser
from vulnprio.core.models import Scan
from vulnprio.core.registry import register_parser

__all__ = ["GenericJsonParser", "detect_parser", "parse_scan", "PARSER_ORDER"]


@register_parser("generic")
class GenericJsonParser(ScannerParser):
    """``ScannerParser`` for the canonical ``Scan`` JSON document."""

    name = "generic"

    def sniff(self, path: str | Path) -> bool:
        """True for a JSON object carrying the canonical ``Scan`` key set."""
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        stripped = text.lstrip()
        if not stripped.startswith("{"):
            return False
        try:
            data = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return False
        return isinstance(data, dict) and {"scan_id", "app_id", "findings"} <= set(data)

    def parse(self, path: str | Path, app_id: str | None = None) -> Scan:
        """Validate the document into a :class:`Scan`.

        ``app_id`` is honoured as an assertion only: the canonical file already carries
        identifiers derived from its own application id, and silently relabelling it would
        invalidate every ``endpoint_id`` and ``finding_id`` inside.
        """
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ParseError(f"invalid canonical scan JSON {source}: {error}") from error
        try:
            scan = Scan.model_validate(data)
        except Exception as error:  # pydantic ValidationError and friends
            raise ParseError(f"{source} is not a valid Scan document: {error}") from error
        if app_id is not None and app_id != scan.app_id:
            raise ParseError(
                f"{source} carries app_id {scan.app_id!r}, which cannot be relabelled to "
                f"{app_id!r} without invalidating every identifier it contains"
            )
        return scan


def _parser_order() -> tuple[type[ScannerParser], ...]:
    """Sniff order: most specific format first so a canonical scan is never mistaken for ZAP."""
    from vulnprio.ingest.burp import BurpParser
    from vulnprio.ingest.nikto import NiktoParser
    from vulnprio.ingest.nuclei import NucleiParser
    from vulnprio.ingest.wapiti import WapitiParser
    from vulnprio.ingest.zap import ZapParser

    return (GenericJsonParser, ZapParser, BurpParser, NucleiParser, WapitiParser, NiktoParser)


#: Public, stable detection order.
PARSER_ORDER: tuple[str, ...] = ("generic", "zap", "burp", "nuclei", "wapiti", "nikto")


def detect_parser(path: str | Path) -> ScannerParser:
    """Return the parser that recognises ``path``.

    Raises :class:`ParseError` when no registered parser claims the file, rather than
    guessing: a silently mis-parsed report would corrupt every identifier downstream.
    """
    source = Path(path)
    if not source.exists():
        raise ParseError(f"scan file not found: {source}")
    for parser_class in _parser_order():
        parser = parser_class()
        if parser.sniff(source):
            return parser
    raise ParseError(f"no registered scanner parser recognises {source}")


def parse_scan(path: str | Path, app_id: str | None = None) -> Scan:
    """Auto-detect the format of ``path`` and parse it."""
    return detect_parser(path).parse(path, app_id=app_id)
