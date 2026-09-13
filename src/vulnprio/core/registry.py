"""Name to class registries: the framework's plugin points."""

from __future__ import annotations

from typing import Callable, TypeVar

from vulnprio.core.enums import LLMBackendKind, RankerName

__all__ = [
    "RANKERS",
    "LLM_BACKENDS",
    "SCANNER_PARSERS",
    "register_ranker",
    "register_backend",
    "register_parser",
    "get_ranker",
    "get_backend",
    "get_parser",
]

C = TypeVar("C", bound=type)

RANKERS: dict[RankerName, type] = {}
LLM_BACKENDS: dict[LLMBackendKind, type] = {}
SCANNER_PARSERS: dict[str, type] = {}


def register_ranker(name: RankerName) -> Callable[[C], C]:
    def decorator(cls: C) -> C:
        cls.name = name  # type: ignore[attr-defined]
        RANKERS[name] = cls
        return cls

    return decorator


def register_backend(kind: LLMBackendKind) -> Callable[[C], C]:
    def decorator(cls: C) -> C:
        cls.kind = kind  # type: ignore[attr-defined]
        LLM_BACKENDS[kind] = cls
        return cls

    return decorator


def register_parser(name: str) -> Callable[[C], C]:
    def decorator(cls: C) -> C:
        cls.name = name  # type: ignore[attr-defined]
        SCANNER_PARSERS[name] = cls
        return cls

    return decorator


def get_ranker(name: RankerName) -> type:
    if name not in RANKERS:
        raise KeyError(f"ranker not registered: {name}. Import vulnprio.rank to register the built-ins.")
    return RANKERS[name]


def get_backend(kind: LLMBackendKind) -> type:
    if kind not in LLM_BACKENDS:
        raise KeyError(f"LLM backend not registered: {kind}. Import vulnprio.llm to register the built-ins.")
    return LLM_BACKENDS[kind]


def get_parser(name: str) -> type:
    if name not in SCANNER_PARSERS:
        raise KeyError(f"scanner parser not registered: {name}. Import vulnprio.ingest to register the built-ins.")
    return SCANNER_PARSERS[name]
