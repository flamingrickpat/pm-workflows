"""Optional phase-kind extensions for the workflow kernel.

The default registry is empty.  An empty registry preserves the built-in
manifest grammar and execution behavior.  Embedders can add a phase kind or
replace a built-in phase executor without changing the kernel's dispatch
code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

if TYPE_CHECKING:
    from .manifest import Workflow
    from .protocol import PhaseConfig


PhaseParser = Callable[[Mapping[str, Any], Path], Mapping[str, Any]]
PhaseValidator = Callable[["PhaseConfig", "Workflow"], None]
PhaseExecutor = Callable[[Any, "PhaseConfig"], Mapping[str, Any]]


@dataclass(frozen=True)
class PhaseKindExtension:
    """One optional manifest phase kind or built-in executor override."""

    kind: str
    execute: PhaseExecutor
    parse: PhaseParser | None = None
    validate: PhaseValidator | None = None

    def __post_init__(self) -> None:
        if not self.kind or not self.kind.strip():
            raise ValueError("phase extension kind must be a non-empty string")


class PhaseExtensionRegistry:
    """An immutable lookup table of phase extensions."""

    def __init__(self, extensions: Iterable[PhaseKindExtension] = ()) -> None:
        values: dict[str, PhaseKindExtension] = {}
        for extension in extensions:
            if extension.kind in values:
                raise ValueError(f"duplicate phase extension kind '{extension.kind}'")
            values[extension.kind] = extension
        self._extensions = MappingProxyType(values)

    def get(self, kind: str) -> PhaseKindExtension | None:
        return self._extensions.get(kind)

    def __contains__(self, kind: object) -> bool:
        return kind in self._extensions

    @property
    def kinds(self) -> frozenset[str]:
        return frozenset(self._extensions)


EMPTY_PHASE_EXTENSIONS = PhaseExtensionRegistry()
