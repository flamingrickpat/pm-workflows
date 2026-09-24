"""The working-target contract a workflow invocation reads and writes.

A ``WorkingTarget`` opens one ``WorkingEnvironment`` per invocation. The
environment is a small POSIX-like file contract: relative paths, text I/O,
one move operation, a direct-child listing, and a shell command.

File paths are relative POSIX paths, such as ``inbox/request.yaml``.
``list_files()`` returns sorted direct-child paths relative to the
environment root. ``write_text()`` creates missing parent directories.
``move()`` creates the destination parent and moves one file.

This package declares the interface only. Real-folder and virtual (in-memory)
implementations live in embedders, not here. The kernel never opens an
environment or runs a command itself; it hands ``WorkflowRuntime`` to a role.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int


class WorkingEnvironment(ABC):
    """One invocation-scoped read/write/exec view of a working target."""

    @abstractmethod
    def read_text(self, path: str) -> str:
        """Return the exact text of one file."""

    @abstractmethod
    def write_text(self, path: str, text: str) -> None:
        """Write one file, creating missing parent directories."""

    @abstractmethod
    def move(self, source: str, destination: str) -> None:
        """Move one file, creating the destination parent directory."""

    @abstractmethod
    def list_files(self, directory: str) -> list[str]:
        """Return sorted direct-child file paths relative to the root."""

    @abstractmethod
    def exec(self, command: str) -> CommandResult:
        """Run one shell command and return stdout, stderr, and exit code."""


class WorkingTarget(ABC):
    """Opens a fresh environment for one invocation."""

    @abstractmethod
    def open(self, invocation_id: str) -> WorkingEnvironment:
        """Create an environment scoped to one invocation id."""