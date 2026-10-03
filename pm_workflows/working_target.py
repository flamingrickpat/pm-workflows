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
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .inputs import FileMetadata


@dataclass(frozen=True)
class AgentTarget:
    """Explicit host cwd or virtual machine and user for all coder tools."""

    cwd: Path
    bash_machine: Any = None
    user: str = "user"


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int


class WorkingEnvironment(ABC):
    """One invocation-scoped read/write/exec view of a working target."""

    def agent_target(self, artifact_workspace: Path) -> AgentTarget:
        """Bind agent tools explicitly. Python-only environments need no binding."""
        raise NotImplementedError("This working environment does not support agents.")

    def file_metadata(self, path: str) -> FileMetadata:
        """Return a text digest version with no invented time or producer.

        Older environments need no new method. Native implementations can
        supply an opaque revision and actual modification time instead.
        """
        import hashlib
        from .inputs import FileMetadata, relative_path

        path = relative_path(path)
        raw = self.read_text(path).encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()
        return FileMetadata(path, len(raw), sha, sha)

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


class HostInputEnvironment:
    """Read-only host adapter for kernels without a supplied working environment.

    Paths remain relative to root. Resolved symlinks cannot escape that root.
    This adapter never executes commands or opens a virtual path on the host.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def _path(self, path: str, *, folder: bool = False) -> Path:
        from .inputs import InputError, relative_path
        target = (self.root / relative_path(path, folder=folder)).resolve()
        if target != self.root and self.root not in target.parents:
            raise InputError("inaccessible_path", f"symlink escapes environment: {path}")
        return target

    def read_text(self, path: str) -> str:
        """Read exact UTF-8 text without newline conversion."""
        with self._path(path).open(encoding="utf-8", newline="") as handle:
            return handle.read()

    def check_file_boundary(self, path: str, folder: str) -> None:
        """Reject symlinks whose real target escapes the declared input folder."""
        from .inputs import InputError
        target = self._path(path)
        boundary = self._path(folder, folder=True)
        if boundary not in target.parents:
            raise InputError("inaccessible_path", f"file target escapes input_folder: {path}")

    def canonical_relative_path(self, path: str) -> str:
        """Return the resolved path inside this root for grant checks."""
        return self._path(path).relative_to(self.root).as_posix()

    def list_files(self, directory: str) -> list[str]:
        """Return sorted direct-child paths. Missing directories raise an error."""
        import posixpath
        from .inputs import relative_path
        folder = relative_path(directory, folder=True)
        target = self._path(folder, folder=True)
        return sorted(posixpath.normpath(posixpath.join(folder, path.name))
                      for path in target.iterdir() if path.is_file())

    def file_metadata(self, path: str) -> FileMetadata:
        """Capture host stat identity and content hash between two stat reads."""
        import hashlib
        from datetime import datetime, timezone
        from .inputs import FileMetadata, InputError, relative_path
        path = relative_path(path)
        target = self._path(path)
        before = target.stat()
        raw = target.read_bytes()
        after = target.stat()
        version = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if version(before) != version(after) or len(raw) != after.st_size:
            raise InputError("stale_input", f"{path} changed during metadata capture")
        return FileMetadata(path, len(raw), repr(version(after)), hashlib.sha256(raw).hexdigest(),
                            datetime.fromtimestamp(after.st_mtime, timezone.utc).isoformat())
