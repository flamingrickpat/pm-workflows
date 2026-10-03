"""Capture scoped inputs without inference or host-path conversion.

The caller owns selection. A folder binding returns a listing for that caller.
Every capture retains its value and version evidence for replay.
"""
from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from .manifest import ManifestError
from .protocol import InputBinding, OutputBinding

JSON_TYPES = frozenset({"string", "object", "array", "integer", "number", "boolean", "null"})


class InputError(ManifestError):
    """An unusable input. Classification callers can route its code to INVALID."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.evidence: dict[str, Any] = {}
        super().__init__(f"{code}: {detail}")


def canonical(value: Any) -> str:
    """Encode JSON data deterministically. Reject foreign values and NaN."""
    def inspect(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise InputError("malformed_value", "JSON object keys must be strings")
            for nested in item.values():
                inspect(nested)
        elif isinstance(item, list):
            for nested in item:
                inspect(nested)
        elif type(item) not in {str, int, float, bool, type(None)}:
            raise InputError("malformed_value", f"unsupported JSON value: {type(item).__name__}")
        elif isinstance(item, float) and not math.isfinite(item):
            raise InputError("malformed_value", "JSON numbers must be finite")
    inspect(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    """Return the SHA-256 of canonical JSON, including scalar type identity."""
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def relative_path(path: str, *, folder: bool = False) -> str:
    """Normalize a relative POSIX path. Reject absolute paths and traversal."""
    if not isinstance(path, str) or not path or "\\" in path or ":" in path or "\x00" in path:
        raise InputError("inaccessible_path", f"expected a relative POSIX path: {path!r}")
    if path.startswith("/") or ".." in path.split("/"):
        raise InputError("inaccessible_path", f"path escapes the environment: {path!r}")
    normalized = posixpath.normpath(path)
    if normalized == "." and not folder:
        raise InputError("incorrect_type", "a file path cannot name the environment root")
    return normalized


def within(path: str, folder: str) -> bool:
    """Return whether a normalized file lies inside the declared folder."""
    return folder == "." or path.startswith(folder.rstrip("/") + "/")


def select_pointer(value: Any, pointer: str) -> Any:
    """Resolve a JSON pointer. Empty text selects the whole value."""
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise InputError("malformed_value", "pointer must be empty or start with '/'")
    current = value
    for encoded in pointer[1:].split("/") if pointer else []:
        # Reject malformed escapes before RFC 6901 decoding.
        if re.search(r"~(?![01])", encoded):
            raise InputError("malformed_value", f"invalid pointer escape: {pointer}")
        part = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and (part == "0" or not part.startswith("0")) and int(part) < len(current):
            current = current[int(part)]
        else:
            raise InputError("missing_field", f"pointer {pointer!r} has no field {part!r}")
    return current


def require_type(value: Any, expected: str | None) -> None:
    """Reject values that do not match the declared JSON type."""
    canonical(value)
    matches = {
        "string": isinstance(value, str), "object": isinstance(value, dict),
        "array": isinstance(value, list), "integer": type(value) is int,
        "number": type(value) in {int, float}, "boolean": type(value) is bool,
        "null": value is None,
    }
    if expected is not None and not matches.get(expected, False):
        raise InputError("incorrect_type", f"expected {expected}, got {type(value).__name__}")


@dataclass(frozen=True)
class FileMetadata:
    """Environment-relative identity and version. Unknown fields remain null.

    Size uses UTF-8 bytes for text-only environments. Revision is opaque.
    Producer metadata comes from the environment, never from report prose.
    """

    path: str
    size_bytes: int
    revision: str
    content_sha256: str
    modified_time: str | None = None
    producer: dict[str, Any] | None = None

    @property
    def candidate_id(self) -> str:
        """Return a path identity that stays stable across file revisions."""
        return hashlib.sha256(self.path.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, **asdict(self)}


@dataclass(frozen=True)
class FolderListing:
    """A captured direct-child candidate set with a deterministic revision."""

    folder: str
    candidates: tuple[FileMetadata, ...]
    data_selection: str = ""

    def to_dict(self) -> dict[str, Any]:
        candidates = [item.to_dict() for item in self.candidates]
        return {"folder": self.folder, "revision": digest(candidates),
                "candidates": candidates, "data_selection": self.data_selection,
                "data_selection_sha256": digest(self.data_selection)}


@dataclass(frozen=True)
class InputCapture:
    """The exact value and replay evidence supplied to a consumer."""

    value: Any
    evidence: dict[str, Any]


@dataclass(frozen=True)
class InputScope:
    """The producer namespace and current logical loop item."""

    task_id: str
    run_id: str
    item: str | None = None


def metadata(environment: Any, path: str) -> FileMetadata:
    """Read environment metadata, or use text digests for an older embedder."""
    path = relative_path(path)
    try:
        method = getattr(environment, "file_metadata", None)
        if method is not None:
            result = method(path)
        else:
            text = environment.read_text(path)
            raw = text.encode("utf-8")
            sha = hashlib.sha256(raw).hexdigest()
            result = FileMetadata(path, len(raw), sha, sha)
    except (FileNotFoundError, KeyError) as exc:
        raise InputError("missing_file", path) from exc
    except UnicodeError as exc:
        raise InputError("malformed_value", f"{path} is not UTF-8 text") from exc
    except (OSError, RuntimeError) as exc:
        raise InputError("inaccessible_path", f"{path}: {exc}") from exc
    if not isinstance(result, FileMetadata) or result.path != path:
        raise InputError("malformed_value", f"invalid environment metadata for {path}")
    if (type(result.size_bytes) is not int or result.size_bytes < 0
            or not isinstance(result.revision, str) or not result.revision
            or not isinstance(result.content_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", result.content_sha256)
            or (result.modified_time is not None and not isinstance(result.modified_time, str))
            or (result.producer is not None and not isinstance(result.producer, dict))):
        raise InputError("malformed_value", f"invalid environment metadata fields for {path}")
    canonical(result.to_dict())
    # A native producer mapping can change after listing. Keep the captured
    # metadata independent so the next version check can detect that change.
    return deepcopy(result)


class InputResolver:
    """Resolve explicit inputs within one environment and producer namespace.

    Resolution performs read effects only. It never retries a stale capture.
    The caller persists receipts and controls cancellation between operations.
    """

    def __init__(self, environment: Any, scope: InputScope, *, request: Any = None,
                 entries: list[dict[str, Any]] | None = None, max_bytes: int = 1_048_576,
                 max_candidates: int = 256):
        self.environment = environment
        self.scope = scope
        self.request = request
        self.entries = entries or []
        self.max_bytes = max_bytes
        self.max_candidates = max_candidates

    def list_folder(self, folder: str, data_selection: str = "") -> FolderListing:
        """Capture metadata for sorted direct-child files. Empty folders fail."""
        folder = relative_path(folder, folder=True)
        try:
            paths = self.environment.list_files(folder)
        except (OSError, RuntimeError, KeyError) as exc:
            raise InputError("inaccessible_path", f"{folder}: {exc}") from exc
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise InputError("malformed_value", "listing must contain relative file paths")
        if not paths:
            raise InputError("empty_listing", folder)
        if len(paths) > self.max_candidates:
            raise InputError("oversized_input", f"listing exceeds {self.max_candidates} candidates")
        normalized = [relative_path(path) for path in paths]
        if len(set(normalized)) != len(normalized):
            raise InputError("malformed_value", "listing contains duplicate paths")
        if any(posixpath.dirname(path) != ("" if folder == "." else folder) for path in normalized):
            raise InputError("inaccessible_path", "listing escaped the declared folder")
        candidates: list[FileMetadata] = []
        for path in sorted(normalized):
            boundary_check = getattr(self.environment, "check_file_boundary", None)
            if boundary_check is not None:
                boundary_check(path, folder)
            candidates.append(metadata(self.environment, path))
        return FolderListing(folder, tuple(candidates), data_selection)

    def capture_selected(self, listing: FolderListing, candidate_id: str) -> InputCapture:
        """Capture one listed ID after a fresh version check. Unknown IDs fail."""
        candidate = next((item for item in listing.candidates if item.candidate_id == candidate_id), None)
        if candidate is None:
            raise InputError("unknown_candidate", str(candidate_id))
        fresh = self.list_folder(listing.folder, listing.data_selection)
        if fresh.to_dict() != listing.to_dict():
            raise InputError("stale_input", "folder changed after selection")
        try:
            capture = self.capture_file(candidate.path, expected=candidate, folder=listing.folder)
        except InputError as exc:
            exc.evidence = {"listing": listing.to_dict(), "selected_id": candidate_id}
            raise
        return InputCapture(capture.value, {**capture.evidence, "listing": listing.to_dict(), "selected_id": candidate_id})

    def capture_file(self, path: str, *, expected: FileMetadata | None = None,
                     folder: str | None = None) -> InputCapture:
        """Capture exact text between metadata checks. Changes reject the read."""
        path = relative_path(path)
        if folder is not None and not within(path, relative_path(folder, folder=True)):
            raise InputError("inaccessible_path", "exact file escaped input_folder")
        boundary_check = getattr(self.environment, "check_file_boundary", None)
        if folder is not None and boundary_check is not None:
            boundary_check(path, folder)
        before = metadata(self.environment, path)
        if expected is not None and before != expected:
            raise InputError("stale_input", f"{path} changed after listing")
        if before.size_bytes > self.max_bytes:
            raise InputError("oversized_input", f"{path} exceeds {self.max_bytes} bytes")
        try:
            text = self.environment.read_text(path)
        except (FileNotFoundError, KeyError) as exc:
            raise InputError("missing_file", path) from exc
        except UnicodeError as exc:
            raise InputError("malformed_value", f"{path} is not UTF-8 text") from exc
        except (OSError, RuntimeError) as exc:
            raise InputError("inaccessible_path", f"{path}: {exc}") from exc
        raw = text.encode("utf-8")
        after = metadata(self.environment, path)
        sha = hashlib.sha256(raw).hexdigest()
        if before != after or sha != before.content_sha256 or len(raw) != before.size_bytes:
            raise InputError("stale_input", f"{path} changed during capture")
        producer = before.producer
        if producer is not None:
            for key, current in asdict(self.scope).items():
                if key in producer and producer[key] != current:
                    raise InputError("stale_input", f"file producer belongs to another {key}")
        for entry in self.entries:
            for evidence in entry.get("input_evidence", {}).values():
                prior = evidence.get("file", {})
                if (self.scope.item is not None and evidence.get("scope", {}).get("item") != self.scope.item
                        and prior.get("path") == path and prior.get("content_sha256") == sha):
                    raise InputError("stale_input", "unchanged file was captured for a previous loop item")
        evidence = {"schema": "pm.input-capture.v1", "mode": "input_file", "scope": asdict(self.scope),
                    "file": before.to_dict(), "content": text, "value": text, "value_sha256": digest(text),
                    "limitations": [] if producer else ["producer metadata unavailable"]}
        return InputCapture(text, evidence)

    def named_output(self, name: str) -> InputCapture:
        """Bind the latest producer attempt. Failed or foreign items cannot reuse it."""
        for entry in reversed(self.entries):
            declared = entry.get("named_outputs", {})
            if name not in declared:
                continue
            record = declared[name]
            if not entry.get("ok") or not isinstance(record, dict):
                raise InputError("stale_output", f"latest producer of {name!r} did not validate")
            scope = record.get("scope")
            if scope != asdict(self.scope):
                raise InputError("stale_output", f"{name!r} belongs to another invocation or loop item")
            if digest(record.get("value")) != record.get("value_sha256"):
                raise InputError("stale_output", f"{name!r} digest does not match its value")
            return InputCapture(deepcopy(record["value"]), {"schema": "pm.input-capture.v1", "mode": "output", **deepcopy(record)})
        raise InputError("missing_output", name)

    def resolve(self, binding: InputBinding) -> InputCapture:
        """Resolve one binding and assert its projection type. No inference runs."""
        if binding.mode == "input_file":
            capture = self.capture_file(binding.source, folder=binding.input_folder)
            value = capture.value
            if binding.pointer:
                from .child import _structured_text
                try:
                    value = _structured_text(binding.source, value)
                except Exception as exc:
                    raise InputError("malformed_value", str(exc)) from exc
        elif binding.mode == "input_folder":
            listing = self.list_folder(binding.source, binding.data_selection)
            value = listing.to_dict()
            capture = InputCapture(value, {"schema": "pm.input-capture.v1", "mode": "input_folder", "scope": asdict(self.scope), "listing": value})
        elif binding.mode == "output":
            capture = self.named_output(binding.source)
            value = capture.value
        elif binding.mode == "request":
            from .child import dotted
            try:
                value = dotted(self.request, binding.source)
            except (ManifestError, IndexError) as exc:
                raise InputError("missing_field", str(exc)) from exc
            capture = InputCapture(value, {"schema": "pm.input-capture.v1", "mode": "request", "field": binding.source,
                                          "scope": asdict(self.scope), "request_sha256": digest(self.request)})
        elif binding.mode == "literal":
            value = binding.source
            capture = InputCapture(value, {"schema": "pm.input-capture.v1", "mode": "literal", "scope": asdict(self.scope)})
        else:
            raise InputError("malformed_value", f"unknown input mode: {binding.mode}")
        source_value = value
        value = select_pointer(value, binding.pointer)
        require_type(value, binding.expected_type)
        if len(canonical(value).encode("utf-8")) > self.max_bytes:
            raise InputError("oversized_input", f"value exceeds {self.max_bytes} bytes")
        return InputCapture(deepcopy(value), {**capture.evidence, "pointer": binding.pointer,
                                    "source_value": deepcopy(source_value), "source_sha256": digest(source_value),
                                    "value": deepcopy(value), "value_sha256": digest(value)})

    def project_outputs(self, declarations: dict[str, OutputBinding], result: Any,
                        *, phase: str, attempt: int, revision: str | None) -> dict[str, Any]:
        """Project named values only after the producer result passes its contract."""
        outputs: dict[str, Any] = {}
        for name, binding in declarations.items():
            value = select_pointer(result, binding.pointer)
            require_type(value, binding.expected_type)
            if len(canonical(value).encode("utf-8")) > self.max_bytes:
                raise InputError("oversized_input", f"output {name!r} exceeds {self.max_bytes} bytes")
            outputs[name] = {"value": deepcopy(value), "value_sha256": digest(value), "scope": asdict(self.scope),
                             "producer": {"phase": phase, "attempt": attempt, "revision": revision}, "pointer": binding.pointer}
        return outputs


class GrantedInputEnvironment:
    """Restrict kernel capture to a role's declared paths within its runtime view."""

    def __init__(self, environment: Any, readable: list[str], denied: list[str]):
        self.environment = environment
        self.readable = readable
        self.denied = denied

    @staticmethod
    def _matches(path: str, pattern: str) -> bool:
        import fnmatch
        normalized = pattern.rstrip("/")
        return normalized == "." or path == normalized or path.startswith(normalized + "/") or fnmatch.fnmatchcase(path, pattern)

    def _allow(self, path: str) -> None:
        canonical_path = getattr(self.environment, "canonical_relative_path", lambda value: value)(path)
        if any(self._matches(value, pattern) for pattern in self.denied for value in {path, canonical_path}):
            raise InputError("inaccessible_path", f"role denies {path}")
        if self.readable and not all(any(self._matches(value, pattern) for pattern in self.readable) for value in {path, canonical_path}):
            raise InputError("inaccessible_path", f"role has no read grant for {path}")

    def read_text(self, path: str) -> str:
        self._allow(path)
        return self.environment.read_text(path)

    def file_metadata(self, path: str) -> FileMetadata:
        self._allow(path)
        return metadata(self.environment, path)

    def list_files(self, directory: str) -> list[str]:
        # List through the granted environment. Exclude denied candidates
        # before metadata or content can enter the consumer's prompt.
        paths = self.environment.list_files(directory)
        visible: list[str] = []
        for path in paths:
            try:
                self._allow(path)
            except InputError:
                continue
            visible.append(path)
        return visible

    def check_file_boundary(self, path: str, folder: str) -> None:
        """Preserve native boundary checks through the role's granted view."""
        method = getattr(self.environment, "check_file_boundary", None)
        if method is not None:
            method(path, folder)
