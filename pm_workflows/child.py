"""Deterministic helpers for opt-in child-workflow phases."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .manifest import ManifestError

EXACT_VARIABLE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_.]*)\}$")
VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_.]*)\}")
FENCED_DATA = re.compile(r"```(?:yaml|yml|json)\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def dotted(value: Any, path: str) -> Any:
    """Resolve a dotted field path through mappings and sequence indexes."""
    current = value
    if not path:
        return current
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            raise ManifestError(f"cannot resolve field '{path}' at '{part}'")
    return current


def expand_runtime(value: Any, variables: dict[str, Any]) -> Any:
    """Expand foreach variables while preserving structured exact matches."""
    if isinstance(value, list):
        return [expand_runtime(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand_runtime(item, variables) for key, item in value.items()}
    if not isinstance(value, str):
        return value

    exact = EXACT_VARIABLE.match(value)
    if exact:
        key = exact.group(1)
        root, _, tail = key.partition(".")
        if root in variables:
            return dotted(variables[root], tail)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        root, _, tail = key.partition(".")
        if root not in variables:
            return match.group(0)
        resolved = dotted(variables[root], tail)
        if isinstance(resolved, (dict, list)):
            return json.dumps(resolved, ensure_ascii=False, sort_keys=True)
        return str(resolved)

    return VARIABLE.sub(replace, value)


def _structured_text(path: str | Path, text: str) -> Any:
    path = PurePosixPath(str(path))
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if text.startswith("---\n"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            return yaml.safe_load(parts[0][4:])
    fenced = FENCED_DATA.search(text)
    if fenced:
        return yaml.safe_load(fenced.group(1))
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ManifestError(f"cannot parse structured artifact {path}: {exc}") from exc


def _json_pointer(value: Any, pointer: str, reference: str) -> Any:
    current = value
    for encoded in pointer.split("/"):
        part = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ManifestError(f"reference '{reference}' has no JSON pointer '/{pointer}'")
    return current


def load_reference(reference: str, workspace: Path, environment: Any = None) -> Any:
    """Read a scoped file or glob and an optional JSON pointer.

    Supplied environments own all paths. Legacy host references remain scoped
    to workspace, including absolute references inside that workspace.
    """
    from .inputs import InputResolver, InputScope, relative_path, select_pointer
    from .working_target import HostInputEnvironment
    source, marker, pointer = reference.partition("#/")
    if environment is None:
        candidate = Path(source)
        if candidate.is_absolute():
            try:
                source = candidate.resolve().relative_to(workspace.resolve()).as_posix()
            except ValueError as exc:
                raise ManifestError(f"reference escapes the workspace: {reference}") from exc
        environment = HostInputEnvironment(workspace)
    resolver = InputResolver(environment, InputScope("reference", "reference"))
    if any(token in source for token in ("*", "?", "[")):
        if marker:
            raise ManifestError(f"glob reference cannot use a JSON pointer: {reference}")
        import fnmatch
        source = relative_path(source)
        if isinstance(environment, HostInputEnvironment):
            return [{"path": path.relative_to(workspace.resolve()).as_posix(),
                     "content": resolver.capture_file(path.relative_to(workspace.resolve()).as_posix()).value}
                    for path in sorted(workspace.resolve().glob(relative_path(source))) if path.is_file()]
        # Virtual folder components stay exact. The environment exposes only
        # direct-child file listings, with no recursive directory contract.
        components = source.split("/")
        if any(any(token in part for token in ("*", "?", "[")) for part in components[:-1]):
            raise ManifestError("glob folder components must be exact scoped paths")
        folder = "/".join(components[:-1]) or "."
        return [{"path": path, "content": resolver.capture_file(path).value}
                for path in environment.list_files(folder) if fnmatch.fnmatchcase(path, source)]
    text = resolver.capture_file(source).value
    if not marker:
        return text
    return select_pointer(_structured_text(source, text), "/" + pointer)


def order_items(items: list[Any], stable_id: str, order: str) -> list[Any]:
    """Return deterministic stable-id or dependency-topological order."""
    keyed: dict[str, Any] = {}
    for item in items:
        key = str(dotted(item, stable_id))
        if not key:
            raise ManifestError("foreach stable IDs cannot be empty")
        if key in keyed:
            raise ManifestError(f"foreach stable ID is duplicated: {key}")
        keyed[key] = item
    if order == "stable_id":
        return [keyed[key] for key in sorted(keyed)]

    dependencies: dict[str, set[str]] = {}
    for key, item in keyed.items():
        raw = item.get("depends_on", []) if isinstance(item, dict) else []
        if isinstance(raw, str):
            raw = [raw]
        dependencies[key] = {str(dep) for dep in raw if str(dep) in keyed}
    result: list[Any] = []
    remaining = set(keyed)
    while remaining:
        ready = sorted(key for key in remaining if not (dependencies[key] & remaining))
        if not ready:
            raise ManifestError(
                "foreach dependency graph contains a cycle among: "
                + ", ".join(sorted(remaining))
            )
        for key in ready:
            result.append(keyed[key])
            remaining.remove(key)
    return result


def copy_declared_artifacts(
    child_task_dir: Path,
    parent_task_dir: Path,
    artifact_prefix: str,
    artifact_names: list[str],
    attempt: int,
) -> list[str]:
    """Copy only declared child artifacts into an immutable attempt folder."""
    if not artifact_prefix or not artifact_names:
        return []
    root = (parent_task_dir / artifact_prefix / f"attempt-{attempt:04d}").resolve()
    parent_root = parent_task_dir.resolve()
    if root != parent_root and parent_root not in root.parents:
        raise ManifestError(f"child artifact_prefix escapes the parent task: {artifact_prefix}")
    copied: list[str] = []
    for name in artifact_names:
        source = (child_task_dir / name).resolve()
        child_root = child_task_dir.resolve()
        if source != child_root and child_root not in source.parents:
            raise ManifestError(f"declared child artifact escapes its task folder: {name}")
        if not source.is_file():
            continue
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination.relative_to(parent_task_dir.parent.parent.parent).as_posix())
    return copied
