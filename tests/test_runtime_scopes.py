"""Role-level tool grants, child namespaces, and runtime MCP descriptors.

These use real Python roles, a real in-memory environment, and the real
kernel. They prove that a role sees only its declared Python tools, that a
child workflow receives a scoped runtime instead of the parent's, and that a
runtime can supply per-run MCP server descriptors without a shared
``.mcp.json``.
"""
from __future__ import annotations

import json
import posixpath
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from pm_workflows.kernel import Kernel
from pm_workflows.manifest import ManifestError
from pm_workflows.runtime import WorkflowRuntime
from pm_workflows.working_target import CommandResult

TASK_ID = "T-scope"


class MemoryEnvironment:
    def __init__(self) -> None:
        self._files: dict[str, str] = {}

    def read_text(self, path: str) -> str:
        return self._files[posixpath.normpath(path)]

    def write_text(self, path: str, text: str) -> None:
        self._files[posixpath.normpath(path)] = text

    def move(self, source: str, destination: str) -> None:
        self._files[posixpath.normpath(destination)] = self._files.pop(
            posixpath.normpath(source)
        )

    def list_files(self, directory: str) -> list[str]:
        prefix = posixpath.normpath(directory).rstrip("/") + "/"
        return sorted(
            posixpath.basename(path)
            for path in self._files
            if path.startswith(prefix) and "/" not in path[len(prefix):]
        )

    def exec(self, command: str) -> CommandResult:
        return CommandResult(stdout="", stderr="", exit_code=0)


ROLE_MANIFEST = """\
---
name: scope-t
driver: {kind: python}
human_resolver: {mode: forbid}
roles:
  role:
    skill: skills/native/role.py
    tools: [TOOLGRANT]
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: work
    kind: role
    role: role
    on_status: {done: stop}
    on_invalid: {action: stop}
---
"""

PARENT_MANIFEST = """\
---
name: parent-t
driver: {kind: python}
human_resolver: {mode: forbid}
phases:
  - name: fan
    kind: workflow
    workflow: child
    task:
      id: ${TASK_ID}.child
      input: {}
    result: {statuses: [done]}
    on_status: {done: stop}
    on_invalid: {action: stop}
---
"""

CHILD_MANIFEST = """\
---
name: child
driver: {kind: python}
human_resolver: {mode: forbid}
roles:
  child-role:
    skill: skills/native/child.py
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: work
    kind: role
    role: child-role
    on_status: {done: stop}
    on_invalid: {action: stop}
---
"""


def _write_role(base: Path, code: str) -> None:
    (base / "skills" / "native").mkdir(parents=True, exist_ok=True)
    (base / "skills" / "native" / "role.py").write_text(code, encoding="utf-8")


def _kernel(base: Path, tmp_path: Path, runtime: WorkflowRuntime) -> Kernel:
    return Kernel(
        manifest_path=base / "w.workflow.md",
        workspace=tmp_path / "repo",
        task_id=TASK_ID,
        base_dir=base,
        kernel_data_root=tmp_path / "kernel_data",
        runtime=runtime,
    )


def test_role_tool_grant_narrows_the_run_tools(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _write_role(
        base,
        "def run(context):\n"
        "    keys = sorted(context.runtime.tools)\n"
        "    context.runtime.environment.write_text('outbox/keys.txt', ','.join(keys))\n"
        "    return {'status': 'done', 'summary': 'ok'}\n",
    )
    (base / "w.workflow.md").write_text(ROLE_MANIFEST.replace("TOOLGRANT", "alpha"), encoding="utf-8")
    environment = MemoryEnvironment()
    runtime = WorkflowRuntime(
        environment, threading.Event(), {"alpha": object(), "beta": object()},
    )

    result = _kernel(base, tmp_path, runtime).run()

    assert result["terminal_status"] == "done"
    assert environment.read_text("outbox/keys.txt") == "alpha"


def test_role_requesting_an_ungranted_tool_is_a_config_error(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _write_role(base, "def run(context):\n    return {'status': 'done', 'summary': 'x'}\n")
    (base / "w.workflow.md").write_text(ROLE_MANIFEST.replace("TOOLGRANT", "missing"), encoding="utf-8")
    runtime = WorkflowRuntime(
        MemoryEnvironment(), threading.Event(), {"alpha": object()},
    )

    with pytest.raises(ManifestError, match="missing"):
        _kernel(base, tmp_path, runtime).run()


def test_child_workflow_receives_a_scoped_runtime(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _write_role(
        base,
        "def run(context):\n"
        "    keys = sorted(context.runtime.tools)\n"
        "    context.runtime.environment.write_text('child-tools.txt', ','.join(keys))\n"
        "    return {'status': 'done', 'summary': 'child'}\n",
    )
    (base / "w.workflow.md").write_text(PARENT_MANIFEST, encoding="utf-8")
    child_dir = base / "workflows" / "child"
    child_dir.mkdir(parents=True)
    (child_dir / "child.workflow.md").write_text(CHILD_MANIFEST, encoding="utf-8")
    (base / "skills" / "native" / "child.py").write_text(
        "def run(context):\n"
        "    keys = sorted(context.runtime.tools)\n"
        "    context.runtime.environment.write_text('child-tools.txt', ','.join(keys))\n"
        "    return {'status': 'done', 'summary': 'child'}\n",
        encoding="utf-8",
    )
    environment = MemoryEnvironment()
    scopes: list[str] = []

    def child_factory(scope: str) -> WorkflowRuntime:
        scopes.append(scope)
        return WorkflowRuntime(environment, threading.Event(), {"child_only": object()})

    runtime = WorkflowRuntime(
        environment, threading.Event(), {"parent_only": object()},
        child_factory=child_factory,
    )

    result = _kernel(base, tmp_path, runtime).run()

    assert result["terminal_status"] == "done", result
    assert scopes and scopes[0].startswith("fan")
    assert environment.read_text("child-tools.txt") == "child_only"


def test_runtime_mcp_descriptors_replace_a_shared_workspace_config(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _write_role(base, "def run(context):\n    return {'status': 'done', 'summary': 'x'}\n")
    manifest = ROLE_MANIFEST.replace("TOOLGRANT", "").replace(
        "    result_contract:",
        "    mcp: [ghost]\n    result_contract:",
    )
    (base / "w.workflow.md").write_text(manifest, encoding="utf-8")
    runtime = WorkflowRuntime(
        MemoryEnvironment(), threading.Event(), {},
        mcp_servers={"ghost": {"url": "http://127.0.0.1:9/mcp", "type": "http"}},
    )
    kernel = _kernel(base, tmp_path, runtime)
    kernel.allowed_mcp = {"ghost"}
    role = kernel.manifest.roles["role"]
    phase = kernel.manifest.phase_by_name("work")

    config = kernel._filtered_mcp_config(phase, role, 1)

    assert json.loads(config.read_text(encoding="utf-8")) == {
        "mcpServers": {"ghost": {"url": "http://127.0.0.1:9/mcp", "type": "http"}}
    }


def test_runtime_mcp_descriptor_missing_a_requested_name_is_a_config_error(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _write_role(base, "def run(context):\n    return {'status': 'done', 'summary': 'x'}\n")
    manifest = ROLE_MANIFEST.replace("TOOLGRANT", "").replace(
        "    result_contract:",
        "    mcp: [ghost]\n    result_contract:",
    )
    (base / "w.workflow.md").write_text(manifest, encoding="utf-8")
    runtime = WorkflowRuntime(
        MemoryEnvironment(), threading.Event(), {}, mcp_servers={},
    )
    kernel = _kernel(base, tmp_path, runtime)
    kernel.allowed_mcp = {"ghost"}
    role = kernel.manifest.roles["role"]
    phase = kernel.manifest.phase_by_name("work")

    with pytest.raises(ManifestError, match="ghost"):
        kernel._filtered_mcp_config(phase, role, 1)