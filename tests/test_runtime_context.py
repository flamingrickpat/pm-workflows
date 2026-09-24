"""Integration checks for the per-invocation ``WorkflowRuntime`` contract.

These use a real Python-role skill and a real (in-memory) working
environment. They verify that a role reads and writes through
``context.runtime.environment``, that cooperative cancellation terminates
one invocation without retry, and that Python exceptions either propagate or
convert to a contract violation depending on ``propagate_python_errors``.
"""
from __future__ import annotations

import json
import posixpath
import threading
from pathlib import Path

import pytest

from pm_workflows.kernel import Kernel
from pm_workflows.runtime import WorkflowRuntime, WorkflowTerminated
from pm_workflows.working_target import CommandResult

TASK_ID = "T-rt"


class MemoryEnvironment:
    """A minimal real working environment backed by a dict."""

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


MANIFEST = """\
---
name: runtime-t
driver: {kind: python}
human_resolver: {mode: forbid}
roles:
  role:
    skill: skills/native/role.py
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


def _write(tmp_path: Path, role_code: str) -> Path:
    base = tmp_path / "base"
    (base / "skills" / "native").mkdir(parents=True)
    (base / "skills" / "native" / "role.py").write_text(role_code, encoding="utf-8")
    (base / "w.workflow.md").write_text(MANIFEST, encoding="utf-8")
    return base


def _kernel(tmp_path: Path, base: Path, runtime: WorkflowRuntime) -> Kernel:
    return Kernel(
        manifest_path=base / "w.workflow.md",
        workspace=tmp_path / "repo",
        task_id=TASK_ID,
        base_dir=base,
        kernel_data_root=tmp_path / "kernel_data",
        runtime=runtime,
    )


def _journal(tmp_path: Path) -> list[dict]:
    path = tmp_path / "kernel_data" / TASK_ID / "journal.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_runtime_role_reads_and_writes_environment(tmp_path: Path) -> None:
    base = _write(
        tmp_path,
        "def run(context):\n"
        "    env = context.runtime.environment\n"
        "    value = env.read_text('inbox/request.txt')\n"
        "    env.write_text('outbox/report.txt', f'got:{value}')\n"
        "    return {'status': 'done', 'summary': 'wrote report'}\n",
    )
    environment = MemoryEnvironment()
    environment.write_text("inbox/request.txt", "incoming\n")
    runtime = WorkflowRuntime(
        environment=environment, stop_event=threading.Event(), tools={},
    )

    result = _kernel(tmp_path, base, runtime).run()

    assert result["terminal_status"] == "done"
    assert environment.read_text("outbox/report.txt") == "got:incoming\n"


def test_runtime_cancellation_terminates_without_retry(tmp_path: Path) -> None:
    base = _write(
        tmp_path,
        "def run(context):\n"
        "    context.runtime.stop_event.set()\n"
        "    context.runtime.check_cancelled()\n"
        "    return {'status': 'done', 'summary': 'never reached'}\n",
    )
    runtime = WorkflowRuntime(
        environment=MemoryEnvironment(), stop_event=threading.Event(), tools={},
    )

    kernel = _kernel(tmp_path, base, runtime)
    result = kernel.run()
    repeated = kernel.step()
    assert repeated.terminal_status == "terminated"
    assert repeated.workflow_ok is False

    assert result["ok"] is False
    assert result["exit_reason"] == "terminated"
    assert result["terminal_status"] == "terminated"
    entries = _journal(tmp_path)
    terminations = [e for e in entries if e.get("kind") == "termination"]
    assert len(terminations) == 1
    assert terminations[0]["status"] == "terminated"
    done = [e for e in entries if e.get("kind") == "role" and e.get("verdict") == "done"]
    assert done == []


def test_runtime_propagates_python_error(tmp_path: Path) -> None:
    base = _write(
        tmp_path,
        "def run(context):\n    raise ValueError('boom')\n",
    )
    runtime = WorkflowRuntime(
        environment=MemoryEnvironment(),
        stop_event=threading.Event(),
        tools={},
        propagate_python_errors=True,
    )

    with pytest.raises(ValueError, match="boom"):
        _kernel(tmp_path, base, runtime).run()


def test_runtime_legacy_exception_conversion(tmp_path: Path) -> None:
    base = _write(
        tmp_path,
        "def run(context):\n    raise ValueError('boom')\n",
    )
    runtime = WorkflowRuntime(
        environment=MemoryEnvironment(),
        stop_event=threading.Event(),
        tools={},
        propagate_python_errors=False,
    )

    result = _kernel(tmp_path, base, runtime).run()

    assert result["ok"] is False
    role_entries = [e for e in _journal(tmp_path) if e.get("kind") == "role"]
    assert role_entries and role_entries[-1]["ok"] is False
    assert "ValueError: boom" in role_entries[-1]["errors"][0]
