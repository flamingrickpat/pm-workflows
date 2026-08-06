"""A role whose skill is a `.py` file runs in-process, not through an agent.

The kernel treats its returned dict exactly like an agentic role's final JSON
message: same contract validation, same journal entries, same routing.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pm_workflows.kernel import Kernel
from pm_workflows.protocol import AgentResult

TASK_ID = "T-py"


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=True,
    )
    return result.stdout.strip()


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "work")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "test")
    (repo / "product.txt").write_text("base\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "initial")
    return repo


def make_base(tmp_path: Path, workflow: str, python_skill: str) -> Path:
    base = tmp_path / "base"
    (base / "coding").mkdir(parents=True)
    (base / "skills" / "native").mkdir(parents=True)
    (base / "coding" / "w.workflow.md").write_text(workflow, encoding="utf-8")
    (base / "skills" / "native" / "role.py").write_text(python_skill, encoding="utf-8")
    return base


class StubDriver:
    """An agentic driver that must never be asked to run a `.py` skill."""

    kind = "stub"

    def __init__(self, results: list[dict]) -> None:
        self.results = list(results)
        self.calls = 0

    def run_session(self, run_id, attempt, skill, prompt, work_dir, tools=None,
                     result_file=None, trace_file=None) -> AgentResult:
        assert not str(skill).endswith(".py"), (
            "a python-role skill must never reach the agentic driver"
        )
        self.calls += 1
        value = self.results.pop(0)
        return AgentResult(exit_code=0, stdout=json.dumps(value), result_json=value)


def run_kernel(base: Path, repo: Path, driver, tmp_path: Path, **kwargs) -> dict:
    kernel = Kernel(
        manifest_path=base / "coding" / "w.workflow.md",
        workspace=repo,
        task_id=TASK_ID,
        task_text="do the thing",
        base_dir=base,
        kernel_data_root=tmp_path / "kernel_data",
        driver=driver,
        **kwargs,
    )
    return kernel.run()


def _journal(tmp_path: Path) -> list[dict]:
    path = tmp_path / "kernel_data" / TASK_ID / "journal.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


ONE_STEP = """\
---
name: t
driver: {kind: stub}
checkpoint_backend: {kind: git, repo_path: "${TARGET}"}
human_resolver: {mode: forbid}
roles:
  native:
    skill: skills/native/role.py
    result_contract:
      type: json
      schema:
        status: {enum: [done, again, blocked]}
        summary: string
phases:
  - name: work
    kind: role
    role: native
    checkpoint_after: true
    on_status:
      done: finish
      again: work
      blocked: stop
    on_invalid: {action: retry_with_feedback, target: work, max_attempts: 5}
  - name: finish
    kind: script
    script: scripts/noop.py
---
"""

NOOP_SCRIPT = "import json; print(json.dumps({'ok': True, 'errors': []}))\n"


def test_python_skill_runs_in_process_and_its_result_routes_normally(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    base = make_base(
        tmp_path, ONE_STEP,
        "def run(context):\n"
        "    return {'status': 'done', 'summary': 'native role ran'}\n",
    )
    (base / "scripts").mkdir()
    (base / "scripts" / "noop.py").write_text(NOOP_SCRIPT, encoding="utf-8")
    driver = StubDriver([])

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    assert driver.calls == 0
    entries = _journal(tmp_path)
    role_entries = [e for e in entries if e.get("kind") == "role"]
    assert role_entries[-1]["verdict"] == "done"
    assert role_entries[-1]["result"] == {"status": "done", "summary": "native role ran"}


def test_python_skill_receives_structured_runtime_state(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(
        tmp_path, ONE_STEP,
        "def run(context):\n"
        "    return {\n"
        "        'status': 'done',\n"
        "        'summary': (\n"
        "            f'{context.task_id}|{context.role}|{context.phase}|'\n"
        "            f'{context.workspace}|{context.task_text}'\n"
        "        ),\n"
        "    }\n",
    )
    (base / "scripts").mkdir()
    (base / "scripts" / "noop.py").write_text(NOOP_SCRIPT, encoding="utf-8")

    result = run_kernel(base, repo, StubDriver([]), tmp_path)

    assert result["ok"], result["exit_reason"]
    entries = [e for e in _journal(tmp_path) if e.get("kind") == "role"]
    summary = entries[-1]["result"]["summary"]
    assert summary == f"{TASK_ID}|native|work|{repo.resolve()}|do the thing"


def test_python_skill_can_read_and_write_the_workspace(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(
        tmp_path, ONE_STEP,
        "def run(context):\n"
        "    (context.workspace / 'from_python_role.txt').write_text('written\\n')\n"
        "    return {'status': 'done', 'summary': 'wrote a file'}\n",
    )
    (base / "scripts").mkdir()
    (base / "scripts" / "noop.py").write_text(NOOP_SCRIPT, encoding="utf-8")

    result = run_kernel(base, repo, StubDriver([]), tmp_path)

    assert result["ok"], result["exit_reason"]
    assert (repo / "from_python_role.txt").read_text(encoding="utf-8") == "written\n"


def test_python_skill_retries_with_feedback_like_any_other_role(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(
        tmp_path, ONE_STEP,
        "def run(context):\n"
        "    status = 'done' if context.attempt >= 2 else 'again'\n"
        "    return {'status': status, 'summary': f'attempt {context.attempt}'}\n",
    )
    (base / "scripts").mkdir()
    (base / "scripts" / "noop.py").write_text(NOOP_SCRIPT, encoding="utf-8")

    result = run_kernel(base, repo, StubDriver([]), tmp_path)

    assert result["ok"], result["exit_reason"]
    role_entries = [e for e in _journal(tmp_path) if e.get("kind") == "role"]
    assert [e["verdict"] for e in role_entries] == ["again", "done"]


def test_python_skill_uncaught_exception_is_a_contract_violation_not_a_crash(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    base = make_base(
        tmp_path, ONE_STEP,
        "def run(context):\n"
        "    if context.attempt == 1:\n"
        "        raise RuntimeError('transient failure')\n"
        "    return {'status': 'done', 'summary': 'recovered'}\n",
    )
    (base / "scripts").mkdir()
    (base / "scripts" / "noop.py").write_text(NOOP_SCRIPT, encoding="utf-8")

    result = run_kernel(base, repo, StubDriver([]), tmp_path)

    assert result["ok"], result["exit_reason"]
    entries = [e for e in _journal(tmp_path) if e.get("kind") == "role"]
    assert entries[0]["ok"] is False
    assert "RuntimeError: transient failure" in entries[0]["errors"][0]
    assert entries[1]["verdict"] == "done"


def test_mixed_workflow_dispatches_each_role_to_its_own_driver(tmp_path: Path) -> None:
    """An agentic role and a python role in the same workflow each run correctly."""
    repo = make_repo(tmp_path)
    workflow = """\
---
name: mixed
driver: {kind: stub}
checkpoint_backend: {kind: git, repo_path: "${TARGET}"}
human_resolver: {mode: forbid}
roles:
  agentic:
    skill: skills/x/SKILL.md
    result_contract:
      schema: {status: {enum: [done]}}
  native:
    skill: skills/native/role.py
    result_contract:
      schema: {status: {enum: [done]}}
phases:
  - name: agentic_step
    kind: role
    role: agentic
    on_status: {done: python_step}
    on_invalid: {action: stop}
  - name: python_step
    kind: role
    checkpoint_after: true
    role: native
    on_status: {done: finish}
    on_invalid: {action: stop}
  - name: finish
    kind: script
    script: scripts/noop.py
---
"""
    base = make_base(
        tmp_path, workflow,
        "def run(context):\n    return {'status': 'done'}\n",
    )
    (base / "skills" / "x").mkdir(parents=True)
    (base / "skills" / "x" / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    (base / "scripts").mkdir()
    (base / "scripts" / "noop.py").write_text(NOOP_SCRIPT, encoding="utf-8")
    driver = StubDriver([{"status": "done"}])

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    assert driver.calls == 1
    kinds = [(e["phase"], e.get("verdict")) for e in _journal(tmp_path) if e.get("kind") == "role"]
    assert ("agentic_step", "done") in kinds
    assert ("python_step", "done") in kinds
