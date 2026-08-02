from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pm_workflows.kernel import Kernel
from pm_workflows.protocol import AgentResult


class StubDriver:
    kind = "stub"

    def __init__(self, results: list[dict | str], on_call=None) -> None:
        self.results = list(results)
        self.on_call = on_call
        self.calls = 0
        self.prompts: list[str] = []

    def run_session(
        self, run_id, attempt, skill, prompt, work_dir, tools=None,
        result_file=None, trace_file=None,
    ) -> AgentResult:
        self.calls += 1
        self.prompts.append(prompt)
        if self.on_call:
            self.on_call(Path(work_dir), self.calls)
        value = self.results.pop(0)
        if isinstance(value, str):
            return AgentResult(exit_code=0, stdout=value, result_json=None)
        return AgentResult(exit_code=0, stdout=json.dumps(value), result_json=value)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "work"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    return repo


def _base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    (base / "workflows").mkdir(parents=True)
    (base / "skills" / "x").mkdir(parents=True)
    (base / "skills" / "x" / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    return base


def test_external_policy_retains_workspace_and_archives_attempt_receipts(
    tmp_path: Path,
) -> None:
    base = _base(tmp_path)
    repo = _repo(tmp_path)
    manifest = base / "workflows" / "external.workflow.md"
    manifest.write_text(
        """---
name: external
driver: {kind: stub}
checkpoint_backend: {kind: git, repo_path: "${TARGET}"}
state_policy: {mutation_model: external}
roles:
  actor:
    skill: skills/x/SKILL.md
    result_contract:
      schema: {status: {enum: [done]}}
phases:
  - name: act
    kind: role
    role: actor
    on_status: {done: stop}
    on_invalid: {action: retry_with_feedback, target: act, max_attempts: 2}
---
""",
        encoding="utf-8",
    )

    def mutate(workspace: Path, call: int) -> None:
        marker = workspace / "world-evidence.txt"
        previous = marker.read_text(encoding="utf-8") if marker.exists() else ""
        marker.write_text(previous + f"attempt {call}\n", encoding="utf-8")

    driver = StubDriver(["not json", {"status": "done"}], on_call=mutate)
    kernel = Kernel(
        manifest, repo, "external-1", base_dir=base,
        kernel_data_root=tmp_path / "private", driver=driver,
    )
    kernel.run()

    assert (repo / "world-evidence.txt").read_text(encoding="utf-8") == (
        "attempt 1\nattempt 2\n"
    )
    assert "Previous files and external-world effects were retained" in driver.prompts[1]
    assert (kernel.kernel_data / "attempts" / "act" / "attempt-0001" / "receipt.json").is_file()
    assert not any(
        entry.get("kind") == "revert" for entry in kernel.journal.read_all()
    )


def _write_child_workflows(base: Path, *, parent_budgets: str = "max_depth: 2") -> Path:
    child_dir = base / "workflows" / "leaf"
    child_dir.mkdir(parents=True)
    (child_dir / "leaf.workflow.md").write_text(
        """---
name: leaf
driver: {kind: stub}
checkpoint_backend: null
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      schema: {status: {enum: [pass, blocked]}}
phases:
  - name: work
    kind: role
    role: worker
    on_status: {pass: stop, blocked: stop}
    on_invalid: {action: stop}
---
""",
        encoding="utf-8",
    )
    parent = base / "workflows" / "parent.workflow.md"
    parent.write_text(
        f"""---
name: parent
driver: {{kind: stub}}
checkpoint_backend: null
budgets: {{{parent_budgets}}}
phases:
  - name: child
    kind: workflow
    workflow: leaf
    task: {{id: "${{TASK_ID}}.leaf", input: {{request: literal}}}}
    limits: {{decrement_depth: 1, max_attempts: 2}}
    result:
      statuses: [completed, blocked, decomposition_limit]
      status_map: {{pass: completed}}
    on_status: {{completed: stop, blocked: stop, decomposition_limit: stop}}
    on_invalid: {{action: retry_child_clean, target: child, max_attempts: 2}}
---
""",
        encoding="utf-8",
    )
    return parent


def test_child_workflow_maps_terminal_status_and_writes_receipt(tmp_path: Path) -> None:
    base = _base(tmp_path)
    parent = _write_child_workflows(base)
    repo = _repo(tmp_path)
    driver = StubDriver([{"status": "pass"}])
    kernel = Kernel(
        parent, repo, "parent-1", base_dir=base,
        kernel_data_root=tmp_path / "private", driver=driver,
    )
    result = kernel.run()

    assert result["terminal_status"] == "completed"
    execution = [
        entry for entry in kernel.journal.read_all() if entry.get("kind") == "workflow"
    ][0]
    assert execution["status"] == "completed"
    assert execution["result"]["children"][0]["raw_status"] == "pass"
    assert (kernel.kernel_data / "child-receipts" / "child" / "attempt-0001.json").is_file()


def test_child_manifest_resolves_when_base_is_the_workflow_root(tmp_path: Path) -> None:
    base = tmp_path / "library"
    parent = _write_child_workflows(base)
    workflow_root = base / "workflows"
    kernel = Kernel(
        parent,
        tmp_path / "repo",
        "parent-root-layout",
        base_dir=workflow_root,
        kernel_data_root=tmp_path / "kernel",
        driver=StubDriver([{"status": "pass", "artifact": "result.md"}]),
    )

    resolved = kernel._resolve_child_manifest("leaf")

    assert resolved == (workflow_root / "leaf" / "leaf.workflow.md").resolve()


def test_completed_child_invocation_is_reused_after_parent_crash_window(
    tmp_path: Path,
) -> None:
    base = _base(tmp_path)
    parent = _write_child_workflows(base)
    repo = _repo(tmp_path)
    driver = StubDriver([{"status": "pass"}])
    kernel = Kernel(
        parent, repo, "parent-1", base_dir=base,
        kernel_data_root=tmp_path / "private", driver=driver,
    )
    phase = kernel.manifest.phases[0]

    first = kernel._run_child(phase, attempt=1, index=1, item=None, depth_remaining=1)
    second = kernel._run_child(phase, attempt=1, index=1, item=None, depth_remaining=1)

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert second["reused"] is True
    assert driver.calls == 1


def test_depth_exhaustion_is_a_typed_child_result(tmp_path: Path) -> None:
    base = _base(tmp_path)
    parent = _write_child_workflows(base, parent_budgets="max_depth: 0")
    repo = _repo(tmp_path)
    driver = StubDriver([])
    kernel = Kernel(
        parent, repo, "parent-1", base_dir=base,
        kernel_data_root=tmp_path / "private", driver=driver,
    )
    result = kernel.run()

    assert result["terminal_status"] == "decomposition_limit"
    assert driver.calls == 0


def test_foreach_orders_stable_ids_and_invokes_one_fresh_child_each(
    tmp_path: Path,
) -> None:
    base = _base(tmp_path)
    _write_child_workflows(base)
    parent = base / "workflows" / "fan.workflow.md"
    parent.write_text(
        """---
name: fan
driver: {kind: stub}
checkpoint_backend: null
budgets: {max_depth: 2}
phases:
  - name: children
    kind: workflow
    workflow: leaf
    foreach:
      from: "${TASK_DIR}/items.json#/items"
      item: thing
      stable_id: thing.id
      max_items: 4
    task: {id: "${TASK_ID}.${thing.id}", input: {thing: "${thing}"}}
    limits: {decrement_depth: 1, max_attempts: 1}
    result:
      statuses: [completed, blocked]
      status_map: {pass: completed}
      aggregate: all_children
    on_status: {completed: stop, blocked: stop}
    on_invalid: {action: stop}
---
""",
        encoding="utf-8",
    )
    repo = _repo(tmp_path)
    task_dir = repo / "agents" / "tasks" / "fan-1"
    task_dir.mkdir(parents=True)
    (task_dir / "items.json").write_text(
        json.dumps({"items": [{"id": "b"}, {"id": "a"}]}), encoding="utf-8"
    )
    driver = StubDriver([{"status": "pass"}, {"status": "pass"}])
    kernel = Kernel(
        parent, repo, "fan-1", base_dir=base,
        kernel_data_root=tmp_path / "private", driver=driver,
    )
    result = kernel.run()

    assert result["terminal_status"] == "completed"
    assert '"id": "a"' in driver.prompts[0]
    assert '"id": "b"' in driver.prompts[1]


def test_recursive_child_task_ids_are_bounded_without_changing_short_ids(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "repo" / "agents" / "tasks"

    short = Kernel._child_task_id(tasks_root, "parent.leaf", 1, "leaf-a")
    assert short == "parent.leaf.__attempt_0001.__item_leaf-a"

    configured = ".".join(["parent"] + ["nested-child-with-a-readable-name"] * 12)
    first = Kernel._child_task_id(tasks_root, configured, 1, "stable-a")
    repeated = Kernel._child_task_id(tasks_root, configured, 1, "stable-a")
    next_attempt = Kernel._child_task_id(tasks_root, configured, 2, "stable-a")
    other_item = Kernel._child_task_id(tasks_root, configured, 1, "stable-b")

    assert first == repeated
    assert first != next_attempt
    assert first != other_item
    assert len(first) <= 120
    assert len(str(tasks_root / first)) <= 220
    assert ".__h_" in first


def test_child_mcp_allowlist_is_materialized_for_the_driver(tmp_path: Path) -> None:
    class McpDriver(StubDriver):
        supports_explicit_mcp_config = True

        def __init__(self) -> None:
            super().__init__([{"status": "done"}])
            self.servers: list[str] = []

        def run_session(self, *args, mcp_config=None, **kwargs) -> AgentResult:
            payload = json.loads(Path(mcp_config).read_text(encoding="utf-8"))
            self.servers = sorted(payload["mcpServers"])
            return super().run_session(*args, **kwargs)

    base = _base(tmp_path)
    manifest = base / "workflows" / "mcp.workflow.md"
    manifest.write_text(
        """---
name: mcp
driver: {kind: stub}
checkpoint_backend: null
roles:
  actor:
    skill: skills/x/SKILL.md
    mcp: [minecraft]
    result_contract:
      schema: {status: {enum: [done]}}
phases:
  - name: act
    kind: role
    role: actor
    on_status: {done: stop}
    on_invalid: {action: stop}
---
""",
        encoding="utf-8",
    )
    repo = _repo(tmp_path)
    (repo / ".mcp.json").write_text(
        json.dumps({"mcpServers": {
            "minecraft": {"url": "http://127.0.0.1:1/mcp"},
            "unrelated": {"command": "do-not-expose"},
        }}),
        encoding="utf-8",
    )
    driver = McpDriver()
    Kernel(
        manifest, repo, "mcp-1", base_dir=base,
        kernel_data_root=tmp_path / "private", driver=driver,
        allowed_mcp={"minecraft"}, require_http_mcp=False,
    ).run()

    assert driver.servers == ["minecraft"]
