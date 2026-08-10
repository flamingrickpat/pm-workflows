from __future__ import annotations

import json
from pathlib import Path

import pytest

from pm_workflows import (
    AgentResult,
    Kernel,
    ManifestError,
    PhaseExtensionRegistry,
    PhaseKindExtension,
    StepResult,
    parse_workflow,
)


class StubDriver:
    kind = "stub"

    def __init__(self, results: list[dict] | None = None) -> None:
        self.results = list(results or [])
        self.prompts: list[str] = []
        self.skills: list[str] = []

    def run_session(
        self,
        run_id,
        attempt,
        skill,
        prompt,
        work_dir,
        tools=None,
        result_file=None,
        trace_file=None,
    ) -> AgentResult:
        self.prompts.append(prompt)
        self.skills.append(Path(skill).read_text(encoding="utf-8"))
        value = self.results.pop(0)
        return AgentResult(
            exit_code=0,
            stdout=json.dumps(value),
            result_json=value,
        )


def make_base(tmp_path: Path, manifest_text: str) -> tuple[Path, Path, Path]:
    base = tmp_path / "base"
    workspace = tmp_path / "workspace"
    manifest = base / "workflows" / "test.workflow.md"
    manifest.parent.mkdir(parents=True)
    (base / "skills" / "worker").mkdir(parents=True)
    (base / "skills" / "worker" / "SKILL.md").write_text(
        "original skill\n", encoding="utf-8"
    )
    (base / "scripts").mkdir()
    (base / "scripts" / "good.py").write_text(
        "import json; print(json.dumps({'ok': True, 'errors': []}))\n",
        encoding="utf-8",
    )
    manifest.write_text(manifest_text, encoding="utf-8")
    workspace.mkdir()
    return base, workspace, manifest


TWO_PHASE = """---
name: step-test
driver: {kind: stub}
checkpoint_backend: null
roles:
  worker:
    skill: skills/worker/SKILL.md
    result_contract:
      schema: {status: {enum: [done]}}
phases:
  - name: work
    kind: role
    role: worker
    on_status: {done: check}
    on_invalid: {action: retry_with_feedback, target: work, max_attempts: 2}
  - name: check
    kind: gate
    predicate: scripts/good.py
    on_pass: stop
    on_fail: {action: retry_with_feedback, target: work, max_attempts: 2}
---
"""


def make_kernel(
    base: Path,
    workspace: Path,
    manifest: Path,
    private: Path,
    driver: StubDriver,
    **kwargs,
) -> Kernel:
    return Kernel(
        manifest_path=manifest,
        workspace=workspace,
        task_id="task-1",
        task_text="test stepping",
        base_dir=base,
        kernel_data_root=private,
        run_id="run-1",
        driver=driver,
        **kwargs,
    )


def test_step_executes_exactly_one_phase(tmp_path: Path) -> None:
    base, workspace, manifest = make_base(tmp_path, TWO_PHASE)
    kernel = make_kernel(
        base, workspace, manifest, tmp_path / "private", StubDriver([{"status": "done"}])
    )

    assert kernel.pending_phase_name == "work"
    first = kernel.step()
    assert isinstance(first, StepResult)
    assert first.phase == "work"
    assert kernel.pending_phase_name == "check"
    assert first.kind == "role"
    assert first.next_phase == "check"
    assert first.disposition == "continue"
    assert first.attempt == 1

    second = kernel.step()
    assert second.phase == "check"
    assert second.valid is True
    assert second.terminal is True
    assert second.exit_reason == "check: stop"

    repeated = kernel.step()
    assert repeated.phase is None
    assert repeated.terminal is True


def test_fresh_kernel_resumes_at_next_phase(tmp_path: Path) -> None:
    base, workspace, manifest = make_base(tmp_path, TWO_PHASE)
    private = tmp_path / "private"
    first_kernel = make_kernel(
        base, workspace, manifest, private, StubDriver([{"status": "done"}])
    )
    assert first_kernel.step().next_phase == "check"

    resumed = make_kernel(base, workspace, manifest, private, StubDriver([]))
    boundary = resumed.step()
    assert boundary.phase == "check"
    assert boundary.terminal is True
    assert resumed.journal.attempts_for_phase("work") == 1
    assert resumed.journal.attempts_for_phase("check") == 1


def test_run_uses_step_and_preserves_summary_shape(tmp_path: Path) -> None:
    base, workspace, manifest = make_base(tmp_path, TWO_PHASE)
    kernel = make_kernel(
        base, workspace, manifest, tmp_path / "private", StubDriver([{"status": "done"}])
    )
    summary = kernel.run()
    assert summary["task_id"] == "task-1"
    assert summary["workflow"] == "step-test"
    assert summary["journal_entries"] >= 4
    assert summary["exit_reason"] == "check: stop"


def test_manifest_and_skill_are_reloaded_at_phase_boundary(tmp_path: Path) -> None:
    manifest_text = """---
name: reload-test
driver: {kind: stub}
checkpoint_backend: null
roles:
  first:
    skill: skills/worker/SKILL.md
    instruction: first instruction
    result_contract: {schema: {status: {enum: [done]}}}
  second:
    skill: skills/worker/SKILL.md
    instruction: old second instruction
    result_contract: {schema: {status: {enum: [done]}}}
phases:
  - {name: first, kind: role, role: first, on_status: {done: second}, on_invalid: {action: stop}}
  - {name: second, kind: role, role: second, on_status: {done: stop}, on_invalid: {action: stop}}
---
"""
    base, workspace, manifest = make_base(tmp_path, manifest_text)
    driver = StubDriver([{"status": "done"}, {"status": "done"}])
    kernel = make_kernel(base, workspace, manifest, tmp_path / "private", driver)
    assert kernel.step().phase == "first"

    manifest.write_text(
        manifest_text.replace("old second instruction", "new second instruction"),
        encoding="utf-8",
    )
    (base / "skills" / "worker" / "SKILL.md").write_text(
        "updated skill\n", encoding="utf-8"
    )
    assert kernel.step().phase == "second"
    assert "new second instruction" in driver.prompts[1]
    assert "updated skill" in driver.skills[1]


def test_removed_next_phase_fails_loudly(tmp_path: Path) -> None:
    base, workspace, manifest = make_base(tmp_path, TWO_PHASE)
    kernel = make_kernel(
        base, workspace, manifest, tmp_path / "private", StubDriver([{"status": "done"}])
    )
    assert kernel.step().next_phase == "check"
    manifest.write_text(TWO_PHASE.replace("name: check", "name: renamed"), encoding="utf-8")
    with pytest.raises(ManifestError):
        kernel.step()


def test_custom_phase_parser_validator_executor_and_journal(tmp_path: Path) -> None:
    manifest_text = """---
name: extension-test
driver: {kind: stub}
checkpoint_backend: null
roles: {}
phases:
  - name: native
    kind: native_test
    token: GOOD
    on_pass: stop
    on_fail: {action: stop}
---
"""
    base, workspace, manifest = make_base(tmp_path, manifest_text)
    calls: list[str] = []

    def parse(raw, path):
        return {"token": str(raw.get("token", ""))}

    def validate(phase, workflow):
        if phase.extension["token"] not in {"GOOD", "BAD"}:
            raise ManifestError("unsupported test token")

    def execute(kernel, phase):
        calls.append(phase.extension["token"])
        return {
            "valid": phase.extension["token"] == "GOOD",
            "status": "checked",
            "data": {"token": phase.extension["token"]},
            "errors": [],
        }

    registry = PhaseExtensionRegistry([
        PhaseKindExtension("native_test", execute=execute, parse=parse, validate=validate)
    ])
    kernel = make_kernel(
        base,
        workspace,
        manifest,
        tmp_path / "private",
        StubDriver([]),
        phase_extensions=registry,
    )
    boundary = kernel.step()
    assert boundary.terminal is True
    assert boundary.data == {"token": "GOOD"}
    assert calls == ["GOOD"]
    entries = kernel.journal.entries_for_phase("native")
    assert len(entries) == 1
    assert entries[0]["kind"] == "native_test"
    assert entries[0]["attempt"] == 1


def test_empty_registry_rejects_custom_phase(tmp_path: Path) -> None:
    manifest_text = """---
name: extension-test
driver: {kind: stub}
checkpoint_backend: null
roles: {}
phases:
  - {name: native, kind: native_test, on_pass: stop, on_fail: {action: stop}}
---
"""
    _, _, manifest = make_base(tmp_path, manifest_text)
    with pytest.raises(ManifestError, match="unknown kind"):
        parse_workflow(manifest)


def test_extension_can_override_builtin_role_executor(tmp_path: Path) -> None:
    base, workspace, manifest = make_base(tmp_path, TWO_PHASE)
    calls: list[str] = []

    def simulated_role(kernel, phase):
        calls.append(phase.name)
        return {"valid": True, "status": "done", "data": {}, "errors": []}

    registry = PhaseExtensionRegistry([
        PhaseKindExtension("role", execute=simulated_role)
    ])
    driver = StubDriver([])
    kernel = make_kernel(
        base,
        workspace,
        manifest,
        tmp_path / "private",
        driver,
        phase_extensions=registry,
    )
    boundary = kernel.step()
    assert boundary.phase == "work"
    assert calls == ["work"]
    assert driver.prompts == []


def test_duplicate_extension_kind_is_rejected() -> None:
    extension = PhaseKindExtension(
        "x", execute=lambda kernel, phase: {"valid": True, "data": {}, "errors": []}
    )
    with pytest.raises(ValueError, match="duplicate"):
        PhaseExtensionRegistry([extension, extension])


def test_external_suspension_returns_resume_boundary_and_is_durable(tmp_path: Path) -> None:
    manifest_text = """---
name: suspension-test
driver: {kind: stub}
checkpoint_backend: null
roles:
  worker:
    skill: skills/worker/SKILL.md
    result_contract: {schema: {status: {enum: [needs_user, done]}}}
phases:
  - name: ask
    kind: role
    role: worker
    on_status:
      needs_user: {action: suspend, waiting: user, resume_at: verify}
      done: stop
    on_invalid: {action: stop}
  - name: verify
    kind: role
    role: worker
    on_status: {needs_user: stop, done: stop}
    on_invalid: {action: stop}
---
"""
    base, workspace, manifest = make_base(tmp_path, manifest_text)
    private = tmp_path / "private"
    kernel = make_kernel(
        base,
        workspace,
        manifest,
        private,
        StubDriver([{"status": "needs_user"}]),
    )
    boundary = kernel.step()
    assert boundary.disposition == "suspend"
    assert boundary.waiting_kind == "user"
    assert boundary.next_phase == "verify"
    assert boundary.to_dict()["schema"] == "pm.step-result.v1"
    route = kernel.journal.last_route()
    assert route["verdict"] == "verify"
    assert route["result"]["action"] == "suspend"
    with pytest.raises(RuntimeError, match="suspended"):
        kernel.step()

    resumed = make_kernel(
        base,
        workspace,
        manifest,
        private,
        StubDriver([{"status": "done"}]),
    )
    final = resumed.step()
    assert final.phase == "verify"
    assert final.terminal is True


def test_invalid_suspension_route_is_rejected(tmp_path: Path) -> None:
    manifest_text = """---
name: suspension-test
driver: {kind: stub}
checkpoint_backend: null
roles:
  worker:
    skill: skills/worker/SKILL.md
    result_contract: {schema: {status: {enum: [wait]}}}
phases:
  - name: ask
    kind: role
    role: worker
    on_status: {wait: {action: suspend, waiting: forever, resume_at: ask}}
    on_invalid: {action: stop}
---
"""
    _, _, manifest = make_base(tmp_path, manifest_text)
    with pytest.raises(ManifestError, match="waiting"):
        parse_workflow(manifest)


def test_durable_external_answer_is_added_to_next_role_once(tmp_path: Path) -> None:
    base, workspace, manifest = make_base(tmp_path, TWO_PHASE)
    private = tmp_path / "private"
    answers = private / "run-1" / "answers"
    answers.mkdir(parents=True)
    (answers / "answer-1.yaml").write_text(
        """schema: pm.answer-receipt.v1
id: answer-1
resume_phase: work
answer: The external condition is resolved.
""",
        encoding="utf-8",
    )
    driver = StubDriver([{"status": "done"}])
    kernel = make_kernel(base, workspace, manifest, private, driver)
    kernel.step()
    assert "The external condition is resolved." in driver.prompts[0]
    consumed = [
        entry for entry in kernel.journal.read_all()
        if entry.get("kind") == "external_answer"
    ]
    assert [entry["verdict"] for entry in consumed] == ["answer-1"]


def test_first_role_after_fatal_recovery_receives_one_explicit_notice(
    tmp_path: Path,
) -> None:
    manifest_text = """---
name: recovery-notice
driver: {kind: stub}
checkpoint_backend: null
roles:
  worker:
    skill: skills/worker/SKILL.md
    result_contract: {schema: {status: {enum: [done]}}}
phases:
  - name: first
    kind: role
    role: worker
    on_status: {done: second}
    on_invalid: {action: stop}
  - name: second
    kind: role
    role: worker
    on_status: {done: stop}
    on_invalid: {action: stop}
---
"""
    base, workspace, manifest = make_base(tmp_path, manifest_text)
    driver = StubDriver([{"status": "done"}, {"status": "done"}])
    kernel = make_kernel(base, workspace, manifest, tmp_path / "private", driver)
    kernel.journal.append_recovery(
        restored_revision="retained-external-state",
        resume_phase="first",
        active_through_entry=0,
        abandoned_entry_range="none",
        external_state_retained=True,
    )
    kernel.step()
    kernel.step()
    assert "BEGIN SUPERVISOR RECOVERY NOTICE" in driver.prompts[0]
    assert "External state can differ" in driver.prompts[0]
    assert "BEGIN SUPERVISOR RECOVERY NOTICE" not in driver.prompts[1]
