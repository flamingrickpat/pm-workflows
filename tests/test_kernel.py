"""Kernel runtime behaviour against real git repositories.

The agent is a stub — the point is not to test a model, it is to test what the
kernel does with what a model returns. Everything else is real: real commits,
real hard resets, real journal files, real check scripts.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pm_workflows.kernel import Kernel
from pm_workflows.protocol import AgentResult
from pm_workflows.ratelimit import TokenLimitError

TASK_ID = "T-1"


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
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (repo / "product.txt").write_text("base\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "initial")
    return repo


def make_base(tmp_path: Path, workflow: str, checks: dict[str, str] | None = None) -> Path:
    """A minimal instruction base: one workflow, one skill file, some checks."""
    base = tmp_path / "base"
    (base / "coding").mkdir(parents=True)
    (base / "skills" / "x").mkdir(parents=True)
    (base / "scripts").mkdir(parents=True)
    (base / "coding" / "w.workflow.md").write_text(workflow, encoding="utf-8")
    (base / "skills" / "x" / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    for name, text in (checks or {}).items():
        (base / "scripts" / name).write_text(text, encoding="utf-8")
    return base


class StubDriver:
    """Returns queued results and records the prompt it was given."""

    def __init__(self, script: list[dict | str], on_call=None) -> None:
        self.script = list(script)
        self.prompts: list[str] = []
        self.skills: list[str] = []
        self.mcp_configs: list[Path | None] = []
        self.on_call = on_call
        self.calls = 0

    def run_session(self, run_id, attempt, skill, prompt, work_dir, tools=None,
                    result_file=None, trace_file=None, mcp_config=None) -> AgentResult:
        self.calls += 1
        self.prompts.append(prompt)
        self.skills.append(skill)
        self.mcp_configs.append(Path(mcp_config) if mcp_config else None)
        if self.on_call is not None:
            self.on_call(Path(work_dir), self.calls, prompt)
        payload = self.script.pop(0) if self.script else {"status": "done"}
        if isinstance(payload, str):
            # Not JSON at all: a contract violation.
            return AgentResult(exit_code=0, stdout=payload, result_json=None)
        return AgentResult(
            exit_code=0,
            stdout=json.dumps(payload),
            result_json=payload,
        )


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


PASSING_CHECK = "import json; print(json.dumps({'ok': True, 'errors': []}))\n"
FAILING_CHECK = (
    "import json, sys\n"
    "print(json.dumps({'ok': False, 'errors': ['the artifact is missing a Status line']}))\n"
    "sys.exit(1)\n"
)

TWO_STEP = """\
---
name: t
driver: {kind: claude, model: sonnet}
checkpoint_backend: {kind: git, repo_path: "${TARGET}"}
human_resolver: {mode: forbid}
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [done, again]}
        summary: string
phases:
  - name: work
    kind: role
    role: worker
    on_status:
      done: check
      again: work
    on_invalid: {action: retry_with_feedback, target: work, max_attempts: 5}
  - name: check
    kind: gate
    checkpoint_after: true
    predicate: scripts/check.py
    on_pass: finish
    on_fail: {action: retry_with_feedback, target: work, max_attempts: 3}
  - name: finish
    kind: script
    script: scripts/check.py
---
"""


REVIEW_PENDING_CANDIDATE = """\
---
name: review-pending-candidate
driver: {kind: claude, model: sonnet}
checkpoint_backend: {kind: git, repo_path: "${TARGET}"}
human_resolver: {mode: forbid}
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
  reviewer:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [pass]}
phases:
  - name: work
    kind: role
    role: worker
    on_status: {done: check}
    on_invalid: {action: stop}
  - name: check
    kind: gate
    predicate: scripts/check.py
    on_pass: review
    on_fail: {action: stop}
  - name: review
    kind: role
    role: reviewer
    checkpoint_after: true
    on_status: {pass: finish}
    on_invalid: {action: retry_with_feedback, target: review, preserve_candidate: true}
  - name: finish
    kind: script
    script: scripts/check.py
---
"""


def test_declared_status_routes_to_the_declared_phase(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, TWO_STEP, {"check.py": PASSING_CHECK})
    driver = StubDriver([{"status": "done", "summary": "did it"}])

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    assert driver.calls == 1
    kinds = [(e["phase"], e["kind"], e.get("verdict")) for e in _journal(tmp_path)]
    assert ("work", "role", "done") in kinds
    assert ("check", "gate", "pass") in kinds


def test_auto_repair_uses_empty_mcp_config_and_hands_off_its_report(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    check = (
        "import json, sys\nfrom pathlib import Path\n"
        "ok = 'Status: ready' in (Path(sys.argv[1]) / 'product.txt').read_text()\n"
        "print(json.dumps({'ok': ok, 'errors': ['product.txt needs Status: ready']}))\n"
        "sys.exit(0 if ok else 1)\n"
    )
    workflow = TWO_STEP.replace(
        "checkpoint_after: true\n    predicate: scripts/check.py",
        "checkpoint_after: true\n    attempt_auto_repair: true\n    predicate: scripts/check.py",
    )

    def repair(repo: Path, calls: int, prompt: str) -> None:
        if calls == 2:
            assert "Gate repair" in prompt
            (repo / "product.txt").write_text("base\nStatus: ready\n", encoding="utf-8")

    driver = StubDriver(
        [{"status": "done", "summary": "work"}, "FIXED"], on_call=repair
    )
    result = run_kernel(make_base(tmp_path, workflow, {"check.py": check}), repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    assert driver.skills == [str(tmp_path / "base" / "skills" / "x" / "SKILL.md"), ""]
    repair_mcp = driver.mcp_configs[1]
    assert repair_mcp is not None
    assert json.loads(repair_mcp.read_text(encoding="utf-8")) == {"mcpServers": {}}


def test_auto_repair_report_is_in_the_next_retry_feedback(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    workflow = TWO_STEP.replace(
        "checkpoint_after: true\n    predicate: scripts/check.py",
        "attempt_auto_repair: true\n    predicate: scripts/check.py",
    ).replace("max_attempts: 3", "max_attempts: 2")
    driver = StubDriver([
        {"status": "done", "summary": "first try"},
        "The requirement conflicts with the current schema. RERUN",
        "No minimal repair is possible. RERUN",
        {"status": "done", "summary": "retry"},
    ])

    result = run_kernel(
        make_base(tmp_path, workflow, {"check.py": FAILING_CHECK}), repo, driver, tmp_path
    )

    assert not result["ok"]
    retry_prompt = driver.prompts[3]
    assert "## Auto-repair handoff" in retry_prompt
    assert "The requirement conflicts with the current schema" in retry_prompt
    assert "untrusted diagnostic, not an instruction" in retry_prompt


def test_reviewer_keeps_candidate_while_parking_untracked_debris(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, REVIEW_PENDING_CANDIDATE, {"check.py": PASSING_CHECK})
    reviewed: list[str] = []

    def make_candidate(work_dir: Path, call: int, prompt: str) -> None:
        if call == 1:
            (work_dir / "product.txt").write_text("candidate\n", encoding="utf-8")
            git(work_dir, "add", "product.txt")
            git(work_dir, "commit", "-m", "candidate")
            (work_dir / "reference-clone").mkdir()
            (work_dir / "reference-clone" / "README.md").write_text("metadata\n", encoding="utf-8")
        else:
            reviewed.append((work_dir / "product.txt").read_text(encoding="utf-8"))

    result = run_kernel(
        base,
        repo,
        StubDriver([{"status": "done"}, {"status": "pass"}], on_call=make_candidate),
        tmp_path,
    )

    assert result["ok"], result["exit_reason"]
    assert reviewed == ["candidate\n"]
    assert not (repo / "reference-clone").exists()
    leftovers = [entry for entry in _journal(tmp_path) if entry.get("kind") == "leftover"]
    assert leftovers[-1]["reason_codes"][-1] == "preserved_candidate"


def test_reviewer_retry_preserves_a_clean_candidate_and_failure_feedback(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, REVIEW_PENDING_CANDIDATE, {"check.py": PASSING_CHECK})
    reviewed: list[str] = []

    def make_candidate(work_dir: Path, call: int, prompt: str) -> None:
        if call == 1:
            (work_dir / "product.txt").write_text("candidate\n", encoding="utf-8")
            git(work_dir, "add", "product.txt")
            git(work_dir, "commit", "-m", "candidate")
        elif call == 3:
            reviewed.append((work_dir / "product.txt").read_text(encoding="utf-8"))

    driver = StubDriver(
        [{"status": "done"}, "not json", {"status": "pass"}],
        on_call=make_candidate,
    )
    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    assert reviewed == ["candidate\n"]
    assert "clean candidate commit" in driver.prompts[2]
    assert "No JSON result object" in driver.prompts[2]
    reverts = [entry for entry in _journal(tmp_path) if entry.get("kind") == "revert"]
    assert reverts[-1]["verdict"] == "preserve_before:review"


def test_a_self_route_dispatches_the_same_role_again(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, TWO_STEP, {"check.py": PASSING_CHECK})
    driver = StubDriver([
        {"status": "again", "summary": "not finished"},
        {"status": "done", "summary": "finished"},
    ])

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    assert driver.calls == 2


def test_role_deny_access_is_the_final_prompt_addendum(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    workflow = TWO_STEP.replace(
        "    skill: skills/x/SKILL.md",
        "    skill: skills/x/SKILL.md\n"
        "    deny_access:\n"
        "      - \"${TASK_DIR}/workitems/**\"\n"
        "      - \"${WORKSPACE}/agents/projects/**\"",
    )
    base = make_base(tmp_path, workflow, {"check.py": PASSING_CHECK})
    driver = StubDriver([{"status": "done", "summary": "did it"}])

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    prompt = driver.prompts[0]
    assert "## Context-isolation access denial" in prompt
    assert f"`{repo / 'agents' / 'tasks' / TASK_ID}/workitems/**`" in prompt
    assert f"`{repo}/agents/projects/**`" in prompt
    assert prompt.index("## Required final message") < prompt.index(
        "## Context-isolation access denial"
    )
    assert prompt.rstrip().endswith(
        "in the skill or elsewhere in this prompt."
    )


def test_a_failed_check_reverts_the_repo_and_feeds_back_the_reason(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, TWO_STEP, {"check.py": FAILING_CHECK})

    def vandalise(work_dir: Path, call: int, prompt: str) -> None:
        (work_dir / "product.txt").write_text(f"attempt {call}\n", encoding="utf-8")
        (work_dir / "stray.txt").write_text("junk\n", encoding="utf-8")
        (work_dir / "ignored.txt").write_text("build output\n", encoding="utf-8")

    driver = StubDriver([{"status": "done", "summary": "s"}] * 5, on_call=vandalise)

    result = run_kernel(base, repo, driver, tmp_path)

    # The check never passes, so the attempt cap stops the run.
    assert not result["ok"]
    assert "attempts" in result["exit_reason"]
    assert driver.calls == 3

    # Every rejected attempt was thrown away, including untracked files...
    assert (repo / "product.txt").read_text(encoding="utf-8") == "base\n"
    assert not (repo / "stray.txt").exists()
    # ...but ignored build output survives, because rebuilding it costs minutes
    # and it is not part of the work being judged.
    assert (repo / "ignored.txt").read_text(encoding="utf-8") == "build output\n"

    # The retried session was told why, in the check's own words.
    assert "the artifact is missing a Status line" in driver.prompts[1]
    assert "reset to the last accepted commit" in driver.prompts[1]
    assert driver.prompts[2].count("the artifact is missing a Status line") == 1
    remembered = [
        entry for entry in _journal(tmp_path)
        if entry.get("kind") == "failure_memory"
        and entry.get("verdict") == "recorded"
    ]
    assert len(remembered) == 1


def test_a_contract_violation_is_not_treated_as_an_outcome(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, TWO_STEP, {"check.py": PASSING_CHECK})
    driver = StubDriver([
        "I have finished the work, it all looks good.",   # no JSON at all
        {"status": "nonsense"},                            # not in the enum
        {"status": "done"},                                # missing `summary`
        {"status": "done", "summary": "ok"},
    ])

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    assert driver.calls == 4
    violations = [
        entry for entry in _journal(tmp_path)
        if entry["phase"] == "work" and entry.get("verdict") == "contract_violation"
    ]
    assert len(violations) == 3
    reasons = " ".join(" ".join(entry.get("errors", [])) for entry in violations)
    assert "final message" in reasons
    assert "not one of the allowed" in reasons
    assert "missing required field" in reasons
    assert "final message" in driver.prompts[1]
    assert "final message" in driver.prompts[2]
    assert "not one of the allowed" in driver.prompts[2]
    assert "final message" in driver.prompts[3]
    assert "not one of the allowed" in driver.prompts[3]
    assert "missing required field" in driver.prompts[3]


LOOP = """\
---
name: t
driver: {kind: claude, model: sonnet}
checkpoint_backend: {kind: git, repo_path: "${TARGET}"}
human_resolver: {mode: forbid}
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [item_done, item_again]}
phases:
  - name: items
    kind: loop
    iterator_source: pending_work_items
    exit: finish
    body:
      - name: do_item
        kind: role
        role: worker
        checkpoint_after: true
        on_status:
          item_done: next_item
          item_again: do_item
        on_invalid: {action: retry_with_feedback, target: do_item, max_attempts: 2}
  - name: finish
    kind: script
    script: scripts/check.py
---
"""


def _work_items(repo: Path, *names: str) -> None:
    directory = repo / "agents" / "tasks" / TASK_ID / "workitems"
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text(f"# {name}\n\nAcceptance: do it.\n", encoding="utf-8")


def test_the_loop_walks_work_items_in_filename_order(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    _work_items(repo, "WI-02-second.md", "WI-01-first.md", "WI-03-third.md")
    base = make_base(tmp_path, LOOP, {"check.py": PASSING_CHECK})
    driver = StubDriver([{"status": "item_done"}] * 3)

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    completed = [
        Path(entry["item"]).name for entry in _journal(tmp_path)
        if entry["kind"] == "item_complete"
    ]
    assert completed == ["WI-01-first.md", "WI-02-second.md", "WI-03-third.md"]


def test_the_selected_work_item_is_in_the_prompt(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    _work_items(repo, "WI-01-first.md")
    base = make_base(tmp_path, LOOP, {"check.py": PASSING_CHECK})
    driver = StubDriver([{"status": "item_done"}])

    run_kernel(base, repo, driver, tmp_path)

    assert "WI-01-first.md" in driver.prompts[0]
    assert "Acceptance: do it." in driver.prompts[0]


def test_a_work_item_added_mid_run_is_picked_up(tmp_path: Path) -> None:
    """Fix planning adds work items after a review. The loop recomputes pending
    items on every entry, so nothing extra has to be wired for that."""
    repo = make_repo(tmp_path)
    _work_items(repo, "WI-01-first.md")
    base = make_base(tmp_path, LOOP, {"check.py": PASSING_CHECK})

    def add_second(work_dir: Path, call: int, prompt: str) -> None:
        if call == 1:
            _work_items(work_dir, "WI-02-added-later.md")

    driver = StubDriver([{"status": "item_done"}] * 2, on_call=add_second)

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    assert driver.calls == 2
    completed = [
        Path(entry["item"]).name for entry in _journal(tmp_path)
        if entry["kind"] == "item_complete"
    ]
    assert completed == ["WI-01-first.md", "WI-02-added-later.md"]


def test_an_empty_work_item_list_exits_the_loop(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, LOOP, {"check.py": PASSING_CHECK})
    driver = StubDriver([])

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    assert driver.calls == 0


HUMAN = """\
---
name: t
driver: {kind: claude, model: sonnet}
checkpoint_backend: {kind: git, repo_path: "${TARGET}"}
human_resolver: {mode: stdin}
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [done, needs_user]}
        question: string
  triage:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [answered]}
phases:
  - name: work
    kind: role
    role: worker
    checkpoint_after: true
    on_status:
      done: finish
      needs_user: ask
    on_invalid: {action: retry_with_feedback, target: work, max_attempts: 2}
  - name: ask
    kind: human
    question: fallback question
    question_from_result: question
    next: answered
  - name: answered
    kind: role
    role: triage
    on_status:
      answered: work
    on_invalid: {action: retry_with_feedback, target: answered, max_attempts: 2}
  - name: finish
    kind: script
    script: scripts/check.py
---
"""


def test_a_role_question_reaches_the_user_and_the_answer_reaches_the_role(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, HUMAN, {"check.py": PASSING_CHECK})
    typed = iter(["Use the second reading.", "It is the documented one.", "."])
    monkeypatch.setattr("builtins.input", lambda: next(typed))

    driver = StubDriver([
        {"status": "needs_user", "question": "Which reading of the ticket is right?"},
        {"status": "answered"},
        {"status": "done", "question": ""},
    ])

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    # The role's own words were shown, not a paraphrase.
    assert "Which reading of the ticket is right?" in capsys.readouterr().out
    # The user's own words reached the next session verbatim.
    assert "Use the second reading.\nIt is the documented one." in driver.prompts[1]
    assert "BEGIN USER ANSWER" in driver.prompts[1]
    answers = [e["answer"] for e in _journal(tmp_path) if e["kind"] == "human"]
    assert answers == ["Use the second reading.\nIt is the documented one."]


def test_a_killed_run_continues_at_the_phase_that_was_in_flight(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, TWO_STEP, {"check.py": PASSING_CHECK})

    class Killed(RuntimeError):
        pass

    def die_in_the_check(work_dir: Path, call: int, prompt: str) -> None:
        (work_dir / "product.txt").write_text("half done\n", encoding="utf-8")

    first = StubDriver([{"status": "done", "summary": "s"}], on_call=die_in_the_check)
    kernel = Kernel(
        manifest_path=base / "coding" / "w.workflow.md",
        workspace=repo, task_id=TASK_ID, task_text="t", base_dir=base,
        kernel_data_root=tmp_path / "kernel_data", driver=first,
    )
    # Reach the point where `work` succeeded and `check` is the next phase, then
    # stop as abruptly as a kill would.
    phase = kernel.manifest.phases[0]
    outcome = kernel._execute_phase(phase)
    target = kernel._route(phase, outcome)
    from pm_workflows.protocol import JournalEntry
    kernel.journal.append(JournalEntry(
        run_id=TASK_ID, phase=phase.name, kind="route", ok=True, verdict=target
    ))
    # A fresh process, same task id.
    second = StubDriver([])
    result = run_kernel(base, repo, second, tmp_path)

    assert result["ok"], result["exit_reason"]
    # It resumed at the check instead of re-dispatching the role that had
    # already succeeded, and the half-finished work was still there to check.
    assert second.calls == 0
    assert (repo / "product.txt").read_text(encoding="utf-8") == "half done\n"


def test_a_usage_limit_can_resume_with_a_different_driver(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, TWO_STEP, {"check.py": PASSING_CHECK})

    class LimitedClaude:
        kind = "claude"

        def run_session(self, *args, **kwargs):
            raise TokenLimitError("claude", "credits exhausted")

    first = run_kernel(base, repo, LimitedClaude(), tmp_path)
    assert not first["ok"]
    assert "another --coding-agent" in first["exit_reason"]
    assert any(entry["kind"] == "rate_limit" for entry in _journal(tmp_path))

    replacement = StubDriver([{"status": "done", "summary": "from pi"}])
    replacement.kind = "pi"
    second = run_kernel(base, repo, replacement, tmp_path)

    assert second["ok"], second["exit_reason"]
    assert replacement.calls == 1


def test_the_journal_and_traces_stay_out_of_the_repository(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, TWO_STEP, {"check.py": PASSING_CHECK})
    run_kernel(base, repo, StubDriver([{"status": "done", "summary": "s"}]), tmp_path)

    inside = [path.name for path in repo.rglob("*") if path.is_file()]
    assert "journal.jsonl" not in inside
    assert not list(repo.rglob("*trace*"))
    assert (tmp_path / "kernel_data" / TASK_ID / "journal.jsonl").is_file()

    # The agents get a read-only projection of the journal, and nothing else.
    state = repo / "agents" / "tasks" / TASK_ID / "state.md"
    assert state.is_file()
    assert "read-only for every role" in state.read_text(encoding="utf-8").lower() \
        or "Read-only for every role" in state.read_text(encoding="utf-8")


def _journal(tmp_path: Path) -> list[dict]:
    path = tmp_path / "kernel_data" / TASK_ID / "journal.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


CYCLE = """\
---
name: t
driver: {kind: claude, model: sonnet}
checkpoint_backend: {kind: git, repo_path: "${TARGET}"}
human_resolver: {mode: forbid}
failure_policy: {max_attempts_per_phase: 4}
roles:
  reviewer:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [findings]}
  fixer:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [nothing_to_fix]}
phases:
  - name: review
    kind: role
    role: reviewer
    on_status:
      findings: fix
    on_invalid: {action: retry_with_feedback, target: review}
  - name: fix
    kind: role
    role: fixer
    on_status:
      nothing_to_fix: review
    on_invalid: {action: retry_with_feedback, target: fix}
---
"""


def test_a_routing_cycle_is_stopped_by_the_declared_phase_budget(tmp_path: Path) -> None:
    """Retry caps only bound failures. Two phases can also ping-pong forever on
    outcomes that are each valid, burning a session per lap."""
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, CYCLE)
    driver = StubDriver([{"status": "findings"}, {"status": "nothing_to_fix"}] * 20)

    result = run_kernel(base, repo, driver, tmp_path)

    assert not result["ok"]
    assert "cycling" in result["exit_reason"]
    # 4 executions of each phase, and then it refuses to start a fifth.
    assert driver.calls == 8


ITEM_CYCLE = """\
---
name: t
driver: {kind: claude, model: sonnet}
checkpoint_backend: {kind: git, repo_path: "${TARGET}"}
human_resolver: {mode: forbid}
failure_policy: {max_attempts_per_phase: 3}
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [item_done]}
  checker:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [findings]}
phases:
  - name: items
    kind: loop
    iterator_source: pending_work_items
    max_iterations: 4
    exit: finish
    body:
      - name: do_item
        kind: role
        role: worker
        on_status:
          item_done: check_item
        on_invalid: {action: retry_with_feedback, target: do_item}
      - name: check_item
        kind: role
        role: checker
        on_status:
          findings: do_item
        on_invalid: {action: retry_with_feedback, target: check_item}
  - name: finish
    kind: script
    script: scripts/check.py
---
"""


def test_a_cycle_inside_a_loop_is_bounded_per_item(tmp_path: Path) -> None:
    """implement -> review -> findings -> implement is a cycle of valid
    outcomes. It is bounded by the loop's own max_iterations, counted per work
    item -- a workflow with fifty items legitimately runs implement fifty
    times, so a lifetime count would be wrong."""
    repo = make_repo(tmp_path)
    _work_items(repo, "WI-01-first.md", "WI-02-second.md")
    base = make_base(tmp_path, ITEM_CYCLE, {"check.py": PASSING_CHECK})
    driver = StubDriver([{"status": "item_done"}, {"status": "findings"}] * 20)

    result = run_kernel(base, repo, driver, tmp_path)

    assert not result["ok"]
    assert "cycling" in result["exit_reason"]
    assert "WI-01-first.md" in result["exit_reason"]
    # Four passes over the first item, never reaching the second.
    assert driver.calls == 8


def test_many_work_items_do_not_trip_the_top_level_phase_budget(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    _work_items(repo, *[f"WI-{n:02d}-item.md" for n in range(1, 8)])
    base = make_base(tmp_path, LOOP, {"check.py": PASSING_CHECK})
    driver = StubDriver([{"status": "item_done"}] * 7)

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    assert driver.calls == 7


ITEM_SCOPED_RETRY = """\
---
name: t
driver: {kind: claude, model: sonnet}
checkpoint_backend: {kind: git, repo_path: "${TARGET}"}
human_resolver: {mode: forbid}
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [item_done]}
phases:
  - name: items
    kind: loop
    iterator_source: pending_work_items
    exit: finish
    body:
      - name: do_item
        kind: role
        role: worker
        checkpoint_after: true
        on_status:
          item_done: next_item
        on_invalid: {action: retry_with_feedback, target: do_item, max_attempts: 2}
  - name: finish
    kind: script
    script: scripts/check.py
---
"""


def test_on_invalid_max_attempts_resets_for_each_work_item(tmp_path: Path) -> None:
    """A hard item's retries must not eat into the next item's attempt budget.

    Both items here fail their first attempt and succeed on their second, so
    each individually stays within max_attempts: 2. Before attempt counting
    was scoped to the loop item, the second item's first (failing) attempt
    would already read as the task's third attempt on this phase — over the
    declared cap of 2 — and the run would give up on it without ever retrying.
    """
    repo = make_repo(tmp_path)
    _work_items(repo, "WI-01-first.md", "WI-02-second.md")
    base = make_base(tmp_path, ITEM_SCOPED_RETRY, {"check.py": PASSING_CHECK})
    driver = StubDriver([
        "not json", {"status": "item_done"},
        "not json", {"status": "item_done"},
    ])

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    assert driver.calls == 4
    completed = [
        Path(entry["item"]).name for entry in _journal(tmp_path)
        if entry["kind"] == "item_complete"
    ]
    assert completed == ["WI-01-first.md", "WI-02-second.md"]
    attempts_by_item = [
        (Path(entry["item"]).name, entry["attempt"])
        for entry in _journal(tmp_path) if entry.get("kind") == "role"
    ]
    assert attempts_by_item == [
        ("WI-01-first.md", 1), ("WI-01-first.md", 2),
        ("WI-02-second.md", 1), ("WI-02-second.md", 2),
    ]


def test_trace_and_result_artifacts_do_not_collide_across_work_items(tmp_path: Path) -> None:
    """Attempt numbers reset per item, so paths keyed only on phase+attempt
    would otherwise have the second item silently overwrite the first item's
    trace and result artifacts."""

    class RecordingDriver:
        """A driver that records the paths the kernel handed it, like a real
        one would receive them, without depending on any real driver's own
        file-writing behaviour."""

        def __init__(self, results: list[dict]) -> None:
            self.results = list(results)
            self.seen_paths: list[tuple[Path | None, Path | None]] = []

        def run_session(self, run_id, attempt, skill, prompt, work_dir, tools=None,
                        result_file=None, trace_file=None) -> AgentResult:
            self.seen_paths.append((result_file, trace_file))
            payload = self.results.pop(0)
            return AgentResult(exit_code=0, stdout=json.dumps(payload), result_json=payload)

    repo = make_repo(tmp_path)
    _work_items(repo, "WI-01-first.md", "WI-02-second.md")
    base = make_base(tmp_path, ITEM_SCOPED_RETRY, {"check.py": PASSING_CHECK})
    driver = RecordingDriver([{"status": "item_done"}, {"status": "item_done"}])

    result = run_kernel(base, repo, driver, tmp_path)

    assert result["ok"], result["exit_reason"]
    assert len(driver.seen_paths) == 2
    result_files = [str(rf) for rf, _ in driver.seen_paths]
    trace_files = [str(tf) for _, tf in driver.seen_paths]
    assert len(set(result_files)) == 2, result_files
    assert len(set(trace_files)) == 2, trace_files


def test_park_ref_names_include_the_item_so_different_items_never_collide(
    tmp_path: Path,
) -> None:
    """`_park` is reached with an attempt number that resets per item; the item
    must be folded into the branch name or two items exhausting at the same
    local attempt would silently retarget one branch onto the other's commit."""
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, ITEM_SCOPED_RETRY, {"check.py": PASSING_CHECK})
    kernel = Kernel(
        manifest_path=base / "coding" / "w.workflow.md",
        workspace=repo, task_id=TASK_ID, task_text="t", base_dir=base,
        kernel_data_root=tmp_path / "kernel_data",
        driver=StubDriver([]),
    )

    (repo / "product.txt").write_text("item one's rejected attempt\n", encoding="utf-8")
    ref1 = f"do_item-{kernel._safe_name(Path('WI-01-first.md').name)}-2"
    kernel._park("do_item", 2, item="agents/tasks/T-1/workitems/WI-01-first.md")

    (repo / "product.txt").write_text("item two's rejected attempt\n", encoding="utf-8")
    ref2 = f"do_item-{kernel._safe_name(Path('WI-02-second.md').name)}-2"
    kernel._park("do_item", 2, item="agents/tasks/T-1/workitems/WI-02-second.md")

    assert ref1 != ref2
    branches = git(repo, "branch", "--list", "rejected/*")
    assert f"rejected/{TASK_ID}/{ref1}" in branches
    assert f"rejected/{TASK_ID}/{ref2}" in branches
    assert git(
        repo, "show", f"rejected/{TASK_ID}/{ref1}:product.txt"
    ) == "item one's rejected attempt"
    assert git(
        repo, "show", f"rejected/{TASK_ID}/{ref2}:product.txt"
    ) == "item two's rejected attempt"


BLOCKING = """\
---
name: t
driver: {kind: claude, model: sonnet}
checkpoint_backend: {kind: git, repo_path: "${TARGET}"}
human_resolver: {mode: forbid}
roles:
  first:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
  second:
    skill: skills/x/SKILL.md
    result_contract:
      type: json
      schema:
        status: {enum: [done, blocked]}
phases:
  - name: step_one
    kind: role
    role: first
    checkpoint_after: true
    on_status:
      done: step_two
    on_invalid: {action: retry_with_feedback, target: step_one}
  - name: step_two
    kind: role
    role: second
    on_status:
      done: finish
      blocked: stop
    on_invalid: {action: retry_with_feedback, target: step_two}
  - name: finish
    kind: script
    script: scripts/check.py
---
"""


def test_a_blocked_run_resumes_at_the_phase_that_blocked(tmp_path: Path) -> None:
    """The earlier phases were accepted; re-running them would throw away work
    that already passed its checks. This is what makes "fix the cause and run
    the same command again" the whole recovery story -- including when the cause
    was the role's own instructions rather than the repository."""
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, BLOCKING, {"check.py": PASSING_CHECK})

    first_run = StubDriver([{"status": "done"}, {"status": "blocked"}])
    result = run_kernel(base, repo, first_run, tmp_path)
    assert not result["ok"]
    assert result["exit_reason"] == "step_two: stop"
    assert first_run.calls == 2

    # Whatever made step_two block is fixed; the same command runs again.
    second_run = StubDriver([{"status": "done"}])
    result = run_kernel(base, repo, second_run, tmp_path)

    assert result["ok"], result["exit_reason"]
    # Only step_two was re-dispatched: step_one's accepted work was not redone.
    assert second_run.calls == 1
    assert "`second`" in second_run.prompts[0]


def test_a_blocked_loop_body_resumes_by_reselecting_the_item(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    _work_items(repo, "WI-01-first.md")
    base = make_base(tmp_path, LOOP.replace(
        "        status: {enum: [item_done, item_again]}",
        "        status: {enum: [item_done, item_again, blocked]}",
    ).replace(
        "          item_again: do_item",
        "          item_again: do_item\n          blocked: stop",
    ), {"check.py": PASSING_CHECK})

    first_run = StubDriver([{"status": "blocked"}])
    result = run_kernel(base, repo, first_run, tmp_path)
    assert not result["ok"]

    second_run = StubDriver([{"status": "item_done"}])
    result = run_kernel(base, repo, second_run, tmp_path)

    assert result["ok"], result["exit_reason"]
    # The loop was re-entered, so the item was selected again rather than
    # assumed to still be current.
    assert "WI-01-first.md" in second_run.prompts[0]


def test_resuming_never_lands_inside_a_loop_without_an_item(tmp_path: Path) -> None:
    """The selected work item lives in memory. A process that resumed straight
    into a loop body would run the phase with no item at all -- the prompt would
    carry no work item and the per-item checks would have nothing to check."""
    repo = make_repo(tmp_path)
    _work_items(repo, "WI-01-first.md", "WI-02-second.md")
    base = make_base(tmp_path, LOOP, {"check.py": PASSING_CHECK})

    # Reach the point where the loop selected an item and routed into the body,
    # then stop as abruptly as a reboot would.
    kernel = Kernel(
        manifest_path=base / "coding" / "w.workflow.md",
        workspace=repo, task_id=TASK_ID, task_text="t", base_dir=base,
        kernel_data_root=tmp_path / "kernel_data", driver=StubDriver([]),
    )
    loop_phase = kernel.manifest.phases[0]
    outcome = kernel._execute_phase(loop_phase)
    target = kernel._route(loop_phase, outcome)
    assert target == "do_item"
    from pm_workflows.protocol import JournalEntry
    kernel.journal.append(JournalEntry(
        run_id=TASK_ID, phase=loop_phase.name, kind="route", ok=True, verdict=target
    ))

    resumed = StubDriver([{"status": "item_done"}, {"status": "item_done"}])
    result = run_kernel(base, repo, resumed, tmp_path)

    assert result["ok"], result["exit_reason"]
    # Both items ran, and the first prompt names the item it is working on.
    assert resumed.calls == 2
    assert "WI-01-first.md" in resumed.prompts[0]


def test_a_dirty_tree_is_discarded_when_a_role_runs_next(tmp_path: Path) -> None:
    """Debris from a killed session never passed a check. If the next thing to
    run is a role, leaving it would credit files to a session that did not
    write them — a reboot must not smuggle unreviewed changes into a commit."""
    repo = make_repo(tmp_path)
    _work_items(repo, "WI-01-first.md", "WI-02-second.md")
    base = make_base(tmp_path, LOOP, {"check.py": PASSING_CHECK})

    # Finish the first item so there is an accepted revision to fall back to,
    # then leave the run poised to implement the second.
    kernel = Kernel(
        manifest_path=base / "coding" / "w.workflow.md",
        workspace=repo, task_id=TASK_ID, task_text="t", base_dir=base,
        kernel_data_root=tmp_path / "kernel_data",
        driver=StubDriver([{"status": "item_done"}]),
    )
    from pm_workflows.protocol import JournalEntry
    for phase_name in ("items", "do_item"):
        phase = kernel.manifest.phase_by_name(phase_name)
        outcome = kernel._execute_phase(phase)
        if outcome["valid"] and phase.checkpoint_after:
            kernel._accept(phase.name, 1)
        target = kernel._route(phase, outcome)
        kernel.journal.append(JournalEntry(
            run_id=TASK_ID, phase=phase.name, kind="route", ok=True, verdict=target
        ))
        kernel._resolve_target(phase, target)
    accepted = kernel.accepted_revision

    # The machine dies mid-implementation of the second item.
    (repo / "product.txt").write_text("half written\n", encoding="utf-8")
    (repo / "stray.txt").write_text("debris\n", encoding="utf-8")

    resumed = StubDriver([{"status": "item_done"}])
    run_kernel(base, repo, resumed, tmp_path)

    # The debris is gone, and the second item was implemented from clean state.
    assert not (repo / "stray.txt").exists()
    assert "WI-02-second.md" in resumed.prompts[0]
    assert accepted


def test_a_dirty_tree_is_kept_when_a_check_runs_next(tmp_path: Path) -> None:
    """The mirror case. Resuming at a check means the role already reported and
    only its validation was lost, so the tree is the candidate that check exists
    to judge — discarding it would have the check pass against nothing and
    accept work that was never done."""
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, TWO_STEP, {"check.py": PASSING_CHECK})

    kernel = Kernel(
        manifest_path=base / "coding" / "w.workflow.md",
        workspace=repo, task_id=TASK_ID, task_text="t", base_dir=base,
        kernel_data_root=tmp_path / "kernel_data",
        driver=StubDriver(
            [{"status": "done", "summary": "s"}],
            on_call=lambda d, c, p: (d / "product.txt").write_text(
                "the work\n", encoding="utf-8"),
        ),
    )
    phase = kernel.manifest.phases[0]
    outcome = kernel._execute_phase(phase)
    from pm_workflows.protocol import JournalEntry
    kernel.journal.append(JournalEntry(
        run_id=TASK_ID, phase=phase.name, kind="route", ok=True,
        verdict=kernel._route(phase, outcome),
    ))

    result = run_kernel(base, repo, StubDriver([]), tmp_path)

    assert result["ok"], result["exit_reason"]
    assert (repo / "product.txt").read_text(encoding="utf-8") == "the work\n"


def test_fresh_really_starts_over(tmp_path: Path) -> None:
    """--fresh used to re-baseline the repository but still resume at the phase
    the previous run stopped at, which is neither behaviour anybody asked for."""
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, TWO_STEP, {"check.py": PASSING_CHECK})
    run_kernel(base, repo, StubDriver([{"status": "done", "summary": "s"}]), tmp_path)

    restarted = StubDriver([{"status": "done", "summary": "s"}])
    result = run_kernel(base, repo, restarted, tmp_path, resume=False)

    assert result["ok"], result["exit_reason"]
    # The role ran again from the top rather than resuming past it.
    assert restarted.calls == 1
    data = tmp_path / "kernel_data" / TASK_ID
    assert (data / "journal.attempt1.jsonl").is_file(), "old journal not archived"
    # The new journal is its own history, not a continuation of the old one.
    assert len(_journal(tmp_path)) < len(
        [line for line in (data / "journal.attempt1.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
    ) + 5


def test_a_revert_does_not_destroy_the_code_index(tmp_path: Path) -> None:
    """`.codegraph/` is not in most repos' .gitignore, so the `git clean -fd`
    in every revert would delete it. That failure is silent: the index just
    stops existing, and every later role is still instructed to consult it."""
    repo = make_repo(tmp_path)
    index = repo / ".codegraph"
    index.mkdir()
    (index / "index.db").write_text("expensive to rebuild\n", encoding="utf-8")

    base = make_base(tmp_path, TWO_STEP, {"check.py": FAILING_CHECK})
    driver = StubDriver([{"status": "done", "summary": "s"}] * 5)
    run_kernel(base, repo, driver, tmp_path)

    # The run gave up after reverting several times; the index survived all of them.
    assert (index / "index.db").read_text(encoding="utf-8") == "expensive to rebuild\n"


def test_a_role_never_inherits_the_previous_phase_s_leftovers(tmp_path: Path) -> None:
    """Leftovers would be swept into this role's checkpoint and credited to a
    session that did not write them, corrupting both the scope check and the
    evidence trail. They are parked so nothing is destroyed, then cleared."""
    repo = make_repo(tmp_path)
    _work_items(repo, "WI-01-first.md")
    base = make_base(tmp_path, LOOP, {"check.py": PASSING_CHECK})

    kernel = Kernel(
        manifest_path=base / "coding" / "w.workflow.md",
        workspace=repo, task_id=TASK_ID, task_text="t", base_dir=base,
        kernel_data_root=tmp_path / "kernel_data",
        driver=StubDriver([{"status": "item_done"}]),
    )
    # Something left changes behind without accepting or reverting them.
    (repo / "product.txt").write_text("orphaned edit\n", encoding="utf-8")
    (repo / "orphan.txt").write_text("nobody owns this\n", encoding="utf-8")

    loop = kernel.manifest.phases[0]
    kernel._execute_phase(loop)
    kernel.current_item = "agents/tasks/T-1/workitems/WI-01-first.md"
    kernel._execute_phase(kernel.manifest.phase_by_name("do_item"))

    # The role started from a clean tree...
    assert (repo / "product.txt").read_text(encoding="utf-8") == "base\n"
    assert not (repo / "orphan.txt").exists()
    # ...and the orphaned work is still inspectable rather than destroyed.
    branches = git(repo, "branch", "--list", "leftover/*")
    assert f"leftover/{TASK_ID}/do_item" in branches
    assert git(repo, "show", f"leftover/{TASK_ID}/do_item:orphan.txt") == "nobody owns this"


def test_state_md_shows_the_revert_before_the_retry_reads_it(tmp_path: Path) -> None:
    """The projection is re-rendered after routing, not only before it. A
    session retried after a revert must not read a history in which its own
    discarded attempt still succeeded."""
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, TWO_STEP, {"check.py": FAILING_CHECK})

    seen: list[str] = []

    def capture_state(work_dir: Path, call: int, prompt: str) -> None:
        state = work_dir / "agents" / "tasks" / TASK_ID / "state.md"
        seen.append(state.read_text(encoding="utf-8") if state.is_file() else "")

    run_kernel(base, repo, StubDriver(
        [{"status": "done", "summary": "s"}] * 4, on_call=capture_state), tmp_path)

    # The second dispatch happens after the first attempt was reverted.
    assert len(seen) >= 2
    assert "revert" in seen[1].lower(), seen[1]


def test_state_md_survives_a_revert(tmp_path: Path) -> None:
    """It is a projection of a journal that lives outside the repository, so a
    hard reset cannot lose it."""
    repo = make_repo(tmp_path)
    base = make_base(tmp_path, TWO_STEP, {"check.py": FAILING_CHECK})
    run_kernel(base, repo, StubDriver([{"status": "done", "summary": "s"}] * 4), tmp_path)

    state = repo / "agents" / "tasks" / TASK_ID / "state.md"
    assert state.is_file(), "state.md was destroyed by a revert"
    assert "Workflow state" in state.read_text(encoding="utf-8")
