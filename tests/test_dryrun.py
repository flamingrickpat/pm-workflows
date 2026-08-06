"""Static check and dry-run simulation — no LLM, no coding agent, no script
or skill is ever executed here. These tests build small workflow manifests
directly (no git repo needed, since dryrun never touches one) and check the
file-existence report, the routing graph, cycle bounds, best/worst case, and
the Monte Carlo simulator.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pm_workflows.dryrun import (
    TERMINAL_STOP,
    TERMINAL_SUCCESS,
    build_graph,
    check_workflow,
    find_cycles,
    main,
)
from pm_workflows.manifest import parse_workflow


def _base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    (base / "workflows").mkdir(parents=True)
    (base / "skills" / "x").mkdir(parents=True)
    (base / "skills" / "x" / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    (base / "scripts").mkdir(parents=True)
    (base / "scripts" / "check.py").write_text(
        "import json; print(json.dumps({'ok': True, 'errors': []}))\n", encoding="utf-8",
    )
    return base


def _write(base: Path, name: str, text: str) -> Path:
    path = base / "workflows" / name
    path.write_text(text, encoding="utf-8")
    return path


LINEAR = """\
---
name: linear
driver: {kind: claude}
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      schema: {status: {enum: [done]}}
phases:
  - name: work
    kind: role
    role: worker
    on_status: {done: check}
    on_invalid: {action: retry_with_feedback, target: work, max_attempts: 4}
  - name: check
    kind: script
    script: scripts/check.py
---
"""


def test_ok_workflow_has_no_errors(tmp_path: Path) -> None:
    base = _base(tmp_path)
    manifest = _write(base, "linear.workflow.md", LINEAR)

    report = check_workflow(manifest, base, skip_monte_carlo=True)

    assert report.ok, report.errors
    assert len(report.file_checks) == 2  # the skill and the script
    assert all(c.exists for c in report.file_checks)


def test_missing_skill_and_script_are_reported(tmp_path: Path) -> None:
    base = _base(tmp_path)
    broken = LINEAR.replace(
        "skill: skills/x/SKILL.md", "skill: skills/missing/SKILL.md"
    ).replace("script: scripts/check.py", "script: scripts/missing.py")
    manifest = _write(base, "broken.workflow.md", broken)

    report = check_workflow(manifest, base, skip_monte_carlo=True)

    assert not report.ok
    assert any("skills/missing/SKILL.md" in e for e in report.errors)
    assert any("scripts/missing.py" in e for e in report.errors)


CHILD_PARENT = """\
---
name: parent
driver: {kind: claude}
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      schema: {status: {enum: [done]}}
phases:
  - name: delegate
    kind: workflow
    workflow: missing-child
    task: {id: "child.${TASK_ID}"}
    result: {statuses: [completed]}
    on_status: {completed: work}
    on_invalid: {action: stop}
  - name: work
    kind: role
    role: worker
    on_status: {done: stop}
    on_invalid: {action: stop}
---
"""


def test_missing_child_workflow_is_reported(tmp_path: Path) -> None:
    base = _base(tmp_path)
    manifest = _write(base, "parent.workflow.md", CHILD_PARENT)

    report = check_workflow(manifest, base, skip_monte_carlo=True)

    assert not report.ok
    assert any("missing-child" in e for e in report.errors)


def test_child_workflow_files_are_checked_recursively(tmp_path: Path) -> None:
    base = _base(tmp_path)
    child = CHILD_PARENT.replace("workflow: missing-child", "workflow: child")
    _write(base, "child.workflow.md", LINEAR.replace(
        "skill: skills/x/SKILL.md", "skill: skills/child-only/SKILL.md"
    ))
    manifest = _write(base, "parent2.workflow.md", child)

    report = check_workflow(manifest, base, skip_monte_carlo=True)

    assert not report.ok
    assert any("child-only" in e for e in report.errors)


SELF_RETRY = """\
---
name: self_retry
driver: {kind: claude}
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      schema: {status: {enum: [done]}}
phases:
  - name: work
    kind: role
    role: worker
    on_status: {done: stop}
    on_invalid: {action: retry_with_feedback, target: work, max_attempts: 5}
---
"""


def test_self_retry_cycle_bound_matches_declared_max_attempts(tmp_path: Path) -> None:
    base = _base(tmp_path)
    manifest = _write(base, "self_retry.workflow.md", SELF_RETRY)
    workflow = parse_workflow(manifest, {"TASK_ID": "t", "WORKSPACE": str(base), "TARGET": str(base), "TASK_DIR": str(base), "BASE": str(base)})
    nodes = build_graph(workflow)

    cycles = find_cycles(nodes)

    assert len(cycles) == 1
    assert cycles[0].phases == ["work"]
    assert cycles[0].bound_per_phase["work"] == 5
    assert not cycles[0].per_item


PING_PONG = """\
---
name: ping_pong
driver: {kind: claude}
roles:
  reviewer:
    skill: skills/x/SKILL.md
    result_contract:
      schema: {status: {enum: [findings]}}
  fixer:
    skill: skills/x/SKILL.md
    result_contract:
      schema: {status: {enum: [nothing_to_fix]}}
phases:
  - name: review
    kind: role
    role: reviewer
    on_status: {findings: fix}
    on_invalid: {action: retry_with_feedback, target: review}
  - name: fix
    kind: role
    role: fixer
    on_status: {nothing_to_fix: review}
    on_invalid: {action: retry_with_feedback, target: fix}
---
"""


def test_ping_pong_with_no_explicit_cap_is_bounded_by_the_backstop(tmp_path: Path) -> None:
    """No self-retry `max_attempts` anywhere in this cycle — the only thing
    that ever stops it is `failure_policy.max_attempts_per_phase` (999 by
    default), which is exactly the default the kernel itself applies."""
    base = _base(tmp_path)
    manifest = _write(base, "ping_pong.workflow.md", PING_PONG)
    workflow = parse_workflow(manifest, {"TASK_ID": "t", "WORKSPACE": str(base), "TARGET": str(base), "TASK_DIR": str(base), "BASE": str(base)})
    nodes = build_graph(workflow)

    cycles = find_cycles(nodes)

    assert len(cycles) == 1
    assert set(cycles[0].phases) == {"review", "fix"}
    assert cycles[0].bound_per_phase == {"review": 999, "fix": 999}
    assert not cycles[0].per_item


LOOP_WORKFLOW = """\
---
name: loop_wf
driver: {kind: claude}
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      schema: {status: {enum: [item_done]}}
phases:
  - name: items
    kind: loop
    iterator_source: pending_work_items
    max_iterations: 50
    exit: finish
    on_failure: {action: stop}
    body:
      - name: do_item
        kind: role
        role: worker
        on_status: {item_done: next_item}
        on_invalid: {action: retry_with_feedback, target: do_item, max_attempts: 3}
  - name: finish
    kind: script
    script: scripts/check.py
---
"""


def test_loop_body_bound_is_per_item(tmp_path: Path) -> None:
    base = _base(tmp_path)
    manifest = _write(base, "loop_wf.workflow.md", LOOP_WORKFLOW)
    workflow = parse_workflow(manifest, {"TASK_ID": "t", "WORKSPACE": str(base), "TARGET": str(base), "TASK_DIR": str(base), "BASE": str(base)})
    nodes = build_graph(workflow)

    assert nodes["do_item"].effective_repeat_bound == 3
    assert nodes["do_item"].loop == "items"

    cycles = find_cycles(nodes)
    assert len(cycles) == 1
    assert cycles[0].phases == ["do_item"]
    assert cycles[0].per_item


def test_best_and_worst_case_for_a_loop_workflow(tmp_path: Path) -> None:
    base = _base(tmp_path)
    manifest = _write(base, "loop_wf2.workflow.md", LOOP_WORKFLOW)

    report = check_workflow(manifest, base, skip_monte_carlo=True)

    assert report.ok, report.errors
    # 0 items: just the loop taking its exhausted edge, then finish (1 dispatch).
    assert report.best["zero_items"].dispatches == 1
    # 1 item: do_item once (1 dispatch), then finish (1 dispatch).
    assert report.best["one_item"].dispatches == 2
    # worst case: do_item can run 3 times per item, plus 1 for finish outside the loop.
    assert report.worst["fixed"] == 1
    assert report.worst["per_item"] == 3
    assert report.worst["examples"][5] == 1 + 3 * 5


def test_manifest_errors_are_surfaced_without_crashing(tmp_path: Path) -> None:
    base = _base(tmp_path)
    manifest = _write(base, "bad.workflow.md", "not even yaml frontmatter\n")

    report = check_workflow(manifest, base, skip_monte_carlo=True)

    assert not report.ok
    assert report.parse_error is not None
    assert "frontmatter" in report.errors[0]


# ----------------------------------------------------------------- monte carlo

ALWAYS_SUCCEEDS = """\
---
name: always_succeeds
driver: {kind: claude}
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      schema: {status: {enum: [done]}}
phases:
  - name: work
    kind: role
    role: worker
    on_status: {done: finish}
    on_invalid: {action: stop}
  - name: finish
    kind: script
    script: scripts/check.py
---
"""


def test_monte_carlo_always_succeeds_with_a_single_declared_outcome(tmp_path: Path) -> None:
    base = _base(tmp_path)
    manifest = _write(base, "always.workflow.md", ALWAYS_SUCCEEDS)

    report = check_workflow(
        manifest, base, monte_carlo_runs=200, p_invalid=0.0, p_fail=0.0, seed=1,
    )

    summary = report.monte_carlo.summary()
    assert summary["outcomes_pct"]["success"] == 100.0
    # 'work' then the implicit-success 'finish' script: 2 dispatches, every run.
    assert summary["dispatches_min"] == summary["dispatches_max"] == 2


ALWAYS_FAILS_TINY_CAP = """\
---
name: always_fails
driver: {kind: claude}
roles:
  worker:
    skill: skills/x/SKILL.md
    result_contract:
      schema: {status: {enum: [done]}}
phases:
  - name: gate
    kind: gate
    predicate: scripts/check.py
    on_pass: stop
    on_fail: {action: retry_with_feedback, target: gate, max_attempts: 2}
---
"""


def test_monte_carlo_reaches_exhausted_when_forced_to_fail(tmp_path: Path) -> None:
    base = _base(tmp_path)
    manifest = _write(base, "fails.workflow.md", ALWAYS_FAILS_TINY_CAP)

    report = check_workflow(
        manifest, base, monte_carlo_runs=200, p_fail=1.0, seed=1,
    )

    summary = report.monte_carlo.summary()
    assert summary["outcomes_pct"]["exhausted"] == 100.0
    # Exactly the declared cap: 2 dispatches of 'gate' before giving up.
    assert summary["dispatches_max"] == 2


def test_loop_item_count_is_sampled_once_and_actually_exhausts(tmp_path: Path) -> None:
    """Regression test: the loop's random item count must be drawn once per
    run and only decremented, never regenerated while > 0 is impossible to
    reach — otherwise the loop can never legitimately exit and every run
    would either hang or fail via unrelated attrition instead of finishing.
    With zero chance of any failure here, every run must reach success in a
    bounded number of dispatches (items + 1 for `finish`), never 'exhausted'.
    """
    base = _base(tmp_path)
    manifest = _write(base, "loop_wf3.workflow.md", LOOP_WORKFLOW)

    report = check_workflow(
        manifest, base, monte_carlo_runs=300, items_range=(3, 3),
        p_invalid=0.0, p_fail=0.0, seed=7,
    )

    summary = report.monte_carlo.summary()
    assert summary["outcomes_pct"].get("exhausted", 0.0) == 0.0
    assert summary["outcomes_pct"]["success"] == 100.0
    # Exactly 3 items + 1 'finish' dispatch, every single run.
    assert summary["dispatches_min"] == summary["dispatches_max"] == 4


def test_to_dict_round_trips_through_json(tmp_path: Path) -> None:
    base = _base(tmp_path)
    manifest = _write(base, "linear2.workflow.md", LINEAR)

    report = check_workflow(manifest, base, monte_carlo_runs=50, seed=1)
    payload = json.loads(json.dumps(report.to_dict(), default=str))

    assert payload["ok"] is True
    assert payload["best_case"]["zero_items"] is not None
    assert payload["monte_carlo"]["runs"] == 50


def test_render_text_mentions_the_result_line(tmp_path: Path) -> None:
    base = _base(tmp_path)
    manifest = _write(base, "linear3.workflow.md", LINEAR)

    report = check_workflow(manifest, base, skip_monte_carlo=True)
    text = report.render_text()

    assert "RESULT: 0 blocking error(s)" in text


# ------------------------------------------------------------------------ cli

def test_cli_exits_zero_on_a_clean_workflow(tmp_path: Path, capsys) -> None:
    base = _base(tmp_path)
    manifest = _write(base, "cli_ok.workflow.md", LINEAR)

    code = main([
        str(tmp_path), "--manifest", str(manifest), "--base", str(base),
        "--no-monte-carlo",
    ])

    assert code == 0
    out = capsys.readouterr().out
    last_line = [line for line in out.splitlines() if line.strip()][-1]
    verdict = json.loads(last_line)
    assert verdict["ok"] is True
    assert verdict["errors"] == []


def test_cli_exits_nonzero_on_a_broken_workflow(tmp_path: Path, capsys) -> None:
    base = _base(tmp_path)
    broken = LINEAR.replace("skill: skills/x/SKILL.md", "skill: skills/missing/SKILL.md")
    manifest = _write(base, "cli_broken.workflow.md", broken)

    code = main([
        str(tmp_path), "--manifest", str(manifest), "--base", str(base),
        "--no-monte-carlo",
    ])

    assert code == 1
    out = capsys.readouterr().out
    last_line = [line for line in out.splitlines() if line.strip()][-1]
    verdict = json.loads(last_line)
    assert verdict["ok"] is False
    assert verdict["errors"]


def test_cli_json_mode_emits_the_full_report(tmp_path: Path, capsys) -> None:
    base = _base(tmp_path)
    manifest = _write(base, "cli_json.workflow.md", LINEAR)

    code = main([
        str(tmp_path), "--manifest", str(manifest), "--base", str(base),
        "--no-monte-carlo", "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workflow"] == "linear"
    assert payload["ok"] is True
