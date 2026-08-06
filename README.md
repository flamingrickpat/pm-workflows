# pm-workflows

`pm-workflows` is the reusable workflow kernel. It loads a workflow manifest,
dispatches isolated role sessions, runs declared checks, routes declared
outcomes, journals attempts, and manages git checkpoints.

The import package is `pm_workflows`. The distribution installs `pm-coder` from
the GitHub repository.

Workflow metadata is stored outside the target repository in
`~/.pm/pm-workflows/<timestamp>_<task-id>/`.

## Attempt counting inside a loop

A phase's `on_invalid`/`on_failure` `max_attempts`, and the attempt number a
role sees, are counted per work item when that phase is inside a loop body —
not over the task's lifetime. A hard item's retries do not eat into the next
item's budget, and each new item starts its own count at 1. A phase that
runs once per task (outside any loop) is still counted for the task's whole
lifetime, as there is no item to scope it by.

This means `max_attempts` can be set to what one item actually needs (e.g.
`8`) instead of a number large enough to survive every item the loop will
ever see.

## Checking a workflow without running it

`pm_workflows.dryrun` statically checks a manifest and simulates it — no LLM,
no coding agent, no script or skill is ever executed. Every route in a
manifest is a plain value in YAML, so the kernel's own routing/retry math can
be replayed without dispatching anything:

```
pm-workflow-check --manifest workflows/auto_task.workflow.md --base agents-deploy
```

It reports, in one pass:

* every role's skill, every gate/script's path, and every `kind: workflow`
  phase's child manifest (checked recursively) — missing files are blocking
  errors.
* every cycle the routing graph contains and what bounds it — a self-retry's
  `max_attempts`, or the ambient `failure_policy.max_attempts_per_phase` /
  loop `max_iterations` backstop, whichever is tighter — flagging phases that
  fall back to the 999 default instead of declaring one.
* best case (fewest dispatches to success, with 0 and with 1 work item) and a
  conservative worst-case dispatch-count ceiling as a formula in the number
  of work items.
* a Monte Carlo simulation (a weighted random walk over the same graph,
  since the manifest has no outcome probabilities to reason about
  statically) reporting success/stopped/exhausted rates, a dispatch-count
  distribution, and the phases visited most often — `--runs`, `--items-min`/
  `--items-max`, `--p-invalid`, `--p-fail`, and `--p-primary` tune the
  assumptions behind it.

Exit code and the JSON line on the last line of output
(`{"ok": ..., "errors": [...]}`) follow the same contract as a `kind: script`
gate (see `pm_workflows.gates.run_gate`), so a meta-workflow that generates
new manifests can check its own output before ever running it, by adding a
thin wrapper script and a gate phase:

```python
# scripts/check_generated_workflow.py
from pm_workflows.dryrun import main
if __name__ == "__main__":
    raise SystemExit(main())
```

```yaml
- name: check_workflow
  kind: gate
  predicate: scripts/check_generated_workflow.py
  args: ["--manifest", "${TASK_DIR}/generated.workflow.md", "--base", "${BASE}"]
  on_pass: finalize
  on_fail: {action: retry_with_feedback, target: build_workflow, max_attempts: 3}
```

The worst-case ceiling counts each phase's bound independently; it does not
tighten a phase's count by how few times an upstream retry target that feeds
it can actually succeed, so it is a safe but sometimes loose upper bound.

## State policies

Manifests without `state_policy` use the original transactional behavior. The
kernel settles dirty work before a role, restores the last accepted Git
revision on retry, and parks rejected work when attempts are exhausted.

External environments use an opt-in policy:

```yaml
state_policy:
  mutation_model: external
  on_resume: retain
  before_role: retain
  on_retry: retain
  on_exhaustion: retain
  attempt_receipts: files
  feedback: accumulated
```

The `external` and `append_only` presets supply these retention values. Every
field can be overridden. A retained retry starts a fresh agent session, keeps
workspace files and external effects, archives a private attempt receipt, and
gives the next role a compact, deduplicated failure list. The next role must
observe the external environment again. Retention is not rollback, and an
external compensation must be modeled as a new action.

`transactional` remains the default preset. Its values are `on_resume: legacy`,
`before_role: settle`, `on_retry: restore`, `on_exhaustion:
park_and_restore`, `attempt_receipts: journal`, and `feedback: accumulated`.

## Python roles

A role whose `skill:` path ends in `.py` is not dispatched to a coding-agent
CLI. The kernel loads that file and calls its `run(context)` entry point
directly, in its own process:

```yaml
roles:
  train_and_wait:
    skill: skills/train_and_wait/role.py
    result_contract:
      schema:
        status: { enum: [done, failed] }
        summary: string
```

```python
# skills/train_and_wait/role.py
def run(context):
    ...  # context: pm_workflows.python_role.RoleContext
    return {"status": "done", "summary": "trained and merged three runs"}
```

The returned dict is validated against the role's `result_contract` exactly
like an agentic role's final JSON message, and it is journalled, checkpointed
and routed the same way. `context` carries the run id, task id, workspace,
task folder, current work item, retry feedback and the same task text an
agentic role would read from its prompt — see `pm_workflows/python_role.py`.

There is no sandbox. The script runs with the kernel's own privileges and may
do anything a Python program can do: write outside `writable_paths`, start
subprocesses, open sockets, train a model, or shell out to other
`pm-workflows` runs and fan them out in parallel before collecting their
results. It is exactly as trusted as `pm-workflows` itself — this is the
integration point for work an agentic skill cannot do on its own. A raised
exception, a missing entry point, or a non-dict return value is journalled as
a contract violation and routed through `on_invalid` like any other.

A workflow may mix python and agentic roles freely; each role's skill
extension decides how it runs regardless of the workflow's configured
`--coding-agent`. A workflow made entirely of python roles can also select
`--coding-agent python` (or `driver: {kind: python}`) so no external agent
CLI is required at all.

## Child workflows

`kind: workflow` invokes a statically named manifest. Each invocation has a
fresh task folder, private journal, request budget, and depth budget. The parent
receives a typed receipt and routes only on declared statuses.

```yaml
- name: execute_leaf
  kind: workflow
  workflow: minecraft-action
  task:
    id: "${TASK_ID}.leaf.${leaf.id}"
    input: {leaf: "${leaf}"}
  foreach:
    from: "${TASK_DIR}/plan.yaml#/leaves"
    item: leaf
    stable_id: leaf.id
    order: dependency_topological_then_id
    max_items: 32
  workspace:
    mode: shared
    merge: artifacts_only
    artifact_prefix: "children/leaves/${leaf.id}"
  context:
    inherit: false
    include: ["memory/minecraft/**", "artifacts/minecraft/**"]
    exclude: [".pm/**"]
  capabilities:
    inherit: false
    allow_mcp: [minecraft]
    allow_effects: [minecraft_survival]
    require_http_reachable: true
    http_timeout_seconds: 5
  limits:
    decrement_depth: 1
    max_attempts: 4
    max_agent_requests: 100
  result:
    statuses: [completed, recoverable_failure, blocked, decomposition_limit]
    status_map: {pass: completed, postcondition_failed: recoverable_failure}
    artifacts: [attempt.md, attempt-review.md]
    aggregate: dependency_graph
    status_priority: [blocked, recoverable_failure]
  on_status:
    completed: review
    recoverable_failure: replan
    blocked: replan
    decomposition_limit: replan
  on_invalid: {action: retry_child_clean, target: execute_leaf, max_attempts: 2}
```

The child status is its last role/workflow status unless `result.status_from`
selects a structured artifact. `status_map` makes normalization explicit. The
kernel does not guess that a review status such as `pass` means `completed`.
Depth exhaustion returns `decomposition_limit` when that status is declared.

`retry_child_clean` means a new child task and journal. It does not remove the
previous child task, copied artifacts, workspace changes, or external effects.
Declared child artifacts are copied to immutable `attempt-NNNN` folders below
the parent's `artifact_prefix`.

When a parent grants MCP servers, each built-in driver receives a generated MCP
configuration containing only the servers requested by the active role. If
`require_http_reachable` is true, a missing, non-HTTP, or unreachable server is
a hard contract failure before the role starts. Manifests without a parent MCP
grant still use the deployed `.mcp.json` exactly as before.

Current child execution is sequential and uses a shared workspace. Isolated
workspaces, automatic Git merges, parallel fan-out, and `foreach.stop_when`
hooks are not implemented. `context.include` and `context.exclude` are recorded
and placed in the child request, but they are not a filesystem ACL.
`allow_effects` is recorded in receipts but cannot be enforced until effect
adapters expose typed capability IDs.
