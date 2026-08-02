# pm-workflows

`pm-workflows` is the reusable workflow kernel. It loads a workflow manifest,
dispatches isolated role sessions, runs declared checks, routes declared
outcomes, journals attempts, and manages git checkpoints.

The import package is `pm_workflows`. The distribution installs `pm-coder` from
the GitHub repository.

Workflow metadata is stored outside the target repository in
`~/.pm/pm-workflows/<timestamp>_<task-id>/`.

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
