"""Kernel protocol: dataclasses shared across the kernel."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Force UTF-8 stdout for Unicode symbols on Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# Reserved route targets. A workflow may name these instead of a phase.
ROUTE_NEXT_ITEM = "next_item"
ROUTE_EXIT_LOOP = "exit_loop"
ROUTE_STOP = "stop"
RESERVED_ROUTES = frozenset({ROUTE_NEXT_ITEM, ROUTE_EXIT_LOOP, ROUTE_STOP})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WorkflowResolution:
    """One catalog result with the deployment base needed by a child kernel."""

    manifest_path: Path
    deployment_base: Path
    qualified_name: str | None = None


@dataclass
class AgentResult:
    """Result of an agent subsession."""
    exit_code: int
    stdout: str
    result_json: dict[str, Any] | None
    usage: dict[str, Any] = field(default_factory=dict)
    trace_path: str | None = None
    error: str | None = None
    session_ref: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.result_json is not None


@dataclass
class GateResult:
    """Result of a gate/script check."""
    ok: bool
    errors: list[str] = field(default_factory=list)
    output: str = ""
    exit_code: int = 0


@dataclass
class JournalEntry:
    """One append-only journal record."""
    run_id: str
    phase: str
    # role | gate | script | human | checkpoint | failure_memory | escalate | loop
    kind: str
    role: str | None = None
    attempt: int = 0
    ok: bool = False
    base_rev: str | None = None
    candidate_rev: str | None = None
    verdict: str | None = None
    status: str | None = None
    item: str | None = None
    answer: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    trace_path: str | None = None
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != []}


@dataclass(frozen=True)
class StepResult:
    """The durable boundary produced by one kernel phase execution."""

    task_id: str
    run_id: str
    workflow: str
    phase: str | None
    kind: str | None
    attempt: int
    status: str | None
    valid: bool
    next_phase: str | None
    disposition: str
    exit_reason: str
    accepted_revision: str | None
    journal: str
    duration_seconds: float
    data: dict[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    workflow_ok: bool = True
    waiting_kind: str | None = None
    terminal_status: str | None = None
    summary: str = ""
    schema: str = "pm.step-result.v1"

    @property
    def terminal(self) -> bool:
        return self.disposition == "terminal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "workflow": self.workflow,
            "phase_executed": self.phase,
            "phase_kind": self.kind,
            "phase_status": self.status,
            "attempt": self.attempt,
            "valid": self.valid,
            "next_phase": self.next_phase,
            "disposition": self.disposition,
            "terminal": self.terminal,
            "workflow_ok": self.workflow_ok,
            "waiting_kind": self.waiting_kind,
            "terminal_status": self.terminal_status,
            "exit_reason": self.exit_reason,
            "accepted_revision": self.accepted_revision,
            "journal_path": self.journal,
            "duration_seconds": self.duration_seconds,
            "summary": self.summary,
            "data": self.data,
            "errors": list(self.errors),
        }


@dataclass
class ResultContract:
    """What a role must return. The status field's enum drives routing."""
    type: str = "json"
    status_field: str = "status"
    schema: dict[str, Any] = field(default_factory=dict)

    @property
    def status_values(self) -> list[str]:
        """The declared enum for the status field, or [] when undeclared."""
        spec = self.schema.get(self.status_field)
        if isinstance(spec, dict):
            values = spec.get("enum")
            if isinstance(values, list):
                return [str(v) for v in values]
        return []

    def required_fields(self) -> list[str]:
        return [k for k in self.schema if k != self.status_field]


@dataclass
class RoleConfig:
    name: str
    skill: str
    tools: list[str] = field(default_factory=list)
    mcp: list[str] = field(default_factory=list)
    readable_paths: list[str] = field(default_factory=list)
    writable_paths: list[str] = field(default_factory=list)
    deny_access: list[str] = field(default_factory=list)
    can_call_kernel: list[str] = field(default_factory=list)
    result_contract: ResultContract = field(default_factory=ResultContract)
    instruction: str = ""


@dataclass
class ChildTaskConfig:
    """Task identity and explicit inputs for one child workflow invocation."""

    id: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildWorkspaceConfig:
    """How child artifacts are exposed in a shared workspace."""

    mode: str = "shared"
    merge: str = "artifacts_only"
    artifact_prefix: str = ""


@dataclass
class ChildContextConfig:
    """Declarative context boundary recorded in the child receipt and prompt."""

    inherit: bool = False
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class ChildCapabilitiesConfig:
    """Capabilities a child may delegate to its roles."""

    inherit: bool = False
    allow_mcp: list[str] = field(default_factory=list)
    allow_effects: list[str] = field(default_factory=list)
    require_http_reachable: bool = True
    http_timeout_seconds: float = 5.0


@dataclass
class ChildLimitsConfig:
    """Limits applied to each child invocation."""

    decrement_depth: int = 1
    max_depth: int | None = None
    max_attempts: int = 1
    max_agent_requests: int | None = None


@dataclass
class ChildResultConfig:
    """Typed public result projected from a child's private execution history."""

    statuses: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    status_from: str = ""
    status_map: dict[str, str] = field(default_factory=dict)
    default_status: str = ""
    aggregate: str = ""
    status_priority: list[str] = field(default_factory=list)


@dataclass
class ForeachConfig:
    """Deterministic fan-out over a structured artifact."""

    source: str = ""
    item: str = "item"
    stable_id: str = "id"
    order: str = "stable_id"
    max_items: int = 32
    stop_when: str = ""


@dataclass
class PhaseConfig:
    name: str
    kind: str  # role | script | gate | loop | human | workflow
    route: str = "static"
    role: str | None = None
    next: str | None = None

    # Enum routing: declared status value -> phase name (or reserved route).
    on_status: dict[str, Any] = field(default_factory=dict)
    # Contract violation (no JSON, unknown status, missing field) or crash.
    on_invalid: dict[str, Any] | None = None

    # Gate routing.
    on_pass: str | None = None
    on_fail: Any = None
    # Role/script failure routing (non-enum drivers, script phases).
    on_failure: dict[str, Any] | None = None

    checkpoint_after: bool = False
    stop_point: bool = False

    # gate / script phases
    predicate: str | None = None
    script: str | None = None
    args: list[str] = field(default_factory=list)

    # human phases
    question: str = ""
    question_from_result: str = ""

    # loop phases
    iterator_source: str | None = None
    body: list["PhaseConfig"] = field(default_factory=list)
    exit: str | None = None
    max_iterations: int = 200

    workflow: str | None = None
    allowed_roles: list[str] = field(default_factory=list)
    task: ChildTaskConfig | None = None
    workspace: ChildWorkspaceConfig | None = None
    context: ChildContextConfig | None = None
    capabilities: ChildCapabilitiesConfig | None = None
    limits: ChildLimitsConfig | None = None
    child_result: ChildResultConfig | None = None
    foreach: ForeachConfig | None = None

    # Data owned by an optional phase-kind extension. Built-in phases leave
    # this empty.
    extension: dict[str, Any] = field(default_factory=dict)

    def body_by_name(self, name: str) -> "PhaseConfig | None":
        for p in self.body:
            if p.name == name:
                return p
        return None

    def route_targets(self) -> list[str]:
        """Every phase name this phase can route to, for load-time validation."""
        targets: list[str] = []
        for value in (self.next, self.on_pass, self.exit):
            if isinstance(value, str) and value:
                targets.append(value)
        for value in self.on_status.values():
            if isinstance(value, str) and value:
                targets.append(value)
            elif isinstance(value, dict) and value.get("action") == "suspend":
                resume_at = value.get("resume_at")
                if isinstance(resume_at, str) and resume_at:
                    targets.append(resume_at)
        for cfg in (self.on_invalid, self.on_failure, self.on_fail):
            if isinstance(cfg, str) and cfg:
                targets.append(cfg)
            elif isinstance(cfg, dict):
                target = cfg.get("target")
                if isinstance(target, str) and target:
                    targets.append(target)
        return targets
