"""Kernel protocol: dataclasses shared across the kernel."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
class PhaseConfig:
    name: str
    kind: str  # role | script | gate | loop | human
    route: str = "static"
    role: str | None = None
    next: str | None = None

    # Enum routing: declared status value -> phase name (or reserved route).
    on_status: dict[str, str] = field(default_factory=dict)
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
        targets.extend(self.on_status.values())
        for cfg in (self.on_invalid, self.on_failure, self.on_fail):
            if isinstance(cfg, str) and cfg:
                targets.append(cfg)
            elif isinstance(cfg, dict):
                target = cfg.get("target")
                if isinstance(target, str) and target:
                    targets.append(target)
        return targets
