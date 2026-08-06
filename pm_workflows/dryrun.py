"""Static check and dry-run simulation of a workflow manifest.

No LLM, no coding agent, no script or skill is ever executed. The manifest
declares every route as a plain value in YAML — which status maps to which
phase, which gate failure retries where, how many times — and that is cold,
hard data the kernel itself never has to guess at either. This module reads
the same manifest the kernel would, and answers three questions from that
data alone:

  * Are the files every role/gate/script/child-workflow phase points at
    actually there?
  * What can the routing graph actually do — which phases cycle, what bounds
    that cycling, what is the best and worst case dispatch count?
  * What does a typical run look like? (a weighted random walk over the same
    graph, since "typical" is not something static bounds can answer)

The bound math mirrors the kernel's own retry accounting exactly (see
`pm_workflows.kernel.Kernel._item_scope` and `_on_failure`): a phase inside a
loop body is bounded per work item, a phase outside any loop is bounded per
task lifetime, and whichever numeric cap is smaller — an explicit
`max_attempts` on a self-targeting retry, or the ambient
`failure_policy.max_attempts_per_phase` / loop `max_iterations` backstop —
is the one that actually fires first.
"""
from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import ManifestError, Workflow, parse_workflow
from .protocol import RESERVED_ROUTES, ROUTE_EXIT_LOOP, ROUTE_NEXT_ITEM, ROUTE_STOP, PhaseConfig

TERMINAL_SUCCESS = "<success>"
TERMINAL_STOP = "<stop>"
DISPATCH_KINDS = frozenset({"role", "gate", "script", "human", "workflow"})


# --------------------------------------------------------------------- files

@dataclass
class FileCheck:
    kind: str          # skill | gate | script | child_workflow
    owner: str          # "role 'x'" or "phase 'y'"
    declared: str
    resolved: str
    exists: bool


def resolve_path(base_dir: Path, relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else (base_dir / path).resolve()


def resolve_child_manifest(base_dir: Path, name: str) -> Path | None:
    """Mirrors `Kernel._resolve_child_manifest` without raising."""
    candidate = Path(name)
    choices = [candidate] if candidate.is_absolute() else [
        base_dir / "workflows" / name / f"{name}.workflow.md",
        base_dir / "workflows" / f"{name}.workflow.md",
        base_dir / name / f"{name}.workflow.md",
        base_dir / f"{name}.workflow.md",
    ]
    for choice in choices:
        if choice.is_file():
            return choice.resolve()
    return None


def _walk_phases(phases: list[PhaseConfig]):
    for phase in phases:
        yield phase
        yield from _walk_phases(phase.body)


def check_files(workflow: Workflow, base_dir: Path) -> list[FileCheck]:
    checks: list[FileCheck] = []
    for role in workflow.roles.values():
        path = resolve_path(base_dir, role.skill)
        checks.append(FileCheck(
            kind="skill", owner=f"role '{role.name}'", declared=role.skill,
            resolved=str(path), exists=path.is_file(),
        ))
    for phase in _walk_phases(workflow.phases):
        if phase.kind == "gate" and phase.predicate:
            path = resolve_path(base_dir, phase.predicate)
            checks.append(FileCheck(
                kind="gate", owner=f"phase '{phase.name}'", declared=phase.predicate,
                resolved=str(path), exists=path.is_file(),
            ))
        elif phase.kind == "script" and phase.script:
            path = resolve_path(base_dir, phase.script)
            checks.append(FileCheck(
                kind="script", owner=f"phase '{phase.name}'", declared=phase.script,
                resolved=str(path), exists=path.is_file(),
            ))
        elif phase.kind == "workflow" and phase.workflow:
            path = resolve_child_manifest(base_dir, phase.workflow)
            checks.append(FileCheck(
                kind="child_workflow", owner=f"phase '{phase.name}'",
                declared=phase.workflow,
                resolved=str(path) if path else "(not found)",
                exists=path is not None,
            ))
    return checks


def check_child_workflows(
    workflow: Workflow, base_dir: Path, max_depth: int, _visited: frozenset[str] = frozenset(),
) -> list[str]:
    """Recursively verify every child manifest a `kind: workflow` phase names
    parses and that its own files exist. Returns human-readable errors,
    prefixed with the chain of phases that led to them."""
    errors: list[str] = []
    if max_depth < 0:
        return errors
    for phase in _walk_phases(workflow.phases):
        if phase.kind != "workflow" or not phase.workflow:
            continue
        path = resolve_child_manifest(base_dir, phase.workflow)
        if path is None:
            continue  # already reported by check_files
        key = str(path)
        if key in _visited:
            errors.append(
                f"phase '{phase.name}': child workflow '{phase.workflow}' recurses "
                f"back to an ancestor ({path}) — this can only terminate on depth budget"
            )
            continue
        try:
            child = parse_workflow(path, {
                "TASK_ID": "dryrun", "WORKSPACE": str(base_dir), "TARGET": str(base_dir),
                "TASK_DIR": str(base_dir / "agents" / "tasks" / "dryrun"), "BASE": str(base_dir),
            })
        except ManifestError as exc:
            errors.append(f"phase '{phase.name}': child workflow '{phase.workflow}' is invalid: {exc}")
            continue
        for check in check_files(child, base_dir):
            if not check.exists:
                errors.append(
                    f"phase '{phase.name}': child workflow '{phase.workflow}' "
                    f"{check.kind} {check.owner} not found: {check.resolved}"
                )
        errors.extend(check_child_workflows(child, base_dir, max_depth - 1, _visited | {key}))
    return errors


# -------------------------------------------------------------------- graph

@dataclass
class Edge:
    outcome: str
    target: str              # phase name, TERMINAL_SUCCESS, or TERMINAL_STOP
    edge_kind: str            # advance | retry | terminal
    max_attempts: int | None  # explicit cap from this rule; None = not capped by this rule alone
    note: str = ""
    explicit_max_attempts: bool = True  # False when max_attempts fell back to the 999 default


@dataclass
class PhaseNode:
    name: str
    kind: str
    loop: str | None                    # enclosing loop's name, if this is a body phase
    edges: list[Edge] = field(default_factory=list)
    backstop_limit: int | None = None    # loop.max_iterations if in a loop, else max_attempts_per_phase

    @property
    def self_retry_cap(self) -> int | None:
        """The tightest explicit cap on a rule that retries this phase into itself."""
        caps = [
            e.max_attempts for e in self.edges
            if e.edge_kind == "retry" and e.target == self.name and e.max_attempts is not None
        ]
        return min(caps) if caps else None

    @property
    def effective_repeat_bound(self) -> int:
        """Worst-case number of times this phase can be dispatched.

        Whichever numeric cap is smaller — an explicit self-retry
        `max_attempts`, or the ambient backstop — is the one that actually
        fires first in the kernel.
        """
        candidates = [c for c in (self.backstop_limit, self.self_retry_cap) if c is not None]
        return min(candidates) if candidates else 999


def _normalize_target(
    raw: str | None, enclosing_loop: PhaseConfig | None,
) -> str:
    if not raw:
        return TERMINAL_SUCCESS
    if raw == ROUTE_STOP:
        return TERMINAL_STOP
    if raw == ROUTE_NEXT_ITEM:
        return enclosing_loop.name if enclosing_loop else TERMINAL_STOP
    if raw == ROUTE_EXIT_LOOP:
        return _normalize_target(enclosing_loop.exit if enclosing_loop else None, None)
    return raw


def _classify_failure(
    config: Any, phase: PhaseConfig, enclosing_loop: PhaseConfig | None,
) -> Edge:
    """Mirrors `Kernel._on_failure`'s routing decision, without running it."""
    if config is None:
        return Edge("invalid", TERMINAL_STOP, "terminal", 1, "no failure route declared")
    if isinstance(config, str):
        return Edge(
            "invalid", _normalize_target(config, enclosing_loop), "advance", None,
            "bare route — no retry cap of its own, still backstopped",
        )
    if not isinstance(config, dict):
        return Edge("invalid", TERMINAL_STOP, "terminal", 1, "malformed failure route")

    action = config.get("action", "retry_with_feedback")
    target = _normalize_target(config.get("target", phase.name), enclosing_loop)

    if action == "route_to":
        return Edge("invalid", target, "advance", None, "route_to — uncapped by this rule")
    if action in {"stop", "stop_subtree", "stop_with_failure", "fail"}:
        return Edge("invalid", TERMINAL_STOP, "terminal", 1, f"action={action}")
    if action not in {"retry_with_feedback", "retry_child_clean"}:
        return Edge("invalid", TERMINAL_STOP, "terminal", 1, f"unknown action '{action}'")

    max_attempts = config.get("max_attempts", 999)
    explicit = "max_attempts" in config
    note = f"retry cap {max_attempts}" + ("" if explicit else " (default; not declared explicitly)")
    return Edge("invalid", target, "retry", max_attempts, note, explicit_max_attempts=explicit)


def build_graph(workflow: Workflow) -> dict[str, PhaseNode]:
    nodes: dict[str, PhaseNode] = {}

    def backstop_for(enclosing_loop: PhaseConfig | None) -> int:
        if enclosing_loop is not None:
            return enclosing_loop.max_iterations
        return workflow.failure_policy.max_attempts_per_phase

    def add(phase: PhaseConfig, enclosing_loop: PhaseConfig | None) -> None:
        node = PhaseNode(
            name=phase.name, kind=phase.kind,
            loop=enclosing_loop.name if enclosing_loop else None,
        )
        if phase.kind == "role":
            role = workflow.roles[phase.role or ""]
            declared = role.result_contract.status_values
            if declared:
                for status in declared:
                    target = _normalize_target(phase.on_status.get(status), enclosing_loop)
                    node.edges.append(Edge(status, target, "advance", None))
            elif phase.next:
                node.edges.append(
                    Edge("*", _normalize_target(phase.next, enclosing_loop), "advance", None)
                )
            else:
                for status, raw_target in phase.on_status.items():
                    node.edges.append(
                        Edge(status, _normalize_target(raw_target, enclosing_loop), "advance", None)
                    )
            node.edges.append(_classify_failure(phase.on_invalid, phase, enclosing_loop))
            node.backstop_limit = backstop_for(enclosing_loop)

        elif phase.kind == "workflow":
            declared = phase.child_result.statuses if phase.child_result else []
            for status in declared:
                target = _normalize_target(phase.on_status.get(status), enclosing_loop)
                node.edges.append(Edge(status, target, "advance", None))
            node.edges.append(_classify_failure(phase.on_invalid, phase, enclosing_loop))
            invocation_limit = phase.limits.max_attempts if phase.limits else 1
            node.backstop_limit = min(backstop_for(enclosing_loop), invocation_limit)

        elif phase.kind in {"gate", "script"}:
            pass_target = _normalize_target(phase.on_pass or phase.next, enclosing_loop)
            node.edges.append(Edge("pass", pass_target, "advance", None))
            fail_config = phase.on_fail if phase.on_fail else phase.on_failure
            node.edges.append(_classify_failure(fail_config, phase, enclosing_loop))
            node.backstop_limit = backstop_for(enclosing_loop)

        elif phase.kind == "human":
            node.edges.append(Edge("answered", _normalize_target(phase.next, enclosing_loop), "advance", None))
            node.backstop_limit = backstop_for(enclosing_loop)

        elif phase.kind == "loop":
            node.edges.append(Edge("pending", phase.body[0].name if phase.body else TERMINAL_STOP, "advance", None))
            node.edges.append(Edge("exhausted", _normalize_target(phase.exit, None), "advance", None))
            if phase.on_failure is not None:
                node.edges.append(_classify_failure(phase.on_failure, phase, None))
            else:
                node.edges.append(Edge("loop_own_max_iterations", TERMINAL_STOP, "terminal", 1, "no on_failure declared"))
            node.backstop_limit = phase.max_iterations

        nodes[phase.name] = node

    for phase in workflow.phases:
        add(phase, None)
        if phase.kind == "loop":
            for body_phase in phase.body:
                add(body_phase, phase)

    return nodes


# -------------------------------------------------------------- cycle bounds

@dataclass
class CycleReport:
    phases: list[str]
    per_item: bool
    bound_per_phase: dict[str, int]
    total_bound: int
    note: str


def find_cycles(nodes: dict[str, PhaseNode]) -> list[CycleReport]:
    """Strongly connected components of size > 1, plus self-loops.

    Only `advance`/`retry` edges count — a `terminal` edge cannot be part of
    a cycle by construction. Loop control nodes are excluded: every body
    phase's `next_item` edge points back at its enclosing loop, so the loop
    node would otherwise show up in every single body cycle it contains —
    that repetition is the loop doing its job, already reported separately
    under STRUCTURE, not a retry cycle worth flagging on its own.
    """
    real = {name: node for name, node in nodes.items() if node.kind != "loop"}
    graph: dict[str, set[str]] = {
        name: {e.target for e in node.edges if e.edge_kind != "terminal" and e.target in real}
        for name, node in real.items()
    }

    index_counter = [0]
    stack: list[str] = []
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    result: list[list[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in graph.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w):
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            if len(component) > 1 or v in graph.get(v, ()):
                result.append(component)

    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(real) * 4 + 100))
    for name in real:
        if name not in index:
            strongconnect(name)

    reports = []
    for component in result:
        per_item = any(real[n].loop is not None for n in component)
        bounds = {n: real[n].effective_repeat_bound for n in component}
        reports.append(CycleReport(
            phases=sorted(component),
            per_item=per_item,
            bound_per_phase=bounds,
            total_bound=sum(bounds.values()),
            note="scoped per work item" if per_item else "scoped for the task's lifetime",
        ))
    return sorted(reports, key=lambda r: (-r.total_bound, r.phases))


def distance_to_success(nodes: dict[str, PhaseNode]) -> dict[str, float]:
    """Dijkstra on the reverse graph: dispatches from each phase to the
    nearest terminal-success node, assuming every phase along the way takes
    its most cooperative outcome.

    Used to weight Monte Carlo sampling: a role's declared outcomes are not
    equally likely in practice, and the manifest gives no probabilities to
    go on, so the outcome that actually moves toward success is treated as
    the common case and the rest split the remaining probability mass.
    """
    reverse: dict[str, list[tuple[str, int]]] = {}
    for name, node in nodes.items():
        weight = 1 if node.kind in DISPATCH_KINDS else 0
        for edge in node.edges:
            if edge.edge_kind == "advance":
                reverse.setdefault(edge.target, []).append((name, weight))

    dist: dict[str, float] = {TERMINAL_SUCCESS: 0.0}
    heap: list[tuple[float, str]] = [(0.0, TERMINAL_SUCCESS)]
    while heap:
        d, current = heapq.heappop(heap)
        if d > dist.get(current, float("inf")):
            continue
        for prev, weight in reverse.get(current, []):
            candidate = d + weight
            if candidate < dist.get(prev, float("inf")):
                dist[prev] = candidate
                heapq.heappush(heap, (candidate, prev))
    return dist


# --------------------------------------------------------------- best/worst

@dataclass
class PathResult:
    dispatches: int
    path: list[str]
    reachable: bool


def _shortest_path(
    nodes: dict[str, PhaseNode],
    start: str,
    is_goal,
) -> PathResult:
    """BFS over `advance` edges only — the optimistic case always takes the
    outcome that moves fastest toward the goal, never a retry."""
    from collections import deque

    if is_goal(start):
        return PathResult(0, [start], True)

    seen = {start}
    queue: deque[tuple[str, int, list[str]]] = deque([(start, 0, [start])])
    while queue:
        current, cost, path = queue.popleft()
        node = nodes.get(current)
        if node is None:
            continue
        for edge in node.edges:
            if edge.edge_kind != "advance":
                continue
            weight = 1 if node.kind in DISPATCH_KINDS else 0
            new_cost = cost + weight
            if is_goal(edge.target):
                return PathResult(new_cost, path + [edge.target], True)
            if edge.target in nodes and edge.target not in seen:
                seen.add(edge.target)
                queue.append((edge.target, new_cost, path + [edge.target]))
    return PathResult(-1, [], False)


def best_case(workflow: Workflow, nodes: dict[str, PhaseNode]) -> dict[str, Any]:
    start = workflow.phases[0].name
    zero_items = _shortest_path(nodes, start, lambda t: t == TERMINAL_SUCCESS)

    loops = [p for p in workflow.phases if p.kind == "loop"]
    one_item: PathResult | None = None
    if loops:
        loop = loops[0]
        entry = loop.body[0].name
        # Only paths back through `next_item` count as "one item done" — a
        # path out via `exit_loop` skips the rest of the body, which is not
        # what "how long does the whole item take" is asking.
        per_item = _shortest_path(nodes, entry, lambda t: t == loop.name)
        prefix = _shortest_path(nodes, start, lambda t: t == loop.name)
        suffix = _shortest_path(nodes, loop.name, lambda t: t == TERMINAL_SUCCESS)
        if prefix.reachable and per_item.reachable and suffix.reachable:
            one_item = PathResult(
                prefix.dispatches + per_item.dispatches + suffix.dispatches,
                prefix.path + per_item.path[1:] + suffix.path[1:],
                True,
            )

    return {
        "zero_items": zero_items,
        "one_item": one_item,
    }


def worst_case(workflow: Workflow, nodes: dict[str, PhaseNode]) -> dict[str, Any]:
    """A phase only gets more than one worst-case dispatch if something can
    actually bring execution back to it — being part of a detected cycle.
    A phase with no retry route and nothing routing back to it (e.g. a
    trailing `finish` script with no `on_failure`) runs at most once,
    however generous its nominal `effective_repeat_bound` looks; scoring it
    at the 999-default backstop would wildly overstate the ceiling for
    exactly the phases that are safest.
    """
    loop_names = {p.name for p in workflow.phases if p.kind == "loop"}
    cyclic = {name for cycle in find_cycles(nodes) for name in cycle.phases}
    fixed = 0
    per_item = 0
    for name, node in nodes.items():
        if node.kind not in DISPATCH_KINDS:
            continue
        bound = node.effective_repeat_bound if name in cyclic else 1
        if node.loop is not None:
            per_item += bound
        elif name not in loop_names:
            fixed += bound

    loops = [p for p in workflow.phases if p.kind == "loop"]
    loop_cap = min((loop.max_iterations for loop in loops), default=None)
    return {
        "fixed": fixed,
        "per_item": per_item,
        "loop_cap": loop_cap,
        "formula": f"{fixed} + {per_item} * N" if loops else f"{fixed}",
        "examples": {
            n: fixed + per_item * min(n, loop_cap or n) for n in (1, 5, 20)
        } if loops else {},
    }


# --------------------------------------------------------------- monte carlo

@dataclass
class MonteCarloResult:
    runs: int
    outcomes: dict[str, int]
    dispatch_counts: list[int]
    phase_visits: dict[str, int]

    def summary(self) -> dict[str, Any]:
        counts = sorted(self.dispatch_counts) or [0]
        n = len(counts)
        return {
            "runs": self.runs,
            "outcomes_pct": {k: round(100 * v / self.runs, 1) for k, v in self.outcomes.items()},
            "dispatches_min": counts[0],
            "dispatches_max": counts[-1],
            "dispatches_mean": round(sum(counts) / n, 1),
            "dispatches_median": counts[n // 2],
            "dispatches_p90": counts[min(n - 1, int(n * 0.9))],
            "hottest_phases": sorted(
                ((name, round(v / self.runs, 2)) for name, v in self.phase_visits.items()),
                key=lambda kv: -kv[1],
            )[:8],
        }


def simulate(
    workflow: Workflow,
    nodes: dict[str, PhaseNode],
    *,
    runs: int = 2000,
    items_range: tuple[int, int] = (1, 5),
    p_invalid: float = 0.05,
    p_fail: float = 0.15,
    p_primary: float = 0.85,
    seed: int | None = None,
) -> MonteCarloResult:
    rng = random.Random(seed)
    start = workflow.phases[0].name
    outcomes: dict[str, int] = {"success": 0, "stopped": 0, "exhausted": 0}
    dispatch_counts: list[int] = []
    phase_visits: dict[str, int] = {}
    distance = distance_to_success(nodes)

    def pick_advance(advance: list[Edge]) -> Edge:
        """The manifest gives no outcome probabilities, so the edge that
        actually moves toward success (the "primary" outcome, e.g. `done`
        over `blocked`) gets `p_primary` by default; the rest of a role's
        declared outcomes share what's left, uniformly."""
        if len(advance) == 1:
            return advance[0]
        primary = min(
            advance,
            key=lambda e: 0.0 if e.target == TERMINAL_SUCCESS else distance.get(e.target, float("inf")),
        )
        if rng.random() < p_primary:
            return primary
        rest = [e for e in advance if e is not primary]
        return rng.choice(rest) if rest else primary

    for _ in range(runs):
        attempts: dict[tuple[str, str | None], int] = {}
        current = start
        current_item: str | None = None
        # Sampled once per loop phase per run, then only ever decremented —
        # regenerating it every time it hits zero would mean the loop never
        # legitimately exhausts, and every run would run until something
        # else (an unlucky "blocked") stopped it.
        items_remaining: dict[str, int] = {}
        outcome = "success"
        dispatches = 0
        guard = 0
        while True:
            guard += 1
            if guard > 200_000:
                outcome = "exhausted"
                break
            node = nodes.get(current)
            if node is None:
                break
            if node.kind in DISPATCH_KINDS:
                # Mirrors `Kernel._over_phase_budget`/`_on_failure` exactly: a
                # phase gets dispatched at most `effective_repeat_bound`
                # times. The kernel checks this *before* dispatching (the
                # backstop) or right after the bound-th failure (an explicit
                # retry cap) — either way, there is never a (bound + 1)-th
                # dispatch, so the check has to happen before counting this
                # one, not after.
                key = (current, current_item if node.loop else None)
                if attempts.get(key, 0) >= node.effective_repeat_bound:
                    outcome = "exhausted"
                    break
                attempts[key] = attempts.get(key, 0) + 1
                dispatches += 1
                phase_visits[current] = phase_visits.get(current, 0) + 1

            if node.kind == "loop":
                if current not in items_remaining:
                    items_remaining[current] = rng.randint(*items_range)
                target_edge = next(e for e in node.edges if e.outcome == "exhausted")
                enter_edge = next(e for e in node.edges if e.outcome == "pending")
                if items_remaining[current] > 0:
                    items_remaining[current] -= 1
                    current_item = f"item-{items_remaining[current]}"
                    current = enter_edge.target
                else:
                    current = target_edge.target
                continue

            advance = [e for e in node.edges if e.edge_kind == "advance"]
            invalid = next((e for e in node.edges if e.outcome == "invalid"), None)
            chosen: Edge
            if node.kind in {"gate", "script"}:
                pass_edge = next(e for e in advance if e.outcome == "pass")
                chosen = invalid if (invalid and rng.random() < p_fail) else pass_edge
            elif invalid is not None and rng.random() < p_invalid:
                chosen = invalid
            elif advance:
                chosen = pick_advance(advance)
            elif invalid is not None:
                chosen = invalid
            else:
                outcome = "stopped"
                break

            target = chosen.target
            if target == TERMINAL_SUCCESS:
                outcome = "success"
                break
            if target == TERMINAL_STOP:
                outcome = "stopped"
                break
            if target == node.loop and node.loop is not None:
                # next_item — head back to the enclosing loop.
                current = node.loop
                current_item = None
                continue
            if target not in nodes:
                outcome = "stopped"
                break
            current = target

        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        dispatch_counts.append(dispatches)

    return MonteCarloResult(runs, outcomes, dispatch_counts, phase_visits)


# ------------------------------------------------------------------ report

@dataclass
class DryRunReport:
    workflow_name: str
    manifest_path: str
    base_dir: str
    file_checks: list[FileCheck]
    child_errors: list[str]
    nodes: dict[str, PhaseNode]
    cycles: list[CycleReport]
    best: dict[str, Any]
    worst: dict[str, Any]
    monte_carlo: MonteCarloResult | None
    parse_error: str | None
    warnings: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        errors = []
        if self.parse_error:
            errors.append(self.parse_error)
        for check in self.file_checks:
            if not check.exists:
                errors.append(f"{check.kind} {check.owner} not found: {check.resolved} (declared: {check.declared})")
        errors.extend(self.child_errors)
        return errors

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "workflow": self.workflow_name,
            "manifest": self.manifest_path,
            "errors": self.errors,
            "warnings": self.warnings,
            "file_checks": [vars(c) for c in self.file_checks],
            "cycles": [
                {
                    "phases": c.phases, "per_item": c.per_item,
                    "bound_per_phase": c.bound_per_phase, "total_bound": c.total_bound,
                    "note": c.note,
                }
                for c in self.cycles
            ],
            "best_case": {
                name: (
                    {"dispatches": result.dispatches, "path": result.path}
                    if result and result.reachable else None
                )
                for name, result in (
                    ("zero_items", self.best.get("zero_items")),
                    ("one_item", self.best.get("one_item")),
                )
            },
            "worst_case": {k: v for k, v in self.worst.items() if k != "loop_cap"} if self.worst else {},
            "monte_carlo": self.monte_carlo.summary() if self.monte_carlo else None,
        }

    def render_text(self) -> str:
        lines = [f"Workflow: {self.workflow_name}  ({self.manifest_path})", ""]

        if self.parse_error:
            lines += ["MANIFEST", f"  INVALID: {self.parse_error}", ""]
            return "\n".join(lines)

        lines.append("FILES")
        missing = [c for c in self.file_checks if not c.exists]
        for check in missing:
            lines.append(f"  MISSING  {check.kind} {check.owner}: {check.resolved}")
        lines.append(f"  {len(self.file_checks)} checked, {len(missing)} missing")
        for error in self.child_errors:
            lines.append(f"  CHILD WORKFLOW ERROR  {error}")
        lines.append("")

        lines.append("STRUCTURE")
        loop_nodes = [n for n in self.nodes.values() if n.kind == "loop"]
        lines.append(f"  {len(self.nodes)} phases total ({len(loop_nodes)} loop(s))")
        for loop in loop_nodes:
            lines.append(f"  loop '{loop.name}': max_iterations={loop.backstop_limit}")
        lines.append("")

        lines.append("CYCLES / RETRY BUDGETS")
        if not self.cycles:
            lines.append("  none — the routing graph has no cycles")
        for cycle in self.cycles:
            scope = "per item" if cycle.per_item else "per task"
            lines.append(
                f"  {' <-> '.join(cycle.phases)}: worst case {cycle.total_bound} dispatches "
                f"({scope}, {cycle.note})"
            )
        lines.append("")

        lines.append("BEST / WORST CASE (dispatches = role/gate/script/human/workflow executions)")
        zero = self.best.get("zero_items")
        one = self.best.get("one_item")
        if zero and zero.reachable:
            lines.append(f"  best case, 0 work items:  {zero.dispatches} dispatches -> success")
            lines.append(f"    path: {' -> '.join(zero.path)}")
        if one and one.reachable:
            lines.append(f"  best case, 1 work item:   {one.dispatches} dispatches -> success")
            lines.append(f"    path: {' -> '.join(one.path)}")
        if self.worst:
            lines.append(f"  worst case formula:       {self.worst['formula']}")
            for n, total in self.worst.get("examples", {}).items():
                lines.append(f"    N={n:<4} -> {total} dispatches (worst case)")
            lines.append(
                "  (a conservative ceiling: real runs stop at the first exhaustion, and "
                "each phase's bound is counted independently — a gate that retries a "
                "shared target is not further tightened by that target's own cap, so "
                "this can overstate the true ceiling)"
            )
        lines.append("")

        if self.monte_carlo:
            summary = self.monte_carlo.summary()
            lines.append(f"MONTE CARLO ({summary['runs']} simulated runs)")
            lines.append(f"  outcomes: {summary['outcomes_pct']}")
            lines.append(
                f"  dispatches: min {summary['dispatches_min']}  "
                f"mean {summary['dispatches_mean']}  median {summary['dispatches_median']}  "
                f"p90 {summary['dispatches_p90']}  max {summary['dispatches_max']}"
            )
            lines.append(f"  hottest phases (avg visits/run): {summary['hottest_phases']}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        lines.append(f"RESULT: {len(self.errors)} blocking error(s), {len(self.warnings)} warning(s)")
        return "\n".join(lines)


DEFAULT_HOT_THRESHOLD = 20


def _build_warnings(workflow: Workflow, nodes: dict[str, PhaseNode], hot_threshold: int) -> list[str]:
    warnings: list[str] = []
    loop_phases = [p for p in workflow.phases if p.kind == "loop"]
    if len(loop_phases) > 1:
        warnings.append(
            "more than one work-item loop shares the same pending-items pool "
            f"({', '.join(p.name for p in loop_phases)}); a later loop only sees "
            "items an earlier loop has not already marked complete"
        )
    for name, node in nodes.items():
        for edge in node.edges:
            if (
                edge.edge_kind == "retry" and not edge.explicit_max_attempts
                and edge.max_attempts and edge.max_attempts >= hot_threshold
            ):
                warnings.append(
                    f"phase '{name}' relies on the default max_attempts={edge.max_attempts} "
                    f"for its '{edge.outcome}' route to '{edge.target}' — consider declaring "
                    "a tighter cap explicitly"
                )
        if node.effective_repeat_bound >= hot_threshold and any(
            e.edge_kind != "terminal" and e.target in nodes for e in node.edges
        ):
            scope = "per work item" if node.loop else "per task"
            warnings.append(
                f"phase '{name}' can be dispatched up to {node.effective_repeat_bound} "
                f"times ({scope}) before the workflow gives up on it — verify this is intentional"
            )
    return warnings


def check_workflow(
    manifest_path: Path,
    base_dir: Path,
    *,
    max_child_depth: int | None = None,
    monte_carlo_runs: int = 2000,
    items_range: tuple[int, int] = (1, 5),
    p_invalid: float = 0.05,
    p_fail: float = 0.15,
    p_primary: float = 0.85,
    hot_threshold: int = DEFAULT_HOT_THRESHOLD,
    seed: int | None = None,
    skip_monte_carlo: bool = False,
) -> DryRunReport:
    manifest_path = Path(manifest_path)
    base_dir = Path(base_dir)
    variables = {
        "TASK_ID": "dryrun", "WORKSPACE": str(base_dir), "TARGET": str(base_dir),
        "TASK_DIR": str(base_dir / "agents" / "tasks" / "dryrun"), "BASE": str(base_dir),
    }
    try:
        workflow = parse_workflow(manifest_path, variables)
    except ManifestError as exc:
        return DryRunReport(
            workflow_name=manifest_path.stem, manifest_path=str(manifest_path),
            base_dir=str(base_dir), file_checks=[], child_errors=[], nodes={},
            cycles=[], best={}, worst={}, monte_carlo=None, parse_error=str(exc),
        )

    file_checks = check_files(workflow, base_dir)
    depth = workflow.budgets.max_depth if max_child_depth is None else max_child_depth
    child_errors = check_child_workflows(workflow, base_dir, depth)

    nodes = build_graph(workflow)
    cycles = find_cycles(nodes)
    best = best_case(workflow, nodes)
    worst = worst_case(workflow, nodes)
    monte_carlo = None if skip_monte_carlo else simulate(
        workflow, nodes, runs=monte_carlo_runs, items_range=items_range,
        p_invalid=p_invalid, p_fail=p_fail, p_primary=p_primary, seed=seed,
    )
    warnings = _build_warnings(workflow, nodes, hot_threshold)

    return DryRunReport(
        workflow_name=workflow.name, manifest_path=str(manifest_path), base_dir=str(base_dir),
        file_checks=file_checks, child_errors=child_errors, nodes=nodes, cycles=cycles,
        best=best, worst=worst, monte_carlo=monte_carlo, parse_error=None, warnings=warnings,
    )


# ---------------------------------------------------------------------- cli

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Static check and dry-run simulation of a pm-workflows manifest. "
            "Runs no LLM, no coding agent, no script or skill."
        ),
    )
    parser.add_argument(
        "workspace", nargs="?", default=".",
        help="Unused by the checker itself; accepted so this doubles as a "
             "gate/script phase (the kernel always passes the workspace first).",
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Workflow manifest to check.")
    parser.add_argument("--base", required=True, type=Path, help="Instruction base holding workflows/, skills/, scripts/.")
    parser.add_argument("--runs", type=int, default=2000, help="Monte Carlo run count (default 2000).")
    parser.add_argument("--items-min", type=int, default=1)
    parser.add_argument("--items-max", type=int, default=5)
    parser.add_argument("--p-invalid", type=float, default=0.05, help="Simulated contract-violation rate.")
    parser.add_argument("--p-fail", type=float, default=0.15, help="Simulated gate/script failure rate.")
    parser.add_argument(
        "--p-primary", type=float, default=0.85,
        help="Simulated probability a role/child-workflow takes the outcome "
             "that actually moves toward success, vs. its other declared outcomes.",
    )
    parser.add_argument("--hot-threshold", type=int, default=DEFAULT_HOT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-monte-carlo", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable report instead of text.")
    parser.add_argument("--fail-on-warnings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_workflow(
        args.manifest, args.base,
        monte_carlo_runs=args.runs, items_range=(args.items_min, args.items_max),
        p_invalid=args.p_invalid, p_fail=args.p_fail, p_primary=args.p_primary,
        hot_threshold=args.hot_threshold, seed=args.seed, skip_monte_carlo=args.no_monte_carlo,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.render_text())
        print()
        print(json.dumps({"ok": report.ok, "errors": report.errors}))

    ok = report.ok and not (args.fail_on_warnings and report.warnings)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
