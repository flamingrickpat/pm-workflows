"""Parse and validate workflow manifests (.workflow.md with YAML frontmatter).

The manifest is the contract. Everything the kernel will do — which role runs,
which check gates it, where each declared outcome routes — is declared here and
validated at load time. A workflow that can reach an undeclared outcome or an
unknown phase fails to load, before a single token is spent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .protocol import (
    RESERVED_ROUTES,
    ROUTE_EXIT_LOOP,
    ROUTE_NEXT_ITEM,
    PhaseConfig,
    ResultContract,
    RoleConfig,
)

# Iterator sources the kernel knows how to enumerate.
ITERATOR_SOURCES = frozenset({"pending_work_items"})
VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_.]*)\}")


class ManifestError(ValueError):
    """A workflow manifest is malformed or internally inconsistent."""


@dataclass
class DriverConfig:
    # Agent selection is normally a runtime concern. These fields remain for
    # backwards-compatible manifests and shared execution defaults.
    kind: str = ""
    model: str = ""
    effort: str = ""
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    add_dirs: list[str] = field(default_factory=list)
    trace: bool = True
    timeout_seconds: int = 7200


@dataclass
class CheckpointConfig:
    kind: str = "git"
    repo_path: str = ""
    base_revision: str = ""


@dataclass
class HumanResolverConfig:
    mode: str = "forbid"  # forbid | stdin
    on_no_default: str = "fail"


@dataclass
class FailurePolicyConfig:
    max_attempts_per_phase: int = 999
    result_classes: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetsConfig:
    max_total_tokens: int = 999_999_999
    max_phase_tokens: int = 999_999_999
    max_wallclock_seconds: int = 999_999_999
    max_depth: int = 1


@dataclass
class Workflow:
    name: str
    description: str
    profile: str
    driver: DriverConfig
    checkpoint_backend: CheckpointConfig | None
    roles: dict[str, RoleConfig]
    phases: list[PhaseConfig]
    failure_policy: FailurePolicyConfig
    human_resolver: HumanResolverConfig
    budgets: BudgetsConfig
    body: str = ""
    path: str = ""

    def phase_by_name(self, name: str) -> PhaseConfig | None:
        """Find a phase by name at any nesting depth."""
        for phase in self.phases:
            if phase.name == name:
                return phase
            nested = phase.body_by_name(name)
            if nested is not None:
                return nested
        return None

    def loop_containing(self, phase_name: str) -> PhaseConfig | None:
        for phase in self.phases:
            if phase.kind == "loop" and phase.body_by_name(phase_name) is not None:
                return phase
        return None


def substitute(value: Any, variables: dict[str, str]) -> Any:
    """Expand ${VAR} references. Unknown names are left untouched on purpose:
    a typo shows up in the failing path instead of silently becoming empty."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            # ${WORKSPACE.path} is accepted as a legacy spelling of ${WORKSPACE}.
            if key not in variables and "." in key:
                key = key.split(".", 1)[0]
            return variables.get(key, match.group(0))
        return VAR_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [substitute(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, variables) for k, v in value.items()}
    return value


def _split_frontmatter(path: Path, text: str) -> tuple[str, str]:
    """Split on `---` delimiter *lines*.

    Not on the first `---` anywhere in the text: a `# ---- section ----`
    comment inside the frontmatter would silently truncate it, and the manifest
    would parse as valid with half its phases missing.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ManifestError(f"{path}: missing YAML frontmatter")
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1:])
    raise ManifestError(f"{path}: frontmatter is never closed by a `---` line")


def _parse_contract(raw: Any) -> ResultContract:
    if not isinstance(raw, dict):
        return ResultContract()
    return ResultContract(
        type=raw.get("type", "json"),
        status_field=raw.get("status_field", "status"),
        schema=raw.get("schema", {}) or {},
    )


def _parse_phase(raw: dict[str, Any], path: Path) -> PhaseConfig:
    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: phase entries must be mappings, got {type(raw)}")
    name = raw.get("name", "")
    if not name:
        raise ManifestError(f"{path}: a phase is missing its name")

    on_status = raw.get("on_status", {}) or {}
    if not isinstance(on_status, dict):
        raise ManifestError(f"{path}: phase '{name}' on_status must be a mapping")

    body_raw = raw.get("body", []) or []
    body = [_parse_phase(entry, path) for entry in body_raw]

    args = raw.get("args", []) or []
    if isinstance(args, str):
        args = [args]

    return PhaseConfig(
        name=name,
        kind=raw.get("kind", "role"),
        route=raw.get("route", "static"),
        role=raw.get("role"),
        next=raw.get("next"),
        on_status={str(k): str(v) for k, v in on_status.items()},
        on_invalid=raw.get("on_invalid"),
        on_pass=raw.get("on_pass"),
        on_fail=raw.get("on_fail"),
        on_failure=raw.get("on_failure"),
        checkpoint_after=raw.get("checkpoint_after", False),
        stop_point=raw.get("stop_point", False),
        predicate=raw.get("predicate"),
        script=raw.get("script"),
        args=[str(a) for a in args],
        question=raw.get("question", "") or "",
        question_from_result=raw.get("question_from_result", "") or "",
        iterator_source=raw.get("iterator_source"),
        body=body,
        exit=raw.get("exit"),
        max_iterations=raw.get("max_iterations", 200),
        workflow=raw.get("workflow"),
        allowed_roles=raw.get("allowed_roles", []) or [],
    )


def _validate(workflow: Workflow) -> None:
    """Fail loudly on any contract hole. This runs before the first dispatch."""
    path = workflow.path
    if not workflow.phases:
        raise ManifestError(f"{path}: no phases defined")

    known: set[str] = set()
    for phase in workflow.phases:
        if phase.name in known:
            raise ManifestError(f"{path}: duplicate phase name '{phase.name}'")
        known.add(phase.name)
        for nested in phase.body:
            if nested.name in known:
                raise ManifestError(f"{path}: duplicate phase name '{nested.name}'")
            known.add(nested.name)

    def check_phase(phase: PhaseConfig, in_loop: bool) -> None:
        if phase.kind not in {"role", "gate", "script", "loop", "human"}:
            raise ManifestError(f"{path}: phase '{phase.name}' has unknown kind '{phase.kind}'")

        for target in phase.route_targets():
            if target in RESERVED_ROUTES:
                if target in {ROUTE_NEXT_ITEM, ROUTE_EXIT_LOOP} and not in_loop:
                    raise ManifestError(
                        f"{path}: phase '{phase.name}' routes to '{target}' "
                        "but is not inside a loop"
                    )
                continue
            if target not in known:
                raise ManifestError(
                    f"{path}: phase '{phase.name}' routes to unknown phase '{target}'"
                )

        if phase.kind == "role":
            if not phase.role:
                raise ManifestError(f"{path}: role phase '{phase.name}' names no role")
            role = workflow.roles.get(phase.role)
            if role is None:
                raise ManifestError(
                    f"{path}: phase '{phase.name}' uses role '{phase.role}' "
                    "which is not in the role catalogue"
                )
            declared = role.result_contract.status_values
            if declared:
                # Every outcome the role is allowed to report must have a route.
                # This is the check that keeps a workflow edit from silently
                # creating a dead end at runtime.
                missing = [v for v in declared if v not in phase.on_status]
                if missing:
                    raise ManifestError(
                        f"{path}: phase '{phase.name}' (role '{phase.role}') has no "
                        f"on_status route for declared status value(s): "
                        f"{', '.join(missing)}"
                    )
                stray = [v for v in phase.on_status if v not in declared]
                if stray:
                    raise ManifestError(
                        f"{path}: phase '{phase.name}' routes status value(s) "
                        f"{', '.join(stray)} that role '{phase.role}' cannot report; "
                        f"declared: {', '.join(declared)}"
                    )
            elif not phase.on_status and not phase.next:
                raise ManifestError(
                    f"{path}: phase '{phase.name}' has neither an on_status map "
                    "(with a status enum in its result contract) nor a next phase"
                )
            if not phase.on_invalid:
                raise ManifestError(
                    f"{path}: role phase '{phase.name}' must declare on_invalid "
                    "for contract violations"
                )

        if phase.kind == "gate":
            if not phase.predicate:
                raise ManifestError(f"{path}: gate phase '{phase.name}' names no predicate")
            if not phase.on_pass:
                raise ManifestError(f"{path}: gate phase '{phase.name}' declares no on_pass")
            if phase.on_fail is None:
                raise ManifestError(f"{path}: gate phase '{phase.name}' declares no on_fail")

        if phase.kind == "script" and not phase.script:
            raise ManifestError(f"{path}: script phase '{phase.name}' names no script")

        if phase.kind == "human":
            if workflow.human_resolver.mode != "stdin":
                raise ManifestError(
                    f"{path}: phase '{phase.name}' is kind 'human' but "
                    f"human_resolver.mode is '{workflow.human_resolver.mode}'"
                )
            if not phase.next:
                raise ManifestError(f"{path}: human phase '{phase.name}' declares no next")
            if not phase.question and not phase.question_from_result:
                raise ManifestError(
                    f"{path}: human phase '{phase.name}' has neither question nor "
                    "question_from_result"
                )

        if phase.kind == "loop":
            if phase.iterator_source not in ITERATOR_SOURCES:
                raise ManifestError(
                    f"{path}: loop '{phase.name}' has unknown iterator_source "
                    f"'{phase.iterator_source}'; known: {sorted(ITERATOR_SOURCES)}"
                )
            if not phase.body:
                raise ManifestError(f"{path}: loop '{phase.name}' has an empty body")
            if not phase.exit:
                raise ManifestError(f"{path}: loop '{phase.name}' declares no exit phase")
            for nested in phase.body:
                if nested.kind == "loop":
                    raise ManifestError(f"{path}: nested loops are not supported ('{nested.name}')")
                check_phase(nested, in_loop=True)

    for phase in workflow.phases:
        check_phase(phase, in_loop=False)

    for name, role in workflow.roles.items():
        if not role.skill:
            raise ManifestError(f"{path}: role '{name}' names no skill")

    _check_reachable(workflow, known)


def _check_reachable(workflow: Workflow, known: set[str]) -> None:
    """A phase nothing can route to is an authoring mistake, not a spare part.

    Usually it means a phase was renamed in one place and not the other, which
    otherwise shows up as a run that quietly never does the step.
    """
    start = workflow.phases[0].name
    reached = {start}
    frontier = [start]
    while frontier:
        phase = workflow.phase_by_name(frontier.pop())
        if phase is None:
            continue
        targets = [t for t in phase.route_targets() if t not in RESERVED_ROUTES]
        if phase.kind == "loop" and phase.body:
            targets.append(phase.body[0].name)
            # `next_item` re-enters the loop, `exit_loop` leaves by its exit.
            targets.extend(
                t for nested in phase.body for t in nested.route_targets()
                if t not in RESERVED_ROUTES
            )
        for target in targets:
            if target in known and target not in reached:
                reached.add(target)
                frontier.append(target)

    unreachable = sorted(known - reached)
    if unreachable:
        raise ManifestError(
            f"{workflow.path}: no route reaches phase(s): {', '.join(unreachable)}. "
            f"A phase nothing can route to is usually a rename that was only "
            f"half applied."
        )


def parse_workflow(path: Path, variables: dict[str, str] | None = None) -> Workflow:
    """Read, expand ${VAR}s, and validate a workflow manifest."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(path, text)

    try:
        fm = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path}: invalid YAML frontmatter: {exc}") from exc
    if not isinstance(fm, dict):
        raise ManifestError(f"{path}: frontmatter is not a mapping")

    fm = substitute(fm, variables or {})

    roles: dict[str, RoleConfig] = {}
    for rname, rraw in (fm.get("roles", {}) or {}).items():
        if not isinstance(rraw, dict):
            raise ManifestError(f"{path}: role '{rname}' is not a mapping")
        raw_deny_access = rraw.get("deny_access")
        deny_access = [] if raw_deny_access is None else raw_deny_access
        if (
            not isinstance(deny_access, list)
            or any(not isinstance(item, str) or not item.strip() for item in deny_access)
        ):
            raise ManifestError(
                f"{path}: role '{rname}' deny_access must be a list of "
                "non-empty path strings"
            )
        roles[rname] = RoleConfig(
            name=rname,
            skill=rraw.get("skill", ""),
            tools=rraw.get("tools", []) or [],
            mcp=rraw.get("mcp", []) or [],
            readable_paths=rraw.get("readable_paths", []) or [],
            writable_paths=rraw.get("writable_paths", []) or [],
            deny_access=deny_access,
            can_call_kernel=rraw.get("can_call_kernel", []) or [],
            result_contract=_parse_contract(rraw.get("result_contract")),
            instruction=rraw.get("instruction", "") or "",
        )

    phases = [_parse_phase(p, path) for p in (fm.get("phases", []) or [])]

    cb_raw = fm.get("checkpoint_backend")
    checkpoint = None
    if isinstance(cb_raw, dict) and cb_raw.get("kind") not in (None, "null", "none"):
        checkpoint = CheckpointConfig(
            kind=cb_raw.get("kind", "git"),
            repo_path=cb_raw.get("repo_path", ""),
            base_revision=cb_raw.get("base_revision", ""),
        )

    fp_raw = fm.get("failure_policy", {}) or {}
    hr_raw = fm.get("human_resolver", {}) or {}
    b_raw = fm.get("budgets", {}) or {}
    drv_raw = fm.get("driver", {}) or {}

    workflow = Workflow(
        name=fm.get("name", path.stem),
        description=fm.get("description", ""),
        profile=fm.get("profile", ""),
        driver=DriverConfig(
            kind=drv_raw.get("kind", "") or "",
            model=drv_raw.get("model", "") or "",
            effort=drv_raw.get("effort", "") or "",
            base_url=drv_raw.get("base_url", "") or "",
            api_key_env=drv_raw.get("api_key_env", "OPENAI_API_KEY"),
            add_dirs=[str(d) for d in (drv_raw.get("add_dirs", []) or [])],
            trace=drv_raw.get("trace", True),
            timeout_seconds=drv_raw.get("timeout_seconds", 7200),
        ),
        checkpoint_backend=checkpoint,
        roles=roles,
        phases=phases,
        failure_policy=FailurePolicyConfig(
            max_attempts_per_phase=fp_raw.get("max_attempts_per_phase", 999),
            result_classes=fp_raw.get("result_classes", {}) or {},
        ),
        human_resolver=HumanResolverConfig(
            mode=hr_raw.get("mode", "forbid"),
            on_no_default=hr_raw.get("on_no_default", "fail"),
        ),
        budgets=BudgetsConfig(
            max_total_tokens=b_raw.get("max_total_tokens", 999_999_999),
            max_phase_tokens=b_raw.get("max_phase_tokens", 999_999_999),
            max_wallclock_seconds=b_raw.get("max_wallclock_seconds", 999_999_999),
            max_depth=b_raw.get("max_depth", 1),
        ),
        body=body,
        path=str(path),
    )

    _validate(workflow)
    return workflow
