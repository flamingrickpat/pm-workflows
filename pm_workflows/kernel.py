"""The kernel: reads a workflow manifest and executes it.

The manifest is the contract. The kernel contributes exactly four things and
nothing else:

  * it dispatches one role per fresh agent session,
  * it runs the checks the workflow declares,
  * it routes on the outcome the workflow declared for that check or status,
  * it appends every attempt to a journal the agents cannot see.

There is no heuristic reading of agent prose, no classifier, and no forward
repair. A failed phase reverts the repo to the last accepted commit and the
same role is dispatched again with the failure as feedback.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.request
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any

from .checkpoint import GitCheckpoint
from .child import (
    copy_declared_artifacts,
    dotted,
    expand_runtime,
    load_reference,
    order_items,
)
from .drivers import build_driver
from .gates import run_gate
from .journal import Journal
from .manifest import ManifestError, Workflow, parse_workflow
from .protocol import (
    ROUTE_EXIT_LOOP,
    ROUTE_NEXT_ITEM,
    ROUTE_STOP,
    AgentResult,
    JournalEntry,
    PhaseConfig,
    RoleConfig,
)
from .ratelimit import TokenLimitError

WORK_ITEM_GLOB = "WI-*.md"


class Kernel:
    def __init__(
        self,
        manifest_path: Path,
        workspace: Path,
        task_id: str,
        task_text: str = "",
        base_dir: Path | None = None,
        coding_agent: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        kernel_data_root: Path | None = None,
        run_id: str | None = None,
        max_agent_requests: int | None = None,
        resume: bool = True,
        driver: Any = None,
        depth_remaining: int | None = None,
        allowed_mcp: set[str] | None = None,
        allowed_effects: set[str] | None = None,
        require_http_mcp: bool = False,
        mcp_http_timeout: float = 5.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.workspace = Path(workspace).resolve()
        self.task_id = task_id
        # `--base`: where central workflows, skills and gate scripts live. Not
        # inside the target repo, so agents cannot edit their own instructions.
        self.base_dir = Path(base_dir or self.manifest_path.parent.parent).resolve()
        self.task_dir = self.workspace / "agents" / "tasks" / task_id

        # Kernel-owned data lives outside the repo and outside the agent's
        # reach: journal, traces, per-attempt results. Keyed by task id so a
        # killed run can be found again and continued.
        root = Path(kernel_data_root) if kernel_data_root else Path.home() / ".pm" / "pm-workflows"
        # Explicit roots are used by embedders and tests; retain their old
        # task-id layout unless the caller supplies a run id. The CLI always
        # supplies a timestamped id for durable workflow metadata.
        self.run_id = run_id or (
            task_id
            if kernel_data_root is not None
            else f"{datetime.now().astimezone():%Y-%m-%d_%H-%M-%S}_{task_id}"
        )
        self.kernel_data = root / self.run_id
        self.kernel_data.mkdir(parents=True, exist_ok=True)

        variables = {
            "TASK_ID": task_id,
            "WORKSPACE": str(self.workspace),
            "TARGET": str(self.workspace),
            "TASK_DIR": str(self.task_dir),
            "BASE": str(self.base_dir),
        }
        self.manifest: Workflow = parse_workflow(self.manifest_path, variables)
        self.depth_remaining = (
            self.manifest.budgets.max_depth
            if depth_remaining is None
            else min(depth_remaining, self.manifest.budgets.max_depth)
        )
        self.allowed_mcp = allowed_mcp
        self.allowed_effects = allowed_effects
        self.require_http_mcp = require_http_mcp
        self.mcp_http_timeout = mcp_http_timeout

        selected_agent = (
            coding_agent
            or self.manifest.driver.kind
            or (getattr(driver, "kind", "") if driver is not None else "")
        )
        if not selected_agent:
            raise ManifestError(
                "this workflow is coding-agent neutral; pass --coding-agent "
                "(claude, codex, pi, or pm-coder)"
            )
        self.manifest.driver.kind = selected_agent
        if model:
            self.manifest.driver.model = model
        if effort:
            self.manifest.driver.effort = effort
        if base_url is not None:
            self.manifest.driver.base_url = base_url
        if api_key_env is not None:
            self.manifest.driver.api_key_env = api_key_env
        add_dirs = self.manifest.driver.add_dirs or [str(self.base_dir)]

        # A caller may supply its own driver; otherwise it comes from the
        # manifest, with the CLI overrides applied above.
        try:
            self.driver = driver or build_driver(
                kind=self.manifest.driver.kind,
                model=self.manifest.driver.model,
                effort=self.manifest.driver.effort,
                add_dirs=add_dirs,
                base_url=self.manifest.driver.base_url,
                api_key_env=self.manifest.driver.api_key_env,
                max_agent_requests=max_agent_requests,
                timeout_seconds=self.manifest.driver.timeout_seconds,
            )
        except ValueError as exc:
            raise ManifestError(str(exc)) from exc

        journal_path = self.kernel_data / "journal.jsonl"
        if not resume and journal_path.is_file() and journal_path.stat().st_size:
            # Starting over means starting over. Leaving the old journal in
            # place would re-baseline the repository but still resume at the
            # phase the previous run stopped at, which is neither behaviour
            # anybody asked for. The old journal is kept beside the new one so
            # the previous attempt stays auditable.
            archive = self._next_archive_name()
            journal_path.rename(archive)
            print(f"FRESH START: previous journal archived as {archive.name}")
        self.journal = Journal(journal_path)

        self.checkpoint: GitCheckpoint | None = None
        if self.manifest.checkpoint_backend and self.manifest.checkpoint_backend.kind == "git":
            repo = self.manifest.checkpoint_backend.repo_path or str(self.workspace)
            self.checkpoint = GitCheckpoint(repo)
            self.checkpoint.init_if_needed()
            # `state.md` is controller-owned; `.codegraph/` is the code index
            # every role is told to consult first. Neither is part of the work
            # being judged, and both sit in the target repo — so unless git is
            # told to ignore them, the `git clean -fd` in every revert deletes
            # them. Losing the index is the worse one: it does not fail loudly,
            # it just leaves each later role without the tool it was told to use.
            self.checkpoint.exclude_locally("agents/tasks/*/state.md", ".codegraph/")

        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._write_task_index()
        if task_text:
            request = self.task_dir / "request.md"
            if not request.exists():
                request.write_text(task_text, encoding="utf-8")
        self.task_text = task_text or self._read_request()

        self.accepted_revision = self._establish_baseline(resume)
        self.pending_feedback: str | None = None
        self.pending_answer: str | None = None
        self.current_item: str | None = None
        self.exit_reason = ""
        self._render_state_md("init")

    # ------------------------------------------------------------------ setup

    def _next_archive_name(self) -> Path:
        index = 1
        while (self.kernel_data / f"journal.attempt{index}.jsonl").exists():
            index += 1
        return self.kernel_data / f"journal.attempt{index}.jsonl"

    def _write_task_index(self) -> None:
        """Index the task folder the controller just created.

        Every non-empty directory under `agents/` needs an `AGENTS.md`, and the
        controller creates this one — so it writes the index rather than
        failing a role for a directory the role did not make. Roles extend it.
        """
        index = self.task_dir / "AGENTS.md"
        if index.exists():
            return
        index.write_text(
            f"# Task {self.task_id}\n\n"
            "- `request.md` — the original request, as given.\n"
            "- `state.md` — written by the controller. Read only; never edit,\n"
            "  stage, or commit a change to it.\n"
            "- other files and folders here are role artifacts. Each role adds\n"
            "  its own and indexes any directory it creates.\n",
            encoding="utf-8",
        )

    def _read_request(self) -> str:
        request = self.task_dir / "request.md"
        return request.read_text(encoding="utf-8") if request.exists() else ""

    def _establish_baseline(self, resume: bool) -> str | None:
        """Decide which commit 'known good' means for this run."""
        if self.checkpoint is None:
            return None
        prior = self.journal.last_accepted_revision()
        head = self.checkpoint.current_rev()
        if resume and prior:
            print(f"RESUME: last accepted revision {prior[:8]}")
            if self.manifest.state_policy.on_resume == "retain":
                print("  retaining current workspace state by configured state_policy")
                return prior
            if head != prior:
                resume_phase = self._resume_point(announce=False)
                if self._candidate_is_pending_followup(
                    resume_phase, head, accepted_revision=prior
                ):
                    print(f"  HEAD is candidate {head[:8]} — preserving for follow-up")
                    return prior
                print(f"  HEAD is {head[:8]} — reverting to the accepted revision")
                self.checkpoint.restore(prior)
            elif self.checkpoint.is_dirty() and self._resume_discards_worktree():
                # Debris from a session killed by a crash, a reboot, or a kill
                # during a rate-limit wait. It never passed a check, so it is
                # rejected work — and leaving it would attribute it to the
                # session about to run.
                print("  discarding an interrupted session's uncommitted changes")
                self.checkpoint.restore(prior)
            return prior
        rev = self.checkpoint.snapshot(f"kernel baseline for {self.task_id}")
        self.journal.append(JournalEntry(
            run_id=self.run_id, phase="baseline", kind="checkpoint",
            ok=True, candidate_rev=rev, verdict="baseline",
        ))
        return rev

    # -------------------------------------------------------------------- run

    def run(self) -> dict[str, Any]:
        print(f"\n{'=' * 68}")
        print(f"KERNEL  task={self.task_id}")
        print(f"  workflow : {self.manifest.name}  ({self.manifest_path.name})")
        print(f"  target   : {self.workspace}")
        print(f"  base     : {self.base_dir}")
        print(f"  agent    : {self.manifest.driver.kind} "
              f"model={self.manifest.driver.model or '(default)'} "
              f"effort={self.manifest.driver.effort or '(default)'}")
        print(f"  journal  : {self.journal.path}")
        print(f"{'=' * 68}\n")

        phase = self._resume_point()
        while phase is not None:
            if self._over_phase_budget(phase):
                break
            try:
                result = self._execute_phase(phase)
            except TokenLimitError as limit:
                self.exit_reason = (
                    f"{phase.name}: {limit.agent_kind} usage limit; rerun this "
                    "task with another --coding-agent to continue"
                )
                print(f"\n!!! {self.exit_reason}")
                break
            # A revision is accepted only by a phase that declares it, and the
            # phase that declares it should be the last check validating the
            # work — never the role that produced it. Accepting a role's output
            # before its checks run would make the revert target the very
            # attempt the checks are about to reject.
            if result["valid"] and phase.checkpoint_after:
                self._accept(phase.name, self.journal.attempts_for_phase(phase.name))
            self._render_state_md(phase.name)
            target = self._route(phase, result)
            self.journal.append(JournalEntry(
                run_id=self.run_id, phase=phase.name, kind="route",
                ok=True, verdict=target or ROUTE_STOP, item=self.current_item,
            ))
            phase = self._resolve_target(phase, target)
            # Routing is where reverts, item completions and give-ups happen.
            # Re-render so the role dispatched next reads a state file that
            # already shows them — otherwise the session retried after a revert
            # sees a history in which its discarded attempt still succeeded.
            self._render_state_md(phase.name if phase else "finished")

        self._render_state_md("finished")
        summary = {
            "task_id": self.task_id,
            "workflow": self.manifest.name,
            "target": str(self.workspace),
            "accepted_revision": self.accepted_revision,
            "journal": str(self.journal.path),
            "journal_entries": len(self.journal.read_all()),
            "exit_reason": self.exit_reason or "completed",
            "ok": self.exit_reason == "",
        }
        terminal = self._terminal_execution()
        summary["terminal_phase"] = terminal.get("phase") if terminal else None
        summary["terminal_status"] = terminal.get("status") if terminal else None
        summary["terminal_result"] = terminal.get("result") if terminal else None
        print(f"\n{'=' * 68}")
        print(f"KERNEL finished: {summary['exit_reason']} "
              f"({summary['journal_entries']} journal entries)")
        print(f"{'=' * 68}\n")
        return summary

    def _terminal_execution(self) -> dict[str, Any] | None:
        for entry in reversed(self.journal.read_all()):
            if entry.get("kind") in {"role", "workflow"}:
                return entry
        return None

    def _resume_discards_worktree(self) -> bool:
        """Is an interrupted session's uncommitted work debris, or a candidate?

        It depends entirely on what runs next. Resuming at a check means that
        check exists to judge exactly what is in the tree — the role already
        reported and only its validation was lost, so throwing the tree away
        would have the check pass vacuously against nothing. Resuming at a role
        or a loop means fresh work is about to be produced, and leftover files
        would be credited to a session that did not write them.
        """
        phase = self._resume_point(announce=False)
        return phase is None or phase.kind not in {"gate", "script"}

    def _over_phase_budget(self, phase: PhaseConfig) -> bool:
        """Backstop against a routing cycle nobody declared a cap for.

        Retry caps only bound failures. Two phases can also ping-pong on
        outcomes that are each perfectly valid — a reviewer reporting findings,
        a fix planner deciding there is nothing to fix, back to the reviewer —
        and that costs a session every lap.

        Inside a loop the count has to be per item, not lifetime: a workflow
        with fifty work items legitimately runs its implement phase fifty times.
        There the bound is the loop's own `max_iterations`.
        """
        if phase.kind == "loop":
            # A loop bounds its own re-entry in _execute_loop.
            return False

        loop = self.manifest.loop_containing(phase.name)
        if loop is not None:
            limit = loop.max_iterations
            executed = sum(
                1 for entry in self.journal.entries_for_phase(phase.name)
                if entry.get("item") == self.current_item
            )
            scope = f"for {Path(self.current_item).name}" if self.current_item else ""
            setting = f"loop '{loop.name}' max_iterations"
        else:
            limit = self.manifest.failure_policy.max_attempts_per_phase
            executed = self.journal.attempts_for_phase(phase.name)
            scope = ""
            setting = "max_attempts_per_phase"

        if executed < limit:
            return False

        self.exit_reason = (
            f"{phase.name}: executed {executed} times {scope}".rstrip()
            + f", at the declared {setting} of {limit} — the workflow is cycling"
        )
        print(f"\n!!! {self.exit_reason}")
        self.journal.append(JournalEntry(
            run_id=self.run_id, phase=phase.name, kind="escalate", ok=False,
            verdict="phase_budget_exhausted", item=self.current_item,
            reason_codes=[f"executed={executed}", f"limit={limit}"],
        ))
        return True

    def _resume_point(self, announce: bool = True) -> PhaseConfig | None:
        """Continue where the previous run left off, or start at the top.

        A run that stopped resumes at the phase that stopped it, not at the
        beginning: the earlier phases were accepted, and re-running them would
        throw away work that already passed its checks. This is what makes
        "fix the cause and run the same command again" the whole recovery
        story — including when the cause was the role's instructions rather
        than the repository.
        """
        route = self.journal.last_route()
        if route is None:
            return self.manifest.phases[0]

        target = route.get("verdict")
        resume_at = target
        if target in {ROUTE_STOP, ROUTE_NEXT_ITEM, ROUTE_EXIT_LOOP}:
            # Not phase names. Go back to the phase that produced the outcome.
            resume_at = route.get("phase")

        # Never resume straight into a loop body: the selected work item lives
        # in memory, so a new process would run the phase with no item at all.
        # Re-entering the loop reselects it deterministically from the journal
        # and the pending files.
        loop = self.manifest.loop_containing(resume_at or "")
        if loop is not None:
            resume_at = loop.name

        phase = self.manifest.phase_by_name(resume_at or "")
        if phase is None:
            return self.manifest.phases[0]
        if announce and phase is not self.manifest.phases[0]:
            print(f"RESUME: continuing at phase '{phase.name}'")
        return phase

    # ------------------------------------------------------------- execution

    def _execute_phase(self, phase: PhaseConfig) -> dict[str, Any]:
        if phase.kind == "role":
            return self._execute_role(phase)
        if phase.kind == "gate":
            return self._execute_check(phase, phase.predicate or "", "gate")
        if phase.kind == "script":
            return self._execute_check(phase, phase.script or "", "script")
        if phase.kind == "human":
            return self._execute_human(phase)
        if phase.kind == "loop":
            return self._execute_loop(phase)
        if phase.kind == "workflow":
            return self._execute_workflow(phase)
        raise ManifestError(f"unsupported phase kind '{phase.kind}' in '{phase.name}'")

    def _execute_role(self, phase: PhaseConfig) -> dict[str, Any]:
        role = self.manifest.roles[phase.role or ""]
        attempt = self.journal.attempts_for_phase(phase.name) + 1
        if self.allowed_mcp is not None:
            denied = sorted(set(role.mcp) - self.allowed_mcp)
            if denied:
                errors = [
                    f"role '{role.name}' requests MCP capability not granted by its "
                    f"parent workflow: {', '.join(denied)}"
                ]
                self.journal.append(JournalEntry(
                    run_id=self.run_id, phase=phase.name, kind="role", role=role.name,
                    attempt=attempt, ok=False, verdict="capability_denied", errors=errors,
                ))
                return {"valid": False, "status": None, "errors": errors, "data": {}}
        skill_path = self._resolve_path(role.skill)
        if not skill_path.is_file():
            raise ManifestError(
                f"role '{role.name}' points at a missing skill: {skill_path}"
            )

        feedback = self.pending_feedback or self._feedback(phase.name)
        self.pending_feedback = None
        answer = self.pending_answer
        self.pending_answer = None

        self._settle_worktree(phase)

        print(f"\n>>> {phase.name}  role={role.name}  attempt {attempt}")
        if self.current_item:
            print(f"    work item: {self.current_item}")
        if feedback:
            print(f"    feedback: {feedback.splitlines()[0][:140]}")

        prompt = self._build_prompt(role, phase, attempt, feedback, answer, skill_path)
        base_rev = self.checkpoint.current_rev() if self.checkpoint else None
        driver_kind = getattr(self.driver, "kind", "agent")
        trace_file = (
            self.kernel_data / "traces"
            / f"{phase.name}_attempt{attempt}_{driver_kind}.jsonl"
        )
        result_file = (
            self.kernel_data / "results"
            / f"{phase.name}_attempt{attempt}_{driver_kind}.json"
        )
        session_options: dict[str, Any] = {}
        if self.allowed_mcp is not None:
            if not getattr(self.driver, "supports_explicit_mcp_config", False):
                errors = [
                    f"driver '{driver_kind}' cannot enforce a child MCP capability boundary"
                ]
                self.journal.append(JournalEntry(
                    run_id=self.run_id, phase=phase.name, kind="role", role=role.name,
                    attempt=attempt, ok=False, verdict="capability_unenforceable",
                    errors=errors,
                ))
                return {"valid": False, "status": None, "errors": errors, "data": {}}
            session_options["mcp_config"] = self._filtered_mcp_config(
                phase, role, attempt
            )

        while True:
            try:
                agent = self.driver.run_session(
                    run_id=f"{self.run_id}_{phase.name}_attempt{attempt}",
                    attempt=attempt,
                    skill=str(skill_path),
                    prompt=prompt,
                    work_dir=self.workspace,
                    tools=role.tools,
                    result_file=result_file,
                    trace_file=trace_file,
                    **session_options,
                )
                break
            except TokenLimitError as limit:
                self.journal.append(JournalEntry(
                    run_id=self.run_id, phase=phase.name, kind="rate_limit",
                    role=role.name, attempt=attempt, ok=False,
                    verdict="usage_limit", errors=[str(limit)[-500:]],
                ))
                raise

        status, errors = self._read_contract(role, agent)
        candidate_rev = self.checkpoint.current_rev() if self.checkpoint else None
        valid = status is not None
        archived_artifacts = self._write_attempt_receipt(
            phase, attempt, status, errors, agent.result_json, agent.trace_path,
            base_rev, candidate_rev,
        )

        self.journal.append(JournalEntry(
            run_id=self.run_id, phase=phase.name, kind="role", role=role.name,
            attempt=attempt, ok=valid, base_rev=base_rev, candidate_rev=candidate_rev,
            verdict=status or "contract_violation", status=status,
            item=self.current_item, errors=errors,
            result=agent.result_json, trace_path=agent.trace_path,
            artifacts=archived_artifacts,
        ))

        print(f"    -> status={status or 'CONTRACT VIOLATION'}")
        for line in errors[:4]:
            print(f"       {line[:160]}")

        return {"valid": valid, "status": status, "errors": errors,
                "data": agent.result_json or {}}

    def _filtered_mcp_config(
        self, phase: PhaseConfig, role: RoleConfig, attempt: int
    ) -> Path:
        source = self.workspace / ".mcp.json"
        servers: dict[str, Any] = {}
        if source.is_file():
            payload = json.loads(source.read_text(encoding="utf-8"))
            raw_servers = payload.get("mcpServers")
            if not isinstance(raw_servers, dict):
                raise ManifestError(f"MCP config must contain mcpServers: {source}")
            missing = sorted(set(role.mcp) - set(raw_servers))
            if missing:
                raise ManifestError(
                    f"role '{role.name}' requires MCP server(s) absent from {source}: "
                    + ", ".join(missing)
                )
            servers = {name: raw_servers[name] for name in role.mcp}
        elif role.mcp:
            raise ManifestError(
                f"role '{role.name}' requires MCP server(s) but {source} does not exist"
            )
        if self.require_http_mcp:
            for name, server in servers.items():
                if not isinstance(server, dict) or not isinstance(server.get("url"), str):
                    raise ManifestError(
                        f"MCP server '{name}' must declare an HTTP url for this child"
                    )
                self._require_http_reachable(name, server["url"])
        output = (
            self.kernel_data / "mcp" / f"{phase.name}-attempt-{attempt:04d}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"mcpServers": servers}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output

    def _require_http_reachable(self, name: str, url: str) -> None:
        if not url.lower().startswith(("http://", "https://")):
            raise ManifestError(f"MCP server '{name}' is not HTTP: {url}")
        request = urllib.request.Request(url, method="OPTIONS")
        try:
            with urllib.request.urlopen(request, timeout=self.mcp_http_timeout):
                return
        except urllib.error.HTTPError:
            # Any HTTP response proves that the configured process is reachable.
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ManifestError(
                f"MCP server '{name}' is not reachable at {url}: {exc}"
            ) from exc

    def _write_attempt_receipt(
        self,
        phase: PhaseConfig,
        attempt: int,
        status: str | None,
        errors: list[str],
        result: dict[str, Any] | None,
        trace_path: str | None,
        base_rev: str | None,
        candidate_rev: str | None,
    ) -> list[str]:
        """Archive opt-in external attempt evidence outside the workspace."""
        if self.manifest.state_policy.attempt_receipts != "files":
            return []
        root = self.kernel_data / "attempts" / phase.name / f"attempt-{attempt:04d}"
        artifact_root = root / "artifacts"
        copied: list[str] = []
        candidates: list[str] = []
        if isinstance(result, dict):
            for key, value in result.items():
                if key == "artifact" and isinstance(value, str):
                    candidates.append(value)
                elif key == "artifacts" and isinstance(value, list):
                    candidates.extend(str(item) for item in value)
                elif (key.endswith("_ref") or key.endswith("_refs")):
                    if isinstance(value, str):
                        candidates.append(value)
                    elif isinstance(value, list):
                        candidates.extend(str(item) for item in value)
        seen: set[str] = set()
        for raw in candidates:
            if raw in seen or "#" in raw:
                continue
            seen.add(raw)
            source = Path(raw)
            if not source.is_absolute():
                source = self.workspace / source
            source = source.resolve()
            if not source.is_file() or self.workspace not in source.parents:
                continue
            relative = source.relative_to(self.workspace)
            destination = artifact_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(relative.as_posix())
        root.mkdir(parents=True, exist_ok=True)
        receipt = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "phase": phase.name,
            "attempt": attempt,
            "status": status,
            "ok": status is not None,
            "errors": errors,
            "result": result,
            "trace_path": trace_path,
            "base_revision": base_rev,
            "candidate_revision": candidate_rev,
            "retained_workspace": self.manifest.state_policy.on_retry == "retain",
            "archived_artifacts": copied,
        }
        (root / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return copied

    def _read_contract(
        self, role: RoleConfig, agent: AgentResult
    ) -> tuple[str | None, list[str]]:
        """Validate the agent's result against the role's declared contract."""
        errors: list[str] = []
        if agent.error:
            errors.append(agent.error[:1000])
        payload = agent.result_json
        if not isinstance(payload, dict):
            errors.append(
                "No JSON result object in the final message. The final message "
                "must be the JSON result object and nothing else."
            )
            if agent.stdout:
                errors.append(f"final message tail: {agent.stdout[-400:]}")
            return None, errors

        contract = role.result_contract
        raw_status = payload.get(contract.status_field)
        declared = contract.status_values
        if raw_status is None:
            errors.append(f"result object is missing the '{contract.status_field}' field")
            return None, errors
        status = str(raw_status)
        if declared and status not in declared:
            errors.append(
                f"'{status}' is not one of the allowed {contract.status_field} "
                f"values: {', '.join(declared)}"
            )
            return None, errors

        missing = [f for f in contract.required_fields() if f not in payload]
        if missing:
            errors.append(f"result object is missing required field(s): {', '.join(missing)}")
            return None, errors

        return status, errors

    def _execute_check(self, phase: PhaseConfig, script: str, kind: str) -> dict[str, Any]:
        path = self._resolve_path(script)
        print(f"\n>>> {phase.name}  {kind}={Path(script).name}")
        result = run_gate(path, self.workspace, self._check_env(), phase.args)
        self.journal.append(JournalEntry(
            run_id=self.run_id, phase=phase.name, kind=kind, ok=result.ok,
            verdict="pass" if result.ok else "fail", item=self.current_item,
            errors=result.errors,
        ))
        print(f"    -> {'PASS' if result.ok else 'FAIL'}")
        for line in result.errors[:6]:
            print(f"       {line[:160]}")
        return {"valid": result.ok, "status": None, "errors": result.errors, "data": {}}

    def _check_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update({
            "WORKSPACE": str(self.workspace),
            "TASK_ID": self.task_id,
            "TASK_DIR": str(self.task_dir),
            "BASE_DIR": str(self.base_dir),
            "KERNEL_DATA": str(self.kernel_data),
            "BASE_REV": self.accepted_revision or "",
            "CANDIDATE_REV": (self.checkpoint.current_rev() if self.checkpoint else ""),
            "CURRENT_ITEM": self.current_item or "",
        })
        last = self.journal.last_role()
        if last:
            role = self.manifest.roles.get(last)
            env["ROLE"] = last
            if role:
                env["ROLE_WRITABLE_PATHS"] = os.pathsep.join(role.writable_paths)
        return env

    def _execute_human(self, phase: PhaseConfig) -> dict[str, Any]:
        question = phase.question
        if phase.question_from_result:
            question = self._last_result_field(phase.question_from_result) or question
        print(f"\n{'-' * 68}")
        print(f"USER INPUT REQUIRED  ({phase.name})")
        print(f"{'-' * 68}")
        print(question.strip())
        print("\nType your answer. Finish with a single '.' on its own line.")
        answer = self._read_multiline()
        self.pending_answer = answer
        self.journal.append(JournalEntry(
            run_id=self.run_id, phase=phase.name, kind="human", ok=True,
            verdict="answered", answer=answer, item=self.current_item,
        ))
        return {"valid": True, "status": None, "errors": [], "data": {"answer": answer}}

    @staticmethod
    def _read_multiline() -> str:
        lines: list[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == ".":
                break
            lines.append(line)
        return "\n".join(lines).strip()

    def _last_result_field(self, field: str) -> str:
        for entry in reversed(self.journal.read_all()):
            if entry.get("kind") == "role" and isinstance(entry.get("result"), dict):
                value = entry["result"].get(field)
                if isinstance(value, str) and value.strip():
                    return value
        return ""

    def _execute_loop(self, phase: PhaseConfig) -> dict[str, Any]:
        entries = self.journal.entries_for_phase(phase.name)
        if len(entries) >= phase.max_iterations:
            self.journal.append(JournalEntry(
                run_id=self.run_id, phase=phase.name, kind="escalate", ok=False,
                verdict="max_iterations", reason_codes=[str(phase.max_iterations)],
            ))
            return {"valid": False, "status": None,
                    "errors": [f"loop '{phase.name}' hit max_iterations"], "data": {}}

        pending = self._pending_work_items()
        print(f"\n>>> {phase.name}  loop  pending={len(pending)}")
        if not pending:
            self.current_item = None
            self.journal.append(JournalEntry(
                run_id=self.run_id, phase=phase.name, kind="loop", ok=True,
                verdict="exhausted",
            ))
            return {"valid": True, "status": None, "errors": [], "data": {"exhausted": True}}

        self.current_item = pending[0]
        print(f"    selected: {self.current_item}  ({len(pending)} pending)")
        self.journal.append(JournalEntry(
            run_id=self.run_id, phase=phase.name, kind="loop", ok=True,
            verdict="selected", item=self.current_item,
            reason_codes=[f"pending={len(pending)}"],
        ))
        return {"valid": True, "status": None, "errors": [], "data": {"item": self.current_item}}

    def _pending_work_items(self) -> list[str]:
        """Work items on disk, in filename order, minus the ones journalled done."""
        directory = self.task_dir / "workitems"
        if not directory.is_dir():
            return []
        done = self.journal.completed_items()
        items = []
        for path in sorted(directory.glob(WORK_ITEM_GLOB)):
            relative = path.relative_to(self.workspace).as_posix()
            if relative not in done:
                items.append(relative)
        return items

    def _execute_workflow(self, phase: PhaseConfig) -> dict[str, Any]:
        """Invoke one or more statically named children in fresh kernel runs."""
        attempt = self.journal.attempts_for_phase(phase.name) + 1
        limits = phase.limits
        max_attempts = limits.max_attempts if limits else 1
        if attempt > max_attempts:
            errors = [
                f"child phase '{phase.name}' reached its invocation limit of {max_attempts}"
            ]
            self.journal.append(JournalEntry(
                run_id=self.run_id, phase=phase.name, kind="workflow",
                attempt=attempt, ok=False, verdict="max_attempts", errors=errors,
            ))
            return {"valid": False, "status": None, "errors": errors, "data": {}}

        decrement = limits.decrement_depth if limits else 1
        remaining = self.depth_remaining - decrement
        if limits and limits.max_depth is not None:
            remaining = min(remaining, limits.max_depth)
        declared = phase.child_result.statuses if phase.child_result else []
        if remaining < 0:
            status = "decomposition_limit" if "decomposition_limit" in declared else None
            errors = [] if status else [
                f"child phase '{phase.name}' exhausted its depth budget"
            ]
            self.journal.append(JournalEntry(
                run_id=self.run_id, phase=phase.name, kind="workflow",
                attempt=attempt, ok=status is not None,
                verdict=status or "depth_exhausted", status=status, errors=errors,
                result={"depth_remaining": self.depth_remaining, "decrement": decrement},
            ))
            return {
                "valid": status is not None, "status": status, "errors": errors,
                "data": {"depth_remaining": self.depth_remaining},
            }

        try:
            items = self._child_items(phase)
            receipts = [
                self._run_child(phase, attempt, index, item, remaining)
                for index, item in enumerate(items, start=1)
            ]
            errors = [
                error for receipt in receipts for error in receipt.get("errors", [])
            ]
            statuses = [
                str(receipt["status"]) for receipt in receipts if receipt.get("status")
            ]
            status = self._aggregate_child_status(phase, statuses)
            valid = len(statuses) == len(receipts) and status in declared
            if not valid and not errors:
                errors = [
                    f"child phase '{phase.name}' did not produce one declared status per child"
                ]
        except (ManifestError, OSError, ValueError, json.JSONDecodeError) as exc:
            receipts = []
            status = None
            valid = False
            errors = [f"{type(exc).__name__}: {exc}"]

        result = {"children": receipts, "status": status}
        artifacts = [
            artifact for receipt in receipts for artifact in receipt.get("artifacts", [])
        ]
        self.journal.append(JournalEntry(
            run_id=self.run_id, phase=phase.name, kind="workflow", attempt=attempt,
            ok=valid, verdict=status or "invalid_child_result", status=status,
            errors=errors, result=result, artifacts=artifacts,
        ))
        receipt_dir = self.kernel_data / "child-receipts" / phase.name
        receipt_dir.mkdir(parents=True, exist_ok=True)
        (receipt_dir / f"attempt-{attempt:04d}.json").write_text(
            json.dumps(result | {"errors": errors, "valid": valid}, indent=2, default=str),
            encoding="utf-8",
        )
        print(
            f"\n>>> {phase.name}  workflow={phase.workflow}  attempt {attempt}"
            f"\n    -> status={status or 'INVALID CHILD RESULT'} children={len(receipts)}"
        )
        return {"valid": valid, "status": status, "errors": errors, "data": result}

    def _child_items(self, phase: PhaseConfig) -> list[Any | None]:
        if phase.foreach is None:
            return [None]
        raw = load_reference(phase.foreach.source, self.workspace)
        if not isinstance(raw, list):
            raise ManifestError(
                f"foreach source for '{phase.name}' must resolve to a list"
            )
        if len(raw) > phase.foreach.max_items:
            raise ManifestError(
                f"foreach source for '{phase.name}' has {len(raw)} items, above "
                f"max_items={phase.foreach.max_items}"
            )
        stable_field = phase.foreach.stable_id
        prefix = f"{phase.foreach.item}."
        if stable_field.startswith(prefix):
            stable_field = stable_field[len(prefix):]
        return order_items(raw, stable_field, phase.foreach.order)

    def _run_child(
        self,
        phase: PhaseConfig,
        attempt: int,
        index: int,
        item: Any | None,
        depth_remaining: int,
    ) -> dict[str, Any]:
        task = phase.task
        result_contract = phase.child_result
        if task is None or result_contract is None:
            raise ManifestError(f"workflow phase '{phase.name}' has no child contract")
        variables: dict[str, Any] = {}
        stable_id = str(index)
        if phase.foreach is not None:
            variables[phase.foreach.item] = item
            stable_field = phase.foreach.stable_id
            prefix = f"{phase.foreach.item}."
            if stable_field.startswith(prefix):
                stable_field = stable_field[len(prefix):]
            stable_id = str(dotted(item, stable_field))
        configured_task_id = str(expand_runtime(task.id, variables))
        child_task_id = self._child_task_id(
            self.workspace / "agents" / "tasks",
            configured_task_id,
            attempt,
            stable_id if phase.foreach is not None else None,
        )
        invocation = f"{self._safe_name(phase.name)}-{attempt:04d}-{index:04d}"
        child_root = self.kernel_data / "children" / invocation
        durable_receipt = child_root / "receipt.json"
        if durable_receipt.is_file():
            try:
                cached = json.loads(durable_receipt.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cached = None
            if isinstance(cached, dict):
                cached["reused"] = True
                return cached
        inputs = self._resolve_child_value(task.input, variables)
        task_text = json.dumps(
            {
                "parent_task_id": self.task_id,
                "parent_phase": phase.name,
                "configured_task_id": configured_task_id,
                "input": inputs,
                "context": phase.context.__dict__ if phase.context else {},
                "capabilities": phase.capabilities.__dict__ if phase.capabilities else {},
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        manifest = self._resolve_child_manifest(phase.workflow or "")
        allowed_mcp, allowed_effects = self._child_capabilities(phase)
        child_driver = self.driver
        if (
            phase.limits
            and phase.limits.max_agent_requests is not None
            and hasattr(self.driver, "max_turns")
        ):
            child_driver = copy(self.driver)
            child_driver.max_turns = phase.limits.max_agent_requests
        child = Kernel(
            manifest_path=manifest,
            workspace=self.workspace,
            task_id=child_task_id,
            task_text=task_text,
            base_dir=self.base_dir,
            coding_agent=getattr(child_driver, "kind", None),
            kernel_data_root=child_root,
            run_id="run",
            resume=True,
            driver=child_driver,
            depth_remaining=depth_remaining,
            allowed_mcp=allowed_mcp,
            allowed_effects=allowed_effects,
            require_http_mcp=(
                phase.capabilities.require_http_reachable
                if phase.capabilities else False
            ),
            mcp_http_timeout=(
                phase.capabilities.http_timeout_seconds
                if phase.capabilities else 5.0
            ),
        )
        summary = child.run()
        raw_status: Any = summary.get("terminal_status")
        if result_contract.status_from:
            raw_status = load_reference(result_contract.status_from, child.task_dir)
        if raw_status is None:
            raw_status = result_contract.default_status or None
        status = result_contract.status_map.get(str(raw_status), str(raw_status)) \
            if raw_status is not None else None
        errors: list[str] = []
        if status not in result_contract.statuses:
            errors.append(
                f"child '{phase.workflow}' returned '{raw_status}', which maps to "
                f"'{status}' but declared statuses are {result_contract.statuses}"
            )
            status = None
        artifacts: list[str] = []
        workspace_cfg = phase.workspace
        if workspace_cfg and workspace_cfg.merge == "artifacts_only":
            artifacts = copy_declared_artifacts(
                child.task_dir,
                self.task_dir,
                str(expand_runtime(workspace_cfg.artifact_prefix, variables)),
                result_contract.artifacts,
                attempt,
            )
        receipt = {
            "invocation": invocation,
            "workflow": phase.workflow,
            "configured_task_id": configured_task_id,
            "task_id": child_task_id,
            "item_id": stable_id if phase.foreach else None,
            "status": status,
            "raw_status": raw_status,
            "ok": status is not None,
            "errors": errors,
            "artifacts": artifacts,
            "journal": summary.get("journal"),
            "exit_reason": summary.get("exit_reason"),
            "depth_remaining": depth_remaining,
            "capabilities": {
                "mcp": sorted(allowed_mcp) if allowed_mcp is not None else None,
                "effects": sorted(allowed_effects) if allowed_effects is not None else None,
            },
        }
        child_root.mkdir(parents=True, exist_ok=True)
        temporary = child_root / "receipt.json.tmp"
        temporary.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(durable_receipt)
        return receipt

    def _resolve_child_value(self, value: Any, variables: dict[str, Any]) -> Any:
        expanded = expand_runtime(value, variables)
        if isinstance(expanded, list):
            return [self._resolve_child_value(item, variables) for item in expanded]
        if not isinstance(expanded, dict):
            return expanded
        if "from" in expanded:
            reference = str(expanded["from"])
            try:
                return load_reference(reference, self.workspace)
            except ManifestError:
                if expanded.get("required", True):
                    raise
                return None
        if "from_child" in expanded:
            source = str(expanded["from_child"])
            for entry in reversed(self.journal.read_all()):
                if entry.get("kind") == "workflow" and entry.get("phase") == source:
                    return entry.get("result")
            raise ManifestError(f"no completed child result exists for phase '{source}'")
        return {
            key: self._resolve_child_value(item, variables)
            for key, item in expanded.items()
        }

    def _aggregate_child_status(self, phase: PhaseConfig, statuses: list[str]) -> str | None:
        if not statuses:
            return None
        if len(set(statuses)) == 1:
            return statuses[0]
        contract = phase.child_result
        if contract is None:
            return None
        if not contract.aggregate:
            return contract.default_status or None
        priority = contract.status_priority or [
            status for status in contract.statuses if status != "completed"
        ]
        for status in priority:
            if status in statuses:
                return status
        return "completed" if "completed" in contract.statuses else statuses[0]

    def _resolve_child_manifest(self, name: str) -> Path:
        candidate = Path(name)
        choices = [candidate] if candidate.is_absolute() else [
            self.base_dir / "workflows" / name / f"{name}.workflow.md",
            self.base_dir / "workflows" / f"{name}.workflow.md",
            self.base_dir / name / f"{name}.workflow.md",
            self.base_dir / f"{name}.workflow.md",
        ]
        for choice in choices:
            if choice.is_file():
                return choice.resolve()
        raise ManifestError(
            f"child workflow '{name}' was not found; checked: "
            + ", ".join(str(choice) for choice in choices)
        )

    def _child_capabilities(
        self, phase: PhaseConfig
    ) -> tuple[set[str] | None, set[str] | None]:
        config = phase.capabilities
        if config is None or config.inherit:
            return self.allowed_mcp, self.allowed_effects
        requested_mcp = set(config.allow_mcp)
        requested_effects = set(config.allow_effects)
        if self.allowed_mcp is not None and not requested_mcp <= self.allowed_mcp:
            raise ManifestError(
                f"child phase '{phase.name}' requests MCP outside its parent grant: "
                + ", ".join(sorted(requested_mcp - self.allowed_mcp))
            )
        if self.allowed_effects is not None and not requested_effects <= self.allowed_effects:
            raise ManifestError(
                f"child phase '{phase.name}' requests effects outside its parent grant: "
                + ", ".join(sorted(requested_effects - self.allowed_effects))
            )
        return requested_mcp, requested_effects

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._") or "child"

    @classmethod
    def _child_task_id(
        cls,
        tasks_root: Path,
        configured_task_id: str,
        attempt: int,
        stable_id: str | None,
    ) -> str:
        """Keep recursive child paths bounded without changing short task IDs."""
        candidate = f"{configured_task_id}.__attempt_{attempt:04d}"
        if stable_id is not None:
            candidate += f".__item_{cls._safe_name(stable_id)}"
        candidate_path = tasks_root / candidate
        if len(candidate) <= 120 and len(str(candidate_path)) <= 220:
            return candidate

        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
        suffix = f".__h_{digest}.__attempt_{attempt:04d}"
        component_limit = min(120, max(48, 220 - len(str(tasks_root)) - 1))
        prefix_length = max(1, component_limit - len(suffix))
        prefix = cls._safe_name(configured_task_id)[:prefix_length].rstrip("-._")
        return f"{prefix or 'child'}{suffix}"

    # --------------------------------------------------------------- routing

    def _route(self, phase: PhaseConfig, result: dict[str, Any]) -> str | None:
        if phase.kind == "loop":
            if result["data"].get("exhausted"):
                return phase.exit
            if not result["valid"]:
                return self._on_failure(phase, phase.on_failure, result)
            return phase.body[0].name

        if phase.kind == "human":
            return phase.next

        if phase.kind in {"role", "workflow"}:
            if result["valid"]:
                status = result["status"]
                if status is not None and status in phase.on_status:
                    return phase.on_status[status]
                if phase.next:
                    return phase.next
                return self._on_failure(phase, phase.on_invalid, result)
            return self._on_failure(phase, phase.on_invalid, result)

        # gate / script
        if result["valid"]:
            return phase.on_pass or phase.next
        return self._on_failure(phase, phase.on_fail or phase.on_failure, result)

    def _on_failure(
        self, phase: PhaseConfig, config: Any, result: dict[str, Any]
    ) -> str | None:
        if config is None:
            self.exit_reason = f"{phase.name}: no failure route declared"
            self.journal.append(JournalEntry(
                run_id=self.run_id, phase=phase.name, kind="escalate", ok=False,
                verdict="no_failure_route",
            ))
            return None
        if isinstance(config, str):
            return config
        if not isinstance(config, dict):
            self.exit_reason = f"{phase.name}: malformed failure route"
            return None

        action = config.get("action", "retry_with_feedback")
        target = config.get("target", phase.name)

        if action == "route_to":
            return target

        if action in {"stop", "stop_subtree", "stop_with_failure", "fail"}:
            self.exit_reason = f"{phase.name}: {action}"
            self.journal.append(JournalEntry(
                run_id=self.run_id, phase=phase.name, kind="escalate", ok=False,
                verdict=action, errors=result.get("errors", []),
            ))
            return None

        if action not in {"retry_with_feedback", "retry_child_clean"}:
            self.exit_reason = f"{phase.name}: unknown failure action '{action}'"
            return None

        max_attempts = config.get("max_attempts", 999)
        attempts = self.journal.attempts_for_phase(target)
        if attempts >= max_attempts:
            self.exit_reason = (
                f"{target}: {attempts} attempts reached the declared limit of {max_attempts}"
            )
            parked = ""
            if self.manifest.state_policy.on_exhaustion == "park_and_restore":
                parked = self._park(target, attempts)
            self.journal.append(JournalEntry(
                run_id=self.run_id, phase=target, kind="escalate", ok=False,
                verdict="max_attempts", candidate_rev=parked,
                reason_codes=[f"attempts={attempts}"] + ([f"parked={parked}"] if parked else []),
                errors=result.get("errors", []),
            ))
            return None

        self._remember_failures(target, phase.name, result)
        workspace_action = str(config.get("workspace_action", ""))
        if action == "retry_child_clean":
            workspace_action = "retain"
        if not workspace_action:
            workspace_action = self.manifest.state_policy.on_retry
        if bool(config.get("preserve_candidate", False)):
            workspace_action = "preserve_candidate"
        reset = False
        if workspace_action in {"restore", "preserve_candidate"}:
            reset = self._revert(
                phase.name,
                target,
                preserve_candidate=workspace_action == "preserve_candidate",
            )
        elif workspace_action != "retain":
            self.exit_reason = (
                f"{phase.name}: unknown failure workspace_action '{workspace_action}'"
            )
            return None
        self.pending_feedback = self._feedback(
            target,
            reset=reset,
            retained_workspace=workspace_action == "retain",
        )
        return target

    def _resolve_target(self, phase: PhaseConfig, target: str | None) -> PhaseConfig | None:
        if target is None or target == ROUTE_STOP:
            if target == ROUTE_STOP:
                self.exit_reason = self.exit_reason or f"{phase.name}: stop"
            return None

        if target in {ROUTE_NEXT_ITEM, ROUTE_EXIT_LOOP}:
            loop = self.manifest.loop_containing(phase.name)
            if loop is None:
                self.exit_reason = f"{phase.name}: '{target}' outside a loop"
                return None
            if target == ROUTE_NEXT_ITEM:
                if self.current_item:
                    self.journal.append(JournalEntry(
                        run_id=self.run_id, phase=loop.name, kind="item_complete",
                        ok=True, verdict="completed", item=self.current_item,
                        candidate_rev=(self.checkpoint.current_rev() if self.checkpoint else None),
                    ))
                    print(f"    item complete: {self.current_item}")
                self._clear_failure_memory(f"item_complete={self.current_item or '(none)'}")
                self.current_item = None
                return loop
            self.current_item = None
            return self.manifest.phase_by_name(loop.exit or "")

        resolved = self.manifest.phase_by_name(target)
        if resolved is None:
            # Unreachable for a validated manifest; treat as a hard stop.
            self.exit_reason = f"{phase.name}: routed to unknown phase '{target}'"
        return resolved

    def _remember_failures(
        self,
        target: str,
        source: str,
        result: dict[str, Any],
    ) -> None:
        existing = {
            " ".join(value.split()).casefold()
            for value in self.journal.active_failure_causes(target)
        }
        errors = result.get("errors") or ["The controller recorded no error detail."]
        additions: list[str] = []
        for value in errors:
            cause = " ".join(str(value).split())
            key = cause.casefold()
            if cause and key not in existing:
                existing.add(key)
                additions.append(cause)
        if not additions:
            return
        self.journal.append(JournalEntry(
            run_id=self.run_id,
            phase=target,
            kind="failure_memory",
            ok=False,
            verdict="recorded",
            reason_codes=[f"source={source}"],
            errors=additions,
            item=self.current_item,
        ))

    def _feedback(
        self,
        target: str,
        *,
        reset: bool | None = None,
        retained_workspace: bool | None = None,
    ) -> str | None:
        if self.manifest.state_policy.feedback == "none":
            return None
        errors = self.journal.active_failure_causes(target)
        if self.manifest.state_policy.feedback == "latest" and errors:
            errors = errors[-1:]
        if not errors:
            return None
        if retained_workspace is None:
            retained_workspace = self.manifest.state_policy.on_retry == "retain"
        if reset is None:
            reset = not retained_workspace
        lines = [
            f"Earlier attempts routed back to the '{target}' step.",
            "",
            "Fix every failure cause in this accumulated list:",
        ]
        for error in errors:
            lines.append(f"  - {error}")
        lines.append("")
        if reset:
            lines += [
                "The repository has been reset to the last accepted commit, so none of",
                "your previous changes are present. Start from the current state, fix",
                "the cause of these problems, and do not repeat the same approach.",
            ]
        elif retained_workspace:
            lines += [
                "Previous files and external-world effects were retained. Treat them as",
                "evidence, observe the current state again, and make a new plan from what",
                "is true now. Do not assume that a failed attempt made no progress.",
            ]
        else:
            lines += [
                "The clean candidate commit being validated has been preserved for this",
                "retry. Review the current state; do not recreate or discard that work.",
            ]
        return "\n".join(lines)

    def _clear_failure_memory(self, reason: str) -> None:
        for target in self.journal.phases_with_failure_memory():
            self.journal.append(JournalEntry(
                run_id=self.run_id,
                phase=target,
                kind="failure_memory",
                ok=True,
                verdict="cleared",
                reason_codes=[reason],
                item=self.current_item,
            ))

    # ------------------------------------------------------------ checkpoints

    def _accept(self, phase_name: str, attempt: int) -> None:
        if self.checkpoint is None:
            self._clear_failure_memory(f"accepted={phase_name}")
            return
        rev = self.checkpoint.snapshot(f"{self.task_id} {phase_name} attempt {attempt}")
        if rev and rev != self.accepted_revision:
            print(f"    accepted {rev[:8]}")
        self.accepted_revision = rev
        self.journal.append(JournalEntry(
            run_id=self.run_id, phase=phase_name, kind="checkpoint", ok=True,
            candidate_rev=rev, verdict="accepted", item=self.current_item,
        ))
        self._clear_failure_memory(f"accepted={phase_name}")

    def _settle_worktree(self, phase: PhaseConfig) -> None:
        """Never hand a role a repository that already has changes in it.

        In normal flow the tree is clean here: the check that accepts a step
        commits everything, and a failed step is reverted. A dirty tree at this
        point is therefore leftover from something that neither accepted nor
        reverted — a crash, or a phase that routed onward without either. Left
        alone it would be swept into this role's checkpoint and attributed to a
        session that did not write it, which quietly corrupts both the scope
        check and the evidence trail.

        The leftovers are parked on a branch first, so nothing is destroyed and
        the diff stays readable, and only then discarded from the tree.
        """
        if self.manifest.state_policy.before_role == "retain":
            return
        if self.checkpoint is None or not self.checkpoint.is_dirty():
            return
        head_before_settle = self.checkpoint.current_rev()
        preserve_candidate = self._candidate_is_pending_followup(
            phase, head_before_settle
        )
        ref = f"leftover/{self.task_id}/{phase.name}"
        parked = self.checkpoint.park(ref)
        print(f"    leftover changes parked on {ref} ({parked[:8]}) and cleared")
        if self.accepted_revision:
            self.checkpoint.restore(
                head_before_settle if preserve_candidate else self.accepted_revision
            )
        self.journal.append(JournalEntry(
            run_id=self.run_id, phase=phase.name, kind="leftover", ok=True,
            verdict="parked", candidate_rev=parked, item=self.current_item,
            reason_codes=[ref] + (["preserved_candidate"] if preserve_candidate else []),
        ))

    def _candidate_is_pending_followup(
        self,
        phase: PhaseConfig | None,
        head: str,
        accepted_revision: str | None = None,
    ) -> bool:
        """Keep a committed candidate visible to the role that validates it.

        A successful implementation can be followed by several gates before a
        reviewer checkpoint accepts it. If that implementation also left
        untracked inspection material behind, settle it without resetting the
        committed candidate that the reviewer must inspect.
        """
        accepted = accepted_revision or getattr(self, "accepted_revision", None)
        if phase is None or not accepted or head == accepted:
            return False
        route = self.journal.last_route()
        if not route or route.get("verdict") != phase.name:
            return False
        preceding = self.manifest.phase_by_name(str(route.get("phase") or ""))
        return preceding is not None and preceding.kind in {"gate", "script"}

    def _park(self, phase_name: str, attempt: int) -> str:
        """Give up on a phase without either keeping or losing the rejected work.

        The attempt is preserved on a `rejected/...` branch so its diff can be
        read afterwards, and the working tree is reset to the last accepted
        commit — so the repository is never left holding changes that failed
        their checks.
        """
        if self.checkpoint is None:
            return ""
        ref = f"rejected/{self.task_id}/{phase_name}-{attempt}"
        parked = ""
        if self.checkpoint.is_dirty() or (
            self.accepted_revision and self.checkpoint.current_rev() != self.accepted_revision
        ):
            parked = self.checkpoint.park(ref)
            print(f"    rejected work parked on branch {ref} ({parked[:8]})")
        if self.accepted_revision:
            self.checkpoint.restore(self.accepted_revision)
            print(f"    repository reset to {self.accepted_revision[:8]}")
        return parked

    def _revert(
        self, from_phase: str, target: str, *, preserve_candidate: bool = False
    ) -> bool:
        if self.checkpoint is None or not self.accepted_revision:
            return False
        head = self.checkpoint.current_rev()
        if (
            preserve_candidate
            and not self.checkpoint.is_dirty()
            and head != self.accepted_revision
        ):
            print(f"    preserving candidate {head[:8]} before retrying '{target}'")
            self.journal.append(JournalEntry(
                run_id=self.run_id, phase=from_phase, kind="revert", ok=True,
                base_rev=self.accepted_revision, candidate_rev=head,
                verdict=f"preserve_before:{target}", item=self.current_item,
            ))
            return False
        print(f"    revert to {self.accepted_revision[:8]} before retrying '{target}'")
        self.checkpoint.restore(self.accepted_revision)
        self.journal.append(JournalEntry(
            run_id=self.run_id, phase=from_phase, kind="revert", ok=True,
            base_rev=self.accepted_revision, verdict=f"revert_before:{target}",
            item=self.current_item,
        ))
        return True

    # ---------------------------------------------------------------- prompts

    def _build_prompt(
        self,
        role: RoleConfig,
        phase: PhaseConfig,
        attempt: int,
        feedback: str | None,
        answer: str | None,
        skill_path: Path,
    ) -> str:
        lines = [
            f"You are running one role in a controlled workflow: `{role.name}`.",
            "",
            f"Read `{skill_path}` in full before doing anything else, then follow it",
            "exactly. Perform this role only. Do not take on another role's work.",
            "",
            f"Repository: `{self.workspace}`",
            f"Task id: `{self.task_id}`",
            f"Task folder: `{self.task_dir}`",
            f"Original request: `{self.task_dir / 'request.md'}`",
            "",
            "`state.md` in the task folder is written by the controller. You may read",
            "it. Never edit, stage, or commit a change to it.",
            "",
        ]
        if role.instruction:
            lines += [role.instruction.strip(), ""]
        if self.current_item:
            item_path = self.workspace / self.current_item
            lines += [
                f"Selected work item: `{self.current_item}`",
                "The controller selected it deterministically from the pending work",
                "items in filename order. Implement only this one.",
                "",
            ]
            if item_path.is_file():
                lines += [
                    "--- BEGIN WORK ITEM ---",
                    item_path.read_text(encoding="utf-8"),
                    "--- END WORK ITEM ---",
                    "",
                ]
        if self.task_text:
            lines += [
                "--- BEGIN ORIGINAL REQUEST ---",
                self.task_text.strip(),
                "--- END ORIGINAL REQUEST ---",
                "",
            ]
        if answer:
            lines += [
                "--- BEGIN USER ANSWER ---",
                answer,
                "--- END USER ANSWER ---",
                "",
                "This is the user's own reply. Treat it as authoritative.",
                "",
            ]
        if feedback:
            lines += [
                f"--- BEGIN CONTROLLER FEEDBACK (attempt {attempt}) ---",
                feedback,
                "--- END CONTROLLER FEEDBACK ---",
                "",
            ]
        lines += [self._contract_text(role)]
        if role.deny_access:
            lines += [
                "",
                "## Context-isolation access denial",
                "",
                "To prevent context poisoning from previous agents, do not access,",
                "read, inspect, list, search, grep, glob, summarize, or recover content",
                "from any of these paths or path patterns:",
                "",
            ]
            lines += [f"- `{path}`" for path in role.deny_access]
            lines += [
                "",
                "Do not obtain their content indirectly through Git history, generated",
                "indexes, caches, links, another tool, or another process. If a denied",
                "path is needed to complete this role, report the resulting limitation",
                "instead of reading it. This denial overrides any conflicting direction",
                "in the skill or elsewhere in this prompt.",
            ]
        return "\n".join(lines)

    def _contract_text(self, role: RoleConfig) -> str:
        contract = role.result_contract
        lines = [
            "## Required final message",
            "",
            "Your final message must be exactly one JSON object and nothing else —",
            "no prose before or after, no code fence. Anything else is a contract",
            "violation and the attempt is discarded.",
            "",
            "Fields:",
        ]
        for field, spec in contract.schema.items():
            lines.append(f"  {field}: {self._describe(spec)}")
        values = contract.status_values
        if values:
            lines += [
                "",
                f"`{contract.status_field}` must be one of: {', '.join(values)}.",
                "Report the outcome that actually happened. Reporting success for",
                "work you did not finish is the one unrecoverable failure here: the",
                "controller verifies every claim against the repository.",
            ]
        return "\n".join(lines)

    @staticmethod
    def _describe(spec: Any) -> str:
        if isinstance(spec, dict):
            if "enum" in spec:
                return "one of " + ", ".join(str(v) for v in spec["enum"])
            if spec.get("type") == "array":
                item = spec.get("items", "string")
                item_name = item if isinstance(item, str) else "object"
                return f"array of {item_name}"
            return str(spec.get("type", "string"))
        return str(spec)

    # ------------------------------------------------------------------ state

    def _render_state_md(self, current_phase: str) -> None:
        """Project the journal into a read-only state file for the agents.

        The journal is the source of truth; this is a view of it. Agents get the
        history of what happened, never the machinery that judges them.
        """
        entries = self.journal.read_all()
        lines = [
            f"# Workflow state — {self.task_id}",
            "",
            "Written by the controller. Read-only for every role: never edit,",
            "stage, or commit a change to this file.",
            "",
            f"- workflow: {self.manifest.name}",
            f"- current phase: {current_phase}",
            f"- accepted revision: {(self.accepted_revision or '(none)')[:12]}",
            f"- current work item: {self.current_item or '(none)'}",
            "",
            "## History",
            "",
        ]
        if not entries:
            lines.append("(nothing yet)")
        for index, entry in enumerate(entries, 1):
            kind = entry.get("kind", "?")
            if kind in {"route", "checkpoint", "failure_memory"}:
                continue
            verdict = entry.get("verdict", "")
            attempt = entry.get("attempt")
            suffix = f" attempt={attempt}" if attempt else ""
            item = entry.get("item")
            item_text = f" item={Path(item).name}" if item else ""
            errors = entry.get("errors") or []
            error_text = f" — {errors[0][:120]}" if errors else ""
            lines.append(
                f"{index}. [{kind}] {entry.get('phase', '?')}{suffix}{item_text}: "
                f"{verdict}{error_text}"
            )
        text = "\n".join(lines) + "\n"
        self.task_dir.mkdir(parents=True, exist_ok=True)
        (self.task_dir / "state.md").write_text(text, encoding="utf-8")
        # Reference copy outside the repo, so a gate can prove the file was not
        # edited by an agent.
        (self.kernel_data / "state.md").write_text(text, encoding="utf-8")

    def _resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()
