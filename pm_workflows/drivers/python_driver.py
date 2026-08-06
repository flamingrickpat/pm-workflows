"""Executes a role by loading a Python file and calling it directly.

No subprocess, no CLI, no sandbox: the skill file is imported into the
kernel's own process and its `run(context)` entry point is called in-line.
See :mod:`pm_workflows.python_role` for the interface a script implements.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

from ..protocol import AgentResult
from ..python_role import RoleContext
from .common import trace_write, write_result_artifact

ENTRY_POINT = "run"


class PythonDriver:
    """One fresh module load and one call to `run(context)` per attempt."""

    kind = "python"
    # Nothing here can enforce an MCP allowlist boundary on arbitrary code.
    supports_explicit_mcp_config = False

    def run_session(
        self,
        run_id: str,
        attempt: int,
        skill: str,
        prompt: str,
        work_dir: Path,
        tools: list[str] | None = None,
        result_file: Path | None = None,
        trace_file: Path | None = None,
        mcp_config: Path | None = None,
        context: RoleContext | None = None,
    ) -> AgentResult:
        skill_path = Path(skill)
        if context is None:
            # Direct use outside the kernel (tests, scripts): a minimal
            # context built from what this call was actually given.
            context = RoleContext(
                run_id=run_id, task_id="", attempt=attempt,
                workspace=Path(work_dir), task_dir=Path(work_dir),
                base_dir=Path(work_dir), kernel_data=Path(work_dir),
                role="", phase="", prompt=prompt, task_text="",
                current_item=None, feedback=None, answer=None,
                tools=list(tools or []), result_file=result_file,
                trace_file=trace_file,
            )

        trace_write(trace_file, {
            "event": "start", "agent": self.kind, "skill": str(skill_path),
            "work_dir": str(work_dir), "run_id": run_id, "attempt": attempt,
        })

        result, error = self._call_entry_point(skill_path, context)

        if error is not None:
            trace_write(trace_file, {"event": "end", "error": error[-4000:]})
            return AgentResult(
                exit_code=1, stdout="", result_json=None, error=error[:2000],
                trace_path=str(trace_file) if trace_file else None,
            )

        trace_write(trace_file, {"event": "end", "result": result})
        write_result_artifact(result_file, result, "", {})
        return AgentResult(
            exit_code=0, stdout="", result_json=result, session_ref=run_id,
            trace_path=str(trace_file) if trace_file else None,
        )

    @staticmethod
    def _call_entry_point(
        skill_path: Path, context: RoleContext
    ) -> tuple[dict[str, Any] | None, str | None]:
        module_name = f"pm_workflows_role_{skill_path.stem}_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, skill_path)
        if spec is None or spec.loader is None:
            return None, f"could not load python role skill: {skill_path}"

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            entry = getattr(module, ENTRY_POINT, None)
            if not callable(entry):
                return None, (
                    f"{skill_path} defines no callable "
                    f"'{ENTRY_POINT}(context)' entry point"
                )
            result = entry(context)
        except Exception as exc:  # arbitrary code, arbitrary failure
            return None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        finally:
            sys.modules.pop(module_name, None)

        if not isinstance(result, dict):
            return None, (
                f"{skill_path}:{ENTRY_POINT}() returned "
                f"{type(result).__name__}, expected a dict result object"
            )
        return result, None
