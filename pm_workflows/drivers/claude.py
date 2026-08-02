"""Claude Code CLI driver."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from ..protocol import AgentResult
from .common import (
    deployed_mcp_config,
    extract_json,
    invoke_cli,
    trace_write,
    write_result_artifact,
)


class ClaudeDriver:
    kind = "claude"
    supports_explicit_mcp_config = True

    def __init__(
        self,
        model: str = "",
        effort: str = "",
        add_dirs: list[str] | None = None,
        timeout_seconds: int = 7200,
    ) -> None:
        self.model = model
        self.effort = effort
        self.add_dirs = [str(Path(directory)) for directory in (add_dirs or [])]
        self.timeout_seconds = timeout_seconds

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
    ) -> AgentResult:
        work_dir = Path(work_dir)
        session_id = str(uuid.uuid4())
        command = ["claude", "-p", "--dangerously-skip-permissions"]
        if self.model:
            command.extend(("--model", self.model))
        if self.effort:
            command.extend(("--effort", self.effort))
        command.extend((
            "--session-id",
            session_id,
            "--no-session-persistence",
            "--output-format",
            "json",
        ))
        for directory in self.add_dirs:
            command.extend(("--add-dir", directory))
        mcp_config = (
            Path(mcp_config) if mcp_config is not None else deployed_mcp_config(work_dir)
        )
        if mcp_config is not None:
            command.extend((
                "--mcp-config",
                str(mcp_config),
                "--strict-mcp-config",
            ))

        trace_write(trace_file, {
            "event": "start",
            "agent": self.kind,
            "model": self.model,
            "effort": self.effort,
            "session_id": session_id,
            "skill": skill,
            "work_dir": str(work_dir),
            "command": command,
            "prompt": prompt,
        })
        outcome = invoke_cli(
            self.kind, command, work_dir, prompt, self.timeout_seconds
        )
        final_text, usage = self._unwrap(outcome.stdout)
        result_json = extract_json(final_text)
        error = None
        if outcome.timed_out:
            error = f"claude session timed out after {self.timeout_seconds}s"
        elif outcome.returncode != 0:
            error = (
                outcome.stderr or final_text or "claude exited non-zero"
            )[-2000:]

        trace_write(trace_file, {
            "event": "end",
            "returncode": outcome.returncode,
            "usage": usage,
            "final_text": final_text,
            "stderr": outcome.stderr[-4000:],
            "result": result_json,
        })
        write_result_artifact(result_file, result_json, final_text, usage)
        return AgentResult(
            exit_code=outcome.returncode,
            stdout=final_text,
            result_json=result_json,
            usage=usage,
            error=error,
            trace_path=str(trace_file) if trace_file else None,
            session_ref=session_id,
        )

    @staticmethod
    def _unwrap(stdout: str) -> tuple[str, dict[str, Any]]:
        text = (stdout or "").strip()
        if not text:
            return "", {}
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError:
            return text, {}
        if not isinstance(envelope, dict):
            return text, {}
        final = envelope.get("result")
        if not isinstance(final, str):
            return text, {}
        usage: dict[str, Any] = {}
        if isinstance(envelope.get("usage"), dict):
            usage.update(envelope["usage"])
        for key in ("total_cost_usd", "num_turns", "duration_ms", "session_id"):
            if key in envelope:
                usage[key] = envelope[key]
        return final, usage
