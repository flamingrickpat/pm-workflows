"""Pi coding-agent CLI driver."""
from __future__ import annotations

from pathlib import Path

from ..protocol import AgentResult
from .common import (
    extract_json,
    invoke_cli,
    trace_write,
    write_result_artifact,
)


class PiDriver:
    kind = "pi"

    def __init__(
        self,
        model: str = "",
        effort: str = "",
        timeout_seconds: int = 7200,
    ) -> None:
        self.model = model
        self.effort = effort
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
    ) -> AgentResult:
        work_dir = Path(work_dir)
        command = [
            "pi",
            "--print",
            "--approve",
            "--no-session",
            "--mode",
            "text",
            "--skill",
            str(skill),
        ]
        if self.model:
            command.extend(("--model", self.model))
        if self.effort:
            command.extend(("--thinking", self.effort))
        mcp_config = work_dir / ".mcp.json"
        if mcp_config.is_file():
            command.extend(("--mcp-config", str(mcp_config)))

        trace_write(trace_file, {
            "event": "start",
            "agent": self.kind,
            "model": self.model,
            "effort": self.effort,
            "skill": skill,
            "work_dir": str(work_dir),
            "command": command,
            "prompt": prompt,
        })
        outcome = invoke_cli(
            self.kind, command, work_dir, prompt, self.timeout_seconds
        )
        final_text = outcome.stdout.strip()
        result_json = extract_json(final_text)
        error = None
        if outcome.timed_out:
            error = f"pi session timed out after {self.timeout_seconds}s"
        elif outcome.returncode != 0:
            error = (
                outcome.stderr or final_text or "pi exited non-zero"
            )[-2000:]

        trace_write(trace_file, {
            "event": "end",
            "returncode": outcome.returncode,
            "final_text": final_text,
            "stderr": outcome.stderr[-4000:],
            "result": result_json,
        })
        write_result_artifact(result_file, result_json, final_text)
        return AgentResult(
            exit_code=outcome.returncode,
            stdout=final_text,
            result_json=result_json,
            error=error,
            trace_path=str(trace_file) if trace_file else None,
        )
