"""OpenAI Codex CLI driver using `codex exec`."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from ..protocol import AgentResult
from .common import (
    codex_mcp_override,
    extract_json,
    invoke_cli,
    trace_write,
    write_result_artifact,
)


def _events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


class CodexDriver:
    kind = "codex"
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
        with tempfile.TemporaryDirectory(prefix="workflow-codex-") as temp:
            last_message = Path(temp) / "last-message.txt"
            command = [
                "codex",
                "exec",
                "-C",
                str(work_dir),
                "--dangerously-bypass-approvals-and-sandbox",
                "--ephemeral",
                "--json",
                "--output-last-message",
                str(last_message),
            ]
            if self.model:
                command.extend(("--model", self.model))
            if self.effort:
                command.extend((
                    "--config",
                    f'model_reasoning_effort="{self.effort}"',
                ))
            mcp_override = codex_mcp_override(work_dir, mcp_config)
            if mcp_override is not None:
                command.extend(("--config", mcp_override))
            for directory in self.add_dirs:
                command.extend(("--add-dir", directory))
            command.append("-")

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
            events = _events(outcome.stdout)
            for event in events:
                trace_write(trace_file, event)

            final_text = (
                last_message.read_text(encoding="utf-8")
                if last_message.is_file()
                else self._last_agent_message(events)
            ).strip()
            session_ref = self._thread_id(events)
            usage = self._usage(events)
            result_json = extract_json(final_text)
            error = None
            if outcome.timed_out:
                error = f"codex session timed out after {self.timeout_seconds}s"
            elif outcome.returncode != 0:
                error = (
                    outcome.stderr or final_text or "codex exited non-zero"
                )[-2000:]

            trace_write(trace_file, {
                "event": "driver.end",
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
                session_ref=session_ref,
            )

    @staticmethod
    def _thread_id(events: list[dict[str, Any]]) -> str:
        for event in events:
            if event.get("type") == "thread.started":
                value = event.get("thread_id")
                return value if isinstance(value, str) else ""
        return ""

    @staticmethod
    def _usage(events: list[dict[str, Any]]) -> dict[str, Any]:
        for event in reversed(events):
            if event.get("type") == "turn.completed":
                usage = event.get("usage")
                return usage if isinstance(usage, dict) else {}
        return {}

    @staticmethod
    def _last_agent_message(events: list[dict[str, Any]]) -> str:
        for event in reversed(events):
            item = event.get("item")
            if (
                event.get("type") == "item.completed"
                and isinstance(item, dict)
                and item.get("type") == "agent_message"
                and isinstance(item.get("text"), str)
            ):
                return item["text"]
        return ""
