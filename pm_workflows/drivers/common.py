"""Shared plumbing for CLI-backed agent drivers."""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ratelimit import ProcessError, TokenLimitError, run_agent_process


def deployed_mcp_config(work_dir: Path) -> Path | None:
    """Return the target repository's workflow-local MCP configuration."""
    path = Path(work_dir) / ".mcp.json"
    return path if path.is_file() else None


def codex_mcp_override(
    work_dir: Path, mcp_config: Path | None = None
) -> str | None:
    """Serialize the target .mcp.json servers as a Codex TOML override."""
    path = Path(mcp_config) if mcp_config is not None else deployed_mcp_config(work_dir)
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        raise ValueError(f"MCP config must contain mcpServers: {path}")

    def toml(value: object) -> str:
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if value is True:
            return "true"
        if value is False:
            return "false"
        if value is None:
            raise ValueError("Codex MCP config does not support null values")
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            return "[" + ",".join(toml(item) for item in value) + "]"
        if isinstance(value, dict):
            entries = []
            for key, item in value.items():
                entries.append(f"{json.dumps(str(key))}={toml(item)}")
            return "{" + ",".join(entries) + "}"
        raise ValueError(f"unsupported MCP config value: {type(value).__name__}")

    return "mcp_servers=" + toml(servers)


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the last complete JSON object out of an agent's final message."""
    if not text:
        return None
    for pattern in (r"```json\s*\n(.*?)\n```", r"```\s*\n(.*?)\n```"):
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    candidates: list[dict[str, Any]] = []
    for start, character in enumerate(text):
        if character != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            current = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:end + 1])
                    except json.JSONDecodeError:
                        pass
                    else:
                        if isinstance(parsed, dict):
                            candidates.append(parsed)
                    break
    return candidates[-1] if candidates else None


def trace_write(trace_file: Path | None, entry: dict[str, Any]) -> None:
    if trace_file is None:
        return
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def write_result_artifact(
    result_file: Path | None,
    result_json: dict[str, Any] | None,
    final_text: str,
    usage: dict[str, Any] | None = None,
) -> None:
    if result_file is None:
        return
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(
        json.dumps(
            {
                "result": result_json,
                "final_text": final_text,
                "usage": usage or {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class CommandOutcome:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


def invoke_cli(
    agent_kind: str,
    command: list[str],
    work_dir: Path,
    prompt: str,
    timeout_seconds: int,
) -> CommandOutcome:
    """Run a CLI while preserving token-limit exceptions for the kernel."""
    # The kernel may need PYTHONPATH for a source checkout or repaired venv.
    # Do not leak it into a coding agent: its target interpreter must resolve
    # compiled packages (for example ``regex``) from the target environment.
    child_environment = os.environ.copy()
    child_environment.pop("PYTHONPATH", None)
    try:
        completed = run_agent_process(
            agent_kind,
            command,
            work_dir,
            timeout=timeout_seconds,
            input_text=prompt,
            environment=child_environment,
        )
        return CommandOutcome(
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            returncode=completed.returncode,
        )
    except TokenLimitError:
        raise
    except ProcessError as exc:
        return CommandOutcome(
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            returncode=exc.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return CommandOutcome(
            stdout=stdout,
            stderr=stderr,
            returncode=-1,
            timed_out=True,
        )
