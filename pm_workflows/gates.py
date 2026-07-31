"""Check execution: run a gate or script phase and read its verdict.

A check is a plain program. It passes by exiting 0, fails by exiting non-zero,
and may print a JSON object with `ok` and `errors` to say why. The `errors` it
prints become the feedback the retried role sees, so they should read as
instructions to a developer, not as a stack trace.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .protocol import GateResult

CHECK_TIMEOUT_SECONDS = 900


def run_gate(
    script_path: str | Path,
    workspace: Path,
    env: dict | None = None,
    args: list[str] | None = None,
) -> GateResult:
    script = Path(script_path)
    workspace = Path(workspace)

    if not script.is_absolute():
        script = script.resolve()
    if not script.exists():
        return GateResult(
            ok=False,
            errors=[f"check script not found: {script}"],
            exit_code=-1,
        )

    extra = [str(a) for a in (args or [])]
    if script.suffix == ".py":
        command = [sys.executable, str(script), str(workspace), *extra]
    elif script.suffix == ".ps1":
        command = [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(script), str(workspace), *extra,
        ]
    else:
        command = ["bash", str(script), str(workspace), *extra]

    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CHECK_TIMEOUT_SECONDS,
            env=env,
            cwd=str(workspace),
        )
    except subprocess.TimeoutExpired:
        return GateResult(
            ok=False,
            errors=[f"check timed out after {CHECK_TIMEOUT_SECONDS}s: {script.name}"],
            exit_code=-1,
        )

    output = (proc.stdout or "").strip()
    ok = proc.returncode == 0
    errors: list[str] = []

    parsed = _parse_verdict(output)
    if parsed is not None:
        ok = bool(parsed.get("ok", ok))
        raw_errors = parsed.get("errors", [])
        if isinstance(raw_errors, list):
            errors = [str(e) for e in raw_errors]
        elif raw_errors:
            errors = [str(raw_errors)]
    elif not ok:
        detail = (proc.stderr or "").strip() or output
        errors = [detail[-1500:] if detail else f"{script.name} failed with no output"]

    if not ok and not errors:
        errors = [f"{script.name} reported failure"]

    return GateResult(ok=ok, errors=errors, output=output, exit_code=proc.returncode)


def _parse_verdict(output: str) -> dict | None:
    """Read a JSON verdict from the check's output, ignoring leading chatter."""
    if not output:
        return None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        pass
    else:
        return parsed if isinstance(parsed, dict) else None
    # Tolerate a check that prints progress lines before its verdict object.
    start = output.rfind("{")
    while start != -1:
        try:
            parsed = json.loads(output[start:])
        except json.JSONDecodeError:
            start = output.rfind("{", 0, start)
            continue
        return parsed if isinstance(parsed, dict) else None
    return None
