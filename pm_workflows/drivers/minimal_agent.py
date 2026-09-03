"""Driver adapter for the separately packaged :mod:`pm_coder` agent."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

from pm_coder import run_auto

from ..protocol import AgentResult
from .common import deployed_mcp_config, extract_json, trace_write


class ProviderExhaustedError(RuntimeError):
    """The local model endpoint stayed unusable after the provider retries.

    Live loop deployments treat this as a fatal infrastructure fault rather
    than an invalid agent response. Standalone callers keep the historical
    normalization by leaving ``fatal_provider_exhaustion`` disabled.
    """


class PmCoderDriver:
    """Run one fresh role session through the installed ``pm-coder`` package."""

    kind = "pm-coder"
    supports_explicit_mcp_config = True

    def __init__(
        self,
        model: str = "",
        effort: str = "",
        base_url: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        max_turns: int = 80,
        timeout_seconds: int = 7200,
        fatal_provider_exhaustion: bool = False,
        log_root: Path | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.base_url = base_url or "http://127.0.0.1:8080/v1"
        self.api_key_env = api_key_env
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds
        self.fatal_provider_exhaustion = fatal_provider_exhaustion
        self.log_root = Path(log_root) if log_root is not None else None

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
        work_dir = Path(work_dir).resolve()
        mcp_config = (
            Path(mcp_config) if mcp_config is not None else deployed_mcp_config(work_dir)
        )
        trace_write(trace_file, {
            "event": "start",
            "agent": self.kind,
            "model": self.model,
            "effort": self.effort,
            "skill": skill,
            "work_dir": str(work_dir),
        })
        try:
            payload = run_auto(
                prompt,
                cwd=work_dir,
                run_id=f"{run_id}_{attempt}",
                log_root=(
                    self.log_root
                    or (trace_file.parent.parent / "pm-coder")
                    if trace_file
                    else Path.home() / ".pm" / "pm-coder"
                ),
                base_url=self.base_url,
                api_key=os.environ.get(self.api_key_env) or "local",
                model=self.model or None,
                mcp_config=mcp_config,
                enable_thinking=self.effort in {"high", "xhigh", "max", "ultra"},
            )
            stdout = str(payload.get("response", ""))
            result_json = extract_json(stdout)
            usage = payload.get("tokens_used", {})
            session_ref = str(payload.get("run_id", "") or "")
            error = None
            exit_code = 0
        except Exception as exc:  # Agent failures are normalized for the kernel.
            if self.fatal_provider_exhaustion and _is_provider_exhaustion(exc):
                raise ProviderExhaustedError(
                    f"pm-coder exhausted provider retries: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            stdout = ""
            result_json = None
            session_ref = ""
            usage = {}
            exception_type = type(exc).__name__
            exception_message = str(exc)
            exception_traceback = traceback.format_exc()
            # Keep the public/kernel error short. The complete diagnostic is
            # written to the private trace artifact below so it cannot be fed
            # back into the model as workflow feedback.
            error = f"{exception_type}: {exception_message}"[-2000:]
            exit_code = 1

        if result_file is not None:
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(
                json.dumps(result_json, ensure_ascii=False, indent=2)
                if result_json is not None else stdout,
                encoding="utf-8",
            )
        trace_write(trace_file, {
            "event": "end",
            "returncode": exit_code,
            "usage": usage,
            "final_text": stdout,
            "error": error,
            "result": result_json,
            "exception_type": exception_type if exit_code else None,
            "exception_message": exception_message if exit_code else None,
            "exception_traceback": exception_traceback if exit_code else None,
        })
        return AgentResult(
            exit_code=exit_code,
            stdout=stdout,
            result_json=result_json,
            usage=usage if isinstance(usage, dict) else {},
            error=error,
            trace_path=str(trace_file) if trace_file else None,
            session_ref=session_ref,
        )


_PROVIDER_EXHAUSTION_MARKERS = (
    "endpoint did not recover",
    "endpoint unavailable",
    "wall-clock limit",
)


def _is_provider_exhaustion(exc: BaseException) -> bool:
    """True when the failure is the model endpoint, not the model output."""
    # pm-coder does not export a specific exhaustion error type.
    if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
        return True
    message = str(exc).casefold()
    return any(marker in message for marker in _PROVIDER_EXHAUSTION_MARKERS)


# Compatibility name for workflow manifests that still say ``minimal_agent``.
MinimalAgentDriver = PmCoderDriver
