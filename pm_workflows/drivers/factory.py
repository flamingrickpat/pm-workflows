"""Runtime selection of an agent driver."""
from __future__ import annotations

from .base import AgentDriver
from .claude import ClaudeDriver
from .codex import CodexDriver
from .minimal_agent import PmCoderDriver
from .pi import PiDriver
from .python_driver import PythonDriver

SUPPORTED_DRIVERS = ("claude", "codex", "pi", "pm-coder", "minimal_agent", "python")


def build_driver(
    kind: str,
    model: str = "",
    effort: str = "",
    add_dirs: list[str] | None = None,
    base_url: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    max_agent_requests: int | None = None,
    timeout_seconds: int = 7200,
) -> AgentDriver:
    if kind == "claude":
        return ClaudeDriver(
            model=model,
            effort=effort,
            add_dirs=add_dirs,
            timeout_seconds=timeout_seconds,
        )
    if kind == "codex":
        return CodexDriver(
            model=model,
            effort=effort,
            add_dirs=add_dirs,
            timeout_seconds=timeout_seconds,
        )
    if kind == "pi":
        return PiDriver(
            model=model,
            effort=effort,
            timeout_seconds=timeout_seconds,
        )
    if kind in {"pm-coder", "minimal_agent"}:
        return PmCoderDriver(
            model=model,
            effort=effort,
            base_url=base_url,
            api_key_env=api_key_env,
            max_turns=max_agent_requests or 80,
            timeout_seconds=timeout_seconds,
        )
    if kind == "python":
        return PythonDriver()
    supported = ", ".join(SUPPORTED_DRIVERS)
    raise ValueError(f"unknown agent '{kind}'; choose one of: {supported}")
