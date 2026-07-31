"""The small interface every agent driver implements."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..protocol import AgentResult


class AgentDriver(Protocol):
    """One fresh role session in, one normalized result out."""

    kind: str

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
    ) -> AgentResult: ...
