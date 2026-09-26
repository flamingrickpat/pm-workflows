"""Per-invocation runtime context handed into a workflow role.

The kernel does not interpret ``tools`` or import application-specific
objects such as a Ghost. A role reads and writes through ``environment``,
checks cooperative cancellation through ``stop_event``, and calls only the
tool functions the application supplied for this invocation.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .working_target import WorkingEnvironment


class WorkflowTerminated(Exception):
    """Raised when an invocation is cancelled between cooperative steps."""


@dataclass(frozen=True)
class AgentModel:
    """Resolved connection and context size. Credentials never enter repr."""

    base_url: str
    model: str
    api_key: str = field(repr=False)
    context_window: int
    enable_thinking: bool = True
    live_test: bool = False

    def __post_init__(self) -> None:
        if not self.model or self.context_window <= 0:
            raise ValueError("Supply an agent model and a positive context window.")


@dataclass
class WorkflowRuntime:
    environment: WorkingEnvironment
    stop_event: threading.Event
    tools: Mapping[str, Callable[..., object]]
    propagate_python_errors: bool = False
    agent_model: AgentModel | None = None

    def check_cancelled(self) -> None:
        if self.stop_event.is_set():
            raise WorkflowTerminated()
