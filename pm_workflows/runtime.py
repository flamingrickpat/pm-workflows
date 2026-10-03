"""Per-invocation runtime context handed into a workflow role.

The kernel does not interpret ``tools`` or import application-specific
objects such as a Ghost. A role reads and writes through ``environment``,
checks cooperative cancellation through ``stop_event``, and calls only the
tool functions the application supplied for this invocation.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Mapping

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
    # Named scoped services the run may use over MCP. The application starts
    # the listeners and supplies connection descriptors here; the kernel only
    # selects and writes a per-run config. ``None`` means no runtime services
    # (the legacy shared ``.mcp.json`` discovery still applies).
    mcp_servers: Mapping[str, Mapping[str, Any]] | None = None
    # Opens the runtime for one child invocation. ``None`` reuses this runtime
    # so existing embedders keep their single-environment behavior.
    child_factory: Callable[[str], "WorkflowRuntime"] | None = None
    # Classification connections are explicit and independent of the agent.
    # An absent name produces INVALID. Existing roles need no new connection.
    classification_models: Mapping[str, AgentModel] = field(default_factory=dict)
    # The application can activate a connection only at a classification call.
    # The session holds service ownership until direct inference or recovery ends.
    # Arguments are the connection name, recovery flag, and cancellation signal.
    classification_session: Callable[[str, bool, threading.Event], ContextManager[AgentModel]] | None = None

    def check_cancelled(self) -> None:
        if self.stop_event.is_set():
            raise WorkflowTerminated()

    def child(self, scope: str) -> "WorkflowRuntime":
        """Return the runtime for one child invocation of this run."""
        if self.child_factory is None:
            return self
        child = self.child_factory(scope)
        if child.classification_session is None and self.classification_session is not None:
            from dataclasses import replace
            child = replace(child, classification_session=self.classification_session)
        return child
