"""The interface a Python role implements.

A role whose ``skill`` is a ``.py`` file is not dispatched to a coding-agent
CLI. The kernel loads that file as a module, in its own process, and calls
its entry point directly:

    def run(context: RoleContext) -> dict:
        ...

The returned dict is the role's result object. It is validated against the
role's ``result_contract`` exactly like an agentic role's final JSON message
— same status field, same required fields, same routing.

There is no sandbox and no permission model. The script runs with the
kernel's own privileges, in the kernel's own process, synchronously. It may
do anything a Python program can do: read and write the workspace outside
`writable_paths` (that boundary is only ever enforced by prompting an agent
to respect it — nothing stops code), start subprocesses, open sockets, train
a model, shell out to another `pm-workflows` invocation and wait on it, fan
out several of those in parallel and collect their results. It is exactly as
trusted as `pm-workflows` itself. This is the point: some roles need to do
things no agentic skill can do on its own.

A failing entry point — an uncaught exception, a missing `run`, a non-dict
return value — becomes a contract violation, journalled and routed through
`on_invalid` like any other, with the exception traceback attached as the
attempt's error detail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RoleContext:
    """Everything the kernel knows about this attempt.

    This is the same information an agentic role receives folded into prose
    in `prompt` — `task_text`, `feedback`, `current_item`, and the contract
    description are all already embedded in it. The rest is provided as
    structured values because a script should not have to parse them back
    out of natural language.
    """

    run_id: str
    task_id: str
    attempt: int
    workspace: Path
    task_dir: Path
    base_dir: Path
    kernel_data: Path
    role: str
    phase: str
    prompt: str
    task_text: str
    current_item: str | None
    feedback: str | None
    answer: str | None
    tools: list[str] = field(default_factory=list)
    result_file: Path | None = None
    trace_file: Path | None = None
