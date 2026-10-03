"""Actual Qwen roles through the kernel, with folder and virtual file tools."""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml
from pm_bash_machine import Access, BashMachine

from pm_workflows.kernel import Kernel
from pm_workflows.drivers.minimal_agent import PmCoderDriver
from pm_workflows.runtime import AgentModel, WorkflowRuntime, WorkflowTerminated
from pm_workflows.working_target import AgentTarget


class BoundEnvironment:
    def __init__(self, target):
        self.target = target

    def agent_target(self, artifact_workspace):
        return self.target


@pytest.mark.parametrize("virtual", [False, True])
def test_real_kernel_agent_uses_supplied_target(tmp_path, virtual):
    root = tmp_path / "target"
    root.mkdir()
    marker = "ACTUAL_TARGET_79263"
    machine = BashMachine() if virtual else None
    if virtual:
        machine.write_text("/work/input.txt", marker, access=Access.R)
        machine.add_user("workflow", cwd="/work")
    else:
        (root / "input.txt").write_text(marker, encoding="utf-8")
    target = AgentTarget(root, machine, "workflow")
    skill = tmp_path / "role.md"
    skill.write_text("Use the read tool to read input.txt in your tool working directory. "
                     "Use the write tool to copy its contents to output.txt. "
                     "Do not use shell commands or subagents. Use only relative file paths. "
                     "Return JSON with status done and summary equal to the input text.", encoding="utf-8")
    manifest = tmp_path / "probe.workflow.md"
    manifest.write_text("---\n" + yaml.safe_dump({
        "name": "agent-target", "driver": {"kind": "pm-coder"},
        "roles": {"work": {"skill": str(skill), "result_contract": {"schema": {
            "status": {"enum": ["done"]}, "summary": "string",
        }}}},
        "phases": [{"name": "work", "kind": "role", "role": "work",
                    "on_status": {"done": "stop"}, "on_invalid": {"action": "stop"}}],
    }) + "---\n", encoding="utf-8")
    runtime = WorkflowRuntime(BoundEnvironment(target), threading.Event(), {}, agent_model=AgentModel(
        base_url=os.environ.get("LOCAL_AGENT_BASE_URL", "http://127.0.0.1:8080/v1"),
        model=os.environ.get("LOCAL_AGENT_MODEL", "qwen"),
        api_key=os.environ.get("LOCAL_AGENT_API_KEY", "local"),
        context_window=96000, enable_thinking=False, live_test=True,
    ))
    kernel = Kernel(manifest_path=manifest, workspace=tmp_path / "journal-workspace",
                    task_id="copy", base_dir=tmp_path, kernel_data_root=tmp_path / "kernel",
                    run_id="bound", runtime=runtime)
    result = kernel.run()
    assert result["terminal_status"] == "done", result
    output = machine.read_text("/work/output.txt") if virtual else (root / "output.txt").read_text(encoding="utf-8")
    assert output.strip() == marker
    assert not (kernel.workspace / "output.txt").exists()
    if virtual:
        assert not (root / "output.txt").exists()
        assert machine.read_text("/work/input.txt") == marker
    sessions = list((kernel.kernel_data / "pm-coder").glob("*/messages.json"))
    assert len(sessions) == 1
    messages = json.loads(sessions[0].read_text(encoding="utf-8"))
    calls = {part["tool_name"] for message in messages for part in message["parts"] if part["part_kind"] == "tool-call"}
    assert {"read", "write"} <= calls


def test_cancellation_during_actual_inference(tmp_path):
    stop = threading.Event()
    logs = tmp_path / "logs"
    runtime = WorkflowRuntime(BoundEnvironment(AgentTarget(tmp_path)), stop, {}, agent_model=AgentModel(
        base_url=os.environ.get("LOCAL_AGENT_BASE_URL", "http://127.0.0.1:8080/v1"),
        model=os.environ.get("LOCAL_AGENT_MODEL", "qwen"),
        api_key=os.environ.get("LOCAL_AGENT_API_KEY", "local"),
        context_window=96000, live_test=True,
    ))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(PmCoderDriver(log_root=logs).run_session,
                             "cancel", 1, "", "Write a detailed essay about every integer from 1 to 1000. Do not use tools.",
                             tmp_path, runtime=runtime)
        try:
            deadline = time.monotonic() + 30
            while not list(logs.glob("*/turn_*_agent_*.json")):
                if future.done():
                    pytest.fail(f"Agent finished before cancellation: {future.result()}")
                assert time.monotonic() < deadline, "No inference request was observed."
                time.sleep(0.01)
            stop.set()
            with pytest.raises(WorkflowTerminated):
                future.result(timeout=10)
        finally:
            stop.set()
    assert list(logs.glob("*/turn_*_agent_*.json"))


def test_bound_session_deadline_cancels_actual_inference(tmp_path):
    runtime = WorkflowRuntime(BoundEnvironment(AgentTarget(tmp_path)), threading.Event(), {}, agent_model=AgentModel(
        base_url=os.environ.get("LOCAL_AGENT_BASE_URL", "http://127.0.0.1:8080/v1"),
        model=os.environ.get("LOCAL_AGENT_MODEL", "qwen"),
        api_key=os.environ.get("LOCAL_AGENT_API_KEY", "local"),
        context_window=96000, live_test=True,
    ))
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="session deadline"):
        PmCoderDriver(log_root=tmp_path / "logs", timeout_seconds=0.05).run_session(
            "deadline", 1, "", "Write a detailed essay about every integer from 1 to 1000. Do not use tools.",
            tmp_path, runtime=runtime)
    assert time.monotonic() - started < 10
    assert not runtime.stop_event.is_set()
