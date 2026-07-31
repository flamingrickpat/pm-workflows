from __future__ import annotations

import subprocess
import json
from pathlib import Path

from pm_workflows.drivers import (
    ClaudeDriver,
    CodexDriver,
    MinimalAgentDriver,
    PiDriver,
    build_driver,
)
from pm_workflows.drivers.common import CommandOutcome
from pm_workflows.manifest import parse_workflow


def test_factory_has_one_driver_per_supported_agent() -> None:
    assert isinstance(build_driver("claude"), ClaudeDriver)
    assert isinstance(build_driver("codex"), CodexDriver)
    assert isinstance(build_driver("pi"), PiDriver)
    assert isinstance(build_driver("minimal_agent"), MinimalAgentDriver)


def test_codex_uses_exec_stdin_and_last_message(
    tmp_path: Path, monkeypatch
) -> None:
    work = tmp_path / "repo"
    work.mkdir()
    (work / ".mcp.json").write_text(json.dumps({
        "mcpServers": {
            "workflow": {"command": "example-mcp", "args": ["--stdio"]}
        }
    }), encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_invoke(kind, command, work_dir, prompt, timeout):
        seen.update(kind=kind, command=command, prompt=prompt)
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text('{"status":"done","summary":"ok"}', encoding="utf-8")
        return CommandOutcome(
            stdout=(
                '{"type":"thread.started","thread_id":"thread-1"}\n'
                '{"type":"turn.completed","usage":{"input_tokens":10,'
                '"output_tokens":4}}\n'
            ),
            stderr="progress belongs on stderr",
            returncode=0,
        )

    monkeypatch.setattr("pm_workflows.drivers.codex.invoke_cli", fake_invoke)
    result = CodexDriver(model="gpt-test", effort="high").run_session(
        "run", 1, "skill.md", "do it", work
    )

    command = seen["command"]
    assert seen["kind"] == "codex"
    assert seen["prompt"] == "do it"
    assert command[-1] == "-"
    assert "--ephemeral" in command
    assert command[command.index("--config", command.index("--config") + 1) + 1] == (
        'mcp_servers={"workflow"={"command"="example-mcp",'
        '"args"=["--stdio"]}}'
    )
    assert result.result_json == {"status": "done", "summary": "ok"}
    assert result.session_ref == "thread-1"
    assert result.usage["input_tokens"] == 10


def test_pi_loads_the_selected_skill_and_pipes_the_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    work = tmp_path / "repo"
    work.mkdir()
    seen: dict[str, object] = {}

    def fake_invoke(kind, command, work_dir, prompt, timeout):
        seen.update(kind=kind, command=command, prompt=prompt)
        return CommandOutcome(
            stdout='{"status":"done","summary":"ok"}',
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("pm_workflows.drivers.pi.invoke_cli", fake_invoke)
    result = PiDriver(model="provider/model", effort="medium").run_session(
        "run", 1, r"C:\base\skill.md", "do it", work
    )

    command = seen["command"]
    assert seen["kind"] == "pi"
    assert seen["prompt"] == "do it"
    assert command[command.index("--skill") + 1] == r"C:\base\skill.md"
    assert result.result_json == {"status": "done", "summary": "ok"}


def test_claude_uses_only_the_deployed_mcp_config(
    tmp_path: Path, monkeypatch
) -> None:
    work = tmp_path / "repo"
    work.mkdir()
    mcp = work / ".mcp.json"
    mcp.write_text('{"mcpServers":{"workflow":{"command":"example-mcp"}}}', encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_invoke(kind, command, work_dir, prompt, timeout):
        seen.update(kind=kind, command=command)
        return CommandOutcome(stdout='{"status":"done"}', stderr="", returncode=0)

    monkeypatch.setattr("pm_workflows.drivers.claude.invoke_cli", fake_invoke)
    ClaudeDriver().run_session("run", 1, "skill.md", "do it", work)

    command = seen["command"]
    assert command[command.index("--mcp-config") + 1] == str(mcp)
    assert "--strict-mcp-config" in command


def test_pm_coder_uses_the_deployed_mcp_config(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "repo"
    work.mkdir()
    mcp = work / ".mcp.json"
    mcp.write_text('{"mcpServers":{"workflow":{"command":"example-mcp"}}}', encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_run(prompt, **kwargs):
        seen.update(prompt=prompt, kwargs=kwargs)
        return {"response": '{"status":"done"}', "tokens_used": {"requests": 1}}

    monkeypatch.setattr("pm_workflows.drivers.minimal_agent.run_auto_sync", fake_run)
    result = MinimalAgentDriver().run_session("run", 1, "skill.md", "do it", work)
    assert seen["kwargs"]["mcp_config"] == mcp
    assert result.result_json == {"status": "done"}


def test_pm_coder_writes_kernel_artifacts(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "repo"
    work.mkdir()
    trace = tmp_path / "private" / "trace.jsonl"
    result_file = tmp_path / "private" / "result.json"
    def fake_run(prompt, **kwargs):
        return {"response": '{"status":"done","summary":"ok"}', "tokens_used": {"requests": 3}}

    monkeypatch.setattr("pm_workflows.drivers.minimal_agent.run_auto_sync", fake_run)
    result = MinimalAgentDriver(model="qwen").run_session(
        "run",
        1,
        "skill.md",
        "do it",
        work,
        result_file=result_file,
        trace_file=trace,
    )

    assert result.result_json == {"status": "done", "summary": "ok"}
    assert result.usage["requests"] == 3
    assert Path(result.trace_path).is_file()


def test_pm_coder_preserves_full_exception_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    work = tmp_path / "repo"
    work.mkdir()
    trace = tmp_path / "private" / "trace.jsonl"

    def fake_run(prompt, **kwargs):
        raise RuntimeError("diagnostic test failure")

    monkeypatch.setattr("pm_workflows.drivers.minimal_agent.run_auto_sync", fake_run)
    result = MinimalAgentDriver().run_session(
        "run", 1, "skill.md", "do it", work, trace_file=trace
    )

    assert result.exit_code == 1
    event = json.loads(trace.read_text(encoding="utf-8").splitlines()[-1])
    assert event["exception_type"] == "RuntimeError"
    assert event["exception_message"] == "diagnostic test failure"
    assert "RuntimeError: diagnostic test failure" in event["exception_traceback"]


def test_shipped_workflows_are_agent_neutral() -> None:
    root = Path(__file__).resolve().parents[1]
    variables = {
        "BASE": str(root / "agents-deploy"),
        "TARGET": str(root),
        "WORKSPACE": str(root),
        "TASK_ID": "test",
        "TASK_DIR": str(root / "agents" / "tasks" / "test"),
    }
    for path in sorted(
        (root / "agents-deploy" / "workflows").glob("*.workflow.md")
    ):
        workflow = parse_workflow(path, variables)
        assert workflow.driver.kind == "", path.name
        assert workflow.driver.model == "", path.name
