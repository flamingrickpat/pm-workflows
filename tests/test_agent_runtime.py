"""Runtime-bound drivers reject missing bindings before model access."""
import threading

import pytest

from pm_workflows.drivers.minimal_agent import PmCoderDriver
from pm_workflows.runtime import WorkflowRuntime, WorkflowTerminated
from pm_workflows.working_target import WorkingEnvironment


def test_cancelled_agent_never_opens_a_session(tmp_path):
    stop = threading.Event()
    stop.set()
    runtime = WorkflowRuntime(None, stop, {})
    with pytest.raises(WorkflowTerminated):
        PmCoderDriver().run_session("cancel", 1, "", "unused", tmp_path, runtime=runtime)


def test_agent_requires_explicit_model(tmp_path):
    runtime = WorkflowRuntime(None, threading.Event(), {})
    with pytest.raises(ValueError, match="model"):
        PmCoderDriver().run_session("missing", 1, "", "unused", tmp_path, runtime=runtime)


def test_old_environment_rejects_agent_binding(tmp_path):
    with pytest.raises(NotImplementedError):
        WorkingEnvironment.agent_target(None, tmp_path)
