"""Runtime-bound drivers reject missing bindings before model access."""
import threading

import pytest

from pm_workflows.drivers.minimal_agent import (
    PmCoderDriver, _is_provider_exhaustion,
)
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


def test_permanent_rejection_counts_as_provider_exhaustion():
    try:
        from pm_coder import PermanentProviderError
    except ImportError:  # pragma: no cover - older installed coder
        pytest.skip("installed pm-coder predates the permanent-error policy")
    assert _is_provider_exhaustion(PermanentProviderError("invalid api key"))
    assert not _is_provider_exhaustion(ValueError("the model returned malformed JSON"))
