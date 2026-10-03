"""Manifest and receipt checks. Live semantic acceptance uses the root gate."""
from copy import deepcopy
from dataclasses import asdict
import json
import threading

import pytest
import yaml

from pm_workflows.classification import ClassificationEnvironment, parse_completion, replay_classification, validate_answer
from pm_workflows.dryrun import build_graph, check_workflow
from pm_workflows.inputs import InputError, InputResolver, InputScope, digest
from pm_workflows.kernel import Kernel
from pm_workflows.manifest import ManifestError, parse_workflow
from pm_workflows.runtime import WorkflowRuntime
from pm_workflows.working_target import HostInputEnvironment
from pathlib import Path


def test_captured_array_label_is_rejected():
    response = json.loads((Path(__file__).parent / "fixtures/classification/array-label-response.json").read_text(encoding="utf-8"))
    answer = parse_completion(response)
    assert answer["status"] == ["all_good"]
    with pytest.raises(InputError, match="unknown_label"):
        validate_answer(answer, {"all_good": "All tests passed."}, {})


def manifest(tmp_path, **changes):
    skill = tmp_path / "inspect.md"
    skill.write_text("Inspect the supplied data and return the declared JSON answer.", encoding="utf-8")
    source = {"name": "check", "kind": "classify", "input_file": "logs/result.log",
              "instruction": "Did all tests pass?", "result_contract": {"type": "classes", "enum": {"passed": "All tests passed.", "failed": "Some tests failed."}},
              "on_class": {"passed": "stop", "failed": "stop", "INVALID": "recover"}}
    source.update(changes)
    raw = {"name": "classification", "driver": {"kind": "pm-coder"}, "failure_policy": {"max_attempts_per_phase": 2},
           "roles": {"inspect": {"skill": str(skill), "readable_paths": ["logs/**"]}},
           "phases": [source, {"name": "recover", "kind": "classify", "role": "inspect", "fallback_for": "check",
                                 "on_class": {"passed": "stop", "failed": "stop", "INVALID": "stop"}}]}
    path = tmp_path / "classify.workflow.md"
    path.write_text("---\n" + yaml.safe_dump(raw, sort_keys=False) + "---\n", encoding="utf-8")
    return path, raw


def write_manifest(path, raw):
    path.write_text("---\n" + yaml.safe_dump(raw, sort_keys=False) + "---\n", encoding="utf-8")


@pytest.mark.parametrize("fault", ["missing_label", "stray_label", "missing_invalid", "unknown_target", "no_handler", "recursive", "indirect_recursive", "different_routes", "bad_role", "conflicting_mode", "missing_description", "laya", "bad_limit", "bad_connection"])
def test_reject_contract_holes(tmp_path, fault):
    path, raw = manifest(tmp_path)
    source, fallback = raw["phases"]
    if fault == "missing_label": del source["on_class"]["failed"]
    elif fault == "stray_label": source["on_class"]["other"] = "stop"
    elif fault == "missing_invalid": del source["on_class"]["INVALID"]
    elif fault == "unknown_target": source["on_class"]["passed"] = "absent"
    elif fault == "no_handler": source["on_class"]["INVALID"] = "stop"
    elif fault == "recursive": fallback["on_class"]["INVALID"] = "check"
    elif fault == "indirect_recursive":
        fallback["on_class"]["INVALID"] = "wait"
        raw["human_resolver"] = {"mode": "external"}
        raw["phases"].append({"name": "wait", "kind": "human", "question": "Provide data", "next": "recover"})
    elif fault == "different_routes": fallback["on_class"]["passed"] = "recover"
    elif fault == "bad_role": raw["roles"]["inspect"]["writable_paths"] = ["logs/**"]
    elif fault == "conflicting_mode": source["literal"] = "passed"
    elif fault == "missing_description": source["result_contract"]["enum"]["passed"] = ""
    elif fault == "laya": source["backend"] = "laya"
    elif fault == "bad_limit": source["classification_limits"] = {"max_input_tokens": True}
    elif fault == "bad_connection": source["connection"] = ""
    write_manifest(path, raw)
    with pytest.raises(ManifestError): parse_workflow(path)


def test_dryrun_contains_primary_and_invalid_handler(tmp_path):
    path, _ = manifest(tmp_path)
    report = check_workflow(path, tmp_path, seed=8, monte_carlo_runs=100)
    assert report.errors == []
    assert {edge.outcome for edge in report.nodes["check"].edges} == {"passed", "failed", "INVALID"}
    assert next(edge.target for edge in report.nodes["check"].edges if edge.outcome == "INVALID") == "recover"
    assert report.nodes["recover"].kind == "classify"
    assert next(edge.note for edge in report.nodes["check"].edges if edge.outcome == "INVALID") == "scoped agent recovery"


@pytest.mark.parametrize("mode", ["missing", "malformed", "oversized", "no_connection", "denied"])
def test_invalid_routes_without_inference(tmp_path, mode):
    path, raw = manifest(tmp_path)
    logs = tmp_path / "target" / "logs"
    logs.mkdir(parents=True)
    if mode != "missing": (logs / "result.log").write_text("actual retained input", encoding="utf-8")
    if mode == "malformed": raw["phases"][0]["pointer"] = "/missing"
    if mode == "oversized": raw["phases"][0]["classification_limits"] = {"max_capture_bytes": 1}
    if mode == "denied": raw["roles"]["inspect"]["deny_access"] = ["logs/**"]
    write_manifest(path, raw)
    kernel = Kernel(path, tmp_path / "artifacts", "test", base_dir=tmp_path, kernel_data_root=tmp_path / "kernel",
                    runtime=WorkflowRuntime(HostInputEnvironment(tmp_path / "target"), threading.Event(), {}))
    step = kernel.step()
    assert step.status == "INVALID" and not step.valid and step.next_phase == "recover"
    assert step.data["receipt"]["invalid_code"] == {"missing": "missing_file", "malformed": "missing_field", "oversized": "oversized_input", "no_connection": "missing_connection", "denied": "inaccessible_path"}[mode]
    assert not step.data["receipt"].get("classification")
    resumed = Kernel(path, tmp_path / "artifacts", "test", base_dir=tmp_path, kernel_data_root=tmp_path / "kernel", runtime=kernel.runtime)
    assert resumed.step().phase == "recover"


def test_snapshot_cannot_edit_source_or_read_host(tmp_path):
    env = ClassificationEnvironment({"logs/result.log": "original"})
    target = env.agent_target(tmp_path)
    with pytest.raises(PermissionError): target.bash_machine.write_text_as(target.user, "/work/logs/result.log", "changed")
    (tmp_path / "host-secret").write_text("secret", encoding="utf-8")
    assert target.bash_machine.exec(target.user, "cat host-secret").exit_code != 0
    assert target.bash_machine.exec(target.user, "cat logs/result.log").stdout == "original"


def test_receipt_replay_rejects_changed_value_contract_and_answer(tmp_path):
    # A source capture is a protocol boundary, independent of inference.
    (tmp_path / "log").write_text("Ran 261 tests\nOK\n", encoding="utf-8")
    capture = InputResolver(HostInputEnvironment(tmp_path), InputScope("task", "run")).capture_file("log")
    classes = {"passed": "All tests passed."}
    answer = {"status": "passed", "reason": "The captured footer records OK.", "evidence": ["Ran 261 tests", "OK"], "candidate_id": None, "input_sha256": capture.evidence["value_sha256"]}
    receipt = {"schema": "pm.classification-receipt.v1", "instruction": "Read the footer", "classes": classes,
               "captures": {"exact": capture.evidence}, "answer": answer}
    receipt["contract_sha256"] = digest({"instruction": receipt["instruction"], "classes": classes})
    assert replay_classification(receipt) == answer
    for mutation in ("value", "classes", "quote", "digest", "label", "field"):
        altered = deepcopy(receipt)
        if mutation == "value": altered["captures"]["exact"]["value"] += "FAILED"
        if mutation == "classes": altered["classes"]["passed"] = "No tests passed."
        if mutation == "quote": altered["answer"]["evidence"] = ["FAILED"]
        if mutation == "digest": altered["answer"]["input_sha256"] = "old"
        if mutation == "label": altered["answer"]["status"] = "unknown"
        if mutation == "field": del altered["answer"]["status"]
        with pytest.raises(InputError): replay_classification(altered)


def test_cancelled_classifier_publishes_no_answer(tmp_path):
    path, _ = manifest(tmp_path)
    stop = threading.Event()
    stop.set()
    kernel = Kernel(path, tmp_path / "workspace", "cancel", base_dir=tmp_path, kernel_data_root=tmp_path / "kernel",
                    runtime=WorkflowRuntime(HostInputEnvironment(tmp_path), stop, {}))
    assert kernel.step().exit_reason == "terminated"
    assert not any(e["kind"] == "classify" for e in kernel.journal.read_all())


def test_captured_incomplete_response_keeps_its_cause():
    path = Path(__file__).parent / "fixtures/classification/incomplete-response.json"
    with pytest.raises(InputError) as caught:
        parse_completion(json.loads(path.read_text(encoding="utf-8")))
    assert caught.value.code == "incomplete_response"


def test_attempt_count_and_fallback_recursion_guard(tmp_path):
    path, _ = manifest(tmp_path)
    kernel = Kernel(path, tmp_path / "workspace", "attempts", base_dir=tmp_path, kernel_data_root=tmp_path / "kernel",
                    runtime=WorkflowRuntime(HostInputEnvironment(tmp_path), threading.Event(), {}))
    assert kernel.step().attempt == 1
    assert kernel.step().attempt == 1
    source = kernel.manifest.phase_by_name("check")
    handler = kernel.manifest.phase_by_name("recover")
    assert kernel.phase_attempts(source.name) == 1
    assert kernel._execute_classify(handler)["data"]["receipt"]["invalid_code"] == "fallback_exhausted"


@pytest.mark.parametrize("change", ["file", "contract", "item"])
def test_recovery_rejects_changed_file_contract_or_item(tmp_path, change):
    path, raw = manifest(tmp_path)
    target = tmp_path / "target"
    (target / "logs").mkdir(parents=True)
    (target / "logs/result.log").write_text("Ran 261 tests\nOK", encoding="utf-8")
    kernel = Kernel(path, tmp_path / "workspace", "stale", base_dir=tmp_path, kernel_data_root=tmp_path / "kernel",
                    runtime=WorkflowRuntime(HostInputEnvironment(target), threading.Event(), {}))
    assert kernel.step().data["receipt"]["invalid_code"] == "missing_connection"
    if change == "file": (target / "logs/result.log").write_text("FAILED", encoding="utf-8")
    if change == "contract":
        raw["phases"][0]["instruction"] = "Another question"
        write_manifest(path, raw)
    if change == "item": kernel.current_item = "another-item"
    step = kernel.step()
    assert step.status == "INVALID"
    assert step.data["receipt"]["invalid_code"] == ("missing_invalid_receipt" if change == "item" else "stale_input")
    assert not step.data["receipt"].get("session_ref")
