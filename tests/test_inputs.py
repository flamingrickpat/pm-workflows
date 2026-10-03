"""Input capture tests use actual host and in-memory filesystem operations."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import threading

import pytest
import yaml

from pm_workflows.child import load_reference
from pm_workflows.inputs import FileMetadata, GrantedInputEnvironment, InputError, InputResolver, InputScope, digest, select_pointer
from pm_workflows.kernel import Kernel
from pm_workflows.manifest import ManifestError, parse_workflow
from pm_workflows.protocol import InputBinding, OutputBinding
from pm_workflows.runtime import WorkflowRuntime
from pm_workflows.working_target import CommandResult, HostInputEnvironment, WorkingEnvironment


class MemoryEnvironment(WorkingEnvironment):
    def __init__(self, files=None):
        self.files = dict(files or {})

    def read_text(self, path):
        return self.files[path]

    def write_text(self, path, text):
        self.files[path] = text

    def move(self, source, destination):
        self.files[destination] = self.files.pop(source)

    def list_files(self, directory):
        prefix = "" if directory == "." else directory.rstrip("/") + "/"
        return sorted(path for path in self.files if path.startswith(prefix) and "/" not in path[len(prefix):])

    def exec(self, command):
        raise NotImplementedError


SCOPE = InputScope("task", "run", None)


@pytest.mark.parametrize("host", [True, False])
def test_listing_capture_versions_and_unknown_producer(tmp_path, host):
    files = {"logs/older.txt": "12 passed\r\n", "logs/newer.txt": "1 failed: \u00df\n"}
    if host:
        environment = HostInputEnvironment(tmp_path)
        for name, text in files.items():
            target = tmp_path / name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
    else:
        environment = MemoryEnvironment(files)
    resolver = InputResolver(environment, SCOPE)
    listing = resolver.list_folder("logs/", "Select the latest log.")
    assert [item.path for item in listing.candidates] == sorted(files)
    for item in listing.candidates:
        assert (item.modified_time is not None) == host
        assert item.producer is None
        assert item.size_bytes == len(files[item.path].encode("utf-8"))
        captured = resolver.capture_selected(listing, item.candidate_id)
        assert captured.value == files[item.path]
        assert captured.evidence["limitations"] == ["producer metadata unavailable"]
        assert captured.evidence["listing"]["revision"] == listing.to_dict()["revision"]
    assert len({item.candidate_id for item in listing.candidates}) == 2


def test_file_change_after_listing_and_during_read():
    environment = MemoryEnvironment({"logs/a.txt": "passed"})
    resolver = InputResolver(environment, SCOPE)
    listing = resolver.list_folder("logs")
    environment.write_text("logs/a.txt", "failed")
    assert listing.candidates[0].candidate_id == resolver.list_folder("logs").candidates[0].candidate_id
    with pytest.raises(InputError, match="stale_input"):
        resolver.capture_selected(listing, listing.candidates[0].candidate_id)

    class ChangingEnvironment(MemoryEnvironment):
        def file_metadata(self, path):
            result = super().file_metadata(path)
            self.files[path] += "!"
            return result
    with pytest.raises(InputError, match="stale_input"):
        InputResolver(ChangingEnvironment({"a": "old"}), SCOPE).capture_file("a")


def test_host_metadata_detects_replacement_with_identical_content(tmp_path):
    path = tmp_path / "log"
    path.write_text("same", encoding="utf-8")
    resolver = InputResolver(HostInputEnvironment(tmp_path), SCOPE)
    before = resolver.list_folder(".")
    replacement = tmp_path / "next"
    replacement.write_text("same", encoding="utf-8")
    replacement.replace(path)
    with pytest.raises(InputError, match="stale_input"):
        resolver.capture_selected(before, before.candidates[0].candidate_id)


def test_host_symlink_cannot_escape_folder_or_role_grant(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    secret_folder = tmp_path / "secrets"
    secret_folder.mkdir()
    secret = secret_folder / "private.txt"
    secret.write_text("private", encoding="utf-8")
    alias = "logs/alias.txt"
    file_link = True
    try:
        (logs / "alias.txt").symlink_to(secret)
    except OSError as exc:
        if os.name != "nt":
            raise
        # Windows permits directory junctions without file-symlink privilege.
        junction = subprocess.run(["cmd", "/c", "mklink", "/J", str(logs / "alias"), str(secret_folder)], capture_output=True)
        assert junction.returncode == 0, junction.stderr
        alias, file_link = "logs/alias/private.txt", False
    environment = HostInputEnvironment(tmp_path)
    with pytest.raises(InputError, match="input_folder"):
        InputResolver(environment, SCOPE).capture_file(alias, folder="logs")
    if file_link:
        with pytest.raises(InputError, match="input_folder"):
            InputResolver(environment, SCOPE).list_folder("logs")
    granted = GrantedInputEnvironment(environment, [], ["secrets"])
    with pytest.raises(InputError, match="role denies"):
        InputResolver(granted, SCOPE).capture_file(alias)


@pytest.mark.parametrize("path", ["../secret", "/host/secret", "C:/secret", "logs\\secret", "logs/../secret", ""])
def test_reject_unscoped_paths(path):
    with pytest.raises(InputError, match="inaccessible_path"):
        InputResolver(MemoryEnvironment(), SCOPE).capture_file(path)


def test_missing_empty_wrong_boundary_and_candidate():
    resolver = InputResolver(MemoryEnvironment({"logs/a": "ok", "outside": "secret"}), SCOPE)
    with pytest.raises(InputError, match="missing_file"):
        resolver.capture_file("missing")
    with pytest.raises(InputError, match="empty_listing"):
        resolver.list_folder("empty")
    with pytest.raises(InputError, match="inaccessible_path"):
        resolver.capture_file("outside", folder="logs")
    with pytest.raises(InputError, match="unknown_candidate"):
        resolver.capture_selected(resolver.list_folder("logs"), "invented-path")
    with pytest.raises(InputError, match="oversized_input"):
        InputResolver(resolver.environment, SCOPE, max_bytes=1).capture_file("logs/a")
    with pytest.raises(InputError, match="oversized_input"):
        InputResolver(MemoryEnvironment({"a": "", "b": ""}), SCOPE, max_candidates=1).list_folder(".")


def test_listing_cannot_invent_host_paths_or_duplicate_ids():
    class BadListing(MemoryEnvironment):
        def list_files(self, directory):
            return ["outside"]
    with pytest.raises(InputError, match="inaccessible_path"):
        InputResolver(BadListing({"outside": "secret"}), SCOPE).list_folder("logs")
    class DuplicateListing(MemoryEnvironment):
        def list_files(self, directory):
            return ["logs/a", "logs/a"]
    with pytest.raises(InputError, match="malformed_value"):
        InputResolver(DuplicateListing(), SCOPE).list_folder("logs")


def test_granted_reads_and_filtered_candidates():
    environment = GrantedInputEnvironment(MemoryEnvironment({"logs/good": "pass", "logs/secret": "private", "outside": "x"}), ["logs"], ["logs/secret"])
    resolver = InputResolver(environment, SCOPE)
    assert [item.path for item in resolver.list_folder("logs").candidates] == ["logs/good"]
    for path in ["logs/secret", "outside"]:
        with pytest.raises(InputError, match="inaccessible_path"):
            resolver.capture_file(path)


def test_nested_structured_request_literal_and_types():
    environment = MemoryEnvironment({"report.json": '{"data": [{"a/b": {"~value": 3}}]}'})
    resolver = InputResolver(environment, SCOPE, request={"job": {"id": "current"}})
    capture = resolver.resolve(InputBinding("input_file", "report.json", pointer="/data/0/a~1b/~0value", expected_type="integer"))
    assert capture.value == 3
    assert json.loads(capture.evidence["content"])["data"][0]["a/b"]["~value"] == 3
    assert resolver.resolve(InputBinding("request", "job.id", expected_type="string")).value == "current"
    assert resolver.resolve(InputBinding("literal", None, expected_type="null")).value is None
    with pytest.raises(InputError, match="incorrect_type"):
        resolver.resolve(InputBinding("literal", True, expected_type="integer"))
    with pytest.raises(InputError, match="missing_field"):
        resolver.resolve(InputBinding("request", "missing"))
    for value in [float("nan"), {1: "not JSON"}]:
        with pytest.raises(InputError, match="malformed_value"):
            resolver.resolve(InputBinding("literal", value))
    for pointer in ["/bad~2", "/data/99", "/data/00"]:
        with pytest.raises(InputError):
            select_pointer({"data": [3]}, pointer)
    environment.write_text("broken.yaml", "x: [")
    with pytest.raises(InputError, match="malformed_value"):
        resolver.resolve(InputBinding("input_file", "broken.yaml", pointer="/x"))


def test_mutation_of_consumer_values_cannot_change_captured_evidence():
    source = {"nested": [1]}
    capture = InputResolver(MemoryEnvironment(), SCOPE).resolve(InputBinding("literal", source))
    capture.value["nested"].append(2)
    source["nested"].append(3)
    assert capture.evidence["value"] == {"nested": [1]}
    assert capture.evidence["source_value"] == {"nested": [1]}
    assert capture.evidence["value_sha256"] == digest({"nested": [1]})


def test_producer_metadata_changes_cannot_rewrite_a_captured_listing():
    class ProducerEnvironment(MemoryEnvironment):
        producer = {"task_id": "task", "run_id": "run", "item": None, "attempt": 1}
        def file_metadata(self, path):
            return replace(super().file_metadata(path), producer=self.producer)
    environment = ProducerEnvironment({"logs/a": "same content"})
    resolver = InputResolver(environment, SCOPE)
    listing = resolver.list_folder("logs")
    environment.producer["attempt"] = 2
    assert listing.candidates[0].producer["attempt"] == 1
    with pytest.raises(InputError, match="stale_input"):
        resolver.capture_selected(listing, listing.candidates[0].candidate_id)


def test_named_output_attempt_digest_and_loop_scope():
    producer = InputResolver(MemoryEnvironment(), replace(SCOPE, item="first"))
    outputs = producer.project_outputs({"report": OutputBinding("/data", "object")}, {"data": {"path": "logs/current"}}, phase="produce", attempt=2, revision="rev")
    entries = [{"ok": True, "named_outputs": outputs}]
    consumer = InputResolver(producer.environment, producer.scope, entries=entries)
    capture = consumer.resolve(InputBinding("output", "report", pointer="/path", expected_type="string"))
    assert capture.value == "logs/current"
    assert capture.evidence["producer"] == {"phase": "produce", "attempt": 2, "revision": "rev"}
    for scope in [replace(SCOPE, item="second"), replace(producer.scope, task_id="child"), replace(producer.scope, run_id="other")]:
        with pytest.raises(InputError, match="stale_output"):
            InputResolver(producer.environment, scope, entries=entries).named_output("report")
    entries.append({"ok": False, "named_outputs": {"report": None}})
    with pytest.raises(InputError, match="stale_output"):
        consumer.named_output("report")
    entries.pop()
    outputs["report"]["value"]["path"] = "tampered"
    with pytest.raises(InputError, match="stale_output"):
        consumer.named_output("report")


def test_unchanged_file_cannot_be_reused_in_next_loop_item():
    environment = MemoryEnvironment({"logs/a": "12 passed"})
    first = InputResolver(environment, replace(SCOPE, item="first")).capture_file("logs/a")
    second = InputResolver(environment, replace(SCOPE, item="second"), entries=[{"input_evidence": {"report": first.evidence}}])
    with pytest.raises(InputError, match="previous loop item"):
        second.capture_file("logs/a")
    environment.write_text("logs/a", "14 passed")
    assert second.capture_file("logs/a").value == "14 passed"


def test_legacy_scoped_reference_and_host_glob(tmp_path):
    environment = MemoryEnvironment({"reports/a.yaml": "x: [4, 5]"})
    assert load_reference("reports/a.yaml#/x/1", tmp_path, environment) == 5
    assert load_reference("reports/*.yaml", tmp_path, environment)[0]["content"] == "x: [4, 5]"
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/a.yaml").write_text("host: decoy", encoding="utf-8")
    assert load_reference("reports/a.yaml#/x/1", tmp_path, environment) == 5
    assert load_reference("reports/**/*.yaml", tmp_path) == [{"path": "reports/a.yaml", "content": "host: decoy"}]
    with pytest.raises(ManifestError, match="escapes"):
        load_reference(str(tmp_path.parent / "outside.txt"), tmp_path)


def manifest(tmp_path, phases):
    document = {"name": "input-test", "driver": {"kind": "python"}, "checkpoint_backend": None,
                "roles": {"worker": {"skill": "worker.py", "result_contract": {"schema": {"status": {"enum": ["done"]}}}}},
                "phases": phases}
    path = tmp_path / "test.workflow.md"
    path.write_text("---\n" + yaml.safe_dump(document) + "---\n", encoding="utf-8")
    return path


def role_phase(name="work", **extras):
    return {"name": name, "kind": "role", "role": "worker", "on_status": {"done": "stop"}, "on_invalid": {"action": "stop"}, **extras}


@pytest.mark.parametrize("binding", [
    {"request": "field", "literal": "conflict"}, {"input_folder": "logs"},
    {"input_file": 7}, {"literal": 1, "expected_type": ["integer"]},
    {"literal": 1, "typo": True}, {"output": "unknown"}, {"literal": 1, "pointer": "a"},
])
def test_manifest_rejects_ambiguous_or_malformed_bindings(tmp_path, binding):
    path = manifest(tmp_path, [role_phase(inputs={"data": binding})])
    with pytest.raises(ManifestError):
        parse_workflow(path)


def test_kernel_dynamic_file_named_result_and_request_in_virtual_environment(tmp_path):
    path = manifest(tmp_path, [
        role_phase("produce", on_status={"done": "consume"}, inputs={"job": {"request": "job", "expected_type": "string"}}, outputs={"report_path": {"pointer": "/path", "expected_type": "string"}}),
        role_phase("consume", inputs={"report": {"input_file": "${outputs.report_path}", "input_folder": "logs", "pointer": "/facts/0", "expected_type": "integer"}}),
    ])
    (tmp_path / "worker.py").write_text(
        "def run(context):\n"
        "    if context.phase == 'produce':\n"
        "        assert context.inputs['job'] == 'current'\n"
        "        context.runtime.environment.write_text('logs/current.json', '{\"facts\": [42]}')\n"
        "        return {'status': 'done', 'path': 'logs/current.json'}\n"
        "    assert context.inputs['report'] == 42\n"
        "    return {'status': 'done'}\n", encoding="utf-8",
    )
    environment = MemoryEnvironment()
    kernel = Kernel(path, tmp_path / "host-decoy", "task", base_dir=tmp_path,
                    kernel_data_root=tmp_path / "journal", runtime=WorkflowRuntime(environment, threading.Event(), {}, True), request_data={"job": "current"})
    assert kernel.run()["terminal_status"] == "done"
    entries = [entry for entry in kernel.journal.read_all() if entry["kind"] == "role"]
    assert entries[0]["named_outputs"]["report_path"]["producer"]["attempt"] == 1
    assert entries[1]["input_evidence"]["report"]["value"] == 42
    assert not (tmp_path / "host-decoy/logs/current.json").exists()


def test_kernel_bad_output_type_or_input_uses_invalid_route(tmp_path):
    path = manifest(tmp_path, [role_phase(outputs={"count": {"pointer": "/count", "expected_type": "integer"}})])
    (tmp_path / "worker.py").write_text("def run(context):\n    return {'status': 'done', 'count': True}\n", encoding="utf-8")
    kernel = Kernel(path, tmp_path / "workspace", "task", base_dir=tmp_path, kernel_data_root=tmp_path / "journal")
    kernel.run()
    entry = next(entry for entry in kernel.journal.read_all() if entry["kind"] == "role")
    assert not entry["ok"]
    assert entry["named_outputs"] == {"count": None}


def test_old_manifest_has_no_bindings_and_still_runs(tmp_path):
    path = manifest(tmp_path, [role_phase()])
    (tmp_path / "worker.py").write_text("def run(context):\n    assert context.inputs == {}\n    return {'status': 'done'}\n", encoding="utf-8")
    kernel = Kernel(path, tmp_path / "workspace", "task", base_dir=tmp_path, kernel_data_root=tmp_path / "journal")
    assert kernel.run()["terminal_status"] == "done"


def test_invalid_input_never_dispatches_and_keeps_partial_capture(tmp_path):
    path = manifest(tmp_path, [role_phase(inputs={"a_good": {"literal": "captured"}, "z_missing": {"input_file": "missing"}})])
    (tmp_path / "worker.py").write_text("def run(context):\n    raise AssertionError('must not dispatch')\n", encoding="utf-8")
    kernel = Kernel(path, tmp_path / "workspace", "task", base_dir=tmp_path, kernel_data_root=tmp_path / "journal")
    kernel.run()
    entry = next(entry for entry in kernel.journal.read_all() if entry["kind"] == "role")
    assert entry["verdict"] == "invalid_input"
    assert entry["input_evidence"]["a_good"]["value"] == "captured"
    assert entry["input_evidence"]["z_missing"]["code"] == "missing_file"


def test_cancelled_capture_cannot_dispatch_or_publish_outputs(tmp_path):
    stop = threading.Event()
    class CancellingEnvironment(MemoryEnvironment):
        def read_text(self, path):
            stop.set()
            return super().read_text(path)
    path = manifest(tmp_path, [role_phase(inputs={"data": {"input_file": "log"}}, outputs={"answer": {}})])
    (tmp_path / "worker.py").write_text("def run(context):\n    raise AssertionError('must not dispatch')\n", encoding="utf-8")
    kernel = Kernel(path, tmp_path / "workspace", "task", base_dir=tmp_path, kernel_data_root=tmp_path / "journal",
                    runtime=WorkflowRuntime(CancellingEnvironment({"log": "ok"}), stop, {}))
    assert kernel.run()["terminal_status"] == "terminated"
    assert not any(entry.get("named_outputs") for entry in kernel.journal.read_all())


def test_foreach_inputs_use_current_item_and_child_request_namespace(tmp_path):
    child = manifest(tmp_path, [role_phase(inputs={"job": {"request": "job", "expected_type": "string"}})])
    child.rename(tmp_path / "child.workflow.md")
    (tmp_path / "worker.py").write_text(
        "def run(context):\n"
        "    context.runtime.environment.write_text('outbox/seen', context.inputs['job'])\n"
        "    return {'status': 'done'}\n", encoding="utf-8")
    parent = manifest(tmp_path, [{"name": "fan", "kind": "workflow", "workflow": "child",
        "foreach": {"from": "items.json#/items", "item": "job", "stable_id": "id"},
        "inputs": {"job": {"input_file": "logs/${job.id}.txt"}}, "task": {"id": "task.${job.id}", "input": {}},
        "result": {"statuses": ["done"], "aggregate": "all_children"},
        "on_status": {"done": "stop"}, "on_invalid": {"action": "stop"}}])
    children = []
    def child_factory(scope):
        environment = MemoryEnvironment()
        children.append(environment)
        return WorkflowRuntime(environment, threading.Event(), {}, True)
    environment = MemoryEnvironment({"items.json": '{"items": [{"id": "b"}, {"id": "a"}]}', "logs/a.txt": "a", "logs/b.txt": "b"})
    runtime = WorkflowRuntime(environment, threading.Event(), {}, True, child_factory=child_factory)
    kernel = Kernel(parent, tmp_path / "host", "task", base_dir=tmp_path, kernel_data_root=tmp_path / "journal", runtime=runtime,
                    workflow_resolver=lambda name, path, base: tmp_path / "child.workflow.md")
    assert kernel.run()["terminal_status"] == "done"
    assert [child.files["outbox/seen"] for child in children] == ["a", "b"]
    assert "outbox/seen" not in environment.files
    entry = next(entry for entry in kernel.journal.read_all() if entry["kind"] == "workflow")
    captures = [receipt["input_evidence"]["job"] for receipt in entry["result"]["children"]]
    assert [capture["scope"]["item"] for capture in captures] == ["/fan/a", "/fan/b"]


def test_foreach_rejects_reuse_of_previous_items_file(tmp_path):
    child = manifest(tmp_path, [role_phase()])
    child.rename(tmp_path / "child.workflow.md")
    (tmp_path / "worker.py").write_text("def run(context):\n    return {'status': 'done'}\n", encoding="utf-8")
    parent = manifest(tmp_path, [{"name": "fan", "kind": "workflow", "workflow": "child",
        "foreach": {"from": "items.json#/items", "stable_id": "id"},
        "inputs": {"job": {"input_file": "logs/reused.txt"}}, "task": {"id": "task.${item.id}", "input": {}},
        "result": {"statuses": ["done"], "aggregate": "all_children"},
        "on_status": {"done": "stop"}, "on_invalid": {"action": "stop"}}])
    environment = MemoryEnvironment({"items.json": '{"items": [{"id": "a"}, {"id": "b"}]}', "logs/reused.txt": "old item"})
    kernel = Kernel(parent, tmp_path / "host", "task", base_dir=tmp_path, kernel_data_root=tmp_path / "journal",
                    runtime=WorkflowRuntime(environment, threading.Event(), {}, True),
                    workflow_resolver=lambda name, path, base: tmp_path / "child.workflow.md")
    kernel.run()
    entry = next(entry for entry in kernel.journal.read_all() if entry["kind"] == "workflow")
    assert not entry["ok"]
    assert "previous loop item" in entry["errors"][0]


def test_loop_output_cannot_reuse_previous_producer_after_restart(tmp_path):
    path = manifest(tmp_path, [role_phase(outputs={"answer": {"pointer": "/answer"}})])
    (tmp_path / "worker.py").write_text("def run(context):\n    return {'status': 'done', 'answer': 7}\n", encoding="utf-8")
    options = dict(manifest_path=path, workspace=tmp_path / "host", task_id="task", base_dir=tmp_path, kernel_data_root=tmp_path / "journal", run_id="same")
    first = Kernel(**options)
    first.current_item = "item-a"
    assert first._execute_role(first.manifest.phases[0])["valid"]
    resumed = Kernel(**options)
    resumed.current_item = "item-b"
    with pytest.raises(InputError, match="stale_output"):
        resumed._input_resolver().named_output("answer")


def test_latest_failed_producer_blocks_old_output(tmp_path):
    path = manifest(tmp_path, [role_phase(outputs={"answer": {"pointer": "/answer"}})])
    (tmp_path / "worker.py").write_text(
        "def run(context):\n"
        "    if context.attempt > 1:\n"
        "        raise ValueError('actual producer error')\n"
        "    return {'status': 'done', 'answer': 7}\n", encoding="utf-8")
    kernel = Kernel(path, tmp_path / "host", "task", base_dir=tmp_path, kernel_data_root=tmp_path / "journal")
    phase = kernel.manifest.phases[0]
    assert kernel._execute_role(phase)["valid"]
    assert kernel._input_resolver().named_output("answer").value == 7
    assert not kernel._execute_role(phase)["valid"]
    with pytest.raises(InputError, match="stale_output"):
        kernel._input_resolver().named_output("answer")
