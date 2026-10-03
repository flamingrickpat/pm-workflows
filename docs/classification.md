# Classification and INVALID recovery

The optional `kind: classify` node uses the normal LLM to select and classify captured evidence.
Existing manifests require no new phases or connections.
The parent guide is the [package README](../README.md).
The consuming PM manual supplies detailed [contracts and verification](https://github.com/flamingrickpat/pm_next_v2/tree/master/docs/subsystems/workflow-inputs).

## Manifest

```yaml
roles:
  inspect_tests:
    skill: inspect-tests.md
    readable_paths: ["logs/**"]
phases:
  - name: tests
    kind: classify
    backend: llm
    connection: Default
    input_folder: logs
    data_selection: Select the latest foundation test log by its name and metadata.
    instruction: Classify the completed test summary.
    result_contract:
      type: classes
      enum:
        passed: At least one test ran and every test passed.
        failed: The completed run reports a failure.
    on_class: {passed: stop, failed: stop, INVALID: recover_tests}
  - name: recover_tests
    kind: classify
    fallback_for: tests
    role: inspect_tests
    on_class: {passed: stop, failed: stop, INVALID: stop}
```

Every class and INVALID requires a destination.
The handler inherits the question and classes.
Its successful routes match the primary routes.
Its unresolved route cannot reach either node again.
The handler must use a non-Python coding role without MCP or writable-path grants.

Known files skip selection. Exact-file plus folder enforces the folder boundary.
One A03 binding under `inputs` can also supply request fields, literal values, or named outputs.
Pointers and type assertions apply to the selected content.
Contradictory source or route declarations fail loading.

## Runtime

Supply resolved connections through `WorkflowRuntime.classification_models`.
The mapping key matches the manifest's `connection` name.
Its `AgentModel.context_window` must describe the selected chat service.
Recovery uses the separate `runtime.agent_model` and the pm-coder driver.
No connection activates implicitly, and Laya remains disabled.

Direct prompts use UTF-8 bytes plus 256 as a conservative token bound.
The bound and reserved output must fit the supplied serving context.
The node never silently clips evidence. Oversized input takes INVALID.
`classification_limits` controls prompt, output, capture, candidate, recovery, and deadline limits.
`ClassificationConfig` declares their defaults and units.

Recovery checks source versions and creates an isolated BashMachine with read-only eligible snapshots.
New virtual scratch files cannot change the original environment.
The agent receives no application tools or MCP services.
One handler attempt can answer each invalid source receipt.
The driver enforces the session deadline and cancellation cleanup.
The installed pm-coder retains its existing compaction mechanism.
Its `live_test=True` diagnostic mode deliberately bypasses that recovery loop.

## Results and inspection

Both model paths return exactly `status`, `reason`, `evidence`, `candidate_id`, and `input_sha256`.
A class requires exact source quotes and the captured value digest.
Exact and structured sources use null candidate IDs.
INVALID requires a reason and null identifiers.
The value digest hashes canonical JSON and can differ from the raw file digest.

The journal and `StepResult.data.receipt` retain complete requests, responses, captures, scope, attempts, and concrete errors.
`replay_classification` checks those retained values without another read or model call.
Static and dry-run graphs include the INVALID handler.
Classification neither restores a worktree nor removes external effects.
Checkpoint and ordinary role retry policies remain unchanged.

## Validation

Run the package suite:

```powershell
.venv/Scripts/python.exe -m pytest -q
```

Run the consuming PM demonstration against actual local chat:

```powershell
uv run --locked python scripts/alpha_classification_gate.py --out .runtime/alpha/a04-repeat
```

The gate uses original captured logs and Spamuel history.
It verifies host and virtual selection, exact digests, INVALID conditions, scoped agent recovery, and actual server-context overflow.
Synthetic boundary cases use real inference and independently defined expected results.
The retained response fixture only checks protocol parsing.
