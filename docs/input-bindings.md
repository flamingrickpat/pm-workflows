# Scoped workflow inputs

## Summary

Phase bindings capture files, folder metadata, named outputs, request fields, and literal JSON data.
The kernel reads through the supplied working environment.
Each receipt retains the exact value, digest, source version, and invocation identity.
Folder bindings supply candidates for a later selection step.
They do not run classification.

Parent: [package README](../README.md).
The consuming PM manual generates API pages from its pinned package.
Its [input guide](https://github.com/flamingrickpat/pm_next_v2/blob/master/docs/subsystems/workflow-inputs/index.md) explains application operation and evidence.

## Manifest interface

Bindings belong to `inputs` on a `role` or `workflow` phase.
The mapping key names the captured value.
Names use Python identifier characters.
Each binding declares one source mode.
An exact file can also declare `input_folder` as a boundary.

```yaml
inputs:
  report:
    input_file: "${request.report_path}"
    input_folder: reports
    pointer: /tests/0/result
    expected_type: object
  candidates:
    input_folder: logs
    data_selection: Select the most recent unit-test log.
  earlier_result:
    output: tested
    pointer: /summary
    expected_type: string
  task:
    request: job.id
    expected_type: string
  policy:
    literal: {allow_incomplete: false}
    expected_type: object
```

`pointer` uses RFC 6901 syntax. Empty text selects the whole value.
`expected_type` accepts `string`, `object`, `array`, `integer`, `number`, `boolean`, or `null`.
Booleans do not satisfy integer or number assertions.
Unknown keys, ambiguous sources, unknown named outputs, and duplicate output names fail manifest loading.
Unresolved path variables fail capture before dispatch.
Paths use relative POSIX spelling. Traversal, drive letters, and absolute runtime paths fail.

An exact file skips selection. Its optional folder bounds the file and its resolved host target.
A folder-only binding requires `data_selection` and returns the listing as structured data.
The caller selects an existing candidate ID, then calls `InputResolver.capture_selected()`.
The caller owns inference, uncertainty, and classification routes.

## Producers and consumers

```yaml
outputs:
  tested:
    pointer: /test_result
    expected_type: object
  report_path:
    pointer: /path
    expected_type: string
```

Outputs project values from a validated role or child-workflow result.
Each record binds the task, run, item, phase, attempt, revision, pointer, value, and digest.
The recorded revision describes the producer candidate. It does not imply a later Git checkpoint.
A failed latest attempt leaves null declarations. It blocks reuse of an older accepted value.
A missing projection or incorrect type invalidates the whole producer attempt.
Outputs from another item or invocation cannot bind.
Each child kernel has a separate output namespace.
Pass required values through child `task.input` or phase `inputs`.

File paths can use `${request.field}` and `${outputs.name}`.
Child phase paths can also use their existing foreach variables.
Dynamic path evidence retains the original binding and the exact output dependency.
Python roles receive `RoleContext.inputs` and `RoleContext.input_evidence`.
Agent roles receive those captured receipts in their prompt.
Child phase inputs merge into the child's explicit request. Duplicate request keys fail.
`Kernel.request_data` supplies request fields. PM's runner supplies its payload automatically.

## Environment metadata

`WorkingEnvironment.file_metadata(path)` returns `FileMetadata`.
The default implementation reads text and uses its UTF-8 digest as the revision.
It leaves modification time and producer metadata unknown.
Older duck-typed environments receive the same digest fallback.
Native environments can supply an opaque revision, actual time, and producer metadata.

Each candidate includes `candidate_id`, `path`, `size_bytes`, `revision`, `content_sha256`, `modified_time`, and `producer`.
The ID hashes the relative path. It stays stable across file versions within one environment.
A listing revision hashes its ordered candidate metadata.
The listing also records the selection instruction and its digest.
Host metadata uses stat identity, size, nanosecond modification/change values, and a content digest.
Host reads preserve exact UTF-8 newlines.
Virtual environments never inherit host timestamps.

The resolver checks metadata before and after a read.
It also compares the captured bytes with the metadata digest.
Selection checks the complete listing again before capture.
A changed file or candidate set rejects the attempt without retry.
Producer fields, when present, must match the current task, run, and item.
Absent producer metadata remains an explicit limitation.
An unchanged file captured for another loop or foreach item cannot become current evidence.
Digest-only environments cannot detect replacement with identical bytes.

## State, failures, and compatibility

The kernel stores capture receipts and named outputs in its append-only journal.
The active recovery cursor controls which records consumers can use.
Captured values support replay without another environment read.
Input resolution reads files and writes no working-target state.
The normal kernel result, route, and checkpoint boundaries remain in control.
Cancellation propagates between captures and before dispatch. It does not undo completed external effects.

`InputError.code` distinguishes missing files, empty listings, inaccessible paths, stale inputs, missing fields, incorrect types, and oversized data.
Additional codes cover missing/stale outputs, unknown candidates, and malformed values.
Invalid role bindings take the declared `on_invalid` route before model dispatch.
Partial successful captures remain in the failed journal entry.
Classification extensions can translate these causes into INVALID in A04.
Defaults permit 256 candidates and 1,048,576 bytes per captured value.
Native or fallback metadata can read content before the size check. These limits do not bound filesystem I/O.
Model consumers need their own token budget. No text is silently clipped.

Old manifests have empty bindings and retain their previous dispatch behavior.
Legacy `load_reference()` now reads through the supplied environment.
Without a runtime, its read-only host adapter bounds references to the workspace.
Legacy host globs retain recursive patterns. Virtual globs require exact folder components and direct-child file patterns.
Child runtime factories and existing tool/MCP grants remain in use.
Role capture obeys readable and denied paths, including resolved host targets.
Trusted Python roles and shell tools retain their existing trust model.
pm-coder request handling and automatic compaction remain unchanged.

## Validation

```powershell
.venv/Scripts/python.exe -m pytest tests/test_inputs.py -q
.venv/Scripts/python.exe -m pytest -q
```

The tests use actual host files, a memory filesystem, real kernel dispatch, and existing live agent checks.
They cover listing changes, replacement, path grants, nested pointers, type errors, dynamic paths, child isolation, and restart reuse.
Windows junctions exercise resolved folder and role boundaries when file-symlink privileges are unavailable.
PM's `scripts/alpha_input_gate.py` adds actual application logs, the complete Spamuel Ghost projection, and a real agent report.
