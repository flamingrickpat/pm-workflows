#!/usr/bin/env python3
"""Run a workflow against a target repository.

The workflow file is the contract: it names the roles, the skills, the checks,
and where every declared outcome routes. This script only wires up the target
repo, the task input, and the coding agent, then hands over to the kernel.

Examples
--------
Autonomous run of a single task:

    pm-workflows --base C:\\source\\coding-workflow\\agents-deploy ^
        --target C:\\source\\target --workflow auto_task.workflow.md ^
        --task-id dsk-1 --input C:\\source\\coding-workflow\\dsk-1\\task.md ^
        --coding-agent claude --model sonnet --effort medium

Interactive run (blocks on stdin at every user gate):

    pm-workflows --base ...\\agents-deploy --target C:\\source\\target ^
        --workflow interactive_task.workflow.md --task-id dsk-2 ^
        --input ...\\dsk-2\\task.md --coding-agent codex

A killed or usage-limited run is continued by repeating the command. You may
change --coding-agent and --model: the kernel reads the same journal, resets the
repository to the last accepted commit, and re-dispatches the phase that was in
flight in a fresh session.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from pm_workflows import Kernel  # noqa: E402
from pm_workflows.drivers import SUPPORTED_DRIVERS  # noqa: E402
from pm_workflows.manifest import ManifestError  # noqa: E402

# Real CLI values. Friendly aliases exist only for the names people type; an
# unknown model is passed through to the agent so its own error surfaces.
CLAUDE_MODELS = ("opus", "sonnet", "haiku", "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
CLAUDE_MODEL_ALIASES = {
    "sonnet5": "sonnet",
    "sonnet-5": "sonnet",
    "opus5": "opus",
    "opus-5": "opus",
    "haiku45": "haiku",
    "haiku-4-5": "haiku",
}
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
BASE_COMMIT = re.compile(r"^-?\s*base_commit:\s*([0-9a-f]{7,40})", re.MULTILINE | re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base", required=True, type=Path,
        help="Central instruction base: the folder holding workflows/, "
             "skills/ and scripts/. Stays outside the target repo.",
    )
    parser.add_argument(
        "--target", required=True, type=Path,
        help="The repository to work in. Must exist and be a git repo on the "
             "branch you want the work on.",
    )
    parser.add_argument(
        "--workflow", required=True,
        help="Workflow file name inside <base>/workflows/, or an absolute path.",
    )
    parser.add_argument("--task-id", required=True, help="Task identifier, e.g. dsk-8119.")
    parser.add_argument(
        "--input", type=Path,
        help="Markdown file describing the task. Copied to the task folder as "
             "request.md on the first run.",
    )
    parser.add_argument(
        "--coding-agent", "--agent", dest="coding_agent",
        default=None, choices=SUPPORTED_DRIVERS,
        help="Agent selected for this run. Change it when resuming to fail over "
             "without changing the workflow or journal.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model passed to the selected coding agent. Claude aliases include: "
             f"{', '.join(CLAUDE_MODELS)}.",
    )
    parser.add_argument(
        "--effort", default=None, choices=CLAUDE_EFFORTS,
        help="Reasoning effort for the coding agent.",
    )
    parser.add_argument(
        "--kernel-data", type=Path, default=None,
        help="Where the journal and traces live (default ~/.pm/pm-workflows). Never "
             "inside the target repo: agents must not see how they are judged.",
    )
    parser.add_argument(
        "--max-agent-requests", type=int, default=None,
        help="Per-role request budget for agents that support it (default 80). "
             "This is a runtime limit, not part of the workflow contract.",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible endpoint used by agents that support it.",
    )
    parser.add_argument(
        "--api-key-env", default=None,
        help="Environment variable holding the minimal-agent API key "
             "(default OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--expect-base-commit", default=None,
        help="Refuse to start unless the target's HEAD is this commit. Defaults "
             "to the base_commit recorded in the input file, when it has one.",
    )
    parser.add_argument(
        "--ignore-base-commit", action="store_true",
        help="Run even though HEAD is not the expected base commit.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Start from the top instead of continuing an existing journal.",
    )
    return parser


def git(repo: Path, *arguments: str) -> tuple[int, str]:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )
    return result.returncode, (result.stdout or "").strip()


def resolve_model(agent: str, model: str | None) -> str | None:
    if not model:
        return None
    if agent != "claude":
        return model
    resolved = CLAUDE_MODEL_ALIASES.get(model.lower(), model)
    if resolved not in CLAUDE_MODELS:
        print(
            f"note: '{model}' is not one of the known claude models "
            f"({', '.join(CLAUDE_MODELS)}); passing it through as given.",
            file=sys.stderr,
        )
    return resolved


def check_target(
    target: Path, expected: str | None, ignore: bool, continuing: bool
) -> None:
    if not target.is_dir():
        raise SystemExit(f"--target does not exist: {target}")
    code, _ = git(target, "rev-parse", "--git-dir")
    if code != 0:
        raise SystemExit(f"--target is not a git repository: {target}")

    _, branch = git(target, "rev-parse", "--abbrev-ref", "HEAD")
    _, head = git(target, "rev-parse", "HEAD")
    print(f"target : {target}")
    print(f"branch : {branch}  HEAD {head[:12]}")

    if branch in {"main", "master"}:
        print(
            "\nrefusing to run on the default branch. The workflow commits and "
            "hard-resets in this repository; check out a task branch first.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if not expected:
        return

    if continuing:
        # A task in progress has commits of its own, so HEAD is *supposed* to
        # have moved past the base. What still has to hold is the lineage: the
        # work must sit on top of the commit the task was scoped against.
        code, _ = git(target, "merge-base", "--is-ancestor", expected, "HEAD")
        if code == 0:
            print(f"resuming : work descends from base commit {expected}")
            return
        message = (
            f"HEAD {head[:12]} does not descend from the task's base commit "
            f"{expected}, so this is not a continuation of the same work. "
            f"Check out the right branch, or start over with --fresh."
        )
    elif head.startswith(expected.lower()):
        return
    else:
        message = (
            f"HEAD is {head[:12]} but the task's base commit is {expected}. "
            f"Check out the base commit before starting, or pass "
            f"--ignore-base-commit."
        )

    if ignore:
        print(f"warning: {message}", file=sys.stderr)
    else:
        raise SystemExit(f"\n{message}")


def read_task_input(path: Path | None) -> tuple[str, str | None]:
    if path is None:
        return "", None
    if not path.is_file():
        raise SystemExit(f"--input does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    match = BASE_COMMIT.search(text)
    return text, (match.group(1) if match else None)


def find_run_id(data_root: Path, task_id: str, fresh: bool) -> str | None:
    """Find the newest resumable run for a task, if one exists."""
    if fresh:
        return None
    legacy = data_root / task_id / "journal.jsonl"
    if legacy.is_file() and legacy.stat().st_size:
        return task_id
    candidates = [
        path.name
        for path in data_root.glob(f"*_{task_id}")
        if (path / "journal.jsonl").is_file()
        and (path / "journal.jsonl").stat().st_size
    ]
    return max(candidates) if candidates else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_agent_requests is not None and args.max_agent_requests < 1:
        raise SystemExit("--max-agent-requests must be greater than zero")

    base = args.base.resolve()
    if not (base / "workflows").is_dir() or not (base / "skills").is_dir():
        raise SystemExit(
            f"--base must be the agents/ folder holding workflows/ and skills/: {base}"
        )

    manifest = Path(args.workflow)
    if not manifest.is_absolute():
        manifest = base / "workflows" / args.workflow
    if not manifest.is_file():
        raise SystemExit(f"workflow not found: {manifest}")

    task_text, recorded_base = read_task_input(args.input)
    expected = args.expect_base_commit or recorded_base

    # A journal for this task means the work is already under way, which changes
    # what "the right HEAD" means.
    data_root = Path(args.kernel_data or (Path.home() / ".pm" / "pm-workflows"))
    existing_run_id = find_run_id(data_root, args.task_id, args.fresh)
    continuing = existing_run_id is not None

    target = args.target.resolve()
    check_target(target, expected, args.ignore_base_commit, continuing)

    agent = args.coding_agent or ""
    model = resolve_model(agent, args.model)

    try:
        kernel = Kernel(
            manifest_path=manifest,
            workspace=target,
            task_id=args.task_id,
            task_text=task_text,
            base_dir=base,
            coding_agent=args.coding_agent,
            model=model,
            effort=args.effort,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            kernel_data_root=args.kernel_data,
            run_id=existing_run_id,
            max_agent_requests=args.max_agent_requests,
            resume=not args.fresh,
        )
    except ManifestError as error:
        print(f"\nthe workflow contract is not valid:\n  {error}", file=sys.stderr)
        return 2

    result = kernel.run()
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
