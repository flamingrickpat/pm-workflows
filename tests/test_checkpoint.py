"""Test the git checkpoint backend."""
import subprocess
from pathlib import Path

import pytest

from pm_workflows.checkpoint import GitCheckpoint


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def test_init_if_needed(tmp_path):
    cp = GitCheckpoint(tmp_path)
    assert not (tmp_path / ".git").exists()

    cp.init_if_needed()
    assert (tmp_path / ".git").exists()


def test_snapshot_and_current_rev(tmp_path):
    cp = GitCheckpoint(tmp_path)
    cp.init_if_needed()

    # Empty repo — no commits yet, rev should be empty or error
    rev0 = cp.current_rev()

    # Write a file and snapshot
    (tmp_path / "file.txt").write_text("hello")
    rev1 = cp.snapshot("test1")
    assert rev1 != ""
    assert cp.current_rev() == rev1

    # Write another file
    (tmp_path / "file2.txt").write_text("world")
    rev2 = cp.snapshot("test2")
    assert rev2 != rev1


def test_restore(tmp_path):
    cp = GitCheckpoint(tmp_path)
    cp.init_if_needed()

    (tmp_path / "file.txt").write_text("v1")
    rev1 = cp.snapshot("v1")

    (tmp_path / "file.txt").write_text("v2")
    (tmp_path / "extra.txt").write_text("extra")
    cp.snapshot("v2")

    # Restore to v1
    cp.restore(rev1)

    assert (tmp_path / "file.txt").read_text() == "v1"
    assert not (tmp_path / "extra.txt").exists()


def test_restore_discards_uncommitted(tmp_path):
    cp = GitCheckpoint(tmp_path)
    cp.init_if_needed()

    (tmp_path / "file.txt").write_text("committed")
    rev = cp.snapshot("checkpoint")

    # Make uncommitted changes
    (tmp_path / "file.txt").write_text("uncommitted")
    (tmp_path / "new.txt").write_text("new")

    cp.restore(rev)

    assert (tmp_path / "file.txt").read_text() == "committed"
    assert not (tmp_path / "new.txt").exists()


def test_empty_snapshot(tmp_path):
    cp = GitCheckpoint(tmp_path)
    cp.init_if_needed()

    (tmp_path / "file.txt").write_text("data")
    rev1 = cp.snapshot("first")

    # Snapshot with no changes
    rev2 = cp.snapshot("noop")
    assert rev1 == rev2  # No new commit when nothing changed


def test_restore_does_not_delete_build_output(tmp_path):
    """`git clean -fd` spares ignored files, but plenty of repositories never
    ignore bin/ and obj/ — and there a revert would silently cost a full
    rebuild of a large native tree on every retry."""
    cp = GitCheckpoint(tmp_path)
    cp.init_if_needed()
    (tmp_path / "src.cs").write_text("code\n", encoding="utf-8")
    rev = cp.snapshot("base")

    # Untracked and *not* gitignored, exactly as in the real checkout.
    for directory in ("bin", "obj", "node_modules", ".Assemblies"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "artifact.dll").write_text("expensive\n", encoding="utf-8")
    (tmp_path / "agent-junk.txt").write_text("scratch\n", encoding="utf-8")
    (tmp_path / "src.cs").write_text("vandalised\n", encoding="utf-8")

    cp.restore(rev)

    assert (tmp_path / "src.cs").read_text(encoding="utf-8") == "code\n"
    assert not (tmp_path / "agent-junk.txt").exists(), "agent debris must still go"
    for directory in ("bin", "obj", "node_modules", ".Assemblies"):
        assert (tmp_path / directory / "artifact.dll").is_file(), (
            f"{directory} was deleted; every retry would force a rebuild"
        )
