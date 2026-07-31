"""Test the append-only journal."""
import json
from pathlib import Path

import pytest

from pm_workflows.journal import Journal
from pm_workflows.protocol import JournalEntry


def test_journal_append_and_read(tmp_path):
    j = Journal(tmp_path / "journal.jsonl")

    e1 = JournalEntry(run_id="r1", phase="implement", kind="role", ok=True, attempt=1)
    e2 = JournalEntry(run_id="r1", phase="verify", kind="gate", ok=False, errors=["no file"])

    j.append(e1)
    j.append(e2)

    entries = j.read_all()
    assert len(entries) == 2
    assert entries[0]["phase"] == "implement"
    assert entries[0]["ok"] is True
    assert entries[1]["phase"] == "verify"
    assert entries[1]["ok"] is False


def test_journal_recent(tmp_path):
    j = Journal(tmp_path / "journal.jsonl")
    for i in range(15):
        j.append(JournalEntry(run_id="r1", phase=f"p{i}", kind="role", ok=True, attempt=i))

    recent = j.recent(5)
    assert len(recent) == 5
    assert recent[-1]["phase"] == "p14"
    assert recent[0]["phase"] == "p10"


def test_journal_attempts_for_phase(tmp_path):
    """An attempt is a dispatch or a check run. The bookkeeping entries around
    it -- the revert, the route, the checkpoint -- are not attempts, or the
    attempt caps would count the same try several times."""
    j = Journal(tmp_path / "journal.jsonl")
    j.append(JournalEntry(run_id="r1", phase="implement", kind="role", ok=False, attempt=1))
    j.append(JournalEntry(run_id="r1", phase="implement", kind="revert", ok=True))
    j.append(JournalEntry(run_id="r1", phase="implement", kind="route", ok=True))
    j.append(JournalEntry(run_id="r1", phase="implement", kind="role", ok=True, attempt=2))
    j.append(JournalEntry(run_id="r1", phase="verify", kind="gate", ok=True))

    assert j.attempts_for_phase("implement") == 2
    assert j.attempts_for_phase("verify") == 1


def test_journal_last_errors(tmp_path):
    j = Journal(tmp_path / "journal.jsonl")
    j.append(JournalEntry(run_id="r1", phase="verify", kind="gate", ok=False, errors=["no file", "too small"]))
    j.append(JournalEntry(run_id="r1", phase="verify", kind="gate", ok=True))

    errors = j.last_errors_for_phase("verify")
    assert "no file" in errors
    assert "too small" in errors


def test_journal_creates_parent_dirs(tmp_path):
    j = Journal(tmp_path / "deep" / "nested" / "journal.jsonl")
    assert j.path.parent.exists()
    assert j.path.exists()
