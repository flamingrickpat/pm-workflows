"""Check execution: how a verdict is read out of a check script."""
import json
from pathlib import Path

import pytest

from pm_workflows.gates import run_gate


def test_gate_pass(tmp_path):
    script = tmp_path / "check.py"
    script.write_text("""import json, sys
print(json.dumps({"ok": True, "errors": []}))
sys.exit(0)
""")
    result = run_gate(script, tmp_path)
    assert result.ok is True
    assert result.errors == []


def test_gate_fail(tmp_path):
    script = tmp_path / "check.py"
    script.write_text("""import json, sys
print(json.dumps({"ok": False, "errors": ["file missing", "too small"]}))
sys.exit(1)
""")
    result = run_gate(script, tmp_path)
    assert result.ok is False
    assert "file missing" in result.errors
    assert "too small" in result.errors


def test_gate_non_json_fail(tmp_path):
    script = tmp_path / "check.py"
    script.write_text("""import sys
print("something went wrong")
sys.exit(1)
""")
    result = run_gate(script, tmp_path)
    assert result.ok is False
    assert len(result.errors) > 0


def test_gate_not_found(tmp_path):
    result = run_gate(tmp_path / "nonexistent.py", tmp_path)
    assert result.ok is False
    assert "not found" in result.errors[0]


def test_gate_reads_a_verdict_printed_after_progress_output(tmp_path):
    """Checks are allowed to be chatty; the verdict is the last object."""
    script = tmp_path / "check.py"
    script.write_text("""import json, sys
print('scanning 120 files...')
print(json.dumps({'ok': False, 'errors': ['product.cs is out of scope']}))
sys.exit(1)
""", encoding="utf-8")
    result = run_gate(script, tmp_path)
    assert result.ok is False
    assert result.errors == ["product.cs is out of scope"]


def test_gate_receives_its_arguments(tmp_path):
    script = tmp_path / "check.py"
    script.write_text("""import json, sys
print(json.dumps({'ok': sys.argv[2:] == ['a', 'b'], 'errors': sys.argv[2:]}))
""", encoding="utf-8")
    assert run_gate(script, tmp_path, args=["a", "b"]).ok is True


def test_a_failing_check_always_reports_a_reason(tmp_path):
    """Silence from a failing check would give the retried role nothing to act
    on, so the runtime substitutes something rather than passing an empty list."""
    script = tmp_path / "check.py"
    script.write_text("import sys; sys.exit(3)\n", encoding="utf-8")
    result = run_gate(script, tmp_path)
    assert result.ok is False
    assert result.errors
