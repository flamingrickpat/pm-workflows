"""Append-only JSONL journal. The kernel is the only writer.

The journal is the source of truth for what happened: every dispatch, every
check, every revert, every accepted commit. It lives outside the target
repository so no agent can read how it is being judged or rewrite its own
history.
"""
from __future__ import annotations

import json
from pathlib import Path

from .protocol import JournalEntry

# Kinds that count as an execution of a phase (used for attempt limits).
EXECUTION_KINDS = frozenset({"role", "gate", "script", "human", "loop"})


def _failure_key(value: str) -> str:
    return " ".join(value.split()).casefold()


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, entry: JournalEntry) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False, default=str) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line (killed mid-write) must not block recovery.
                continue
        return entries

    def recent(self, count: int = 10) -> list[dict]:
        return self.read_all()[-count:]

    def entries_for_phase(self, phase: str) -> list[dict]:
        return [
            entry for entry in self.read_all()
            if entry.get("phase") == phase and entry.get("kind") in EXECUTION_KINDS
        ]

    def attempts_for_phase(self, phase: str) -> int:
        return len(self.entries_for_phase(phase))

    def last_route(self) -> dict | None:
        for entry in reversed(self.read_all()):
            if entry.get("kind") == "route":
                return entry
        return None

    def last_route_target(self) -> str | None:
        route = self.last_route()
        return route.get("verdict") if route else None

    def last_accepted_revision(self) -> str | None:
        for entry in reversed(self.read_all()):
            if entry.get("kind") == "checkpoint" and entry.get("candidate_rev"):
                return entry["candidate_rev"]
        return None

    def last_role(self) -> str | None:
        for entry in reversed(self.read_all()):
            if entry.get("kind") == "role" and entry.get("role"):
                return entry["role"]
        return None

    def completed_items(self) -> set[str]:
        return {
            entry["item"] for entry in self.read_all()
            if entry.get("kind") == "item_complete" and entry.get("item")
        }

    def last_errors_for_phase(self, phase: str) -> list[str]:
        errors: list[str] = []
        for entry in self.read_all():
            if entry.get("phase") == phase and not entry.get("ok", False):
                errors.extend(entry.get("errors", []))
        return errors

    def active_failure_causes(self, phase: str) -> list[str]:
        """Return the deduplicated retry memory for one destination phase."""
        causes: list[str] = []
        keys: set[str] = set()
        for entry in self.read_all():
            if entry.get("kind") != "failure_memory" or entry.get("phase") != phase:
                continue
            if entry.get("verdict") == "cleared":
                causes.clear()
                keys.clear()
                continue
            if entry.get("verdict") != "recorded":
                continue
            for value in entry.get("errors", []):
                if not isinstance(value, str) or not value.strip():
                    continue
                cause = " ".join(value.split())
                key = _failure_key(cause)
                if key not in keys:
                    keys.add(key)
                    causes.append(cause)
        return causes

    def phases_with_failure_memory(self) -> list[str]:
        phases: list[str] = []
        for entry in self.read_all():
            if entry.get("kind") != "failure_memory":
                continue
            phase = entry.get("phase")
            if isinstance(phase, str) and phase not in phases:
                phases.append(phase)
        return [phase for phase in phases if self.active_failure_causes(phase)]
