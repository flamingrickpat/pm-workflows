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
EXECUTION_KINDS = frozenset({"role", "gate", "script", "human", "loop", "workflow"})


def _failure_key(value: str) -> str:
    return " ".join(value.split()).casefold()


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, entry: JournalEntry) -> None:
        # A process can be killed after it writes only part of the final JSON
        # object. Keep that torn audit fragment, but isolate it before the
        # next complete entry. Without this separator, the recovery entry is
        # concatenated to invalid JSON and both records disappear from the
        # active journal view.
        if self.path.stat().st_size:
            with self.path.open("rb+") as binary:
                binary.seek(-1, 2)
                if binary.read(1) not in {b"\n", b"\r"}:
                    binary.seek(0, 2)
                    binary.write(b"\n")
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False, default=str) + "\n")

    def read_all(self) -> list[dict]:
        entries = self.read_complete()
        recovery_index = None
        active_count = None
        for index, entry in enumerate(entries):
            if entry.get("kind") != "fatal_recovery":
                continue
            result = entry.get("result") or {}
            cursor = result.get("active_through_entry")
            if isinstance(cursor, int) and cursor >= 0:
                recovery_index = index
                active_count = cursor
        if recovery_index is not None and active_count is not None:
            return entries[:active_count] + entries[recovery_index:]
        return entries

    def read_complete(self) -> list[dict]:
        """Read the complete audit log without applying a recovery cursor."""
        if not self.path.exists():
            return []
        source_lines = [
            (line_number, line.strip())
            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1
            )
            if line.strip()
        ]
        decoded: list[tuple[int, dict | None]] = []
        for line_number, line in source_lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                decoded.append((line_number, None))
                continue
            if not isinstance(value, dict):
                raise ValueError(
                    f"{self.path}: journal line {line_number} is not a JSON object"
                )
            decoded.append((line_number, value))

        entries: list[dict] = []
        for index, (line_number, value) in enumerate(decoded):
            if value is not None:
                entries.append(value)
                continue
            # A kill can tear only the final append. Recovery preserves that
            # fragment and writes a fatal_recovery record immediately after it.
            # Any other malformed middle record is corruption and must fail.
            final_fragment = index == len(decoded) - 1
            followed_by_recovery = (
                not final_fragment
                and decoded[index + 1][1] is not None
                and decoded[index + 1][1].get("kind") == "fatal_recovery"
            )
            if not final_fragment and not followed_by_recovery:
                raise ValueError(
                    f"{self.path}: malformed journal JSON at line {line_number}"
                )
        return entries

    def append_recovery(
        self,
        *,
        restored_revision: str,
        resume_phase: str,
        active_through_entry: int,
        abandoned_entry_range: str,
        external_state_retained: bool = False,
    ) -> None:
        self.append(JournalEntry(
            run_id="fatal-recovery",
            phase=resume_phase,
            kind="fatal_recovery",
            ok=True,
            candidate_rev=restored_revision,
            verdict="restored",
            result={
                "restored_revision": restored_revision,
                "resume_phase": resume_phase,
                "active_through_entry": active_through_entry,
                "abandoned_entry_range": abandoned_entry_range,
                "external_state_retained": external_state_retained,
            },
        ))

    def pending_recovery_notice(self) -> bool:
        """Return true until the first role result after the latest recovery."""
        entries = self.read_all()
        recovery_index = None
        for index, entry in enumerate(entries):
            if entry.get("kind") == "fatal_recovery":
                recovery_index = index
        if recovery_index is None:
            return False
        return not any(
            entry.get("kind") == "role"
            for entry in entries[recovery_index + 1:]
        )

    def recent(self, count: int = 10) -> list[dict]:
        return self.read_all()[-count:]

    def entries_for_phase(self, phase: str) -> list[dict]:
        return [
            entry for entry in self.read_all()
            if entry.get("phase") == phase
            and (
                entry.get("kind") in EXECUTION_KINDS
                or int(entry.get("attempt", 0) or 0) > 0
            )
        ]

    def attempts_for_phase(self, phase: str, item: str | None = None) -> int:
        """Executions of `phase`, or only those for one loop item when given.

        A phase that runs once per task (never inside a loop) is counted
        globally: pass no `item`. A phase inside a loop runs once per work
        item, and each item's attempts are its own budget — pass the current
        item so a hard-to-implement item cannot consume the attempts a later
        item needs, and so a later item does not inherit an earlier item's
        count.
        """
        entries = self.entries_for_phase(phase)
        if item is not None:
            entries = [entry for entry in entries if entry.get("item") == item]
        return len(entries)

    def last_route(self) -> dict | None:
        for entry in reversed(self.read_all()):
            if entry.get("kind") == "route":
                return entry
        return None

    def last_route_target(self) -> str | None:
        route = self.last_route()
        return route.get("verdict") if route else None

    def last_lease_boundary(self) -> dict | None:
        entries = self.read_all()
        if entries and entries[-1].get("kind") == "lease_boundary":
            return entries[-1]
        return None

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
