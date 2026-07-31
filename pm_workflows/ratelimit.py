"""Provider usage-limit detection.

The kernel journals a detected Claude or Codex limit and exits without
accepting the interrupted phase. Re-running with any supported driver resumes
that phase from the last accepted revision. The old wait helper remains for
callers that explicitly prefer in-process waiting.
"""
from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
import time
from datetime import datetime, timedelta
from pathlib import Path

TOKEN_LIMIT_FALLBACK_WAIT_SECONDS = 300
TOKEN_LIMIT_TOAST_THRESHOLD_SECONDS = 60


class ProcessError(RuntimeError):
    def __init__(
        self,
        command: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"command failed ({returncode}): {command}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )


class TokenLimitError(RuntimeError):
    def __init__(
        self,
        agent_kind: str,
        message: str,
        retry_at: datetime | None = None,
    ) -> None:
        self.agent_kind = agent_kind
        self.retry_at = retry_at
        super().__init__(message)

    def wait_seconds(self, now: datetime | None = None) -> int:
        if self.retry_at is None:
            return TOKEN_LIMIT_FALLBACK_WAIT_SECONDS
        current = now or datetime.now().astimezone()
        retry_at = self.retry_at
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=current.tzinfo)
        return max(1, int((retry_at - current).total_seconds()))


def token_limit_from_process_error(
    agent_kind: str, error: ProcessError
) -> TokenLimitError | None:
    if agent_kind not in {"codex", "claude"}:
        return None
    text = f"{error.stdout}\n{error.stderr}"
    if not is_token_limit_message(text):
        return None
    return TokenLimitError(
        agent_kind,
        compact_error_message(text),
        parse_retry_at(text),
    )


def is_token_limit_message(text: str) -> bool:
    lowered = text.lower()
    if "context length" in lowered or "maximum context" in lowered:
        return False
    indicators = (
        "usage limit",
        "session limit",
        "rate limit",
        "too many requests",
        "try again",
        "retry after",
        "429",
    )
    return any(indicator in lowered for indicator in indicators)


def compact_error_message(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "token or usage limit reached"
    return "\n".join(lines[-6:])


def parse_retry_at(text: str, now: datetime | None = None) -> datetime | None:
    current = now or datetime.now().astimezone()
    relative = parse_relative_retry(text, current)
    if relative is not None:
        return relative
    absolute = parse_absolute_retry(text, current)
    if absolute is not None:
        return absolute
    time_only = parse_time_only_retry(text, current)
    if time_only is not None:
        return time_only
    return None


def parse_relative_retry(text: str, now: datetime) -> datetime | None:
    match = re.search(
        r"\b(?:try again|retry|reset|resets|retry after)[^\n.]*?\b(?:in\s+)?"
        r"(\d+)\s*(seconds?|minutes?|hours?)",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("second"):
        return now + timedelta(seconds=amount)
    if unit.startswith("minute"):
        return now + timedelta(minutes=amount)
    return now + timedelta(hours=amount)


def parse_absolute_retry(text: str, now: datetime) -> datetime | None:
    pattern = (
        r"\b(?:try again|retry|reset|resets)[^\n.]*?\bat\s+"
        r"([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?"
        r",?\s+(\d{4})\s+(\d{1,2})(?::(\d{2}))?\s*([AP]M)\b"
    )
    match = re.search(pattern, text, re.IGNORECASE)
    if match is None:
        return None
    month_name, day, year, hour, minute, meridiem = match.groups()
    month = month_number(month_name)
    if month is None:
        return None
    hour_int = int(hour) % 12
    if meridiem.upper() == "PM":
        hour_int += 12
    return datetime(
        int(year),
        month,
        int(day),
        hour_int,
        int(minute or "0"),
        tzinfo=now.tzinfo,
    )


def parse_time_only_retry(text: str, now: datetime) -> datetime | None:
    pattern = (
        r"\b(?:try again|retry|reset|resets)[^\n.]*?\b(?:at\s+)?"
        r"(\d{1,2})(?::(\d{2}))?\s*([AP]M)\b"
    )
    match = re.search(pattern, text, re.IGNORECASE)
    if match is None:
        return None
    hour, minute, meridiem = match.groups()
    hour_int = int(hour) % 12
    if meridiem.upper() == "PM":
        hour_int += 12
    candidate = now.replace(
        hour=hour_int,
        minute=int(minute or "0"),
        second=0,
        microsecond=0,
    )
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def month_number(month_name: str) -> int | None:
    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "sept": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    return months.get(month_name[:4].lower()) or months.get(
        month_name[:3].lower()
    )


def format_duration(seconds: int) -> str:
    seconds = max(0, seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def show_windows_toast(title: str, message: str) -> None:
    try:
        from windows_toasts import Toast, WindowsToaster

        toaster = WindowsToaster("Workflow Kernel")
        toast = Toast()
        toast.text_fields = [title, message]
        toaster.show_toast(toast)
    except Exception:
        # Toasts are helpful, not part of workflow correctness.
        pass


def freeze_until_limit_resets(error: TokenLimitError) -> None:
    """Block until the provider limit resets. Never raises."""
    seconds = error.wait_seconds()
    message = (
        f"{error.agent_kind} usage limit reached; waiting "
        f"{format_duration(seconds)} before retrying the same role."
    )
    print(f"\n!!! {message}")
    print(f"    provider said: {error}")
    if seconds >= TOKEN_LIMIT_TOAST_THRESHOLD_SECONDS:
        show_windows_toast("Workflow paused", message)
    time.sleep(seconds)


def resolve_executable(command: str) -> str:
    """Resolve a bare command name to a real executable path on Windows."""
    if Path(command).is_absolute():
        return command
    from shutil import which

    found = which(command)
    if found:
        return found
    if __import__("os").name == "nt":
        for suffix in (".cmd", ".exe", ".bat", ".ps1"):
            found = which(command + suffix)
            if found:
                return found
    return command


def run_process(
    command: list[str],
    cwd: Path,
    timeout: int = 7200,
    input_text: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [resolve_executable(command[0]), *command[1:]]
    result = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=environment,
    )
    if result.returncode != 0:
        raise ProcessError(command, result.returncode, result.stdout, result.stderr)
    return result


def run_agent_process(
    agent_kind: str,
    command: list[str],
    cwd: Path,
    timeout: int = 7200,
    input_text: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an agent CLI, converting usage limits into TokenLimitError."""
    try:
        return run_process(
            command, cwd, timeout, input_text=input_text, environment=environment
        )
    except ProcessError as exc:
        token_limit = token_limit_from_process_error(agent_kind, exc)
        if token_limit is not None:
            raise token_limit from exc
        raise
