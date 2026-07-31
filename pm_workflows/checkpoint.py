"""Git checkpoint backend. A snapshot is a commit; a revert is a hard reset.

This is deliberately the whole recovery story. When anything goes wrong the
repository is reset to the last accepted commit and the same role is dispatched
again in a fresh session. There is no session resume, no archive to restore
from, and no partial forward repair — those cost more to keep working than they
ever paid back.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


# Build output a revert must not delete. `git clean -fd` spares files git
# ignores, but plenty of real repositories never ignore `bin/` and `obj/` — and
# there, every retry would silently cost a full rebuild of a large native tree.
# Rebuilding is minutes to hours and none of it is the work being judged.
BUILD_OUTPUT = (
    "bin/", "obj/", ".vs/", ".vscode/",
    "node_modules/", "packages/", "dist/", "build/", "target/",
    ".Assemblies/", ".codegraph/",
)


class GitCheckpoint:
    def __init__(self, repo_path: str | Path):
        self.repo = Path(repo_path)

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )

    def _git_checked(self, *args: str) -> str:
        result = self._git(*args)
        if result.returncode != 0:
            raise GitError(
                f"git {' '.join(args)} failed in {self.repo}: "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )
        return result.stdout.strip()

    def init_if_needed(self) -> None:
        """Only ever initialises a repo that is not one yet."""
        if (self.repo / ".git").exists():
            return
        self._git_checked("init")
        # Local identity only — never touches the user's global git config.
        self._git("config", "user.email", "kernel@localhost")
        self._git("config", "user.name", "workflow kernel")

    def exclude_locally(self, *patterns: str) -> None:
        """Make git ignore paths the controller owns, without touching a
        tracked `.gitignore`.

        `state.md` is written by the controller after every phase. If git could
        see it, it would show up as a change in every scope and commit check,
        and either be committed by a checkpoint or deleted by a revert. This
        goes in `.git/info/exclude`, which is local to the clone and not part of
        the repository's own files.
        """
        exclude = self.repo / ".git" / "info" / "exclude"
        if not exclude.parent.is_dir():
            return
        existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        missing = [p for p in patterns if p not in existing.splitlines()]
        if not missing:
            return
        prefix = "" if existing.endswith("\n") or not existing else "\n"
        exclude.write_text(
            existing + prefix + "# written by the workflow controller\n"
            + "\n".join(missing) + "\n",
            encoding="utf-8",
        )

    def current_rev(self) -> str:
        result = self._git("rev-parse", "HEAD")
        return result.stdout.strip() if result.returncode == 0 else ""

    def is_dirty(self) -> bool:
        return bool(self._git("status", "--porcelain").stdout.strip())

    def snapshot(self, label: str) -> str:
        """Accept the current state. Commits only what is not committed yet.

        A role that commits its own work (as the conventions require) leaves a
        clean tree, and this just records its commit as the accepted revision.
        """
        if not self.is_dirty():
            return self.current_rev()
        self._git_checked("add", "-A")
        if not self._git("diff", "--cached", "--quiet").returncode:
            return self.current_rev()
        self._git_checked("commit", "-m", f"checkpoint: {label}")
        return self.current_rev()

    def park(self, ref_name: str) -> str:
        """Preserve the current state on a throwaway ref, then report its sha.

        Used when the runtime gives up on a phase: the rejected work stays
        inspectable as a real commit (`git diff <accepted>..<ref>`) while the
        working tree is left clean. Without this, either the repository keeps a
        state that failed its checks or the diff is gone for good.
        """
        if not self.is_dirty() :
            head = self.current_rev()
            if head:
                self._git("branch", "-f", ref_name, head)
            return head
        self._git_checked("add", "-A")
        self._git_checked("commit", "-m", f"rejected: {ref_name}")
        sha = self.current_rev()
        self._git("branch", "-f", ref_name, sha)
        return sha

    def restore(self, rev: str) -> None:
        """Discard everything after `rev`, but never the build output.

        `clean` runs without `-x` so ignored files survive, and with an explicit
        exclude list for build directories on top — because whether `bin/` and
        `obj/` are ignored varies by repository, and getting it wrong turns
        every retry into a full rebuild of a large native tree.
        """
        if not rev:
            return
        self._git_checked("reset", "--hard", rev)
        excludes: list[str] = []
        for pattern in BUILD_OUTPUT:
            excludes += ["-e", pattern]
        self._git("clean", "-fd", *excludes)

    def commits_since(self, rev: str) -> list[str]:
        if not rev:
            return []
        result = self._git("rev-list", f"{rev}..HEAD")
        return [line for line in result.stdout.splitlines() if line.strip()]

    def changed_files_since(self, rev: str) -> list[str]:
        """Paths that differ from `rev`, including uncommitted and untracked."""
        if not rev:
            return []
        changed: set[str] = set()
        tracked = self._git("diff", "--name-only", rev)
        changed.update(line.strip() for line in tracked.stdout.splitlines() if line.strip())
        untracked = self._git("ls-files", "--others", "--exclude-standard")
        changed.update(line.strip() for line in untracked.stdout.splitlines() if line.strip())
        return sorted(changed)

    def diff(self, base: str, candidate: str) -> str:
        return self._git("diff", base, candidate).stdout
