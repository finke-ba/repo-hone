"""Thin git wrapper. Every call is explicit about cwd and env; nothing shells out."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import deadline


class GitError(RuntimeError):
    def __init__(self, args, returncode, stderr):
        super().__init__(f"git {' '.join(args)} failed ({returncode}): {stderr.strip()}")
        self.args_ = args
        self.returncode = returncode
        self.stderr = stderr


def run(args: List[str], cwd, env: Optional[Dict[str, str]] = None,
        check: bool = True, timeout: Optional[float] = None) -> str:
    import os
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    limit = deadline.current()
    if timeout is None and limit is not None:
        timeout = max(0.01, limit - time.monotonic())
    try:
        proc = subprocess.run(["git"] + args, cwd=str(cwd), env=full_env,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise GitError(args, -1, "capture deadline exceeded") from exc
    if check and proc.returncode != 0:
        raise GitError(args, proc.returncode, proc.stderr)
    return proc.stdout.strip()


def index_flagged_paths(cwd, env: Optional[Dict[str, str]] = None) -> List[str]:
    """Tracked paths whose index flags can make ``git status``/``git add`` lie.

    ``git ls-files -v`` lower-cases its tag for ``assume-unchanged`` entries;
    ``S`` is ``skip-worktree``.  Read NUL-delimited bytes so unusual filenames
    retain the same fidelity as the porcelain status parser.
    """
    import os
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    timeout = None
    limit = deadline.current()
    if limit is not None:
        timeout = max(0.01, limit - time.monotonic())
    try:
        proc = subprocess.run(["git", "ls-files", "-v", "-z"], cwd=str(cwd),
                              env=full_env, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise GitError(["ls-files", "-v", "-z"], -1,
                       "capture deadline exceeded") from exc
    if proc.returncode != 0:
        raise GitError(["ls-files", "-v", "-z"], proc.returncode,
                       os.fsdecode(proc.stderr))
    found = []
    for entry in proc.stdout.split(b"\0"):
        if len(entry) < 3 or entry[1:2] != b" ":
            continue
        tag = entry[0]
        if tag == ord("S") or chr(tag).islower():
            found.append(os.fsdecode(entry[2:]))
    return found


def ok(args: List[str], cwd, **kw) -> bool:
    try:
        run(args, cwd, check=True, **kw)
        return True
    except (GitError, OSError):
        return False


def toplevel(cwd) -> Optional[Path]:
    try:
        return Path(run(["rev-parse", "--show-toplevel"], cwd))
    except (GitError, OSError):
        return None


def git_dir(cwd) -> Optional[Path]:
    """Worktree-specific, not the common dir: each worktree is its own checkout (§15)."""
    try:
        return Path(run(["rev-parse", "--absolute-git-dir"], cwd))
    except (GitError, OSError):
        return None


def common_dir(cwd) -> Optional[Path]:
    """Shared by every worktree of one clone."""
    try:
        return Path(run(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd))
    except (GitError, OSError):
        return None


def head(cwd) -> Optional[str]:
    try:
        return run(["rev-parse", "HEAD"], cwd)
    except (GitError, OSError):
        return None


def branch(cwd) -> Optional[str]:
    try:
        name = run(["symbolic-ref", "--quiet", "--short", "HEAD"], cwd)
        return name or None
    except (GitError, OSError):
        return None


def config(key: str, cwd) -> Optional[str]:
    try:
        return run(["config", "--get", key], cwd) or None
    except (GitError, OSError):
        return None
