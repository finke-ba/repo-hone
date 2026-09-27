"""Short-lived subprocess supervision shared by reasoning and validation.

A process group handles the normal parent/child case. Some build tools also
start an independently managed worker. Those workers retain their environment,
so every RepoHone run receives a unique marker and cleanup checks the host's
process table for that marker before reporting the run finished.

The marker is local coordination data. Process command lines and environments
are inspected only in memory and are never logged or persisted.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from typing import Dict, Set, Tuple

MARKER_ENV = "REPOHONE_PROCESS_TOKEN"


class TrackingUnavailable(RuntimeError):
    """The host cannot verify that a bounded run left no worker behind."""


def tracked_env(env: Dict[str, str]) -> Tuple[Dict[str, str], str]:
    """Return a copy of *env* carrying a collision-resistant run marker.

    Capability is checked before the command starts. A host that cannot inspect
    same-user process environments must decline validation rather than make a
    lifetime claim it cannot verify.
    """
    _marked_pids("repohone-capability-check")
    token = uuid.uuid4().hex
    prepared = dict(env)
    prepared[MARKER_ENV] = token
    return prepared, token


def _linux_marked_pids(needle: bytes) -> Set[int]:
    root = "/proc"
    if not os.path.isdir(root):
        raise TrackingUnavailable("/proc process metadata is unavailable")
    found: Set[int] = set()
    for name in os.listdir(root):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == os.getpid():
            continue
        try:
            with open(f"{root}/{name}/environ", "rb") as handle:
                values = handle.read().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        except OSError:
            continue
        if needle in values:
            found.add(pid)
    return found


def _ps_marked_pids(needle: str) -> Set[int]:
    tool = "/bin/ps" if os.path.exists("/bin/ps") else "ps"
    try:
        result = subprocess.run(
            [tool, "eww", "-axo", "pid=,command="], capture_output=True,
            text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TrackingUnavailable(f"process lifetime inspection is unavailable: {exc}") from exc
    if result.returncode != 0:
        raise TrackingUnavailable(
            "process lifetime inspection was refused by the operating system")
    found: Set[int] = set()
    for line in result.stdout.splitlines():
        if needle not in line:
            continue
        head = line.lstrip().split(None, 1)
        if not head or not head[0].isdigit():
            continue
        pid = int(head[0])
        if pid != os.getpid():
            found.add(pid)
    return found


def _marked_pids(token: str) -> Set[int]:
    assignment = f"{MARKER_ENV}={token}"
    if sys.platform.startswith("linux"):
        return _linux_marked_pids(assignment.encode("utf-8"))
    if os.name == "posix":
        return _ps_marked_pids(assignment)
    raise TrackingUnavailable(
        "this platform has no RepoHone process lifetime implementation")


def terminate(proc, token: str) -> None:
    """End the original process group and every marked worker.

    A bounded retry closes the small race where one marked worker starts another
    while cleanup begins. Failure to inspect or remove a marked process is an
    explicit validation failure.
    """
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    elif proc.poll() is None:
        proc.kill()

    remaining: Set[int] = set()
    for _attempt in range(5):
        remaining = _marked_pids(token)
        if not remaining:
            break
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise TrackingUnavailable(
                    f"could not end validation worker {pid}: {exc}") from exc
        time.sleep(0.05)
    remaining = _marked_pids(token)
    if remaining:
        raise TrackingUnavailable(
            f"{len(remaining)} validation worker(s) remained after cleanup")

    # communicate also closes pipe objects after a leader that was already
    # reaped or killed above; skipping it leaks descriptors in repeated probes.
    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
