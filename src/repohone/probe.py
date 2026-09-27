"""Running the project's own tooling safely (ARCH §12.2, §25).

RepoHone may invoke project tooling during discovery and validation. It must
never do so against the live working tree: project tooling writes caches,
coverage files and build artifacts, and RepoHone captures snapshots
concurrently with the developer's work.

So a probe runs against a **detached checkout of a snapshot, in scratch, with
tool state redirected there and writes confined to scratch**. If confinement
cannot start, no project command runs. Afterwards it verifies the developer's
working tree is unchanged. A difference is a RepoHone defect, never a project
finding.
"""
from __future__ import annotations

import hashlib
import json
import os
import select
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from . import gitcmd, paths, processes

DEFAULT_TIMEOUT_S = 120.0


class WorkingTreeTouched(RuntimeError):
    """The developer's tree changed across a probe. A RepoHone defect."""


class FingerprintUnavailable(RuntimeError):
    """The live checkout could not be compared reliably; never claim unchanged."""


@dataclass
class ProbeResult:
    ran: bool
    exit_code: Optional[int]
    duration_ms: Optional[int]
    note: Optional[str] = None
    detail: Optional[str] = None       # the command's last line of output, minimized

    @property
    def could_not_run(self) -> bool:
        """126 and 127 are the shell saying the command never started."""
        return self.exit_code in (126, 127)


def _last_line(stderr: str, stdout: str, scratch: Path, repo) -> Optional[str]:
    """What a maintainer needs to see why a run failed, and nothing more."""
    from . import redact, toolinput
    lines = [line.strip() for line in (stderr or stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    text = lines[-1].replace(str(scratch / "work"), "<repo>").replace(str(scratch), "<scratch>")
    cleaned, _ = redact.redact(text)
    return toolinput.scrub_paths(cleaned, repo)[:200]


def tree_fingerprint(repo, deep: bool = False) -> Tuple[Optional[str], bytes, dict]:
    """HEAD, raw NUL-delimited status, and optionally all dirty file contents.

    Porcelain's human-readable form C-quotes unusual names. Treating that text
    as a path silently omitted files containing newlines, tabs or quotes.
    """
    # -uall lists files inside an untracked directory individually; the default
    # collapses them to `dir/`, which cannot be hashed and hid rewrites under it.
    # --ignored covers build output and caches: project tooling writes there, and
    # excluding them made "the tree is unchanged" false for exactly the artifacts
    # a build most often leaves behind.
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "-uall",
             "--ignored=traditional"], cwd=str(repo), check=True,
            capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FingerprintUnavailable(f"cannot read Git status: {exc}") from exc
    if not deep:
        return gitcmd.head(repo), status, {}
    # Index hints can suppress a real tracked-file edit from porcelain status.
    # Hash those paths explicitly so the before/after guard still sees them.
    entries = list(_status_entries(status))
    ignored = {name for code, name in entries if code == b"!!"}
    names = {name for _, name in entries} | set(gitcmd.index_flagged_paths(repo))
    digests = {}
    for name in names:
        target = Path(repo) / name
        digests[name] = (_stat_digest(target) if name in ignored
                         else _digest(target))
    return gitcmd.head(repo), status, digests


def _status_entries(status: bytes):
    """Decode Git's NUL-delimited status without losing unusual path bytes.

    Yields ``(two-letter code, name)``; a rename reports its source under the
    same code.
    """
    entries = status.split(b"\0")
    index = 0
    while index < len(entries) and entries[index]:
        entry = entries[index]
        if len(entry) < 4 or entry[2:3] != b" ":
            raise FingerprintUnavailable("malformed Git porcelain status")
        code = entry[:2]
        record_names = [entry[3:]]
        if b"R" in code or b"C" in code:
            index += 1
            if index >= len(entries) or not entries[index]:
                raise FingerprintUnavailable("incomplete Git rename status")
            record_names.append(entries[index])
        for raw_name in record_names:
            yield code, os.fsdecode(raw_name)
        index += 1


def status_paths(status: bytes) -> set[str]:
    return {name for _, name in _status_entries(status)}


def _ignored_paths(status: bytes) -> set[str]:
    return {name for code, name in _status_entries(status) if code == b"!!"}


def _stat_digest(target: Path) -> str:
    """Size and mtime, for paths Git is ignoring.

    Content hashing a `node_modules` costs seconds twice per probe. A build that
    rewrites a file moves its mtime; one that restores byte-identical content at
    the same nanosecond does not, and that is the detection this trades away.
    """
    try:
        if target.is_symlink():
            return "link:" + hashlib.sha256(os.fsencode(os.readlink(target))).hexdigest()
        if target.is_dir():
            marks = []
            for child in sorted(target.rglob("*")):
                info = child.lstat()
                marks.append((str(child.relative_to(target)), info.st_size,
                              info.st_mtime_ns, stat.S_IFMT(info.st_mode)))
            return "directory:" + hashlib.sha256(
                repr(marks).encode("utf-8", "surrogateescape")).hexdigest()
        info = target.lstat()
        return f"stat:{info.st_size}:{info.st_mtime_ns}"
    except FileNotFoundError:
        return "absent"
    except OSError as exc:
        raise FingerprintUnavailable(f"cannot stat {target}: {exc}") from exc


def _digest(target: Path) -> str:
    try:
        if target.is_symlink():
            return "link:" + hashlib.sha256(os.fsencode(os.readlink(target))).hexdigest()
        if target.is_dir():
            entries = []
            for child in sorted(target.rglob("*")):
                entries.append((str(child.relative_to(target)), _digest(child)))
            return "directory:" + hashlib.sha256(
                repr(entries).encode("utf-8", "surrogateescape")).hexdigest()
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except FileNotFoundError:
        return "absent"
    except OSError as exc:
        raise FingerprintUnavailable(f"cannot hash {target}: {exc}") from exc


def materialise(repo, tree_ish: str, destination: Path) -> None:
    """`git archive` writes only into the destination — no worktree registration,
    no index mutation, nothing added to the repository's metadata."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "archive", "--format=tar", tree_ish],
                             cwd=str(repo), capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(destination)],
                   input=archive.stdout, check=True)


def _within(base: Path, relative: str) -> Path:
    """A proposed path comes from a model, so it must not escape the checkout."""
    target = (base / relative).resolve()
    if not str(target).startswith(str(base.resolve()) + os.sep):
        raise ValueError(f"path escapes the checkout: {relative!r}")
    return target


def _isolated_env(scratch: Path) -> dict:
    """Tool caches and temporary state are redirected into scratch so a probe
    cannot warm, poison or grow anything in the developer's environment."""
    env = dict(os.environ)
    home = scratch / "home"
    for directory in ("home", "tmp", "cache"):
        (scratch / directory).mkdir(parents=True, exist_ok=True)
    env.update({
        "HOME": str(home),
        "TMPDIR": str(scratch / "tmp"),
        "XDG_CACHE_HOME": str(scratch / "cache"),
        "XDG_CONFIG_HOME": str(scratch / "home" / ".config"),
        "XDG_DATA_HOME": str(scratch / "home" / ".local" / "share"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_NO_INPUT": "1",
        "CI": "1",
        "REPOHONE_PROBE": "1",
    })
    env.pop("REPOHONE_DATA_DIR", None)
    return env


def _confined_argv(scratch: Path, argv, protected_repo=None,
                   denial_marker: Optional[str] = None) -> Optional[list]:
    """Permit writes only inside scratch. Unsupported hosts fail closed."""
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file():
        marker = denial_marker or "repohone-unmonitored-denial"
        message = f"(with telemetry) (with message {json.dumps(marker)})"
        terminating = message + " (with send-signal SIGKILL)"
        # Every policy denial is tagged for the out-of-process monitor below.
        # The narrow service and device permissions are required by ordinary
        # program startup; persistent writes remain limited to scratch.
        rules = ["(version 1)", f"(deny default {message})",
                 f"(deny file-write* {terminating})",
                 f"(deny network* {terminating})",
                 "(allow file-read*)", "(allow process*)",
                 "(allow sysctl-read)", "(allow mach-lookup)",
                 "(allow ipc-posix-shm-read-data)",
                 '(allow file-write-data (literal "/dev/dtracehelper"))',
                 '(allow file-ioctl (literal "/dev/dtracehelper"))',
                 '(allow file-write-data (literal "/dev/null"))',
                 '(allow file-write-data (literal "/dev/tty"))',
                 f"(allow file-write* (subpath {json.dumps(str(scratch))}))"]
        profile = "\n".join(rules)
        return ["/usr/bin/sandbox-exec", "-p", profile, *argv]
    if sys.platform.startswith("linux"):
        bubblewrap = shutil.which("bwrap")
        if bubblewrap:
            command = [bubblewrap, "--die-with-parent", "--unshare-all",
                       "--ro-bind", "/", "/", "--bind", str(scratch), str(scratch)]
            if protected_repo is not None:
                # An absolute reference to the live checkout resolves to the
                # detached candidate instead. A write therefore succeeds in
                # scratch instead of producing a catchable read-only-filesystem
                # error that could change the command's verdict.
                command += ["--bind", str(scratch / "work"),
                            str(Path(protected_repo).resolve())]
            return command + ["--dev-bind", "/dev", "/dev", "--proc", "/proc",
                              "--", *argv]
    return None


def _start_denial_monitor(marker: str):
    """Start macOS Unified Logging before a Seatbelt-confined command.

    Seatbelt applies to descendants, so a parent can catch or suppress a
    child's failed operation and still exit zero. The tagged kernel event is
    outside that process tree and cannot be hidden by project code.
    """
    log_tool = Path("/usr/bin/log")
    if not log_tool.is_file():
        return None, "macOS sandbox denial telemetry is unavailable"
    predicate = f'eventMessage CONTAINS[c] "{marker}"'
    monitor = None
    try:
        monitor = subprocess.Popen(
            [str(log_tool), "stream", "--level", "debug", "--style", "ndjson",
             "--predicate", predicate],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "LC_ALL": "C"})
        assert monitor.stdout is not None
        ready, _, _ = select.select([monitor.stdout], [], [], 2.0)
        banner = monitor.stdout.readline() if ready else ""
        if (not banner.startswith("Filtering the log data using")
                or monitor.poll() is not None):
            monitor.terminate()
            output, _ = monitor.communicate(timeout=2)
            detail = (banner + output).strip()[:250]
            return None, ("macOS sandbox denial telemetry did not start"
                          + (f": {detail}" if detail else ""))
        return monitor, None
    except (OSError, subprocess.SubprocessError) as exc:
        if monitor is not None and monitor.poll() is None:
            monitor.kill()
            monitor.communicate()
        return None, f"macOS sandbox denial telemetry did not start: {exc}"


def _finish_denial_monitor(monitor, marker: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Drain tagged kernel events and stop the monitor.

    A short drain window accounts for the asynchronous handoff from the kernel
    to Unified Logging. Losing the monitor is itself a confinement failure.
    """
    assert monitor.stdout is not None
    lines = []
    deadline = time.monotonic() + 0.75
    exited_early = False
    while time.monotonic() < deadline:
        ready, _, _ = select.select(
            [monitor.stdout], [], [],
            max(0.0, min(0.1, deadline - time.monotonic())))
        if ready:
            line = monitor.stdout.readline()
            if line:
                lines.append(line)
                continue
        if monitor.poll() is not None:
            exited_early = True
            break
    if monitor.poll() is None:
        monitor.terminate()
    try:
        output, _ = monitor.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        monitor.kill()
        output, _ = monitor.communicate()
    lines.append(output)
    for line in "".join(lines).splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        message = str(event.get("eventMessage", ""))
        if marker and marker in message:
            return True, message.splitlines()[0][:250]
    if exited_early:
        return False, "macOS sandbox denial telemetry stopped unexpectedly"
    return False, None


def _run_bounded(argv, cwd, env, timeout: float):
    """Run one marked process family and leave no observed worker behind.

    Returns ``(returncode, stdout, stderr, timed_out)``. Kept separate from the
    confinement wrapper so the lifetime guarantee can be tested without a
    platform sandbox or process-list access.
    """
    run_env, token = processes.tracked_env(env)
    # Never the caller's terminal: the sandbox denies writes to it.
    proc = subprocess.Popen(
        argv, cwd=str(cwd), env=run_env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True)
    cleanup_error = None
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        returncode, timed_out = proc.returncode, False
    except subprocess.TimeoutExpired:
        stdout, stderr = "", ""
        returncode, timed_out = None, True
    finally:
        try:
            processes.terminate(proc, token)
        except processes.TrackingUnavailable as exc:
            cleanup_error = exc
    if cleanup_error is not None:
        raise cleanup_error
    return returncode, stdout, stderr, timed_out


def run(repo, checkout_id: str, tree_ish: str, argv,
        timeout: float = DEFAULT_TIMEOUT_S, overlay=None) -> ProbeResult:
    """Takes an argv list, never a shell string: filenames and config values are
    repository-controlled, and `shell=True` would turn them into commands.

    Raises WorkingTreeTouched if the developer's tree moved during the probe.
    """
    if isinstance(argv, str) or not argv:
        raise TypeError("probe.run takes a non-empty argv list, not a shell string")
    overlay = overlay or {}
    before = tree_fingerprint(repo, deep=True)
    scratch = Path(tempfile.mkdtemp(prefix="repohone-probe-",
                                    dir=str(paths.ensure(paths.scratch_dir(checkout_id))))).resolve()
    result = ProbeResult(False, None, None)
    try:
        if scratch == Path(repo).resolve() or Path(repo).resolve() in scratch.parents:
            return ProbeResult(False, None, None,
                               "scratch directory is inside the live repository")
        denial_marker = ("repohone-denial-" + uuid.uuid4().hex
                         if sys.platform == "darwin" else None)
        confined = _confined_argv(scratch, list(argv), protected_repo=repo,
                                  denial_marker=denial_marker)
        if confined is None:
            return ProbeResult(False, None, None,
                               "filesystem confinement is unavailable on this host")
        work = scratch / "work"
        materialise(repo, tree_ish, work)
        for relative, contents in overlay.items():
            target = _within(work, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents, encoding="utf-8")
        env = _isolated_env(scratch)
        started = time.perf_counter()
        monitor = None
        if denial_marker is not None:
            monitor, monitor_error = _start_denial_monitor(denial_marker)
            if monitor is None:
                return ProbeResult(False, None, None, monitor_error)
        try:
            returncode, stdout, stderr, timed_out = _run_bounded(
                confined, work, env, timeout)
            if timed_out:
                result = ProbeResult(True, None,
                                     int((time.perf_counter() - started) * 1000),
                                     f"exceeded {timeout:.0f}s")
            elif stderr.startswith(("sandbox-exec:", "bwrap:")):
                result = ProbeResult(False, None, None,
                                     "filesystem confinement did not start: "
                                     + stderr[:250])
            else:
                result = ProbeResult(True, returncode,
                                     int((time.perf_counter() - started) * 1000),
                                     detail=_last_line(stderr, stdout, scratch, repo))
        finally:
            if monitor is not None:
                denied, monitor_error = _finish_denial_monitor(
                    monitor, denial_marker)
                if denied:
                    result = ProbeResult(
                        False, None, None,
                        "filesystem confinement refused an operation: "
                        + (monitor_error or "tagged macOS sandbox denial"))
                elif monitor_error:
                    result = ProbeResult(False, None, None, monitor_error)
    except (subprocess.CalledProcessError, OSError,
            processes.TrackingUnavailable) as exc:
        # A missing executable is a fact about the mechanism, not a crash.
        result = ProbeResult(False, None, None, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if tree_fingerprint(repo, deep=True) != before:
        raise WorkingTreeTouched(
            "the developer's working tree changed during a probe; this is a "
            "RepoHone defect, not a project finding")
    return result
