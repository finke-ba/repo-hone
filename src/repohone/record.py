"""Session record construction and persistence (ARCH §19, §22.1).

Records are written in the `observed` phase only: Phase 1 delivers the
two-phase structure, not reconciliation. Every mutation happens inside a file
lock, because concurrent hooks on one event may both reach the same record.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from . import CONTRACT_VERSION, SCHEMA_ID, SCHEMA_VERSION, artifacts, deadline, identity, paths

try:
    import fcntl
except ImportError:  # Windows: the O_EXCL fallback below takes over
    fcntl = None  # type: ignore[assignment]

LOCK_WAIT_S = 30.0
LOCK_POLL_S = 0.01


class LockUnavailable(RuntimeError):
    """The lock could not be taken. Never a reason to proceed anyway."""


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def record_path(checkout_id: str, session_id: str) -> Path:
    artifacts.require_artifact_id("session", session_id)
    return paths.records_dir(checkout_id) / f"{session_id}.json"


def new_record(session_id: str, host: str, host_session_id: str, adapter: str,
               adapter_version: str, content_policy: str, repository: Dict[str, Any],
               developer_id: Optional[str], agent_name: str,
               agent_version: Optional[str], started_at: str, cwd: Optional[str]) -> dict:
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "host": {"name": host, "session_id": host_session_id},
        "lifecycle": {"state": "observed", "started_at": started_at, "ended_at": None,
                      "end_reason": None, "reconciled_at": None,
                      "starts": [{"at": started_at, "source": None, "cwd": cwd}]},
        "capture": {"adapter": adapter, "adapter_version": adapter_version,
                    "content_policy": content_policy,
                    "cwds": [cwd] if cwd else [], "errors": []},
        "agent": {"name": agent_name, "version": agent_version},
        "models": [],
        "developer": {"id": developer_id},
        "repository": repository,
        "turns": [],
        "session_end_snapshots": [],
        "transcript": {"source_path": None, "stored_path": None, "stored_sha256": None,
                       "lines": None, "redactions": 0, "observed_models": [],
                       "observed_agent_versions": []},
        "labels": {"developer_label": None, "labeled_at": None},
        "reconciliation": {"outcome": "pending", "pull_request": None,
                           "agent_final_snapshot": None, "accepted_snapshot": None,
                           "related_session_ids": []},
        "extensions": {},
    }


def new_turn(index: int, logical_turn_id: str, cwd: Optional[str],
             workspace_changed: Optional[bool], permission_mode: Optional[str]) -> dict:
    return {"index": index, "logical_turn_id": logical_turn_id, "cwd": cwd,
            "prompt_events": [], "stop_events": [],
            "workspace_changed_before_turn": workspace_changed,
            "completion": "pending", "completion_source": None, "completed_at": None,
            "permission_mode": permission_mode, "model": None}


def add_error(rec: dict, event: str, error: str) -> None:
    rec["capture"]["errors"].append({"at": now(), "event": event, "error": str(error)[:500]})


@contextmanager
def _locked(path: Path, suffix: str = ".lock") -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(suffix)
    stop = deadline.until(LOCK_WAIT_S)
    if fcntl is None:
        with _exclusive_file(lock, stop):
            yield
        return
    handle = open(lock, "a+")  # noqa: SIM115 -- held across the lock loop below
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError) as exc:
                if not isinstance(exc, BlockingIOError) and getattr(exc, "errno", None) \
                        not in (11, 13):
                    raise
                if time.monotonic() >= stop:
                    raise LockUnavailable(
                        f"could not take {lock.name} before the capture deadline; "
                        "another RepoHone process is holding it") from exc
                time.sleep(min(LOCK_POLL_S, max(0.0, stop - time.monotonic())))
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def _exclusive_file(lock: Path, stop: Optional[float] = None) -> Iterator[None]:
    """Portable fallback with serialized stale-lock recovery.

    Every create, stale check, unlink and release is guarded. Without that
    guard two waiters can both judge one old lock stale; the second then unlinks
    the first waiter's new lock and enters its critical section.
    """
    stop = deadline.until(LOCK_WAIT_S) if stop is None else stop
    while True:
        acquired = False
        with _recovery_guard(lock, stop):
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, f"{os.getpid()} {time.time()}".encode())
                finally:
                    os.close(fd)
                acquired = True
            except FileExistsError:
                if _abandoned(lock):
                    _discard(lock)
        if acquired:
            break
        if time.monotonic() >= stop:
            raise LockUnavailable(
                f"could not take {lock.name} before the capture deadline; "
                "another RepoHone process is holding it")
        time.sleep(min(LOCK_POLL_S, max(0.0, stop - time.monotonic())))
    try:
        yield
    finally:
        # Releasing the lock is cleanup, not more primary capture work.  Give it
        # a small slice of the caller's reserved time even when the operation
        # budget itself has just expired.
        with _recovery_guard(lock, time.monotonic() + min(0.1, LOCK_WAIT_S)):
            _discard(lock)


@contextmanager
def _recovery_guard(lock: Path, deadline: float) -> Iterator[None]:
    """Short-lived O_EXCL guard. A crashed guard fails closed, never self-reaps.

    Recovering this guard with another unprotected stale check would recreate
    the same race one level higher. Doctor can surface a persistent refusal.
    """
    guard = lock.with_name(lock.name + ".guard")
    while True:
        try:
            fd = os.open(str(guard), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise LockUnavailable(
                    f"could not take {lock.name} before the capture deadline; "
                    "the recovery guard is held") from None
            time.sleep(min(LOCK_POLL_S, max(0.0, deadline - time.monotonic())))
    try:
        yield
    finally:
        _discard(guard)


def _abandoned(lock: Path) -> bool:
    try:
        pid_text, _, stamp = lock.read_text(encoding="utf-8").partition(" ")
        pid = int(pid_text)
    except (OSError, ValueError):
        return False
    try:
        age = time.time() - float(stamp)
    except ValueError:
        age = LOCK_WAIT_S + 1
    if age < LOCK_WAIT_S:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _discard(lock: Path) -> None:
    try:
        lock.unlink()
    except OSError:
        pass


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=artifacts.STAGING_PREFIX,
                               suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@contextmanager
def session_lock(checkout_id: str, session_id: str) -> Iterator[None]:
    """Serializes one session's ordinal allocation, snapshot and append.

    Ordinals are atomic on their own, but a later prompt whose snapshot finishes
    first would otherwise be appended first and become `prompt_events[0]` — the
    immutable turn baseline (§18). Different sessions take different locks, so
    concurrent sessions are unaffected.
    """
    # Its own lock file: sharing one with open_record would deadlock, because
    # flock blocks a second handle on the same file within one process.
    path = record_path(checkout_id, session_id).with_suffix(".ordering")
    with _locked(path, suffix=".ordering.lock"):
        yield


def validate_context(rec: dict, checkout_id: str, session_id: str,
                     expected_host: Optional[str] = None,
                     expected_host_session_id: Optional[str] = None) -> None:
    """Require the artifact body, storage path and caller to name one session."""
    problems = []
    if rec.get("session_id") != session_id:
        problems.append(
            f"record claims session_id {rec.get('session_id')!r}; expected {session_id!r}")
    repository = rec.get("repository") or {}
    if repository.get("checkout_id") != checkout_id:
        problems.append(
            f"record claims checkout_id {repository.get('checkout_id')!r}; "
            f"expected {checkout_id!r}")
    host = rec.get("host") or {}
    derived = identity.session_id(host.get("name"), host.get("session_id"))
    if derived != session_id:
        problems.append("record host identity does not derive its session_id")
    if expected_host is not None and host.get("name") != expected_host:
        problems.append(f"record host {host.get('name')!r}; expected {expected_host!r}")
    if (expected_host_session_id is not None
            and host.get("session_id") != expected_host_session_id):
        problems.append(
            f"record host session {host.get('session_id')!r}; "
            f"expected {expected_host_session_id!r}")
    if problems:
        raise MalformedArtifact("; ".join(problems))


@contextmanager
def open_record(checkout_id: str, session_id: str, factory,
                expected_host: Optional[str] = None,
                expected_host_session_id: Optional[str] = None) -> Iterator[dict]:
    """Read-modify-write under one lock; existing evidence is never replaced."""
    path = record_path(checkout_id, session_id)
    with _locked(path):
        # A damaged record read as absent would restart the session mid-way:
        # wrong evidence, where a refused event is only missing evidence.
        if paths.present(path):
            rec = artifacts.load(path, "session",
                                 expected_checkout_id=checkout_id,
                                 expected_artifact_id=session_id)
            if rec is None:
                raise artifacts.MalformedArtifact(f"{path.name}: vanished while opened")
            validate_context(rec, checkout_id, session_id, expected_host,
                             expected_host_session_id)
        else:
            rec = factory()
            artifacts.validate(rec, "session")
            validate_context(rec, checkout_id, session_id, expected_host,
                             expected_host_session_id)
        yield rec
        artifacts.validate(rec, "session")
        validate_context(rec, checkout_id, session_id, expected_host,
                         expected_host_session_id)
        _atomic_write(path, rec)


ArtifactError = artifacts.ArtifactError
MalformedArtifact = artifacts.MalformedArtifact
UnsupportedVersion = artifacts.UnsupportedVersion


# 1.5 changed the activation model and the egress enum; 1.6 changed the ref
# namespace. Neither changed the record shape, so older records stay readable.
SUPPORTED_CONTRACTS = artifacts.SUPPORTED_CONTRACTS


def check_versions(rec: dict) -> None:
    """ARCH §22 — supported, migratable, or refused. Never silently reinterpreted."""
    artifacts.validate(rec, "session")


def load(checkout_id: str, session_id: str, strict: bool = True,
         expected_host: Optional[str] = None,
         expected_host_session_id: Optional[str] = None) -> Optional[dict]:
    """Raises UnsupportedVersion rather than returning a record whose semantics
    this Core cannot interpret. `strict=False` is for inspection only."""
    path = record_path(checkout_id, session_id)
    rec = artifacts.load(path, "session", strict=strict,
                         expected_checkout_id=checkout_id,
                         expected_artifact_id=session_id)
    if strict and rec is not None:
        validate_context(rec, checkout_id, session_id, expected_host,
                         expected_host_session_id)
    return rec


def list_sessions(checkout_id: str):
    return sorted(p.stem for p in artifacts.members(paths.records_dir(checkout_id)) or [])
