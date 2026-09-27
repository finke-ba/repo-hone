"""Derived identities (ARCH §14, §15).

Nothing here looks anything up in shared state: hooks on one event run
concurrently, so every identity must be computable independently.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from . import gitcmd

_SCP_LIKE = re.compile(r"^(?:(?P<user>[^@/]+)@)?(?P<host>[^:/]+):(?P<path>.+)$")
_URL_LIKE = re.compile(r"^(?P<scheme>[a-z][a-z0-9+.-]*)://(?:[^@/]+@)?(?P<host>[^:/]+)"
                       r"(?::\d+)?/(?P<path>.+)$", re.I)


def session_id(host: Optional[str], host_session_id: Optional[str]) -> Optional[str]:
    """rh_ + stable_hash(host, host_session_id). Order-independent and idempotent.

    None when the host supplied no id: hashing "" is stable, so every id-less
    session would otherwise share one record and interleave unrelated turns."""
    if not (host_session_id or "").strip():
        return None
    digest = hashlib.sha256(f"{host}\x00{host_session_id}".encode()).hexdigest()
    return "rh_" + digest[:16]


def normalise_remote(url: str) -> Optional[str]:
    """ssh and https forms of one remote must normalise to the same string."""
    if not url:
        return None
    url = url.strip().rstrip("/")
    m = _URL_LIKE.match(url)
    if not m:
        m = _SCP_LIKE.match(url)
        if not m:
            return None
    host = m.group("host").lower()
    path = m.group("path").lstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return f"{host}/{path}"


def repository_id(remote_url: Optional[str]) -> Optional[str]:
    """Null when there is no remote: a local-only repo has no cross-checkout identity."""
    normalised = normalise_remote(remote_url) if remote_url else None
    if not normalised:
        return None
    return "repo_" + hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def origin_url(cwd) -> Optional[str]:
    return gitcmd.config("remote.origin.url", cwd)


def _read_marker(marker: Path, attempts: int = 20) -> Optional[str]:
    """A winner may still be writing when a loser looks, so a blank read is
    retried rather than treated as 'no id'."""
    for _ in range(attempts):
        try:
            value = marker.read_text().strip()
        except OSError:
            value = ""
        if value:
            return value
        time.sleep(0.005)
    return None


HOME_FILE = "checkout-home"


def _home_of(marker: Path) -> Optional[str]:
    try:
        return (marker.parent / HOME_FILE).read_text().strip() or None
    except OSError:
        return None


def _set_home(marker: Path, git_dir: str) -> None:
    staging = marker.parent / f".{HOME_FILE}.{os.getpid()}.{uuid.uuid4().hex}"
    staging.write_text(git_dir + "\n")
    os.replace(staging, marker.parent / HOME_FILE)


def _held_elsewhere(recorded: str, here: str, checkout: str) -> bool:
    """True when another Git directory still holds this id: this one is a copy."""
    if recorded == here:
        return False
    other = Path(recorded) / "repohone" / "checkout-id"
    return _read_marker(other, attempts=1) == checkout


def copy_of(cwd) -> Optional[str]:
    """The Git directory whose identity this checkout was copied with, if any."""
    gd = gitcmd.git_dir(cwd)
    if gd is None:
        return None
    marker = Path(gd) / "repohone" / "checkout-id"
    checkout, recorded = _read_marker(marker, attempts=1), _home_of(marker)
    here = str(Path(gd).resolve())
    if checkout and recorded and _held_elsewhere(recorded, here, checkout):
        return recorded
    return None


def _settle(marker: Path, git_dir: Path, checkout: str) -> str:
    """A copied directory carries this marker along. The id stays with the Git
    directory that recorded it; a copy is given its own, a move keeps it."""
    here = str(git_dir.resolve())
    recorded = _home_of(marker)
    if recorded == here:
        return checkout
    from . import record  # record imports this module
    with record._locked(marker.parent / "identity"):
        checkout = _read_marker(marker, attempts=1) or checkout
        recorded = _home_of(marker)
        if recorded is not None and _held_elsewhere(recorded, here, checkout):
            staging = marker.parent / f".checkout-id.{os.getpid()}.{uuid.uuid4().hex}"
            checkout = str(uuid.uuid4())
            staging.write_text(checkout + "\n")
            os.replace(staging, marker)
        if recorded != here:
            _set_home(marker, here)
    return checkout


def checkout_id(cwd) -> Optional[str]:
    """A UUID in worktree-local git metadata. Never committed, never derived from
    a path, so moving or renaming a checkout does not change its telemetry scope,
    and a copy does not share one.

    Establishing it is atomic: concurrent first hooks must not each mint an id
    and split the checkout across several telemetry stores. Exactly one
    contender's file becomes the marker; every other returns what it reads.
    """
    gd = gitcmd.git_dir(cwd)
    if gd is None:
        return None
    marker = Path(gd) / "repohone" / "checkout-id"
    existing = _read_marker(marker, attempts=1)
    if existing:
        try:
            return _settle(marker, Path(gd), existing)
        except OSError:
            return None

    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    candidate = str(uuid.uuid4())
    staging = marker.parent / f".checkout-id.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        staging.write_text(candidate + "\n")
        try:
            os.link(staging, marker)      # atomic: fails if the marker exists
            _set_home(marker, str(Path(gd).resolve()))
            return candidate
        except FileExistsError:
            return _read_marker(marker)
        except OSError:
            # Filesystems without hard links: O_EXCL is the fallback.
            try:
                handle = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                return _read_marker(marker)
            with os.fdopen(handle, "w") as fh:
                fh.write(candidate + "\n")
            _set_home(marker, str(Path(gd).resolve()))
            return candidate
    except OSError:
        return _read_marker(marker)
    finally:
        try:
            staging.unlink()
        except OSError:
            pass


def peek_checkout_id(cwd) -> Optional[str]:
    """Read-only: never creates the marker. Used before the privacy boundary (§5).
    None for a copy that has not captured yet: its id still belongs to the original."""
    if copy_of(cwd):
        return None
    gd = gitcmd.git_dir(cwd)
    if gd is None:
        return None
    return _read_marker(Path(gd) / "repohone" / "checkout-id", attempts=1)


def developer_id(cwd) -> Optional[str]:
    email = gitcmd.config("user.email", cwd)
    if not email:
        return None
    return "dev_" + hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:16]
