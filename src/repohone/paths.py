"""Platform-standard application data locations (ARCH §4).

Never hardcodes XDG paths: honours the platform convention, and the env var
where the platform defines one.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import List, Optional

APP = "RepoHone"
ENV_DATA_DIR = "REPOHONE_DATA_DIR"
ENV_TMP = "REPOHONE_TMP"


def data_dir() -> Path:
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / APP
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP


def global_db() -> Path:
    """Pre-initialization state only: hash-keyed, never repository content (§5)."""
    return data_dir() / "global.db"


def checkout_dir(checkout_id: str) -> Path:
    return data_dir() / "checkouts" / checkout_id


def state_db(checkout_id: str) -> Path:
    return checkout_dir(checkout_id) / "state.db"


def records_dir(checkout_id: str) -> Path:
    return checkout_dir(checkout_id) / "records"


def scratch_dir(checkout_id: str) -> Path:
    """Core's guaranteed writable scratch; excluded from the captured file set."""
    override = os.environ.get(ENV_TMP)
    root = Path(override).expanduser() if override else checkout_dir(checkout_id) / "tmp"
    return root


def error_log() -> Path:
    return data_dir() / "errors.log"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def present(path: Path) -> bool:
    """False only when nothing is there. A question that cannot be answered —
    an unreadable parent, say — raises instead of reading as absent."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


class Damaged(OSError):
    """Something is at the path, but not what was expected there."""


def read_regular(path: Path) -> Optional[str]:
    """The text of the regular file at `path`, or None only when nothing is there.

    `Path.is_file()` also answers False for a directory, a dangling link or an
    unreadable path, which is how damaged evidence and settings read as absent.
    """
    path = Path(path)
    try:
        os.lstat(path)
    except FileNotFoundError:
        return None
    try:
        mode = os.stat(path).st_mode
    except FileNotFoundError as exc:
        raise Damaged(f"{path} is a link to nothing") from exc
    if not stat.S_ISREG(mode):
        raise Damaged(f"{path} is not a regular file")
    return path.read_text(encoding="utf-8")


def entries(path: Path) -> Optional[List[Path]]:
    """The children of the directory at `path`, or None only when nothing is there.

    `Path.glob` yields nothing for a directory it cannot read.
    """
    path = Path(path)
    try:
        os.lstat(path)
    except FileNotFoundError:
        return None
    try:
        return sorted(path.iterdir())
    except FileNotFoundError as exc:
        raise Damaged(f"{path} is a link to nothing") from exc
