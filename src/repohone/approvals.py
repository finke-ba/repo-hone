"""Approvals the gate recorded when it asked (ARCH §9): one per gated command,
used once by the command it approved.

A file per approval, claimed by renaming it, so two processes can never both use
one. Kept under the data directory because `init` is approved before consent."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import List

from . import paths

EXPIRY_S = 3600


def root() -> Path:
    return paths.data_dir() / "approvals"


def _dir(host_session: str) -> Path:
    return root() / hashlib.sha256(host_session.encode("utf-8")).hexdigest()[:32]


def digest(argv: List[str]) -> str:
    return hashlib.sha256(json.dumps(list(argv)).encode("utf-8")).hexdigest()[:32]


def mint(host_session: str, argv: List[str]) -> None:
    _prune()
    directory = paths.ensure(_dir(host_session))
    name = f"{digest(argv)}.{uuid.uuid4().hex}"
    staging = directory / f".{name}"
    staging.write_text(repr(time.time()), encoding="utf-8")
    os.replace(staging, directory / name)


def consume(host_session: str, argv: List[str]) -> bool:
    directory = _dir(host_session)
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return False
    prefix = digest(argv) + "."
    for name in names:
        if not name.startswith(prefix) or ".claimed-" in name:
            continue
        claimed = directory / f"{name}.claimed-{uuid.uuid4().hex}"
        try:
            os.rename(directory / name, claimed)
        except FileNotFoundError:
            continue
        try:
            minted = float(claimed.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            minted = 0.0
        claimed.unlink()
        if time.time() - minted <= EXPIRY_S:
            return True
    return False


def void(host_session: str) -> None:
    shutil.rmtree(_dir(host_session), ignore_errors=True)


def _prune() -> None:
    """Sessions that ended without a SessionEnd leave their directory behind."""
    try:
        entries = list(os.scandir(root()))
    except FileNotFoundError:
        return
    cutoff = time.time() - EXPIRY_S
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False) and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry.path, ignore_errors=True)
        except OSError:
            continue
